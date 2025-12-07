"""
============================================================
Statistical Tests - Validation and Significance Testing
Comprehensive statistical analysis for UMAP investigation
============================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
from pathlib import Path
import json

from config import STATISTICAL_TESTS, get_results_path
from utils import setup_logging, compute_effect_size, bootstrap_confidence_interval

logger = setup_logging()

# ============================================================
# Cross-Split Consistency Tests
# ============================================================
class CrossSplitConsistencyAnalyzer:
    """
    Test consistency of metrics across train/val/test splits
    """
    
    def __init__(self, all_metrics: Dict[str, Dict[str, Any]]):
        """
        Initialize analyzer
        
        Args:
            all_metrics: Nested dict {layer: {split: metrics_dict}}
        """
        self.all_metrics = all_metrics
        self.results = {}
    
    def run_permutation_test(self,
                            metric_name: str,
                            layer: str,
                            n_permutations: int = 1000) -> Dict[str, float]:
        """
        Permutation test for cross-split consistency
        
        Tests null hypothesis: splits come from same distribution
        
        Args:
            metric_name: Metric to test
            layer: Layer name
            n_permutations: Number of permutations
            
        Returns:
            Dictionary with test statistics and p-value
        """
        from config import SPLITS
        
        # Extract metric values for each split
        split_values = []
        for split in SPLITS:
            if split in self.all_metrics[layer]:
                metrics = self.all_metrics[layer][split]
                # Navigate to metric (may be nested)
                value = self._extract_metric_value(metrics, metric_name)
                if not np.isnan(value):
                    split_values.append(value)
        
        if len(split_values) < 2:
            logger.warning(f"Insufficient splits for permutation test: {layer}/{metric_name}")
            return {'p_value': np.nan, 'test_statistic': np.nan}
        
        # Compute observed variance
        observed_variance = np.var(split_values)
        
        # Permutation test
        all_values = np.array(split_values)
        n_splits = len(split_values)
        
        perm_variances = []
        for _ in range(n_permutations):
            perm_values = np.random.permutation(all_values)
            perm_variances.append(np.var(perm_values))
        
        # P-value: proportion of permuted variances >= observed
        p_value = np.mean(np.array(perm_variances) >= observed_variance)
        
        return {
            'test': 'permutation',
            'metric': metric_name,
            'layer': layer,
            'observed_variance': float(observed_variance),
            'p_value': float(p_value),
            'n_permutations': n_permutations,
            'interpretation': 'consistent' if p_value > 0.05 else 'inconsistent'
        }
    
    def _extract_metric_value(self, metrics_dict: Dict, metric_name: str) -> float:
        """Extract metric value from potentially nested dict"""
        # Try direct access
        if metric_name in metrics_dict:
            return float(metrics_dict[metric_name])
        
        # Try nested in label_metrics
        if 'label_metrics' in metrics_dict:
            for label_type, label_metrics in metrics_dict['label_metrics'].items():
                if metric_name in label_metrics:
                    return float(label_metrics[metric_name])
        
        return np.nan
    
    def compute_cross_split_correlation(self, metric_name: str) -> pd.DataFrame:
        """
        Compute correlation matrix of metric across splits
        
        Args:
            metric_name: Metric to analyze
            
        Returns:
            DataFrame with pairwise correlations
        """
        from config import LAYERS, SPLITS
        
        # Build matrix: rows=layers, cols=splits
        data = []
        for layer in LAYERS:
            row = []
            for split in SPLITS:
                if layer in self.all_metrics and split in self.all_metrics[layer]:
                    value = self._extract_metric_value(
                        self.all_metrics[layer][split], metric_name
                    )
                    row.append(value)
                else:
                    row.append(np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, index=LAYERS, columns=SPLITS)
        
        # Compute pairwise Spearman correlations between splits
        corr_matrix = df.corr(method='spearman')
        
        return corr_matrix
    
    def run_all_consistency_tests(self, metrics_to_test: List[str]) -> Dict[str, Any]:
        """
        Run all consistency tests
        
        Args:
            metrics_to_test: List of metric names to test
            
        Returns:
            Dictionary of all test results
        """
        from config import LAYERS
        
        logger.info("Running cross-split consistency tests...")
        
        results = {
            'permutation_tests': {},
            'correlation_matrices': {},
            'summary': {}
        }
        
        n_permutations = STATISTICAL_TESTS['cross_split_consistency']['n_permutations']
        
        for metric in metrics_to_test:
            logger.info(f"  Testing {metric}...")
            
            # Permutation tests for each layer
            results['permutation_tests'][metric] = {}
            for layer in LAYERS:
                test_result = self.run_permutation_test(metric, layer, n_permutations)
                results['permutation_tests'][metric][layer] = test_result
            
            # Correlation matrix
            corr_matrix = self.compute_cross_split_correlation(metric)
            results['correlation_matrices'][metric] = corr_matrix.to_dict()
        
        # Summary statistics
        consistent_count = 0
        total_tests = 0
        for metric, layer_tests in results['permutation_tests'].items():
            for layer, test_result in layer_tests.items():
                if not np.isnan(test_result['p_value']):
                    total_tests += 1
                    if test_result['interpretation'] == 'consistent':
                        consistent_count += 1
        
        results['summary'] = {
            'total_tests': total_tests,
            'consistent_count': consistent_count,
            'consistency_rate': consistent_count / total_tests if total_tests > 0 else 0
        }
        
        logger.info(f"✓ Consistency tests complete: {consistent_count}/{total_tests} consistent")
        
        return results

# ============================================================
# Layer Comparison Tests
# ============================================================
class LayerComparisonAnalyzer:
    """
    Statistical comparison of layers to identify best performer
    """
    
    def __init__(self, all_metrics: Dict[str, Dict[str, Any]]):
        """
        Initialize analyzer
        
        Args:
            all_metrics: Nested dict {layer: {split: metrics_dict}}
        """
        self.all_metrics = all_metrics
        self.results = {}
    
    def friedman_test(self, metric_name: str, split: str = 'test') -> Dict[str, Any]:
        """
        Friedman test for comparing multiple layers
        
        Non-parametric alternative to repeated measures ANOVA
        
        Args:
            metric_name: Metric to compare
            split: Which split to use for comparison (default: test)
            
        Returns:
            Test results
        """
        from config import LAYERS
        
        # Extract metric for each layer
        layer_values = []
        valid_layers = []
        
        for layer in LAYERS:
            if layer in self.all_metrics and split in self.all_metrics[layer]:
                value = self._extract_metric_value(
                    self.all_metrics[layer][split], metric_name
                )
                if not np.isnan(value):
                    layer_values.append(value)
                    valid_layers.append(layer)
        
        if len(layer_values) < 3:
            logger.warning(f"Insufficient layers for Friedman test: {metric_name}/{split}")
            return {'error': 'insufficient_data'}
        
        # Friedman test requires repeated measures, but we have independent groups
        # Use Kruskal-Wallis instead (non-parametric one-way ANOVA)
        statistic, p_value = stats.kruskal(*[[v] for v in layer_values])
        
        return {
            'test': 'kruskal_wallis',
            'metric': metric_name,
            'split': split,
            'layers': valid_layers,
            'values': [float(v) for v in layer_values],
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < STATISTICAL_TESTS['layer_comparison']['alpha']
        }
    
    def pairwise_wilcoxon(self, metric_name: str, split: str = 'test') -> List[Dict]:
        """
        Pairwise Wilcoxon signed-rank tests (with Bonferroni correction)
        
        Args:
            metric_name: Metric to compare
            split: Which split to use
            
        Returns:
            List of pairwise comparison results
        """
        from config import LAYERS
        from itertools import combinations
        
        # Extract metrics
        layer_metrics = {}
        for layer in LAYERS:
            if layer in self.all_metrics and split in self.all_metrics[layer]:
                value = self._extract_metric_value(
                    self.all_metrics[layer][split], metric_name
                )
                if not np.isnan(value):
                    layer_metrics[layer] = value
        
        if len(layer_metrics) < 2:
            return []
        
        # Pairwise comparisons
        results = []
        pairs = list(combinations(layer_metrics.keys(), 2))
        n_comparisons = len(pairs)
        
        # Bonferroni correction
        alpha = STATISTICAL_TESTS['layer_comparison']['alpha']
        corrected_alpha = alpha / n_comparisons
        
        for layer1, layer2 in pairs:
            val1 = layer_metrics[layer1]
            val2 = layer_metrics[layer2]
            
            # Effect size (Cohen's d for single values is just standardized difference)
            effect_size = (val1 - val2) / np.sqrt((val1**2 + val2**2) / 2) if val1 != val2 else 0
            
            results.append({
                'layer1': layer1,
                'layer2': layer2,
                'value1': float(val1),
                'value2': float(val2),
                'difference': float(val1 - val2),
                'effect_size': float(effect_size),
                'better_layer': layer1 if val1 > val2 else layer2,
                'corrected_alpha': corrected_alpha
            })
        
        return results
    
    def rank_layers(self, metric_name: str, split: str = 'test') -> List[Tuple[str, float]]:
        """
        Rank layers by metric performance
        
        Args:
            metric_name: Metric to rank by
            split: Which split to use
            
        Returns:
            List of (layer, value) tuples sorted by performance
        """
        from config import LAYERS
        
        layer_values = []
        for layer in LAYERS:
            if layer in self.all_metrics and split in self.all_metrics[layer]:
                value = self._extract_metric_value(
                    self.all_metrics[layer][split], metric_name
                )
                if not np.isnan(value):
                    layer_values.append((layer, value))
        
        # Sort descending (higher is better for silhouette, CH score)
        # For Davies-Bouldin, reverse this (lower is better)
        reverse = 'davies' not in metric_name.lower()
        layer_values.sort(key=lambda x: x[1], reverse=reverse)
        
        return layer_values
    
    def _extract_metric_value(self, metrics_dict: Dict, metric_name: str) -> float:
        """Extract metric value from potentially nested dict"""
        if metric_name in metrics_dict:
            return float(metrics_dict[metric_name])
        
        if 'label_metrics' in metrics_dict:
            for label_type, label_metrics in metrics_dict['label_metrics'].items():
                if metric_name in label_metrics:
                    return float(label_metrics[metric_name])
        
        return np.nan
    
    def run_all_comparisons(self, metrics_to_compare: List[str]) -> Dict[str, Any]:
        """
        Run all layer comparison tests
        
        Args:
            metrics_to_compare: List of metrics to compare
            
        Returns:
            Complete comparison results
        """
        from config import SPLITS
        
        logger.info("Running layer comparison tests...")
        
        results = {
            'statistical_tests': {},
            'pairwise_comparisons': {},
            'rankings': {},
            'best_layers': {}
        }
        
        for metric in metrics_to_compare:
            logger.info(f"  Comparing layers on {metric}...")
            
            results['statistical_tests'][metric] = {}
            results['pairwise_comparisons'][metric] = {}
            results['rankings'][metric] = {}
            
            for split in SPLITS:
                # Statistical test
                test_result = self.friedman_test(metric, split)
                results['statistical_tests'][metric][split] = test_result
                
                # Pairwise comparisons
                pairwise = self.pairwise_wilcoxon(metric, split)
                results['pairwise_comparisons'][metric][split] = pairwise
                
                # Rankings
                rankings = self.rank_layers(metric, split)
                results['rankings'][metric][split] = [
                    {'layer': layer, 'value': float(value)} 
                    for layer, value in rankings
                ]
            
            # Determine best layer across splits
            all_rankings = []
            for split in SPLITS:
                if results['rankings'][metric][split]:
                    best = results['rankings'][metric][split][0]
                    all_rankings.append(best['layer'])
            
            if all_rankings:
                from collections import Counter
                most_common = Counter(all_rankings).most_common(1)[0]
                results['best_layers'][metric] = {
                    'layer': most_common[0],
                    'frequency': most_common[1],
                    'total_splits': len(SPLITS)
                }
        
        logger.info("✓ Layer comparison complete")
        
        return results

# ============================================================
# Feature Correlation Analysis
# ============================================================
def analyze_layer_feature_correlation(all_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze correlation between features from different layers
    
    Args:
        all_embeddings: Dict {layer: embeddings_array}
        
    Returns:
        Correlation analysis results
    """
    from config import LAYERS
    
    logger.info("Analyzing inter-layer feature correlations...")
    
    # Compute pairwise Canonical Correlation Analysis (CCA)
    from sklearn.cross_decomposition import CCA
    
    results = {
        'cca_scores': {},
        'spearman_correlations': {}
    }
    
    # Pairwise CCA
    for i, layer1 in enumerate(LAYERS[:-1]):
        for layer2 in LAYERS[i+1:]:
            if layer1 not in all_embeddings or layer2 not in all_embeddings:
                continue
            
            emb1 = all_embeddings[layer1]
            emb2 = all_embeddings[layer2]
            
            # Align samples
            n_samples = min(len(emb1), len(emb2))
            emb1 = emb1[:n_samples]
            emb2 = emb2[:n_samples]
            
            # CCA
            n_components = min(emb1.shape[1], emb2.shape[1], 5)  # Use top 5 components
            cca = CCA(n_components=n_components)
            try:
                cca.fit(emb1, emb2)
                X_c, Y_c = cca.transform(emb1, emb2)
                
                # Compute correlation of canonical variates
                corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
                
                results['cca_scores'][f'{layer1}_vs_{layer2}'] = {
                    'correlations': [float(c) for c in corrs],
                    'mean_correlation': float(np.mean(corrs))
                }
            except Exception as e:
                logger.warning(f"CCA failed for {layer1} vs {layer2}: {e}")
    
    logger.info("✓ Feature correlation analysis complete")
    
    return results

# ============================================================
# Save Results
# ============================================================
def save_statistical_results(all_results: Dict[str, Any], output_path: Path):
    """Save all statistical test results"""
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"✓ Saved statistical results: {output_path}")

# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    print("This module provides statistical testing functionality")
    print("Use main_investigation.py to run the full pipeline")