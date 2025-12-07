"""
============================================================
UMAP Analyzer - Core UMAP Computation and Metrics
Handles UMAP projection and clustering quality evaluation
============================================================
"""
import numpy as np
import umap
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from itertools import product

from config import (
    UMAP_DEFAULT_PARAMS, 
    UMAP_GRID_PARAMS,
    LABEL_TYPES,
    ARTIFACT_TYPES,
    get_results_path
)
from utils import (
    setup_logging,
    safe_silhouette_score,
    safe_davies_bouldin_score,
    safe_calinski_harabasz_score,
    check_sufficient_samples,
    print_section_header,
    print_progress
)

logger = setup_logging()

# ============================================================
# UMAP Analyzer Class
# ============================================================
class UMAPAnalyzer:
    """
    Comprehensive UMAP analysis for a single layer/split combination
    CRITICAL FIX 2: Support pre-fitted UMAP reducers for val/test consistency
    """
    
    def __init__(self, 
                 layer: str,
                 split: str,
                 embeddings: np.ndarray,
                 labels: Dict[str, np.ndarray],
                 umap_params: Optional[Dict] = None,
                 fitted_reducer: Optional[umap.UMAP] = None,
                 primary_label_type: str = 'quality_class'):
        """
        Initialize UMAP Analyzer
        
        Args:
            layer: Layer name (stage1, stage2, etc.)
            split: Data split (train, val, test)
            embeddings: Feature embeddings [N, D]
            labels: Dictionary of labels (mos, quality_class, artifact flags)
            umap_params: UMAP parameters (uses label-specific default if None)
            fitted_reducer: Pre-fitted UMAP reducer (for val/test splits)
            primary_label_type: Primary label type for selecting default params
        """
        self.layer = layer
        self.split = split
        self.embeddings = embeddings
        self.labels = labels
        self.primary_label_type = primary_label_type
        self.fitted_reducer = fitted_reducer
        
        # CRITICAL FIX 1: Use label-specific parameters if not provided
        if umap_params is None:
            from config import UMAP_OPTIMIZED_PARAMS
            self.umap_params = UMAP_OPTIMIZED_PARAMS.get(
                primary_label_type, 
                UMAP_OPTIMIZED_PARAMS['quality_class']
            ).copy()
        else:
            self.umap_params = umap_params
        
        # Computed attributes
        self.umap_projection = None
        self.reducer = None
        self.metrics = {}
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info(f"✓ UMAPAnalyzer initialized: {layer}/{split}")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Primary label: {primary_label_type}")
        logger.info(f"  Using {'fitted' if fitted_reducer else 'new'} UMAP reducer")
        logger.info(f"  UMAP params: {self.umap_params}")
    
    def _validate_inputs(self):
        """Validate input data"""
        n_samples = self.embeddings.shape[0]
        
        for label_type, label_array in self.labels.items():
            if len(label_array) != n_samples:
                raise ValueError(
                    f"Label {label_type} length mismatch: "
                    f"embeddings={n_samples}, labels={len(label_array)}"
                )
        
        # Check for NaN/Inf
        if np.isnan(self.embeddings).any() or np.isinf(self.embeddings).any():
            raise ValueError("Embeddings contain NaN or Inf values")
    
    def fit_transform(self) -> np.ndarray:
        """
        Fit UMAP and transform embeddings
        CRITICAL FIX 2: Use pre-fitted reducer for val/test (consistent projections)
        
        Returns:
            2D UMAP projection [N, 2]
        """
        if self.fitted_reducer is not None:
            # Use pre-fitted reducer (for val/test)
            logger.info(f"Applying pre-fitted UMAP to {self.layer}/{self.split}...")
            self.reducer = self.fitted_reducer
            self.umap_projection = self.reducer.transform(self.embeddings)
            logger.info(f"✓ Transform complete using fitted reducer: {self.umap_projection.shape}")
        else:
            # Fit new reducer (for train only)
            logger.info(f"Fitting new UMAP on {self.layer}/{self.split}...")
            self.reducer = umap.UMAP(**self.umap_params)
            self.umap_projection = self.reducer.fit_transform(self.embeddings)
            logger.info(f"✓ Fit-transform complete: {self.umap_projection.shape}")
        
        return self.umap_projection
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive clustering quality metrics
        
        Returns:
            Dictionary of metrics for each label type
        """
        if self.umap_projection is None:
            raise ValueError("Must call fit_transform() before computing metrics")
        
        logger.info(f"Computing metrics for {self.layer}/{self.split}...")
        
        self.metrics = {
            'layer': self.layer,
            'split': self.split,
            'n_samples': len(self.embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'umap_params': self.umap_params,
            'label_metrics': {}
        }
        
        # Compute metrics for each label type
        for label_type in LABEL_TYPES:
            if label_type not in self.labels:
                logger.warning(f"Label type {label_type} not found, skipping")
                continue
            
            labels = self.labels[label_type]
            label_metrics = self._compute_label_metrics(labels, label_type)
            self.metrics['label_metrics'][label_type] = label_metrics
        
        logger.info(f"✓ Metrics computation complete")
        return self.metrics
    
    def _compute_label_metrics(self, labels: np.ndarray, label_type: str) -> Dict[str, Any]:
        """
        Compute metrics for a specific label type
        
        Args:
            labels: Label array
            label_type: Type of label (mos, quality_class, artifact flag)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'label_type': label_type,
            'n_unique_labels': len(np.unique(labels))
        }
        
        # For continuous labels (MOS), discretize for clustering metrics
        if label_type == 'mos':
            # Discretize MOS into bins for clustering evaluation
            labels_discrete = self._discretize_mos(labels)
            metrics['discretization'] = 'quartiles'
        else:
            labels_discrete = labels
        
        # Check if we have sufficient samples per class
        if not check_sufficient_samples(labels_discrete, min_samples_per_class=2):
            logger.warning(f"Insufficient samples for {label_type}, metrics may be unreliable")
        
        # Compute clustering metrics
        try:
            # Silhouette Score (higher is better, range [-1, 1])
            metrics['silhouette_score'] = safe_silhouette_score(
                self.umap_projection, labels_discrete
            )
            
            # Davies-Bouldin Index (lower is better, range [0, ∞))
            metrics['davies_bouldin_score'] = safe_davies_bouldin_score(
                self.umap_projection, labels_discrete
            )
            
            # Calinski-Harabasz Score (higher is better, range [0, ∞))
            metrics['calinski_harabasz_score'] = safe_calinski_harabasz_score(
                self.umap_projection, labels_discrete
            )
            
        except Exception as e:
            logger.error(f"Error computing metrics for {label_type}: {e}")
            metrics['error'] = str(e)
        
        # For binary labels (artifacts), compute additional metrics
        if 'flag' in label_type:
            metrics.update(self._compute_binary_metrics(labels))
        
        # For MOS, compute correlation with UMAP coordinates
        if label_type == 'mos':
            metrics.update(self._compute_mos_correlation(labels))
        
        return metrics
    
    def _discretize_mos(self, mos_scores: np.ndarray, n_bins: int = 4) -> np.ndarray:
        """
        Discretize MOS scores into bins for clustering evaluation
        
        Args:
            mos_scores: Continuous MOS scores
            n_bins: Number of bins (default: 4 quartiles)
            
        Returns:
            Discretized labels
        """
        # Use quantile-based binning
        labels = np.digitize(
            mos_scores, 
            bins=np.percentile(mos_scores, np.linspace(0, 100, n_bins + 1)[1:-1])
        )
        return labels
    
    def _compute_binary_metrics(self, binary_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics specific to binary labels (artifact flags)
        CRITICAL FIX 3: Use imbalance-aware metrics instead of silhouette
        
        Args:
            binary_labels: Binary (0/1) labels
            
        Returns:
            Dictionary of binary classification metrics
        """
        n_positive = np.sum(binary_labels == 1)
        n_negative = np.sum(binary_labels == 0)
        
        metrics = {
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'positive_ratio': float(n_positive / len(binary_labels)) if len(binary_labels) > 0 else 0.0,
            'class_balance': float(min(n_positive, n_negative) / max(n_positive, n_negative)) if max(n_positive, n_negative) > 0 else 0.0
        }
        
        # Skip if too few positive samples (< 10 is unreliable)
        if n_positive < 10:
            logger.warning(f"Only {n_positive} positive samples - skipping advanced metrics")
            metrics['skipped'] = True
            metrics['skip_reason'] = 'insufficient_positive_samples'
            return metrics
        
        # Compute centroid distance (existing metric, keep this)
        if n_positive > 0 and n_negative > 0:
            pos_centroid = self.umap_projection[binary_labels == 1].mean(axis=0)
            neg_centroid = self.umap_projection[binary_labels == 0].mean(axis=0)
            centroid_distance = np.linalg.norm(pos_centroid - neg_centroid)
            metrics['centroid_distance'] = float(centroid_distance)
        
        # CRITICAL FIX 3: Imbalance-aware classification metrics
        # Use UMAP space to train simple classifier and evaluate
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import (
                f1_score, precision_score, recall_score, 
                roc_auc_score, average_precision_score
            )
            
            # Standardize UMAP coordinates
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.umap_projection)
            
            # Train balanced logistic regression on UMAP space
            clf = LogisticRegression(
                class_weight='balanced',  # Handle imbalance
                random_state=42,
                max_iter=1000
            )
            clf.fit(X_scaled, binary_labels)
            
            # Predictions
            y_pred = clf.predict(X_scaled)
            y_proba = clf.predict_proba(X_scaled)[:, 1]
            
            # Compute imbalance-aware metrics
            metrics['f1_score'] = float(f1_score(binary_labels, y_pred, zero_division=0))
            metrics['precision'] = float(precision_score(binary_labels, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(binary_labels, y_pred, zero_division=0))
            metrics['roc_auc'] = float(roc_auc_score(binary_labels, y_proba))
            metrics['average_precision'] = float(average_precision_score(binary_labels, y_proba))
            
            # Interpretation help
            if metrics['roc_auc'] > 0.7:
                metrics['separation_quality'] = 'good'
            elif metrics['roc_auc'] > 0.6:
                metrics['separation_quality'] = 'moderate'
            else:
                metrics['separation_quality'] = 'poor'
            
        except Exception as e:
            logger.error(f"Imbalanced binary metrics computation failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _compute_mos_correlation(self, mos_scores: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation between MOS and UMAP coordinates
        
        Args:
            mos_scores: MOS scores
            
        Returns:
            Dictionary of correlation metrics
        """
        from scipy.stats import spearmanr, pearsonr
        
        metrics = {}
        
        # Correlation with each UMAP dimension
        for dim in range(self.umap_projection.shape[1]):
            umap_dim = self.umap_projection[:, dim]
            
            # Spearman correlation (rank-based)
            rho, p_value = spearmanr(mos_scores, umap_dim)
            metrics[f'spearman_dim{dim}'] = float(rho)
            metrics[f'spearman_pvalue_dim{dim}'] = float(p_value)
            
            # Pearson correlation (linear)
            r, p_value = pearsonr(mos_scores, umap_dim)
            metrics[f'pearson_dim{dim}'] = float(r)
            metrics[f'pearson_pvalue_dim{dim}'] = float(p_value)
        
        return metrics
    
    def save_results(self, output_dir: Path):
        """
        Save UMAP projection and metrics
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save UMAP projection
        projection_path = output_dir / f'umap_projection_{self.layer}_{self.split}.npy'
        np.save(projection_path, self.umap_projection)
        logger.info(f"✓ Saved UMAP projection: {projection_path}")
        
        # Save metrics
        metrics_path = output_dir / f'metrics_{self.layer}_{self.split}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"✓ Saved metrics: {metrics_path}")
        
        # Save UMAP model (for future transformations)
        import pickle
        model_path = output_dir / f'umap_model_{self.layer}_{self.split}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.reducer, f)
        logger.info(f"✓ Saved UMAP model: {model_path}")

# ============================================================
# UMAP Grid Search
# ============================================================
class UMAPGridSearch:
    """
    Perform grid search over UMAP hyperparameters
    """
    
    def __init__(self,
                 layer: str,
                 split: str,
                 embeddings: np.ndarray,
                 labels: Dict[str, np.ndarray],
                 param_grid: Dict[str, List] = None):
        """
        Initialize grid search
        
        Args:
            layer: Layer name
            split: Data split
            embeddings: Feature embeddings
            labels: Dictionary of labels
            param_grid: Parameter grid (uses default if None)
        """
        self.layer = layer
        self.split = split
        self.embeddings = embeddings
        self.labels = labels
        self.param_grid = param_grid or UMAP_GRID_PARAMS
        
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        
        logger.info(f"✓ UMAPGridSearch initialized: {layer}/{split}")
        logger.info(f"  Parameter grid: {self.param_grid}")
    
    def run(self, scoring_metric: str = 'silhouette_score',
            scoring_label: str = 'quality_class') -> Dict[str, Any]:
        """
        Run grid search
        
        Args:
            scoring_metric: Metric to optimize (default: silhouette_score)
            scoring_label: Label type to optimize for (default: quality_class)
            
        Returns:
            Best configuration and results
        """
        print_section_header(f"UMAP Grid Search: {self.layer}/{self.split}")
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_configs = len(param_combinations)
        logger.info(f"Testing {total_configs} parameter configurations...")
        
        for idx, param_vals in enumerate(param_combinations):
            # Create config dict
            config = dict(zip(param_names, param_vals))
            
            try:
                # Create analyzer with this config
                analyzer = UMAPAnalyzer(
                    layer=self.layer,
                    split=self.split,
                    embeddings=self.embeddings,
                    labels=self.labels,
                    umap_params=config
                )
                
                # Fit and compute metrics
                analyzer.fit_transform()
                metrics = analyzer.compute_metrics()
                
                # Extract score
                score = metrics['label_metrics'].get(scoring_label, {}).get(scoring_metric, -np.inf)
                
                # Store result
                result = {
                    'config': config,
                    'score': score,
                    'metrics': metrics
                }
                self.results.append(result)
                
                # Update best
                if not np.isnan(score) and score > self.best_score:
                    self.best_score = score
                    self.best_config = config
                
                print_progress(idx + 1, total_configs, 
                             prefix=f"Config {idx+1}/{total_configs} - Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Config {config} failed: {e}")
                continue
        
        print(f"\n✓ Grid search complete!")
        print(f"  Best {scoring_metric}: {self.best_score:.4f}")
        print(f"  Best config: {self.best_config}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def save_results(self, output_dir: Path):
        """Save grid search results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f'grid_search_{self.layer}_{self.split}.json'
        with open(results_path, 'w') as f:
            json.dump({
                'best_config': self.best_config,
                'best_score': float(self.best_score),
                'all_results': self.results
            }, f, indent=2, default=str)
        
        logger.info(f"✓ Saved grid search results: {results_path}")

# ============================================================
# Batch Processing Utilities
# ============================================================
def analyze_all_layers_splits(
    layers: List[str],
    splits: List[str],
    data_loader_fn: callable,
    run_grid_search: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all layer/split combinations
    
    Args:
        layers: List of layer names to analyze
        splits: List of splits to analyze
        data_loader_fn: Function that loads data for (layer, split)
        run_grid_search: Whether to run grid search (default: False)
        
    Returns:
        Nested dictionary of results
    """
    print_section_header("Analyzing All Layers and Splits")
    
    results = {}
    total = len(layers) * len(splits)
    current = 0
    
    for layer in layers:
        results[layer] = {}
        
        for split in splits:
            current += 1
            logger.info(f"\n[{current}/{total}] Processing {layer}/{split}")
            
            # Load data
            data = data_loader_fn(layer, split)
            embeddings = data['embeddings']
            labels = {k: v for k, v in data.items() if k != 'embeddings'}
            
            if run_grid_search:
                # Run grid search
                grid_search = UMAPGridSearch(layer, split, embeddings, labels)
                grid_results = grid_search.run()
                grid_search.save_results(get_results_path('grid_search'))
                
                # Use best config for final analysis
                best_config = grid_results['best_config']
            else:
                best_config = UMAP_DEFAULT_PARAMS
            
            # Run final analysis with best/default config
            analyzer = UMAPAnalyzer(layer, split, embeddings, labels, best_config)
            analyzer.fit_transform()
            analyzer.compute_metrics()
            analyzer.save_results(get_results_path('umap_results'))
            
            results[layer][split] = {
                'analyzer': analyzer,
                'metrics': analyzer.metrics,
                'projection': analyzer.umap_projection
            }
    
    print("\n✅ All layers and splits analyzed!")
    return results

# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    print("This module provides UMAP analysis functionality")
    print("Use main_investigation.py to run the full pipeline")