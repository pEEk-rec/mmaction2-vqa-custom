"""
============================================================
Layer Comparator - Cross-Layer Analysis and Recommendations
Synthesizes results across layers to identify best features
============================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from config import LAYERS, SPLITS, ARTIFACT_TYPES, get_results_path
from utils import setup_logging, print_section_header

logger = setup_logging()

# ============================================================
# Layer Comparator Class
# ============================================================
class LayerComparator:
    """
    Comprehensive cross-layer analysis and recommendation system
    """
    
    def __init__(self, all_analysis_results: Dict[str, Dict[str, Any]]):
        """
        Initialize comparator
        
        Args:
            all_analysis_results: Complete results from UMAP analysis
                                 {layer: {split: {metrics, projection, labels}}}
        """
        self.all_results = all_analysis_results
        self.comparison_summary = {}
        self.recommendations = {}
        
        logger.info("✓ LayerComparator initialized")
        logger.info(f"  Analyzing {len(self.all_results)} layers across {len(SPLITS)} splits")
    
    def create_metrics_dataframe(self, metric_name: str, label_type: str = 'quality_class') -> pd.DataFrame:
        """
        Create DataFrame of a specific metric across all layers and splits
        
        Args:
            metric_name: Metric to extract (e.g., 'silhouette_score')
            label_type: Label type for the metric (e.g., 'quality_class')
            
        Returns:
            DataFrame with rows=layers, columns=splits
        """
        data = []
        
        for layer in LAYERS:
            row = []
            for split in SPLITS:
                if layer in self.all_results and split in self.all_results[layer]:
                    metrics = self.all_results[layer][split]['metrics']
                    
                    # Navigate to metric
                    if 'label_metrics' in metrics and label_type in metrics['label_metrics']:
                        value = metrics['label_metrics'][label_type].get(metric_name, np.nan)
                    else:
                        value = np.nan
                    
                    row.append(value)
                else:
                    row.append(np.nan)
            
            data.append(row)
        
        df = pd.DataFrame(data, index=LAYERS, columns=SPLITS)
        return df
    
    def compute_layer_scores(self) -> Dict[str, float]:
        """
        Compute aggregate score for each layer across all metrics
        
        Returns:
            Dictionary {layer: aggregate_score}
        """
        logger.info("Computing aggregate layer scores...")
        
        # Metrics to consider (with weights)
        metric_weights = {
            'silhouette_score': 0.4,  # Most important
            'calinski_harabasz_score': 0.3,
            'davies_bouldin_score': 0.3  # Lower is better, will negate
        }
        
        layer_scores = {}
        
        for layer in LAYERS:
            scores = []
            
            # Average across splits
            for split in SPLITS:
                if layer in self.all_results and split in self.all_results[layer]:
                    metrics = self.all_results[layer][split]['metrics']
                    
                    # Extract quality_class metrics
                    if 'label_metrics' in metrics and 'quality_class' in metrics['label_metrics']:
                        qc_metrics = metrics['label_metrics']['quality_class']
                        
                        # Compute weighted score
                        split_score = 0
                        for metric, weight in metric_weights.items():
                            value = qc_metrics.get(metric, np.nan)
                            
                            if not np.isnan(value):
                                # Normalize Davies-Bouldin (lower is better)
                                if metric == 'davies_bouldin_score':
                                    value = -value  # Negate so higher is better
                                
                                split_score += weight * value
                        
                        scores.append(split_score)
            
            # Average across splits
            if scores:
                layer_scores[layer] = np.mean(scores)
            else:
                layer_scores[layer] = -np.inf
        
        # Normalize scores to [0, 1]
        min_score = min(s for s in layer_scores.values() if s != -np.inf)
        max_score = max(layer_scores.values())
        
        for layer in layer_scores:
            if layer_scores[layer] != -np.inf:
                layer_scores[layer] = (layer_scores[layer] - min_score) / (max_score - min_score)
        
        logger.info("✓ Layer scores computed")
        return layer_scores
    
    def analyze_artifact_separation(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze which layer best separates each artifact type
        
        Returns:
            Dictionary {artifact_type: {layer: avg_silhouette}}
        """
        logger.info("Analyzing artifact separation across layers...")
        
        artifact_analysis = {artifact: {} for artifact in ARTIFACT_TYPES}
        
        for artifact in ARTIFACT_TYPES:
            for layer in LAYERS:
                scores = []
                
                for split in SPLITS:
                    if layer in self.all_results and split in self.all_results[layer]:
                        metrics = self.all_results[layer][split]['metrics']
                        
                        if 'label_metrics' in metrics and artifact in metrics['label_metrics']:
                            silhouette = metrics['label_metrics'][artifact].get('silhouette_score', np.nan)
                            if not np.isnan(silhouette):
                                scores.append(silhouette)
                
                if scores:
                    artifact_analysis[artifact][layer] = np.mean(scores)
                else:
                    artifact_analysis[artifact][layer] = np.nan
        
        logger.info("✓ Artifact separation analysis complete")
        return artifact_analysis
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations for best layers
        
        Returns:
            Dictionary with recommendations
        """
        print_section_header("Generating Recommendations")
        
        recommendations = {
            'overall_best_layer': None,
            'best_for_quality_assessment': None,
            'best_for_artifacts': {},
            'justifications': {},
            'confidence_scores': {}
        }
        
        # 1. Overall best layer (based on aggregate scores)
        layer_scores = self.compute_layer_scores()
        best_layer = max(layer_scores.items(), key=lambda x: x[1])
        recommendations['overall_best_layer'] = {
            'layer': best_layer[0],
            'score': float(best_layer[1])
        }
        
        logger.info(f"  Overall best: {best_layer[0]} (score: {best_layer[1]:.3f})")
        
        # 2. Best for quality assessment (highest avg silhouette on quality_class)
        quality_df = self.create_metrics_dataframe('silhouette_score', 'quality_class')
        quality_avg = quality_df.mean(axis=1)
        best_quality = quality_avg.idxmax()
        
        recommendations['best_for_quality_assessment'] = {
            'layer': best_quality,
            'avg_silhouette': float(quality_avg[best_quality])
        }
        
        logger.info(f"  Best for quality: {best_quality} (silhouette: {quality_avg[best_quality]:.3f})")
        
        # 3. Best for each artifact type
        artifact_analysis = self.analyze_artifact_separation()
        
        for artifact, layer_scores in artifact_analysis.items():
            valid_scores = {l: s for l, s in layer_scores.items() if not np.isnan(s)}
            if valid_scores:
                best_artifact_layer = max(valid_scores.items(), key=lambda x: x[1])
                recommendations['best_for_artifacts'][artifact] = {
                    'layer': best_artifact_layer[0],
                    'avg_silhouette': float(best_artifact_layer[1])
                }
                logger.info(f"  Best for {artifact}: {best_artifact_layer[0]} (silhouette: {best_artifact_layer[1]:.3f})")
        
        # 4. Justifications
        recommendations['justifications'] = self._generate_justifications(
            recommendations, layer_scores, quality_df, artifact_analysis
        )
        
        # 5. Confidence scores
        recommendations['confidence_scores'] = self._compute_confidence_scores(
            layer_scores, quality_df, artifact_analysis
        )
        
        self.recommendations = recommendations
        
        print("\n✅ Recommendations generated!")
        return recommendations
    
    def _generate_justifications(self,
                                 recommendations: Dict,
                                 layer_scores: Dict[str, float],
                                 quality_df: pd.DataFrame,
                                 artifact_analysis: Dict) -> Dict[str, str]:
        """Generate human-readable justifications for recommendations"""
        justifications = {}
        
        # Overall best
        best_layer = recommendations['overall_best_layer']['layer']
        score = recommendations['overall_best_layer']['score']
        
        justifications['overall'] = (
            f"{best_layer} achieves the highest aggregate score ({score:.3f}) "
            f"considering silhouette score, Calinski-Harabasz score, and Davies-Bouldin index "
            f"across all splits. This indicates strong cluster separation and quality discrimination."
        )
        
        # Quality assessment
        quality_layer = recommendations['best_for_quality_assessment']['layer']
        quality_score = recommendations['best_for_quality_assessment']['avg_silhouette']
        
        justifications['quality'] = (
            f"{quality_layer} shows the strongest clustering for quality classes "
            f"(avg silhouette: {quality_score:.3f}), suggesting this layer's features "
            f"are most effective for discriminating between good/poor/bad quality videos."
        )
        
        # Artifacts
        artifact_justifications = {}
        for artifact, info in recommendations['best_for_artifacts'].items():
            artifact_name = artifact.replace('_flag', '').replace('_', ' ').title()
            layer = info['layer']
            score = info['avg_silhouette']
            
            # Interpret layer semantic meaning
            layer_semantics = {
                'stage1': 'low-level texture and spatial features',
                'stage2': 'mid-level object parts and local patterns',
                'stage3': 'high-level semantic features',
                'stage4': 'abstract global features',
                'final': 'task-specific compressed representation'
            }
            
            semantic = layer_semantics.get(layer, 'features at this level')
            
            artifact_justifications[artifact] = (
                f"{layer} best separates {artifact_name} artifacts (silhouette: {score:.3f}). "
                f"This suggests {semantic} are most informative for detecting this artifact type."
            )
        
        justifications['artifacts'] = artifact_justifications
        
        return justifications
    
    def _compute_confidence_scores(self,
                                   layer_scores: Dict[str, float],
                                   quality_df: pd.DataFrame,
                                   artifact_analysis: Dict) -> Dict[str, float]:
        """Compute confidence scores for recommendations"""
        confidence = {}
        
        # Overall: based on score separation
        scores = sorted(layer_scores.values(), reverse=True)
        if len(scores) >= 2:
            gap = scores[0] - scores[1]
            confidence['overall'] = min(1.0, gap / 0.5)  # Normalize by expected gap
        else:
            confidence['overall'] = 0.5
        
        # Quality: based on cross-split consistency
        quality_std = quality_df.std(axis=1).mean()
        confidence['quality'] = max(0.0, 1.0 - quality_std)
        
        # Artifacts: average confidence across artifacts
        artifact_confidences = []
        for artifact, layer_scores_art in artifact_analysis.items():
            valid_scores = [s for s in layer_scores_art.values() if not np.isnan(s)]
            if len(valid_scores) >= 2:
                scores_sorted = sorted(valid_scores, reverse=True)
                gap = scores_sorted[0] - scores_sorted[1]
                conf = min(1.0, gap / 0.3)
                artifact_confidences.append(conf)
        
        confidence['artifacts'] = np.mean(artifact_confidences) if artifact_confidences else 0.5
        
        return confidence
    
    def create_comparison_summary(self) -> pd.DataFrame:
        """
        Create comprehensive comparison summary table
        
        Returns:
            DataFrame summarizing all metrics across layers
        """
        logger.info("Creating comparison summary table...")
        
        summary_data = []
        
        for layer in LAYERS:
            row = {'layer': layer}
            
            # Aggregate metrics across splits
            metrics_to_aggregate = [
                ('quality_class', 'silhouette_score'),
                ('quality_class', 'calinski_harabasz_score'),
                ('quality_class', 'davies_bouldin_score'),
                ('mos', 'spearman_dim0'),
                ('mos', 'spearman_dim1')
            ]
            
            for label_type, metric_name in metrics_to_aggregate:
                values = []
                for split in SPLITS:
                    if layer in self.all_results and split in self.all_results[layer]:
                        metrics = self.all_results[layer][split]['metrics']
                        if 'label_metrics' in metrics and label_type in metrics['label_metrics']:
                            value = metrics['label_metrics'][label_type].get(metric_name, np.nan)
                            if not np.isnan(value):
                                values.append(value)
                
                if values:
                    row[f'{label_type}_{metric_name}_mean'] = np.mean(values)
                    row[f'{label_type}_{metric_name}_std'] = np.std(values)
                else:
                    row[f'{label_type}_{metric_name}_mean'] = np.nan
                    row[f'{label_type}_{metric_name}_std'] = np.nan
            
            # Artifact metrics
            for artifact in ARTIFACT_TYPES:
                values = []
                for split in SPLITS:
                    if layer in self.all_results and split in self.all_results[layer]:
                        metrics = self.all_results[layer][split]['metrics']
                        if 'label_metrics' in metrics and artifact in metrics['label_metrics']:
                            value = metrics['label_metrics'][artifact].get('silhouette_score', np.nan)
                            if not np.isnan(value):
                                values.append(value)
                
                artifact_short = artifact.replace('_flag', '')
                row[f'{artifact_short}_silhouette_mean'] = np.mean(values) if values else np.nan
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        self.comparison_summary = summary_df
        
        logger.info("✓ Comparison summary created")
        return summary_df
    
    def save_results(self, output_dir: Path):
        """
        Save all comparison results
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save summary DataFrame
        summary_path = output_dir / 'layer_comparison_summary.csv'
        self.comparison_summary.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved comparison summary: {summary_path}")
        
        # 2. Save recommendations
        recommendations_path = output_dir / 'recommendations.json'
        with open(recommendations_path, 'w') as f:
            json.dump(self.recommendations, f, indent=2, default=str)
        logger.info(f"✓ Saved recommendations: {recommendations_path}")
        
        # 3. Save human-readable report
        report_path = output_dir / 'recommendations_report.txt'
        self._generate_text_report(report_path)
        logger.info(f"✓ Saved text report: {report_path}")
    
    def _generate_text_report(self, output_path: Path):
        """Generate human-readable text report"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MULTI-LAYER UMAP INVESTIGATION - RECOMMENDATIONS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overall best
            f.write("OVERALL BEST LAYER\n")
            f.write("-" * 70 + "\n")
            overall = self.recommendations['overall_best_layer']
            f.write(f"Layer: {overall['layer']}\n")
            f.write(f"Score: {overall['score']:.4f}\n")
            f.write(f"\nJustification:\n{self.recommendations['justifications']['overall']}\n\n")
            
            # Quality assessment
            f.write("\nBEST LAYER FOR QUALITY ASSESSMENT\n")
            f.write("-" * 70 + "\n")
            quality = self.recommendations['best_for_quality_assessment']
            f.write(f"Layer: {quality['layer']}\n")
            f.write(f"Avg Silhouette: {quality['avg_silhouette']:.4f}\n")
            f.write(f"\nJustification:\n{self.recommendations['justifications']['quality']}\n\n")
            
            # Artifacts
            f.write("\nBEST LAYERS FOR ARTIFACT DETECTION\n")
            f.write("-" * 70 + "\n")
            for artifact, info in self.recommendations['best_for_artifacts'].items():
                artifact_name = artifact.replace('_flag', '').replace('_', ' ').title()
                f.write(f"\n{artifact_name}:\n")
                f.write(f"  Layer: {info['layer']}\n")
                f.write(f"  Avg Silhouette: {info['avg_silhouette']:.4f}\n")
                f.write(f"  Justification: {self.recommendations['justifications']['artifacts'][artifact]}\n")
            
            # Confidence
            f.write("\n\nCONFIDENCE SCORES\n")
            f.write("-" * 70 + "\n")
            for key, conf in self.recommendations['confidence_scores'].items():
                f.write(f"{key.capitalize()}: {conf:.2%}\n")
            
            f.write("\n" + "="*70 + "\n")

# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    print("This module provides layer comparison functionality")
    print("Use main_investigation.py to run the full pipeline")