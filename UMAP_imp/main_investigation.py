"""
============================================================
Main Investigation Pipeline
Orchestrates complete multi-layer UMAP analysis workflow
============================================================
"""
import sys
from pathlib import Path
import argparse
import time
from typing import Dict, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    LAYERS, SPLITS, LABEL_TYPES, ARTIFACT_TYPES,
    UMAP_DEFAULT_PARAMS, UMAP_GRID_PARAMS,
    RESULTS_DIR, PLOTS_DIR, COMPARISON_DIR,
    get_feature_path
)
from utils import (
    setup_logging,
    load_layer_data,
    print_section_header,
    print_progress,
    set_random_seeds
)
from umap_analyzer import UMAPAnalyzer, UMAPGridSearch
from visualization import (
    visualize_layer_split,
    plot_layer_comparison_grid,
    plot_metrics_heatmap,
    plot_layer_comparison_bar,
    create_all_visualizations
)
from statistical_tests import (
    CrossSplitConsistencyAnalyzer,
    LayerComparisonAnalyzer,
    analyze_layer_feature_correlation,
    save_statistical_results
)
from layer_comparator import LayerComparator

# Setup logging
logger = setup_logging()

# ============================================================
# Main Investigation Pipeline
# ============================================================
class InvestigationPipeline:
    """
    Complete multi-layer UMAP investigation pipeline
    """
    
    def __init__(self, 
                 run_grid_search: bool = False,
                 skip_visualizations: bool = False,
                 layers_to_analyze: list = None,
                 splits_to_analyze: list = None):
        """
        Initialize pipeline
        
        Args:
            run_grid_search: Whether to perform UMAP hyperparameter search
            skip_visualizations: Skip visualization generation (faster)
            layers_to_analyze: Specific layers to analyze (None = all)
            splits_to_analyze: Specific splits to analyze (None = all)
        """
        self.run_grid_search = run_grid_search
        self.skip_visualizations = skip_visualizations
        self.layers = layers_to_analyze or LAYERS
        self.splits = splits_to_analyze or SPLITS
        
        # Results storage
        self.all_results = {}
        self.best_umap_configs = {}
        self.statistical_results = {}
        self.comparison_results = {}
        
        logger.info("="*70)
        logger.info("MULTI-LAYER UMAP INVESTIGATION PIPELINE")
        logger.info("="*70)
        logger.info(f"Layers to analyze: {self.layers}")
        logger.info(f"Splits to analyze: {self.splits}")
        logger.info(f"Grid search: {run_grid_search}")
        logger.info(f"Visualizations: {not skip_visualizations}")
        logger.info("="*70 + "\n")
    
    def validate_data_availability(self) -> bool:
        """
        Validate that all required data files exist
        
        Returns:
            True if all data available, False otherwise
        """
        print_section_header("Validating Data Availability")
        
        missing_files = []
        
        for layer in self.layers:
            for split in self.splits:
                # Check embeddings
                emb_path = get_feature_path(split, layer, 'embeddings')
                if not emb_path.exists():
                    missing_files.append(str(emb_path))
                
                # Check labels
                for label_type in LABEL_TYPES:
                    label_path = get_feature_path(split, layer, label_type)
                    if not label_path.exists():
                        missing_files.append(str(label_path))
        
        if missing_files:
            logger.error("Missing data files:")
            for f in missing_files[:10]:  # Show first 10
                logger.error(f"  - {f}")
            if len(missing_files) > 10:
                logger.error(f"  ... and {len(missing_files) - 10} more")
            return False
        
        logger.info("✅ All data files found!")
        return True
    
    def run_umap_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Run UMAP analysis for all layers and splits
        CRITICAL FIX: Fit on train, transform val/test with same reducer
        
        Returns:
            Complete analysis results
        """
        print_section_header("UMAP Analysis with Unified Projections")
        
        from config import UMAP_OPTIMIZED_PARAMS, ARTIFACT_TYPES
        
        for layer in self.layers:
            self.all_results[layer] = {}
            
            # Step 1: Process TRAIN split FIRST and fit UMAP reducers
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {layer} - TRAIN split (fitting reducers)")
            logger.info(f"{'='*70}")
            
            try:
                # Load train data
                train_data = load_layer_data('train', layer, LABEL_TYPES)
                train_embeddings = train_data['embeddings']
                train_labels = {k: v for k, v in train_data.items() if k != 'embeddings'}
                
                # Store fitted reducers per label type
                fitted_reducers = {}
                train_projections = {}
                
                # Fit UMAP for each label type with optimized parameters
                label_types_to_process = ['mos', 'quality_class'] + ARTIFACT_TYPES
                
                for label_type in label_types_to_process:
                    # Check if should skip (e.g., rendering_flag with extreme imbalance)
                    if UMAP_OPTIMIZED_PARAMS.get(label_type, {}).get('skip'):
                        logger.warning(f"  Skipping {label_type}: {UMAP_OPTIMIZED_PARAMS[label_type].get('reason')}")
                        continue
                    
                    logger.info(f"\n  Fitting UMAP for {label_type}...")
                    
                    # Get label-specific parameters
                    if label_type in UMAP_OPTIMIZED_PARAMS:
                        umap_params = UMAP_OPTIMIZED_PARAMS[label_type].copy()
                    elif 'flag' in label_type:
                        # Use artifact params for unlisted flags
                        umap_params = UMAP_OPTIMIZED_PARAMS['hallucination_flag'].copy()
                    else:
                        umap_params = UMAP_DEFAULT_PARAMS.copy()
                    
                    # Grid search option (if enabled)
                    if self.run_grid_search:
                        logger.info(f"    Running grid search for {label_type}...")
                        grid_search = UMAPGridSearch(layer, 'train', train_embeddings, train_labels)
                        grid_results = grid_search.run(
                            scoring_metric='silhouette_score' if label_type == 'quality_class' else 'roc_auc',
                            scoring_label=label_type
                        )
                        grid_search.save_results(RESULTS_DIR / 'grid_search')
                        umap_params = grid_results['best_config']
                        logger.info(f"    Best config: {umap_params}")
                    
                    # Fit UMAP on train data
                    analyzer_train = UMAPAnalyzer(
                        layer=layer,
                        split='train',
                        embeddings=train_embeddings,
                        labels=train_labels,
                        umap_params=umap_params,
                        fitted_reducer=None,  # Fit new
                        primary_label_type=label_type
                    )
                    
                    projection_train = analyzer_train.fit_transform()
                    metrics_train = analyzer_train.compute_metrics()
                    
                    # Store the fitted reducer for val/test
                    fitted_reducers[label_type] = analyzer_train.reducer
                    train_projections[label_type] = projection_train
                    
                    logger.info(f"    ✓ {label_type} train UMAP fitted")
                
                # Store train results
                self.all_results[layer]['train'] = {
                    'projections': train_projections,
                    'labels': train_labels,
                    'embeddings': train_embeddings,
                    'fitted_reducers': fitted_reducers
                }
                
                logger.info(f"✓ Train split complete: {len(fitted_reducers)} reducers fitted")
                
            except Exception as e:
                logger.error(f"✗ Failed on {layer}/train: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Step 2: Process VAL and TEST splits using fitted reducers
            for split in ['val', 'test']:
                if split not in self.splits:
                    continue
                
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing {layer} - {split.upper()} split (applying fitted reducers)")
                logger.info(f"{'='*70}")
                
                try:
                    # Load data
                    data = load_layer_data(split, layer, LABEL_TYPES)
                    embeddings = data['embeddings']
                    labels = {k: v for k, v in data.items() if k != 'embeddings'}
                    
                    # Apply fitted reducers from train
                    projections = {}
                    
                    for label_type, fitted_reducer in fitted_reducers.items():
                        logger.info(f"  Transforming {label_type} using train reducer...")
                        
                        # Apply pre-fitted UMAP
                        analyzer = UMAPAnalyzer(
                            layer=layer,
                            split=split,
                            embeddings=embeddings,
                            labels=labels,
                            umap_params=None,  # Not used when fitted_reducer provided
                            fitted_reducer=fitted_reducer,  # Use train reducer
                            primary_label_type=label_type
                        )
                        
                        projection = analyzer.fit_transform()  # Actually just transform
                        metrics = analyzer.compute_metrics()
                        analyzer.save_results(RESULTS_DIR / 'umap_results')
                        
                        projections[label_type] = projection
                        
                        logger.info(f"    ✓ {label_type} transformed")
                    
                    # Store results
                    self.all_results[layer][split] = {
                        'projections': projections,
                        'labels': labels,
                        'embeddings': embeddings
                    }
                    
                    logger.info(f"✓ {split} split complete")
                    
                except Exception as e:
                    logger.error(f"✗ Failed on {layer}/{split}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        logger.info("\n✅ UMAP analysis complete with unified projections!")
        logger.info("   Train: Fitted reducers")
        logger.info("   Val/Test: Applied train reducers (consistent projections)")
        
        return self.all_results
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        if self.skip_visualizations:
            logger.info("Skipping visualizations (as requested)")
            return
        
        print_section_header("Generating Visualizations")
        
        # 1. Individual layer/split plots
        for layer in self.layers:
            for split in self.splits:
                if layer in self.all_results and split in self.all_results[layer]:
                    try:
                        # FIX: Pass 'projections' not 'projection'
                        visualize_layer_split(
                            layer,
                            split,
                            self.all_results[layer][split]['projections'],  # <- Changed from 'projection'
                            self.all_results[layer][split]['labels'],
                            PLOTS_DIR
                        )
                    except Exception as e:
                        logger.error(f"Visualization failed for {layer}/{split}: {e}")
        
        # 2. Comparison grids
        try:
            for label_type in ['mos', 'quality_class'] + ARTIFACT_TYPES:
                plot_layer_comparison_grid(
                    self.all_results,
                    label_type,
                    COMPARISON_DIR / f'layer_comparison_{label_type}.png'
                )
        except Exception as e:
            logger.error(f"Comparison grid generation failed: {e}")
        
        # 3. Metrics heatmaps
        try:
            from layer_comparator import LayerComparator
            comparator = LayerComparator(self.all_results)
            
            for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']:
                df = comparator.create_metrics_dataframe(metric, 'quality_class')
                plot_metrics_heatmap(
                    df,
                    metric,
                    COMPARISON_DIR / f'heatmap_{metric}.png'
                )
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
        
        logger.info("✅ Visualizations complete!")
    
    def run_statistical_tests(self):
        """Run comprehensive statistical tests"""
        print_section_header("Statistical Testing")
        
        self.statistical_results = {}
        
        # 1. Cross-split consistency
        try:
            logger.info("Testing cross-split consistency...")
            consistency_analyzer = CrossSplitConsistencyAnalyzer(self.all_results)
            
            metrics_to_test = [
                'silhouette_score',
                'calinski_harabasz_score',
                'davies_bouldin_score'
            ]
            
            consistency_results = consistency_analyzer.run_all_consistency_tests(metrics_to_test)
            self.statistical_results['consistency'] = consistency_results
            
            logger.info(f"  ✓ Consistency rate: {consistency_results['summary']['consistency_rate']:.1%}")
        
        except Exception as e:
            logger.error(f"Consistency testing failed: {e}")
        
        # 2. Layer comparison
        try:
            logger.info("Comparing layers statistically...")
            comparison_analyzer = LayerComparisonAnalyzer(self.all_results)
            
            comparison_results = comparison_analyzer.run_all_comparisons(metrics_to_test)
            self.statistical_results['layer_comparison'] = comparison_results
            
            # Print best layers
            for metric, info in comparison_results['best_layers'].items():
                logger.info(f"  Best for {metric}: {info['layer']}")
        
        except Exception as e:
            logger.error(f"Layer comparison failed: {e}")
        
        # 3. Feature correlation analysis
        try:
            logger.info("Analyzing inter-layer feature correlations...")
            
            # Extract embeddings for correlation analysis
            train_embeddings = {}
            for layer in self.layers:
                if layer in self.all_results and 'train' in self.all_results[layer]:
                    train_embeddings[layer] = self.all_results[layer]['train']['embeddings']
            
            correlation_results = analyze_layer_feature_correlation(train_embeddings)
            self.statistical_results['feature_correlation'] = correlation_results
            
        except Exception as e:
            logger.error(f"Feature correlation analysis failed: {e}")
        
        # Save all statistical results
        save_statistical_results(
            self.statistical_results,
            RESULTS_DIR / 'statistical_tests.json'
        )
        
        logger.info("✅ Statistical testing complete!")
    
    def generate_recommendations(self):
        """Generate final recommendations"""
        print_section_header("Generating Recommendations")
        
        try:
            comparator = LayerComparator(self.all_results)
            
            # Create comparison summary
            summary_df = comparator.create_comparison_summary()
            
            # Generate recommendations
            recommendations = comparator.generate_recommendations()
            
            # Save results
            comparator.save_results(RESULTS_DIR)
            
            self.comparison_results = {
                'summary': summary_df.to_dict(),
                'recommendations': recommendations
            }
            
            # Print summary
            logger.info("\n" + "="*70)
            logger.info("RECOMMENDATIONS SUMMARY")
            logger.info("="*70)
            
            overall = recommendations['overall_best_layer']
            logger.info(f"\n🏆 Overall Best Layer: {overall['layer']} (score: {overall['score']:.3f})")
            
            quality = recommendations['best_for_quality_assessment']
            logger.info(f"\n📊 Best for Quality Assessment: {quality['layer']}")
            logger.info(f"   Avg Silhouette: {quality['avg_silhouette']:.3f}")
            
            logger.info("\n🔍 Best for Artifact Detection:")
            for artifact, info in recommendations['best_for_artifacts'].items():
                artifact_name = artifact.replace('_flag', '').replace('_', ' ').title()
                logger.info(f"   {artifact_name}: {info['layer']} (silhouette: {info['avg_silhouette']:.3f})")
            
            logger.info("\n💯 Confidence Scores:")
            for key, conf in recommendations['confidence_scores'].items():
                logger.info(f"   {key.capitalize()}: {conf:.1%}")
            
            logger.info("\n" + "="*70)
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
        
        logger.info("✅ Recommendations generated!")
    
    def save_pipeline_summary(self):
        """Save complete pipeline summary"""
        print_section_header("Saving Pipeline Summary")
        
        summary = {
            'pipeline_config': {
                'layers_analyzed': self.layers,
                'splits_analyzed': self.splits,
                'grid_search_performed': self.run_grid_search,
                'visualizations_generated': not self.skip_visualizations
            },
            'best_umap_configs': self.best_umap_configs,
            'statistical_results_summary': {
                'consistency_rate': self.statistical_results.get('consistency', {}).get('summary', {}).get('consistency_rate'),
                'best_layers': self.statistical_results.get('layer_comparison', {}).get('best_layers', {})
            },
            'final_recommendations': self.comparison_results.get('recommendations', {})
        }
        
        summary_path = RESULTS_DIR / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✓ Saved pipeline summary: {summary_path}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete investigation pipeline
        
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        # Validate data
        if not self.validate_data_availability():
            logger.error("❌ Data validation failed! Please ensure all feature files exist.")
            logger.error("    Run extract_multilayer_features.py first.")
            return None
        
        # Run pipeline stages
        try:
            # 1. UMAP Analysis
            self.run_umap_analysis()
            
            # 2. Visualizations
            self.generate_visualizations()
            
            # 3. Statistical Tests
            self.run_statistical_tests()
            
            # 4. Generate Recommendations
            self.generate_recommendations()
            
            # 5. Save Summary
            self.save_pipeline_summary()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        elapsed = time.time() - start_time
        
        # Final summary
        print_section_header("PIPELINE COMPLETE")
        logger.info(f"Total execution time: {elapsed/60:.1f} minutes")
        logger.info(f"\nResults saved to: {RESULTS_DIR}")
        logger.info(f"Visualizations saved to: {PLOTS_DIR}")
        logger.info(f"Comparisons saved to: {COMPARISON_DIR}")
        
        logger.info("\n📁 Key output files:")
        logger.info(f"  - Recommendations: {RESULTS_DIR / 'recommendations_report.txt'}")
        logger.info(f"  - Summary Table: {RESULTS_DIR / 'layer_comparison_summary.csv'}")
        logger.info(f"  - Statistical Tests: {RESULTS_DIR / 'statistical_tests.json'}")
        logger.info(f"  - Pipeline Summary: {RESULTS_DIR / 'pipeline_summary.json'}")
        
        return {
            'umap_results': self.all_results,
            'statistical_results': self.statistical_results,
            'comparison_results': self.comparison_results
        }

# ============================================================
# CLI Interface
# ============================================================
def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="Multi-Layer UMAP Investigation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full investigation with grid search
  python main_investigation.py --grid-search
  
  # Quick run with default parameters
  python main_investigation.py
  
  # Analyze specific layers only
  python main_investigation.py --layers stage1 stage3 final
  
  # Analyze only test split (faster)
  python main_investigation.py --splits test
  
  # Skip visualizations for faster execution
  python main_investigation.py --no-viz
        """
    )
    
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Perform UMAP hyperparameter grid search (slower but finds optimal configs)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation (faster, analysis only)'
    )
    
    parser.add_argument(
        '--layers',
        nargs='+',
        choices=LAYERS,
        default=None,
        help='Specific layers to analyze (default: all)'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=SPLITS,
        default=None,
        help='Specific splits to analyze (default: all)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seeds(args.seed)
    
    # Initialize and run pipeline
    pipeline = InvestigationPipeline(
        run_grid_search=args.grid_search,
        skip_visualizations=args.no_viz,
        layers_to_analyze=args.layers,
        splits_to_analyze=args.splits
    )
    
    results = pipeline.run()
    
    if results is None:
        sys.exit(1)
    
    logger.info("\n✅ Investigation pipeline completed successfully!")
    sys.exit(0)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    main()