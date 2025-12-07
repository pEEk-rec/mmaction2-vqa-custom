"""
============================================================
Central Configuration for Multi-Layer UMAP Investigation
============================================================
"""
import os
from pathlib import Path

# ============================================================
# Directory Structure
# ============================================================
BASE_DIR = Path(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\UMAP_imp\swin_AI-VQA')
EMBEDDING_DIR = BASE_DIR / 'embeddings_workstation'
UMAP_DIR = BASE_DIR / 'UMAP_workstation'
RESULTS_DIR = UMAP_DIR / 'analysis_results'
PLOTS_DIR = UMAP_DIR / 'per_layer_plots'
COMPARISON_DIR = UMAP_DIR / 'comparative_plots'

# Create all directories
for dir_path in [EMBEDDING_DIR, UMAP_DIR, RESULTS_DIR, PLOTS_DIR, COMPARISON_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# Model Configuration - Video Swin-T Architecture
# ============================================================
LAYERS = ['stage1', 'stage2', 'stage3', 'stage4']

LAYER_CONFIG = {
    'stage1': {
        'dim': 96,
        'hook_name': 'backbone.layers.0',  # Adjust based on your model
        'description': 'Local window attention, fine-grained spatial features'
    },
    'stage2': {
        'dim': 192,
        'hook_name': 'backbone.layers.1',
        'description': 'Shifted windows, mid-level features'
    },
    'stage3': {
        'dim': 384,
        'hook_name': 'backbone.layers.2',
        'description': 'Higher semantic features'
    },
    'stage4': {
        'dim': 768,
        'hook_name': 'backbone.layers.3',
        'description': 'Abstract features before pooling'
    },
    'final': {
        'dim': 768,
        'hook_name': 'cls_head',  # Or wherever your final embedding comes from
        'description': 'Compressed quality representation'
    }
}

# ============================================================
# Dataset Configuration
# ============================================================
SPLITS = ['train', 'val', 'test']

ARTIFACT_TYPES = [
    'hallucination_flag',
    'lighting_flag',
    'spatial_flag',
    'rendering_flag'
]

LABEL_TYPES = ['mos', 'quality_class'] + ARTIFACT_TYPES

# Quality class mapping (adjust based on your dataset)
QUALITY_CLASS_MAPPING = {
    'good': 0,
    'poor': 1,
    'bad': 2
}

# ============================================================
# UMAP Configuration - LABEL-SPECIFIC OPTIMIZATION
# ============================================================
# CRITICAL FIX 1: Different UMAP parameters for different label types
# MOS (continuous) needs smooth gradients, artifacts (binary) need global separation

# Optimized parameters per label type (based on data characteristics)
UMAP_OPTIMIZED_PARAMS = {
    'mos': {
        'n_neighbors': 50,      # Large neighborhood for smooth gradients
        'min_dist': 0.5,        # High min_dist preserves continuity
        'n_components': 2,
        'metric': 'euclidean',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': False
    },
    'quality_class': {
        'n_neighbors': 30,      # Medium neighborhood for balanced clustering
        'min_dist': 0.1,        # Low for clear cluster separation
        'n_components': 2,
        'metric': 'euclidean',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': False
    },
    'hallucination_flag': {
        'n_neighbors': 100,     # LARGE for minority class (32.55% positive)
        'min_dist': 0.0,        # Zero for maximum separation
        'n_components': 2,
        'metric': 'manhattan',  # Manhattan often better for binary
        'random_state': 42,
        'n_jobs': -1,
        'verbose': False
    },
    'lighting_flag': {
        'n_neighbors': 100,     # Large for minority class (23.50% positive)
        'min_dist': 0.0,
        'n_components': 2,
        'metric': 'manhattan',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': False
    },
    'spatial_flag': {
        'n_neighbors': 150,     # VERY LARGE for severe imbalance (9.65% positive)
        'min_dist': 0.0,
        'n_components': 2,
        'metric': 'manhattan',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': False
    },
    'rendering_flag': {
        # SKIP: Only 0.10% positive (2/2000 samples) - not analyzable
        'skip': True,
        'reason': 'extreme_imbalance'
    }
}

# Default fallback (for backward compatibility)
UMAP_DEFAULT_PARAMS = UMAP_OPTIMIZED_PARAMS['quality_class'].copy()

# Hyperparameter grid for systematic search (per label type)
UMAP_GRID_PARAMS_PER_LABEL = {
    'mos': {
        'n_neighbors': [30, 50, 100],
        'min_dist': [0.3, 0.5, 0.7],
        'n_components': [2],
        'metric': ['euclidean'],
        'random_state': [42],
        'n_jobs': [-1],
        'verbose': [False]
    },
    'quality_class': {
        'n_neighbors': [15, 30, 50],
        'min_dist': [0.0, 0.1, 0.2],
        'n_components': [2],
        'metric': ['euclidean'],
        'random_state': [42],
        'n_jobs': [-1],
        'verbose': [False]
    },
    'artifacts': {  # Used for all artifact flags
        'n_neighbors': [50, 100, 150],
        'min_dist': [0.0, 0.05],
        'n_components': [2],
        'metric': ['euclidean', 'manhattan'],
        'random_state': [42],
        'n_jobs': [-1],
        'verbose': [False]
    }
}

# Legacy grid search params (for backward compatibility)
UMAP_GRID_PARAMS = UMAP_GRID_PARAMS_PER_LABEL['quality_class'].copy()

# Best config per layer will be determined during grid search
UMAP_BEST_CONFIGS = {}  # Will be populated during analysis

# ============================================================
# Clustering & Metrics Configuration
# ============================================================
CLUSTERING_METHODS = {
    'quality_class': {
        'method': 'ground_truth',  # Use provided labels
        'metric': 'silhouette'
    },
    'artifacts': {
        'method': 'ground_truth',
        'metric': 'silhouette'
    }
}

# Metrics to compute
METRICS_TO_COMPUTE = [
    'silhouette_score',
    'davies_bouldin_score',
    'calinski_harabasz_score',
    'adjusted_rand_score',
    'normalized_mutual_info'
]

# ============================================================
# Statistical Testing Configuration
# ============================================================
STATISTICAL_TESTS = {
    'cross_split_consistency': {
        'method': 'permutation_test',
        'n_permutations': 1000,
        'alpha': 0.05
    },
    'layer_comparison': {
        'method': 'friedman_test',  # Non-parametric for multiple groups
        'post_hoc': 'nemenyi',
        'alpha': 0.05
    },
    'correlation_analysis': {
        'method': 'spearman',  # Non-parametric correlation
        'alpha': 0.05
    }
}

# ============================================================
# Visualization Configuration
# ============================================================
VIS_CONFIG = {
    'figure_size': (8, 6),
    'dpi': 300,
    'format': 'png',
    'font_size': 12,
    'title_size': 14,
    'label_size': 11,
    'legend_size': 10,
    'marker_size': 20,
    'alpha': 0.7,
    'style': 'seaborn-v0_8-darkgrid'  # Matplotlib style
}

# Colormaps
COLORMAPS = {
    'mos': 'viridis',
    'quality_class': 'tab10',
    'artifacts': ['#3498db', '#e74c3c'],  # Blue for 0, Red for 1
    'heatmap': 'RdYlGn'
}

# ============================================================
# Logging Configuration
# ============================================================
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': RESULTS_DIR / 'investigation.log'
}

# ============================================================
# Computational Configuration
# ============================================================
COMPUTE_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'device': 'cuda',  # or 'cpu'
    'seed': 42,
    'normalize_features': True,  # Apply standardization before UMAP
    'cache_umap': True  # Cache UMAP transformations
}

# ============================================================
# Output File Naming Convention
# ============================================================
def get_feature_path(split, layer, label_type):
    """Generate standardized feature file path"""
    layer_dir = EMBEDDING_DIR / layer
    layer_dir.mkdir(parents=True, exist_ok=True)
    if label_type == 'embeddings':
        return layer_dir / f'features_{split}_embeddings.npy'
    else:
        return layer_dir / f'features_{split}_{label_type}.npy'

def get_umap_path(split, layer, label_type, suffix=''):
    """Generate standardized UMAP plot path"""
    layer_plot_dir = PLOTS_DIR / layer
    layer_plot_dir.mkdir(parents=True, exist_ok=True)
    filename = f'umap_{split}_{label_type}{suffix}.png'
    return layer_plot_dir / filename

def get_results_path(filename):
    """Generate standardized results path"""
    return RESULTS_DIR / filename

def get_comparison_path(filename):
    """Generate standardized comparison plot path"""
    return COMPARISON_DIR / filename

# ============================================================
# Validation
# ============================================================
def validate_config():
    """Validate configuration consistency"""
    errors = []
    
    # Check layer dimensions match expected
    expected_dims = [96, 192, 384, 768]
    actual_dims = [LAYER_CONFIG[layer]['dim'] for layer in LAYERS]
    if actual_dims != expected_dims:
        errors.append(f"Layer dimensions mismatch: expected {expected_dims}, got {actual_dims}")
    
    # Check all directories are accessible
    for dir_path in [BASE_DIR, EMBEDDING_DIR]:
        if not dir_path.exists():
            errors.append(f"Directory not found: {dir_path}")
    
    # Check UMAP grid has valid ranges
    if not all(isinstance(v, list) for v in UMAP_GRID_PARAMS.values()):
        errors.append("UMAP_GRID_PARAMS values must be lists")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Run validation on import
validate_config()

print("✓ Configuration loaded and validated successfully")
print(f"  Base directory: {BASE_DIR}")
print(f"  Layers to analyze: {LAYERS}")
print(f"  Splits to process: {SPLITS}")