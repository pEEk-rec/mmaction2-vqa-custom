"""
============================================================
Utility Functions for Multi-Layer UMAP Investigation
============================================================
"""
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import json
import pickle

from config import LOG_CONFIG, COMPUTE_CONFIG

# ============================================================
# Logging Setup
# ============================================================
def setup_logging():
    """Configure logging for the investigation"""
    import sys
    
    # Create file handler (supports all Unicode)
    file_handler = logging.FileHandler(LOG_CONFIG['file'], encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))
    
    # Create console handler with fallback encoding for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))
    
    # Try UTF-8, fallback to ASCII if Windows console doesn't support it
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Fallback for older Python or restricted consoles
        pass
    
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG['level']),
        handlers=[file_handler, console_handler]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# Data Loading & Validation
# ============================================================
def load_embeddings(file_path: Path) -> np.ndarray:
    """
    Load embeddings with validation
    
    Args:
        file_path: Path to .npy file
        
    Returns:
        Loaded embeddings array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If array has invalid shape
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {file_path}")
    
    embeddings = np.load(file_path)
    
    # Validate shape
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    
    if embeddings.shape[0] == 0:
        raise ValueError(f"Empty embeddings array: {file_path}")
    
    # Check for NaN or Inf
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logger.warning(f"Found NaN/Inf values in {file_path}, cleaning...")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"✓ Loaded embeddings: {embeddings.shape} from {file_path.name}")
    return embeddings

def load_labels(file_path: Path, label_type: str) -> np.ndarray:
    """
    Load labels with type-specific validation
    
    Args:
        file_path: Path to .npy file
        label_type: Type of label (mos, quality_class, artifact flags)
        
    Returns:
        Loaded labels array
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Labels not found: {file_path}")
    
    labels = np.load(file_path, allow_pickle=True)
    
    # Type-specific validation
    if label_type == 'mos':
        # MOS should be continuous, typically 1-5
        if not (labels.min() >= 0 and labels.max() <= 5):
            logger.warning(f"MOS values outside expected range [0, 5]: [{labels.min()}, {labels.max()}]")
    
    elif label_type == 'quality_class':
        # Quality class should be categorical
        unique_classes = np.unique(labels)
        logger.info(f"  Quality classes found: {unique_classes}")
    
    elif 'flag' in label_type:
        # Artifact flags should be binary
        unique_vals = np.unique(labels)
        if not np.array_equal(unique_vals, [0, 1]) and not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [1]):
            logger.warning(f"Artifact flag {label_type} has non-binary values: {unique_vals}")
        labels = labels.astype(int)
    
    logger.info(f"✓ Loaded labels: {labels.shape} from {file_path.name}")
    return labels

def load_layer_data(split: str, layer: str, label_types: List[str]) -> Dict[str, np.ndarray]:
    """
    Load all data for a specific split and layer
    
    Args:
        split: Data split (train/val/test)
        layer: Layer name (stage1/stage2/...)
        label_types: List of label types to load
        
    Returns:
        Dictionary with embeddings and all labels
    """
    from config import get_feature_path
    
    data = {}
    
    # Load embeddings
    emb_path = get_feature_path(split, layer, 'embeddings')
    data['embeddings'] = load_embeddings(emb_path)
    
    # Load all labels
    n_samples = data['embeddings'].shape[0]
    for label_type in label_types:
        label_path = get_feature_path(split, layer, label_type)
        labels = load_labels(label_path, label_type)
        
        # Validate alignment
        if len(labels) != n_samples:
            raise ValueError(f"Label length mismatch: embeddings={n_samples}, {label_type}={len(labels)}")
        
        data[label_type] = labels
    
    logger.info(f"✓ Loaded complete dataset: {split}/{layer} - {n_samples} samples")
    return data

# ============================================================
# Feature Preprocessing
# ============================================================
def normalize_features(embeddings: np.ndarray, 
                      scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using standardization
    
    Args:
        embeddings: Raw embeddings [N, D]
        scaler: Pre-fitted scaler (for val/test), None for train
        
    Returns:
        Normalized embeddings and fitted scaler
    """
    if scaler is None:
        # Fit new scaler (for training set)
        scaler = StandardScaler()
        normalized = scaler.fit_transform(embeddings)
        logger.info(f"✓ Fitted new scaler: mean={scaler.mean_[:5]}, std={scaler.scale_[:5]}")
    else:
        # Use pre-fitted scaler (for val/test sets)
        normalized = scaler.transform(embeddings)
        logger.info(f"✓ Applied pre-fitted scaler")
    
    return normalized, scaler

# ============================================================
# Data Saving
# ============================================================
def save_umap_projection(projection: np.ndarray, 
                         file_path: Path,
                         metadata: Optional[Dict] = None):
    """Save UMAP projection with metadata"""
    save_data = {
        'projection': projection,
        'metadata': metadata or {}
    }
    np.save(file_path, save_data, allow_pickle=True)
    logger.info(f"✓ Saved UMAP projection: {file_path}")

def save_metrics(metrics: Dict[str, Any], file_path: Path):
    """Save metrics as JSON"""
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"✓ Saved metrics: {file_path}")

def save_scaler(scaler: StandardScaler, file_path: Path):
    """Save fitted scaler for consistent normalization"""
    with open(file_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ Saved scaler: {file_path}")

def load_scaler(file_path: Path) -> StandardScaler:
    """Load pre-fitted scaler"""
    with open(file_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"✓ Loaded scaler: {file_path}")
    return scaler

# ============================================================
# Metric Utilities
# ============================================================
def safe_silhouette_score(embeddings: np.ndarray, 
                          labels: np.ndarray,
                          metric: str = 'euclidean') -> float:
    """
    Compute silhouette score with error handling
    
    Args:
        embeddings: 2D UMAP projection
        labels: Cluster labels
        metric: Distance metric
        
    Returns:
        Silhouette score or NaN if computation fails
    """
    from sklearn.metrics import silhouette_score
    
    # Check if we have at least 2 clusters
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise if present
    if len(unique_labels) < 2:
        logger.warning(f"Cannot compute silhouette: only {len(unique_labels)} cluster(s)")
        return np.nan
    
    # Check if any cluster has only 1 sample
    for label in unique_labels:
        if np.sum(labels == label) < 2:
            logger.warning(f"Cluster {label} has < 2 samples, silhouette may be unreliable")
    
    try:
        score = silhouette_score(embeddings, labels, metric=metric)
        return float(score)
    except Exception as e:
        logger.error(f"Silhouette computation failed: {e}")
        return np.nan

def safe_davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Davies-Bouldin score with error handling"""
    from sklearn.metrics import davies_bouldin_score
    
    unique_labels = np.unique(labels[labels != -1])
    if len(unique_labels) < 2:
        return np.nan
    
    try:
        score = davies_bouldin_score(embeddings, labels)
        return float(score)
    except Exception as e:
        logger.error(f"Davies-Bouldin computation failed: {e}")
        return np.nan

def safe_calinski_harabasz_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Calinski-Harabasz score with error handling"""
    from sklearn.metrics import calinski_harabasz_score
    
    unique_labels = np.unique(labels[labels != -1])
    if len(unique_labels) < 2:
        return np.nan
    
    try:
        score = calinski_harabasz_score(embeddings, labels)
        return float(score)
    except Exception as e:
        logger.error(f"Calinski-Harabasz computation failed: {e}")
        return np.nan

# ============================================================
# Statistical Utilities
# ============================================================
def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size
    
    Args:
        group1, group2: Metric arrays for two groups
        
    Returns:
        Cohen's d effect size
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return mean_diff / pooled_std

def bootstrap_confidence_interval(data: np.ndarray,
                                  statistic_func: callable,
                                  n_bootstrap: int = 1000,
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic
    
    Args:
        data: Input data
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        
    Returns:
        (lower_bound, upper_bound) of confidence interval
    """
    np.random.seed(COMPUTE_CONFIG['seed'])
    bootstrap_stats = []
    
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return lower, upper

# ============================================================
# Progress Utilities
# ============================================================
def print_section_header(title: str, width: int = 70):
    """Print formatted section header"""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")

def print_progress(current: int, total: int, prefix: str = ""):
    """Print progress indicator"""
    percent = 100 * (current / total)
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()

# ============================================================
# Validation Utilities
# ============================================================
def validate_embeddings_labels_alignment(embeddings_dict: Dict[str, np.ndarray]) -> bool:
    """
    Validate that all arrays in dict have consistent first dimension
    
    Args:
        embeddings_dict: Dict with 'embeddings' and label arrays
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    n_samples = embeddings_dict['embeddings'].shape[0]
    
    for key, arr in embeddings_dict.items():
        if key == 'embeddings':
            continue
        if len(arr) != n_samples:
            raise ValueError(f"Length mismatch: embeddings={n_samples}, {key}={len(arr)}")
    
    return True

def check_sufficient_samples(labels: np.ndarray, min_samples_per_class: int = 2) -> bool:
    """Check if each class has sufficient samples for clustering metrics"""
    unique, counts = np.unique(labels, return_counts=True)
    insufficient = counts < min_samples_per_class
    
    if insufficient.any():
        bad_classes = unique[insufficient]
        bad_counts = counts[insufficient]
        logger.warning(f"Classes with < {min_samples_per_class} samples: {dict(zip(bad_classes, bad_counts))}")
        return False
    
    return True

# ============================================================
# Reproducibility
# ============================================================
def set_random_seeds(seed: int = None):
    """Set random seeds for reproducibility"""
    if seed is None:
        seed = COMPUTE_CONFIG['seed']
    
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"✓ Random seeds set to {seed}")

# ============================================================
# Memory Management
# ============================================================
def estimate_memory_usage(embeddings_shape: Tuple[int, int], dtype=np.float32) -> str:
    """Estimate memory usage for embeddings"""
    bytes_per_elem = np.dtype(dtype).itemsize
    total_bytes = np.prod(embeddings_shape) * bytes_per_elem
    
    # Convert to human-readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_bytes < 1024:
            return f"{total_bytes:.2f} {unit}"
        total_bytes /= 1024
    
    return f"{total_bytes:.2f} TB"

# ============================================================
# Initialize on import
# ============================================================
set_random_seeds()
logger.info("✓ Utilities module loaded successfully")