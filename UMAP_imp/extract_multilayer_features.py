"""
============================================================
Swin-T Feature Extraction WITH CSV Labels
Loads actual labels from annotation files
============================================================
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict
from tqdm import tqdm
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

# Block problematic imports
from unittest.mock import MagicMock
mock = MagicMock()
sys.modules['mmaction.models.multimodal'] = mock
sys.modules['mmaction.models.multimodal.vindlu'] = mock
sys.modules['mmpretrain.models.multimodal'] = mock

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS

from config import LAYERS, SPLITS, get_feature_path, COMPUTE_CONFIG
from utils import setup_logging, print_section_header, save_scaler, normalize_features

logger = setup_logging()

# ============================================================
# Swin-T Feature Extractor (same as before)
# ============================================================
class SwinFeatureExtractor:
    """Extract features from Video Swin Transformer"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = 'cuda'):
        self.device = device
        logger.info(f"Loading Swin-T model...")
        
        cfg = Config.fromfile(config_path)
        self.model = MODELS.build(cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location='cpu')
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.cfg = cfg
        self.features = OrderedDict()
        self.hooks = []
        
        self._register_hooks()
        
        logger.info(f"✓ Model loaded with {len(self.hooks)} hooks")
    
    def _register_hooks(self):
        """Register hooks - handles (B, D, H, W, C) format"""
        
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    feat = output
                    if feat.dim() == 5:  # (B, D, H, W, C)
                        feat = feat.mean(dim=[1, 2, 3])  # (B, C)
                    elif feat.dim() == 4:
                        feat = feat.mean(dim=[1, 2])
                    elif feat.dim() == 3:
                        feat = feat.mean(dim=1)
                    elif feat.dim() == 2:
                        pass
                    else:
                        return
                    
                    self.features[name] = feat.detach().cpu()
                    
                elif isinstance(output, (tuple, list)):
                    feat = output[0]
                    if isinstance(feat, torch.Tensor) and feat.dim() == 5:
                        feat = feat.mean(dim=[1, 2, 3])
                        self.features[name] = feat.detach().cpu()
            
            return hook
        
        logger.info("\nRegistering hooks:")
        for i in range(4):
            try:
                layer = self.model.backbone.layers[i]
                handle = layer.register_forward_hook(get_hook(f'stage{i+1}'))
                self.hooks.append(handle)
                logger.info(f"  ✓ stage{i+1}")
            except Exception as e:
                logger.error(f"  ✗ Failed stage{i+1}: {e}")
        
        if len(self.hooks) == 0:
            raise RuntimeError("No hooks registered!")
    
    def preprocess_video(self, video_path: str):
        """Load and preprocess video"""
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        target_frames = 8
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Cannot read: {video_path}")
        
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        
        current_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_idx in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            
            current_idx += 1
        
        cap.release()
        
        while len(frames) < target_frames:
            frames.append(frames[-1])
        
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)
        
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1, 1)
        frames = (frames - mean) / std
        
        return frames.unsqueeze(0)
    
    @torch.no_grad()
    def extract_video(self, video_path: str):
        """Extract features"""
        self.features.clear()
        
        try:
            video_tensor = self.preprocess_video(video_path)
            video_tensor = video_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load {video_path}: {e}")
            return {}
        
        try:
            _ = self.model.backbone(video_tensor)
        except Exception as e:
            logger.error(f"Forward failed: {e}")
            return {}
        
        return {name: feat.numpy() for name, feat in self.features.items()}
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

# ============================================================
# CSV-based Data Loading
# ============================================================
def load_split_from_csv(csv_path: Path, video_root: Path):
    """
    Load video list and labels from CSV annotation file
    
    Args:
        csv_path: Path to train.csv / val.csv / test.csv
        video_root: Root directory containing all videos
        
    Returns:
        List of dicts with video_path and labels
    """
    logger.info(f"\nLoading annotations from {csv_path.name}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"  ✓ Found {len(df)} samples in CSV")
    
    # Expected columns (adjust based on your CSV structure)
    required_cols = ['filename']  # At minimum, need filename
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    
    # Print available columns
    logger.info(f"  CSV columns: {list(df.columns)}")
    
    samples = []
    missing_videos = []
    
    for idx, row in df.iterrows():
        # Get video path
        video_filename = row['filename']
        video_path = video_root / video_filename
        
        # Check if video exists
        if not video_path.exists():
            missing_videos.append(video_filename)
            continue
        
        # Extract labels from CSV (with defaults if missing)
        sample = {
            'video_path': video_path,
            'filename': video_filename,
            'mos': float(row.get('mos', 0.0)),
            'quality_class': str(row.get('quality_class', 'unknown')),
            'hallucination_flag': int(row.get('hallucination_flag', 0)),
            'lighting_flag': int(row.get('lighting_flag', 0)),
            'spatial_flag': int(row.get('spatial_flag', 0)),
            'rendering_flag': int(row.get('rendering_flag', 0))
        }
        
        samples.append(sample)
    
    logger.info(f"  ✓ Loaded {len(samples)} valid samples")
    
    if missing_videos:
        logger.warning(f"  ⚠️  {len(missing_videos)} videos not found (first 5):")
        for v in missing_videos[:5]:
            logger.warning(f"     - {v}")
    
    # Print label statistics
    if samples:
        logger.info(f"\n  📊 Label statistics:")
        logger.info(f"     MOS range: [{min(s['mos'] for s in samples):.1f}, {max(s['mos'] for s in samples):.1f}]")
        logger.info(f"     Quality classes: {set(s['quality_class'] for s in samples)}")
        logger.info(f"     Hallucination: {sum(s['hallucination_flag'] for s in samples)} positive")
        logger.info(f"     Lighting: {sum(s['lighting_flag'] for s in samples)} positive")
        logger.info(f"     Spatial: {sum(s['spatial_flag'] for s in samples)} positive")
        logger.info(f"     Rendering: {sum(s['rendering_flag'] for s in samples)} positive")
    
    return samples

# ============================================================
# Main Extraction
# ============================================================
def extract_all_splits(checkpoint_path: str, config_path: str, video_root: str, 
                       csv_dir: str, splits: list = None):
    """
    Extract features using CSV annotations
    
    Args:
        checkpoint_path: Model checkpoint
        config_path: Model config
        video_root: Directory containing ALL videos
        csv_dir: Directory containing train.csv, val.csv, test.csv
        splits: Splits to process
    """
    video_root = Path(video_root)
    csv_dir = Path(csv_dir)
    splits = splits or SPLITS
    
    print_section_header("Swin-T Feature Extraction (CSV-based)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Video root: {video_root}")
    logger.info(f"CSV directory: {csv_dir}")
    
    extractor = SwinFeatureExtractor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device
    )
    
    for split in splits:
        print_section_header(f"Processing {split.upper()}")
        
        # Load samples from CSV
        csv_path = csv_dir / f"{split}.csv"
        if not csv_path.exists():
            logger.error(f"✗ CSV not found: {csv_path}")
            continue
        
        samples = load_split_from_csv(csv_path, video_root)
        
        if not samples:
            logger.warning(f"⚠ No valid samples for {split}")
            continue
        
        # Storage
        layer_features = {layer: [] for layer in LAYERS}
        all_labels = {
            'mos': [],
            'quality_class': [],
            'hallucination_flag': [],
            'lighting_flag': [],
            'spatial_flag': [],
            'rendering_flag': []
        }
        
        # Extract features
        success_count = 0
        for sample in tqdm(samples, desc=f"Extracting {split}"):
            try:
                features = extractor.extract_video(str(sample['video_path']))
                
                if not features:
                    continue
                
                # Store features
                for layer in LAYERS:
                    if layer in features:
                        layer_features[layer].append(features[layer])
                
                # Store ACTUAL labels from CSV
                for key in all_labels.keys():
                    all_labels[key].append(sample[key])
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed {sample['filename']}: {e}")
                continue
        
        logger.info(f"\n✓ Processed {success_count}/{len(samples)} videos")
        
        # Concatenate
        logger.info("\n📊 Feature shapes:")
        for layer in LAYERS:
            if layer_features[layer]:
                layer_features[layer] = np.concatenate(layer_features[layer], axis=0)
                logger.info(f"  {layer}: {layer_features[layer].shape}")
        
        # Convert labels to arrays
        for key in all_labels.keys():
            all_labels[key] = np.array(all_labels[key])
        
        # Print final label stats
        logger.info(f"\n📊 Final {split} label statistics:")
        logger.info(f"  Total samples: {len(all_labels['mos'])}")
        logger.info(f"  MOS range: [{all_labels['mos'].min():.1f}, {all_labels['mos'].max():.1f}]")
        logger.info(f"  Quality classes: {np.unique(all_labels['quality_class'])}")
        logger.info(f"  Hallucination positives: {all_labels['hallucination_flag'].sum()}")
        logger.info(f"  Lighting positives: {all_labels['lighting_flag'].sum()}")
        logger.info(f"  Spatial positives: {all_labels['spatial_flag'].sum()}")
        logger.info(f"  Rendering positives: {all_labels['rendering_flag'].sum()}")
        
        # Save
        for layer in LAYERS:
            if layer not in layer_features or len(layer_features[layer]) == 0:
                continue
            
            logger.info(f"\n💾 Saving {layer}...")
            
            embeddings = layer_features[layer]
            
            # Normalize
            if COMPUTE_CONFIG.get('normalize_features', True):
                if split == 'train':
                    embeddings, scaler = normalize_features(embeddings, scaler=None)
                    scaler_path = get_feature_path(split, layer, 'scaler').with_suffix('.pkl')
                    scaler_path.parent.mkdir(parents=True, exist_ok=True)
                    save_scaler(scaler, scaler_path)
                else:
                    scaler_path = get_feature_path('train', layer, 'scaler').with_suffix('.pkl')
                    if scaler_path.exists():
                        from utils import load_scaler
                        scaler = load_scaler(scaler_path)
                        embeddings, _ = normalize_features(embeddings, scaler=scaler)
            
            # Save embeddings
            emb_path = get_feature_path(split, layer, 'embeddings')
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(emb_path, embeddings)
            logger.info(f"  ✓ {emb_path}")
            
            # Save labels
            for label_type, label_array in all_labels.items():
                label_path = get_feature_path(split, layer, label_type)
                np.save(label_path, label_array)
            
            logger.info(f"  ✓ Labels saved")
    
    extractor.cleanup()
    print_section_header("DONE!")
    logger.info("\n🎉 Feature extraction complete with REAL labels!")
    logger.info("\nNext: python main_investigation.py")

# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Extract features with CSV labels")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config')
    parser.add_argument('--video_root', type=str, required=True,
                        help='Root directory containing ALL videos')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=None,
                        help='Splits to process')
    
    args = parser.parse_args()
    
    # Validate paths
    for name, path in [('model', args.model_path), ('config', args.config), 
                        ('video_root', args.video_root), ('csv_dir', args.csv_dir)]:
        if not Path(path).exists():
            logger.error(f"✗ {name} not found: {path}")
            sys.exit(1)
    
    extract_all_splits(
        checkpoint_path=args.model_path,
        config_path=args.config,
        video_root=args.video_root,
        csv_dir=args.csv_dir,
        splits=args.splits
    )

if __name__ == "__main__":
    main()