import os, sys, numpy as np, torch

repo_root = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1'
sys.path.insert(0, repo_root)

from mmengine import Config
from mmaction.registry import MODELS, DATASETS
from torch.utils.data import DataLoader

# Import custom modules BEFORE loading config
import mmaction.models.data_preprocessors
import mmaction.models.recognizers.video_quality_recognizer
import mmaction.models.heads.VQA_multihead
import mmaction.datasets.VQA_dataset
import mmaction.datasets.transforms.video_quality_pack
import mmaction.evaluation.metrics.VQA_customMetric


def collate_fn(batch):
    """Custom collate function that returns batch as-is."""
    return batch


def spatial_temporal_pool(x, method='adaptive'):
    """
    Pool spatial dimensions but keep temporal structure for token sequence.
    
    Args:
        x: [B, C, T, H, W] - backbone features
        method: 'adaptive' or 'mean'
    
    Returns:
        [B, T, C] - sequence of video tokens (one per time step)
    """
    if method == 'adaptive':
        # Adaptive pool spatial dims to 1x1, keep temporal
        x = torch.nn.functional.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # [B,C,T,1,1]
    else:
        # Mean pool over spatial dimensions
        x = x.mean(dim=[-2, -1])  # [B,C,T]
    
    x = x.squeeze(-1).squeeze(-1)  # [B,C,T]
    x = x.permute(0, 2, 1)  # [B,T,C] - standard sequence format
    return x


def main():
    # Load config
    cfg = Config.fromfile(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\mmaction\configs\recognition\swin\VQA_swinConfigPipeline.py')
    
    ckpt = r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\VQA_dryrun\epoch_3.pth'

    # Build model
    model = MODELS.build(cfg.model)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval().cuda()

    # Build dataloader
    ds_cfg = cfg.val_dataloader['dataset']
    dataset = DATASETS.build(ds_cfg)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Extract video tokens
    all_tokens, rows = [], []
    with torch.no_grad():
        for batch_list in loader:
            # Convert to MMAction format
            batch = {
                'inputs': [sample['inputs'] for sample in batch_list],
                'data_samples': [sample['data_samples'] for sample in batch_list]
            }
            
            batch = model.data_preprocessor(batch, training=False)
            inputs = batch['inputs'].cuda()
            data_samples = batch['data_samples']

            # Forward through backbone
            feats = model.backbone(inputs)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]  # Get final stage features
            
            # Extract video tokens (keep temporal dimension)
            if feats.dim() == 5:
                # [B,C,T,H,W] -> [B,T,C] video token sequence
                video_tokens = spatial_temporal_pool(feats, method='adaptive')
            else:
                # If already flattened, reshape appropriately
                raise ValueError(f"Unexpected feature shape: {feats.shape}")
            
            # Store tokens
            all_tokens.append(video_tokens.cpu().numpy())
            
            # Store metadata
            ds = data_samples[0]
            rows.append(dict(
                filename=os.path.basename(ds.metainfo.get('filename', 'unk')),
                mos=ds.metainfo.get('mos', None),
                quality_class=ds.metainfo.get('quality_class', None),
                distortion_type=ds.metainfo.get('distortion_type', None),
                num_tokens=video_tokens.shape[1],  # Temporal length
                token_dim=video_tokens.shape[2]     # Feature dimension
            ))

    # Save results
    tokens = np.concatenate(all_tokens, axis=0)  # [N, T, C]
    os.makedirs(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\vqa_dryrun', exist_ok=True)
    np.save(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\output\vqa_dryrun\video_tokens.npy', tokens)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\work_dirs\vqa_dryrun\video_tokens_fusion.csv', index=False)
    
    print(f'✓ Saved video tokens with shape {tokens.shape}')
    print(f'  - {tokens.shape[0]} videos')
    print(f'  - {tokens.shape[1]} temporal tokens per video')
    print(f'  - {tokens.shape[2]} feature dimensions per token')
    print(f'  Format: [num_videos, seq_len, feature_dim]')
    print(f'\nReady for fusion with T5 text tokens!')

if __name__ == '__main__':
    main()