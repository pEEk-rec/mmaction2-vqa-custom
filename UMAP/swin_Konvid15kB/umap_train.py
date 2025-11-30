"""
UMAP + Feature-MOS correlation for Swin-T Stage 3 vs Stage 4 (KoNViD Training Data)

Assumptions:
- features_stage3.npy  shape: (N, 384)
- features_stage4.npy  shape: (N, 768)
- extracted_video_list.csv with column 'stem' listing stems in the same order as features saved
- train CSV contains 'filename' and 'mos' columns (or 'video_name' and 'video_score')

Saves:
- per-stage CSVs with per-dim Pearson/Spearman
- PCA cumulative variance
- UMAP scatter plots (by MOS and by MOS-quantile bins)
- stages_summary.csv

Usage:
python umap_stage_compare_train.py --features_dir "D:\path\to\features" --extracted_list "D:\path\to\extracted_video_list.csv" --train_csv "D:\path\to\konvid150k_train.csv" --out_dir "D:\path\to\output"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
import umap

sns.set(style="whitegrid")

def load_align(features_dir: Path, extracted_list: Path, train_csv: Path):
    # load extracted list
    df_list = pd.read_csv(extracted_list)
    # expected column 'stem' (as saved by extraction). If not, try first column.
    if 'stem' not in df_list.columns:
        df_list.columns = ['stem']
    stems = df_list['stem'].astype(str).tolist()

    # load training csv and build map from stem -> score
    df_train = pd.read_csv(train_csv)
    
    # Handle different possible column names
    if 'filename' in df_train.columns:
        df_train['stem'] = df_train['filename'].apply(lambda x: Path(x).stem if isinstance(x, str) else str(x))
    elif 'video_name' in df_train.columns:
        df_train['stem'] = df_train['video_name'].apply(lambda x: Path(x).stem if isinstance(x, str) else str(x))
    else:
        raise ValueError("Train CSV must contain 'filename' or 'video_name' column.")
    
    # Handle MOS column names
    if 'mos' in df_train.columns:
        df_train['score'] = df_train['mos'].astype(float)
    elif 'video_score' in df_train.columns:
        df_train['score'] = df_train['video_score'].astype(float)
    else:
        raise ValueError("Train CSV must contain 'mos' or 'video_score' column.")

    score_map = dict(zip(df_train['stem'], df_train['score']))

    # Load global feature matrices
    f3 = np.load(features_dir / "features_stage3.npy", allow_pickle=False)
    f4 = np.load(features_dir / "features_stage4.npy", allow_pickle=False)

    # Align features using stems list. extracted list should match row order used when saving
    # But we double-check lengths
    if len(stems) != f3.shape[0] or len(stems) != f4.shape[0]:
        print(f"[WARN] stems length ({len(stems)}) does not match feature matrix rows (f3: {f3.shape[0]}, f4: {f4.shape[0]}). Attempting to align by order fallback.")
    
    # Build mos list in same order as stems
    mos_list = []
    missing = 0
    for s in stems:
        mos = score_map.get(s, None)
        if mos is None:
            missing += 1
            mos = np.nan
        mos_list.append(mos)
    
    mos_arr = np.array(mos_list, dtype=float)
    valid_mask = ~np.isnan(mos_arr)
    
    if missing > 0:
        print(f"[WARN] {missing} stems did not have matching MOS in train CSV. They will be dropped for correlation/UMAP.")
    
    print(f"[INFO] Loaded {f3.shape[0]} stage3 features, {f4.shape[0]} stage4 features")
    print(f"[INFO] Valid MOS entries: {valid_mask.sum()} / {len(mos_arr)}")

    return f3, f4, np.array(stems), mos_arr, valid_mask

def per_feature_correlation(X, mos, out_csv):
    # X: (N, D)
    pear = []
    spear = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        # drop nan pairs
        mask = ~np.isnan(xi) & ~np.isnan(mos)
        if mask.sum() < 3:
            pear.append(np.nan)
            spear.append(np.nan)
            continue
        p = pearsonr(xi[mask], mos[mask])[0]
        s = spearmanr(xi[mask], mos[mask])[0]
        pear.append(p)
        spear.append(s)
    df = pd.DataFrame({'pearson': pear, 'spearman': spear})
    df.to_csv(out_csv, index_label='feature_idx')
    
    # Print summary stats
    print(f"  Pearson - mean: {np.nanmean(pear):.4f}, max: {np.nanmax(np.abs(pear)):.4f}")
    print(f"  Spearman - mean: {np.nanmean(spear):.4f}, max: {np.nanmax(np.abs(spear)):.4f}")
    
    return df

def plot_hist_corr(df_corr, title, out_png):
    plt.figure(figsize=(6,4))
    sns.histplot(df_corr['spearman'].dropna(), bins=40, kde=True)
    plt.title(title)
    plt.xlabel('Spearman correlation (feature vs MOS)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def pca_then_umap(X, n_pca=50, n_neighbors=15, min_dist=0.1, random_state=42):
    # Standardize
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_pca, Xz.shape[1]), random_state=random_state)
    Xp = pca.fit_transform(Xz)
    
    print(f"  PCA: {Xp.shape[1]} components explain {pca.explained_variance_ratio_.sum():.2%} variance")
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
    emb2 = reducer.fit_transform(Xp)
    return emb2, pca, Xp

def plot_umap(emb2, mos, stems, out_png_cont, out_png_bins, stage_name):
    # continuous MOS
    plt.figure(figsize=(7,5))
    sc = plt.scatter(emb2[:,0], emb2[:,1], c=mos, cmap='viridis', s=14, alpha=0.7, edgecolors='none')
    plt.colorbar(sc, label='MOS Score')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'{stage_name} - UMAP colored by MOS (Training Data)')
    plt.tight_layout()
    plt.savefig(out_png_cont, dpi=200)
    plt.close()

    # bins (quantiles)
    # create 3 bins (low/mid/high) using qcut (quantiles)
    try:
        bins = pd.qcut(mos, q=3, labels=['Low Quality','Medium Quality','High Quality'], duplicates='drop')
    except Exception:
        # fallback to simple cut
        bins = pd.cut(mos, bins=3, labels=['Low Quality','Medium Quality','High Quality'])
    
    plt.figure(figsize=(7,5))
    palette = {'Low Quality':'#d73027','Medium Quality':'#fee08b','High Quality':'#1a9850'}
    for b in bins.unique().categories if hasattr(bins.unique(), "categories") else np.unique(bins):
        mask = (bins == b)
        plt.scatter(emb2[mask,0], emb2[mask,1], s=20, alpha=0.7, label=str(b), 
                   color=palette.get(str(b), None), edgecolors='none')
    plt.legend(title='Quality Bins', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'{stage_name} - UMAP by Quality Bins (Training Data)')
    plt.tight_layout()
    plt.savefig(out_png_bins, dpi=200, bbox_inches='tight')
    plt.close()

def main(args):
    features_dir = Path(args.features_dir)
    extracted_list = Path(args.extracted_list)
    train_csv = Path(args.train_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading features from: {features_dir}")
    print(f"[INFO] Using extracted list: {extracted_list}")
    print(f"[INFO] Using training CSV: {train_csv}")
    
    f3, f4, stems, mos_arr, valid_mask = load_align(features_dir, extracted_list, train_csv)

    # for each stage do full pipeline
    summary_rows = []
    for stage_idx, (feat_mat, feat_name) in enumerate(zip([f3, f4], ['Stage 3','Stage 4']), start=3):
        print(f"\n{'='*60}")
        print(f"[INFO] Processing {feat_name} with shape {feat_mat.shape}")
        print(f"{'='*60}")
        
        # Filter to valid MOS entries
        mask = valid_mask & (~np.isnan(mos_arr))
        X = feat_mat[mask, :]
        mos = mos_arr[mask]
        stems_filt = stems[mask]

        print(f"  Samples after filtering: {X.shape[0]}")
        print(f"  MOS range: [{mos.min():.2f}, {mos.max():.2f}]")
        print(f"  MOS mean: {mos.mean():.2f}, std: {mos.std():.2f}")

        # Per-feature correlations
        print(f"\n  Computing per-feature correlations...")
        corr_csv = out_dir / f"{feat_name.lower().replace(' ','')}_feature_mos_corr.csv"
        df_corr = per_feature_correlation(X, mos, corr_csv)
        plot_hist_corr(df_corr, f"{feat_name} Spearman Feature-MOS", 
                      out_dir / f"{feat_name.lower().replace(' ','')}_spearman_hist.png")

        # PCA then UMAP
        print(f"\n  Running PCA + UMAP...")
        emb2, pca, Xp = pca_then_umap(X, n_pca=args.pca_dims, n_neighbors=args.umap_neighbors, 
                                      min_dist=args.umap_dist, random_state=args.random_state)

        # Save PCA cumulative var
        ev = pca.explained_variance_ratio_.cumsum()
        pd.DataFrame({'pc_index': np.arange(1, len(ev)+1), 'cum_var': ev}).to_csv(
            out_dir / f"{feat_name.lower().replace(' ','')}_pca_cumvar.csv", index=False)

        # Correlate top PCs with MOS (top 20)
        n_pc_corr = min(20, Xp.shape[1])
        pear = []; spear = []
        for i in range(n_pc_corr):
            pear.append(pearsonr(Xp[:,i], mos)[0])
            spear.append(spearmanr(Xp[:,i], mos)[0])
        pd.DataFrame({'pearson_pc': pear, 'spearman_pc': spear}).to_csv(
            out_dir / f"{feat_name.lower().replace(' ','')}_pc_mos_corr.csv", index_label='pc')

        # UMAP plots
        print(f"  Creating UMAP plots...")
        out_png_cont = out_dir / f"{feat_name.lower().replace(' ','')}_umap_mos_cont.png"
        out_png_bins = out_dir / f"{feat_name.lower().replace(' ','')}_umap_mos_bins.png"
        plot_umap(emb2, mos, stems_filt, out_png_cont, out_png_bins, feat_name)

        # Correlate UMAP dims with MOS
        p1 = pearsonr(emb2[:,0], mos)[0]; p2 = pearsonr(emb2[:,1], mos)[0]
        s1 = spearmanr(emb2[:,0], mos)[0]; s2 = spearmanr(emb2[:,1], mos)[0]
        
        print(f"  UMAP-MOS correlations:")
        print(f"    Dim 1 - Pearson: {p1:.4f}, Spearman: {s1:.4f}")
        print(f"    Dim 2 - Pearson: {p2:.4f}, Spearman: {s2:.4f}")

        # Silhouette on MOS bins (quantile bins)
        try:
            bins = pd.qcut(mos, q=3, labels=False, duplicates='drop')
            sil = silhouette_score(Xp, bins)
            print(f"  Silhouette score (quality bins): {sil:.4f}")
        except Exception as e:
            print(f"  Could not compute silhouette score: {e}")
            sil = np.nan

        summary_rows.append({
            'stage': feat_name,
            'n_samples': X.shape[0],
            'feat_dim': X.shape[1],
            'pca_dims': Xp.shape[1],
            'mos_mean': float(mos.mean()),
            'mos_std': float(mos.std()),
            'pearson_umap_dim0': float(p1),
            'pearson_umap_dim1': float(p2),
            'spearman_umap_dim0': float(s1),
            'spearman_umap_dim1': float(s2),
            'silhouette_bins': float(sil)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "stages_summary.csv", index=False)
    
    print(f"\n{'='*60}")
    print("[DONE] Analysis complete!")
    print(f"{'='*60}")
    print(f"Output saved to: {out_dir}")
    print("\nFiles created:")
    print("  - stages_summary.csv")
    print("  - stage3/4_feature_mos_corr.csv")
    print("  - stage3/4_spearman_hist.png")
    print("  - stage3/4_pca_cumvar.csv")
    print("  - stage3/4_pc_mos_corr.csv")
    print("  - stage3/4_umap_mos_cont.png")
    print("  - stage3/4_umap_mos_bins.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True, 
                       help="Directory containing features_stage3.npy and features_stage4.npy")
    parser.add_argument("--extracted_list", type=str, required=True, 
                       help="CSV with column 'stem' listing extracted stems in same order")
    parser.add_argument("--train_csv", type=str, required=True, 
                       help="Training CSV with filename/video_name and mos/video_score columns")
    parser.add_argument("--out_dir", type=str, required=True,
                       help="Output directory for plots and results")
    parser.add_argument("--pca_dims", type=int, default=50,
                       help="Number of PCA dimensions before UMAP (default: 50)")
    parser.add_argument("--umap_neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter (default: 15)")
    parser.add_argument("--umap_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter (default: 0.1)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    main(args)