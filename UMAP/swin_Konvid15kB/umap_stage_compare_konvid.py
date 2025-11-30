"""
UMAP + Feature-MOS correlation for Swin-T Stage 3 vs Stage 4 (KoNViD)

Assumptions:
- features_stage3.npy  shape: (N, 384)
- features_stage4.npy  shape: (N, 768)
- extracted_video_list.csv with column 'stem' listing stems in the same order as features saved
- konvid CSV contains 'video_name' and 'video_score' columns

Saves:
- per-stage CSVs with per-dim Pearson/Spearman
- PCA cumulative variance
- UMAP scatter plots (by MOS and by MOS-quantile bins)
- stages_summary.csv

Usage:
python umap_stage_compare_konvid.py --features_dir "<path>" --extracted_list "<path>" --konvid_csv "<path>" --out_dir "<path>"
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

def load_align(features_dir: Path, extracted_list: Path, konvid_csv: Path):
    # load extracted list
    df_list = pd.read_csv(extracted_list)
    # expected column 'stem' (as saved by extraction). If not, try first column.
    if 'stem' not in df_list.columns:
        df_list.columns = ['stem']
    stems = df_list['stem'].astype(str).tolist()

    # load konvid csv and build map from stem -> score
    df_k = pd.read_csv(konvid_csv)
    # Normalize names: konvid video_name contains file names like 'orig_xxx.mp4'
    df_k['stem'] = df_k['video_name'].apply(lambda x: Path(x).stem if isinstance(x, str) else str(x))
    if 'video_score' in df_k.columns:
        df_k['mos'] = df_k['video_score'].astype(float)
    elif 'video_score' not in df_k.columns and 'mos' in df_k.columns:
        df_k['mos'] = df_k['mos'].astype(float)
    else:
        raise ValueError("KonViD CSV must contain 'video_score' or 'mos' column.")

    score_map = dict(zip(df_k['stem'], df_k['mos']))

    # Load global feature matrices
    f3 = np.load(features_dir / "features_stage3.npy", allow_pickle=False)
    f4 = np.load(features_dir / "features_stage4.npy", allow_pickle=False)

    # Align features using stems list. extracted list should match row order used when saving
    # But we double-check lengths
    if len(stems) != f3.shape[0] or len(stems) != f4.shape[0]:
        print("[WARN] stems length does not match feature matrix rows. Attempting to align by order fallback.")
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
        print(f"[WARN] {missing} stems did not have matching MOS in KonViD CSV. They will be dropped for correlation/UMAP.")

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
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
    emb2 = reducer.fit_transform(Xp)
    return emb2, pca, Xp

def plot_umap(emb2, mos, stems, out_png_cont, out_png_bins):
    # continuous MOS
    plt.figure(figsize=(6,5))
    sc = plt.scatter(emb2[:,0], emb2[:,1], c=mos, cmap='viridis', s=14, alpha=0.9)
    plt.colorbar(sc, label='MOS (video_score)')
    plt.title(Path(out_png_cont).stem)
    plt.tight_layout()
    plt.savefig(out_png_cont, dpi=200)
    plt.close()

    # bins (quantiles)
    # create 3 bins (low/mid/high) using qcut (quantiles)
    # handle duplicates if not enough unique values
    try:
        bins = pd.qcut(mos, q=3, labels=['low','mid','high'], duplicates='drop')
    except Exception:
        # fallback to simple cut
        bins = pd.cut(mos, bins=3, labels=['low','mid','high'])
    plt.figure(figsize=(6,5))
    palette = {'low':'#d73027','mid':'#fee08b','high':'#1a9850'}
    for b in bins.unique().categories if hasattr(bins.unique(), "categories") else np.unique(bins):
        mask = (bins == b)
        plt.scatter(emb2[mask,0], emb2[mask,1], s=16, alpha=0.9, label=str(b))
    plt.legend(title='MOS bins', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title(Path(out_png_bins).stem)
    plt.tight_layout()
    plt.savefig(out_png_bins, dpi=200)
    plt.close()

def main(args):
    features_dir = Path(args.features_dir)
    extracted_list = Path(args.extracted_list)
    konvid_csv = Path(args.konvid_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f3, f4, stems, mos_arr, valid_mask = load_align(features_dir, extracted_list, konvid_csv)

    # for each stage do full pipeline
    summary_rows = []
    for stage_idx, (feat_mat, feat_name) in enumerate(zip([f3, f4], ['stage3','stage4']), start=3):
        print(f"[INFO] Processing {feat_name} with shape {feat_mat.shape}")
        # Filter to valid MOS entries
        mask = valid_mask & (~np.isnan(mos_arr))
        X = feat_mat[mask, :]
        mos = mos_arr[mask]
        stems_filt = stems[mask]

        # Per-feature correlations
        corr_csv = out_dir / f"{feat_name}_feature_mos_corr.csv"
        df_corr = per_feature_correlation(X, mos, corr_csv)
        plot_hist_corr(df_corr, f"{feat_name} Spearman feature-MOS", out_dir / f"{feat_name}_spearman_hist.png")

        # PCA then UMAP
        emb2, pca, Xp = pca_then_umap(X, n_pca=args.pca_dims, n_neighbors=args.umap_neighbors, min_dist=args.umap_dist, random_state=args.random_state)

        # Save PCA cumulative var
        ev = pca.explained_variance_ratio_.cumsum()
        pd.DataFrame({'pc_index': np.arange(1, len(ev)+1), 'cum_var': ev}).to_csv(out_dir / f"{feat_name}_pca_cumvar.csv", index=False)

        # Correlate top PCs with MOS (top 20)
        n_pc_corr = min(20, Xp.shape[1])
        pear = []; spear = []
        for i in range(n_pc_corr):
            pear.append(pearsonr(Xp[:,i], mos)[0])
            spear.append(spearmanr(Xp[:,i], mos)[0])
        pd.DataFrame({'pearson_pc': pear, 'spearman_pc': spear}).to_csv(out_dir / f"{feat_name}_pc_mos_corr.csv", index_label='pc')

        # UMAP plots
        out_png_cont = out_dir / f"{feat_name}_umap_mos_cont.png"
        out_png_bins = out_dir / f"{feat_name}_umap_mos_bins.png"
        plot_umap(emb2, mos, stems_filt, out_png_cont, out_png_bins)

        # Correlate UMAP dims with MOS
        p1 = pearsonr(emb2[:,0], mos)[0]; p2 = pearsonr(emb2[:,1], mos)[0]
        s1 = spearmanr(emb2[:,0], mos)[0]; s2 = spearmanr(emb2[:,1], mos)[0]

        # Silhouette on MOS bins (quantile bins)
        try:
            bins = pd.qcut(mos, q=3, labels=False, duplicates='drop')
            sil = silhouette_score(Xp, bins)
        except Exception:
            sil = np.nan

        summary_rows.append({
            'stage': feat_name,
            'n_samples': X.shape[0],
            'feat_dim': X.shape[1],
            'pca_dims': Xp.shape[1],
            'pearson_umap_dim0': float(p1),
            'pearson_umap_dim1': float(p2),
            'spearman_umap_dim0': float(s1),
            'spearman_umap_dim1': float(s2),
            'silhouette_bins': float(sil)
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "stages_summary.csv", index=False)
    print("[DONE] Saved analysis in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--extracted_list", type=str, required=True, help="csv with column 'stem' listing extracted stems in same order")
    parser.add_argument("--konvid_csv", type=str, required=True, help="konvid test csv with video_name and video_score")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pca_dims", type=int, default=50)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_dist", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
