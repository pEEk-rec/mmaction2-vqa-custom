

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

def load_data(features_dir: Path, extracted_list: Path, aivqa_csv: Path):
    # features
    f3 = np.load(features_dir / "features_stage3.npy", allow_pickle=False)
    f4 = np.load(features_dir / "features_stage4.npy", allow_pickle=False)

    # stems list
    df_list = pd.read_csv(extracted_list)
    if 'stem' not in df_list.columns:
        df_list.columns = ['stem']
    stems = df_list['stem'].astype(str).tolist()

    # labels
    df = pd.read_csv(aivqa_csv)
    # detect filename column
    filename_col = None
    for c in ['filename', 'video_name', 'name', 'video_filename', 'file']:
        if c in df.columns:
            filename_col = c
            break
    if filename_col is None:
        filename_col = df.columns[0]
    df['stem'] = df[filename_col].apply(lambda x: Path(str(x)).stem if pd.notna(x) else '')
    # detect MOS column
    if 'mos' in df.columns:
        df['mos_val'] = pd.to_numeric(df['mos'], errors='coerce')
    elif 'video_score' in df.columns:
        df['mos_val'] = pd.to_numeric(df['video_score'], errors='coerce')
    else:
        df['mos_val'] = np.nan

    # detect flags
    flag_names = []
    for name in ['hallucination_flag', 'lighting_flag']:
        if name in df.columns:
            flag_names.append(name)
        else:
            # try shorter names if present
            for col in df.columns:
                if name.split('_')[0] in col and 'flag' in col:
                    flag_names.append(col)
                    break
    # ensure flags exist; if not, fill NaN (will be dropped)
    for n in ['hallucination_flag', 'lighting_flag']:
        if n not in df.columns:
            df[n] = np.nan

    # build maps
    mos_map = dict(zip(df['stem'], df['mos_val']))
    hall_map = dict(zip(df['stem'], df.get('hallucination_flag', pd.Series(np.nan, index=df.index))))
    light_map = dict(zip(df['stem'], df.get('lighting_flag', pd.Series(np.nan, index=df.index))))

    # align to stems list
    mos_list = []
    hall_list = []
    light_list = []
    missing_count = 0
    for s in stems:
        mos = mos_map.get(s, np.nan)
        hall = hall_map.get(s, np.nan)
        light = light_map.get(s, np.nan)
        if (s not in mos_map) and (s not in hall_map) and (s not in light_map):
            missing_count += 1
        mos_list.append(mos)
        hall_list.append(hall)
        light_list.append(light)

    mos_arr = np.array(mos_list, dtype=float)
    hall_arr = np.array(hall_list, dtype=float)
    light_arr = np.array(light_list, dtype=float)

    valid_mask = (~np.isnan(mos_arr)) | (~np.isnan(hall_arr)) | (~np.isnan(light_arr))
    print(f"[INFO] features shapes: stage3={f3.shape}, stage4={f4.shape}")
    print(f"[INFO] stems: {len(stems)}, missing in labels: {missing_count}")

    return f3, f4, np.array(stems), mos_arr, hall_arr, light_arr, valid_mask

def per_feature_correlations_multi(X, mos, hall, light, out_csv):
    # X: (N, D)
    cols = []
    pear_mos = []; spear_mos = []
    spear_hall = []; spear_light = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        # mask for mos
        m_mos = ~np.isnan(xi) & ~np.isnan(mos)
        if m_mos.sum() >= 3:
            p = pearsonr(xi[m_mos], mos[m_mos])[0]
            s = spearmanr(xi[m_mos], mos[m_mos])[0]
        else:
            p = np.nan; s = np.nan
        pear_mos.append(p); spear_mos.append(s)

        # flags correlation: Spearman is OK for binary (point-biserial alternative)
        m_h = ~np.isnan(xi) & ~np.isnan(hall)
        if m_h.sum() >= 3:
            sh = spearmanr(xi[m_h], hall[m_h])[0]
        else:
            sh = np.nan
        spear_hall.append(sh)

        m_l = ~np.isnan(xi) & ~np.isnan(light)
        if m_l.sum() >= 3:
            sl = spearmanr(xi[m_l], light[m_l])[0]
        else:
            sl = np.nan
        spear_light.append(sl)

    df = pd.DataFrame({
        'pearson_mos': pear_mos,
        'spearman_mos': spear_mos,
        'spearman_hallucination': spear_hall,
        'spearman_lighting': spear_light
    })
    df.to_csv(out_csv, index_label='feature_idx')
    # print summary:
    def stats(arr):
        arr = np.array(arr, dtype=float)
        return np.nanmean(arr), np.nanmax(np.abs(arr))
    print("  MOS pearson mean/|max|: ", stats(pear_mos))
    print("  MOS spearman mean/|max|:", stats(spear_mos))
    print("  Halluc spearman mean/|max|:", stats(spear_hall))
    print("  Light spearman mean/|max|:", stats(spear_light))
    return df

def pca_then_umap(X, n_pca=50, n_neighbors=15, min_dist=0.1, random_state=42):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    n_pca = min(n_pca, Xz.shape[1])
    pca = PCA(n_components=n_pca, random_state=random_state)
    Xp = pca.fit_transform(Xz)
    print(f"  PCA -> {Xp.shape[1]} comps, cumvar {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
    emb2 = reducer.fit_transform(Xp)
    return emb2, pca, Xp

def plot_umap_variants(emb2, mos, hall, light, stems, out_prefix, stage_name):
    out_prefix = Path(out_prefix)
    # MOS continuous
    plt.figure(figsize=(7,5))
    sc = plt.scatter(emb2[:,0], emb2[:,1], c=mos, cmap='viridis', s=18, alpha=0.8)
    plt.colorbar(sc, label='MOS')
    plt.title(f"{stage_name} - UMAP colored by MOS")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix('.mos.png'), dpi=200)
    plt.close()

    # Hallucination - binary
    # Map values to 0/1 if possible
    hall_cat = pd.Series(hall).copy()
    unique_h = np.unique(hall[~np.isnan(hall)]) if np.any(~np.isnan(hall)) else []
    if len(unique_h) > 0:
        plt.figure(figsize=(7,5))
        # mask for valid hall labels
        valid_h = ~np.isnan(hall)
        plt.scatter(emb2[~valid_h,0], emb2[~valid_h,1], s=8, alpha=0.2, color='lightgray', label='no label')
        c0 = emb2[(valid_h) & (hall==0),:]
        c1 = emb2[(valid_h) & (hall!=0),:]
        if c0.shape[0] > 0:
            plt.scatter(c0[:,0], c0[:,1], s=20, alpha=0.8, label='non-hallucinated', color='#2b83ba')
        if c1.shape[0] > 0:
            plt.scatter(c1[:,0], c1[:,1], s=20, alpha=0.8, label='hallucinated', color='#d7191c')
        plt.legend(loc='best')
        plt.title(f"{stage_name} - UMAP by Hallucination Flag")
        plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix('.halluc.png'), dpi=200)
        plt.close()

    # Lighting flag
    light_cat = pd.Series(light).copy()
    unique_l = np.unique(light[~np.isnan(light)]) if np.any(~np.isnan(light)) else []
    if len(unique_l) > 0:
        plt.figure(figsize=(7,5))
        valid_l = ~np.isnan(light)
        plt.scatter(emb2[~valid_l,0], emb2[~valid_l,1], s=8, alpha=0.2, color='lightgray', label='no label')
        c0 = emb2[(valid_l) & (light==0),:]
        c1 = emb2[(valid_l) & (light!=0),:]
        if c0.shape[0] > 0:
            plt.scatter(c0[:,0], c0[:,1], s=20, alpha=0.8, label='normal lighting', color='#4daf4a')
        if c1.shape[0] > 0:
            plt.scatter(c1[:,0], c1[:,1], s=20, alpha=0.8, label='lighting issue', color='#984ea3')
        plt.legend(loc='best')
        plt.title(f"{stage_name} - UMAP by Lighting Flag")
        plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(out_prefix.with_suffix('.lighting.png'), dpi=200)
        plt.close()

def analyze_stage(feat_mat, mos_arr, hall_arr, light_arr, stems, out_dir: Path, stage_name, args):
    # Filter rows where at least one label exists (we did outer-valid earlier)
    mask = (~np.isnan(mos_arr)) | (~np.isnan(hall_arr)) | (~np.isnan(light_arr))
    X = feat_mat[mask, :]
    mos = mos_arr[mask]
    hall = hall_arr[mask]
    light = light_arr[mask]
    stems_filt = stems[mask]
    print(f"[INFO] {stage_name}: using {X.shape[0]} samples (features dim {X.shape[1]})")

    # Per-feature correlations (MOS + flags)
    corr_csv = out_dir / f"{stage_name.replace(' ','').lower()}_feature_label_corr.csv"
    df_corr = per_feature_correlations_multi(X, mos, hall, light, corr_csv)

    # PCA + UMAP
    emb2, pca, Xp = pca_then_umap(X, n_pca=args.pca_dims, n_neighbors=args.umap_neighbors, min_dist=args.umap_dist, random_state=args.random_state)
    # Save PCA cumulative var
    ev = pca.explained_variance_ratio_.cumsum()
    pd.DataFrame({'pc_index': np.arange(1, len(ev)+1), 'cum_var': ev}).to_csv(out_dir / f"{stage_name.replace(' ','').lower()}_pca_cumvar.csv", index=False)

    # UMAP plots
    plot_prefix = out_dir / f"{stage_name.replace(' ','').lower()}_umap"
    plot_umap_variants(emb2, mos, hall, light, stems_filt, plot_prefix, stage_name)

    # UMAP ↔ MOS correlations
    p1 = pearsonr(emb2[:,0], mos)[0] if emb2.shape[0] >= 3 else np.nan
    p2 = pearsonr(emb2[:,1], mos)[0] if emb2.shape[0] >= 3 else np.nan
    s1 = spearmanr(emb2[:,0], mos)[0] if emb2.shape[0] >= 3 else np.nan
    s2 = spearmanr(emb2[:,1], mos)[0] if emb2.shape[0] >= 3 else np.nan
    # silhouette on PCA space using halluc flag bins if available
    sil = np.nan
    try:
        if np.any(~np.isnan(hall)):
            bins = pd.Series(hall).dropna().values
            # align bins to Xp rows (drop nan rows)
            # but here hall already filtered; compute silhouette on Xp with integer bins
            bins_int = pd.Series(hall).astype(float)
            bins_int = bins_int.reset_index(drop=True)
            sil = silhouette_score(Xp, bins_int)
    except Exception:
        sil = np.nan

    return {
        'stage': stage_name,
        'n_samples': int(X.shape[0]),
        'feat_dim': int(X.shape[1]),
        'pca_dims': int(Xp.shape[1]),
        'pearson_umap_dim0': float(p1) if not np.isnan(p1) else None,
        'pearson_umap_dim1': float(p2) if not np.isnan(p2) else None,
        'spearman_umap_dim0': float(s1) if not np.isnan(s1) else None,
        'spearman_umap_dim1': float(s2) if not np.isnan(s2) else None,
        'silhouette_halluc_bins': float(sil) if not np.isnan(sil) else None
    }

def main(args):
    features_dir = Path(args.features_dir)
    extracted_list = Path(args.extracted_list)
    aivqa_csv = Path(args.aivqa_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f3, f4, stems, mos_arr, hall_arr, light_arr, valid_mask = load_data(features_dir, extracted_list, aivqa_csv)

    summary = []
    # Stage 3
    s3 = analyze_stage(f3, mos_arr, hall_arr, light_arr, stems, out_dir, "Stage 3", args)
    summary.append(s3)
    # Stage 4
    s4 = analyze_stage(f4, mos_arr, hall_arr, light_arr, stems, out_dir, "Stage 4", args)
    summary.append(s4)

    pd.DataFrame(summary).to_csv(out_dir / "stages_summary.csv", index=False)
    print("[DONE] Saved analysis to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, required=True, help="dir with features_stage3.npy and features_stage4.npy")
    parser.add_argument("--extracted_list", type=str, required=True, help="csv with column 'stem' listing extracted stems in same order")
    parser.add_argument("--aivqa_csv", type=str, required=True, help="AI-VQA csv with filename and labels")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pca_dims", type=int, default=50)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_dist", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)
