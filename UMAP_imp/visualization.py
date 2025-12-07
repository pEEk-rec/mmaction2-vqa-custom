"""
Visualization Module - FIXED & ROBUST VERSION
Handles:
 - Missing projections
 - Missing labels
 - Missing metrics
 - Empty clusters
 - Multi-projection dicts (MOS, quality_class, artifacts)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import seaborn as sns

from config import (
    VIS_CONFIG, COLORMAPS, LAYERS, SPLITS, ARTIFACT_TYPES,
    get_umap_path, get_comparison_path, PLOTS_DIR, COMPARISON_DIR
)
from utils import setup_logging

logger = setup_logging()

plt.style.use(VIS_CONFIG["style"])
mpl.rcParams.update({
    "font.size": VIS_CONFIG["font_size"],
    "axes.titlesize": VIS_CONFIG["title_size"],
    "axes.labelsize": VIS_CONFIG["label_size"],
    "legend.fontsize": VIS_CONFIG["legend_size"],
})


# -------------------------------------------------------------
# Safe getter (prevents KeyError crashes)
# -------------------------------------------------------------
def safe_get(d: Dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# -------------------------------------------------------------
# UMAP Scatter: MOS
# -------------------------------------------------------------
def plot_umap_by_mos(umap_xy, mos, layer, split, save_path):
    try:
        fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"], dpi=VIS_CONFIG["dpi"])
        sc = ax.scatter(
            umap_xy[:, 0], umap_xy[:, 1],
            c=mos, cmap=COLORMAPS["mos"],
            s=VIS_CONFIG["marker_size"], alpha=VIS_CONFIG["alpha"],
            edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label="MOS Score")

        ax.set_title(f"{layer.upper()} - {split.capitalize()} (MOS)")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

        plt.tight_layout()
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()
        logger.info(f"✓ Saved MOS plot: {save_path}")
    except Exception as e:
        logger.error(f"Failed MOS plot {layer}/{split}: {e}")


# -------------------------------------------------------------
# Quality Class
# -------------------------------------------------------------
def plot_umap_by_quality_class(umap_xy, qc, layer, split, save_path):
    try:
        fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"], dpi=VIS_CONFIG["dpi"])

        classes = np.unique(qc)
        cmap = plt.cm.get_cmap(COLORMAPS["quality_class"], len(classes))

        for i, cls in enumerate(classes):
            mask = qc == cls
            ax.scatter(
                umap_xy[mask, 0], umap_xy[mask, 1],
                c=[cmap(i)], label=f"{cls} (n={mask.sum()})",
                s=VIS_CONFIG["marker_size"], alpha=VIS_CONFIG["alpha"],
                edgecolors="none"
            )

        ax.legend()
        ax.set_title(f"{layer.upper()} - {split.capitalize()} (Quality Class)")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

        plt.tight_layout()
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()
        logger.info(f"✓ Saved QC plot: {save_path}")

    except Exception as e:
        logger.error(f"Failed QC plot {layer}/{split}: {e}")


# -------------------------------------------------------------
# Artifact Flag (binary)
# -------------------------------------------------------------
def plot_umap_by_artifact(umap_xy, flags, artifact_type, layer, split, save_path):
    try:
        fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"], dpi=VIS_CONFIG["dpi"])
        flags = flags.astype(int)

        colors = COLORMAPS["artifacts"]
        neg, pos = flags == 0, flags == 1

        ax.scatter(
            umap_xy[neg, 0], umap_xy[neg, 1],
            c=colors[0], s=VIS_CONFIG["marker_size"],
            alpha=0.4, label=f"No ({neg.sum()})", edgecolors="none"
        )
        ax.scatter(
            umap_xy[pos, 0], umap_xy[pos, 1],
            c=colors[1], s=VIS_CONFIG["marker_size"],
            alpha=0.8, label=f"Yes ({pos.sum()})",
            edgecolors="black", linewidths=0.4
        )

        name = artifact_type.replace("_flag", "").title()
        ax.set_title(f"{layer.upper()} - {split.capitalize()} ({name})")
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()
        logger.info(f"✓ Saved artifact plot: {save_path}")

    except Exception as e:
        logger.error(f"Failed artifact plot {layer}/{split}/{artifact_type}: {e}")

def plot_layer_comparison_bar(metrics_summary, metric_name, save_path):
        """
        Creates a bar plot comparing metric values across layers.
        metrics_summary: dict {layer: value}
        """

        import matplotlib.pyplot as plt
        import numpy as np

        layers = list(metrics_summary.keys())
        values = [metrics_summary[layer] for layer in layers]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=VIS_CONFIG["dpi"])

        ax.bar(layers, values, color="steelblue", alpha=0.8)
        ax.set_title(f"{metric_name} Comparison Across Layers")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()

        logger.info(f"✓ Saved bar comparison: {save_path}")


# -------------------------------------------------------------
# Main layer/split visualization
# -------------------------------------------------------------
def visualize_layer_split(layer, split, projections_dict, labels, output_dir):
    logger.info(f"▶ Visualizing {layer}/{split}...")

    if projections_dict is None:
        logger.error(f"No projection dict for {layer}/{split}")
        return

    # 1. MOS
    mos_xy = projections_dict.get("mos", None)
    if mos_xy is not None and "mos" in labels:
        plot_umap_by_mos(
            mos_xy,
            labels["mos"],
            layer, split,
            get_umap_path(split, layer, "mos")
        )

    # 2. Quality Class
    qc_xy = projections_dict.get("quality_class", None)
    if qc_xy is not None and "quality_class" in labels:
        plot_umap_by_quality_class(
            qc_xy,
            labels["quality_class"],
            layer, split,
            get_umap_path(split, layer, "quality_class")
        )

    # 3. Artifact flags
    for art in ARTIFACT_TYPES:
        xy = projections_dict.get(art, None)
        if xy is not None and art in labels:
            plot_umap_by_artifact(
                xy,
                labels[art],
                art,
                layer, split,
                get_umap_path(split, layer, art)
            )

    logger.info(f"✓ Completed visualizations for {layer}/{split}")


# -------------------------------------------------------------
# Comparison Grid
# -------------------------------------------------------------
def plot_layer_comparison_grid(all_results, label_type, save_path):
    """
    Creates comparison grid across layers and splits for a specific label type
    """
    fig, axes = plt.subplots(
        len(LAYERS), len(SPLITS),
        figsize=(5 * len(SPLITS), 5 * len(LAYERS)),
        dpi=VIS_CONFIG["dpi"],
        squeeze=False
    )

    for i, layer in enumerate(LAYERS):
        for j, split in enumerate(SPLITS):
            ax = axes[i][j]

            # FIX: Access nested structure correctly
            entry = all_results.get(layer, {}).get(split, {})
            projections = entry.get("projections", {})  # Get projections dict
            labels = entry.get("labels", {})  # Get labels dict
            
            # Now get the specific projection and label for this label_type
            proj = projections.get(label_type)
            lbls = labels.get(label_type)

            if proj is None or lbls is None:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", 
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{layer} - {split}")
                continue

            # Determine colormap based on label type
            if label_type == "mos":
                sc = ax.scatter(
                    proj[:, 0], proj[:, 1],
                    c=lbls, cmap=COLORMAPS["mos"],
                    s=10, alpha=0.6, edgecolors="none"
                )
            elif label_type == "quality_class":
                # Categorical coloring
                classes = np.unique(lbls)
                cmap = plt.cm.get_cmap(COLORMAPS["quality_class"], len(classes))
                for k, cls in enumerate(classes):
                    mask = lbls == cls
                    ax.scatter(
                        proj[mask, 0], proj[mask, 1],
                        c=[cmap(k)], s=10, alpha=0.6, edgecolors="none"
                    )
            else:
                # Binary artifact flags
                lbls_int = lbls.astype(int)
                colors = COLORMAPS["artifacts"]
                for val, color in enumerate(colors):
                    mask = lbls_int == val
                    ax.scatter(
                        proj[mask, 0], proj[mask, 1],
                        c=color, s=10, alpha=0.6, edgecolors="none"
                    )

            ax.set_title(f"{layer} - {split}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
    plt.close()
    logger.info(f"✓ Saved comparison grid: {save_path}")


# -------------------------------------------------------------
# Metrics Heatmap (safe version)
# -------------------------------------------------------------
def plot_metrics_heatmap(metrics_df: pd.DataFrame, metric_name: str, save_path: Path):
    if metrics_df.empty:
        logger.warning(f"No metrics for heatmap {metric_name}")
        return

    try:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=VIS_CONFIG["dpi"])

        sns.heatmap(
            metrics_df, annot=True, fmt=".3f",
            cmap=COLORMAPS["heatmap"],
            linewidths=1, linecolor="white",
            ax=ax
        )

        ax.set_title(f"{metric_name} Across Layers/Splits")
        plt.tight_layout()
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()

        logger.info(f"✓ Saved metrics heatmap: {save_path}")

    except Exception as e:
        logger.error(f"Heatmap failed for {metric_name}: {e}")


# -------------------------------------------------------------
# Top-level Visualization Pipeline
# -------------------------------------------------------------
def create_all_visualizations(all_results: Dict[str, Dict[str, Any]]):
    logger.info("=== Starting Visualization Pipeline ===")

    # 1. Individual UMAP plots
    for layer in LAYERS:
        for split in SPLITS:
            entry = safe_get(all_results, layer, split, default=None)
            if entry is None:
                continue
            visualize_layer_split(
                layer, split,
                entry.get("projection"),
                entry.get("labels"),
                PLOTS_DIR
            )

    # 2. Comparison grids
    for label_type in ["mos", "quality_class"] + ARTIFACT_TYPES:
        plot_layer_comparison_grid(
            all_results,
            label_type,
            get_comparison_path(f"compare_{label_type}.png")
        )

    logger.info("✓ Visualization Pipeline Complete!")
