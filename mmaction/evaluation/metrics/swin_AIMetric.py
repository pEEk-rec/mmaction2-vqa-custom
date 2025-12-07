# mmaction/evaluation/swin_AIMetric.py
from typing import List, Optional, Dict, Any
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS
from mmengine.dist import get_rank


@METRICS.register_module()
class swin_AIMetric(BaseMetric):
    """Robust metric collector for Stage-0 AI-VQA.
    Produces:
      - MOS: SRCC (primary), PLCC, MAE, RMSE
      - Quality (5-class): Macro-F1 (primary), BalancedAcc, Top-1 Acc, Confusion Matrix
      - Binary artifact tasks: AUC-ROC (primary) + AP (PR-AUC), Precision/Recall/F1 @0.5, TP/FP/FN/TN
    """

    default_prefix: Optional[str] = None

    def __init__(self, collect_device: str = "cpu", prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        # internal accumulator
        self.results: List[Dict[str, Any]] = []

    def _extract_value(self, ds, key, default=None):
        """Safely extract value from sample/datasample."""
        if hasattr(ds, "metainfo") and isinstance(ds.metainfo, dict) and key in ds.metainfo:
            return ds.metainfo[key]
        if hasattr(ds, key):
            return getattr(ds, key)
        if isinstance(ds, dict) and key in ds:
            return ds[key]
        return default

    def process(self, data_batch, data_samples):
        """Accumulate predictions and ground truth per sample."""
        for ds in data_samples:
            # extract key fields; allow fallback names
            pred_mos = self._extract_value(ds, "pred_mos", None)
            gt_mos = self._extract_value(ds, "mos", None)
            if pred_mos is None or gt_mos is None:
                # skip incomplete sample
                continue

            pred_qclass = self._extract_value(ds, "pred_qclass", None)
            if pred_qclass is None:
                pred_qclass = self._extract_value(ds, "pred_quality_class", None)

            gt_qclass = self._extract_value(ds, "quality_class", None)

            item = {
                "pred_mos": float(pred_mos),
                "gt_mos": float(gt_mos),
                "pred_qclass": int(pred_qclass) if pred_qclass is not None else 0,
                "gt_qclass": int(gt_qclass) if gt_qclass is not None else 0,
                # artifacts (predictions from head are class indices; convert to 0/1)
                "pred_hallucination": int(self._extract_value(ds, "pred_hallucination", 0)),
                "gt_hallucination": int(self._extract_value(ds, "hallucination_flag", 0)),
                "pred_lighting": int(self._extract_value(ds, "pred_lighting", 0)),
                "gt_lighting": int(self._extract_value(ds, "lighting_flag", 0)),
                "pred_spatial": int(self._extract_value(ds, "pred_spatial", 0)),
                "gt_spatial": int(self._extract_value(ds, "spatial_flag", 0)),
                "pred_rendering": int(self._extract_value(ds, "pred_rendering", 0)),
                "gt_rendering": int(self._extract_value(ds, "rendering_flag", 0)),
            }
            self.results.append(item)

    # helper for binary metrics robustly
    def _binary_summary(self, preds, gts, prefix):
        out = {}
        preds = np.array(preds)
        gts = np.array(gts)
        out[f"{prefix}_acc"] = float(np.mean(preds == gts)) if len(preds) > 0 else 0.0

        # counts
        tn, fp, fn, tp = (0, 0, 0, 0)
        if len(preds) > 0:
            try:
                tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0, 1]).ravel()
            except Exception:
                # fallback safe counting
                tp = int(((preds == 1) & (gts == 1)).sum())
                fp = int(((preds == 1) & (gts == 0)).sum())
                fn = int(((preds == 0) & (gts == 1)).sum())
                tn = int(((preds == 0) & (gts == 0)).sum())

        out[f"{prefix}_tp"] = int(tp)
        out[f"{prefix}_fp"] = int(fp)
        out[f"{prefix}_fn"] = int(fn)
        out[f"{prefix}_tn"] = int(tn)

        # precision/recall/f1 at threshold 0.5 (degenerate-handled)
        unique_gt = np.unique(gts)
        if len(gts) == 0:
            out[f"{prefix}_precision"] = 0.0
            out[f"{prefix}_recall"] = 0.0
            out[f"{prefix}_f1"] = 0.0
        else:
            if len(unique_gt) == 1:
                # degenerate: sklearn may give warnings; produce meaningful fallbacks
                if unique_gt[0] == 0:
                    out[f"{prefix}_precision"] = 1.0 if preds.sum() == 0 else 0.0
                    out[f"{prefix}_recall"] = 1.0
                    out[f"{prefix}_f1"] = 0.0
                else:  # all positive
                    out[f"{prefix}_precision"] = 1.0 if (preds == 1).all() else 0.0
                    out[f"{prefix}_recall"] = out[f"{prefix}_precision"]
                    out[f"{prefix}_f1"] = out[f"{prefix}_precision"]
            else:
                out[f"{prefix}_precision"] = float(precision_score(gts, preds, zero_division=0))
                out[f"{prefix}_recall"] = float(recall_score(gts, preds, zero_division=0))
                out[f"{prefix}_f1"] = float(f1_score(gts, preds, zero_division=0))

        # AUC-ROC & AP: need scores or label probabilities; if we only have hard preds, try to compute using them safely
        # If preds are only class (0/1), roc_auc_score can still be computed but provides limited info.
        try:
            out[f"{prefix}_aucroc"] = float(roc_auc_score(gts, preds)) if len(np.unique(gts)) > 1 else 0.0
        except Exception:
            out[f"{prefix}_aucroc"] = 0.0
        try:
            out[f"{prefix}_ap"] = float(average_precision_score(gts, preds)) if len(np.unique(gts)) > 1 else 0.0
        except Exception:
            out[f"{prefix}_ap"] = 0.0

        return out

    def compute_metrics(self, results: Optional[List[dict]] = None) -> dict:
        """Compute and return consolidated metrics dict."""
        if results is None:
            results = self.results

        if not results:
            # empty safe defaults
            empty = {
                "mos_mae": 0.0,
                "mos_rmse": 0.0,
                "mos_srcc": 0.0,
                "mos_plcc": 0.0,
                "qclass_acc": 0.0,
                "qclass_balanced_acc": 0.0,
                "qclass_macro_f1": 0.0,
                "num_samples": 0,
            }
            # add binary keys for consistency
            for t in ["hallucination", "lighting", "spatial", "rendering"]:
                empty.update({
                    f"{t}_acc": 0.0, f"{t}_precision": 0.0, f"{t}_recall": 0.0, f"{t}_f1": 0.0,
                    f"{t}_tp": 0, f"{t}_fp": 0, f"{t}_fn": 0, f"{t}_tn": 0,
                    f"{t}_aucroc": 0.0, f"{t}_ap": 0.0
                })
            return empty

        metrics: Dict[str, Any] = {}
        pred_mos = np.array([r["pred_mos"] for r in results])
        gt_mos = np.array([r["gt_mos"] for r in results])

        # Regression metrics
        metrics["mos_mae"] = float(np.mean(np.abs(pred_mos - gt_mos)))
        metrics["mos_rmse"] = float(np.sqrt(np.mean((pred_mos - gt_mos) ** 2)))

        if len(pred_mos) > 1:
            try:
                srcc, _ = spearmanr(pred_mos, gt_mos)
                metrics["mos_srcc"] = float(srcc if not np.isnan(srcc) else 0.0)
            except Exception:
                metrics["mos_srcc"] = 0.0
            try:
                plcc, _ = pearsonr(pred_mos, gt_mos)
                metrics["mos_plcc"] = float(plcc if not np.isnan(plcc) else 0.0)
            except Exception:
                metrics["mos_plcc"] = 0.0
        else:
            metrics["mos_srcc"] = 0.0
            metrics["mos_plcc"] = 0.0

        # Quality classification
        pred_q = np.array([r["pred_qclass"] for r in results])
        gt_q = np.array([r["gt_qclass"] for r in results])
        metrics["qclass_acc"] = float(np.mean(pred_q == gt_q))
        try:
            metrics["qclass_balanced_acc"] = float(balanced_accuracy_score(gt_q, pred_q))
        except Exception:
            metrics["qclass_balanced_acc"] = float(metrics["qclass_acc"])
        try:
            metrics["qclass_macro_f1"] = float(f1_score(gt_q, pred_q, average="macro", zero_division=0))
        except Exception:
            metrics["qclass_macro_f1"] = float(metrics["qclass_acc"])

        # Confusion matrix (for debugging)
        try:
            cm = confusion_matrix(gt_q, pred_q)
            metrics["qclass_confusion_matrix"] = cm.tolist()
        except Exception:
            metrics["qclass_confusion_matrix"] = []

        # Binary artifact tasks
        for task in ["hallucination", "lighting", "spatial", "rendering"]:
            pkey = f"pred_{task}"
            gkey = f"gt_{task}"
            preds = [r[pkey] for r in results]
            gts = [r[gkey] for r in results]
            bm = self._binary_summary(preds, gts, task)
            metrics.update(bm)

        metrics["num_samples"] = len(results)

        # Print concise report only on rank 0
        if get_rank() == 0:
            print("\n" + "=" * 60)
            print("AI-VQA Summary")
            print("=" * 60)
            print(f"MOS (primary=SRCC): SRCC={metrics['mos_srcc']:.4f}, PLCC={metrics['mos_plcc']:.4f}, MAE={metrics['mos_mae']:.4f}, RMSE={metrics['mos_rmse']:.4f}")
            print(f"Quality (primary=Macro-F1): MacroF1={metrics['qclass_macro_f1']:.4f}, BalancedAcc={metrics['qclass_balanced_acc']:.4f}, Acc={metrics['qclass_acc']:.4f}")
            for t in ["hallucination", "lighting", "spatial", "rendering"]:
                print(f"{t.capitalize()} → AUC={metrics[f'{t}_aucroc']:.4f}, AP={metrics[f'{t}_ap']:.4f}, F1={metrics[f'{t}_f1']:.4f}, TP={metrics[f'{t}_tp']}, FP={metrics[f'{t}_fp']}, FN={metrics[f'{t}_fn']}, TN={metrics[f'{t}_tn']}")
            print(f"Samples: {metrics['num_samples']}")
            print("=" * 60 + "\n")

        return metrics
