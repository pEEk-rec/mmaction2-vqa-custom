from mmaction.registry import METRICS
from mmengine.evaluator import BaseMetric
import math
from scipy.stats import spearmanr, pearsonr

@METRICS.register_module()
class VQAMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def _get_meta(self, ds):
        if hasattr(ds, 'metainfo'):
            return ds.metainfo
        if isinstance(ds, dict):
            if 'metainfo' in ds:
                return ds['metainfo']
            return ds
        raise TypeError(f'Unsupported data_sample type: {type(ds)}')

    def process(self, data_batch, data_samples):
        for ds in data_samples:
            meta = self._get_meta(ds)
            pm = meta.get('pred_mos', None)
            if pm is None:
                continue
            item = dict(
                pred_mos=float(pm),
                gt_mos=float(meta['mos']),
                pred_qclass=int(meta['pred_qclass']),
                gt_qclass=int(meta['quality_class']),
                pred_dtype=int(meta['pred_dtype']),
                gt_dtype=int(meta['distortion_type']),
            )
            self.results.append(item)

    def compute_metrics(self, results):
        if len(results) == 0:
            return dict(
                mos_mae=float('nan'),
                mos_rmse=float('nan'),
                mos_srcc=float('nan'),
                mos_plcc=float('nan'),
                qclass_acc=float('nan'),
                dtype_acc=float('nan'),
            )

        pred_mos = [r['pred_mos'] for r in results]
        gt_mos   = [r['gt_mos']   for r in results]
        n = max(1, len(gt_mos))
        mae = sum(abs(p - g) for p, g in zip(pred_mos, gt_mos)) / n
        rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(pred_mos, gt_mos)) / n)
        srcc = spearmanr(gt_mos, pred_mos).correlation if len(gt_mos) > 1 else float('nan')
        plcc = pearsonr(gt_mos, pred_mos)[0] if len(gt_mos) > 1 else float('nan')

        q_acc = sum(int(r['pred_qclass'] == r['gt_qclass']) for r in results) / n
        d_acc = sum(int(r['pred_dtype'] == r['gt_dtype']) for r in results) / n

        return dict(
            mos_mae=mae,
            mos_rmse=rmse,
            mos_srcc=srcc,
            mos_plcc=plcc,
            qclass_acc=q_acc,
            dtype_acc=d_acc
        )
