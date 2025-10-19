# mmaction2/mmaction/datasets/transforms/formatting.py
# Add this to the existing formatting.py file

from mmaction.registry import TRANSFORMS
from mmengine.structures import LabelData
import torch

@TRANSFORMS.register_module()
class PackActionInputsMOS(PackActionInputs):
    """Pack inputs for MOS regression.
    
    Extends PackActionInputs to handle MOS scores in addition to labels.
    """
    
    def transform(self, results: dict) -> dict:
        """Transform function.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Packed results.
        """
        packed_results = super().transform(results)
        
        # Add MOS score to gt_label
        if 'mos' in results:
            if not hasattr(packed_results['data_samples'], 'gt_label'):
                packed_results['data_samples'].gt_label = LabelData()
            
            packed_results['data_samples'].gt_label.mos = torch.tensor(
                results['mos'], dtype=torch.float32
            )
        
        return packed_results