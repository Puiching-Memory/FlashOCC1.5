import numpy as np
import torch


def convert_ann_infos_to_tensors(gt_boxes, gt_labels):
    """Convert annotation arrays to torch tensors used by the BEV pipeline."""
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes).float()
    elif not torch.is_tensor(gt_boxes):
        gt_boxes = torch.tensor(np.array(gt_boxes), dtype=torch.float32)
    else:
        gt_boxes = gt_boxes.float()

    if isinstance(gt_labels, np.ndarray):
        gt_labels = torch.from_numpy(gt_labels).long()
    elif not torch.is_tensor(gt_labels):
        gt_labels = torch.tensor(np.array(gt_labels), dtype=torch.long)
    else:
        gt_labels = gt_labels.long()

    return gt_boxes, gt_labels
