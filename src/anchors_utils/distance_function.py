import torch
from torchvision.ops import box_iou


def IoU(clusters: torch.tensor, bboxes: torch.tensor) -> float:
    iou_values = box_iou(clusters, bboxes)
    return iou_values