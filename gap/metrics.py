"""Absolute MPJPE, pelvis position error, and root-relative MPJPE."""

from __future__ import annotations

import torch


def pose_metrics(
    prediction: torch.Tensor, target: torch.Tensor, root_index: int = 0
) -> dict:
    """Average every sample/joint once and report millimeters from meter inputs.

    Root-relative evaluation subtracts the predicted pelvis from prediction and
    the ground-truth pelvis from target. No rotation or scale alignment is used.
    """
    pred = torch.as_tensor(prediction).float()
    gt = torch.as_tensor(target, device=pred.device).float()
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("Expected matching prediction and target arrays [N,J,3]")
    if pred.shape[0] == 0 or not 0 <= root_index < pred.shape[1]:
        raise ValueError("Expected nonempty poses and a valid pelvis index")
    if not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("Poses contain NaN or Inf")
    errors = torch.linalg.norm(pred - gt, dim=-1)
    pred_relative = pred - pred[:, root_index : root_index + 1]
    gt_relative = gt - gt[:, root_index : root_index + 1]
    relative_errors = torch.linalg.norm(pred_relative - gt_relative, dim=-1)
    return {
        "sample_count": int(pred.shape[0]),
        "joint_count": int(pred.shape[1]),
        "absolute_mpjpe_mm": float(errors.mean().item() * 1000.0),
        "root_error_mm": float(errors[:, root_index].mean().item() * 1000.0),
        "root_relative_mpjpe_mm": float(relative_errors.mean().item() * 1000.0),
    }
