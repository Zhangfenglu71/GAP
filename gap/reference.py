"""Coordinate-wise measurement references from padded radar point clouds."""

from __future__ import annotations

import torch


def measurement_reference(
    radar_points: torch.Tensor,
    initial_pelvis: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the xyz median and whether at least one radar point was used.

    radar_points has shape [N,C] with C >= 3 and xyz in the first three columns.
    Without a mask, rows whose every channel is zero are padding. torch.median
    selects the lower middle value for an even number of valid points, matching
    the experiment implementation. An empty frame uses the initial pelvis.
    """
    points = torch.as_tensor(radar_points)
    root = torch.as_tensor(initial_pelvis, dtype=torch.float32, device=points.device)
    if points.ndim != 2 or points.shape[1] < 3 or tuple(root.shape) != (3,):
        raise ValueError("Expected radar points [N,C>=3] and initial pelvis [3]")
    if not torch.isfinite(points).all() or not torch.isfinite(root).all():
        raise ValueError("Coordinates must be finite")
    if valid_mask is None:
        valid = torch.any(points != 0, dim=-1)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=points.device)
        if tuple(valid.shape) != (points.shape[0],):
            raise ValueError("Expected a validity mask [N]")
    if bool(valid.any()):
        return points[valid, :3].float().median(dim=0).values, valid.any()
    return root.clone(), valid.any()
