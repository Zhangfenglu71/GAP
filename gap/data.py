"""Cached pose/reference inputs and causal windows for evaluation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def load_arrays(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {key: torch.from_numpy(payload[key].copy()) for key in payload.files}
    if path.suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, dict):
            raise ValueError("Expected a tensor dictionary")
        return value
    raise ValueError("Expected .npz, .pt, or .pth")


def indicators(value: torch.Tensor) -> torch.Tensor:
    raw = torch.as_tensor(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] == 0:
        raise ValueError("Expected nonempty indicators [N,3]: subject, action, frame")
    if not torch.isfinite(raw).all() or not torch.equal(raw, raw.long().to(raw.dtype)):
        raise ValueError("Indicators must be finite integers")
    result = raw.long()
    if torch.unique(result, dim=0).shape[0] != result.shape[0]:
        raise ValueError("Duplicate subject/action/frame indicators")
    return result


def causal_window_indices(indicator: torch.Tensor, history_length: int = 9) -> torch.Tensor:
    """Order available frames within each subject/action and left-pad the first.

    Windows never cross subject/action boundaries. Frame gaps retain the earlier
    available observations, matching the evaluation protocol.
    """
    values = indicators(indicator)
    if history_length <= 0:
        raise ValueError("History length must be positive")
    groups = {}
    for row, (subject, action, frame) in enumerate(values.tolist()):
        groups.setdefault((subject, action), []).append((frame, row))
    windows = torch.empty((len(values), history_length), dtype=torch.long)
    for items in groups.values():
        ordered = [row for _, row in sorted(items)]
        for position, row in enumerate(ordered):
            history = ordered[max(0, position - history_length + 1) : position + 1]
            windows[row] = torch.tensor(
                [history[0]] * (history_length - len(history)) + history
            )
    return windows


def pose_array(value: torch.Tensor, sample_count: int, scale: float) -> torch.Tensor:
    pose = torch.as_tensor(value).float() * scale
    if tuple(pose.shape) != (sample_count, 22, 3) or not torch.isfinite(pose).all():
        raise ValueError("Expected finite poses [N,22,3]")
    return pose


class GAPDataset(Dataset):
    def __init__(
        self, features: dict, labels: dict, units: str = "m", reference: str = "radar"
    ):
        if units not in {"m", "mm"} or reference not in {"radar", "pose-root"}:
            raise ValueError("Invalid units or reference type")
        self.indicator = indicators(features["indicator"])
        if not torch.equal(self.indicator, indicators(labels["indicator"])):
            raise ValueError("Feature and target indicators must have identical row order")
        scale = 1.0 if units == "m" else 0.001
        count = len(self.indicator)
        self.coarse_pose = pose_array(features["coarse_pose"], count, scale)
        self.target_pose = pose_array(labels["target_pose"], count, scale)
        if reference == "pose-root":
            self.reference = self.coarse_pose[:, 0].clone()
        else:
            key = "measurement_reference" if "measurement_reference" in features else "rpc_anchor"
            self.reference = torch.as_tensor(features[key]).float() * scale
            if tuple(self.reference.shape) != (count, 3) or not torch.isfinite(self.reference).all():
                raise ValueError("Expected finite measurement references [N,3]")
            if "measurement_reference" in features and "rpc_anchor" in features:
                legacy = torch.as_tensor(features["rpc_anchor"]).float() * scale
                if not torch.equal(legacy, self.reference):
                    raise ValueError("Conflicting reference fields")
        self.windows = causal_window_indices(self.indicator)

    def __len__(self) -> int:
        return int(self.indicator.shape[0])

    def __getitem__(self, row: int) -> dict:
        history = self.windows[row]
        return {
            "coarse_pose_seq": self.coarse_pose[history],
            "measurement_reference_seq": self.reference[history],
        }
