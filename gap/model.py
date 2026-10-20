"""Measurement-conditioned temporal prediction of absolute joint coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .vendor.videopose3d import TemporalModel


@dataclass(frozen=True)
class GAPConfig:
    num_joints: int = 22
    channels: int = 1024
    dropout: float = 0.25
    history_length: int = 9

    def __post_init__(self) -> None:
        if self.num_joints != 22 or self.history_length != 9:
            raise ValueError("This release uses 22 joints and a nine-frame history")
        if self.channels <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid channel count or dropout probability")


class GAP(nn.Module):
    """Predict the current absolute pose from nine paired pose/reference frames.

    Input coordinates must share a radar coordinate frame and use meters.
    The temporal parameter names match the original GAP model checkpoints.
    """

    def __init__(self, config: GAPConfig | None = None):
        super().__init__()
        self.config = config or GAPConfig()
        self.temporal = TemporalModel(
            num_joints_in=self.config.num_joints + 1,
            in_features=3,
            num_joints_out=self.config.num_joints,
            filter_widths=[3, 3],
            causal=True,
            dropout=self.config.dropout,
            channels=self.config.channels,
            dense=False,
        )
        if self.temporal.receptive_field() != self.config.history_length:
            raise RuntimeError("Unexpected temporal receptive field")
        nn.init.zeros_(self.temporal.shrink.weight)
        nn.init.zeros_(self.temporal.shrink.bias)

    def set_bn_momentum(self, momentum: float) -> None:
        self.temporal.set_bn_momentum(float(momentum))

    def forward(
        self,
        coarse_pose_seq: torch.Tensor,
        measurement_reference_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = (self.config.history_length, self.config.num_joints, 3)
        if coarse_pose_seq.ndim != 4 or tuple(coarse_pose_seq.shape[1:]) != expected:
            raise ValueError(f"Expected initial poses [B,{expected[0]},22,3]")
        if tuple(measurement_reference_seq.shape) != (
            coarse_pose_seq.shape[0], self.config.history_length, 3
        ):
            raise ValueError("Expected measurement references [B,9,3]")
        paired = torch.cat(
            [coarse_pose_seq.float(), measurement_reference_seq[:, :, None, :].float()],
            dim=2,
        )
        residual = self.temporal(paired)
        if tuple(residual.shape[1:]) != (1, self.config.num_joints, 3):
            raise RuntimeError(f"Unexpected coordinate output {tuple(residual.shape)}")
        joint_residual = residual[:, 0]
        prediction = coarse_pose_seq[:, -1].float() + joint_residual
        return {"pred_pose_absolute": prediction, "joint_residual": joint_residual}
