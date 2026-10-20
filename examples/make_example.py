#!/usr/bin/env python3
"""Create synthetic inputs and an untrained checkpoint for an interface check."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gap import GAP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    ids = np.array([[subject, 0, frame] for subject in range(2) for frame in range(6)])
    target = rng.normal(0, 0.1, (len(ids), 22, 3)).astype(np.float32)
    target[..., 2] += 2.0
    coarse = target + np.array([0.05, 0.0, 0.02], dtype=np.float32)
    np.savez_compressed(
        args.output / "features.npz",
        indicator=ids,
        coarse_pose=coarse,
        measurement_reference=target[:, 0],
    )
    np.savez_compressed(args.output / "labels.npz", indicator=ids, target_pose=target)
    torch.save(GAP().state_dict(), args.output / "untrained_demo.pth")
    print("Created synthetic inputs and an untrained, zero-output checkpoint.")
    print("These files check the interface and do not reproduce paper results.")


if __name__ == "__main__":
    main()
