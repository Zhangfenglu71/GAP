#!/usr/bin/env python3
"""Evaluate GAP checkpoints or saved absolute pose predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gap import GAP, GAPConfig, pose_metrics
from gap.data import GAPDataset, indicators, load_arrays, pose_array


def load_model(checkpoint: Path, device: torch.device) -> GAP:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a state dictionary or a checkpoint dictionary")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Checkpoint model state must contain tensors only")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    key = "temporal.expand_conv.weight"
    if key not in state or state[key].shape[1:] != (69, 3):
        raise ValueError("Expected a GAP checkpoint with 69 input coordinates")
    model = GAP(GAPConfig(channels=int(state[key].shape[0])))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate_model(args: argparse.Namespace) -> dict:
    dataset = GAPDataset(
        load_arrays(args.features), load_arrays(args.labels), args.units, args.reference
    )
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["coarse_pose_seq"].to(device),
                batch["measurement_reference_seq"].to(device),
            )
            outputs.append(output["pred_pose_absolute"].cpu())
    prediction = torch.cat(outputs)
    result = {
        "baseline": pose_metrics(dataset.coarse_pose, dataset.target_pose),
        "gap": pose_metrics(prediction, dataset.target_pose),
        "reference": args.reference,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_predictions,
            indicator=dataset.indicator.numpy(),
            prediction=prediction.numpy(),
        )
    return result


def evaluate_predictions(args: argparse.Namespace) -> dict:
    predictions = load_arrays(args.predictions)
    targets = load_arrays(args.labels)
    ids = indicators(predictions["indicator"])
    if not torch.equal(ids, indicators(targets["indicator"])):
        raise ValueError("Prediction and target indicators must have identical row order")
    scale = 1.0 if args.units == "m" else 0.001
    pred = pose_array(predictions["prediction"], len(ids), scale)
    gt = pose_array(targets["target_pose"], len(ids), scale)
    return {"prediction": pose_metrics(pred, gt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    model = sub.add_parser("model", help="Run GAP inference on cached inputs")
    model.add_argument("--features", type=Path, required=True)
    model.add_argument("--checkpoint", type=Path, required=True)
    model.add_argument("--batch-size", type=int, default=64)
    model.add_argument("--device", default="cpu")
    model.add_argument("--reference", choices=["radar", "pose-root"], default="radar")
    model.add_argument("--save-predictions", type=Path)
    prediction = sub.add_parser("predictions", help="Score saved absolute joint coordinates")
    prediction.add_argument("--predictions", type=Path, required=True)
    for command in [model, prediction]:
        command.add_argument("--labels", type=Path, required=True)
        command.add_argument("--units", choices=["m", "mm"], default="m")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "model" and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    result = evaluate_model(args) if args.command == "model" else evaluate_predictions(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
