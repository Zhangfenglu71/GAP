# GAP

GAP: Grounding Absolute 3D Human Pose Estimation in Millimeter-Wave Radar Measurements

Fenglu Zhang, Yang Yang, and Jianpu Liu

## Release scope

This repository provides the GAP module, radar measurement reference construction,
and evaluation code. GAP takes cached initial poses from an existing estimator and
same-frame radar position references. It predicts the current absolute 22-joint pose.

Training code, base estimators, baseline methods, dataset files, and pretrained
weights are not included. Running the model evaluator requires a compatible GAP
checkpoint. The saved-prediction evaluator can be used without a checkpoint.

## Installation

Python 3.9 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The package can also be installed with `python -m pip install -e .`.

## GAP module

```python
import torch
from gap import GAP, measurement_reference

model = GAP().eval()
initial_poses = torch.zeros(1, 9, 22, 3)
references = torch.zeros(1, 9, 3)
with torch.no_grad():
    prediction = model(initial_poses, references)["pred_pose_absolute"]
```

The example above uses an untrained model. Its zero-initialized output layer
initially preserves the current initial pose.

The temporal encoder uses the VideoPose3D implementation with filter widths
`[3, 3]`, causal convolutions, 1024 channels, and dropout 0.25. Nine paired frames
have shape `[B,9,23,3]`. The output has shape `[B,22,3]`. The default model contains
4,480,066 parameters. `configs/gap.json` records the evaluation configuration.

`measurement_reference(points, initial_pelvis)` computes coordinate-wise xyz
medians from a padded current-frame radar point cloud. A row is padding when all
its channels are zero. A caller can instead supply an explicit validity mask.
Empty frames use the initial pelvis coordinate. Even-sized point sets use the
lower middle order statistic, as in `torch.median` in the experiment code.

## Input format

Prepare separate feature and target files as `.npz` arrays or `.pt` tensor
dictionaries. NumPy object arrays are not supported. PyTorch files are read with
`weights_only=True`.

| Feature field | Shape | Meaning |
| --- | --- | --- |
| `indicator` | `[N,3]` | Integer subject, action, and frame identifiers |
| `coarse_pose` | `[N,22,3]` | Initial absolute joint coordinates |
| `measurement_reference` | `[N,3]` | Per-frame radar measurement references |

The original cache field name `rpc_anchor` is also accepted in place of
`measurement_reference`. Additional cache metadata is not used.

| Target field | Shape | Meaning |
| --- | --- | --- |
| `indicator` | `[N,3]` | The same identifiers in the same row order |
| `target_pose` | `[N,22,3]` | Ground-truth absolute joint coordinates |

The pelvis is joint 0. Every array must use the same 22-joint ordering and radar
coordinate frame. Inputs are already prepared coordinates. The evaluator does not
apply calibration or shift coordinates automatically. The paper uses a -1.5 m
z-shift for radar-frame poses and measurements. Use `--units mm` for millimeter
input files, which are converted to meters before model inference.

Nine-frame windows are built separately for each subject/action sequence using
frame order. Missing history repeats the earliest available frame. Earlier
available frames are retained across gaps. Feature files must contain only the
split being evaluated and include the histories required by its target frames.

## Evaluation

Run a GAP checkpoint on cached inputs:

```bash
python evaluate.py model \
  --features data/features.npz \
  --labels data/labels.npz \
  --checkpoint checkpoints/gap.pth \
  --device cpu \
  --output outputs/metrics.json \
  --save-predictions outputs/predictions.npz
```

Use `--device cuda` on a CUDA-enabled installation. The default batch size is 64.
The loader accepts a tensor state dictionary, `model_state_dict`, or `state_dict`.
Original GAP checkpoints with `temporal.*` parameter names are compatible.

For the Pose root control, use `--reference pose-root` with its separately trained
Pose root checkpoint. That option replaces each radar reference with the initial
pelvis. Changing the input to a radar-trained checkpoint does not reproduce the
trained Pose root control.

Evaluate saved absolute predictions:

```bash
python evaluate.py predictions \
  --predictions outputs/predictions.npz \
  --labels data/labels.npz \
  --output outputs/prediction_metrics.json
```

Prediction files contain `indicator` with shape `[N,3]` and `prediction` with
shape `[N,22,3]`. Predictions saved by the model command are always in meters.

All three metrics are reported in millimeters:

- Absolute MPJPE averages joint position errors over all samples and joints.
- Root error averages the absolute position error of pelvis joint 0.
- Root-relative MPJPE subtracts each pose's own pelvis before computing joint
  errors. It uses no rotation or scale alignment.

Batch outputs are concatenated before computing global means. To summarize the
three trained runs in the paper, evaluate each seed separately, then compute the
mean and sample standard deviation of the three metric values, using `ddof=1`.
Data and checkpoints must match the estimator, split, and seed being reported.

## Interface check

```bash
python -m unittest discover -s tests -v
python examples/make_example.py --output demo
python evaluate.py model \
  --features demo/features.npz \
  --labels demo/labels.npz \
  --checkpoint demo/untrained_demo.pth \
  --output outputs/demo_metrics.json
```

The synthetic example checks file formats and inference. It is not a paper
experiment and does not reproduce the reported accuracy.

## Citation and upstream implementation

```bibtex
@unpublished{zhang2026gap,
  title = {GAP: Grounding Absolute 3D Human Pose Estimation in Millimeter-Wave Radar Measurements},
  author = {Zhang, Fenglu and Yang, Yang and Liu, Jianpu},
  year = {2026},
  note = {Manuscript}
}

@inproceedings{pavllo2019videopose3d,
  title = {3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training},
  author = {Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2019},
  pages = {7753--7762}
}
```

The original temporal backbone is included unchanged under
`gap/vendor/videopose3d/`, from VideoPose3D commit
`1afb1ca0f1237776518469876342fc8669d3f6a9`.

## License

This research code is available under **CC BY-NC 4.0**, which includes a
noncommercial restriction. It is not a permissive MIT or Apache-licensed release.
See `LICENSE`, `NOTICE`, and the original license next to the VideoPose3D source.
