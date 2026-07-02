# memory-qc-edge

[![ci](https://github.com/JPL11/memory_qc_edge/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL11/memory_qc_edge/actions/workflows/ci.yml)

Deployable, **numpy-only** on-device memory-QC scoring layer for MRI
segmentation failure review. This is the edge half of the memory-QC agent,
separated from the research/training code (SpikeBridge, segmenter training,
feature extraction) so it can run on a Jetson-class device with no deep-learning
framework installed.

## What it does

Given per-slice feature rows produced by *any* frozen segmenter, it emits a
per-slice failure-review risk score by combining:

1. a **frozen logistic controller** over 29 base features, and
2. a **class-balanced memory retrieval** score over a small local table,

fused as `0.5 * mem_norm + 0.5 * base_norm`. Reviewed cases can be appended to
the local memory at deployment time (`update`) with no segmenter or controller
retraining — this is the online adaptation reported in the paper.

## Dependencies

- Runtime (device): `numpy` only.
- Artifact build (workstation, once): `scikit-learn` (to fit + freeze the
  logistic controller).

## Files

| File | Role | Deps |
| --- | --- | --- |
| `memory_qc_edge.py` | `MemoryQCScorer`: standardize → controller + memory → fuse → online update | numpy |
| `benchmark_latency.py` | on-device latency + footprint benchmark | numpy |
| `build_edge_artifacts.py` | freeze controller + calibration + initial memory into `.npz` | sklearn |
| `artifacts/edge_controller.npz` | frozen controller, standardization, z01 ranges, initial memory table | — |
| `artifacts/africa_test_features.npz` | example held-out Africa stream (features + labels) | — |

## Input contract

`score(raw_x)` expects `raw_x` of shape `(n_slices, 29)`, columns in the exact
order of `FEATURE_NAMES` (see `memory_qc_edge.py`). Missing values may be `nan`;
they are median-filled from the calibration statistics baked into the artifact.

## Usage

```python
import numpy as np
from memory_qc_edge import MemoryQCScorer

scorer = MemoryQCScorer("artifacts/edge_controller.npz")     # loads frozen artifact
data = np.load("artifacts/africa_test_features.npz")
risk = scorer.score(data["features"])                        # per-slice risk in [0, 1]
scorer.update(data["features"], data["labels"])              # append reviewed cases
```

Benchmark on device:

```bash
python benchmark_latency.py            # per-slice latency + memory footprint
```

## Reproducibility

The artifact reproduces the paper's cap-5 external-Africa calibration
(peds+gli, seed 42). `score()` matches the reference implementation's fixed
fusion to floating-point precision (max per-slice difference ≈ 3.5e-8).
