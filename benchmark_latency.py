"""On-device latency benchmark for the numpy-only memory-QC scorer.

Times the two edge operations (fixed scoring, online case update) over the
shipped Africa test stream. Reports per-slice latency and the memory-table
footprint. Intended to be run directly on a Jetson-class device; it imports
nothing beyond numpy and the local scorer.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from memory_qc_edge import MemoryQCScorer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="artifacts/edge_controller.npz")
    p.add_argument("--features", default="artifacts/africa_test_features.npz")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    data = np.load(args.features)
    feats, labels = data["features"], data["labels"]
    n = feats.shape[0]

    scorer = MemoryQCScorer(args.artifact, k=args.k)
    start_mb = scorer.memory_mb

    # Fixed scoring.
    scorer.score(feats[:8])  # warm up
    fixed = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        scorer.score(feats)
        fixed.append(time.perf_counter() - t)
    fixed = np.array(fixed)

    # Online update: append the whole stream case-agnostically in one pass.
    upd = []
    for _ in range(args.repeats):
        s = MemoryQCScorer(args.artifact, k=args.k)
        t = time.perf_counter()
        s.score(feats)
        s.update(feats, labels)
        upd.append(time.perf_counter() - t)
    upd = np.array(upd)
    end_mb = MemoryQCScorer(args.artifact, k=args.k)
    end_mb.update(feats, labels)

    print(f"Slices: {n}  |  positives: {int(labels.sum())}  |  k={args.k}")
    print(f"Initial memory table: {start_mb*1024:.1f} KB ({start_mb:.4f} MB)")
    print(f"After online update:  {end_mb.memory_mb*1024:.1f} KB ({end_mb.memory_mb:.4f} MB)")
    print()
    print(f"{'Operation':<26}{'Total s (mean±std)':<26}{'s/slice':<14}{'slices/s':<12}")
    for name, arr in [("fixed score", fixed), ("score + online update", upd)]:
        m, sd = arr.mean(), arr.std()
        print(f"{name:<26}{f'{m:.6f} ± {sd:.6f}':<26}{m/n:<14.8f}{n/m:<12.1f}")


if __name__ == "__main__":
    main()
