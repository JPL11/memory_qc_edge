"""Build the frozen edge artifact (.npz) for the memory-QC scorer.

Run ONCE on a workstation (needs scikit-learn). Reproduces the paper's cap-5
external-Africa calibration: peds+gli calibration subjects, seed 42. Freezes the
logistic controller, the feature standardization stats, the two z01 calibration
ranges, and the initial class-balanced memory table into a single .npz that the
numpy-only ``MemoryQCScorer`` loads on the edge device.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jpli/MRI")
import analyze_memory_qc_controller as qc  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slice-glob", default="/home/jpli/MRI/outputs/memory_qc_features_20260618_memory_qc_full_*.slices.csv")
    p.add_argument("--calibration-datasets", default="peds,gli")
    p.add_argument("--test-datasets", default="africa")
    p.add_argument("--calibration-fraction", type=float, default=0.50)
    p.add_argument("--calibration-subject-cap-per-dataset", type=int, default=5)
    p.add_argument("--label-key", default="major_failure_slice")
    p.add_argument("--tumor-relevant-only", action="store_true", default=True)
    p.add_argument("--min-brain-fraction", type=float, default=0.01)
    p.add_argument("--memory-k", type=int, default=15)
    p.add_argument("--memory-batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/home/jpli/MRI/memory_qc_edge/artifacts/edge_controller.npz")
    p.add_argument("--export-test-features", default="/home/jpli/MRI/memory_qc_edge/artifacts/africa_test_features.npz")
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression

    rows = qc.filter_rows(qc.read_csv_rows(sorted(glob.glob(args.slice_glob))), args)
    train_rows, test_rows = qc.split_rows(rows, args)
    train_y = qc.labels(train_rows, args.label_key)
    test_y = qc.labels(test_rows, args.label_key)

    train_raw = qc.matrix(train_rows, qc.BASE_FEATURES)
    test_raw = qc.matrix(test_rows, qc.BASE_FEATURES)

    # Standardization stats (median-fill then z-score), exactly as fill_standardize.
    med = np.nanmedian(train_raw, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(train_raw), train_raw, med)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    train_x = ((filled - mean) / std).astype(np.float32)

    # Frozen logistic controller on base features.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(train_x, train_y)
    coef = clf.coef_.ravel().astype(np.float64)
    intercept = float(clf.intercept_.ravel()[0])
    train_base_prob = clf.predict_proba(train_x)[:, 1]
    base_lo, base_hi = np.nanpercentile(train_base_prob, [1, 99])

    # z01 reference range for the memory density-risk column (leave-one-out on train).
    train_mem = qc.class_balanced_memory_features(
        train_x, train_y, train_x, args.memory_k, args.memory_batch_size, leave_one_out=True
    )[:, 0].astype(np.float64)
    mem_lo, mem_hi = np.nanpercentile(train_mem, [1, 99])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        feature_names=np.array(qc.BASE_FEATURES),
        feat_mean=mean.astype(np.float32),
        feat_std=std.astype(np.float32),
        feat_median=med.astype(np.float32),
        logreg_coef=coef,
        logreg_intercept=np.float64(intercept),
        base_lo=np.float64(base_lo),
        base_hi=np.float64(base_hi),
        mem_lo=np.float64(mem_lo),
        mem_hi=np.float64(mem_hi),
        memory_x=train_x,
        memory_y=train_y.astype(bool),
    )
    # Ship the held-out Africa stream as an example input for the benchmark/demo.
    np.savez(args.export_test_features, features=test_raw.astype(np.float32), labels=test_y.astype(np.int64))

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)")
    print(f"  calibration rows: {len(train_rows)} (pos {int(train_y.sum())}); memory table {train_x.shape}")
    print(f"  base z01 range [{base_lo:.4f}, {base_hi:.4f}]; mem z01 range [{mem_lo:.4f}, {mem_hi:.4f}]")
    print(f"Wrote {args.export_test_features}: {test_raw.shape[0]} test slices (pos {int(test_y.sum())})")


if __name__ == "__main__":
    main()
