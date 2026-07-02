"""On-device online-adaptation experiment for the memory-QC edge scorer.

Splits the shipped Africa test stream in arrival order: the first half is
"reviewed" and appended to the local memory (`update`), the second half is
scored before and after that update. Reports ranking quality (AUROC, computed
with numpy only) so the frozen-controller baseline and the online-adapted
scorer can be compared on-device without sklearn.

Run: python experiment_online_update.py
"""
from __future__ import annotations

import argparse

import numpy as np

from benchmark_latency import print_device_info
from memory_qc_edge import MemoryQCScorer


def auroc(y_true, scores) -> float:
    """Rank-based AUROC (Mann-Whitney), numpy-only, ties handled by mid-rank."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes to compute AUROC")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # mid-ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="artifacts/edge_controller.npz")
    p.add_argument("--features", default="artifacts/africa_test_features.npz")
    p.add_argument("--k", type=int, default=15)
    args = p.parse_args()

    print_device_info()

    data = np.load(args.features)
    feats, labels = data["features"], data["labels"].astype(bool)
    n = feats.shape[0]
    half = n // 2
    rev_x, rev_y = feats[:half], labels[:half]      # reviewed at deployment
    hold_x, hold_y = feats[half:], labels[half:]    # never used for update

    scorer = MemoryQCScorer(args.artifact, k=args.k)

    full_auc = auroc(labels, scorer.score(feats))
    before = auroc(hold_y, scorer.score(hold_x))
    scorer.update(rev_x, rev_y)
    after = auroc(hold_y, scorer.score(hold_x))

    print(f"Slices: {n}  |  positives: {int(labels.sum())}  |  k={args.k}")
    print(f"Reviewed (update) half: {half} slices, {int(rev_y.sum())} positives")
    print(f"Held-out half:          {n - half} slices, {int(hold_y.sum())} positives")
    print()
    print(f"AUROC, frozen scorer, full stream:          {full_auc:.4f}")
    print(f"AUROC, frozen scorer, held-out half:        {before:.4f}")
    print(f"AUROC, after online update, held-out half:  {after:.4f}  "
          f"(delta {after - before:+.4f})")


if __name__ == "__main__":
    main()
