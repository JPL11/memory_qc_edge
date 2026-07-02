"""Minimal numpy-only smoke tests for the edge scorer.

Run: python test_memory_qc_edge.py  (needs only numpy + the shipped artifacts).
Exits non-zero on failure so it can gate CI.
"""
from __future__ import annotations

import os

import numpy as np

from memory_qc_edge import MemoryQCScorer, FEATURE_NAMES, class_balanced_density_risk

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts", "edge_controller.npz")
FEATS = os.path.join(HERE, "artifacts", "africa_test_features.npz")


def test_feature_contract():
    assert len(FEATURE_NAMES) == 29, "expected 29 base features"


def test_score_range_and_shape():
    scorer = MemoryQCScorer(ART)
    data = np.load(FEATS)
    x = data["features"]
    risk = scorer.score(x)
    assert risk.shape == (x.shape[0],), "one score per slice"
    assert np.all(np.isfinite(risk)), "scores must be finite"
    assert risk.min() >= 0.0 and risk.max() <= 1.0, "scores must lie in [0, 1]"


def test_nan_inputs_are_handled():
    scorer = MemoryQCScorer(ART)
    x = np.full((4, 29), np.nan, dtype=np.float32)
    risk = scorer.score(x)
    assert np.all(np.isfinite(risk)), "nan rows must be median-filled, not propagate"


def test_online_update_grows_memory():
    scorer = MemoryQCScorer(ART)
    data = np.load(FEATS)
    n0 = scorer.memory_x.shape[0]
    mb0 = scorer.memory_mb
    scorer.update(data["features"], data["labels"])
    assert scorer.memory_x.shape[0] == n0 + data["features"].shape[0]
    assert scorer.memory_mb > mb0


def test_ranking_is_nontrivial():
    # The QC score should rank held-out failures above non-failures well above chance.
    scorer = MemoryQCScorer(ART)
    data = np.load(FEATS)
    risk = scorer.score(data["features"])
    y = data["labels"].astype(bool)
    pos, neg = risk[y].mean(), risk[~y].mean()
    assert pos > neg, "mean risk on failures must exceed non-failures"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failures else 0)
