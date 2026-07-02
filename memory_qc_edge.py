"""On-device memory-QC scoring layer (numpy-only).

This is the deployable half of the memory-QC agent. It consumes per-slice
feature rows produced by any frozen segmenter (the 29 base features listed in
``FEATURE_NAMES``) and emits a per-slice failure-review risk score. It performs
the class-balanced memory retrieval, the calibrated fusion with a frozen
logistic controller, and the online memory update described in the paper.

It has NO dependency on torch, monai, sklearn, or the SpikeBridge research
stack: the frozen controller and calibration constants live in an ``.npz``
artifact built once on a workstation (see ``build_edge_artifacts.py``). This is
the code path benchmarked for on-device / Jetson-class latency.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "brain_fraction", "entropy_mean", "entropy_top10", "entropy_max",
    "entropy_brain_mean", "uncertainty_mean", "uncertainty_top10",
    "uncertainty_max", "uncertainty_brain_mean", "temporal_prob_delta_mean",
    "temporal_prob_delta_top10", "temporal_prob_delta_max",
    "temporal_prob_delta_brain_mean", "temporal_label_delta_mean",
    "temporal_label_delta_top10", "temporal_label_delta_max",
    "temporal_label_delta_brain_mean", "patch_temporal_score_mean",
    "patch_temporal_score_top10", "patch_temporal_score_max",
    "pred_wt_area_frac", "pred_tc_area_frac", "pred_et_area_frac",
    "pred_wt_area_jump", "pred_tc_area_jump", "pred_et_area_jump",
    "pred_wt_components", "pred_tc_components", "pred_et_components",
]


def class_balanced_density_risk(memory_x, memory_y, query_x, k=15, batch_size=512):
    """Column-0 class-balanced memory feature (density risk), numpy-only.

    Mirrors ``analyze_memory_qc_controller.class_balanced_memory_features``[:, 0]
    exactly so edge scores match the reported experiments.
    """
    memory_x = np.asarray(memory_x, dtype=np.float32)
    query_x = np.asarray(query_x, dtype=np.float32)
    memory_y = np.asarray(memory_y, dtype=bool)
    if int(memory_y.sum()) == 0 or int((~memory_y).sum()) == 0:
        raise ValueError("memory needs both positive and negative entries")
    pos_mask, neg_mask = memory_y, ~memory_y
    large = np.float32(1e12)
    out = np.zeros(query_x.shape[0], dtype=np.float32)
    for start in range(0, query_x.shape[0], batch_size):
        stop = min(start + batch_size, query_x.shape[0])
        diff = query_x[start:stop, None, :] - memory_x[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        pos_d2 = np.where(pos_mask[None, :], dist2, large)
        neg_d2 = np.where(neg_mask[None, :], dist2, large)
        pos_k = min(max(1, int(k)), max(1, int(pos_mask.sum())))
        neg_k = min(max(1, int(k)), max(1, int(neg_mask.sum())))
        pos_idx = np.argpartition(pos_d2, kth=pos_k - 1, axis=1)[:, :pos_k]
        neg_idx = np.argpartition(neg_d2, kth=neg_k - 1, axis=1)[:, :neg_k]
        pos_d = np.sqrt(np.maximum(np.take_along_axis(pos_d2, pos_idx, axis=1), 0.0))
        neg_d = np.sqrt(np.maximum(np.take_along_axis(neg_d2, neg_idx, axis=1), 0.0))
        pos_density = np.mean(1.0 / (pos_d + 1e-3), axis=1)
        neg_density = np.mean(1.0 / (neg_d + 1e-3), axis=1)
        out[start:stop] = (pos_density / (pos_density + neg_density + 1e-6)).astype(np.float32)
    return out


class MemoryQCScorer:
    """Frozen-controller memory-QC scorer with optional online updates."""

    def __init__(self, artifact_path, k=15, blend=0.5, batch_size=512):
        d = np.load(artifact_path, allow_pickle=False)
        self.feat_mean = d["feat_mean"].astype(np.float32)
        self.feat_std = d["feat_std"].astype(np.float32)
        self.feat_median = d["feat_median"].astype(np.float32)
        self.coef = d["logreg_coef"].astype(np.float64)
        self.intercept = float(d["logreg_intercept"])
        self.base_lo, self.base_hi = float(d["base_lo"]), float(d["base_hi"])
        self.mem_lo, self.mem_hi = float(d["mem_lo"]), float(d["mem_hi"])
        self.memory_x = d["memory_x"].astype(np.float32)
        self.memory_y = d["memory_y"].astype(bool)
        self.k, self.blend, self.batch_size = k, blend, batch_size

    def _standardize(self, raw_x):
        x = np.asarray(raw_x, dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.feat_median[None, :])
        return (x - self.feat_mean[None, :]) / self.feat_std[None, :]

    def _base_norm(self, std_x):
        logit = std_x.astype(np.float64) @ self.coef + self.intercept
        prob = 1.0 / (1.0 + np.exp(-logit))
        return np.clip((prob - self.base_lo) / max(self.base_hi - self.base_lo, 1e-6), 0.0, 1.0)

    def score(self, raw_x):
        """Risk score for a batch of raw feature rows (no memory update)."""
        std_x = self._standardize(raw_x)
        base_norm = self._base_norm(std_x)
        mem = class_balanced_density_risk(self.memory_x, self.memory_y, std_x, self.k, self.batch_size)
        mem_norm = np.clip((mem - self.mem_lo) / max(self.mem_hi - self.mem_lo, 1e-6), 0.0, 1.0)
        return self.blend * mem_norm + (1.0 - self.blend) * base_norm

    def update(self, raw_x, y):
        """Append reviewed slices (raw features + labels) to the local memory."""
        std_x = self._standardize(raw_x)
        self.memory_x = np.concatenate([self.memory_x, std_x.astype(np.float32)], axis=0)
        self.memory_y = np.concatenate([self.memory_y, np.asarray(y, dtype=bool)], axis=0)

    @property
    def memory_mb(self):
        return self.memory_x.nbytes / (1024 ** 2)
