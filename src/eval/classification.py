"""Classification metrics with patient-cluster bootstrap (protocol v0.1 Sections 4-5).

AUROC and average precision (AUPRC) accept row weights, so one bootstrap replicate = the same rows re-weighted by
how many times their patient was drawn. Replicates in which a class has no positive (or no negative) rows are
redrawn, as in Strodthoff et al. (2021). Operating-point metrics use a threshold tau fixed beforehand.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def weighted_auroc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), each row weighted."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, float)
    _, group = np.unique(s, return_inverse=True)
    pos = np.bincount(group, weights=w * y)
    neg = np.bincount(group, weights=w * ~y)
    wp, wn = pos.sum(), neg.sum()
    if wp == 0 or wn == 0:
        return float("nan")
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (wp * wn))


def weighted_average_precision(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Average precision (step-wise area under the precision-recall curve), ties grouped, rows weighted."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, float)
    order = np.argsort(-s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * ~y)
    last = np.r_[np.diff(s) != 0, True]  # last row of each tied score group
    tp, fp = tp[last], fp[last]
    total_pos = tp[-1] if tp.size else 0.0
    if total_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / total_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (ties -> the higher threshold). Decision: score >= tau."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    candidates = np.unique(s)
    best_t, best_j = candidates[-1], -np.inf
    for t in candidates:
        pred = s >= t
        sens = (pred & y).sum() / max(y.sum(), 1)
        spec = (~pred & ~y).sum() / max((~y).sum(), 1)
        if sens + spec - 1 >= best_j:
            best_j, best_t = sens + spec - 1, t
    return float(best_t)


def operating_point(scores, labels, tau: float, weights=None) -> dict[str, float]:
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, float)
    pred = s >= tau
    tp, fn = (w * (pred & y)).sum(), (w * (~pred & y)).sum()
    tn, fp = (w * (~pred & ~y)).sum(), (w * (pred & ~y)).sum()
    return {"sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan")}


def cluster_bootstrap_metric(metric: Callable[[np.ndarray], float], labels: np.ndarray, patient_ids: np.ndarray,
                             replicates: int = 10_000, seed: int = 20261030, level: float = 0.95) -> dict[str, Any]:
    """Patient-cluster percentile CI of ``metric(row_weights)``; replicates lacking a class are redrawn."""
    y = np.asarray(labels).astype(bool)
    _, inverse = np.unique(patient_ids, return_inverse=True)
    n_pat = inverse.max() + 1
    rng = np.random.default_rng(seed)
    stats, redrawn = [], 0
    while len(stats) < replicates:
        counts = np.bincount(rng.integers(0, n_pat, size=n_pat), minlength=n_pat)
        w = counts[inverse].astype(float)
        if (w * y).sum() == 0 or (w * ~y).sum() == 0:
            redrawn += 1
            continue
        stats.append(metric(w))
    alpha = (1 - level) / 2
    point = metric(np.ones(len(y)))
    lo, hi = float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
    return {"point": float(point), "ci_low": lo, "ci_high": hi, "half_width": (hi - lo) / 2, "level": level,
            "replicates": replicates, "redrawn_replicates": redrawn, "n_rows": int(len(y)),
            "n_patients": int(n_pat), "n_positive_rows": int(y.sum()),
            "n_positive_patients": int(len(np.unique(np.asarray(patient_ids)[y]))),
            "method": "patient-cluster bootstrap, percentile; replicates without both classes redrawn"}


def auroc_fn(scores: np.ndarray, labels: np.ndarray) -> Callable[[np.ndarray], float]:
    """weighted_auroc with the score sorting done once (for fast bootstrap replicates)."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    _, group = np.unique(s, return_inverse=True)

    def metric(w: np.ndarray) -> float:
        pos = np.bincount(group, weights=w * y)
        neg = np.bincount(group, weights=w * ~y)
        wp, wn = pos.sum(), neg.sum()
        if wp == 0 or wn == 0:
            return float("nan")
        return float((pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (wp * wn))
    return metric


def average_precision_fn(scores: np.ndarray, labels: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda w: weighted_average_precision(scores, labels, w)
