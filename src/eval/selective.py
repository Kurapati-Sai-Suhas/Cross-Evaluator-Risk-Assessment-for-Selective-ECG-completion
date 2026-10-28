"""Selective-release estimands, threshold candidates and the single bootstrap decision rule of protocol v0.2 (proposal P3:
configs/protocol_v0_2.yaml blocks uncertainty, threshold, primary_analysis).

Reference implementation of formulas only. It reads no data file, fits nothing and is exercised by synthetic tests; the M5
evaluation pipeline that will call it is NOT implemented. Rows are record-mask rows. ``w`` is the row weight: 1 for a point
estimate, the patient's multiplicity in a bootstrap replicate. Scores are calibrated gate scores (higher = riskier); a row is
accepted (released) when its score is <= t.

Decision rule (P3): a contrast delta = risk(method) - risk(comparator) is rejected (evidence for delta < 0) iff the one-sided
95% percentile upper bound U = delta*_(k), k = ceil(0.95 (B + 1)), is < 0. With B = 10,000 this is k = 9501, i.e. at most 499
replicates with delta* >= 0. No p-value is computed.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterator

import numpy as np


# ------------------------------------------------------------------------------------------------------------------
# Acceptance weights a_i
# ------------------------------------------------------------------------------------------------------------------
def accept_at_threshold(scores: np.ndarray, t: float) -> np.ndarray:
    """a_i = 1[s_i <= t] (locked-threshold operating point)."""
    return (np.asarray(scores, dtype=np.float64) <= t).astype(np.float64)


def accept_at_coverage(scores: np.ndarray, w: np.ndarray, c: float) -> np.ndarray:
    """Exact-coverage acceptance a_i in {1, phi, 0} (protocol primary_analysis.H2.estimand).

    Rows are ordered by score ascending; tie groups (equal float64 score) are accepted whole while the cumulative weight is
    <= W* = c * sum(w); the next tie group is accepted with fraction phi = (W* - W_before) / W_group on each of its rows.
    """
    return accept_at_coverage_batch(scores, np.asarray(w, dtype=np.float64)[None, :], c)[0]


def accept_at_coverage_batch(scores: np.ndarray, W: np.ndarray, c: float) -> np.ndarray:
    """``accept_at_coverage`` for every row of a weight matrix W (replicates x rows); the ranking is redone per replicate."""
    if not 0.0 < c <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {c}")
    s = np.asarray(scores, dtype=np.float64)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    if W.shape[1] != s.size or not np.isfinite(s).all():
        raise ValueError("weights must have one column per row and scores must be finite")
    uniq, inv = np.unique(s, return_inverse=True)            # tie groups, ascending score
    order = np.argsort(inv, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(inv[order]) != 0])
    gw = np.add.reduceat(W[:, order], starts, axis=1)        # replicate x group weight
    cum = np.cumsum(gw, axis=1)
    before = cum - gw
    target = c * W.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(gw > 0, (target - before) / gw, 0.0)
    phi = np.clip(phi, 0.0, 1.0)
    return phi[:, inv]


def selective_risk(a: np.ndarray, w: np.ndarray, d: np.ndarray) -> float:
    """R = sum a w d / sum a w; NaN (undefined) when no weight is accepted."""
    aw = np.asarray(a, dtype=np.float64) * np.asarray(w, dtype=np.float64)
    den = aw.sum()
    return float((aw * np.asarray(d, dtype=np.float64)).sum() / den) if den > 0 else math.nan


def selective_risk_batch(A: np.ndarray, W: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Row-wise selective_risk for acceptance and weight matrices (replicates x rows); NaN where undefined."""
    AW = np.atleast_2d(A) * np.atleast_2d(W)
    den = AW.sum(axis=1)
    num = AW @ np.asarray(d, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def contrast(risk_method: np.ndarray, risk_comparator: np.ndarray) -> np.ndarray:
    """delta* = risk(method) - risk(comparator); +inf when either risk is undefined (uncertainty.undefined_replicate)."""
    a, b = np.asarray(risk_method, dtype=np.float64), np.asarray(risk_comparator, dtype=np.float64)
    out = a - b
    out[~(np.isfinite(a) & np.isfinite(b))] = np.inf
    return out


# ------------------------------------------------------------------------------------------------------------------
# Order statistics, the decision rule and the descriptive interval
# ------------------------------------------------------------------------------------------------------------------
def _rank(q: float, B: int, up: bool) -> int:
    x = Fraction(str(q)) * (B + 1)
    return math.ceil(x) if up else math.floor(x)


def order_statistic(sorted_values: np.ndarray, k: int) -> float:
    """1-based k-th order statistic; -inf below rank 1 and +inf above rank B (the bound is not reached)."""
    B = len(sorted_values)
    if k < 1:
        return -math.inf
    if k > B:
        return math.inf
    return float(sorted_values[k - 1])


def one_sided_decision(deltas: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """U = delta*_(ceil((1 - alpha)(B + 1))); reject the null delta >= 0 iff U < 0. n_nonneg is a diagnostic, not a p-value."""
    d = np.asarray(deltas, dtype=np.float64)
    if np.isnan(d).any():
        raise ValueError("NaN replicate: undefined contrast replicates must be +inf (uncertainty.undefined_replicate)")
    B = d.size
    k = _rank(1 - alpha, B, up=True)
    U = order_statistic(np.sort(d), k)
    return {"upper_bound": U, "k_upper": k, "reject": bool(U < 0), "n_nonneg": int((d >= 0).sum()),
            "max_nonneg_for_rejection": B - k, "replicates": B}


def two_sided_interval(values: np.ndarray, level: float = 0.95) -> dict[str, Any]:
    """Descriptive interval [v_(floor(a(R+1))), v_(ceil((1-a)(R+1)))], a = (1 - level) / 2, over the R defined replicates.

    NaN values are undefined replicates of a single-quantity endpoint: they are excluded and counted. +inf (undefined
    contrast replicate) sorts last and stays in.
    """
    v = np.asarray(values, dtype=np.float64)
    undefined = int(np.isnan(v).sum())
    v = np.sort(v[~np.isnan(v)])
    R = v.size
    if R == 0:
        return {"low": None, "high": None, "replicates_used": 0, "replicates_undefined": undefined, "note": "not estimable"}
    a = (1 - Fraction(str(level))) / 2
    lo, hi = math.floor(a * (R + 1)), math.ceil((1 - a) * (R + 1))
    return {"low": order_statistic(v, lo), "high": order_statistic(v, hi), "k_low": lo, "k_high": hi,
            "replicates_used": R, "replicates_undefined": undefined}


# ------------------------------------------------------------------------------------------------------------------
# Replicate stream (uncertainty.replicate_stream)
# ------------------------------------------------------------------------------------------------------------------
def patient_multiplicities(patient_ids: np.ndarray, replicates: int, seed: int,
                           chunk: int = 500) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (patients, M) blocks: M[b, j] = times patient j (sorted ascending) is drawn in replicate b.

    The draws equal ``default_rng(seed).integers(0, P, size=(replicates, P))`` taken in one call; blocks of ``chunk`` rows are
    drawn consecutively from the same generator (equality with the single call is tested).
    """
    patients = np.unique(np.asarray(patient_ids))
    P = patients.size
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, chunk):
        n = min(chunk, replicates - start)
        draws = rng.integers(0, P, size=(n, P))
        M = np.bincount((draws + (np.arange(n) * P)[:, None]).ravel(), minlength=n * P).reshape(n, P)
        yield patients, M.astype(np.float64)


def row_weights(patient_ids: np.ndarray, patients: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Row weight matrix: every row takes its patient's multiplicity."""
    return M[:, np.searchsorted(patients, np.asarray(patient_ids))]


# ------------------------------------------------------------------------------------------------------------------
# Threshold selection on 9b (threshold block): candidates = distinct observed calibrated scores
# ------------------------------------------------------------------------------------------------------------------
def select_threshold(scores: np.ndarray, risk_target: np.ndarray, c_min: float = 0.80,
                     candidates: np.ndarray | None = None) -> dict[str, Any]:
    """Minimise pooled accepted-case risk subject to coverage >= c_min; tie-break lowest risk, highest coverage, smallest t.

    ``candidates=None`` uses the distinct observed scores (proposal P3); an explicit array (e.g. the former 0.001 grid) is
    evaluated with the same objective for comparison. Coverage and risk use w = 1 on the selection rows.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(risk_target, dtype=np.float64)
    if s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("scores and target must have equal shape and finite scores")
    t = np.unique(s) if candidates is None else np.asarray(candidates, dtype=np.float64)
    srt = np.sort(s)
    n_acc = np.searchsorted(srt, t, side="right")
    cum_y = np.r_[0.0, np.cumsum(y[np.argsort(s, kind="stable")])]
    cov = n_acc / s.size
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = np.where(n_acc > 0, cum_y[n_acc] / np.maximum(n_acc, 1), np.nan)
    feasible = (cov >= c_min) & np.isfinite(risk)
    if not feasible.any():
        raise ValueError("no candidate reaches the minimum coverage")
    idx = np.flatnonzero(feasible)
    best = idx[np.lexsort((t[idx], -cov[idx], risk[idx]))[0]]
    return {"t": float(t[best]), "coverage": float(cov[best]), "risk": float(risk[best]),
            "n_candidates": int(t.size), "n_feasible": int(feasible.sum())}
