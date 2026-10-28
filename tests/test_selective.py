"""Synthetic tests of the protocol v0.2 (P3) selective-release formulas, threshold candidates and decision rule
(src/eval/selective.py). No data file is read; nothing is fitted on project data."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval import selective as sel


def test_exact_coverage_hand_example_with_ties():
    s = np.array([1, 1, 2, 3, 3, 3, 4], dtype=float)
    a = sel.accept_at_coverage(s, np.ones(7), 0.5)            # W* = 3.5: groups {1,1}, {2} whole, then 0.5/3 of {3,3,3}
    assert np.allclose(a, [1, 1, 1, 1 / 6, 1 / 6, 1 / 6, 0]) and math.isclose(a.sum(), 3.5)


@pytest.mark.parametrize("seed", range(5))
def test_exact_coverage_holds_for_any_weights_and_ties(seed):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 15, 300).astype(float)                # heavy ties
    W = rng.integers(0, 4, size=(20, 300)).astype(float)       # bootstrap-like multiplicities, some zero
    A = sel.accept_at_coverage_batch(s, W, 0.8)
    assert np.allclose((A * W).sum(axis=1), 0.8 * W.sum(axis=1))
    assert ((A >= 0) & (A <= 1)).all()
    for b in range(3):                                         # batch == single, ranking redone per replicate
        assert np.allclose(A[b], sel.accept_at_coverage(s, W[b], 0.8))
    for i, j in zip(*np.nonzero(s[:, None] == s[None, :])):    # tied rows share the same fraction
        assert np.allclose(A[:, i], A[:, j])


def test_acceptance_depends_only_on_score_ranks():
    rng = np.random.default_rng(1)
    s, w = rng.normal(size=500), rng.integers(1, 3, 500).astype(float)
    platt = 1 / (1 + np.exp(-(0.7 * s - 2.0)))                 # strictly increasing calibration
    assert np.array_equal(sel.accept_at_coverage(s, w, 0.8), sel.accept_at_coverage(platt, w, 0.8))


def test_risk_undefined_and_contrast_infinite():
    assert math.isnan(sel.selective_risk(np.zeros(4), np.ones(4), np.array([0, 1, 0, 1])))
    d = sel.contrast(np.array([0.02, np.nan, 0.03]), np.array([0.04, 0.04, np.nan]))
    assert d[0] == pytest.approx(-0.02) and np.isinf(d[1]) and np.isinf(d[2]) and (d[1:] > 0).all()


def test_decision_ranks_for_the_protocol_B():
    out = sel.one_sided_decision(np.arange(10000) - 20000.0)
    assert out["k_upper"] == 9501 and out["max_nonneg_for_rejection"] == 499
    lo_hi = sel.two_sided_interval(np.arange(10000, dtype=float))
    assert (lo_hi["k_low"], lo_hi["k_high"]) == (250, 9751)


@pytest.mark.parametrize("n_nonneg, rejected", [(0, True), (499, True), (500, False), (501, False), (10000, False)])
def test_interval_rule_counts(n_nonneg, rejected):
    rng = np.random.default_rng(n_nonneg)
    d = np.r_[-1 - rng.random(10000 - n_nonneg), rng.random(n_nonneg)]
    out = sel.one_sided_decision(rng.permutation(d))
    assert out["reject"] is rejected and out["n_nonneg"] == n_nonneg


def test_interval_rule_equals_the_superseded_p_rule_for_every_count():
    B, k = 10000, 9501
    for n in range(B + 1):                                     # U = delta*_(k) < 0 iff at least k replicates are < 0
        assert ((B - n) >= k) == ((1 + n) / (B + 1) < 0.05), n


def test_infinite_replicates_count_against_rejection_and_nan_is_refused():
    d = np.r_[np.full(9600, -0.01), np.full(400, np.inf)]
    out = sel.one_sided_decision(d)
    assert out["reject"] and out["n_nonneg"] == 400
    out = sel.one_sided_decision(np.r_[np.full(9400, -0.01), np.full(600, np.inf)])
    assert not out["reject"] and out["upper_bound"] == math.inf
    with pytest.raises(ValueError):
        sel.one_sided_decision(np.r_[np.full(9999, -0.01), np.nan])


def test_h1_undefined_point_estimate_is_not_rejected():
    """Gate B accepting no row: R_B undefined; every replicate has zero accepted weight too, so U = +inf."""
    rng = np.random.default_rng(3)
    pid = np.repeat(np.arange(50), 8)
    d1 = (rng.random(400) < 0.05).astype(float)
    a = sel.accept_at_threshold(rng.random(400) + 1.0, t=0.5)  # nothing accepted
    assert a.sum() == 0 and math.isnan(sel.selective_risk(a, np.ones(400), d1))
    deltas = []
    for patients, M in sel.patient_multiplicities(pid, 1000, seed=20261030, chunk=250):
        W = sel.row_weights(pid, patients, M)
        deltas.append(sel.contrast(sel.selective_risk_batch(np.broadcast_to(a, W.shape), W, d1),
                                   sel.selective_risk_batch(np.ones_like(W), W, d1)))
    out = sel.one_sided_decision(np.concatenate(deltas))
    assert not out["reject"] and out["n_nonneg"] == 1000 and out["upper_bound"] == math.inf


def test_descriptive_interval_excludes_undefined_replicates_and_counts_them():
    v = np.r_[np.arange(999, dtype=float), np.full(25, np.nan)]
    out = sel.two_sided_interval(v)
    assert out["replicates_used"] == 999 and out["replicates_undefined"] == 25
    assert (out["k_low"], out["k_high"]) == (25, 975) and out["low"] == 24 and out["high"] == 974
    assert sel.two_sided_interval(np.full(10, np.nan))["note"] == "not estimable"


def test_replicate_stream_equals_the_literal_specification():
    pid = np.array([7, 3, 3, 11, 7, 2, 2, 2, 19])
    blocks = list(sel.patient_multiplicities(pid, 23, seed=20261030, chunk=5))
    patients = blocks[0][0]
    M = np.vstack([m for _, m in blocks])
    draws = np.random.default_rng(20261030).integers(0, len(patients), size=(23, len(patients)))
    expected = np.stack([np.bincount(row, minlength=len(patients)) for row in draws])
    assert list(patients) == [2, 3, 7, 11, 19] and np.array_equal(M, expected) and (M.sum(axis=1) == 5).all()
    W = sel.row_weights(pid, patients, M)
    assert np.array_equal(W[:, 0], M[:, 2]) and np.array_equal(W[:, 1], W[:, 2])   # rows of one patient share a weight


def test_threshold_unique_scores_never_worse_than_the_grid_on_the_selection_rows():
    """Every grid threshold reproduces the acceptance set of some observed-score threshold, so the optimum over observed
    scores is <= the optimum over the grid (same objective, constraint and tie-break)."""
    grid = np.round(np.arange(0, 1.0005, 0.001), 6)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        s = np.clip(rng.lognormal(-3.5, 0.6, 3000), 0, 1)       # calibrated-looking scores, mostly 0.01-0.08
        y = (rng.random(3000) < s * 1.2).astype(float)
        u, g = sel.select_threshold(s, y, 0.8), sel.select_threshold(s, y, 0.8, candidates=grid)
        assert u["coverage"] >= 0.8 and g["coverage"] >= 0.8
        assert u["risk"] <= g["risk"] + 1e-15
        assert u["t"] in set(s)                                # smallest t: an observed 9b score


def test_grid_overshoots_coverage_when_scores_are_concentrated():
    s = np.r_[np.linspace(0.0100, 0.0199, 700), np.linspace(0.0200, 0.02099, 200), np.linspace(0.05, 0.9, 100)]
    y = np.zeros(1000)
    y[[801, 820, 850, 870]] = 1                                # events in the crowded grid step (0.020, 0.021], above rank 801
    y[900::5] = 1                                              # high-score rows are riskier still
    grid = np.round(np.arange(0, 1.0005, 0.001), 6)
    u, g = sel.select_threshold(s, y, 0.8), sel.select_threshold(s, y, 0.8, candidates=grid)
    assert u["coverage"] == pytest.approx(0.801) and u["risk"] == 0     # highest coverage at the lowest risk
    assert g["coverage"] == pytest.approx(0.9) and g["risk"] == pytest.approx(4 / 900)   # t = 0.020 reaches only 0.701


def test_threshold_tie_break_highest_coverage_then_smallest_t():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.zeros(10)                                           # every feasible t has risk 0
    out = sel.select_threshold(s, y, 0.8)
    assert out["t"] == 1.0 and out["coverage"] == 1.0
    out = sel.select_threshold(s, y, 0.8, candidates=np.array([1.5, 0.95, 1.0]))
    assert out["t"] == 1.0                                     # 1.0 and 1.5 give the same set (coverage 1) -> smallest t
