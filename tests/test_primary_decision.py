"""Synthetic tests of the fixed-sequence H1 -> H2 decision, the zero-accepted-row rule and the threshold rule and
serialisation of protocol v0.2 (P4). No project data is read; fold 10 is never loaded; nothing is fitted."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval import primary_decision as pdm
from src.eval import selective as sel


def reps(n_nonneg: int, B: int = 10000) -> np.ndarray:
    rng = np.random.default_rng(n_nonneg)
    return rng.permutation(np.r_[-1 - rng.random(B - n_nonneg), rng.random(n_nonneg)])


def test_h2_tested_only_if_h1_rejected():
    called = []
    out = pdm.fixed_sequence(-0.01, reps(100), lambda: called.append(1) or reps(0))
    assert out["H1"]["reject"] and out["H2"]["tested"] and out["H2"]["reject"] and called == [1]
    out = pdm.fixed_sequence(-0.01, reps(600), lambda: called.append(2) or reps(0))
    assert not out["H1"]["reject"] and not out["H2"]["tested"] and called == [1]      # H2 never computed
    out = pdm.fixed_sequence(-0.01, reps(499), reps(500))
    assert out["H1"]["reject"] and out["H2"]["tested"] and not out["H2"]["reject"]


def test_zero_accepted_rows_h1_not_rejected_and_h2_skipped_end_to_end():
    """Synthetic 'fold' only: Gate B accepts no row at its threshold."""
    rng = np.random.default_rng(1)
    pid = np.repeat(np.arange(80), 8)
    d1 = (rng.random(pid.size) < 0.04).astype(float)
    score_b = rng.random(pid.size) + 2.0
    a = sel.accept_at_threshold(score_b, t=1.0)
    point = sel.selective_risk(a, np.ones(pid.size), d1) - d1.mean()
    assert math.isnan(point)
    deltas = []
    for patients, M in sel.patient_multiplicities(pid, 10000, seed=20261030, chunk=2500):
        W = sel.row_weights(pid, patients, M)
        deltas.append(sel.contrast(sel.selective_risk_batch(np.broadcast_to(a, W.shape), W, d1),
                                   sel.selective_risk_batch(np.ones_like(W), W, d1)))
    out = pdm.fixed_sequence(point, np.concatenate(deltas), lambda: pytest.fail("H2 must not be computed"))
    assert out["H1"]["point_defined"] is False and out["H1"]["reject"] is False and out["H2"]["tested"] is False
    assert out["H1"]["n_nonneg"] == 10000 and out["H1"]["upper_bound"] == math.inf


def test_undefined_point_overrides_replicates():
    out = pdm.fixed_sequence(float("nan"), reps(0), reps(0))         # even if every replicate were negative
    assert not out["H1"]["reject"] and not out["H2"]["tested"]


def test_threshold_serialisation_round_trips_exactly():
    rng = np.random.default_rng(3)
    for t in list(rng.random(200)) + [0.0, 1.0, 5e-324, 0.1 + 0.2, np.nextafter(0.8, 1.0)]:
        rec = pdm.threshold_record(t)
        back = pdm.threshold_from_record(rec)
        assert back == t and np.float64(back).tobytes() == np.float64(t).tobytes()
    with pytest.raises(ValueError):
        pdm.threshold_from_record({"repr": "0.5", "hex": (0.25).hex()})
    with pytest.raises(ValueError):
        pdm.threshold_record(float("inf"))


def test_threshold_rule_accepts_scores_equal_to_t_and_uses_observed_candidates():
    s = np.array([0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=float)
    out = sel.select_threshold(s, y, 0.8)
    assert out["t"] == 0.07 and out["coverage"] == pytest.approx(0.8) and out["risk"] == pytest.approx(1 / 8)   # 0.06 < 0.80
    assert out["t"] in set(s)
    assert (sel.accept_at_threshold(s, out["t"]) == (s <= out["t"])).all() and sel.accept_at_threshold(s, 0.02)[1:3].all()
    assert out["n_candidates"] == len(np.unique(s))


def test_threshold_tie_break_order_risk_then_coverage_then_t():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float)    # risk 0 at coverage 0.8; 1/9 and 1/10 above
    out = sel.select_threshold(s, y, 0.8)
    assert out["t"] == 0.8 and out["risk"] == 0                  # lowest risk wins over higher coverage
    y2 = np.zeros(10)                                            # all risks equal -> highest coverage
    assert sel.select_threshold(s, y2, 0.8)["t"] == 1.0
    out = sel.select_threshold(s, y2, 0.8, candidates=np.array([2.0, 1.0, 1.5]))
    assert out["t"] == 1.0                                       # same set and coverage -> smallest t
