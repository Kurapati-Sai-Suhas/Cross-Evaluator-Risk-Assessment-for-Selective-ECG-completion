"""Synthetic tests of the M5 evaluator (src/eval/m5.py): fixed sequence, zero-accepted handling, gate_A identity, row checks,
denominators and determinism. Small B; no real data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.eval import m5

PROTOCOL = {"seed": 20261030, "primary_analysis": {"alpha_one_sided": 0.05}, "uncertainty": {"replicates": 10000},
            "threshold": {"secondary_coverages": [0.6, 0.7, 0.9]}}


def table(n_pat=120, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        for k in range(8):
            mask = rng.integers(0, 6)
            risk = 0.02 + 0.03 * (mask % 3)
            noise = rng.random()
            d1 = int(rng.random() < (risk + (0.25 if informative and noise > 0.9 else 0.0)))
            zf = rng.integers(0, 2, size=2)
            zr = zf.copy()
            if d1:
                zr[0] = 1 - zf[0]
            rows.append({"row_id": f"r:{p}:{k}", "patient_id": p, "mask_id": int(mask), "n_observed": int(1 + mask % 3),
                         "D1": int((zf != zr).any()), "y_CLBBB": int(zf[0]), "y_CRBBB": int(zf[1]),
                         "z_full_CLBBB": int(zf[0]), "z_rec_CLBBB": int(zr[0]), "z_full_CRBBB": int(zf[1]), "z_rec_CRBBB": int(zr[1]),
                         "D1_miss_CLBBB": int(zf[0] and not zr[0]), "D1_false_CLBBB": int(zr[0] and not zf[0]),
                         "D1_miss_CRBBB": int(zf[1] and not zr[1]), "D1_false_CRBBB": int(zr[1] and not zf[1]),
                         "gate_A": 0.02 + 0.03 * (mask % 3), "gate_B": 0.02 + 0.03 * (mask % 3) + (0.3 * noise if informative else 0.01 * rng.random()),
                         "M3_signal_quality_only": rng.random(), "M2_uncertainty_only": rng.random()})
    return pd.DataFrame(rows)


def thresholds(t):
    return {m: float(np.quantile(t[m], 0.8)) for m in m5.GATES}


def test_runs_and_reports_everything_with_one_replicate_stream():
    t = table()
    r = m5.evaluate(t, thresholds(t), PROTOCOL, replicates=60, chunk=25)
    assert r["replicates"] == 60 and r["primary"]["H1"]["replicates"] == 60
    assert set(r["S_A"]) == set(m5.GATES) and r["S_A"]["gate_A"]["C_mean_headline"] == 0.5 == r["S_A"]["gate_A"]["C_pair_sensitivity"]
    assert "SB_den_e1_called_any" in r["S_B_S_C_S_D"]["locked_gate_B"] and "SC_den_any" in r["S_B_S_C_S_D"]["c80_M0_accept_all"]
    same = m5.evaluate(t, thresholds(t), PROTOCOL, replicates=60, chunk=25)             # same chunking: bitwise identical
    assert all(np.array_equal(r["replicate_arrays"][k], same["replicate_arrays"][k]) for k in ("h1", "h2"))
    # other chunking: the same draws (replicate stream), but weight sums may round differently in the last bit
    again = m5.evaluate(t, thresholds(t), PROTOCOL, replicates=60, chunk=60)
    assert all(np.allclose(r["replicate_arrays"][k], again["replicate_arrays"][k], rtol=0, atol=1e-12) for k in ("h1", "h2"))


def test_zero_accepted_rows_mean_h1_not_rejected_and_h2_skipped():
    t = table()
    th = thresholds(t)
    th["gate_B"] = -1.0                                         # gate_B releases nothing
    r = m5.evaluate(t, th, PROTOCOL, replicates=40)
    assert r["primary"]["H1"]["point_defined"] is False and r["primary"]["H1"]["reject"] is False
    assert r["primary"]["H2"]["tested"] is False
    assert np.isinf(r["replicate_arrays"]["h1"]).all()          # undefined replicate = +inf, against rejection
    assert r["point"]["locked"]["gate_B"]["risk"] is None and r["point"]["locked"]["gate_B"]["accepted_rows"] == 0


def test_gate_a_identity_violation_stops_the_analysis():
    t = table()
    t.loc[0, "gate_A"] += 1e-9
    with pytest.raises(RuntimeError):
        m5.evaluate(t, thresholds(t), PROTOCOL, replicates=10)


def test_rows_must_be_unique_and_scores_finite():
    t = table()
    with pytest.raises(ValueError):
        m5.evaluate(pd.concat([t, t.iloc[[0]]]), thresholds(t), PROTOCOL, replicates=10)
    t2 = t.copy()
    t2.loc[3, "gate_B"] = np.nan
    with pytest.raises(ValueError):
        m5.evaluate(t2, thresholds(t), PROTOCOL, replicates=10)


def test_aurc_and_reliability_on_known_inputs():
    assert m5.aurc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1])) == pytest.approx(0.25 * (0 + 0 + 1 / 3 + 0.5))
    rel = m5.reliability(np.array([0.05, 0.95]), np.array([0, 1]), bins=10)
    assert rel["brier"] == pytest.approx(0.0025) and rel["bins"][0]["n"] == 1 and rel["bins"][9]["n"] == 1
