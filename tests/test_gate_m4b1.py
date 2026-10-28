"""Synthetic tests of the M4b-1 gate fitting, calibration and threshold-lock helpers (src/gate/m4b1.py). No real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gate import m4b1
from src.gate import features as gf

P2 = {"gate": {"forbidden_inputs": ["d1", "patient_id", "e1_recon_margin_score"]}}


def test_roles_are_fixed_per_stage_and_fold10_is_refused():
    for stage, role in m4b1.STAGE_ROLE.items():
        m4b1.check_role(role, stage)
        with pytest.raises(RuntimeError):
            m4b1.check_role("test_fold10", stage)
    with pytest.raises(RuntimeError):
        m4b1.check_role("9b_threshold", "fit")                 # fitting only on B_gate
    with pytest.raises(RuntimeError):
        m4b1.check_role("B_gate", "threshold")                 # thresholds only on 9b


def test_method_matrix_is_the_exact_whitelist():
    cols = list(gf.G1_COLUMNS) + list(gf.G2_COLUMNS)
    df = pd.DataFrame(np.zeros((3, len(cols))), columns=cols).assign(row_id=["a", "b", "c"], patient_id=1)
    assert list(m4b1.method_matrix(df, "gate_A", P2).columns) == ["row_id", *gf.G1_COLUMNS]
    assert list(m4b1.method_matrix(df, "gate_B", P2).columns) == ["row_id", *gf.G1_COLUMNS, *gf.G2_COLUMNS]
    assert list(m4b1.method_matrix(df, "M3_signal_quality_only", P2).columns) == ["row_id", *gf.G2_COLUMNS]
    with pytest.raises(ValueError):
        m4b1.method_matrix(df.drop(columns="g1_n_observed"), "gate_A", P2)


def test_folds_keep_every_patient_in_one_fold():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 8)                      # 8 record-mask rows per patient
    y = (rng.random(groups.size) < 0.05).astype(int)
    folds = m4b1.make_folds(y, groups, 5, 20261030)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert pd.Series(folds).groupby(groups).nunique().max() == 1
    assert (folds == m4b1.make_folds(y, groups, 5, 20261030)).all()   # deterministic


def test_parameter_selection_and_tie_break():
    r = [{"min_samples_leaf": m, "max_features": f, "log_loss": ll} for m, f, ll in
         [(10, "sqrt", 0.20), (10, 0.5, 0.19), (50, "sqrt", 0.19 + 5e-10), (50, 0.5, 0.19), (200, "sqrt", 0.25), (200, 0.5, 0.3)]]
    assert m4b1.select_params(r) == r[2]                      # within 1e-9: larger leaf first, then 'sqrt'
    r[2]["log_loss"] = 0.19 + 2e-9
    assert m4b1.select_params(r) == r[3]
    r[3]["log_loss"] = 0.191
    assert m4b1.select_params(r) == r[1]


def test_platt_input_and_calibrated_score():
    raw = np.array([0.0, 1e-9, 0.5, 1.0])
    z = m4b1.platt_input(raw, "gate_B").ravel()
    assert z[0] == z[1] == pytest.approx(np.log(1e-6 / (1 - 1e-6))) and z[2] == 0.0 and z[3] == pytest.approx(-z[0])
    assert (m4b1.platt_input(np.array([-0.3, -0.1]), m4b1.M2).ravel() == [-0.3, -0.1]).all()   # M2: raw score
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    s = rng.random(400)
    y = (rng.random(400) < s).astype(int)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(m4b1.platt_input(s, "gate_B"), y)
    c = m4b1.calibrated(lr, s, "gate_B")
    assert np.all(np.diff(c[np.argsort(s)]) >= 0)             # Platt keeps the ranking (positive slope here)
    assert (m4b1.calibrated(None, s, "gate_B") == s).all()


def test_write_readonly_never_overwrites(tmp_path):
    p = tmp_path / "lock.json"
    m4b1.write_readonly(p, "{}\n")
    with pytest.raises(FileExistsError):
        m4b1.write_readonly(p, "{}\n")
