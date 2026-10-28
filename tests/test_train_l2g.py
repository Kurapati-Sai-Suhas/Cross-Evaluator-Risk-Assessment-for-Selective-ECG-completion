"""Tests for Milestone 2c components: cached preprocessing, GPU-side windowing, metrics, baselines,
patient bootstrap, preflight config checks, data-role guards, and (once it exists) the run summary."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data import preprocess as pp  # noqa: E402
from src.data import ptbxl_io as io  # noqa: E402
from src.eval.stats import patient_cluster_bootstrap  # noqa: E402
from src.models import l2g  # noqa: E402
from src.train import preflight  # noqa: E402
from src.train import recon_metrics as rm  # noqa: E402
from src.train.train_l2g import select_role, validation_lead_masks  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402


@pytest.fixture(scope="module")
def pcfg() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "preprocessing.yaml")


@pytest.fixture(scope="module")
def train_cfg() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "train_l2g_v0_1.yaml")


@pytest.fixture(scope="module")
def protocol() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "protocol_v0_1.yaml")


def _ecg_like(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(5000) / 500
    return np.stack([np.sin(2 * np.pi * (1 + i) * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(5000)
                     for i in range(12)])


# --------------------------------------------------------------------------- #
# Preprocessing split and GPU windowing must equal the tested pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("offset", [0, 452])
def test_filter_resample_then_crop_normalize_equals_preprocess(pcfg, offset):
    x = _ecg_like()
    direct = pp.preprocess(x, 500, pcfg, offset=offset).x
    split = pp.crop_normalize(pp.filter_resample(x, 500, pcfg), pcfg, offset).x
    assert np.array_equal(direct, split)


def test_gpu_crop_normalize_matches_numpy_on_float32_cache(pcfg):
    cached = np.stack([pp.filter_resample(_ecg_like(s), 500, pcfg).astype(np.float32) for s in range(3)])
    cached[1, 5] = 0.3  # a flat lead
    offsets = np.array([0, 452, 100])
    ref = np.stack([pp.crop_normalize(c, pcfg, int(o)).x for c, o in zip(cached, offsets)])
    out = rm.gpu_crop_normalize(torch.from_numpy(cached), torch.from_numpy(offsets)).numpy()
    assert out.shape == (3, 12, 2048)
    assert np.abs(out - ref).max() < 1e-6
    assert (out[1, 5] == 0).all()


# --------------------------------------------------------------------------- #
# Metrics and baselines
# --------------------------------------------------------------------------- #
def test_ssim_per_lead_consistent_with_mean():
    s = l2g.SSIM1D()
    a, b = torch.rand(3, 12, 300), torch.rand(3, 12, 300)
    assert torch.isclose(s.per_lead(a, b).mean(), s(a, b), atol=1e-6)


def test_row_metrics_perfect_reconstruction():
    s = l2g.SSIM1D()
    x = torch.rand(2, 12, 400)
    lm = torch.zeros(2, 12)
    lm[:, :3] = 1
    met = rm.row_metrics(s, x.clone(), x, lm)
    assert torch.allclose(met["ssim_missing"], torch.ones(2), atol=1e-4)
    assert torch.allclose(met["mse_missing"], torch.zeros(2))
    assert torch.allclose(met["pcc_missing"], torch.ones(2), atol=1e-5)


def test_missing_metrics_ignore_observed_leads():
    s = l2g.SSIM1D()
    x = torch.rand(1, 12, 400)
    lm = torch.zeros(1, 12)
    lm[0, 0] = 1
    bad = x.clone()
    bad[0, 1:] = 0.5  # every missing lead wrong, observed lead perfect
    met = rm.row_metrics(s, bad, x, lm)
    assert float(met["ssim_missing"]) < 0.2 and float(met["ssim_full"]) > float(met["ssim_missing"])


def test_baseline_b0_and_b1():
    x = torch.rand(1, 12, 50)
    lm = torch.zeros(1, 12)
    lm[0, [0, 2]] = 1  # leads I and III observed
    x_obs = x * lm.unsqueeze(-1)
    b0 = rm.baseline_constant(x_obs, lm)
    assert torch.equal(b0[0, 0], x[0, 0]) and (b0[0, 1] == 0.5).all()
    corr = torch.eye(12)
    corr[1, 0] = corr[0, 1] = 0.9    # II best explained by I
    corr[3, 2] = corr[2, 3] = -0.95  # AVR best explained by III, negatively
    b1 = rm.baseline_best_correlated(x_obs, lm, corr)
    assert torch.equal(b1[0, 1], x[0, 0])
    assert torch.allclose(b1[0, 3], 1 - x[0, 2])
    assert torch.equal(b1[0, 2], x[0, 2])  # observed untouched


def test_lead_correlation_matrix_signs():
    rng = np.random.default_rng(0)
    base = rng.random((20, 1, 300))
    w = np.concatenate([base, base, 1 - base] + [rng.random((20, 1, 300)) for _ in range(9)], axis=1)
    c = rm.lead_correlation_matrix(w, chunk=7)
    assert c[0, 1] == pytest.approx(1.0) and c[0, 2] == pytest.approx(-1.0) and abs(c[0, 5]) < 0.2


# --------------------------------------------------------------------------- #
# Patient-cluster bootstrap (protocol Section 4)
# --------------------------------------------------------------------------- #
def test_patient_bootstrap_basic_properties():
    rng = np.random.default_rng(1)
    patients = np.repeat(np.arange(200), 5)
    values = np.repeat(rng.normal(0, 1, 200), 5) + rng.normal(0, 0.1, 1000)
    res = patient_cluster_bootstrap(values, patients, replicates=2000, seed=3)
    assert res["ci_low"] < res["point"] < res["ci_high"] and res["n_patients"] == 200
    again = patient_cluster_bootstrap(values, patients, replicates=2000, seed=3)
    assert again == res  # reproducible
    row_level = patient_cluster_bootstrap(values, np.arange(1000), replicates=2000, seed=3)
    assert (res["ci_high"] - res["ci_low"]) > (row_level["ci_high"] - row_level["ci_low"])  # clustering widens the CI
    nan_res = patient_cluster_bootstrap(np.append(values, np.nan), np.append(patients, 999), replicates=500)
    assert nan_res["n_rows_dropped_nan"] == 1


# --------------------------------------------------------------------------- #
# Config conformance and data-role guards
# --------------------------------------------------------------------------- #
def test_training_config_conforms_to_lock_and_protocol(train_cfg, protocol):
    assert preflight.check_training_config(train_cfg, protocol) == []


def test_preflight_detects_a_changed_locked_value(train_cfg, protocol):
    import copy
    bad = copy.deepcopy(train_cfg)
    bad["optimizer"]["generator"]["lr"] = 0.001
    bad["data"]["validation_role"] = "9b_threshold"
    problems = preflight.check_training_config(bad, protocol)
    assert any("optimizer.generator.lr" in p for p in problems)
    assert any("validation" in p for p in problems)


def test_select_role_refuses_folds_9_and_10():
    import pandas as pd
    manifest = pd.DataFrame({"ecg_id": [1, 2], "patient_id": [1, 2], "strat_fold": [3, 10],
                             "role": ["A_fit", "A_fit"]})
    with pytest.raises(RuntimeError):
        select_role(manifest, "A_fit", set())


def test_validation_masks_are_the_protocol_masks(protocol):
    from src.data import missingness as mz
    lm = validation_lead_masks([5, 9], protocol["seed"], 8)
    assert lm.shape == (2, 8, 12)
    expected = [mz.lead_mask(c) for c in mz.evaluation_combinations(9, protocol["seed"], k=8)]
    assert np.array_equal(lm[1], np.stack(expected))


def test_real_train_and_validation_roles_are_disjoint(real_data, train_cfg, protocol):
    from src.data.freeze_split_manifest import load_frozen_manifest
    manifest = load_frozen_manifest(real_data.config, REPO_ROOT)
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    tr = select_role(manifest, train_cfg["data"]["train_role"], excluded)
    va = select_role(manifest, train_cfg["data"]["validation_role"], excluded)
    assert not set(tr["patient_id"]) & set(va["patient_id"])
    assert 12722 not in set(tr["ecg_id"])
    assert len(tr) == 12173 and len(va) == 2620


# --------------------------------------------------------------------------- #
# Run summary of the actual training run (skipped until it exists)
# --------------------------------------------------------------------------- #
def test_run_summary_integrity(train_cfg):
    run_dir = REPO_ROOT / train_cfg["output_dir"] / train_cfg["run_name"]
    path = run_dir / "run_summary.json"
    if not path.is_file():
        pytest.skip("training run not executed yet")
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["roles_loaded"] == ["A_fit", "C_dev"] and s["fold10_loaded"] is False
    assert s["downstream_components_trained"] is False
    if s["status"] == "FINISHED":
        integ = s["checkpoint_integrity"]
        assert integ["pass"] is True
        assert io.sha256_file(run_dir / "best.pt") == integ["sha256_now"]  # checkpoint unchanged since evaluation
        if not integ["sha256_at_save"].startswith("NOT LOGGED"):
            assert integ["sha256_at_save"] == integ["sha256_now"]
        assert integ["checkpoint_epoch"] == s["best_epoch"]


def test_bootstrap_or_undefined_handles_all_nan():
    from src.eval.stats import bootstrap_or_undefined
    out = bootstrap_or_undefined(np.full(5, np.nan), np.arange(5))
    assert out["point"] is None and "undefined" in out["note"]
    ok = bootstrap_or_undefined(np.array([1.0, 2.0, np.nan]), np.array([1, 2, 3]), replicates=200)
    assert ok["point"] == 1.5 and ok["n_rows_dropped_nan"] == 1
