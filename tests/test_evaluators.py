"""Tests for Milestone 3a: diagnostic targets, the E1 network, classification metrics, the patient bootstrap
for AUROC, the evaluator config, and (once it exists) the E1 run summary."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from src.data import ptbxl_io as io  # noqa: E402
from src.data.targets import target_matrix  # noqa: E402
from src.eval import classification as cl  # noqa: E402
from src.models.evaluators import ResNet1dWang  # noqa: E402
from src.train.train_l2g import validation_lead_masks  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402


@pytest.fixture(scope="module")
def protocol() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "protocol_v0_1.yaml")


@pytest.fixture(scope="module")
def e1_cfg() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "evaluator_e1_v0_1.yaml")


# --------------------------------------------------------------------------- #
# Targets (protocol Section 1)
# --------------------------------------------------------------------------- #
def test_target_matrix_on_hand_made_rows(small_db, protocol):
    y = target_matrix(small_db, [1, 2, 3, 4, 5, 6, 7], protocol)
    # columns: CLBBB, CRBBB. Likelihood 0 still counts; IRBBB/ILBBB-only rows are negatives.
    expected = [[1, 0], [1, 0], [0, 0], [0, 1], [0, 0], [1, 0], [0, 0]]
    assert y.tolist() == expected
    assert y.dtype == np.uint8


def test_target_counts_match_frozen_role_summary(real_data, protocol):
    manifest = pd.read_csv(REPO_ROOT / "artifacts" / "manifests" / "split_manifest_v1_frozen.csv")
    summary = pd.read_csv(REPO_ROOT / "artifacts" / "manifests" / "split_manifest_v0_draft_role_summary.csv",
                          index_col="role")
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    for role in ("A_fit", "C_dev"):
        rows = manifest[(manifest.role == role) & ~manifest.ecg_id.isin(excluded)].sort_values("ecg_id")
        y = target_matrix(real_data.db, rows.ecg_id.tolist(), protocol)
        for j, code in enumerate(protocol["targets"]["primary_outputs"]):
            assert y[:, j].sum() == summary.loc[role, f"{code}_records"], (role, code)
            assert rows.patient_id[y[:, j] == 1].nunique() == summary.loc[role, f"{code}_patients"], (role, code)


# --------------------------------------------------------------------------- #
# E1 network
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("in_ch", [12, 24])
def test_resnet_shapes_and_gradients(in_ch):
    torch.manual_seed(0)
    model = ResNet1dWang(in_ch, 2)
    x = torch.rand(3, in_ch, 2048)
    out = model(x)
    assert out.shape == (3, 2)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_resnet_is_length_agnostic_and_deterministic_in_eval():
    torch.manual_seed(0)
    model = ResNet1dWang(12, 2).eval()
    x = torch.rand(2, 12, 2048)
    assert torch.equal(model(x), model(x))
    assert model(torch.rand(2, 12, 400)).shape == (2, 2)


# --------------------------------------------------------------------------- #
# Metrics agree with scikit-learn (independent reference implementation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_auroc_and_ap_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(500) < 0.1
    s = np.round(rng.random(500) + 0.8 * y, 2)  # rounding creates ties
    w = rng.integers(0, 4, 500).astype(float)
    assert cl.weighted_auroc(s, y) == pytest.approx(sklearn_metrics.roc_auc_score(y, s), abs=1e-12)
    assert cl.weighted_auroc(s, y, w) == pytest.approx(sklearn_metrics.roc_auc_score(y, s, sample_weight=w), abs=1e-12)
    assert cl.auroc_fn(s, y)(w) == pytest.approx(cl.weighted_auroc(s, y, w), abs=1e-12)
    assert cl.weighted_average_precision(s, y) == pytest.approx(sklearn_metrics.average_precision_score(y, s), abs=1e-12)
    assert cl.weighted_average_precision(s, y, w) == pytest.approx(
        sklearn_metrics.average_precision_score(y, s, sample_weight=w), abs=1e-12)


def test_youden_and_operating_point():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    tau = cl.youden_threshold(s, y)
    assert tau == pytest.approx(0.4)  # sens 1.0, spec 0.75 -> J = 0.75, the unique maximum
    op = cl.operating_point(s, y, tau)
    assert op["sensitivity"] == 1.0 and op["specificity"] == 0.75 and op["ppv"] == pytest.approx(0.8)
    # tie (t = 0.4 and t = 0.7 both give J = 0.75) -> the higher threshold
    assert cl.youden_threshold(s, np.array([0, 0, 0, 1, 0, 1, 1, 1])) == pytest.approx(0.7)


def test_cluster_bootstrap_counts_patients_and_covers_point():
    rng = np.random.default_rng(3)
    pid = np.repeat(np.arange(300), 4)  # 4 rows per patient (like masks per record)
    y = np.repeat(rng.random(300) < 0.15, 4)
    s = np.repeat(rng.random(300), 4) + 0.4 * y + 0.05 * rng.random(1200)  # patient-level score + small row noise
    r = cl.cluster_bootstrap_metric(cl.auroc_fn(s, y), y, pid, replicates=500, seed=1)
    assert r["n_patients"] == 300 and r["n_rows"] == 1200 and r["n_positive_rows"] == 4 * r["n_positive_patients"]
    assert r["ci_low"] < r["point"] < r["ci_high"] and r["half_width"] > 0
    # row-level resampling would ignore the 4x duplication and give a narrower interval
    row = cl.cluster_bootstrap_metric(cl.auroc_fn(s, y), y, np.arange(1200), replicates=500, seed=1)
    assert row["half_width"] < r["half_width"]


def test_cluster_bootstrap_redraws_single_class_replicates():
    pid = np.arange(40)
    y = np.zeros(40, bool)
    y[0] = True  # one positive patient: most replicates lack it and must be redrawn
    s = np.linspace(0, 1, 40)
    r = cl.cluster_bootstrap_metric(cl.auroc_fn(s, y), y, pid, replicates=200, seed=0)
    assert r["redrawn_replicates"] > 0 and r["replicates"] == 200


# --------------------------------------------------------------------------- #
# Config follows the pinned protocol
# --------------------------------------------------------------------------- #
def test_e1_config_conforms_to_protocol(e1_cfg, protocol):
    assert e1_cfg["data"]["train_role"] in protocol["roles"]["fit"]
    assert e1_cfg["data"]["validation_role"] in protocol["roles"]["model_selection"]
    assert e1_cfg["data"]["validation_role"] == protocol["eligibility"]["measured_on"]
    assert 1 <= len(e1_cfg["attempts"]) <= protocol["eligibility"]["max_training_attempts"]
    assert len({a["id"] for a in e1_cfg["attempts"]}) == len(e1_cfg["attempts"])
    assert e1_cfg["operating_threshold"] == protocol["d1"]["evaluator_threshold_rule"]
    assert e1_cfg["input"] == "complete" and e1_cfg["model"]["in_channels"] == 12
    assert e1_cfg["model"]["family"] == "resnet1d_wang"
    text = (REPO_ROOT / "configs" / "evaluator_e1_v0_1.yaml").read_text(encoding="utf-8")
    assert "test_fold10" not in text and "9a_calibration" not in text and "B_gate" not in text


# --------------------------------------------------------------------------- #
# Run summary (skipped until the E1 run exists)
# --------------------------------------------------------------------------- #
def test_e1_run_summary_integrity(e1_cfg, protocol):
    run = REPO_ROOT / e1_cfg["output_dir"] / e1_cfg["run_name"]
    if not (run / "summary.json").is_file():
        pytest.skip("E1 run not finished yet")
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["fold10_loaded"] is False and s["d1_generated"] is False
    assert s["roles_loaded"] == ["A_fit", "C_dev"]
    assert io.sha256_file(REPO_ROOT / s["selected_checkpoint"]) == s["selected_checkpoint_sha256"]
    assert s["reload_matches_logged"] is True
    best = max(a["best_c_dev_macro_auroc"] for a in s["attempts"])
    assert [a for a in s["attempts"] if a["attempt"]["id"] == s["selected_attempt"]][0]["best_c_dev_macro_auroc"] == best
    floor, hw = protocol["eligibility"]["auroc_lower_bound_floor"], protocol["eligibility"]["max_ci_half_width"]
    recomputed = all(v["auroc"]["ci_low"] >= floor and v["auroc"]["half_width"] <= hw for v in s["per_output"].values())
    assert s["eligible"] is recomputed
    pred = pd.read_csv(run / "c_dev_predictions_selected.csv")
    assert len(pred) == s["val_rows"] and pred.patient_id.nunique() == s["val_patients"]
    for code in s["outputs"]:
        assert cl.weighted_auroc(pred[f"prob_{code}"], pred[f"y_{code}"]) == pytest.approx(
            s["per_output"][code]["auroc"]["point"], abs=1e-9)


def test_preflight_evaluator_check_detects_violations(e1_cfg, protocol):
    import copy

    from src.train import preflight
    assert preflight.check_evaluator_config(e1_cfg, protocol) == []
    bad = copy.deepcopy(e1_cfg)
    bad["data"]["validation_role"] = "test_fold10"
    bad["attempts"] = bad["attempts"] + [{"id": 4, "optimizer": "adam", "lr": 0.01, "weight_decay": 0, "seed": 1}]
    bad["model"]["in_channels"] = 24
    problems = preflight.check_evaluator_config(bad, protocol)
    assert len(problems) >= 4


@pytest.mark.parametrize("input_kind", ["complete", "masked"])
def test_train_attempt_runs_on_tiny_synthetic_data(tmp_path, input_kind):
    """Both input paths (E1 complete, E_dir masked) train, log, checkpoint with SHA-256, and predict row-aligned."""
    from src.train import train_evaluator as te
    rng = np.random.default_rng(0)
    n, m, k = 16, 6, 8 if input_kind == "masked" else 1
    y = (rng.random((n, 2)) < 0.4).astype(np.uint8)
    y[0], y[1] = 1, 0
    val_masks = validation_lead_masks(list(range(1, m + 1)), 20261030, 8) if input_kind == "masked" else None
    yv = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.uint8)
    data = {"train_sig": rng.random((n, 12, 2500), dtype=np.float32), "outputs": ["CLBBB", "CRBBB"], "y_train": y,
            "rec_index": np.repeat(np.arange(n), 2), "offsets": np.tile(np.array([0, 452]), n),
            "val_x": rng.random((m, 12, 2048), dtype=np.float32), "val_masks": val_masks,
            "y_val_rows": np.repeat(yv, k, axis=0), "pos_weight": [1.0, 1.0]}
    cfg = {"run_name": "t", "evaluator_id": "T", "input": input_kind,
           "model": {"family": "resnet1d_wang", "in_channels": 24 if input_kind == "masked" else 12, "widths": [4, 4]},
           "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5}}
    att = {"id": 1, "optimizer": "adam", "lr": 1e-3, "weight_decay": 0.0, "seed": 1}
    res = te.train_attempt(att, cfg, data, tmp_path, torch.device("cpu"))
    assert res["epochs_run"] == 2 and np.isfinite(res["best_c_dev_macro_auroc"])
    assert io.sha256_file(tmp_path / "best.pt") == res["best_sha256"]
    model = te.build_model(cfg, 2)
    model.load_state_dict(torch.load(tmp_path / "best.pt", weights_only=False)["model"])
    logits = te.predict(model, data["val_x"], torch.device("cpu"), val_masks)
    assert logits.shape == (m * k, 2)
    assert te.macro_auroc(logits, data["y_val_rows"])[0] == pytest.approx(res["best_c_dev_macro_auroc"], abs=1e-6)
    if input_kind == "masked":  # masked rows of one record differ (different observed leads)
        assert not np.allclose(logits[0], logits[1])


def test_cdev_comparison_integrity():
    path = REPO_ROOT / "artifacts" / "reports" / "m3a_cdev_evaluator_comparison.json"
    if not path.is_file():
        pytest.skip("C_dev comparison not run yet")
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["d1_computed"] is False and s["fold10_loaded"] is False and s["roles_loaded"] == ["C_dev"]
    rows = pd.read_csv(REPO_ROOT / s["rows_file"])
    assert io.sha256_file(REPO_ROOT / s["rows_file"]) == s["rows_file_sha256"]
    assert len(rows) == s["rows"] == s["records"] * s["masks_per_record"]
    assert not any("d1" in c.lower() or "disagree" in c.lower() for c in rows.columns)
    for code, res in s["results"].items():
        for m in ("e1_complete", "e1_recon", "edir"):
            assert cl.weighted_auroc(rows[f"prob_{m}_{code}"], rows[f"y_{code}"]) == pytest.approx(
                res[f"auroc_{m}"]["point"], abs=1e-9)
        # E1 on the complete ECG does not depend on the mask
        assert rows.groupby("ecg_id")[f"prob_e1_complete_{code}"].nunique().max() == 1


def test_m3b_integrity_report_passes():
    path = REPO_ROOT / "artifacts" / "reports" / "m3b_integrity_check.json"
    if not path.is_file():
        pytest.skip("M3b integrity check not run yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
    names = {c["check"] for c in report["checks"]}
    assert {"I3_roles_patient_disjoint", "I4_output_tables_contain_only_C_dev", "I1_G_v0_1_hash_and_read_only"} <= names
