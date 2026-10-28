"""Tests for src/data/make_split_manifest.py.

Acceptance criteria covered here: no patient overlap between roles, fold 10
untouched (every fold-10 record is test, nothing else is test, and fold-10
labels cannot influence the split), determinism under a fixed seed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import make_split_manifest as ms
from src.data import ptbxl_io as io
from tests.conftest import REPO_ROOT


def _build(db: pd.DataFrame, config: dict) -> pd.DataFrame:
    codes, failures = io.parse_all_scp_codes(db)
    assert failures.empty
    return ms.make_manifest(db, codes, config)


def _fails(manifest, db, config) -> list[str]:
    return [c["check"] for c in ms.check_manifest(manifest, db, config) if c["status"] != "PASS"]


# --------------------------------------------------------------------------- #
# allocate_counts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 20, 101, 1000])
def test_allocate_counts_sums_to_n(n):
    counts = ms.allocate_counts(n, {"A": 0.7, "B": 0.15, "C": 0.15}, np.random.default_rng(0))
    assert sum(counts.values()) == n
    assert all(v >= 0 for v in counts.values())


def test_allocate_counts_is_unbiased_for_single_patient_strata():
    rng = np.random.default_rng(1)
    picks = [max(c, key=c.get) for c in
             (ms.allocate_counts(1, {"A": 0.7, "B": 0.15, "C": 0.15}, rng) for _ in range(4000))]
    share_b = picks.count("B") / len(picks)
    assert 0.12 < share_b < 0.18  # expected 0.15


def test_allocate_counts_rejects_bad_proportions():
    with pytest.raises(ValueError):
        ms.allocate_counts(10, {"A": 0.7, "B": 0.2}, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Manifest properties on synthetic data
# --------------------------------------------------------------------------- #
def test_all_split_checks_pass(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    assert _fails(manifest, synthetic_db, config) == []


def test_no_patient_overlap_between_roles(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    assert (manifest.groupby("patient_id")["role"].nunique() == 1).all()
    sets = [set(g["patient_id"]) for _, g in manifest.groupby("role")]
    assert sum(len(s) for s in sets) == len(set().union(*sets))


def test_fold10_untouched(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    in_10 = manifest["strat_fold"] == 10
    assert (manifest.loc[in_10, "role"] == "test_fold10").all()
    assert (manifest.loc[manifest["role"] == "test_fold10", "strat_fold"] == 10).all()
    assert in_10.sum() == (synthetic_db["strat_fold"] == 10).sum()


def test_fold10_labels_cannot_change_the_split(synthetic_db, config):
    """Rewrite every fold-10 label; the manifest must be byte-for-byte identical."""
    altered = synthetic_db.copy()
    altered.loc[altered["strat_fold"] == 10, "scp_codes"] = "{'CLBBB': 100.0, 'ILBBB': 100.0}"
    pd.testing.assert_frame_equal(_build(synthetic_db, config), _build(altered, config))


def test_fold_role_restrictions(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    assert set(manifest.loc[manifest["strat_fold"] == 9, "role"]) <= {"9a_calibration", "9b_threshold"}
    assert set(manifest.loc[manifest["strat_fold"] <= 8, "role"]) <= {"A_fit", "B_gate", "C_dev"}


def test_same_seed_same_manifest_different_seed_different(synthetic_db, config):
    first, second = _build(synthetic_db, config), _build(synthetic_db, config)
    pd.testing.assert_frame_equal(first, second)
    config["split"]["seed"] = int(config["split"]["seed"]) + 1
    assert not first["role"].equals(_build(synthetic_db, config)["role"])


def test_role_proportions_close_to_config(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    dev = manifest[manifest["role_group"] == "development"].drop_duplicates("patient_id")
    share = dev["role"].value_counts(normalize=True)
    for role, target in config["split"]["development_roles"].items():
        assert abs(share.get(role, 0) - target) < 0.05, (role, share.to_dict())


def test_rare_label_spread_across_roles(synthetic_db, config):
    codes, _ = io.parse_all_scp_codes(synthetic_db)
    manifest = _build(synthetic_db, config)
    summary = ms.role_summary(manifest, synthetic_db, codes, config).set_index("role")
    for role in ["A_fit", "B_gate", "C_dev"]:
        assert summary.loc[role, "IRBBB_patients"] >= 1, summary
    assert summary.loc["test_fold10"].filter(like="BBB").isna().all()  # fold 10 not summarised


# --------------------------------------------------------------------------- #
# The checker must catch planted violations
# --------------------------------------------------------------------------- #
def test_checker_detects_patient_in_two_roles(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    multi = manifest[manifest["role"] == "A_fit"].groupby("patient_id").filter(lambda g: len(g) > 1)
    manifest.loc[multi.index[0], "role"] = "B_gate"
    fails = _fails(manifest, synthetic_db, config)
    assert "each_patient_has_one_role" in fails
    assert "no_patient_overlap_between_roles" in fails


def test_checker_detects_fold10_leak(synthetic_db, config):
    manifest = _build(synthetic_db, config)
    idx = manifest.index[manifest["strat_fold"] == 10][0]
    manifest.loc[idx, "role"] = "A_fit"
    fails = _fails(manifest, synthetic_db, config)
    assert "fold10_records_all_in_test_role" in fails


def test_checker_detects_dropped_record(synthetic_db, config):
    manifest = _build(synthetic_db, config).iloc[1:]
    assert "every_record_assigned_once" in _fails(manifest, synthetic_db, config)


# --------------------------------------------------------------------------- #
# Refusal on unsafe input and audit gating
# --------------------------------------------------------------------------- #
def test_validate_inputs_rejects_patient_in_two_folds(synthetic_db):
    db = synthetic_db.copy()
    pid = db["patient_id"].iloc[0]
    other_fold = 1 if db["strat_fold"].iloc[0] != 1 else 2
    db.loc[len(db)] = {**db.iloc[0].to_dict(), "ecg_id": 999999, "strat_fold": other_fold,
                       "filename_hr": "records500/99999/999999_hr", "filename_lr": "x"}
    assert db.loc[db["patient_id"] == pid, "strat_fold"].nunique() == 2
    assert any("more than one official fold" in p for p in ms.validate_split_inputs(db))


def _setup_root(tmp_path, db, config):
    raw = tmp_path / "raw"
    raw.mkdir()
    db.to_csv(raw / "ptbxl_database.csv", index=False)
    config["dataset"]["database_csv"] = str(raw / "ptbxl_database.csv")
    config["audit"]["report_dir"] = str(tmp_path / "reports")
    config["split"]["manifest_dir"] = str(tmp_path / "manifests")
    return config


def test_run_refuses_and_writes_no_manifest_for_bad_input(tmp_path, synthetic_db, config):
    db = synthetic_db.copy()
    db.loc[0, "patient_id"] = np.nan
    config = _setup_root(tmp_path, db, config)
    meta, summary = ms.run(config, tmp_path)
    assert meta["status"] == "REFUSED" and summary is None
    assert not (tmp_path / "manifests" / "split_manifest_v0_draft.csv").exists()


def test_run_marks_draft_blocked_until_audit_passes(tmp_path, synthetic_db, config):
    config = _setup_root(tmp_path, synthetic_db, config)
    meta, _ = ms.run(config, tmp_path)
    assert meta["audit_status"] == "NOT_RUN"
    assert meta["status"].startswith("DRAFT_BLOCKED")
    reports = tmp_path / "reports"
    reports.mkdir()
    sha = io.sha256_file(config["dataset"]["database_csv"])
    (reports / "audit_summary.json").write_text(json.dumps(
        {"audit_status": "PASS", "inputs": [{"file": "ptbxl_database.csv", "sha256": sha}]}))
    meta, _ = ms.run(config, tmp_path)
    assert meta["status"].startswith("DRAFT_CHECKS_PASSED")
    assert meta["seed"] == config["split"]["seed"]
    assert meta["model_training_started"] is False


# --------------------------------------------------------------------------- #
# Real PTB-XL metadata (skipped if not downloaded)
# --------------------------------------------------------------------------- #
def test_real_manifest_passes_all_checks(real_data):
    manifest = ms.make_manifest(real_data.db, real_data.codes, real_data.config)
    assert _fails(manifest, real_data.db, real_data.config) == []
    assert len(manifest) == len(real_data.db)
    assert (manifest.groupby("patient_id")["role"].nunique() == 1).all()
    assert (manifest.loc[manifest["strat_fold"] == 10, "role"] == "test_fold10").all()


def test_real_saved_manifest_is_reproducible(real_data):
    sp = real_data.config["split"]
    saved_path = REPO_ROOT / sp["manifest_dir"] / f"{sp['manifest_name']}.csv"
    if not saved_path.is_file():
        pytest.skip("Manifest not generated yet. Run: python -m src.data.make_split_manifest")
    saved = pd.read_csv(saved_path)
    fresh = ms.make_manifest(real_data.db, real_data.codes, real_data.config)
    cols = ["ecg_id", "patient_id", "strat_fold", "role"]
    pd.testing.assert_frame_equal(saved[cols], fresh[cols], check_dtype=False)
