"""Tests for src/data/ptbxl_io.py and src/data/audit_ptbxl.py.

Synthetic tests check the logic against answers worked out by hand.
Real-data tests (bottom of file) run the same logic on the downloaded PTB-XL CSVs.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data import audit_ptbxl as audit
from src.data import ptbxl_io as io
from tests.conftest import REPO_ROOT, TARGETS


def _status(checks: list[dict], name: str) -> str:
    return next(c["status"] for c in checks if c["check"] == name)


def _quality(db: pd.DataFrame, statements: pd.DataFrame) -> list[dict]:
    codes, failures = io.parse_all_scp_codes(db)
    return audit.data_quality_checks(db, statements, codes, failures)


# --------------------------------------------------------------------------- #
# scp_codes parsing
# --------------------------------------------------------------------------- #
def test_parse_valid_scp_codes():
    assert io.parse_scp_codes("{'NORM': 100.0, 'SR': 0}") == {"NORM": 100.0, "SR": 0.0}


@pytest.mark.parametrize("bad", [
    "not a dict",
    "['NORM']",
    "{'NORM': 'high'}",
    "{'NORM': True}",
    "{'NORM': nan}",
    "{1: 100.0}",
    "__import__('os').getcwd()",
    None,
])
def test_parse_rejects_malformed_scp_codes(bad):
    with pytest.raises(ValueError):
        io.parse_scp_codes(bad)


def test_parse_all_collects_failures_without_crashing(small_db):
    db = small_db.copy()
    db.loc[0, "scp_codes"] = "broken {"
    codes, failures = io.parse_all_scp_codes(db)
    assert codes.iloc[0] == {}
    assert failures["ecg_id"].tolist() == [1]


# --------------------------------------------------------------------------- #
# Label matrix and counts (expected values derived by hand from small_db)
# --------------------------------------------------------------------------- #
def test_label_matrix_keeps_codes_separate_and_respects_threshold(small_db):
    codes, _ = io.parse_all_scp_codes(small_db)
    any_lik = io.label_matrix(codes, TARGETS, 0)
    assert list(any_lik.columns) == TARGETS  # four separate columns, nothing merged
    assert any_lik["CRBBB"].sum() == 1  # likelihood 0 still means "listed"
    assert io.label_matrix(codes, TARGETS, 50)["CRBBB"].sum() == 0
    assert io.label_matrix(codes, TARGETS, 100)["CLBBB"].sum() == 2


def test_label_counts_records_vs_unique_patients(small_db):
    codes, _ = io.parse_all_scp_codes(small_db)
    counts = audit.label_counts(small_db, codes, TARGETS, [0.0, 100.0], list(range(1, 10)))
    row = counts.query("code == 'CLBBB' and min_likelihood == 0").iloc[0]
    assert row["n_records"] == 3  # ecg 1, 2, 6
    assert row["n_unique_patients"] == 2  # patients 101, 104
    assert row["n_records_decision_folds"] == 2  # ecg 6 is fold 10 -> excluded
    assert row["n_patients_decision_folds"] == 1
    irbbb = counts.query("code == 'IRBBB' and min_likelihood == 0").iloc[0]
    assert (irbbb["n_records"], irbbb["n_unique_patients"]) == (2, 2)


def test_fold_distribution_sums_to_totals(small_db):
    codes, _ = io.parse_all_scp_codes(small_db)
    labels = io.label_matrix(codes, TARGETS, 0)
    folds = audit.fold_distribution(small_db, labels)
    for code in TARGETS:
        assert folds.query("code == @code")["n_records"].sum() == labels[code].sum()
    totals = folds.drop_duplicates("strat_fold")
    assert totals["fold_records"].sum() == len(small_db)
    assert set(totals["strat_fold"]) == set(range(1, 11))


def test_record_and_patient_cooccurrence(small_db):
    codes, _ = io.parse_all_scp_codes(small_db)
    labels = io.label_matrix(codes, TARGETS, 0)
    rec = audit.cooccurrence(labels)
    pat = audit.cooccurrence(audit.patient_labels(small_db, labels))
    assert (rec.values == rec.values.T).all()  # symmetric
    assert rec.loc["CLBBB", "CLBBB"] == 3  # diagonal = record total
    assert rec.loc["CLBBB", "ILBBB"] == 1  # same record (ecg 6)
    assert rec.loc["CRBBB", "IRBBB"] == 0  # never on the same record ...
    assert pat.loc["CRBBB", "IRBBB"] == 1  # ... but the same patient (103)
    assert pat.loc["CLBBB", "CLBBB"] == 2  # diagonal = patient total


def test_within_patient_label_changes(small_db):
    codes, _ = io.parse_all_scp_codes(small_db)
    changes = audit.within_patient_changes(small_db, io.label_matrix(codes, TARGETS, 0)).set_index("code")
    assert changes.loc["CLBBB", "code_in_all_their_records"] == 1  # patient 101
    assert changes.loc["CRBBB", "code_in_some_but_not_all_records"] == 1  # patient 103
    assert changes.loc["IRBBB", "code_in_some_but_not_all_records"] == 1  # patient 103


def test_superclass_cooccurrence(small_db, statements):
    codes, _ = io.parse_all_scp_codes(small_db)
    sup = audit.superclass_cooccurrence(codes, io.label_matrix(codes, TARGETS, 0), statements).set_index("code")
    assert sup.loc["IRBBB", "also_NORM"] == 1  # ecg 3
    assert sup.loc["CRBBB", "also_MI"] == 1  # ecg 4 (AMI)
    assert sup.loc["CLBBB", "also_CD"] == 1  # ecg 6 has ILBBB (class CD)


def test_check_target_codes_reports_missing_code(statements):
    defs = audit.check_target_codes(statements.drop(index="ILBBB"), TARGETS).set_index("code")
    assert not defs.loc["ILBBB", "present_in_scp_statements"]
    assert defs.loc["CLBBB", "diagnostic_subclass"] == "CLBBB"


# --------------------------------------------------------------------------- #
# Data-quality checks: each planted problem must produce a FAIL
# --------------------------------------------------------------------------- #
def test_clean_synthetic_data_has_no_fail(small_db, statements):
    assert [c for c in _quality(small_db, statements) if c["status"] == "FAIL"] == []


@pytest.mark.parametrize("column, value, check", [
    ("ecg_id", 1, "ecg_id_unique"),
    ("patient_id", None, "patient_id_valid"),
    ("patient_id", 101.5, "patient_id_valid"),
    ("patient_id", -3.0, "patient_id_valid"),
    ("strat_fold", 11, "strat_fold_in_1_to_10"),
    ("strat_fold", 2, "each_patient_in_one_fold"),  # patient 101 now in folds 1 and 2
    ("scp_codes", "{'XYZ': 100.0}", "scp_codes_known_in_statements"),
    ("scp_codes", "oops", "scp_codes_parseable"),
    ("scp_codes", "{'CLBBB': 150.0}", "likelihood_in_0_to_100"),
    ("filename_hr", "records500/00000/00001_hr", "filename_hr_present_and_unique"),
])
def test_planted_problem_is_detected(small_db, statements, column, value, check):
    db = small_db.copy()
    db[column] = db[column].astype(object)
    db.loc[1, column] = value  # row 1 is ecg_id 2 / patient 101 / fold 1
    assert _status(_quality(db, statements), check) == "FAIL"


# --------------------------------------------------------------------------- #
# Checksums and missing files
# --------------------------------------------------------------------------- #
def test_verify_checksum_statuses(tmp_path):
    data = tmp_path / "scp_statements.csv"
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(f"{io.sha256_file(data)} scp_statements.csv\n", encoding="utf-8")
    assert io.verify_checksum(data, sums)["status"] == "MATCH"
    sums.write_text(f"{'0' * 64} scp_statements.csv\n", encoding="utf-8")
    assert io.verify_checksum(data, sums)["status"] == "MISMATCH"
    sums.write_text(f"{'0' * 64} other.csv\n", encoding="utf-8")
    assert io.verify_checksum(data, sums)["status"] == "NOT_LISTED"
    assert io.verify_checksum(data, tmp_path / "absent.txt")["status"] == "NOT_CHECKED"


def _write_inputs(tmp_path, db, statements, config):
    raw = tmp_path / "raw"
    raw.mkdir()
    db.to_csv(raw / "ptbxl_database.csv", index=False)
    statements.to_csv(raw / "scp_statements.csv")
    config["dataset"].update(
        database_csv=str(raw / "ptbxl_database.csv"),
        statements_csv=str(raw / "scp_statements.csv"),
        checksums_file=str(raw / "SHA256SUMS.txt"),
    )
    config["audit"]["report_dir"] = str(tmp_path / "reports")
    return config


def test_missing_input_file_gives_blocker_and_exit_code_2(tmp_path, config, capsys):
    config["dataset"]["database_csv"] = str(tmp_path / "nope.csv")
    with pytest.raises(io.MissingInputError, match="BLOCKER"):
        audit.run_audit(config, tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data_audit.yaml").write_text(
        (REPO_ROOT / "configs" / "data_audit.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    assert audit.main(["--root", str(tmp_path)]) == 2
    assert "Next action" in capsys.readouterr().err


def test_run_audit_end_to_end_on_synthetic_files(tmp_path, small_db, statements, config):
    config = _write_inputs(tmp_path, small_db, statements, config)
    summary = audit.run_audit(config, tmp_path)
    assert summary["audit_status"] == "PASS"  # no checksum file -> WARN, not FAIL
    assert summary["model_training_started"] is False
    reports = tmp_path / "reports"
    for name in ["audit_summary.json", "audit_report.md", "bbb_label_counts.csv",
                 "bbb_fold_distribution.csv", "bbb_cooccurrence_records.csv",
                 "bbb_cooccurrence_patients.csv", "data_quality_checks.csv"]:
        assert (reports / name).is_file(), name
    saved = json.loads((reports / "audit_summary.json").read_text(encoding="utf-8"))
    assert saved["n_records"] == len(small_db)
    assert saved["n_unique_patients"] == small_db["patient_id"].nunique()


# --------------------------------------------------------------------------- #
# Real PTB-XL metadata (skipped if not downloaded)
# --------------------------------------------------------------------------- #
def test_real_target_codes_exist_in_statements(real_data):
    defs = audit.check_target_codes(real_data.statements, TARGETS)
    assert defs["present_in_scp_statements"].all(), defs


def test_real_data_quality_has_no_fail(real_data):
    checks = audit.data_quality_checks(real_data.db, real_data.statements, real_data.codes, real_data.failures)
    fails = [c for c in checks if c["status"] == "FAIL"]
    assert fails == [], fails


def test_real_each_patient_in_exactly_one_fold(real_data):
    assert (real_data.db.groupby("patient_id")["strat_fold"].nunique() == 1).all()


def test_real_counts_are_internally_consistent(real_data):
    counts = audit.label_counts(real_data.db, real_data.codes, TARGETS, [0.0, 50.0, 100.0], list(range(1, 10)))
    for code in TARGETS:
        rows = counts.query("code == @code").sort_values("min_likelihood")
        assert rows["n_records"].is_monotonic_decreasing  # stricter threshold -> fewer records
        assert (rows["n_unique_patients"] <= rows["n_records"]).all()
        assert (rows["n_records_decision_folds"] <= rows["n_records"]).all()


def test_saved_report_matches_fresh_computation(real_data):
    """The committed report must reflect the current code and data."""
    saved_path = REPO_ROOT / real_data.config["audit"]["report_dir"] / "bbb_label_counts.csv"
    if not saved_path.is_file():
        pytest.skip("Audit report not generated yet. Run: python -m src.data.audit_ptbxl")
    saved = pd.read_csv(saved_path)
    fresh = audit.label_counts(
        real_data.db, real_data.codes, TARGETS,
        [float(t) for t in real_data.config["targets"]["likelihood_thresholds"]],
        [int(f) for f in real_data.config["audit"]["decision_folds"]],
    )
    cols = ["code", "min_likelihood", "n_records", "n_unique_patients"]
    pd.testing.assert_frame_equal(saved[cols].reset_index(drop=True),
                                  fresh[cols].reset_index(drop=True), check_dtype=False)
