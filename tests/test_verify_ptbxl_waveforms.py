"""Tests for src/data/verify_ptbxl_waveforms.py.

A tiny PTB-XL-shaped folder (3 records, both resolutions) is written with the real WFDB
writer, then each kind of problem is planted and must be detected. The last tests read the
report produced from the real C:\\PTB-XL download (skipped if it was not generated).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

from src.data import ptbxl_io as io
from src.data import verify_ptbxl_waveforms as vw
from src.data.ptbxl_waveforms import LEAD_NAMES
from tests.conftest import REPO_ROOT


def _write_record(root: Path, rel: str, fs: int, n: int, rng, flat_lead: int | None = None) -> None:
    folder = root / Path(rel).parent
    folder.mkdir(parents=True, exist_ok=True)
    sig = rng.normal(0, 0.3, (n, 12)).round(3)
    if flat_lead is not None:
        sig[:, flat_lead] = 0.1
    wfdb.wrsamp(Path(rel).name, fs=fs, units=["mV"] * 12, sig_name=list(LEAD_NAMES), p_signal=sig,
                fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(folder))


def _sha_list(root: Path) -> None:
    lines = [f"{vw._sha256(p)} {p.relative_to(root).as_posix()}"
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def mini(tmp_path, config, monkeypatch):
    """Build a clean 3-record dataset and a config pointing at it."""
    monkeypatch.delenv("PTBXL_ROOT", raising=False)
    root, rng = tmp_path / "PTB", np.random.default_rng(0)
    rows = []
    for i in (1, 2, 3):
        lr, hr = f"records100/00000/{i:05d}_lr", f"records500/00000/{i:05d}_hr"
        _write_record(root, lr, 100, 1000, rng)
        _write_record(root, hr, 500, 5000, rng)
        rows.append({"ecg_id": i, "patient_id": float(i), "scp_codes": "{'NORM': 100.0}",
                     "strat_fold": 1, "filename_lr": lr, "filename_hr": hr})
    db = pd.DataFrame(rows)
    db.to_csv(root / "ptbxl_database.csv", index=False)
    (root / "scp_statements.csv").write_text(",description\nNORM,normal ECG\n", encoding="utf-8")
    (root / "RECORDS").write_text("\n".join(list(db.filename_lr) + list(db.filename_hr)) + "\n", encoding="utf-8")
    _sha_list(root)
    ref = tmp_path / "ref"
    ref.mkdir()
    shutil.copy(root / "SHA256SUMS.txt", ref / "SHA256SUMS.txt")
    reports, manifests = tmp_path / "reports", tmp_path / "manifests"
    reports.mkdir()
    manifests.mkdir()
    (reports / "audit_summary.json").write_text(json.dumps({"inputs": [
        {"file": n, "sha256": vw._sha256(root / n)} for n in ("ptbxl_database.csv", "scp_statements.csv")]}))
    frozen = manifests / f"{config['split']['frozen_manifest_name']}.csv"
    frozen.write_text("ecg_id,role\n1,A_fit\n", encoding="utf-8")
    config["dataset"].update(waveform_root=str(root), database_csv=str(root / "ptbxl_database.csv"),
                             checksums_file=str(ref / "SHA256SUMS.txt"))
    config["audit"]["report_dir"] = str(reports)
    config["split"].update(manifest_dir=str(manifests), frozen_manifest_sha256=vw._sha256(frozen))
    return root, config, tmp_path


def _status(summary: dict, name: str) -> str:
    return next(c["status"] for c in summary["checks"] if c["check"] == name)


def test_clean_dataset_passes(mini):
    root, config, repo = mini
    summary = vw.run_verification(config, repo, workers=1)
    assert summary["status"] == "PASS", summary["checks"]
    # 12 waveform files (3 records x 2 resolutions x .hea/.dat) + 2 CSVs + RECORDS
    assert summary["verified"] == {"files_sha256_matched": 15, "records_read_ok_500hz": 3, "records_read_ok_100hz": 3}
    assert (repo / "reports" / "waveform_verification.md").is_file()


def test_missing_dat_file_fails(mini):
    root, config, repo = mini
    (root / "records500/00000/00002_hr.dat").unlink()
    summary = vw.run_verification(config, repo, workers=1)
    for check in ("official_files_present", "metadata_waveform_files_exist", "wfdb_read_500hz"):
        assert _status(summary, check) == "FAIL", check


def test_corrupted_dat_is_caught_by_sha256_and_wfdb_checksum(mini):
    root, config, repo = mini
    dat = root / "records500/00000/00003_hr.dat"
    data = bytearray(dat.read_bytes())
    data[100] ^= 0xFF
    dat.write_bytes(bytes(data))
    assert _status(vw.run_verification(config, repo, workers=1), "official_files_sha256") == "FAIL"
    assert "wfdb_checksum_mismatch" in vw.check_record((str(root), "records500/00000/00003_hr", "hr"))["problems"]


def test_unlisted_file_needs_attention(mini):
    root, config, repo = mini
    (root / "records500/00000/partial.tmp").write_text("x", encoding="utf-8")
    summary = vw.run_verification(config, repo, workers=1)
    assert _status(summary, "unlisted_local_files") == "NEEDS_ATTENTION"
    assert summary["status"] == "NEEDS_ATTENTION"


def test_flat_lead_needs_attention(mini):
    root, config, repo = mini
    _write_record(root, "records500/00000/00001_hr", 500, 5000, np.random.default_rng(1), flat_lead=10)
    _sha_list(root)
    shutil.copy(root / "SHA256SUMS.txt", Path(config["dataset"]["checksums_file"]))
    summary = vw.run_verification(config, repo, workers=1)
    assert _status(summary, "flat_leads") == "NEEDS_ATTENTION"
    assert vw.check_record((str(root), "records500/00000/00001_hr", "hr"))["flat_leads"] == "V5"


def test_records_index_glued_lines_vs_missing_entry(mini):
    root, config, repo = mini
    lines = (root / "RECORDS").read_text(encoding="utf-8").split()
    (root / "RECORDS").write_text("\n".join(lines[:2] + [lines[2] + lines[3]] + lines[4:]) + "\n", encoding="utf-8")
    assert _status(vw.run_verification(config, repo, workers=1), "RECORDS_matches_metadata") == "NEEDS_ATTENTION"
    (root / "RECORDS").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    assert _status(vw.run_verification(config, repo, workers=1), "RECORDS_matches_metadata") == "FAIL"


def test_line_break_explanation_logic():
    assert vw.explained_by_missing_line_breaks({"a/1b/2"}, {"a/1", "b/2"})
    assert not vw.explained_by_missing_line_breaks({"a/1b/2"}, {"a/1", "b/2", "c/3"})
    assert not vw.explained_by_missing_line_breaks(set(), {"a/1"})


def test_changed_frozen_manifest_fails(mini):
    root, config, repo = mini
    config["split"]["frozen_manifest_sha256"] = "0" * 64
    assert _status(vw.run_verification(config, repo, workers=1), "frozen_manifest_unchanged") == "FAIL"


# --------------------------------------------------------------------------- #
# Report from the real download (skipped until generated)
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_report():
    path = REPO_ROOT / "artifacts" / "reports" / "waveform_verification.json"
    if not path.is_file():
        pytest.skip("Run: python -m src.data.verify_ptbxl_waveforms")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_download_has_no_fail_and_full_counts(real_report):
    assert [c["check"] for c in real_report["checks"] if c["status"] == "FAIL"] == []
    exp, ver = real_report["expected"], real_report["verified"]
    assert ver["files_sha256_matched"] == exp["files_listed_in_SHA256SUMS"]
    assert ver["records_read_ok_500hz"] == ver["records_read_ok_100hz"] == exp["records_in_metadata"] == 21799


def test_quality_exclusions_match_flat_lead_records(real_report):
    problems = pd.read_csv(REPO_ROOT / "artifacts" / "reports" / "waveform_problem_records.csv")
    flat_ids = {int(Path(r).name.split("_")[0]) for r in problems.loc[problems["flat_leads"].notna(), "record"]}
    pcfg = io.load_config(REPO_ROOT / "configs" / "preprocessing.yaml")
    assert {e["ecg_id"] for e in pcfg["quality_exclusions"]} == flat_ids
