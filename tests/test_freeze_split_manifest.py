"""Tests for src/data/freeze_split_manifest.py (decisions D-009 and D-012).

The frozen manifest must only be created from a verified draft, must be an exact
copy of it, must never be silently overwritten, and must be rejected if edited.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from src.data import freeze_split_manifest as fz
from src.data import make_split_manifest as ms
from src.data import ptbxl_io as io

NOTE = "test approval of D-010/D-011"


def _prepare(tmp_path, db, config, audit_pass=True):
    """Write the database, a matching audit summary, and a draft manifest under tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    db.to_csv(raw / "ptbxl_database.csv", index=False)
    config["dataset"]["database_csv"] = str(raw / "ptbxl_database.csv")
    config["audit"]["report_dir"] = str(tmp_path / "reports")
    config["split"]["manifest_dir"] = str(tmp_path / "manifests")
    if audit_pass:
        reports = tmp_path / "reports"
        reports.mkdir(exist_ok=True)
        sha = io.sha256_file(raw / "ptbxl_database.csv")
        (reports / "audit_summary.json").write_text(json.dumps(
            {"audit_status": "PASS", "inputs": [{"file": "ptbxl_database.csv", "sha256": sha}]}))
    ms.run(config, tmp_path)
    return config


def _frozen(tmp_path, config):
    paths = fz._paths(config, tmp_path)
    return paths["frozen_csv"], paths["frozen_meta"], paths["draft_csv"]


def _make_writable(*paths):
    for path in paths:
        if path.exists():
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def test_freeze_writes_exact_read_only_copy(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    result = fz.freeze(config, tmp_path, NOTE)
    frozen_csv, frozen_meta, draft_csv = _frozen(tmp_path, config)
    assert result["status"] == "FROZEN"
    assert io.sha256_file(frozen_csv) == io.sha256_file(draft_csv) == result["manifest_sha256"]
    meta = json.loads(frozen_meta.read_text(encoding="utf-8"))
    assert meta["status"] == "FROZEN" and meta["approval_note"] == NOTE
    assert meta["regenerated_identically"] is True
    assert all(c["status"] == "PASS" for c in meta["checks_at_freeze"])
    assert meta["diagnostic_target"]["primary_outputs"] == ["CLBBB", "CRBBB"]
    assert not os.access(frozen_csv, os.W_OK) and not os.access(frozen_meta, os.W_OK)
    _make_writable(frozen_csv, frozen_meta)


def test_freeze_twice_is_idempotent(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    first = fz.freeze(config, tmp_path, NOTE)
    second = fz.freeze(config, tmp_path, NOTE)
    assert second["status"] == "ALREADY_FROZEN"
    assert second["manifest_sha256"] == first["manifest_sha256"]
    _make_writable(*_frozen(tmp_path, config)[:2])


def test_freeze_requires_approval_note(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    with pytest.raises(fz.FreezeError, match="approval"):
        fz.freeze(config, tmp_path, "   ")


def test_freeze_refuses_blocked_draft(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config, audit_pass=False)
    with pytest.raises(fz.FreezeError, match="DRAFT_CHECKS_PASSED"):
        fz.freeze(config, tmp_path, NOTE)
    assert not _frozen(tmp_path, config)[0].exists()


def test_freeze_refuses_edited_draft(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    draft_csv = _frozen(tmp_path, config)[2]
    text = draft_csv.read_text(encoding="utf-8")
    draft_csv.write_text(text.replace(",A_fit,", ",B_gate,", 1), encoding="utf-8")
    with pytest.raises(fz.FreezeError, match="changed after it was created"):
        fz.freeze(config, tmp_path, NOTE)


def test_freeze_refuses_when_config_no_longer_reproduces_draft(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    config["split"]["seed"] = int(config["split"]["seed"]) + 1
    with pytest.raises(fz.FreezeError, match="different roles"):
        fz.freeze(config, tmp_path, NOTE)


def test_freeze_never_overwrites_a_different_frozen_manifest(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    fz.freeze(config, tmp_path, NOTE)
    config["split"]["seed"] = int(config["split"]["seed"]) + 1
    ms.run(config, tmp_path)  # new draft with a different split
    with pytest.raises(fz.FreezeError, match="already exists"):
        fz.freeze(config, tmp_path, NOTE)
    _make_writable(*_frozen(tmp_path, config)[:2])


def test_load_frozen_manifest_verifies_integrity(tmp_path, synthetic_db, config):
    config = _prepare(tmp_path, synthetic_db, config)
    sha = fz.freeze(config, tmp_path, NOTE)["manifest_sha256"]
    config["split"]["frozen_manifest_sha256"] = sha
    loaded = fz.load_frozen_manifest(config, tmp_path)
    assert len(loaded) == len(synthetic_db)

    config["split"]["frozen_manifest_sha256"] = "0" * 64
    with pytest.raises(fz.FreezeError, match="pinned"):
        fz.load_frozen_manifest(config, tmp_path)

    config["split"]["frozen_manifest_sha256"] = sha
    frozen_csv, frozen_meta, _ = _frozen(tmp_path, config)
    _make_writable(frozen_csv, frozen_meta)
    frozen_csv.write_text(frozen_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(fz.FreezeError, match="modified after freezing"):
        fz.load_frozen_manifest(config, tmp_path)


def test_freeze_missing_draft_gives_exit_code_2(tmp_path, config, capsys):
    (tmp_path / "configs").mkdir()
    from tests.conftest import CONFIG_PATH
    (tmp_path / "configs" / "data_audit.yaml").write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    assert fz.main(["--root", str(tmp_path), "--approval-note", NOTE]) == 2
    assert "Next action" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Real frozen manifest (skipped until it exists)
# --------------------------------------------------------------------------- #
def test_real_frozen_manifest_is_intact_and_valid(real_data):
    from tests.conftest import REPO_ROOT
    try:
        frozen = fz.load_frozen_manifest(real_data.config, REPO_ROOT)
    except io.MissingInputError:
        pytest.skip("Frozen manifest not created yet. Run: python -m src.data.freeze_split_manifest")
    assert real_data.config["split"]["frozen_manifest_sha256"], "pin the frozen SHA-256 in the config"
    checks = ms.check_manifest(frozen, real_data.db, real_data.config)
    assert [c["check"] for c in checks if c["status"] != "PASS"] == []
    assert (frozen.loc[frozen["strat_fold"] == 10, "role"] == "test_fold10").all()
    assert (frozen.groupby("patient_id")["role"].nunique() == 1).all()
