"""Freeze the approved draft split manifest (decisions D-009 and D-012).

Run from the repository root, only after the target definition (D-010) is approved:
    .\\.venv\\Scripts\\python.exe -m src.data.freeze_split_manifest --approval-note "<who approved what, when>"

The frozen copy is the ONLY manifest later milestones may use. Freezing refuses unless:
  1. the draft metadata says DRAFT_CHECKS_PASSED (audit PASS and all split checks PASS);
  2. the draft CSV still has the SHA-256 recorded in its metadata (nobody edited it);
  3. the draft was built from the same ptbxl_database.csv that is on disk now;
  4. re-generating the manifest from the current data + config gives identical roles;
  5. every split check passes again; and
  6. no DIFFERENT frozen manifest already exists (a frozen file is never overwritten;
     a changed split must become a new version, e.g. v2, with its own decision entry).
The frozen CSV and its metadata are marked read-only so they are not changed by accident
(for example by opening the CSV in Excel and saving it).

Exit codes: 0 = frozen (or already frozen identically), 1 = refused, 2 = missing input.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import make_split_manifest as ms
from src.data import ptbxl_io as io
from src.data.audit_ptbxl import environment_info

ROLE_COLUMNS = ["ecg_id", "patient_id", "strat_fold", "role"]


class FreezeError(RuntimeError):
    """Freezing (or loading a frozen manifest) was refused. The message says why."""


def _paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    sp = config["split"]
    out = root / sp["manifest_dir"]
    return {
        "draft_csv": out / f"{sp['manifest_name']}.csv",
        "draft_meta": out / f"{sp['manifest_name']}_metadata.json",
        "frozen_csv": out / f"{sp['frozen_manifest_name']}.csv",
        "frozen_meta": out / f"{sp['frozen_manifest_name']}_metadata.json",
    }


def _make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def freeze(config: dict[str, Any], root: Path, approval_note: str) -> dict[str, Any]:
    """Verify the draft and write the frozen copy. Raises FreezeError on any refusal."""
    if not approval_note or not approval_note.strip():
        raise FreezeError("An --approval-note is required (who approved which decisions, and when).")
    p = _paths(config, root)
    for key in ("draft_csv", "draft_meta"):
        if not p[key].is_file():
            raise io.MissingInputError(
                f"BLOCKER: {p[key]} not found.\n"
                "Next action: run the audit and then src.data.make_split_manifest first."
            )

    draft_meta = json.loads(p["draft_meta"].read_text(encoding="utf-8"))
    if not str(draft_meta.get("status", "")).startswith("DRAFT_CHECKS_PASSED"):
        raise FreezeError(f"Draft status is '{draft_meta.get('status')}', not DRAFT_CHECKS_PASSED.")
    draft_sha = io.sha256_file(p["draft_csv"])
    if draft_sha != draft_meta.get("manifest_sha256"):
        raise FreezeError("The draft CSV was changed after it was created (SHA-256 differs from its metadata). "
                          "Re-run src.data.make_split_manifest.")

    db_path = root / config["dataset"]["database_csv"]
    db = io.load_database(db_path)
    db_sha = io.sha256_file(db_path)
    if db_sha != draft_meta.get("inputs", {}).get("ptbxl_database.csv"):
        raise FreezeError("ptbxl_database.csv on disk is not the file the draft was built from.")
    audit = ms.audit_status(root, config, db_sha)
    if audit != "PASS":
        raise FreezeError(f"Audit status is {audit}; run src.data.audit_ptbxl and fix every FAIL first.")

    codes, failures = io.parse_all_scp_codes(db)
    if len(failures):
        raise FreezeError(f"{len(failures)} scp_codes cells cannot be parsed.")
    draft = pd.read_csv(p["draft_csv"])
    regenerated = ms.make_manifest(db, codes, config)
    try:
        pd.testing.assert_frame_equal(draft[ROLE_COLUMNS].reset_index(drop=True),
                                      regenerated[ROLE_COLUMNS].reset_index(drop=True), check_dtype=False)
    except AssertionError:
        raise FreezeError("Re-generating the manifest from the current data and config gives different roles "
                          "(was the seed or a proportion changed?). Re-run make_split_manifest and review.")
    checks = ms.check_manifest(draft, db, config)
    failed = [c["check"] for c in checks if c["status"] != "PASS"]
    if failed:
        raise FreezeError(f"Split checks failed: {failed}")

    if p["frozen_csv"].exists():
        if io.sha256_file(p["frozen_csv"]) == draft_sha:
            return {"status": "ALREADY_FROZEN", "manifest_sha256": draft_sha, "path": str(p["frozen_csv"])}
        raise FreezeError(f"A different frozen manifest already exists at {p['frozen_csv']}. Frozen files are never "
                          "overwritten; create a new version (new frozen_manifest_name) with a decision-log entry.")

    shutil.copyfile(p["draft_csv"], p["frozen_csv"])
    frozen_sha = io.sha256_file(p["frozen_csv"])
    meta = {
        "manifest": p["frozen_csv"].name,
        "status": "FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approval_note": approval_note.strip(),
        "manifest_sha256": frozen_sha,
        "source_draft": p["draft_csv"].name,
        "source_draft_sha256": draft_sha,
        "seed": draft_meta.get("seed"),
        "role_counts": draft_meta.get("role_counts"),
        "diagnostic_target": config.get("diagnostic_target"),
        "inputs": {"ptbxl_database.csv": db_sha},
        "audit_status": audit,
        "checks_at_freeze": checks,
        "regenerated_identically": True,
        "source_draft_metadata": draft_meta,
        "environment": environment_info(),
        "model_training_started": False,
    }
    p["frozen_meta"].write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _make_read_only(p["frozen_csv"])
    _make_read_only(p["frozen_meta"])
    return {"status": "FROZEN", "manifest_sha256": frozen_sha, "path": str(p["frozen_csv"])}


def load_frozen_manifest(config: dict[str, Any], root: Path) -> pd.DataFrame:
    """The ONLY supported way for later milestones to read the split.

    Verifies that the metadata says FROZEN, that the CSV still has the frozen SHA-256, and
    (if pinned in the config) that it is the exact file recorded in decision D-012.
    """
    p = _paths(config, root)
    if not (p["frozen_csv"].is_file() and p["frozen_meta"].is_file()):
        raise io.MissingInputError(f"BLOCKER: frozen manifest not found at {p['frozen_csv']}.")
    meta = json.loads(p["frozen_meta"].read_text(encoding="utf-8"))
    if meta.get("status") != "FROZEN":
        raise FreezeError(f"{p['frozen_meta'].name} status is '{meta.get('status')}', not FROZEN.")
    actual = io.sha256_file(p["frozen_csv"])
    if actual != meta.get("manifest_sha256"):
        raise FreezeError("The frozen manifest CSV was modified after freezing (SHA-256 mismatch).")
    pinned = config["split"].get("frozen_manifest_sha256")
    if pinned and pinned != actual:
        raise FreezeError("The frozen manifest is not the one pinned in configs/data_audit.yaml "
                          "(split.frozen_manifest_sha256).")
    return pd.read_csv(p["frozen_csv"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the approved draft split manifest.")
    parser.add_argument("--config", default="configs/data_audit.yaml")
    parser.add_argument("--root", default=".", help="Repository root (default: current folder)")
    parser.add_argument("--approval-note", required=True,
                        help='e.g. "D-010 and D-011 approved by the project team on 2026-10-19"')
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        config = io.load_config(root / args.config)
        result = freeze(config, root, args.approval_note)
    except io.MissingInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FreezeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"Split manifest: {result['status']}")
    print(f"  file:    {result['path']}")
    print(f"  sha256:  {result['manifest_sha256']}")
    if result["status"] == "FROZEN":
        print("Next action: copy this sha256 into configs/data_audit.yaml (split.frozen_manifest_sha256) "
              "and into docs/decision_log.md (D-012).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
