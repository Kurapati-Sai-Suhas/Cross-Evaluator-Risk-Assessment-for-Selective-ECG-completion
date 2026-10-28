"""Verify the complete local PTB-XL v1.0.3 download (waveforms + metadata).

Run from the repository root (takes several minutes; reads about 3 GB):
    .\\.venv\\Scripts\\python.exe -m src.data.verify_ptbxl_waveforms

Each check is PASS, FAIL or NEEDS_ATTENTION:
  * dataset folder found; checksum list identical to the one fetched from physionet.org (version 1.0.3)
  * metadata CSVs identical to the audited copies and to the official checksums
  * every file in PhysioNet's SHA256SUMS.txt is present, non-empty and has the official SHA-256
  * every .hea/.dat referenced by filename_hr / filename_lr exists; RECORDS agrees with the metadata
  * WFDB reads every record: sampling rate, length, 12 leads in the expected order, mV units,
    per-lead WFDB header checksums, finite values; constant ("flat") leads are reported
  * the frozen split manifest still has its pinned SHA-256
Nothing is downloaded and nothing in the dataset folder is modified.

Exit codes: 0 = no FAIL, 1 = at least one FAIL, 2 = dataset folder not found.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from src.data import ptbxl_io as io
from src.data.audit_ptbxl import _md_table, environment_info
from src.data.ptbxl_waveforms import LEAD_NAMES, RESOLUTIONS, resolve_waveform_root

PASS, FAIL, ATTN, INFO = "PASS", "FAIL", "NEEDS_ATTENTION", "INFO"


def _check(name: str, status: str, value: Any, detail: str) -> dict[str, Any]:
    return {"check": name, "status": status, "value": value, "detail": detail}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_local_files(base: Path) -> dict[str, int]:
    """All files under ``base`` as {posix relative path: size in bytes}."""
    files: dict[str, int] = {}
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            full = Path(dirpath) / name
            files[full.relative_to(base).as_posix()] = full.stat().st_size
    return files


def explained_by_missing_line_breaks(only_in_records: set[str], only_in_metadata: set[str]) -> bool:
    """True if every RECORDS-only line is exactly two metadata entries glued together (a line
    break missing in the published index file) and that accounts for every metadata-only entry."""
    explained: set[str] = set()
    for line in only_in_records:
        splits = [(line[:i], line[i:]) for i in range(1, len(line))
                  if line[:i] in only_in_metadata and line[i:] in only_in_metadata]
        if not splits:
            return False
        explained.update(splits[0])
    return bool(only_in_records) and explained == only_in_metadata


def check_record(task: tuple[str, str, str]) -> dict[str, Any]:
    """Read one record with WFDB and validate it. Runs in a worker process."""
    root, rel, resolution = task
    expected = RESOLUTIONS[resolution]
    out: dict[str, Any] = {"record": rel, "resolution": resolution, "problems": "",
                           "flat_leads": "", "max_abs_mV": np.nan}
    problems: list[str] = []
    try:
        record = wfdb.rdrecord(str(Path(root) / rel), physical=False)
    except Exception as exc:  # any read error makes the record unusable
        out["problems"] = f"unreadable: {type(exc).__name__}: {exc}"
        return out
    if record.fs != expected["fs"]:
        problems.append(f"fs={record.fs}")
    if record.sig_len != expected["n_samples"]:
        problems.append(f"sig_len={record.sig_len}")
    if tuple(record.sig_name) != LEAD_NAMES:
        problems.append(f"leads={record.sig_name}")
    if any(unit != "mV" for unit in record.units):
        problems.append(f"units={record.units}")
    digital = np.asarray(record.d_signal, dtype=np.int64)
    if digital.shape != (expected["n_samples"], len(LEAD_NAMES)):
        problems.append(f"shape={digital.shape}")
    else:
        computed = (digital.sum(axis=0) % 65536).tolist()
        header = [int(c) % 65536 for c in record.checksum]
        if computed != header:
            problems.append("wfdb_checksum_mismatch")
        gain = np.asarray(record.adc_gain, dtype=float)
        if (gain == 0).any():
            problems.append("zero_adc_gain")
        else:
            physical = (digital - np.asarray(record.baseline, dtype=float)) / gain
            if not np.isfinite(physical).all():
                problems.append("non_finite")
            out["max_abs_mV"] = float(np.abs(physical).max())
        flat = [LEAD_NAMES[i] for i in range(digital.shape[1]) if digital[:, i].min() == digital[:, i].max()]
        out["flat_leads"] = ";".join(flat)
    out["problems"] = "; ".join(problems)
    return out


def _read_all(tasks: list[tuple[str, str, str]], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [check_record(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(check_record, tasks, chunksize=64))
    return pd.DataFrame(rows, columns=["record", "resolution", "problems", "flat_leads", "max_abs_mV"])


def run_verification(config: dict[str, Any], repo_root: Path, workers: int = 1,
                     limit: int | None = None) -> dict[str, Any]:
    ds, sp = config["dataset"], config["split"]
    root = resolve_waveform_root(config, repo_root)
    report_dir = repo_root / config["audit"]["report_dir"]
    checks: list[dict[str, Any]] = [_check("dataset_root_found", PASS, str(root), "Folder with records500/ and records100/")]

    # --- version + metadata -------------------------------------------------------------
    local_sums, verified_sums = root / "SHA256SUMS.txt", repo_root / ds["checksums_file"]
    if not local_sums.is_file():
        checks.append(_check("official_checksum_list_present", FAIL, str(local_sums), "SHA256SUMS.txt missing"))
        official: dict[str, str] = {}
    else:
        official = io.read_official_checksums(local_sums)
        if verified_sums.is_file():
            same = _sha256(local_sums) == _sha256(verified_sums)
            checks.append(_check("dataset_version_1_0_3", PASS if same else FAIL, same,
                                 "Local SHA256SUMS.txt is byte-identical to the one fetched from "
                                 "physionet.org/files/ptb-xl/1.0.3/"))
        else:
            checks.append(_check("dataset_version_1_0_3", ATTN, "not compared",
                                 f"Reference copy {verified_sums} not found"))
    audited = {}
    summary_path = report_dir / "audit_summary.json"
    if summary_path.is_file():
        audited = {i["file"]: i["sha256"] for i in json.loads(summary_path.read_text(encoding="utf-8"))["inputs"]}
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        path = root / name
        if not path.is_file():
            checks.append(_check(f"metadata_{name}", FAIL, "missing", str(path)))
            continue
        sha = _sha256(path)
        ok = sha == official.get(name) and (not audited or sha == audited.get(name))
        checks.append(_check(f"metadata_{name}", PASS if ok else FAIL, sha[:16] + "...",
                             "Matches official SHA-256 and the copy audited in Milestone 1"))

    # --- files listed by PhysioNet ------------------------------------------------------
    local = list_local_files(root)
    db = io.load_database(repo_root / ds["database_csv"])
    if limit:
        db = db.head(limit)
        wanted = {f"{db_row}{ext}" for col in ("filename_hr", "filename_lr")
                  for db_row in db[col] for ext in (".hea", ".dat")}
        official = {k: v for k, v in official.items() if k in wanted}
    listed = sorted(official)
    missing = [f for f in listed if f not in local]
    empty = [f for f in listed if local.get(f, 1) == 0]
    present = [f for f in listed if f in local]
    with ThreadPoolExecutor(max_workers=max(4, workers)) as pool:
        hashes = dict(zip(present, pool.map(lambda f: _sha256(root / f), present)))
    mismatched = [f for f in present if hashes[f] != official[f]]
    unlisted = sorted(set(local) - set(official) - {"SHA256SUMS.txt"}) if not limit else []
    checks += [
        _check("official_files_present", PASS if not missing else FAIL,
               {"listed": len(listed), "present": len(present), "missing": len(missing)},
               "Files listed in PhysioNet SHA256SUMS.txt that exist locally"),
        _check("official_files_nonempty", PASS if not empty else FAIL, len(empty), "Listed files with size 0"),
        _check("official_files_sha256", PASS if not mismatched else FAIL,
               {"verified": len(present) - len(mismatched), "mismatched": len(mismatched)},
               "SHA-256 of every present listed file compared with the official value"),
        _check("unlisted_local_files", PASS if not unlisted else ATTN, unlisted[:20],
               "Files in the dataset folder that PhysioNet does not list (e.g. partial downloads)"),
    ]

    # --- metadata <-> files -------------------------------------------------------------
    missing_refs = {res: int(sum(f"{p}{ext}" not in local for p in db[spec["column"]] for ext in (".hea", ".dat")))
                    for res, spec in RESOLUTIONS.items()}
    checks.append(_check("metadata_waveform_files_exist", PASS if not any(missing_refs.values()) else FAIL,
                         {"records": len(db), "missing_files_hr": missing_refs["hr"], "missing_files_lr": missing_refs["lr"]},
                         ".hea and .dat for every filename_hr and filename_lr in ptbxl_database.csv"))
    records_file = root / "RECORDS"
    if records_file.is_file() and not limit:
        listed_records = {line.strip() for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        meta_records = set(db["filename_hr"]) | set(db["filename_lr"])
        only_rec, only_meta = listed_records - meta_records, meta_records - listed_records
        if not only_rec and not only_meta:
            status, detail = PASS, "PhysioNet RECORDS index lists exactly the metadata records"
        elif explained_by_missing_line_breaks(only_rec, only_meta):
            status = ATTN
            detail = ("The official RECORDS file (its SHA-256 matches PhysioNet) is missing a line break, so two "
                      "entries are glued together. Upstream formatting defect; the pipeline uses the metadata "
                      "filename columns, whose files are all present and checksum-verified")
        else:
            status, detail = FAIL, "PhysioNet RECORDS index disagrees with filename_hr/filename_lr"
        checks.append(_check("RECORDS_matches_metadata", status,
                             {"RECORDS": len(listed_records), "metadata": len(meta_records),
                              "only_in_RECORDS": sorted(only_rec), "only_in_metadata": sorted(only_meta)},
                             detail))

    # --- WFDB read of every record ------------------------------------------------------
    tasks = [(str(root), p, res) for res, spec in RESOLUTIONS.items() for p in db[spec["column"]]
             if f"{p}.hea" in local and f"{p}.dat" in local]
    per_record = _read_all(tasks, workers)
    bad = per_record[per_record["problems"] != ""]
    flat = per_record[per_record["flat_leads"] != ""]
    for res, spec in RESOLUTIONS.items():
        sub = per_record[per_record["resolution"] == res]
        n_bad = int((sub["problems"] != "").sum())
        checks.append(_check(f"wfdb_read_{spec['fs']}hz", PASS if n_bad == 0 and len(sub) == len(db) else FAIL,
                             {"read_ok": len(sub) - n_bad, "problems": n_bad, "expected": len(db)},
                             f"fs={spec['fs']} Hz, {spec['n_samples']} samples, leads {','.join(LEAD_NAMES)}, mV, "
                             "WFDB header checksums, finite values"))
    flat_records = {res: int((flat["resolution"] == res).sum()) for res in RESOLUTIONS}
    checks.append(_check("flat_leads", PASS if flat.empty else ATTN, flat_records,
                         "Records with at least one constant lead. Per-lead min-max normalisation is undefined "
                         "for such a lead (see D-014)"))

    # --- frozen split ---------------------------------------------------------------------
    frozen = repo_root / sp["manifest_dir"] / f"{sp['frozen_manifest_name']}.csv"
    frozen_sha = _sha256(frozen) if frozen.is_file() else "missing"
    checks.append(_check("frozen_manifest_unchanged", PASS if frozen_sha == sp.get("frozen_manifest_sha256") else FAIL,
                         frozen_sha, "SHA-256 of split_manifest_v1_frozen.csv equals the value pinned in the config (D-012)"))

    statuses = [c["status"] for c in checks]
    summary = {
        "status": "FAIL" if FAIL in statuses else ("NEEDS_ATTENTION" if ATTN in statuses else "PASS"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "limited_to_first_records": limit,
        "expected": {"records_in_metadata": len(db), "waveform_files_expected": 4 * len(db),
                     "files_listed_in_SHA256SUMS": len(listed)},
        "verified": {"files_sha256_matched": len(present) - len(mismatched),
                     "records_read_ok_500hz": int(((per_record["resolution"] == "hr") & (per_record["problems"] == "")).sum()),
                     "records_read_ok_100hz": int(((per_record["resolution"] == "lr") & (per_record["problems"] == "")).sum())},
        "max_abs_mV_percentiles": {q: float(np.nanpercentile(per_record["max_abs_mV"], q)) for q in (50, 99, 100)}
        if len(per_record) else {},
        "checks": checks,
        "environment": {**environment_info(), "wfdb": wfdb.__version__},
        "model_training_started": False,
    }
    if not limit:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "waveform_verification.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        problem_rows = pd.concat([bad, flat]).drop_duplicates(subset=["record"])
        problem_rows.to_csv(report_dir / "waveform_problem_records.csv", index=False, encoding="utf-8")
        pd.DataFrame({"file": mismatched + missing,
                      "issue": ["sha256_mismatch"] * len(mismatched) + ["missing"] * len(missing)}).to_csv(
            report_dir / "waveform_problem_files.csv", index=False, encoding="utf-8")
        _write_markdown(report_dir / "waveform_verification.md", summary, problem_rows)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any], problems: pd.DataFrame) -> None:
    lines = [
        "# PTB-XL waveform download verification",
        "",
        f"Generated by `src/data/verify_ptbxl_waveforms.py` at {summary['created_utc']} (UTC). "
        "All numbers are computed from the files; nothing is typed in by hand.",
        "",
        f"**Overall: {summary['status']}**. Dataset folder: `{summary['dataset_root']}`. No model was trained.",
        "",
        "## Expected vs verified",
        _md_table(pd.DataFrame([{**{f"expected_{k}": v for k, v in summary["expected"].items()},
                                 **{f"verified_{k}": v for k, v in summary["verified"].items()}}]).T.reset_index()
                  .rename(columns={"index": "quantity", 0: "value"})),
        "",
        "## Checks",
        _md_table(pd.DataFrame(summary["checks"]).astype(str)),
        "",
        f"Peak absolute amplitude per record (mV), percentiles 50/99/100: {summary['max_abs_mV_percentiles']}",
        "",
        "## Records with problems or flat leads (first 50)",
        _md_table(problems.head(50)) if len(problems) else "None.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the local PTB-XL v1.0.3 download.")
    parser.add_argument("--config", default="configs/data_audit.yaml")
    parser.add_argument("--root", default=".", help="Repository root (default: current folder)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    repo_root = Path(args.root).resolve()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        config = io.load_config(repo_root / args.config)
        summary = run_verification(config, repo_root, workers=args.workers)
    except io.MissingInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"PTB-XL download verification: {summary['status']}  (folder: {summary['dataset_root']})")
    for chk in summary["checks"]:
        print(f"  [{chk['status']}] {chk['check']}: {chk['value']}")
    print(f"Reports written to: {repo_root / config['audit']['report_dir']}")
    return 1 if summary["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
