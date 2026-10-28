"""M4b-0: extract G1/G2 gate features for B_gate, 9a_calibration and 9b_threshold. No D1 join, no model fitting.

    .\\.venv\\Scripts\\python.exe -m src.gate.extract_features --config configs/gate_m4b_v0_2.yaml

Reads only: the D1 ``keys/`` files (row_id, identifiers, observed leads), the ``ecg_id`` and ``filename_hr`` columns of the
PTB-XL database (to locate raw files) and the raw 500 Hz WFDB records. Never reads ``offline_labels/``. Per record, the
per-lead SQIs depend only on that lead's own raw window; each row aggregates the SQIs of its OBSERVED leads only (equal to
``features.row_features``, tested). test_fold10 is refused (protocol §7), A_fit and C_dev are not gate-feature roles here.
Outputs (refused if ``output_dir`` exists): features/<role>_features.csv (row_id + 30 feature columns), extraction_summary.json,
hash_manifest.json; artifacts/reports/m4b0_feature_diagnostics_B_gate.csv (B_gate only: value ranges, no D1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import missingness as mz
from src.data import ptbxl_io as io
from src.data.ptbxl_waveforms import LEAD_NAMES, load_record, resolve_waveform_root
from src.eval.d1 import check_feature_columns
from src.gate import features as gf
from src.utils.provenance import code_tree_sha256, environment

ALLOWED_ROLES = ("B_gate", "9a_calibration", "9b_threshold")
FROZEN = {"artifacts/manifests/split_manifest_v1_frozen.csv": "898d0ac707cf86ec9201eb4d3a9d97a49e8b84767374313b01d1a5996f42b83b",
          "configs/protocol_v0_1.yaml": "55a1ceb6ad4a2078f5b856752bd8c8dd9954ea2168e547760eb6529b8ba32b43",
          "artifacts/runs/l2g_v0_1_run02/best.pt": "d515875a7783be5a54934001e13b182ff4f75a5e35554f0e9590d5a1c602bb3b",
          "artifacts/runs/e1_resnet1d_v0_1_run01/attempt_2/best.pt": "e10c250c78782fe23449fa9dc39dfb7fc981baac00eb690511444386a4357c1f",
          "artifacts/runs/edir_resnet1d_v0_1_run02/attempt_1/best.pt": "4b94af6ac6afe21be10fe589cf4f59831399fb17ca6fb9822c51e847844fcbdd"}


def check_role(role: str) -> None:
    if role == "test_fold10":
        raise RuntimeError("test_fold10 is refused: protocol §7 allows fold-10 rows only after the threshold lock")
    if role not in ALLOWED_ROLES:
        raise RuntimeError(f"{role} is not a gate-feature role in M4b-0 (allowed: {ALLOWED_ROLES})")


def _per_lead_table(task: tuple[str, str]) -> np.ndarray:
    """(12, 5) per-lead SQIs of one raw record (NaN where undefined); runs in a worker process."""
    root, rel = task
    raw = load_record(root, rel, "hr")
    return np.array([[gf.sqi_per_lead(raw[j, gf.WINDOW[0]:gf.WINDOW[1]])[s] for s in gf.SQIS] for j in range(len(LEAD_NAMES))])


def preflight(repo: Path, cfg_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    checks = []

    def ok(name: str, cond: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if cond else "FAIL", "detail": detail})
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        if not cond:
            raise RuntimeError(f"preflight failed: {name}: {detail}")

    for rel, sha in FROZEN.items():
        p = repo / rel
        ok(f"frozen_{p.name}", io.sha256_file(p) == sha and not os.access(p, os.W_OK), rel)
    lock = json.loads((repo / "artifacts/locks/protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    ok("protocol_v0_1_lock", lock["sha256"] == FROZEN["configs/protocol_v0_1.yaml"], "lock matches v0.1")
    p2 = io.load_config(repo / "configs/protocol_v0_2.yaml")
    ok("gate_config_is_the_one_referenced_by_v0_2", io.sha256_file(cfg_path) == p2["gate_preregistration"]["sha256"],
       io.sha256_file(cfg_path))
    v02_lock = repo / "artifacts/locks/protocol_v0_2.lock.json"
    checks.append({"check": "protocol_v0_2_pinned", "status": "INFO", "detail": v02_lock.exists()})
    d1_manifest = repo / Path(cfg["d1_keys_dir"]).parent / "hash_manifest.json"
    ok("d1_hash_manifest", io.sha256_file(d1_manifest) == cfg["d1_hash_manifest_sha256"], cfg["d1_hash_manifest_sha256"])
    ok("no_threshold_lock", not (repo / cfg["threshold"]["lock_file"]).exists(), "fold 10 stays closed")
    ok("roles_allowed", all(r in ALLOWED_ROLES for r in cfg["roles"]["feature_extraction"]), cfg["roles"]["feature_extraction"])
    ok("output_dir_absent", not (repo / cfg["output_dir"]).exists(), cfg["output_dir"])
    gf.check_config(cfg)
    ok("formulas_match_config", True, "features.check_config")
    return {"checks": checks, "protocol_v0_2": p2,
            "d1_manifest": json.loads(d1_manifest.read_text(encoding="utf-8"))}


def extract_role(repo: Path, cfg: dict[str, Any], role: str, d1_manifest: dict[str, Any], db_files: pd.Series,
                 root: Path, seed: int, k: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    check_role(role)
    key_path = gf.assert_not_offline(repo / cfg["d1_keys_dir"] / f"{role}_rows.csv")
    rel = key_path.relative_to(repo).as_posix()
    if io.sha256_file(key_path) != d1_manifest["files"][rel]:
        raise RuntimeError(f"{rel} differs from the D1 hash manifest")
    keys = pd.read_csv(key_path)
    if set(keys["role"]) != {role}:
        raise RuntimeError(f"{rel}: unexpected roles {set(keys['role'])}")
    ids = keys["ecg_id"].drop_duplicates().tolist()
    observed = [s.split(";") for s in keys["observed_leads"]]
    for e, mi, obs in zip(keys["ecg_id"], keys["mask_index"], observed):  # masks must be the protocol masks
        if list(mz.evaluation_combinations(int(e), seed, k=k)[int(mi)]) != obs:
            raise RuntimeError(f"{role} ecg {e} mask {mi}: observed leads differ from evaluation_combinations")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        tables = list(pool.map(_per_lead_table, [(str(root), db_files.loc[e]) for e in ids], chunksize=16))
    per_record = dict(zip(ids, tables))
    rows = []
    for e, obs in zip(keys["ecg_id"], observed):
        t = per_record[e]
        per_lead = [dict(zip(gf.SQIS, t[gf.LEAD_INDEX[lead]])) for lead in obs]  # observed leads only, in mask order
        rows.append({**gf.g1_features(obs), **gf.aggregate(per_lead)})
    feats = pd.DataFrame(rows, columns=gf.G1_COLUMNS + gf.G2_COLUMNS)
    feats.insert(0, "row_id", keys["row_id"].to_numpy())
    info = {"rows": int(len(feats)), "records": len(ids), "seconds": round(time.perf_counter() - t0, 1),
            "non_finite_values": int((~np.isfinite(feats[gf.G1_COLUMNS + gf.G2_COLUMNS].to_numpy(float))).sum()),
            "sentinel_counts": {c: int((feats[c] == gf.SENTINEL).sum()) for c in gf.G2_COLUMNS if (feats[c] == gf.SENTINEL).any()},
            "keys_file_sha256": d1_manifest["files"][rel]}
    return feats, info


def diagnostics(feats: pd.DataFrame) -> pd.DataFrame:
    """Value ranges only (B_gate). No D1, no label, no association."""
    rows = []
    for c in gf.G1_COLUMNS + gf.G2_COLUMNS:
        v = feats[c].to_numpy(float)
        valid = v[v != gf.SENTINEL] if c.startswith("g2_") else v
        q = np.quantile(valid, [0, 0.01, 0.5, 0.99, 1]) if valid.size else [np.nan] * 5
        rows.append({"feature": c, "rows": len(v), "n_unique": int(np.unique(v).size), "n_sentinel": int((v == gf.SENTINEL).sum()),
                     "min": q[0], "p01": q[1], "median": q[2], "p99": q[3], "max": q[4], "constant": bool(np.unique(v).size == 1)})
    return pd.DataFrame(rows)


def run(repo: Path, cfg_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = io.load_config(cfg_path)
    tree = code_tree_sha256(repo)["tree_sha256"]
    pf = preflight(repo, cfg_path, cfg)
    p2 = pf["protocol_v0_2"]
    audit_cfg = io.load_config(repo / "configs/data_audit.yaml")
    db_files = pd.read_csv(repo / audit_cfg["dataset"]["database_csv"], usecols=["ecg_id", "filename_hr"]).set_index("ecg_id")["filename_hr"]
    root = resolve_waveform_root(audit_cfg, repo)
    out = repo / cfg["output_dir"]
    (out / "features").mkdir(parents=True)
    seed, k = int(p2["seed"]), int(p2["d1"]["masks_per_record"])
    infos = {}
    for role in cfg["roles"]["feature_extraction"]:
        feats, info = extract_role(repo, cfg, role, pf["d1_manifest"], db_files, root, seed, k)
        bad = check_feature_columns([c for c in feats.columns if c != "row_id"], p2)
        if bad:
            raise RuntimeError(f"forbidden feature columns: {bad}")
        feats.to_csv(out / "features" / f"{role}_features.csv", index=False, float_format="%.17g")
        infos[role] = info
        if role == "B_gate":
            diagnostics(feats).to_csv(repo / "artifacts/reports/m4b0_feature_diagnostics_B_gate.csv", index=False)
        print(f"[{role}] rows {info['rows']} records {info['records']} non-finite {info['non_finite_values']} "
              f"sentinels {sum(info['sentinel_counts'].values())} | {info['seconds']}s", flush=True)
    summary = {"milestone": "M4b-0", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "config": cfg_path.relative_to(repo).as_posix(), "config_sha256": io.sha256_file(cfg_path),
               "protocol_v0_2_sha256": io.sha256_file(repo / "configs/protocol_v0_2.yaml"), "protocol_v0_2_pinned": False,
               "d1_joined": False, "models_fitted": False, "fold10_loaded": False, "offline_labels_read": False,
               "feature_columns": gf.G1_COLUMNS + gf.G2_COLUMNS, "per_role": infos, "preflight": pf["checks"],
               "code_tree_at_start": tree, "code_tree_at_end": code_tree_sha256(repo)["tree_sha256"],
               "environment": environment(), "total_seconds": round(time.perf_counter() - t0, 1)}
    (out / "extraction_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file()) + [repo / "artifacts/reports/m4b0_feature_diagnostics_B_gate.csv"]
    (out / "hash_manifest.json").write_text(json.dumps({"created_utc": summary["created_utc"], "files": {
        p.relative_to(repo).as_posix(): io.sha256_file(p) for p in files}}, indent=2), encoding="utf-8")
    for p in (out / "features").glob("*.csv"):
        os.chmod(p, 0o444)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M4b-0 gate feature extraction (no D1 join, no fitting).")
    ap.add_argument("--config", default="configs/gate_m4b_v0_2.yaml")
    args = ap.parse_args(argv)
    repo = Path(".").resolve()
    s = run(repo, (repo / args.config).resolve())
    print(json.dumps({r: {"rows": v["rows"], "non_finite": v["non_finite_values"]} for r, v in s["per_role"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
