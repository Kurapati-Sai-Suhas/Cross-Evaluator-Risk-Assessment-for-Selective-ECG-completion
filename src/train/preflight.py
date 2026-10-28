"""Pre-training audit (Milestone 2c, Task A). Any FAIL means: do not train; report and ask for a decision.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.train.preflight --config configs/train_l2g_v0_1.yaml

Checks: pinned hashes and read-only flags (frozen manifest, protocol + lock); the split is still reproducible;
saved reports are not stale; no conflicting configs or existing run folder; the training config agrees with the
architecture lock and protocol; train/validation roles are patient-disjoint and never touch folds 9-10;
GPU/environment; the full test suite passes. Writes artifacts/reports/preflight_<run_name>.json.
Exit code: 0 = all PASS, 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data import make_split_manifest as ms
from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.utils.provenance import code_tree_sha256, environment

# Values fixed by docs/architecture_lock_v0_1.md Section 2.3 (D-021). Keys are dotted paths in the training config.
LOCK_V0_1: dict[str, Any] = {
    "model.in_channels": 24,
    "optimizer.generator.type": "sgd", "optimizer.generator.lr": 0.01, "optimizer.generator.momentum": 0.5,
    "optimizer.generator.weight_decay": 0.0,
    "optimizer.discriminator.type": "sgd", "optimizer.discriminator.lr": 0.001,
    "optimizer.discriminator.momentum": 0.5, "optimizer.discriminator.weight_decay": 0.0,
    "optimizer.discriminator_scheduler.patience": 10,
    "training.batch_size": 256, "training.max_epochs": 50, "training.adversarial_start_epoch": 10,
    "training.alpha": 0.1, "training.vttd_crop": 400, "training.early_stopping_patience": 10,
    "training.checkpoint": "best_so_far", "training.shuffle": "fixed_seed", "training.mixed_precision": False,
    "training.training_masks": "uniform_over_258_l2g_lead_sets",
    "data.train_role": "A_fit", "data.validation_role": "C_dev",
    "data.train_window_offsets": [0, 452], "data.validation_window_offset": 0,
}


# G v0.1 (decision D-024): the frozen reconstructor checkpoint. Evaluator milestones must leave it untouched.
G_V0_1_CHECKPOINT = "artifacts/runs/l2g_v0_1_run02/best.pt"
G_V0_1_SHA256 = "d515875a7783be5a54934001e13b182ff4f75a5e35554f0e9590d5a1c602bb3b"


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return "<missing>"
        node = node[part]
    return node


def check_training_config(cfg: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    """Violations of the lock or protocol (empty list = conforms)."""
    problems = [f"{k}: config {_get(cfg, k)!r} != lock {v!r}" for k, v in LOCK_V0_1.items() if _get(cfg, k) != v]
    if cfg.get("seed") != protocol["seed"]:
        problems.append(f"seed {cfg.get('seed')} != protocol seed {protocol['seed']}")
    if _get(cfg, "data.validation_masks_per_record") != protocol["d1"]["masks_per_record"]:
        problems.append("validation_masks_per_record differs from protocol d1.masks_per_record")
    if cfg["data"]["train_role"] not in protocol["roles"]["fit"]:
        problems.append("train role is not the protocol 'fit' role")
    if cfg["data"]["validation_role"] not in protocol["roles"]["model_selection"]:
        problems.append("validation role is not the protocol 'model_selection' role")
    return problems


def check_evaluator_config(cfg: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    """Violations of protocol v0.1 Sections 1, 3 and 5 by an evaluator config (empty list = conforms)."""
    problems = []
    if cfg.get("seed") != protocol["seed"]:
        problems.append(f"seed {cfg.get('seed')} != protocol seed {protocol['seed']}")
    if cfg["data"]["train_role"] not in protocol["roles"]["fit"]:
        problems.append("train role is not the protocol 'fit' role")
    if cfg["data"]["validation_role"] not in protocol["roles"]["model_selection"]:
        problems.append("validation role is not the protocol 'model_selection' role")
    if cfg["data"]["validation_role"] != protocol["eligibility"]["measured_on"]:
        problems.append("eligibility would not be measured on the protocol role")
    if cfg["data"]["validation_masks_per_record"] != protocol["d1"]["masks_per_record"]:
        problems.append("validation_masks_per_record differs from protocol d1.masks_per_record")
    if not 1 <= len(cfg["attempts"]) <= protocol["eligibility"]["max_training_attempts"]:
        problems.append(f"{len(cfg['attempts'])} attempts > protocol maximum")
    if cfg.get("operating_threshold") != protocol["d1"]["evaluator_threshold_rule"]:
        problems.append("operating threshold rule differs from protocol")
    expected_in = {"complete": 12, "masked": 24}.get(cfg.get("input"))
    if expected_in is None or cfg["model"]["in_channels"] != expected_in:
        problems.append(f"input {cfg.get('input')!r} with in_channels {cfg['model']['in_channels']} is inconsistent")
    if cfg["model"]["family"] not in ("resnet1d_wang",):
        problems.append(f"model family {cfg['model']['family']} not approved/implemented for v0.1")
    return problems


def _check(rows: list[dict[str, Any]], name: str, ok: bool, detail: Any, critical: bool = True) -> None:
    rows.append({"check": name, "status": "PASS" if ok else "FAIL", "critical": critical, "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def run(repo: Path, config_path: Path, run_tests: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    cfg = io.load_config(config_path)
    sp = audit_cfg["split"]
    mdir = repo / sp["manifest_dir"]
    frozen = mdir / f"{sp['frozen_manifest_name']}.csv"
    frozen_meta = json.loads((mdir / f"{sp['frozen_manifest_name']}_metadata.json").read_text(encoding="utf-8"))
    frozen_sha = io.sha256_file(frozen)
    _check(rows, "P1_frozen_manifest_hash", frozen_sha == sp["frozen_manifest_sha256"] == frozen_meta["manifest_sha256"],
           frozen_sha)
    _check(rows, "P1_frozen_manifest_read_only", not os.access(frozen, os.W_OK), str(frozen))

    lock_path = repo / "artifacts" / "locks" / "protocol_v0_1.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    proto_path = repo / "configs" / "protocol_v0_1.yaml"
    proto_sha = io.sha256_file(proto_path)
    _check(rows, "P2_protocol_hash_matches_lock", proto_sha == lock["sha256"] and protocol["status"] == "APPROVED", proto_sha)
    _check(rows, "P2_protocol_and_lock_read_only", not os.access(proto_path, os.W_OK) and not os.access(lock_path, os.W_OK),
           "both read-only")

    db = io.load_database(repo / audit_cfg["dataset"]["database_csv"])
    codes, _ = io.parse_all_scp_codes(db)
    manifest = load_frozen_manifest(audit_cfg, repo)
    regen = ms.make_manifest(db, codes, audit_cfg)
    same = (regen[["ecg_id", "role"]].reset_index(drop=True).astype(str)
            .equals(manifest[["ecg_id", "role"]].reset_index(drop=True).astype(str)))
    _check(rows, "P3_split_reproducible_from_config", same, "regenerated roles identical to the frozen manifest")

    reports = repo / "artifacts" / "reports"
    audit = json.loads((reports / "audit_summary.json").read_text(encoding="utf-8"))
    db_sha = io.sha256_file(repo / audit_cfg["dataset"]["database_csv"])
    _check(rows, "P4_audit_report_current",
           audit["audit_status"] == "PASS" and any(i["sha256"] == db_sha for i in audit["inputs"]), audit["created_utc"])
    wave = json.loads((reports / "waveform_verification.json").read_text(encoding="utf-8"))
    wave_frozen = next(c["value"] for c in wave["checks"] if c["check"] == "frozen_manifest_unchanged")
    _check(rows, "P4_waveform_report_current", wave["status"] != "FAIL" and wave_frozen == frozen_sha,
           f"{wave['status']} ({wave['created_utc']})")
    smoke = json.loads((reports / "smoke_l2g_v0_1.json").read_text(encoding="utf-8"))
    _check(rows, "P4_smoke_report_current", smoke["status"] == "PASS" and smoke["protocol_sha256"] == lock["sha256"],
           f"{smoke['status']} ({smoke['created_utc']})")

    protocol_files = sorted(p.name for p in (repo / "configs").glob("protocol_v*.yaml"))
    run_dir = repo / cfg["output_dir"] / cfg["run_name"]
    _check(rows, "P5_no_conflicting_configs_or_runs", protocol_files == ["protocol_v0_1.yaml"] and not run_dir.exists(),
           {"protocol_files": protocol_files, "run_dir_exists": run_dir.exists()})

    if "evaluator_id" in cfg:
        problems = check_evaluator_config(cfg, protocol)
        _check(rows, "P6_evaluator_config_matches_protocol", not problems, problems or "conforms to protocol v0.1")
        g = repo / G_V0_1_CHECKPOINT
        g_sha = io.sha256_file(g)
        _check(rows, "P6b_frozen_G_v0_1_unchanged_and_read_only", g_sha == G_V0_1_SHA256 and not os.access(g, os.W_OK),
               g_sha)
    else:
        problems = check_training_config(cfg, protocol)
        _check(rows, "P6_training_config_matches_lock_and_protocol", not problems, problems or "all locked values match")

    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    train = manifest[(manifest["role"] == cfg["data"]["train_role"]) & ~manifest["ecg_id"].isin(excluded)]
    val = manifest[manifest["role"] == cfg["data"]["validation_role"]]
    disjoint = not (set(train["patient_id"]) & set(val["patient_id"]))
    folds_ok = train["strat_fold"].between(1, 8).all() and val["strat_fold"].between(1, 8).all()
    _check(rows, "P7_no_leakage_in_planned_data", disjoint and folds_ok and not (set(train["ecg_id"]) & excluded),
           {"train_records": len(train), "train_patients": int(train["patient_id"].nunique()),
            "val_records": len(val), "val_patients": int(val["patient_id"].nunique()),
            "patient_overlap": 0 if disjoint else "OVERLAP", "folds_used": "1-8 only" if folds_ok else "INVALID"})

    env = environment()
    gpu_ok = bool(env.get("cuda_available"))
    if gpu_ok:
        import torch
        free, total = torch.cuda.mem_get_info()
        env["gpu_free_MiB_at_preflight"] = round(free / 2 ** 20)
    _check(rows, "P8_gpu_and_environment", gpu_ok, {k: env.get(k) for k in ("gpu", "torch_cuda_build", "nvidia_driver",
                                                                          "gpu_free_MiB_at_preflight")})

    if run_tests:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=repo, capture_output=True, text=True)
        tail = [line for line in proc.stdout.splitlines() if re.search(r"\d+ (passed|failed)", line)]
        _check(rows, "P9_full_test_suite", proc.returncode == 0, {"exit_code": proc.returncode, "summary": tail[-1:]})

    tree = code_tree_sha256(repo)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": cfg["run_name"], "training_config": str(config_path.relative_to(repo)),
        "kind": "evaluator" if "evaluator_id" in cfg else "reconstructor",
        "training_config_sha256": io.sha256_file(config_path),
        "frozen_manifest_sha256": frozen_sha, "protocol_sha256": proto_sha,
        "code_tree_sha256": tree["tree_sha256"], "code_files": tree["n_files"],
        "environment": env, "checks": rows,
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-training audit (no training); reconstructor or evaluator config.")
    parser.add_argument("--config", default="configs/train_l2g_v0_1.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.root).resolve()
    report = run(repo, (repo / args.config).resolve(), run_tests=not args.skip_tests)
    out = repo / "artifacts" / "reports" / f"preflight_{report['run_name']}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Preflight overall: {report['status']}  (report: {out})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
