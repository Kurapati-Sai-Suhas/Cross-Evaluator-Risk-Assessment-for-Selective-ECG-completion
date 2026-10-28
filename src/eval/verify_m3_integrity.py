"""Milestone M3b integrity check (read-only). Writes artifacts/reports/m3b_integrity_check.json.

Run from the repository root (after E1, E_dir and the C_dev comparison exist):
    .\\.venv\\Scripts\\python.exe -m src.eval.verify_m3_integrity

I1 hashes and read-only flags of every frozen artifact (manifest, protocol + lock, G v0.1, selected E1 and E_dir checkpoints)
I2 checkpoint reload: logged SHA-256 = file SHA-256; the C_dev comparison's E1(complete) AUROC equals the E1 run summary
   (an independent reload and re-scoring of the frozen E1 through a different code path)
I3 split boundaries: the six frozen roles are patient-disjoint; E1/E_dir were fitted on A_fit and scored on C_dev only
I4 fold 10 (and B_gate, 9a, 9b): no record of these roles appears in any M3 output table, and the signal cache holds only
   A_fit and C_dev
Nothing is trained or modified.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.train.preflight import G_V0_1_CHECKPOINT, G_V0_1_SHA256
from src.utils.provenance import code_tree_sha256

E1_RUN = "artifacts/runs/e1_resnet1d_v0_1_run01"
EDIR_RUN = "artifacts/runs/edir_resnet1d_v0_1_run02"
COMPARISON = "artifacts/reports/m3a_cdev_evaluator_comparison.json"


def _ro(path: Path) -> bool:
    return not os.access(path, os.W_OK)


def run(repo: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    lock = json.loads((repo / "artifacts" / "locks" / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    manifest_path = repo / audit_cfg["split"]["manifest_dir"] / f"{audit_cfg['split']['frozen_manifest_name']}.csv"
    proto = repo / "configs" / "protocol_v0_1.yaml"
    g = repo / G_V0_1_CHECKPOINT
    e1 = json.loads((repo / E1_RUN / "summary.json").read_text(encoding="utf-8"))
    ed = json.loads((repo / EDIR_RUN / "summary.json").read_text(encoding="utf-8"))
    comp = json.loads((repo / COMPARISON).read_text(encoding="utf-8"))
    e1_ckpt, ed_ckpt = repo / e1["selected_checkpoint"], repo / ed["selected_checkpoint"]

    # I1 hashes and read-only flags
    items = {
        "frozen_manifest": (manifest_path, audit_cfg["split"]["frozen_manifest_sha256"]),
        "protocol_v0_1": (proto, lock["sha256"]),
        "G_v0_1": (g, G_V0_1_SHA256),
        "E1_selected": (e1_ckpt, e1["selected_checkpoint_sha256"]),
        "E_dir_selected": (ed_ckpt, ed["selected_checkpoint_sha256"]),
    }
    for name, (path, expected) in items.items():
        actual = io.sha256_file(path)
        check(f"I1_{name}_hash_and_read_only", actual == expected and _ro(path),
              {"path": str(path.relative_to(repo)), "sha256": actual, "read_only": _ro(path)})
    check("I1_protocol_lock_read_only", _ro(repo / "artifacts" / "locks" / "protocol_v0_1.lock.json"), "lock file")

    # I2 reload consistency
    check("I2_E1_reload_matches_logged", e1["reload_matches_logged"] is True, e1["reloaded_c_dev_macro_auroc"])
    check("I2_E_dir_reload_matches_logged", ed["reload_matches_logged"] is True, ed["reloaded_c_dev_macro_auroc"])
    diffs = {c: abs(comp["results"][c]["auroc_e1_complete"]["point"] - e1["per_output"][c]["auroc"]["point"])
             for c in e1["outputs"]}
    # different batch sizes may change cuDNN kernels and hence logits in the last bits; 1e-4 allows that, not more
    check("I2_E1_rescored_in_comparison_equals_summary", max(diffs.values()) < 1e-4, diffs)
    ed_diffs = {c: abs(comp["results"][c]["auroc_edir"]["point"] - ed["per_output"][c]["auroc"]["point"])
                for c in ed["outputs"]}
    check("I2_E_dir_rescored_in_comparison_equals_summary", max(ed_diffs.values()) < 1e-4, ed_diffs)
    check("I2_comparison_used_frozen_checkpoints",
          comp["E1"]["sha256"] == e1["selected_checkpoint_sha256"] and comp["E_dir"]["sha256"] == ed["selected_checkpoint_sha256"]
          and comp["G"]["sha256"] == G_V0_1_SHA256, "E1, E_dir, G hashes recorded in the comparison")

    # I3 split boundaries
    manifest = load_frozen_manifest(audit_cfg, repo)
    protocol = io.load_config(proto)
    a_fit_used = int(((manifest.role == "A_fit") & ~manifest.ecg_id.isin(protocol["targets"]["excluded_ecg_ids"])).sum())
    pats = {r: set(manifest.loc[manifest.role == r, "patient_id"]) for r in manifest.role.unique()}
    overlaps = {f"{a}|{b}": len(pats[a] & pats[b]) for a, b in combinations(sorted(pats), 2)}
    check("I3_roles_patient_disjoint", all(v == 0 for v in overlaps.values()), {"roles": sorted(pats), "max_overlap": max(overlaps.values())})
    for name, s in (("E1", e1), ("E_dir", ed)):
        check(f"I3_{name}_roles", s["roles_loaded"] == ["A_fit", "C_dev"] and s["fold10_loaded"] is False
              and s["train_records"] == a_fit_used, {"roles_loaded": s["roles_loaded"],
              "train_records": s["train_records"], "fold10_loaded": s["fold10_loaded"]})

    # I4 no forbidden-role record in any M3 output table; cache contents
    role_of = manifest.set_index("ecg_id")["role"]
    tables = [repo / E1_RUN / "c_dev_predictions_selected.csv", repo / EDIR_RUN / "c_dev_predictions_selected.csv",
              repo / comp["rows_file"]]
    found = {}
    for t in tables:
        ids = pd.read_csv(t, usecols=["ecg_id"])["ecg_id"].unique()
        found[str(t.relative_to(repo))] = sorted(set(role_of.loc[ids]))
    check("I4_output_tables_contain_only_C_dev", all(v == ["C_dev"] for v in found.values()), found)
    cache_dir = io.load_config(repo / E1_RUN / "config_snapshot" / "evaluator_e1_v0_1.yaml")["data"]["cache_dir"]
    cache_files = sorted(p.name for p in Path(cache_dir).glob("*"))
    check("I4_cache_holds_only_A_fit_and_C_dev", all(f.startswith(("A_fit_", "C_dev_")) for f in cache_files), cache_files)
    check("I4_no_D1_and_no_gate_outputs", comp["d1_computed"] is False and not (repo / "artifacts" / "locks" /
          "threshold_lock_v0_1.json").exists() and e1["d1_generated"] is False and ed["d1_generated"] is False,
          "no D1 computed; no threshold lock exists")

    return {"milestone": "M3b", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_tree_sha256": code_tree_sha256(repo)["tree_sha256"], "checks": checks,
            "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"}


def main() -> int:
    repo = Path(".").resolve()
    report = run(repo)
    out = repo / "artifacts" / "reports" / "m3b_integrity_check.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"M3b integrity: {report['status']}  (report: {out})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
