"""M4a: by-role D1 count table and post-generation integrity checks (read-only; no rescoring).

    .\\.venv\\Scripts\\python.exe -m src.eval.summarize_d1

Reads artifacts/d1/v0_1/generation_summary.json and hash_manifest.json and writes
artifacts/reports/m4a_d1_counts_by_role.csv and artifacts/reports/m4a_d1_summary.json.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import ptbxl_io as io

OUT = Path("artifacts/d1/v0_1")


def main() -> int:
    repo = Path(".").resolve()
    s = json.loads((repo / OUT / "generation_summary.json").read_text(encoding="utf-8"))
    man = json.loads((repo / OUT / "hash_manifest.json").read_text(encoding="utf-8"))
    cfg = io.load_config(repo / s["config"])
    rows = []
    for role, r in s["per_role"].items():
        row = {"role": role, "use": "sanity check only (development)" if r["sanity_only"] else "held-out D1 role",
               "rows": r["rows"], "records": r["records"], "patients": r["patients"], "D1_rows": r["D1_rows"],
               "D1_rate": round(r["D1_rate_rows"]["point"], 5), "D1_rate_ci_low": round(r["D1_rate_rows"]["ci_low"], 5),
               "D1_rate_ci_high": round(r["D1_rate_rows"]["ci_high"], 5),
               "records_with_any_D1": r["records_with_any_D1"], "patients_with_any_D1": r["patients_with_any_D1"],
               "D1_rows_disagreeing_in_both_outputs": r["D1_rows_disagreeing_in_both_outputs"]}
        for y in s["outputs"]:
            q = r[y]
            row.update({f"{y}_label_positive_records": q["label_positive_records"],
                        f"{y}_label_positive_patients": q["label_positive_patients"],
                        f"{y}_E1_complete_positive_records": q["z_full_positive_records"],
                        f"{y}_D1_miss_rows": q["D1_miss_rows"], f"{y}_D1_false_rows": q["D1_false_rows"],
                        f"{y}_D1_miss_rate": round(q["D1_miss_rate_rows"]["point"], 5),
                        f"{y}_D1_false_rate": round(q["D1_false_rate_rows"]["point"], 5),
                        f"{y}_near_tau_complete_records": r["near_tau_full_records"][y],
                        f"{y}_near_tau_reconstructed_rows": r["near_tau_rec_rows"][y]})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(repo / "artifacts/reports/m4a_d1_counts_by_role.csv", index=False)

    checks = []

    def check(name: str, ok: bool, detail) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    mismatched = [p for p, h in man["files"].items() if io.sha256_file(repo / p) != h]
    check("hash_manifest_matches_all_files", not mismatched, {"files": len(man["files"]), "mismatched": mismatched})
    label_files = sorted((repo / OUT).rglob("*.csv"))
    check("d1_csv_files_read_only", all(not os.access(p, os.W_OK) for p in label_files), len(label_files))
    cache_files = sorted(p.name for p in Path(cfg["data"]["cache_dir"]).glob("*"))
    check("signal_cache_has_no_fold10_files", not any("fold10" in f or f.startswith("test_") for f in cache_files), cache_files)
    check("no_threshold_lock_exists", not (repo / "artifacts/locks/threshold_lock_v0_1.json").exists(), "fold 10 stays closed")
    check("fold10_not_loaded", s["fold10_loaded"] is False and "test_fold10" not in s["per_role"], s["roles"])
    check("model_state_unchanged", s["model_state_unchanged"] is True, s["model_state_sha256"])
    check("reproducibility_exact", all(v["max_abs_diff_p_full"] == 0 and v["max_abs_diff_p_rec"] == 0 and v["decisions_identical"]
                                       for v in s["reproducibility"].values()), "reverse-order re-score of 512 rows per role")
    check("frozen_input_hashes", s["inputs"]["g"] == cfg["frozen"]["g_sha256"] and s["inputs"]["e1"] == cfg["frozen"]["e1_sha256"]
          and s["inputs"]["protocol"] == cfg["frozen"]["protocol_sha256"], s["inputs"])
    out = {"milestone": "M4a", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "generation_summary_sha256": io.sha256_file(repo / OUT / "generation_summary.json"),
           "hash_manifest_sha256": io.sha256_file(repo / OUT / "hash_manifest.json"),
           "tau_float32": s["tau_float32"], "decision_rule": s["decision_rule"],
           "calibration_readiness_9a": s["calibration_readiness_9a"],
           "c_dev_vs_e1_run_predictions": s.get("c_dev_vs_e1_run_predictions"),
           "reproducibility": s["reproducibility"], "checks": checks,
           "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
           "per_role_table": "artifacts/reports/m4a_d1_counts_by_role.csv"}
    (repo / "artifacts/reports/m4a_d1_summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(table.T.to_string())
    print("M4a post-generation checks:", out["status"])
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
