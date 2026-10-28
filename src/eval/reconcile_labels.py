"""M4a Step B: reconcile CLBBB/CRBBB label counts by role with the frozen target rule (protocol v0.1 Section 1).

    .\\.venv\\Scripts\\python.exe -m src.eval.reconcile_labels

Recomputes, for every role EXCEPT test_fold10 (not inspected in M4a), the records/patients positive under the protocol rule
(code listed in scp_codes at any likelihood; ILBBB/IRBBB-only records negative; ecg_id 12722 excluded) and compares them with
two earlier, independent extractions: the frozen role summary (Milestone 1 split) and the Milestone 1 per-fold audit
(`bbb_fold_distribution.csv`, folds 1-9 only; fold-10 rows are dropped before any use). The database-wide totals including
fold 10 (CLBBB 536/430, CRBBB 541/464) are cited from protocol Section 1 / D-022 and are NOT recomputed here, because that
would read fold-10 labels. Writes artifacts/reports/m4a_label_reconciliation.csv and .json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import ptbxl_io as io
from src.data.freeze_split_manifest import load_frozen_manifest
from src.data.targets import target_matrix

ROLES = ["A_fit", "B_gate", "C_dev", "9a_calibration", "9b_threshold"]


def main() -> int:
    repo = Path(".").resolve()
    audit_cfg = io.load_config(repo / "configs" / "data_audit.yaml")
    protocol = io.load_config(repo / "configs" / "protocol_v0_1.yaml")
    outputs = list(protocol["targets"]["primary_outputs"])
    excluded = set(protocol["targets"]["excluded_ecg_ids"])
    manifest = load_frozen_manifest(audit_cfg, repo)
    db = io.load_database(repo / audit_cfg["dataset"]["database_csv"])
    summary = pd.read_csv(repo / "artifacts" / "manifests" / "split_manifest_v0_draft_role_summary.csv", index_col="role")
    m1 = pd.read_csv(repo / "artifacts" / "reports" / "bbb_fold_distribution.csv")
    m1 = m1[m1["strat_fold"].between(1, 9)]  # fold-10 rows are dropped before any use

    rows, checks = [], []
    for role in ROLES:
        r = manifest[manifest["role"] == role]
        kept = r[~r["ecg_id"].isin(excluded)].sort_values("ecg_id")
        y = target_matrix(db, kept["ecg_id"].tolist(), protocol)
        codes = [io.parse_scp_codes(t) for t in db.set_index("ecg_id").loc[kept["ecg_id"], "scp_codes"]]
        row = {"role": role, "records_in_manifest": len(r), "excluded_records": int(r["ecg_id"].isin(excluded).sum()),
               "records_used": len(kept), "patients_used": int(kept["patient_id"].nunique()),
               "both_outputs_positive_records": int((y.sum(axis=1) == 2).sum()),
               "ILBBB_without_CLBBB_negatives": sum(("ILBBB" in c) and ("CLBBB" not in c) for c in codes),
               "IRBBB_without_CRBBB_negatives": sum(("IRBBB" in c) and ("CRBBB" not in c) for c in codes)}
        for j, o in enumerate(outputs):
            row[f"{o}_records"] = int(y[:, j].sum())
            row[f"{o}_patients"] = int(kept.loc[y[:, j] == 1, "patient_id"].nunique())
            row[f"{o}_records_role_summary"] = int(summary.loc[role, f"{o}_records"])
            row[f"{o}_patients_role_summary"] = int(summary.loc[role, f"{o}_patients"])
            row[f"{o}_records_likelihood_below_100"] = sum(0 <= c.get(o, -1) < 100 for c in codes)
            checks.append({"check": f"{role}_{o}_matches_role_summary",
                           "pass": row[f"{o}_records"] == row[f"{o}_records_role_summary"]
                           and row[f"{o}_patients"] == row[f"{o}_patients_role_summary"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    excluded_info = {int(e): {"role": manifest.loc[manifest.ecg_id == e, "role"].tolist(),
                              "scp_codes": db.set_index("ecg_id").loc[e, "scp_codes"]} for e in excluded}
    folds_1_9 = {}
    for o in outputs:
        ours = int(table[f"{o}_records"].sum())
        excl_pos = sum(o in io.parse_scp_codes(v["scp_codes"]) for v in excluded_info.values())
        theirs = int(m1.loc[m1["code"] == o, "n_records"].sum())
        folds_1_9[o] = {"sum_over_roles_folds_1_9": ours, "excluded_positive_records": excl_pos,
                        "m1_audit_folds_1_9": theirs, "match": ours + excl_pos == theirs}
        checks.append({"check": f"{o}_folds_1_9_match_M1_audit", "pass": ours + excl_pos == theirs})
    out = {
        "milestone": "M4a", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "y=1 iff the code is listed in scp_codes at any likelihood (protocol targets.positive_min_likelihood 0); "
                "ILBBB/IRBBB-only records are negatives; excluded ecg_ids removed",
        "excluded": excluded_info, "folds_1_9_cross_check": folds_1_9, "checks": checks,
        "database_wide_totals_cited_not_recomputed": {
            "CLBBB": "536 records / 430 patients (protocol Section 1, M1 audit)",
            "CRBBB": "541 records / 464 patients (protocol Section 1, D-022; the cited 542 is not reproducible from v1.0.3)",
            "why_not_recomputed": "the totals include fold 10, whose labels are not read in M4a"},
        "status": "PASS" if all(c["pass"] for c in checks) else "MISMATCH",
        "input_sha256": {"ptbxl_database.csv": io.sha256_file(repo / audit_cfg["dataset"]["database_csv"]),
                         "frozen_manifest": audit_cfg["split"]["frozen_manifest_sha256"],
                         "role_summary": io.sha256_file(repo / "artifacts/manifests/split_manifest_v0_draft_role_summary.csv"),
                         "bbb_fold_distribution.csv": io.sha256_file(repo / "artifacts/reports/bbb_fold_distribution.csv")},
    }
    table.to_csv(repo / "artifacts" / "reports" / "m4a_label_reconciliation.csv", index=False)
    (repo / "artifacts" / "reports" / "m4a_label_reconciliation.json").write_text(json.dumps(out, indent=2, default=str),
                                                                                  encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({"folds_1_9_cross_check": folds_1_9, "excluded": excluded_info, "status": out["status"]}, indent=1))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
