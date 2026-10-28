"""How the two candidate S-A aggregates would weight lead sets, measured on development roles (B_gate, 9a, 9b).

Reads only D1 keys (mask_id, observed-lead count) and the D1 column of the offline labels: no gate feature, no score, no
model, no fold 10. For each role it reports how many of the 258 lead sets have both D1 classes (eligible for S-A), how many
lack one class, and how the pair-weighted aggregate would distribute weight over lead-set sizes compared with the equal
weight per eligible lead set of the stratum-mean aggregate. B_gate (20,992 rows) is the closest in size to fold 10 (about
17,584 rows).

    .\\.venv\\Scripts\\python.exe -m src.eval.profile_sa_strata
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/reports/m4b0d_sa_strata_profile.json"
ROLES = ("B_gate", "9a_calibration", "9b_threshold")


def profile(role: str) -> dict:
    keys = pd.read_csv(REPO / f"artifacts/d1/v0_1/keys/{role}_rows.csv", usecols=["row_id", "mask_id", "n_observed"])
    d1 = pd.read_csv(REPO / f"artifacts/d1/v0_1/offline_labels/{role}_OFFLINE_d1_labels.csv", usecols=["row_id", "D1"])
    t = keys.merge(d1, on="row_id", validate="one_to_one")
    g = t.groupby("mask_id").agg(rows=("D1", "size"), d1=("D1", "sum"), n_obs=("n_observed", "first"))
    g["non_d1"] = g["rows"] - g["d1"]
    g["pairs"] = g["d1"] * g["non_d1"]
    g["eligible"] = (g["d1"] > 0) & (g["non_d1"] > 0)
    g["size_group"] = pd.cut(g["n_obs"], [0, 1, 3, 6, 12], labels=["1 lead", "2-3 leads", "4-6 leads", "7+ leads"])
    e = g[g["eligible"]]
    by = e.groupby("size_group", observed=False).agg(lead_sets=("pairs", "size"), pairs=("pairs", "sum"))
    by["pair_weight_share"] = by["pairs"] / by["pairs"].sum()
    by["stratum_mean_weight_share"] = by["lead_sets"] / by["lead_sets"].sum()
    return {"rows": int(len(t)), "d1_rows": int(t["D1"].sum()), "lead_sets_present": int(len(g)),
            "lead_sets_eligible": int(g["eligible"].sum()), "lead_sets_without_d1": int((g["d1"] == 0).sum()),
            "lead_sets_without_non_d1": int((g["non_d1"] == 0).sum()),
            "d1_rows_per_eligible_lead_set_median": float(e["d1"].median()),
            "lead_sets_eligible_with_1_d1_row": int((e["d1"] == 1).sum()),
            "by_size": {str(k): {kk: (float(vv) if "share" in kk else int(vv)) for kk, vv in row.items()}
                        for k, row in by.iterrows()}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    res = {r: profile(r) for r in ROLES}
    OUT.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "milestone": "M4b-0d pre-pin", "reads": "D1 keys (mask_id, n_observed) and offline D1 only",
                               "roles": res}, indent=2), encoding="utf-8")
    for r, v in res.items():
        print(r, {k: v[k] for k in ("rows", "d1_rows", "lead_sets_present", "lead_sets_eligible", "lead_sets_without_d1",
                                    "lead_sets_eligible_with_1_d1_row")})
        for k, row in v["by_size"].items():
            print("   ", k, row)


if __name__ == "__main__":
    main()
