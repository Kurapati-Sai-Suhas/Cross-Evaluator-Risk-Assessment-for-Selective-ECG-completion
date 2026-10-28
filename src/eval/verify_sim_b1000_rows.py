"""Spot-check of artifacts/reports/simulations/sim_B1000_sims1000_seed20261021_raw.csv (B = 1,000 run of the rewound first
attempt of the pre-pin milestone, whose generating script version is no longer on disk).

For a fixed, evenly spaced sample of simulations per scenario, the row is recomputed from its recorded data and replicate
seeds with the current data generator and decision code (``src.eval.simulate_selective_inference`` functions and
``src.eval.selective``) and compared with the recorded point estimate, U, n_nonneg and decision.

    .\\.venv\\Scripts\\python.exe -m src.eval.verify_sim_b1000_rows
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.eval import selective as sel
from src.eval import simulate_selective_inference as base

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "artifacts/reports/simulations/sim_B1000_sims1000_seed20261021_raw.csv"
OUT = REPO / "artifacts/reports/m4b0d_simulations_B1000_spotcheck.json"
T = {"H1_null_uninformative_gate": norm.ppf(0.8), "H1_alternative_informative_gate": norm.ppf(0.8) * np.sqrt(3)}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    with RAW.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results = []
    for name, (hyp, _, build) in base.SCENARIOS.items():
        mine = [r for r in rows if r["scenario"] == name]
        pick = [mine[i] for i in np.linspace(0, len(mine) - 1, 8).round().astype(int)]
        for r in pick:
            data = base.make_rows(np.random.default_rng(int(r["data_seed"])), int(r["patients"]))
            scores = build(data)
            if hyp == "H1":
                point, d = base.h1_deltas(data, scores[0], T[name], 1000, int(r["replicate_seed"]))
            else:
                point, d = base.h2_deltas(data, scores[0], scores[1], 1000, int(r["replicate_seed"]))
            dec = sel.one_sided_decision(d)
            same = (int(r["rows"]) == data["d1"].size and float(r["point_delta"]) == point
                    and float(r["upper_bound_U"]) == dec["upper_bound"] and int(r["n_nonneg"]) == dec["n_nonneg"]
                    and (r["reject"] == "True") == dec["reject"])
            results.append({"scenario": name, "sim": int(r["sim"]), "identical": bool(same)})
            print(name, r["sim"], "identical" if same else "DIFFERENT", flush=True)
    OUT.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "raw_csv": RAW.relative_to(REPO).as_posix(),
                               "raw_csv_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
                               "rows_checked": len(results), "rows_identical": sum(x["identical"] for x in results),
                               "rows": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
