"""Archived simulation of the P3 decision rule at the protocol's B = 10,000, one scenario per process, raw output per simulation.

Synthetic data only (no project data is read). Reuses the data generator and the H1/H2 replicate functions of
``src.eval.simulate_selective_inference`` (kept unchanged: it produced the historical B = 1,000 run), which call the reference
implementation in ``src.eval.selective``: patient multiplicities from ``default_rng(seed).integers(0, P, size=(B, P))``,
acceptance s <= t (H1) or exact 80% coverage with fractional ties recomputed per replicate (H2), contrast = risk(method) -
risk(comparator) with +inf for an undefined replicate, and the decision U = delta*_(ceil(0.95 (B + 1))) < 0.

Every simulation writes one raw row (seeds, sizes, point estimate, U, n_nonneg, decision, descriptive interval, number of +inf
replicates, and whether the superseded P2 p-rule agrees). The metadata file records the command, environment and code hashes.

    python -m src.eval.simulate_decision_rule --scenario H2_null_exchangeable_tied_scores --out-dir DIR [--sims 1000]
        [--shard i --shards n]   (simulations i, i + n, i + 2n, ...; seeds do not depend on the sharding)
    python -m src.eval.simulate_decision_rule --summarize DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import norm

from src.eval import selective as sel
from src.eval import simulate_selective_inference as base

REPO = Path(__file__).resolve().parents[2]
T_H1 = norm.ppf(0.8)                                           # population coverage 0.80 for an N(0, 1) score


def _tied(x: np.ndarray) -> np.ndarray:
    return np.floor(x * 2.0) / 2.0                             # half-unit bins: heavy ties, still exchangeable


# name: (index for seeding, hypothesis, true contrast, score builder, H1 threshold)
SCENARIOS = {
    "H1_null_uninformative_gate": (0, "H1", "0", lambda r: (r["noise"],), T_H1),
    "H1_alternative_informative_gate": (1, "H1", "< 0", lambda r: (r["q1"] + r["q2"] + r["noise"],), T_H1 * math.sqrt(3)),
    "H2_null_exchangeable_scores": (2, "H2", "0", lambda r: (r["q2"] + r["noise"], r["q1"] + r["noise2"]), None),
    "H2_null_exchangeable_tied_scores": (3, "H2", "0", lambda r: (_tied(r["q2"] + r["noise"]), _tied(r["q1"] + r["noise2"])), None),
    "H2_alternative_better_ranking": (4, "H2", "< 0", lambda r: (r["q1"] + r["q2"] + 0.5 * r["noise"], r["q1"] + r["noise2"]), None),
}
FIELDS = ["scenario", "sim", "data_seed", "replicate_seed", "rows", "patients", "d1_rate", "accepted_share", "tied_rows_share",
          "point_delta", "upper_bound_U", "k_upper", "n_nonneg", "reject", "p2_rule_reject", "two_sided_low", "two_sided_high",
          "replicates_inf"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine()}


def sim_seeds(name: str, sims: int, seed: int) -> list[tuple[int, int]]:
    """(data seed, replicate seed) of every simulation of a scenario, drawn in order from its own stream, so a simulation's
    result does not depend on how the scenario is split into shards."""
    rng = np.random.default_rng([seed, SCENARIOS[name][0]])
    return [(int(rng.integers(2**63)), int(rng.integers(2**31))) for _ in range(sims)]


def resume_point(raw: Path, name: str, todo: list[int], seeds: list[tuple[int, int]]) -> int:
    """Number of simulations already in an interrupted shard file, after checking that its rows are exactly the first ones
    of the shard, in order, with the scenario's seeds, and that every row is complete."""
    with raw.open(encoding="utf-8", newline="") as f:
        if not f.read().endswith("\n"):
            raise SystemExit(f"{raw}: last row is incomplete; refusing to resume")
    with raw.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"{raw}: unexpected columns")
        done = list(reader)
    if [int(r["sim"]) for r in done] != todo[:len(done)] or any(r["scenario"] != name or None in r.values() for r in done):
        raise SystemExit(f"{raw}: rows are not the first simulations of this shard")
    if any((int(r["data_seed"]), int(r["replicate_seed"])) != seeds[int(r["sim"])] for r in done):
        raise SystemExit(f"{raw}: seeds differ from the scenario stream")
    return len(done)


def run_scenario(name: str, sims: int, B: int, n_patients: int, seed: int, out_dir: Path, shard: int = 0,
                 shards: int = 1, resume: bool = False) -> None:
    idx, hyp, truth, build, t = SCENARIOS[name]
    tag = f"_shard{shard:02d}of{shards:02d}"
    raw, meta = out_dir / f"{name}_raw{tag}.csv", out_dir / f"{name}_meta{tag}.json"
    if meta.exists() or (raw.exists() and not resume):
        raise SystemExit(f"{raw} or {meta} exists; refusing to overwrite")
    seeds = sim_seeds(name, sims, seed)
    todo = list(range(shard, sims, shards))                    # interleaved shards balance the load
    already = resume_point(raw, name, todo, seeds) if raw.exists() else 0
    started_utc, t0 = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.time()
    with raw.open("a" if already else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not already:
            w.writeheader()
        for k in todo[already:]:
            data_seed, rep_seed = seeds[k]
            rows = base.make_rows(np.random.default_rng(data_seed), n_patients)
            scores = build(rows)
            if hyp == "H1":
                point, d = base.h1_deltas(rows, scores[0], t, B, rep_seed)
                acc = float((scores[0] <= t).mean())
                tied = 0.0
            else:
                point, d = base.h2_deltas(rows, scores[0], scores[1], B, rep_seed)
                acc = 0.8
                tied = float(np.mean([(lambda u: (u[2][u[1]] > 1).mean())(np.unique(s, return_inverse=True, return_counts=True))
                                      for s in scores]))         # share of rows in tie groups, averaged over both scores
            dec = sel.one_sided_decision(d)
            iv = sel.two_sided_interval(d)
            w.writerow({"scenario": name, "sim": k, "data_seed": data_seed, "replicate_seed": rep_seed, "rows": rows["d1"].size,
                        "patients": int(np.unique(rows["pid"]).size), "d1_rate": round(float(rows["d1"].mean()), 6),
                        "accepted_share": round(acc, 6), "tied_rows_share": tied, "point_delta": repr(float(point)),
                        "upper_bound_U": repr(dec["upper_bound"]), "k_upper": dec["k_upper"], "n_nonneg": dec["n_nonneg"],
                        "reject": int(dec["reject"]), "p2_rule_reject": int((1 + dec["n_nonneg"]) / (B + 1) < 0.05),
                        "two_sided_low": repr(iv["low"]), "two_sided_high": repr(iv["high"]),
                        "replicates_inf": int(np.isinf(d).sum())})
            f.flush()
    meta.write_text(json.dumps({
        "scenario": name, "hypothesis": hyp, "true_contrast": truth, "simulations": sims, "replicates": B,
        "shard": shard, "shards": shards, "simulations_in_shard": len(range(shard, sims, shards)),
        "patients_per_dataset": n_patients, "master_seed": seed, "scenario_stream": f"numpy.random.default_rng([{seed}, {idx}])",
        "alpha_one_sided": 0.05, "h1_threshold": t, "h2_coverage": 0.8 if hyp == "H2" else None,
        "decision": "reject iff U = delta*_(ceil(0.95 (B + 1))) < 0 (src.eval.selective.one_sided_decision)",
        "command": " ".join(sys.argv), "started_utc": started_utc, "resumed_after_rows": already,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
        "environment": environment(),
        "code_sha256": {p: sha256(REPO / p) for p in ("src/eval/selective.py", "src/eval/simulate_selective_inference.py",
                                                       "src/eval/simulate_decision_rule.py")},
        "raw_csv_sha256": sha256(raw)}, indent=2), encoding="utf-8")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def merge_shards(out_dir: Path, name: str) -> tuple[list[dict], dict, list[str]] | None:
    """All shards of a scenario: hashes checked against their metadata, each simulation index present exactly once, the
    settings equal across shards, and the seeds equal to the scenario stream. Writes <name>_raw.csv sorted by simulation."""
    metas = sorted(out_dir.glob(f"{name}_meta_shard*.json"))
    if not metas:
        return None
    rows, ms, problems = [], [], []
    for mp in metas:
        m = json.loads(mp.read_text(encoding="utf-8"))
        raw = out_dir / mp.name.replace("_meta_", "_raw_").replace(".json", ".csv")
        if sha256(raw) != m["raw_csv_sha256"]:
            problems.append(f"{raw.name}: hash differs from its metadata")
        with raw.open(encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
        ms.append(m)
    keys = ("simulations", "replicates", "patients_per_dataset", "master_seed", "shards")
    if any(any(m[k] != ms[0][k] for k in keys) for m in ms) or len(ms) != ms[0]["shards"]:
        problems.append("shards disagree on settings or a shard is missing")
    rows.sort(key=lambda x: int(x["sim"]))
    if [int(x["sim"]) for x in rows] != list(range(ms[0]["simulations"])):
        problems.append("simulation indices are not exactly 0..N-1 once each")
    seeds = sim_seeds(name, ms[0]["simulations"], ms[0]["master_seed"])
    if any((int(x["data_seed"]), int(x["replicate_seed"])) != seeds[int(x["sim"])] for x in rows):
        problems.append("seeds differ from the scenario stream")
    merged = out_dir / f"{name}_raw.csv"
    if not merged.exists():
        with merged.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows, {**ms[0], "seconds": max(m["seconds"] for m in ms), "merged_raw_csv_sha256": sha256(merged),
                  "shard_raw_csv_sha256": {m["shard"]: m["raw_csv_sha256"] for m in ms},
                  "environments": sorted({json.dumps(m["environment"], sort_keys=True) for m in ms}),
                  "code_sha256_sets": sorted({json.dumps(m["code_sha256"], sort_keys=True) for m in ms})}, problems


def compare_with_aborted(out_dir: Path, name: str, rows: list[dict]) -> dict:
    """Rows of the stopped first launch (if kept) must equal the new rows for the same simulation index."""
    old = out_dir / "aborted_first_launch_unsharded" / f"{name}_raw.csv"
    if not old.exists():
        return {"compared": 0}
    by_sim = {int(x["sim"]): x for x in rows}
    with old.open(encoding="utf-8") as f:
        prev = list(csv.DictReader(f))
    diff = [int(x["sim"]) for x in prev if by_sim.get(int(x["sim"])) != x]
    return {"compared": len(prev), "identical": len(prev) - len(diff), "different_sims": diff[:20]}


def summarize(out_dir: Path) -> None:
    rows = []
    for name in SCENARIOS:
        got = merge_shards(out_dir, name)
        if got is None:
            rows.append({"scenario": name, "status": "MISSING"})
            continue
        r, m, problems = got
        n, k = len(r), sum(int(x["reject"]) for x in r)
        lo, hi = wilson(k, n)
        rows.append({"scenario": name, "status": "COMPLETE" if n == m["simulations"] and not problems else "INCOMPLETE",
                     "problems": problems, "shards": m["shards"], "first_launch_rows_reproduced": compare_with_aborted(out_dir, name, r),
                     "environments": m["environments"], "code_sha256_sets": m["code_sha256_sets"],
                     "hypothesis": m["hypothesis"], "true_contrast": m["true_contrast"], "simulations": n,
                     "replicates": m["replicates"], "rejections": k, "rejection_rate": k / n,
                     "wilson_95_low": lo, "wilson_95_high": hi,
                     "p2_rule_disagreements": sum(int(x["reject"]) != int(x["p2_rule_reject"]) for x in r),
                     "rejected_with_two_sided_interval_containing_0":
                         sum(int(x["reject"]) and float(x["two_sided_high"]) >= 0 for x in r),
                     "sims_with_inf_replicates": sum(int(x["replicates_inf"]) > 0 for x in r),
                     "mean_d1_rate": float(np.mean([float(x["d1_rate"]) for x in r])), "seconds": m["seconds"],
                     "raw_csv": f"{name}_raw.csv", "raw_csv_sha256": m["merged_raw_csv_sha256"],
                     "shard_raw_csv_sha256": m["shard_raw_csv_sha256"]})
    (out_dir / "summary.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                      "scenarios": rows}, indent=2), encoding="utf-8")
    lines = ["| Scenario | True Δ | Sims | B | Rejections | Rate (Wilson 95%) | P2-rule disagreements |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for x in rows:
        if x["status"] == "MISSING":
            lines.append(f"| {x['scenario']} | — | MISSING | | | | |")
            continue
        lines.append(f"| {x['scenario']} | {x['true_contrast']} | {x['simulations']} | {x['replicates']} | {x['rejections']} | "
                     f"{x['rejection_rate']:.3f} ({x['wilson_95_low']:.3f}–{x['wilson_95_high']:.3f}) | {x['p2_rule_disagreements']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=list(SCENARIOS))
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--replicates", type=int, default=10000)
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted shard file after checking its rows")
    ap.add_argument("--summarize", type=Path)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.summarize:
        summarize(a.summarize)
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= a.shard < a.shards:
        raise SystemExit("--shard must be in [0, --shards)")
    run_scenario(a.scenario, a.sims, a.replicates, a.patients, a.seed, a.out_dir, a.shard, a.shards, a.resume)


if __name__ == "__main__":
    main()
