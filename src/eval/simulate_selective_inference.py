"""Synthetic check of the P3 decision rule (one-sided percentile upper bound, paired patient-cluster bootstrap) for H1 and H2.

Synthetic data only: clustered rows (patients -> records -> 8 masks) with lead-set strata, about 4% D1 and record and patient
random effects. Null scenarios have a population contrast of exactly 0 by construction (H1: a score independent of D1 at a
fixed threshold; H2: two exchangeable scores at exact 80% coverage). The rejection rate under the null estimates the size
of the rule; alternative scenarios show it can reject. This is supporting evidence, not a proof: validity rests on the
first-order consistency of the cluster bootstrap for smooth functionals (Field & Welsh 2007) and on CI inversion.

    python -m src.eval.simulate_selective_inference [--sims 400] [--replicates 1000]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.eval import selective as sel

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "reports" / "m4b0d_bootstrap_size_simulation.json"


def make_rows(rng: np.random.Generator, n_patients: int) -> dict[str, np.ndarray]:
    n_rec = 1 + rng.poisson(0.3, n_patients)
    pid = np.repeat(np.arange(n_patients), n_rec * 8)
    rec = np.repeat(np.arange(n_rec.sum()), 8)
    n = pid.size
    severity = rng.random(40)                                  # 40 lead-set strata
    stratum = rng.integers(0, 40, n)
    q1, q2 = rng.normal(size=n), rng.normal(size=n)
    logit = (-4.8 + 2.0 * severity[stratum] + rng.normal(0, 0.6, n_patients)[pid] + rng.normal(0, 0.5, rec.max() + 1)[rec]
             + 0.5 * q1 + 0.5 * q2)
    d1 = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float64)
    return {"pid": pid, "d1": d1, "q1": q1, "q2": q2, "noise": rng.normal(size=n), "noise2": rng.normal(size=n)}


def h1_deltas(rows: dict, score: np.ndarray, t: float, B: int, seed: int) -> tuple[float, np.ndarray]:
    a = sel.accept_at_threshold(score, t)
    point = sel.contrast(np.array([sel.selective_risk(a, 1.0, rows["d1"])]), np.array([rows["d1"].mean()]))[0]
    out = []
    for patients, M in sel.patient_multiplicities(rows["pid"], B, seed, chunk=250):
        W = sel.row_weights(rows["pid"], patients, M)
        out.append(sel.contrast(sel.selective_risk_batch(np.broadcast_to(a, W.shape), W, rows["d1"]),
                                sel.selective_risk_batch(np.ones_like(W), W, rows["d1"])))
    return float(point), np.concatenate(out)


def h2_deltas(rows: dict, s_b: np.ndarray, s_a: np.ndarray, B: int, seed: int, c: float = 0.8) -> tuple[float, np.ndarray]:
    one = np.ones(rows["d1"].size)
    point = sel.selective_risk(sel.accept_at_coverage(s_b, one, c), one, rows["d1"]) - \
        sel.selective_risk(sel.accept_at_coverage(s_a, one, c), one, rows["d1"])
    out = []
    for patients, M in sel.patient_multiplicities(rows["pid"], B, seed, chunk=250):
        W = sel.row_weights(rows["pid"], patients, M)
        out.append(sel.contrast(sel.selective_risk_batch(sel.accept_at_coverage_batch(s_b, W, c), W, rows["d1"]),
                                sel.selective_risk_batch(sel.accept_at_coverage_batch(s_a, W, c), W, rows["d1"])))
    return float(point), np.concatenate(out)


SCENARIOS = {
    # name: (hypothesis, true contrast, builder of the score(s))
    "H1_null_uninformative_gate": ("H1", "0", lambda r: (r["noise"],)),
    "H2_null_exchangeable_scores": ("H2", "0", lambda r: (r["q2"] + r["noise"], r["q1"] + r["noise2"])),
    "H1_alternative_informative_gate": ("H1", "< 0", lambda r: (r["q1"] + r["q2"] + r["noise"],)),
    "H2_alternative_better_ranking": ("H2", "< 0", lambda r: (r["q1"] + r["q2"] + 0.5 * r["noise"], r["q1"] + r["noise2"])),
}


def run(sims: int, B: int, n_patients: int, seed: int) -> dict:
    master = np.random.default_rng(seed)
    t_h1 = {"H1_null_uninformative_gate": norm.ppf(0.8),             # population coverage 0.80 for an N(0, 1) score
            "H1_alternative_informative_gate": norm.ppf(0.8) * np.sqrt(3)}
    results = {}
    for name, (hyp, truth, build) in SCENARIOS.items():
        n_sims = sims if truth == "0" else max(sims // 4, 50)
        rej = ci_contains_zero_when_rejected = 0
        started = time.time()
        for k in range(n_sims):
            rows = make_rows(np.random.default_rng(master.integers(2**63)), n_patients)
            scores = build(rows)
            rep_seed = int(master.integers(2**31))
            if hyp == "H1":
                _, d = h1_deltas(rows, scores[0], t_h1[name], B, rep_seed)
            else:
                _, d = h2_deltas(rows, scores[0], scores[1], B, rep_seed)
            dec = sel.one_sided_decision(d)
            old_rule = (1 + dec["n_nonneg"]) / (B + 1) < 0.05
            assert old_rule == dec["reject"], "P2 and P3 rules disagree"   # identical decisions for B with (B+1)*0.05 non-integer
            if dec["reject"]:
                rej += 1
                iv = sel.two_sided_interval(d)
                ci_contains_zero_when_rejected += int(iv["high"] >= 0)
        p = rej / n_sims
        results[name] = {"hypothesis": hyp, "true_contrast": truth, "simulations": n_sims, "rejections": rej,
                         "rejection_rate": p, "monte_carlo_se": float(np.sqrt(p * (1 - p) / n_sims)),
                         "rejected_with_two_sided_interval_containing_0": ci_contains_zero_when_rejected,
                         "seconds": round(time.time() - started, 1)}
        print(name, results[name], flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    res = run(args.sims, args.replicates, args.patients, args.seed)
    OUT.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "milestone": "M4b-0d",
        "purpose": "size and power of the P3 decision rule on synthetic clustered data (no project data used)",
        "settings": {"sims_null": args.sims, "replicates": args.replicates, "patients": args.patients, "seed": args.seed,
                     "alpha_one_sided": 0.05, "coverage_H2": 0.8, "rows_per_record": 8, "lead_set_strata": 40,
                     "decision": "U = delta*_(ceil(0.95 (B + 1))) < 0"},
        "note": ("B = 1000 here (k = 951, reject iff n_nonneg <= 49) keeps (B + 1) * 0.05 non-integer as for the protocol's "
                 "B = 10000, so the P2 p-rule and the P3 interval rule give identical decisions (asserted in every simulation)"),
        "results": res}, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
