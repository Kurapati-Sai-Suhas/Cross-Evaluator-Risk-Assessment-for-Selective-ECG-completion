"""M5 evaluation (protocol v0.2, pinned): primary H1 -> H2, secondary descriptive contrasts and the descriptive endpoints
S-A to S-D on ONE aligned table of record-mask rows. Pure computation: this module opens no file and knows nothing about
roles; the fold-10 pathway (src.eval.fold10_pathway) decides which rows it may receive.

Input table columns: row_id, patient_id, mask_id, n_observed, the M4a offline-label columns needed by S-A to S-D (D1, y_*,
z_full_*, z_rec_*, D1_miss_*, D1_false_*) and one calibrated-score column per method. Higher score = riskier; a row is released
when its score <= the method's locked threshold (or by exact coverage, primary_analysis.H2.estimand). M0_accept_all releases
every row. One replicate stream (uncertainty.replicate_stream) is shared by every quantity.

D1 is disagreement between the frozen E1's decisions on the complete and on the reconstructed ECG; it is not clinical harm,
clinical truth, or a safety or risk guarantee. The decision rule is a bootstrap-based empirical rule.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.eval import descriptive_endpoints as de
from src.eval import primary_decision as pdec
from src.eval import selective as sel

GATES = ("gate_A", "gate_B", "M3_signal_quality_only", "M2_uncertainty_only")
M0 = "M0_accept_all"
LOCKED_CONTRASTS = (("gate_A", M0), ("gate_B", "M2_uncertainty_only"), ("gate_B", "gate_A"), ("gate_B", "M3_signal_quality_only"))


def _nan(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _interval(v: np.ndarray) -> dict[str, Any]:
    return sel.two_sided_interval(np.asarray(v, dtype=np.float64), 0.95)


def gate_a_identity(scores_a: np.ndarray, lead_set: np.ndarray) -> None:
    """gate_A is a function of the lead set: its calibrated score must be constant within every lead set (protocol S-A)."""
    per = pd.Series(np.asarray(scores_a)).groupby(np.asarray(lead_set)).nunique()
    if (per > 1).any():
        raise RuntimeError("gate_A score is not constant within a lead set: implementation error, analysis stopped")


def aurc(scores: np.ndarray, d1: np.ndarray) -> float:
    """Area under the selective-risk vs coverage curve, releasing whole tie groups in ascending score order (point estimate)."""
    s, d = np.asarray(scores, dtype=np.float64), np.asarray(d1, dtype=np.float64)
    uniq, inv = np.unique(s, return_inverse=True)
    n_g = np.bincount(inv)
    d_g = np.bincount(inv, weights=d)
    cov = np.cumsum(n_g) / s.size
    risk = np.cumsum(d_g) / np.cumsum(n_g)
    return float(np.sum(np.diff(np.r_[0.0, cov]) * risk))


def reliability(cal: np.ndarray, d1: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Brier score and equal-width reliability bins of a calibrated score (descriptive)."""
    p, d = np.asarray(cal, dtype=np.float64), np.asarray(d1, dtype=np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = [{"bin": b, "n": int((idx == b).sum()), "mean_score": _nan(float(p[idx == b].mean())) if (idx == b).any() else None,
             "d1_rate": _nan(float(d[idx == b].mean())) if (idx == b).any() else None} for b in range(bins)]
    return {"brier": float(np.mean((p - d) ** 2)), "bins": rows}


def evaluate(table: pd.DataFrame, thresholds: dict[str, float], protocol: dict[str, Any], replicates: int | None = None,
             chunk: int = 500, outputs: Sequence[str] = de.OUTPUTS) -> dict[str, Any]:
    pa, unc = protocol["primary_analysis"], protocol["uncertainty"]
    B = int(replicates if replicates is not None else unc["replicates"])
    seed, alpha, c = int(protocol["seed"]), float(pa["alpha_one_sided"]), 0.80
    sec_cov = [float(x) for x in protocol["threshold"]["secondary_coverages"]]
    if not table["row_id"].is_unique:
        raise ValueError("duplicate row_id")
    arr = de.offline_arrays(table, outputs)
    d = table["D1"].to_numpy(np.int64)
    pid, lead_set, n_obs = table["patient_id"].to_numpy(), table["mask_id"].to_numpy(), table["n_observed"].to_numpy()
    S = {m: table[m].to_numpy(np.float64) for m in GATES}
    if not all(np.isfinite(v).all() for v in S.values()):
        raise ValueError("non-finite calibrated score")
    gate_a_identity(S["gate_A"], lead_set)
    n = len(table)
    w1 = np.ones(n)
    locked = {m: sel.accept_at_threshold(S[m], thresholds[m]) for m in GATES}
    locked[M0] = w1.copy()
    c80 = {m: sel.accept_at_coverage(S[m], w1, c) for m in GATES}
    c80[M0] = w1.copy()
    risk = lambda a: sel.selective_risk(a, w1, d)  # noqa: E731

    # ---- point estimates ----
    R_B, R_0 = risk(locked["gate_B"]), risk(locked[M0])
    h1_point = R_B - R_0 if not math.isnan(R_B) else math.nan
    h2_point = risk(c80["gate_B"]) - risk(c80["gate_A"])
    point = {"locked": {m: {"coverage": float(locked[m].mean()), "accepted_rows": float(locked[m].sum()), "risk": _nan(risk(locked[m]))}
                        for m in locked},
             "coverage_0.80": {m: {"risk": _nan(risk(c80[m]))} for m in c80},
             "secondary_coverages": {str(cc): {m: _nan(risk(sel.accept_at_coverage(S[m], w1, cc))) for m in GATES} for cc in sec_cov}}

    # ---- one replicate stream for everything ----
    reps: dict[str, list[np.ndarray]] = {k: [] for k in ["h1", "h2", "R_B", "R_0", "R_B80", "R_A80"]
                                         + [f"{a}_vs_{b}" for a, b in LOCKED_CONTRASTS]
                                         + [f"Cmean_{m}" for m in GATES] + [f"Cpair_{m}" for m in GATES]}
    sbcd_reps: dict[str, list[dict[str, float]]] = {f"{op}_{m}": [] for op in ("locked", "c80") for m in [*GATES, M0]}
    for patients, M in sel.patient_multiplicities(pid, B, seed, chunk):
        W = sel.row_weights(pid, patients, M)
        rb = sel.selective_risk_batch(locked["gate_B"][None, :], W, d)
        r0 = sel.selective_risk_batch(locked[M0][None, :], W, d)
        reps["R_B"].append(rb)
        reps["R_0"].append(r0)
        reps["h1"].append(sel.contrast(rb, r0))
        A_B, A_A = sel.accept_at_coverage_batch(S["gate_B"], W, c), sel.accept_at_coverage_batch(S["gate_A"], W, c)
        rb80, ra80 = sel.selective_risk_batch(A_B, W, d), sel.selective_risk_batch(A_A, W, d)
        reps["R_B80"].append(rb80)
        reps["R_A80"].append(ra80)
        reps["h2"].append(sel.contrast(rb80, ra80))
        for a, b in LOCKED_CONTRASTS:
            reps[f"{a}_vs_{b}"].append(sel.contrast(sel.selective_risk_batch(locked[a][None, :], W, d),
                                                    sel.selective_risk_batch(locked[b][None, :], W, d)))
        for m in GATES:
            sa = de.lead_set_concordance(S[m], d, lead_set, W)
            reps[f"Cmean_{m}"].append(sa["c_within_stratum_mean"])
            reps[f"Cpair_{m}"].append(sa["c_within_pair_weighted"])
        A80 = {m: sel.accept_at_coverage_batch(S[m], W, c) for m in GATES}
        for b in range(W.shape[0]):
            w = W[b]
            for m in [*GATES, M0]:
                for op, a in (("locked", locked[m]), ("c80", A80[m][b] if m != M0 else w1)):
                    sbcd_reps[f"{op}_{m}"].append({**{f"SB_{k}": v for k, v in de.bbb_positive_coverage(a, w, arr["z_full"], arr["labels"], outputs).items()},
                                                   **{f"SC_{k}": v for k, v in de.bbb_call_preservation(a, w, arr["z_full"], arr["z_rec"], outputs).items()},
                                                   **{f"SD_{k}": v for k, v in de.label_agreement_flips(a, w, arr["z_full"], arr["z_rec"], arr["labels"], outputs).items()}})
    R = {k: np.concatenate(v) for k, v in reps.items()}

    decision = pdec.fixed_sequence(None if math.isnan(h1_point) else h1_point, R["h1"], R["h2"], alpha)
    decision["H1"].update({"R_B": _nan(R_B), "R_0": _nan(R_0), "realised_coverage_gate_B": float(locked["gate_B"].mean()),
                           "two_sided_95": _interval(R["h1"]), "threshold_gate_B": pdec.threshold_record(thresholds["gate_B"])})
    decision["H2"].update({"point": _nan(h2_point), "R_B_0.80": _nan(risk(c80["gate_B"])), "R_A_0.80": _nan(risk(c80["gate_A"])),
                           "two_sided_95": _interval(R["h2"])})
    if not decision["H2"]["tested"]:
        decision["H2"]["note"] = "not tested (H1 not rejected); reported descriptively only, no inferential claim"
        decision["H2"]["descriptive_one_sided_bound"] = sel.one_sided_decision(R["h2"], alpha)["upper_bound"]

    secondary = {f"{a}_vs_{b}_locked": {"point": _nan(risk(locked[a]) - risk(locked[b])) if not (math.isnan(risk(locked[a])) or math.isnan(risk(locked[b]))) else None,
                                        "two_sided_95": _interval(R[f"{a}_vs_{b}"])} for a, b in LOCKED_CONTRASTS}
    s_a = {}
    for m in GATES:
        pt = de.lead_set_concordance(S[m], d, lead_set)
        s_a[m] = {"C_mean_headline": _nan(float(pt["c_within_stratum_mean"][0])), "C_mean_two_sided_95": _interval(R[f"Cmean_{m}"]),
                  "C_pair_sensitivity": _nan(float(pt["c_within_pair_weighted"][0])), "C_pair_two_sided_95": _interval(R[f"Cpair_{m}"]),
                  "C_unstratified_reference": _nan(float(pt["c_unstratified"][0])),
                  "eligible_lead_sets": int(pt["eligible_lead_sets"][0]), "lead_sets_present": int(pt["lead_sets_present"]),
                  "lead_sets_only_d1": pt["lead_sets_only_d1"], "lead_sets_only_non_d1": pt["lead_sets_only_non_d1"],
                  "comparable_pairs_within": pt["n_comparable_pairs_within"]}
    if s_a["gate_A"]["C_mean_headline"] is not None and not (s_a["gate_A"]["C_mean_headline"] == s_a["gate_A"]["C_pair_sensitivity"] == 0.5):
        raise RuntimeError("gate_A S-A differs from 0.5: implementation error, analysis stopped")
    sbcd = {}
    for key, lst in sbcd_reps.items():
        op, m = key.split("_", 1)
        a = locked[m] if op == "locked" else c80[m]
        pt = {**{f"SB_{k}": v for k, v in de.bbb_positive_coverage(a, w1, arr["z_full"], arr["labels"], outputs).items()},
              **{f"SC_{k}": v for k, v in de.bbb_call_preservation(a, w1, arr["z_full"], arr["z_rec"], outputs).items()},
              **{f"SD_{k}": v for k, v in de.label_agreement_flips(a, w1, arr["z_full"], arr["z_rec"], arr["labels"], outputs).items()}}
        df = pd.DataFrame(lst)
        sbcd[key] = {k: {"point": _nan(v), "two_sided_95": _interval(df[k].to_numpy())} for k, v in pt.items()}
    by_leads = {int(k): {m: {"rows": int((n_obs == k).sum()), "coverage": float(locked[m][n_obs == k].mean()),
                             "risk": _nan(sel.selective_risk(locked[m][n_obs == k], w1[n_obs == k], d[n_obs == k]))}
                         for m in [*GATES, M0]} for k in sorted(set(n_obs.tolist()))}
    return {"rows": n, "patients": int(np.unique(pid).size), "d1_rows": int(d.sum()), "d1_rate": float(d.mean()),
            "replicates": B, "seed": seed, "alpha_one_sided": alpha,
            "thresholds": {m: pdec.threshold_record(t) for m, t in thresholds.items()},
            "primary": decision, "point": point, "secondary_locked_contrasts": secondary,
            "S_A": s_a, "S_B_S_C_S_D": sbcd, "by_number_of_observed_leads": by_leads,
            "aurc": {m: aurc(S[m], d) for m in GATES}, "calibration": {m: reliability(S[m], d) for m in GATES},
            "replicate_arrays": {k: R[k] for k in ("h1", "h2")}}
