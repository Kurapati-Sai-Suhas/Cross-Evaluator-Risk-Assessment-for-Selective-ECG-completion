"""Prespecified descriptive endpoints S-A to S-D of protocol v0.2 (proposal P3/P4: configs/protocol_v0_2.yaml,
descriptive_endpoints). Secondary, descriptive outputs only: they are NOT gate features, NOT models and NOT hypothesis tests.

Every input except the calibrated score and the lead set is OFFLINE evaluation data (E1 decisions on the complete ECG and
on the reconstruction, PTB-XL label bits). It is read from the M4a offline-label schema at evaluation time and never enters
a gate. Reference implementation of formulas only, exercised by synthetic tests; the M5 pipeline is NOT implemented.

Notation: rows are record-mask rows; ``a`` = acceptance weights (selective.accept_at_threshold or accept_at_coverage),
``w`` = row weights (1 for point estimates, patient multiplicities in a bootstrap replicate); ``z_full``, ``z_rec``,
``labels`` are (rows x outputs) 0/1 arrays in the order of OUTPUTS. A zero denominator gives NaN ("not estimable"), never 0.
D1 and S-D measure frozen-E1 decision disagreement and dataset-label agreement; neither is clinical harm or clinical truth.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

OUTPUTS = ("CLBBB", "CRBBB")


def required_offline_columns(outputs: Sequence[str] = OUTPUTS) -> list[str]:
    """Offline-label columns S-A to S-D need (all present in the M4a schema; the fold-10 pathway must keep them)."""
    cols = ["row_id", "D1"]
    for y in outputs:
        cols += [f"y_{y}", f"z_full_{y}", f"z_rec_{y}", f"D1_miss_{y}", f"D1_false_{y}"]
    return cols


def offline_arrays(labels: pd.DataFrame, outputs: Sequence[str] = OUTPUTS) -> dict[str, np.ndarray]:
    """(rows x outputs) arrays from an offline-label table, after checking the columns and the D1 identities."""
    missing = [c for c in required_offline_columns(outputs) if c not in labels.columns]
    if missing:
        raise ValueError(f"offline labels lack {missing}")
    get = lambda p: np.column_stack([labels[f"{p}_{y}"].to_numpy(np.int64) for y in outputs])  # noqa: E731
    zf, zr, lab = get("z_full"), get("z_rec"), get("y")
    if not (np.array_equal(get("D1_miss"), zf & (1 - zr)) and np.array_equal(get("D1_false"), (1 - zf) & zr)
            and np.array_equal(labels["D1"].to_numpy(np.int64), (zf != zr).any(axis=1).astype(np.int64))):
        raise ValueError("offline labels violate the D1 definitions")
    return {"z_full": zf, "z_rec": zr, "labels": lab, "d1": labels["D1"].to_numpy(np.int64)}


def align_rows(keys: pd.DataFrame, labels: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One table in the order of ``keys``, joined one-to-one on row_id. Refuses duplicates, missing or extra rows, so an
    endpoint can never pair a score with another row's decisions or lead set."""
    for name, t in (("keys", keys), ("labels", labels), ("scores", scores)):
        if not t["row_id"].is_unique:
            raise ValueError(f"{name}: duplicate row_id")
    ids = set(keys["row_id"])
    for name, t in (("labels", labels), ("scores", scores)):
        if set(t["row_id"]) != ids:
            raise ValueError(f"{name}: row_id set differs from keys ({len(set(t['row_id']) ^ ids)} rows)")
    out = keys.merge(labels, on="row_id", how="left", validate="one_to_one", suffixes=("", "_lab"))
    out = out.merge(scores, on="row_id", how="left", validate="one_to_one", suffixes=("", "_score"))
    if not out["row_id"].equals(keys["row_id"]):
        raise ValueError("row order changed during alignment")
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


# ------------------------------------------------------------------------------------------------------------------
# S-A: within-lead-set concordance
# ------------------------------------------------------------------------------------------------------------------
def within_stratum_concordance(scores: np.ndarray, d1: np.ndarray, strata: np.ndarray,
                               W: np.ndarray | None = None) -> dict[str, Any]:
    """Within-stratum concordance with K(u, v) = 1[u > v] + 0.5 1[u = v] for (D1 = 1, D1 = 0) pairs of one stratum.

    Per stratum s: c_s = sum_{i in P_s} sum_{j in N_s} w_i w_j K(s_i, s_j) / (W_P,s W_N,s), defined (eligible) only when
    W_P,s > 0 and W_N,s > 0. Two aggregates, per replicate row of ``W`` (default: one row of ones):
      * ``c``       pair-weighted: sum_s num_s / sum_s W_P,s W_N,s  (= the mean of c_s weighted by W_P,s W_N,s);
      * ``c_macro`` equal-weighted: the plain mean of c_s over eligible strata.
    Both are NaN when no stratum is eligible. Ineligible strata (one class or no rows) enter neither aggregate.
    """
    s = np.asarray(scores, dtype=np.float64)
    d = np.asarray(d1).astype(bool)
    st = np.asarray(strata)
    if not (s.shape == d.shape == st.shape) or not np.isfinite(s).all():
        raise ValueError("scores, d1 and strata must be equal-length 1-D arrays with finite scores")
    W = np.ones((1, s.size)) if W is None else np.atleast_2d(np.asarray(W, dtype=np.float64))
    _, st_code = np.unique(st, return_inverse=True)
    order = np.lexsort((s, st_code))
    s_o, st_o, d_o = s[order], st_code[order], d[order]
    n = s.size
    new_stratum = np.r_[True, st_o[1:] != st_o[:-1]]
    new_tie = new_stratum | np.r_[True, s_o[1:] != s_o[:-1]]
    stratum_start = np.maximum.accumulate(np.where(new_stratum, np.arange(n), 0))
    tie_start = np.maximum.accumulate(np.where(new_tie, np.arange(n), 0))
    tie_id = np.cumsum(new_tie) - 1
    tie_end = np.r_[np.flatnonzero(new_tie)[1:], n][tie_id]
    stratum_id = np.cumsum(new_stratum) - 1
    S = int(stratum_id[-1]) + 1 if n else 0
    Wo = W[:, order]
    negw = Wo * ~d_o
    C = np.concatenate([np.zeros((Wo.shape[0], 1)), np.cumsum(negw, axis=1)], axis=1)
    pos = np.flatnonzero(d_o)
    less = C[:, tie_start[pos]] - C[:, stratum_start[pos]]
    equal = C[:, tie_end[pos]] - C[:, tie_start[pos]]
    contrib = Wo[:, pos] * (less + 0.5 * equal)
    onehot = np.zeros((pos.size, S))
    onehot[np.arange(pos.size), stratum_id[pos]] = 1.0
    num_s = contrib @ onehot                                   # replicate x stratum
    starts = np.flatnonzero(new_stratum)
    wp = np.add.reduceat(Wo * d_o, starts, axis=1)
    wn = np.add.reduceat(negw, starts, axis=1)
    den_s = wp * wn
    eligible = den_s > 0
    c_s = _ratio(num_s, den_s)
    n_eligible = eligible.sum(axis=1)
    with np.errstate(invalid="ignore"):
        macro = np.where(n_eligible > 0, np.nansum(np.where(eligible, c_s, 0.0), axis=1) / np.maximum(n_eligible, 1), np.nan)
    out = {"c": _ratio(num_s.sum(axis=1), den_s.sum(axis=1)), "c_macro": macro, "numerator": num_s.sum(axis=1),
           "denominator": den_s.sum(axis=1), "eligible_strata": n_eligible, "strata_present": S,
           "per_stratum_c": c_s}
    p1 = np.add.reduceat(d_o.astype(np.int64), starts)
    n1 = np.add.reduceat((~d_o).astype(np.int64), starts)
    out["n_strata_contributing"] = int(((p1 > 0) & (n1 > 0)).sum())
    out["n_strata_only_d1"] = int(((p1 > 0) & (n1 == 0)).sum())
    out["n_strata_only_non_d1"] = int(((p1 == 0) & (n1 > 0)).sum())
    out["n_comparable_pairs"] = int((p1 * n1).sum())
    return out


def lead_set_concordance(scores: np.ndarray, d1: np.ndarray, lead_set: np.ndarray,
                         W: np.ndarray | None = None) -> dict[str, Any]:
    """S-A: both within-lead-set aggregates, the unstratified concordance for reference, and eligibility counts."""
    within = within_stratum_concordance(scores, d1, lead_set, W)
    overall = within_stratum_concordance(scores, d1, np.zeros(len(scores), dtype=np.int64), W)
    return {"c_within_stratum_mean": within["c_macro"], "c_within_pair_weighted": within["c"],
            "c_unstratified": overall["c"], "eligible_lead_sets": within["eligible_strata"],
            "lead_sets_present": within["strata_present"], "lead_sets_only_d1": within["n_strata_only_d1"],
            "lead_sets_only_non_d1": within["n_strata_only_non_d1"], "n_comparable_pairs_within": within["n_comparable_pairs"],
            "n_comparable_pairs_unstratified": overall["n_comparable_pairs"]}


# ------------------------------------------------------------------------------------------------------------------
# S-B, S-C, S-D at one operating point (a) and one weight vector (w)
# ------------------------------------------------------------------------------------------------------------------
def bbb_positive_coverage(a: np.ndarray, w: np.ndarray, z_full: np.ndarray, labels: np.ndarray,
                          outputs: Sequence[str] = OUTPUTS) -> dict[str, float]:
    """S-B: share released among rows E1 calls BBB on x, and among PTB-XL label-positive rows.

    Per output y: Cov_y = sum a w z_y / sum w z_y. 'any' uses max_y z_y, so a row positive for both outputs counts once in
    each per-output denominator and once in the 'any' denominator; 'both' reports those rows separately.
    """
    a, w = np.asarray(a, dtype=np.float64), np.asarray(w, dtype=np.float64)
    out = {"coverage_overall": float(_ratio((a * w).sum(), w.sum()))}
    for name, Z in (("e1_called", z_full), ("label_positive", labels)):
        Z = np.asarray(Z, dtype=np.float64)
        cols = [Z[:, j] for j in range(len(outputs))] + [Z.max(axis=1), Z.min(axis=1)]
        for y, z in zip(list(outputs) + ["any", "both"], cols):
            out[f"cov_{name}_{y}"] = float(_ratio((a * w * z).sum(), (w * z).sum()))
            out[f"num_{name}_{y}"] = float((a * w * z).sum())
            out[f"den_{name}_{y}"] = float((w * z).sum())
    return out


def bbb_call_preservation(a: np.ndarray, w: np.ndarray, z_full: np.ndarray, z_rec: np.ndarray,
                          outputs: Sequence[str] = OUTPUTS) -> dict[str, float]:
    """S-C: among released rows E1 calls BBB on x, the share whose call(s) E1 still makes on x_hat.

    Per output y: Pres_y = sum a w zf_y zr_y / sum a w zf_y. Any: a row called for at least one output is preserved only
    if every output called on x is still called on x_hat. Reported separately, never only inside 'any': rows called for
    BOTH outputs on x (kept both, lost only one, lost both) and released 'type switches' (one call lost while the other
    output's call is gained on x_hat), by direction. A gained call alone is not a preservation failure (directional D1).
    """
    a, w = np.asarray(a, dtype=np.float64), np.asarray(w, dtype=np.float64)
    zf, zr = np.asarray(z_full, dtype=np.int64), np.asarray(z_rec, dtype=np.int64)
    if len(outputs) != 2:
        raise ValueError("the both-output and type-switch breakdowns are defined for two outputs")
    aw = a * w
    out = {}
    for j, y in enumerate(outputs):
        out[f"num_{y}"] = float((aw * zf[:, j] * zr[:, j]).sum())
        out[f"den_{y}"] = float((aw * zf[:, j]).sum())
        out[f"pres_{y}"] = float(_ratio(out[f"num_{y}"], out[f"den_{y}"]))
    called = zf.max(axis=1)
    kept = np.all((zf == 0) | (zr == 1), axis=1).astype(np.int64)
    out["num_any"], out["den_any"] = float((aw * called * kept).sum()), float((aw * called).sum())
    out["pres_any"] = float(_ratio(out["num_any"], out["den_any"]))
    both = (zf[:, 0] == 1) & (zf[:, 1] == 1)
    lost = (zf == 1) & (zr == 0)
    y0, y1 = outputs
    out["both_called_released"] = float((aw * both).sum())
    out["both_called_kept_both"] = float((aw * (both & ~lost.any(axis=1))).sum())
    out[f"both_called_lost_only_{y0}"] = float((aw * (both & lost[:, 0] & ~lost[:, 1])).sum())
    out[f"both_called_lost_only_{y1}"] = float((aw * (both & lost[:, 1] & ~lost[:, 0])).sum())
    out["both_called_lost_both"] = float((aw * (both & lost[:, 0] & lost[:, 1])).sum())
    gained = (zf == 0) & (zr == 1)
    out[f"type_switch_{y0}_lost_{y1}_gained"] = float((aw * (lost[:, 0] & gained[:, 1])).sum())
    out[f"type_switch_{y1}_lost_{y0}_gained"] = float((aw * (lost[:, 1] & gained[:, 0])).sum())
    out["released_type_switch_weight"] = out[f"type_switch_{y0}_lost_{y1}_gained"] + out[f"type_switch_{y1}_lost_{y0}_gained"]
    return out


def label_agreement_flips(a: np.ndarray, w: np.ndarray, z_full: np.ndarray, z_rec: np.ndarray, labels: np.ndarray,
                          outputs: Sequence[str] = OUTPUTS) -> dict[str, float]:
    """S-D (dataset-label agreement, not clinical harm, not clinical truth), rates among released rows.

    Worse_y = 1[zf_y = l_y and zr_y != l_y]; Corr_y = 1[zf_y != l_y and zr_y = l_y] (Worse_y + Corr_y = D1_y).
    D2 = max_y Worse_y; C2 = max_y Corr_y; mixed = D2 * C2. Worse_y is split into lost label-positive calls (l_y = 1) and
    created false calls (l_y = 0).
    """
    a, w = np.asarray(a, dtype=np.float64), np.asarray(w, dtype=np.float64)
    zf, zr, lab = (np.asarray(x, dtype=np.int64) for x in (z_full, z_rec, labels))
    worse = (zf == lab) & (zr != lab)
    corr = (zf != lab) & (zr == lab)
    aw = a * w
    den = aw.sum()
    rate = lambda x: float(_ratio((aw * x).sum(), den))  # noqa: E731
    out = {"released_weight": float(den), "rate_D2_worsening_any": rate(worse.any(axis=1)),
           "rate_C2_corrective_any": rate(corr.any(axis=1)),
           "rate_mixed": rate(worse.any(axis=1) & corr.any(axis=1))}
    for j, y in enumerate(outputs):
        out[f"rate_worse_{y}"] = rate(worse[:, j])
        out[f"rate_corr_{y}"] = rate(corr[:, j])
        out[f"rate_worse_lost_label_positive_{y}"] = rate(worse[:, j] & (lab[:, j] == 1))
        out[f"rate_worse_created_false_call_{y}"] = rate(worse[:, j] & (lab[:, j] == 0))
    return out
