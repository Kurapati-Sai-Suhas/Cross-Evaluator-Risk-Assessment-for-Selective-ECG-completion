"""Tests of the descriptive endpoints S-A to S-D (src/eval/descriptive_endpoints.py, protocol v0.2 P3).

Synthetic tests check every formula against a brute-force version or a hand count. Real-data tests read only gate features,
D1 keys (identifiers and masks) and column headers: no D1 value is joined to a feature, nothing is fitted on project data and
fold 10 is not touched.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data import ptbxl_io as io
from src.eval import d1 as d1mod
from src.eval import descriptive_endpoints as de
from src.eval import selective as sel
from tests.conftest import REPO_ROOT

GATE_DIR = REPO_ROOT / "artifacts" / "gate" / "v0_2"
D1_DIR = REPO_ROOT / "artifacts" / "d1" / "v0_1"
ROLES = ["B_gate", "9a_calibration", "9b_threshold"]


def brute_force_concordance(s, d, st, w):
    num = den = 0.0
    for i in np.flatnonzero(d == 1):
        for j in np.flatnonzero((d == 0) & (st == st[i])):
            num += w[i] * w[j] * ((s[i] > s[j]) + 0.5 * (s[i] == s[j]))
            den += w[i] * w[j]
    return num / den if den > 0 else math.nan


@pytest.mark.parametrize("seed", range(6))
def test_s_a_matches_brute_force_with_weights_ties_and_empty_strata(seed):
    rng = np.random.default_rng(seed)
    n = 160
    st = rng.integers(0, 12, n)
    s = rng.integers(0, 6, n).astype(float)                    # many ties
    d = (rng.random(n) < 0.25).astype(int)
    d[st == 0] = 1                                             # stratum 0: positives only; stratum 1: negatives only
    d[st == 1] = 0
    W = rng.integers(0, 4, size=(4, n)).astype(float)
    got = de.within_stratum_concordance(s, d, st, W)["c"]
    for b in range(4):
        assert got[b] == pytest.approx(brute_force_concordance(s, d, st, W[b]), abs=1e-12)
    one = de.within_stratum_concordance(s, d, st)
    assert one["c"][0] == pytest.approx(brute_force_concordance(s, d, st, np.ones(n)), abs=1e-12)
    pairs = sum(((d == 1) & (st == k)).sum() * ((d == 0) & (st == k)).sum() for k in range(12))
    assert one["n_comparable_pairs"] == pairs


def test_s_a_gate_a_is_one_half_exactly_in_every_replicate():
    """A score that is a function of the lead set ties every within-lead-set pair: C_within = 0.5 exactly."""
    rng = np.random.default_rng(7)
    n = 4000
    lead_set = rng.integers(0, 258, n)
    gate_a = rng.random(258)[lead_set]                         # any function of the lead set
    d = (rng.random(n) < 0.04 + 0.1 * gate_a).astype(int)      # D1 depends on the lead set, so pooled C > 0.5
    pid = rng.integers(0, 900, n)
    blocks = list(sel.patient_multiplicities(pid, 200, seed=20261030, chunk=100))
    W = np.vstack([sel.row_weights(pid, p, M) for p, M in blocks])
    res = de.lead_set_concordance(gate_a, d, lead_set, W)
    assert np.all(res["c_within_pair_weighted"] == 0.5)        # exact, not approximate
    assert np.all(res["c_within_stratum_mean"] == 0.5)         # both candidate S-A aggregates
    assert np.nanmean(res["c_unstratified"]) > 0.55
    assert (res["eligible_lead_sets"] <= res["lead_sets_present"]).all() and res["eligible_lead_sets"].min() > 0


def test_s_a_undefined_without_comparable_pairs():
    out = de.within_stratum_concordance(np.array([0.1, 0.2, 0.3]), np.array([1, 0, 1]), np.array([0, 1, 2]))
    assert math.isnan(out["c"][0]) and out["n_strata_contributing"] == 0


def test_rf_scores_identical_rows_identical_scores_and_n_jobs_1_reproducible():
    """Gate A's tie property in practice: equal G1 rows get bitwise equal RF scores within one call; n_jobs = 1 repeats."""
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(3000, 15)).astype(float)
    y = (rng.random(3000) < 0.05 + 0.1 * X[:, 0]).astype(int)
    rf = RandomForestClassifier(n_estimators=60, min_samples_leaf=10, random_state=20261030, n_jobs=4).fit(X, y)
    Xd = np.repeat(X[:40], 5, axis=0)
    p = rf.predict_proba(Xd)[:, 1].reshape(40, 5)
    assert (p == p[:, :1]).all()
    rf.set_params(n_jobs=1)
    assert np.array_equal(rf.predict_proba(X)[:, 1], rf.predict_proba(X)[:, 1])


@pytest.mark.parametrize("role", ROLES)
def test_real_g1_is_a_function_of_the_lead_set(role):
    """mask_id <-> G1 is one-to-one on the extracted features, so Gate A cannot rank rows within a lead set."""
    feat, keys = GATE_DIR / "features" / f"{role}_features.csv", D1_DIR / "keys" / f"{role}_rows.csv"
    if not (feat.is_file() and keys.is_file()):
        pytest.skip("features or keys not present")
    g1 = [c for c in pd.read_csv(feat, nrows=0).columns if c.startswith("g1_")]
    f = pd.read_csv(feat, usecols=["row_id"] + g1)
    k = pd.read_csv(keys, usecols=["row_id", "mask_id"])
    m = k.merge(f, on="row_id", validate="one_to_one")
    assert len(m) == len(k) == len(f) and len(g1) == 15
    sig = m[g1].astype(str).agg("|".join, axis=1)
    assert m.groupby("mask_id")[g1].nunique().max().max() == 1
    assert sig.groupby(m["mask_id"]).nunique().max() == 1 and m.groupby(sig)["mask_id"].nunique().max() == 1


def test_real_offline_schema_has_every_endpoint_column_and_they_are_forbidden_as_features():
    path = D1_DIR / "offline_labels" / "9b_threshold_OFFLINE_d1_labels.csv"
    if not path.is_file():
        pytest.skip("offline labels not present")
    header = list(pd.read_csv(path, nrows=0).columns)          # header only
    assert set(de.required_offline_columns()) <= set(header)
    p2 = io.load_config(REPO_ROOT / "configs" / "protocol_v0_2.yaml")
    endpoint_cols = [c for c in de.required_offline_columns() if c != "row_id"]
    assert sorted(d1mod.check_feature_columns(endpoint_cols, p2)) == sorted(endpoint_cols)


def synthetic_rows():
    #            zf(CL,CR) zr(CL,CR) label(CL,CR)
    rows = [((1, 0), (1, 0), (1, 0)),     # CL call kept, label agrees
            ((1, 0), (0, 0), (1, 0)),     # CL call lost: worsening (label-positive lost)
            ((1, 0), (0, 1), (1, 0)),     # type switch: CL lost (worse), CR gained (worse, false call)
            ((0, 1), (0, 1), (0, 0)),     # CR call kept, label disagrees (no flip)
            ((0, 1), (0, 0), (0, 0)),     # CR call lost: corrective
            ((1, 1), (1, 0), (1, 1)),     # both called, CR lost: worse for CR
            ((0, 0), (0, 0), (0, 1)),     # nothing called
            ((0, 0), (1, 0), (1, 0))]     # CL gained: corrective
    zf, zr, lab = (np.array([r[i] for r in rows]) for i in range(3))
    return zf, zr, lab


def test_s_b_coverage_hand_count():
    zf, zr, lab = synthetic_rows()
    a = np.array([1, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    out = de.bbb_positive_coverage(a, np.ones(8), zf, lab)
    assert out["cov_e1_called_CLBBB"] == pytest.approx(3 / 4)           # CL called rows 0,1,2,5; released 0,2,5
    assert out["cov_e1_called_CRBBB"] == pytest.approx(2 / 3)           # CR called rows 3,4,5; released 3,5
    assert out["cov_e1_called_any"] == pytest.approx(4 / 6)
    assert out["cov_label_positive_CLBBB"] == pytest.approx(4 / 5)      # label CL rows 0,1,2,5,7
    assert out["cov_label_positive_CRBBB"] == pytest.approx(2 / 2)      # label CR rows 5,6
    m0 = de.bbb_positive_coverage(np.ones(8), np.ones(8), zf, lab)
    assert all(v == 1.0 for k, v in m0.items() if k.startswith("cov_"))
    none = de.bbb_positive_coverage(np.ones(2), np.ones(2), np.zeros((2, 2)), np.zeros((2, 2)))
    assert math.isnan(none["cov_e1_called_CLBBB"]) and math.isnan(none["cov_label_positive_any"])


def test_s_c_preservation_hand_count_and_identity():
    zf, zr, lab = synthetic_rows()
    a = np.ones(8)
    out = de.bbb_call_preservation(a, np.ones(8), zf, zr)
    assert out["pres_CLBBB"] == pytest.approx(2 / 4)                    # CL called 0,1,2,5; kept 0,5
    assert out["pres_CRBBB"] == pytest.approx(1 / 3)                    # CR called 3,4,5; kept 3
    assert out["pres_any"] == pytest.approx(2 / 6)                      # rows 0 and 3 keep every call
    assert out["released_type_switch_weight"] == 1
    for j in range(2):                                                  # 1 - Pres_y = released D1_miss_y share
        miss = (zf[:, j] == 1) & (zr[:, j] == 0)
        assert 1 - out[f"pres_{de.OUTPUTS[j]}"] == pytest.approx(miss.sum() / zf[:, j].sum())
    empty = de.bbb_call_preservation(np.zeros(8), np.ones(8), zf, zr)
    assert math.isnan(empty["pres_any"])


def test_s_d_flips_hand_count_and_identities():
    zf, zr, lab = synthetic_rows()
    out = de.label_agreement_flips(np.ones(8), np.ones(8), zf, zr, lab)
    assert out["rate_worse_CLBBB"] == pytest.approx(2 / 8)              # rows 1, 2
    assert out["rate_corr_CLBBB"] == pytest.approx(1 / 8)               # row 7
    assert out["rate_worse_CRBBB"] == pytest.approx(2 / 8)              # rows 2 (false call), 5 (lost)
    assert out["rate_corr_CRBBB"] == pytest.approx(1 / 8)               # row 4
    assert out["rate_D2_worsening_any"] == pytest.approx(3 / 8)         # rows 1, 2, 5
    assert out["rate_C2_corrective_any"] == pytest.approx(2 / 8)        # rows 4, 7
    assert out["rate_mixed"] == 0
    assert out["rate_worse_lost_label_positive_CRBBB"] == pytest.approx(1 / 8)
    assert out["rate_worse_created_false_call_CRBBB"] == pytest.approx(1 / 8)


@pytest.mark.parametrize("seed", range(4))
def test_s_d_worse_plus_corrective_equals_d1_per_output(seed):
    rng = np.random.default_rng(seed)
    zf, zr, lab = (rng.integers(0, 2, size=(500, 2)) for _ in range(3))
    a, w = rng.random(500), rng.integers(1, 3, 500).astype(float)
    out = de.label_agreement_flips(a, w, zf, zr, lab)
    for j, y in enumerate(de.OUTPUTS):
        d1y = (zf[:, j] != zr[:, j]).astype(float)
        assert out[f"rate_worse_{y}"] + out[f"rate_corr_{y}"] == pytest.approx(sel.selective_risk(a, w, d1y))
        assert out[f"rate_worse_lost_label_positive_{y}"] + out[f"rate_worse_created_false_call_{y}"] == \
            pytest.approx(out[f"rate_worse_{y}"])


def test_offline_arrays_checks_the_d1_identities():
    zf, zr, lab = synthetic_rows()
    t = pd.DataFrame({"row_id": [f"r{i}" for i in range(8)], "D1": (zf != zr).any(axis=1).astype(int)})
    for j, y in enumerate(de.OUTPUTS):
        t[f"y_{y}"], t[f"z_full_{y}"], t[f"z_rec_{y}"] = lab[:, j], zf[:, j], zr[:, j]
        t[f"D1_miss_{y}"], t[f"D1_false_{y}"] = zf[:, j] & (1 - zr[:, j]), (1 - zf[:, j]) & zr[:, j]
    arr = de.offline_arrays(t)
    assert np.array_equal(arr["z_full"], zf) and np.array_equal(arr["labels"], lab)
    t.loc[0, "D1"] = 1 - t.loc[0, "D1"]
    with pytest.raises(ValueError):
        de.offline_arrays(t)


def test_endpoints_at_exact_coverage_use_fractional_acceptance():
    zf, zr, lab = synthetic_rows()
    s = np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.5, 0.3, 0.4])
    a = sel.accept_at_coverage(s, np.ones(8), 0.5)                      # W* = 4: rows 0,3,6,7 whole
    out = de.bbb_positive_coverage(a, np.ones(8), zf, lab)
    assert a.sum() == pytest.approx(4) and out["coverage_overall"] == pytest.approx(0.5)
    a = sel.accept_at_coverage(s, np.ones(8), 0.6)                      # W* = 4.8: tie group {0.5, 0.5} gets phi = 0.4
    assert np.allclose(a[[2, 5]], 0.4)
    out = de.bbb_call_preservation(a, np.ones(8), zf, zr)
    assert out["pres_CLBBB"] == pytest.approx((1 + 0.4) / (1 + 0.4 + 0.4))   # CL called & released: rows 0, 2 (0.4), 5 (0.4)


# ---------------------------------------------------------------------------------------------------------------------
# M4b-0d pre-pin additions: S-A aggregates, overlap, type switches, undefined denominators, row alignment
# ---------------------------------------------------------------------------------------------------------------------
def brute_force_stratum_mean(s, d, st, w):
    vals = []
    for k in np.unique(st):
        m = st == k
        c = brute_force_concordance(s[m], d[m], st[m], w[m])
        if not math.isnan(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else math.nan


@pytest.mark.parametrize("seed", range(6))
def test_s_a_stratum_mean_matches_brute_force_and_counts_eligibility(seed):
    rng = np.random.default_rng(100 + seed)
    n = 180
    st = rng.integers(0, 15, n)
    s = rng.integers(0, 5, n).astype(float)                    # ties
    d = (rng.random(n) < 0.3).astype(int)
    d[st == 0], d[st == 1] = 1, 0                              # one-class strata
    W = rng.integers(0, 3, size=(5, n)).astype(float)
    out = de.within_stratum_concordance(s, d, st, W)
    for b in range(5):
        assert out["c_macro"][b] == pytest.approx(brute_force_stratum_mean(s, d, st, W[b]), abs=1e-12)
        eligible = sum(((W[b] * d)[st == k].sum() > 0) and ((W[b] * (1 - d))[st == k].sum() > 0) for k in np.unique(st))
        assert out["eligible_strata"][b] == eligible
    assert out["n_strata_only_d1"] >= 1 and out["n_strata_only_non_d1"] >= 1
    assert out["n_strata_contributing"] + out["n_strata_only_d1"] + out["n_strata_only_non_d1"] == out["strata_present"]


def test_s_a_aggregates_differ_when_strata_differ_in_size():
    """Pair weighting and stratum averaging are different estimands: here they give 0.9 versus 0.5."""
    # stratum 0: 2 D1 x 8 non-D1 = 16 pairs, perfectly ranked; stratum 1: 1 x 1 = 1 pair, reversed
    s = np.r_[[9, 9], np.zeros(8), [0], [1]].astype(float)
    d = np.r_[[1, 1], np.zeros(8), [1], [0]].astype(int)
    st = np.r_[np.zeros(10), [1, 1]].astype(int)
    out = de.within_stratum_concordance(s, d, st)
    assert out["c"][0] == pytest.approx(16 / 17) and out["c_macro"][0] == pytest.approx(0.5)


def test_s_a_is_invariant_to_row_order():
    rng = np.random.default_rng(5)
    n = 300
    s, d, st = rng.random(n).round(2), (rng.random(n) < 0.2).astype(int), rng.integers(0, 20, n)
    p = rng.permutation(n)
    a, b = de.lead_set_concordance(s, d, st), de.lead_set_concordance(s[p], d[p], st[p])
    for k in ("c_within_stratum_mean", "c_within_pair_weighted", "c_unstratified"):
        assert a[k][0] == pytest.approx(b[k][0], abs=1e-15)


def test_s_b_rows_positive_for_both_outputs_count_once_per_denominator():
    zf = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    lab = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
    a = np.array([1.0, 0.0, 1.0, 1.0])
    out = de.bbb_positive_coverage(a, np.ones(4), zf, lab)
    assert (out["den_e1_called_CLBBB"], out["den_e1_called_CRBBB"], out["den_e1_called_any"], out["den_e1_called_both"]) == (2, 2, 3, 1)
    assert out["cov_e1_called_any"] == pytest.approx(2 / 3) and out["cov_e1_called_both"] == 1.0
    assert out["num_e1_called_CLBBB"] == 1 and out["cov_e1_called_CLBBB"] == pytest.approx(0.5)
    assert (out["den_label_positive_CLBBB"], out["den_label_positive_any"], out["den_label_positive_both"]) == (2, 3, 1)
    none = de.bbb_positive_coverage(np.ones(2), np.ones(2), np.zeros((2, 2)), np.zeros((2, 2)))
    assert math.isnan(none["cov_e1_called_both"]) and none["den_e1_called_any"] == 0


def test_s_c_both_called_and_type_switches_reported_separately():
    #               zf      zr
    zf = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [0, 1], [1, 0]])
    zr = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [0, 1], [1, 0], [1, 1]])
    out = de.bbb_call_preservation(np.ones(7), np.ones(7), zf, zr)
    assert out["both_called_released"] == 4 and out["both_called_kept_both"] == 1
    assert out["both_called_lost_only_CLBBB"] == 1 and out["both_called_lost_only_CRBBB"] == 1 and out["both_called_lost_both"] == 1
    assert out["type_switch_CLBBB_lost_CRBBB_gained"] == 1 and out["type_switch_CRBBB_lost_CLBBB_gained"] == 1
    assert out["released_type_switch_weight"] == 2
    assert (out["num_CLBBB"], out["den_CLBBB"]) == (3, 6)       # CL called rows 0,1,2,3,4,6; kept 0,2,6
    assert (out["num_any"], out["den_any"]) == (2, 7)            # rows 0 and 6 keep every call (6 gained CR: not a failure)


def test_alignment_by_row_id_is_order_independent_and_strict():
    rng = np.random.default_rng(9)
    n = 50
    keys = pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], "mask_id": rng.integers(0, 5, n)})
    labels = pd.DataFrame({"row_id": keys["row_id"], "D1": rng.integers(0, 2, n)})
    scores = pd.DataFrame({"row_id": keys["row_id"], "score": rng.random(n)})
    ref = de.align_rows(keys, labels, scores)
    shuffled = de.align_rows(keys, labels.sample(frac=1, random_state=1), scores.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(ref, shuffled)
    assert ref["row_id"].tolist() == keys["row_id"].tolist()
    a = de.lead_set_concordance(ref["score"].to_numpy(), ref["D1"].to_numpy(), ref["mask_id"].to_numpy())
    b = de.lead_set_concordance(shuffled["score"].to_numpy(), shuffled["D1"].to_numpy(), shuffled["mask_id"].to_numpy())
    assert a["c_within_pair_weighted"][0] == b["c_within_pair_weighted"][0]
    with pytest.raises(ValueError):
        de.align_rows(keys, labels.iloc[1:], scores)                     # missing row
    with pytest.raises(ValueError):
        de.align_rows(keys, pd.concat([labels, labels.iloc[:1]]), scores)  # duplicate row
