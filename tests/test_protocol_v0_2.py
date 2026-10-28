"""Consistency tests for protocol v0.2, proposal P5 (M4b-0d-pp2; C_mean S-A headline), pinned on 2026-10-24 after the
owner's explicit approval (D-037): YAML structure, the single decision rule, the undefined-H1 rule, threshold candidates,
the descriptive endpoints S-A..S-D (both S-A aggregates with fixed headline/sensitivity roles, overlap and type-switch
reporting), the fold-10 plan, wording, cross-file hashes, the archived predecessors and alternatives, and the provenance of
the already extracted features."""
from __future__ import annotations

import json
import re

import pytest

from src.data import ptbxl_io as io
from src.eval import d1 as d1mod
from src.eval import generate_d1 as gd
from src.gate import extract_features as ef
from tests.conftest import REPO_ROOT

PROTOCOL = REPO_ROOT / "configs" / "protocol_v0_2.yaml"
GATE = REPO_ROOT / "configs" / "gate_m4b_v0_2.yaml"
DOC = REPO_ROOT / "docs" / "evaluation_protocol_v0_2.md"
LOG = REPO_ROOT / "docs" / "protocol_v0_2_change_log.md"
HISTORY = REPO_ROOT / "artifacts" / "protocol_history"
GATE_SHA = "80fdd7a4c7aec27005eeb84c93b47228923b86d2867ce1da540b12e0aa264690"
GATE_SHA_P3 = "e466724b5014e9bf15820370518a5db77f5f0629cf86dcc13e25f4fd68671833"
GATE_SHA_FEATURES = "3705aea7bebd22e2b36dc389dcb16437cf259409407bbbe858ab71dac7f7b171"   # features were extracted under this
P1 = "b45a03b9382de9a629fe69ab939b69fb7c842dc203ab9b8ddd432ee234059138"
P2 = "cce1e70b19d34b32e59072db8076fbae53fd00bdcac255a36842f9979f0a7af7"
P3 = "7a39901128afe1111b6e24bbd51af549abdfe390e8cc650fbcced4d88621e134"
P4A = "e8f4aaadd30306e9179fe1162080b24b16c76ab6e29165df312cbd81adc79028"
P4B = "eec4f4b769f6143278684f7d695d69846282ca305e86002262e1631eb556899e"
CANDIDATES = HISTORY / "candidates"
P5_RECOMMENDED = CANDIDATES / "protocol_v0_2_P5_option_B_C_mean_headline_RECOMMENDED.yaml"
P5_ALTERNATIVE = CANDIDATES / "protocol_v0_2_P5_option_A_C_pair_headline_ALTERNATIVE.yaml"
FEATURE_BLOCKS = ("output_dir", "d1_keys_dir", "d1_hash_manifest_sha256", "roles", "g1", "g2")


@pytest.fixture(scope="module")
def p2() -> dict:
    return io.load_config(PROTOCOL)


def test_gate_preregistration_referenced_by_hash(p2):
    assert io.sha256_file(GATE) == GATE_SHA == p2["gate_preregistration"]["sha256"]


def test_archived_predecessors_are_byte_identical():
    assert io.sha256_file(HISTORY / "protocol_v0_2_P2_cce1e70b.yaml") == P2
    assert io.sha256_file(HISTORY / "gate_m4b_v0_2_3705aea7.yaml") == GATE_SHA_FEATURES
    assert io.sha256_file(HISTORY / "protocol_v0_2_P3_7a399011.yaml") == P3
    assert io.sha256_file(HISTORY / "gate_m4b_v0_2_e466724b.yaml") == GATE_SHA_P3


def test_feature_defining_blocks_unchanged_so_extracted_features_stay_valid(p2):
    old, new = io.load_config(HISTORY / "gate_m4b_v0_2_3705aea7.yaml"), io.load_config(GATE)
    mid = io.load_config(HISTORY / "gate_m4b_v0_2_e466724b.yaml")
    for block in FEATURE_BLOCKS:
        assert old[block] == mid[block] == new[block], block
    texts = [p.read_text(encoding="utf-8") for p in (HISTORY / "gate_m4b_v0_2_3705aea7.yaml", HISTORY / "gate_m4b_v0_2_e466724b.yaml", GATE)]
    for start in ("g1:", "g2:"):                                # raw text of the formula blocks, not only the parsed values
        blocks = [x[x.index("\n" + start):].split("\n# ----")[0] for x in texts]
        assert blocks[0] == blocks[1] == blocks[2], start
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {"rf", "threshold"} and set(old) == set(new)
    assert {k: v for k, v in old["rf"].items()} == {k: v for k, v in new["rf"].items() if k != "scoring"}
    assert GATE_SHA_FEATURES in p2["feature_provenance"] and "block identity" in p2["feature_provenance"]
    path = REPO_ROOT / "artifacts" / "gate" / "v0_2" / "extraction_summary.json"
    if not path.is_file():
        pytest.skip("features not extracted")
    s = json.loads(path.read_text(encoding="utf-8"))
    assert s["config_sha256"] == GATE_SHA_FEATURES        # the formulas the features were computed with
    assert s["protocol_v0_2_sha256"] == P1                # historical record of the proposal at extraction time (not edited)
    man = json.loads((REPO_ROOT / "artifacts" / "gate" / "v0_2" / "hash_manifest.json").read_text(encoding="utf-8"))
    for rel, sha in man["files"].items():
        assert io.sha256_file(REPO_ROOT / rel) == sha, rel


def test_single_interval_decision_rule_and_no_p_value(p2):
    u, pa = p2["uncertainty"], p2["primary_analysis"]
    assert "one_sided_p_value" not in u and "quantiles" not in u
    assert "not a finite-sample guarantee" in u["nature"] and "clinical-safety" in u["nature"]
    for key in ("contrast", "order_statistics", "decision_interval", "decision_rule", "descriptive_interval",
                "diagnostic_count", "undefined_replicate", "replicate_stream"):
        assert key in u, key
    assert u["replicates"] == 10000 and pa["alpha_one_sided"] == 0.05 and pa["procedure"] == "fixed_sequence"
    assert u["method"] == "patient_cluster_bootstrap_percentile" and u["one_sided_upper_quantile"] == 0.95
    assert "9501" in u["decision_interval"] and "U < 0" in u["decision_rule"] and "499" in u["decision_rule"]
    assert "250" in u["descriptive_interval"] and "9751" in u["descriptive_interval"]
    assert "never used for a decision" in u["descriptive_interval"] and "NOT a p-value" in u["diagnostic_count"]
    assert "+inf" in u["undefined_replicate"] and "excluded" in u["undefined_replicate"]
    assert "sorted ascending" in u["replicate_stream"] and "default_rng" in u["replicate_stream"]
    for h in ("H1", "H2"):
        rep = " | ".join(pa[h]["report"])
        assert "one-sided p" not in rep and "p-value" in rep and "upper bound U" in rep and "descriptive" in rep
    assert pa["H1"]["delta"] == "R_B - R_0" and pa["H2"]["delta"] == "R_B(0.80) - R_A(0.80)"
    assert pa["reject_rule"].startswith("reject iff U < 0")
    assert "recomputed inside every bootstrap replicate" in pa["H2"]["estimand"]


def test_undefined_h1_point_estimate_rule(p2):
    rule = p2["primary_analysis"]["H1"]["undefined_point_estimate"]
    assert "H1 is NOT rejected" in rule and "H2 is not tested" in rule and "realised coverage 0" in rule
    assert "No value is substituted" in rule and "U = +inf" in rule


def test_threshold_candidates_are_the_distinct_9b_scores(p2):
    th, g = p2["threshold"], io.load_config(GATE)
    assert not {"grid_start", "grid_stop", "grid_step"} & set(th)
    assert "distinct float64 values" in th["candidates"] and th["selected_on"] == "9b_threshold"
    assert th["min_coverage"] == 0.80 and th["tie_break"] == ["lowest_risk", "highest_coverage", "smallest_t"]
    assert th["objective"] == "minimise_pooled_accepted_d1_rate" and th["accept_if"] == "score <= t"
    assert "n_jobs = 1" in th["deterministic_scoring"] and "bitwise" in th["deterministic_scoring"]
    assert "deterministic within the pinned environment and seed" in th["deterministic_scoring"]
    assert "no cross-environment bitwise reproducibility is claimed" in th["deterministic_scoring"]
    assert "protocol change" in th["protocol_change"] and "owner approves" in th["protocol_change"]
    assert "deterministic within the pinned environment and seed" in g["rf"]["scoring"] and "does not establish" in g["rf"]["scoring"]
    assert "environment" in g["threshold"]["lock_contents"] and "float.fromhex" in g["threshold"]["lock_contents"]
    assert "distinct float64 calibrated scores" in g["threshold"]["rule"] and "0.001 grid" in g["threshold"]["rule"]
    assert "n_jobs = 1" in g["rf"]["scoring"] and "float.hex" in g["threshold"]["lock_contents"]
    assert "bitwise reproduction of the locked 9b calibrated scores" in p2["fold10_pathway"]["checks_before_evaluation"]


def test_descriptive_endpoints_block(p2):
    de = p2["descriptive_endpoints"]
    assert de["methods"] == ["gate_A", "gate_B", "M3_signal_quality_only", "M2_uncertainty_only"]
    assert set(de["operating_points"]) == {"locked", "c80", "reporting"}
    for k in ("S_A_within_lead_set_concordance", "S_B_bbb_positive_coverage", "S_C_bbb_call_preservation",
              "S_D_label_agreement_flips"):
        assert k in de, k
    sa = de["S_A_within_lead_set_concordance"]
    assert {"stratum_concordance", "aggregate_pair_weighted", "aggregate_stratum_mean", "headline"} <= set(sa)
    assert sa["headline"] == "stratum_mean" and sa["sensitivity"] == "pair_weighted"
    assert "eligible only if W_P,s > 0 and W_N,s > 0" in sa["stratum_concordance"]
    assert "never exchanged after any result is seen" in sa["reporting_rule"] and "C_mean, C_pair" in sa["reporting_rule"]
    assert "enters neither aggregate" in sa["ineligible_strata"] and "never imputed" in sa["ineligible_strata"]
    assert "average-configuration question" in sa["headline_rationale"] and "average-pair question" in sa["headline_rationale"]
    assert "not evidence that any gate works" in sa["headline_rationale"]
    assert "0.5 exactly" in sa["gate_A_identity"] and "every eligible lead set" in sa["gate_A_identity"]
    assert "Nothing is asserted for ineligible lead sets" in sa["gate_A_identity"]
    assert "once in each per-output denominator" in de["S_B_bbb_positive_coverage"]["overlap"]
    assert "both" in de["S_B_bbb_positive_coverage"]["e1_called"] and "both" in de["S_B_bbb_positive_coverage"]["label_positive"]
    assert "type switches by direction" in de["S_C_bbb_call_preservation"]["reported_separately"]
    assert "neither D1 nor S-D is clinical harm" in de["S_D_label_agreement_flips"]["meaning"]
    assert "primary_decision.fixed_sequence" in p2["primary_analysis"]["fixed_sequence_rule"]
    assert "Worse_y + Corr_y = D1_y" in de["S_D_label_agreement_flips"]["per_output"]
    assert "never 0" in de["zero_denominator"] and "no hypothesis test" in de["uncertainty"]
    assert "NOT gate features" in open(PROTOCOL, encoding="utf-8").read()
    assert any("S-A to S-D" in s for s in p2["secondary_descriptive"])
    # every reference-derived input is forbidden as a gate feature by name
    cols = ["y_CLBBB", "z_full_CRBBB", "z_rec_CLBBB", "D1_miss_CRBBB", "D1_false_CLBBB", "D1"]
    assert sorted(d1mod.check_feature_columns(cols, p2)) == sorted(cols)
    for name in ("d1", "labels", "e1_outputs_complete", "e1_outputs_reconstruction"):
        assert name in p2["gate"]["forbidden_inputs"]


def test_fold10_rules_and_refusals_outside_the_pathway(p2):
    fp = p2["fold10_pathway"]
    assert fp["status"] == "not_implemented"                 # pinned text written before M4b-1; the pathway is built in stage 04
    assert "threshold_lock_v0_2.json" in fp["precondition"] and "protocol_v0_2.lock.json" in fp["precondition"]
    assert "new output_dir" in fp["configs"] and "without opening any fold-10 record" in fp["pre_authorisation_tests"]
    assert "offline-label column schema" in fp["outputs"]
    assert "fold10_pathway" in p2["d1"]["fold10_rule"] and "fold10_pathway" in p2["gate"]["fold10_features"]
    assert fp["no_choices_on_fold10"].startswith("fold-10 data are never used")
    with pytest.raises(RuntimeError):                       # the code still refuses fold 10 everywhere
        ef.check_role("test_fold10")
    with pytest.raises(RuntimeError):
        gd.select_d1_role(__import__("pandas").DataFrame({"ecg_id": [1], "patient_id": [1], "strat_fold": [10],
                                                          "role": ["test_fold10"]}), "test_fold10", p2)
    # M4b-1 stage 03 wrote the threshold lock; the final fold-10 run still needs the owner's separate authorisation
    from src.eval import fold10_pathway as f10
    assert (REPO_ROOT / "artifacts" / "locks" / "threshold_lock_v0_2.json").exists()
    assert not (REPO_ROOT / f10.FINAL_AUTHORISATION).exists()
    with pytest.raises(RuntimeError):
        f10.run(REPO_ROOT, "final", "test_fold10")           # refused before any data is read


def test_signal_wording(p2):
    note = p2["gate"]["g2_signal_note"]
    assert "as distributed, without filtering by this project" in note and "unverified" in note
    assert any("as distributed, without filtering by this project" in s and "unverified" in s for s in p2["limitations_added"])
    doc = DOC.read_text(encoding="utf-8")
    assert "mV, unfiltered" not in doc
    for m in re.finditer("unfiltered", doc):                # the word only appears where its meaning is pinned down
        assert "without filtering by this project" in doc[max(0, m.start() - 400): m.end() + 400]


def test_risk_by_method_matches_gate_preregistration(p2):
    g = io.load_config(GATE)
    assert p2["threshold"]["risk_by_method"] == {"gate_A": "D1", "gate_B": "D1", "M3_signal_quality_only": "D1",
                                                 "M2_uncertainty_only": "D1", "E_dir_margin": "any_output_label_error"}
    for text in (g["calibration"]["target"], g["threshold"]["risk"]):
        assert "D1 for gates, M3 and M2" in text and "label error" in text and "E_dir_margin" in text


def test_unchanged_blocks_against_p2(p2):
    old = io.load_config(HISTORY / "protocol_v0_2_P2_cce1e70b.yaml")
    for block in ("version", "base_protocol", "seed", "targets", "d1", "roles", "eligibility", "calibration", "methods"):
        assert old[block] == p2[block], block
    for key in ("unit", "method", "replicates", "two_sided_level", "one_sided_upper_quantile", "paired"):
        assert old["uncertainty"][key] == p2["uncertainty"][key], key
    assert old["uncertainty"]["replicate_stream"] == p2["uncertainty"]["replicate_stream"].replace(   # only the sharing list grew
        ", every secondary contrast and every descriptive endpoint on the same rows", " and every secondary contrast on the same rows")
    assert old["gate"]["forbidden_inputs"] == p2["gate"]["forbidden_inputs"]
    for h in ("H1", "H2"):
        for key in ("statement", "delta"):
            assert old["primary_analysis"][h][key] == p2["primary_analysis"][h][key]


def test_p5_candidates_differ_only_in_the_s_a_roles_and_live_file_is_the_recommended_one():
    assert io.sha256_file(P5_RECOMMENDED) == io.sha256_file(PROTOCOL)
    live, alt = PROTOCOL.read_text(encoding="utf-8").splitlines(), P5_ALTERNATIVE.read_text(encoding="utf-8").splitlines()
    assert len(live) == len(alt)
    diff = [(a.strip(), b.strip()) for a, b in zip(live, alt) if a != b]
    assert [d[0].split(":")[0] for d in diff] == ["headline", "headline_rationale", "sensitivity"]
    sa = io.load_config(P5_ALTERNATIVE)["descriptive_endpoints"]["S_A_within_lead_set_concordance"]
    assert sa["headline"] == "pair_weighted" and sa["sensitivity"] == "stratum_mean"


def test_p5_is_p4_option_b_plus_only_the_s_a_additions():
    assert io.sha256_file(CANDIDATES / "protocol_v0_2_P4_option_A_pair_weighted.yaml") == P4A
    assert io.sha256_file(CANDIDATES / "protocol_v0_2_P4_option_B_stratum_mean.yaml") == P4B
    old = io.load_config(CANDIDATES / "protocol_v0_2_P4_option_B_stratum_mean.yaml")
    new = io.load_config(PROTOCOL)
    assert {k for k in new if new[k] != old.get(k)} == {"descriptive_endpoints"} and set(new) == set(old)
    o, n = old["descriptive_endpoints"], new["descriptive_endpoints"]
    assert {k for k in n if n[k] != o.get(k)} == {"S_A_within_lead_set_concordance"}
    so, sn = o["S_A_within_lead_set_concordance"], n["S_A_within_lead_set_concordance"]
    assert set(sn) - set(so) == {"sensitivity", "reporting_rule", "ineligible_strata"}
    assert {k for k in so if so[k] != sn[k]} == {"headline_rationale"} and so["headline"] == sn["headline"] == "stratum_mean"


def test_unchanged_blocks_against_p3(p2):
    old = io.load_config(HISTORY / "protocol_v0_2_P3_7a399011.yaml")
    changed = {k for k in old if old[k] != p2.get(k)}
    assert changed == {"gate_preregistration", "feature_provenance", "uncertainty", "threshold", "primary_analysis",
                       "descriptive_endpoints", "limitations_added"} and set(old) == set(p2)
    assert {k: v for k, v in p2["uncertainty"].items() if k != "nature"} == old["uncertainty"]
    assert {k: v for k, v in p2["threshold"].items() if k not in ("protocol_change", "deterministic_scoring")} == \
        {k: v for k, v in old["threshold"].items() if k != "deterministic_scoring"}
    assert {k: v for k, v in p2["primary_analysis"].items() if k != "fixed_sequence_rule"} == old["primary_analysis"]
    assert p2["limitations_added"][:-1] == old["limitations_added"]


def test_doc_and_change_log_cite_the_current_hashes():
    sha, alt = io.sha256_file(PROTOCOL), io.sha256_file(P5_ALTERNATIVE)
    doc, log = DOC.read_text(encoding="utf-8"), LOG.read_text(encoding="utf-8")
    assert f"APPROVE PIN v0.2 {sha}" in doc and f"APPROVE PIN v0.2 {alt}" in doc
    assert all(f"APPROVE PIN v0.2 {h}" not in doc for h in (P1, P2, P3, P4A, P4B))
    assert all(h in log for h in (sha, alt, P4A, P4B, P3, P2, P1, GATE_SHA, GATE_SHA_P3, GATE_SHA_FEATURES))


def test_protocol_v0_2_pinned_to_the_approved_hash_and_v0_1_intact():
    # owner approval 2026-10-24 (D-037): "APPROVE PIN v0.2 <P5 recommended hash>"; no threshold lock yet
    locks_dir = REPO_ROOT / "artifacts" / "locks"
    assert sorted(p.name for p in locks_dir.iterdir()) == ["protocol_v0_1.lock.json", "protocol_v0_2.lock.json", "threshold_lock_v0_2.json"]
    lock = json.loads((locks_dir / "protocol_v0_1.lock.json").read_text(encoding="utf-8"))
    assert io.sha256_file(REPO_ROOT / "configs" / "protocol_v0_1.yaml") == lock["sha256"]
    v02 = json.loads((locks_dir / "protocol_v0_2.lock.json").read_text(encoding="utf-8"))
    assert v02["status"] == "APPROVED" and v02["file"] == "configs/protocol_v0_2.yaml"
    assert v02["sha256"] == io.sha256_file(PROTOCOL) == io.sha256_file(P5_RECOMMENDED) != io.sha256_file(P5_ALTERNATIVE)
    assert v02["approval_phrase"] == f"APPROVE PIN v0.2 {v02['sha256']}"
    assert v02["gate_preregistration"]["sha256"] == io.sha256_file(GATE) == GATE_SHA
    assert v02["base_protocol"]["sha256"] == lock["sha256"]
    # the threshold lock (M4b-1 stage 03) names exactly this pinned protocol and its lock
    tl = json.loads((locks_dir / "threshold_lock_v0_2.json").read_text(encoding="utf-8"))
    assert tl["protocol_v0_2"]["sha256"] == v02["sha256"]
    assert tl["protocol_v0_2"]["lock_sha256"] == io.sha256_file(locks_dir / "protocol_v0_2.lock.json")
    assert tl["gate_preregistration_sha256"] == GATE_SHA and tl["c_min"] == 0.8
