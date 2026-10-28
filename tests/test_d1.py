"""Tests for Milestone M4a: D1 formulas, thresholds, role guards, masks, copy-through, the fixed scoring path, the
label/feature separation guard, and (once generated) the integrity of the D1 artifacts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data import missingness as mz  # noqa: E402
from src.data import ptbxl_io as io  # noqa: E402
from src.eval import d1 as d1mod  # noqa: E402
from src.eval import generate_d1 as gd  # noqa: E402
from src.models import l2g  # noqa: E402
from src.models.evaluators import ResNet1dWang  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "artifacts" / "d1" / "v0_1"
ROLES = ["C_dev", "B_gate", "9a_calibration", "9b_threshold"]
E1_SUMMARY = REPO_ROOT / "artifacts" / "runs" / "e1_resnet1d_v0_1_run01" / "summary.json"


@pytest.fixture(scope="module")
def protocol() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "protocol_v0_1.yaml")


@pytest.fixture(scope="module")
def outputs(protocol) -> list[str]:
    return list(protocol["targets"]["primary_outputs"])


# --------------------------------------------------------------------------- #
# D1 formulas
# --------------------------------------------------------------------------- #
def test_d1_truth_table(outputs):
    tau = {"CLBBB": np.float32(0.5), "CRBBB": np.float32(0.3)}
    p_full = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.1], [0.1, 0.9], [0.5, 0.3], [0.4999, 0.2999], [0.9, 0.9]], np.float32)
    p_rec = np.array([[0.8, 0.2], [0.1, 0.1], [0.9, 0.1], [0.1, 0.1], [0.5, 0.3], [0.5, 0.3], [0.1, 0.1]], np.float32)
    c = d1mod.d1_columns(p_full, p_rec, tau, outputs)
    assert c["D1"].tolist() == [0, 1, 1, 1, 0, 1, 1]
    assert c["D1_miss_CLBBB"].tolist() == [0, 1, 0, 0, 0, 0, 1]
    assert c["D1_false_CLBBB"].tolist() == [0, 0, 1, 0, 0, 1, 0]
    assert c["D1_miss_CRBBB"].tolist() == [0, 0, 0, 1, 0, 0, 1]
    assert c["D1_false_CRBBB"].tolist() == [0, 0, 0, 0, 0, 1, 0]
    # D1 is exactly the OR over outputs of (miss or false)
    assert np.array_equal(c["D1"], np.maximum.reduce([c[f"D1_{d}_{y}"] for d in ("miss", "false") for y in outputs]))


def test_probability_equal_to_tau_is_positive_in_float32():
    tau = np.float32(0.31639263)
    assert d1mod.decide(np.array([tau]), tau)[0]
    assert not d1mod.decide(np.array([np.nextafter(tau, np.float32(0))]), tau)[0]
    # a float64 value that rounds to tau in float32 is treated as tau (comparison happens in float32)
    assert d1mod.decide(np.array([0.3163926303386688]), tau)[0]


def test_near_threshold_window():
    tau = np.float32(0.5)
    p = np.array([0.5, 0.5000009, 0.5000011, 0.4999991, 0.4999989], np.float32)
    assert d1mod.near_threshold(p, tau, 1e-6).tolist() == [True, True, False, True, False]


def test_thresholds_come_from_the_e1_summary(outputs, tmp_path):
    tau = d1mod.load_thresholds(E1_SUMMARY, outputs)
    assert tau == {"CLBBB": np.float32(0.31639263), "CRBBB": np.float32(0.35816982)}
    assert all(isinstance(v, np.float32) for v in tau.values())
    bad = json.loads(E1_SUMMARY.read_text(encoding="utf-8"))
    bad["eligible"] = False
    (tmp_path / "s.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not marked eligible"):
        d1mod.load_thresholds(tmp_path / "s.json", outputs)


# --------------------------------------------------------------------------- #
# Roles and masks
# --------------------------------------------------------------------------- #
def test_d1_roles_refuse_fold10_and_a_fit(protocol):
    man = pd.DataFrame({"ecg_id": [1, 2, 3, 4, 12722], "patient_id": [1, 2, 3, 4, 5], "strat_fold": [9, 9, 10, 1, 1],
                        "role": ["9a_calibration", "9b_threshold", "test_fold10", "A_fit", "A_fit"]})
    assert gd.select_d1_role(man, "9a_calibration", protocol)["ecg_id"].tolist() == [1]
    with pytest.raises(RuntimeError, match="test_fold10 is refused"):
        gd.select_d1_role(man, "test_fold10", protocol)
    with pytest.raises(RuntimeError, match="not a D1 role"):
        gd.select_d1_role(man, "A_fit", protocol)


def test_evaluation_masks_are_deterministic_distinct_and_order_free(protocol):
    seed, k = int(protocol["seed"]), int(protocol["d1"]["masks_per_record"])
    a = [mz.evaluation_combinations(e, seed, k=k) for e in (5, 17, 99)]
    b = [mz.evaluation_combinations(e, seed, k=k) for e in (99, 17, 5)][::-1]
    assert a == b
    assert all(len(set(cs)) == k for cs in a)
    combos = set(mz.l2g_lead_combinations())
    assert all(c in combos for cs in a for c in cs) and len(combos) == 258


# --------------------------------------------------------------------------- #
# Fixed scoring path
# --------------------------------------------------------------------------- #
def test_reconstruct_copies_observed_samples_exactly():
    torch.manual_seed(0)
    gen = l2g.ECGen(24).eval()
    x = torch.rand(3, 12, 2048)
    lm = torch.from_numpy(np.stack([mz.lead_mask(("I",)), mz.lead_mask(("II", "V1", "V5")), mz.lead_mask(("AVL", "V3"))]))
    x_hat, violations = gd.reconstruct(gen, x, lm, batch_rows=2)
    m = lm.bool().unsqueeze(-1).expand_as(x)
    assert violations == 0 and torch.equal(x_hat[m], x[m])
    assert not torch.equal(x_hat[~m], x[~m])  # missing leads come from G
    gen.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        gd.reconstruct(gen, x, lm, batch_rows=2)


def test_score_e1_fixed_batch_padding_is_order_free_and_stateless():
    torch.manual_seed(0)
    e1 = ResNet1dWang(12, 2, (4, 4))
    with torch.no_grad():  # give the BatchNorm buffers non-trivial values
        for mod in e1.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-0.1, 0.1)
                mod.running_var.uniform_(0.5, 1.5)
    e1.eval()
    before = gd.state_hash(e1)
    x = torch.rand(7, 12, 256)
    p = gd.score_e1(e1, x, batch_rows=4)
    p_rev = gd.score_e1(e1, x.flip(0), batch_rows=4).flip(0)
    assert p.shape == (7, 2) and p.dtype == torch.float32
    assert torch.allclose(p, p_rev, atol=1e-7, rtol=0)
    assert gd.state_hash(e1) == before  # no BatchNorm statistics updated
    assert not p.requires_grad
    e1.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        gd.score_e1(e1, x, batch_rows=4)


# --------------------------------------------------------------------------- #
# Label / feature separation
# --------------------------------------------------------------------------- #
def test_feature_guard_flags_reference_derived_and_identifier_columns(protocol):
    safe = ["n_observed", "obs_flag_I", "obs_limb_count", "sqi_kurtosis_mean"]
    assert d1mod.check_feature_columns(safe, protocol) == []
    bad = ["D1", "d1_miss_CLBBB", "p_full_CLBBB", "p_rec_CRBBB", "tau_CLBBB", "z_full_CLBBB", "y_CLBBB", "ecg_id",
           "patient_id", "row_id", "mask_index", "strat_fold", "vttd_score", "e1_outputs_reconstruction", "baseline_drift",
           "true_reconstruction_error", "complete_reference_ecg", "age", "sex", "prob_CLBBB"]
    assert sorted(d1mod.check_feature_columns(bad, protocol)) == sorted(bad)


def test_attach_d1_target_copies_only_d1(protocol):
    feats = pd.DataFrame({"row_id": ["a", "b", "c"], "n_observed": [1, 3, 5]})
    labels = pd.DataFrame({"row_id": ["c", "a", "b"], "D1": [1, 0, 1], "p_full_CLBBB": [0.9, 0.1, 0.2], "y_CLBBB": [1, 0, 0]})
    X, y = d1mod.attach_d1_target(feats, labels, protocol)
    assert list(X.columns) == ["n_observed"] and y.tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="forbidden"):
        d1mod.attach_d1_target(feats.assign(p_full_CLBBB=0.5), labels, protocol)
    with pytest.raises(ValueError, match="no D1 label"):
        d1mod.attach_d1_target(pd.DataFrame({"row_id": ["z"], "n_observed": [2]}), labels, protocol)
    with pytest.raises(Exception):  # duplicated label rows must not silently multiply feature rows
        d1mod.attach_d1_target(feats, pd.concat([labels, labels]), protocol)


# --------------------------------------------------------------------------- #
# Generated artifacts (skipped until M4a has run)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def generated():
    if not (OUT / "generation_summary.json").is_file():
        pytest.skip("M4a D1 artifacts not generated yet")
    return json.loads((OUT / "generation_summary.json").read_text(encoding="utf-8"))


def test_generated_roles_and_no_fold10(generated):
    assert generated["roles"] == ROLES and generated["fold10_loaded"] is False
    names = [p.name for p in OUT.rglob("*")]
    assert not any("fold10" in n or "test_fold" in n for n in names)
    assert generated["model_state_unchanged"] is True


def test_hash_manifest_matches_files(generated):
    man = json.loads((OUT / "hash_manifest.json").read_text(encoding="utf-8"))
    for rel, sha in man["files"].items():
        assert io.sha256_file(REPO_ROOT / rel) == sha, rel
    assert man["inputs"]["g"] == "d515875a7783be5a54934001e13b182ff4f75a5e35554f0e9590d5a1c602bb3b"
    assert man["inputs"]["e1"] == "e10c250c78782fe23449fa9dc39dfb7fc981baac00eb690511444386a4357c1f"


@pytest.mark.parametrize("role", ROLES)
def test_generated_rows_labels_and_reuse(generated, protocol, outputs, role):
    k = int(protocol["d1"]["masks_per_record"])
    keys = pd.read_csv(OUT / "keys" / f"{role}_rows.csv")
    lab = pd.read_csv(OUT / "offline_labels" / f"{role}_OFFLINE_d1_labels.csv")
    rec = pd.read_csv(OUT / "offline_labels" / f"{role}_OFFLINE_e1_complete_by_record.csv")
    assert tuple(keys.columns) == d1mod.KEY_COLUMNS
    assert len(keys) == len(lab) == k * len(rec) and keys.row_id.is_unique and (keys.row_id == lab.row_id).all()
    assert (keys.groupby("ecg_id").mask_index.nunique() == k).all() and (keys.role == role).all()
    # the stored complete-ECG score is identical for every mask row of a record and equals the per-record file
    df = keys[["row_id", "ecg_id"]].merge(lab, on="row_id")
    tau = d1mod.load_thresholds(E1_SUMMARY, outputs)
    for y in outputs:
        assert (df.groupby("ecg_id")[f"p_full_{y}"].nunique() == 1).all()
        m = df.groupby("ecg_id")[f"p_full_{y}"].first().to_numpy(np.float32)
        assert np.array_equal(m, rec.set_index("ecg_id").loc[df.ecg_id.unique(), f"p_full_{y}"].to_numpy(np.float32))
        assert (lab[f"tau_{y}"].astype(np.float32) == tau[y]).all()
    # D1 recomputed from the stored float32 probabilities equals the stored labels
    pf = np.stack([lab[f"p_full_{y}"].to_numpy(np.float32) for y in outputs], axis=1)
    pr = np.stack([lab[f"p_rec_{y}"].to_numpy(np.float32) for y in outputs], axis=1)
    for name, col in d1mod.d1_columns(pf, pr, tau, outputs).items():
        assert np.array_equal(lab[name].to_numpy(), col), name
    # masks are the protocol masks
    sample = keys[keys.ecg_id.isin(keys.ecg_id.unique()[:20])]
    for e, g in sample.groupby("ecg_id"):
        expect = [";".join(c) for c in mz.evaluation_combinations(int(e), int(protocol["seed"]), k=k)]
        assert g.sort_values("mask_index").observed_leads.tolist() == expect
    # keys hold no reference-derived column
    assert d1mod.check_feature_columns([c for c in keys.columns if c not in d1mod.IDENTIFIER_COLUMNS
                                        and c not in ("observed_leads", "n_observed")], protocol) == []


def test_generated_roles_patient_disjoint_and_not_fold10(generated):
    man = pd.read_csv(REPO_ROOT / "artifacts" / "manifests" / "split_manifest_v1_frozen.csv")
    fold10_patients = set(man.loc[man.role == "test_fold10", "patient_id"])
    seen = {}
    for role in ROLES:
        keys = pd.read_csv(OUT / "keys" / f"{role}_rows.csv", usecols=["ecg_id", "patient_id"])
        assert set(man.set_index("ecg_id").loc[keys.ecg_id.unique(), "role"]) == {role}
        seen[role] = set(keys.patient_id)
        assert not (seen[role] & fold10_patients)
    for a in ROLES:
        for b in ROLES:
            if a < b:
                assert not (seen[a] & seen[b]), (a, b)


def test_generated_reproducibility_and_boundary_report(generated, outputs):
    for role, r in generated["reproducibility"].items():
        assert r["max_abs_diff_p_full"] <= 1e-6 and r["max_abs_diff_p_rec"] <= 1e-6 and r["decisions_identical"], role
    b = pd.read_csv(REPO_ROOT / "artifacts" / "reports" / "m4a_threshold_boundary_cases.csv")
    for role, s in generated["per_role"].items():
        for y in outputs:
            n_full = ((b.role == role) & (b.output == y) & (b.score == "p_full (per record)")).sum()
            n_rec = ((b.role == role) & (b.output == y) & (b.score == "p_rec (per row)")).sum()
            assert n_full == s["near_tau_full_records"][y] and n_rec == s["near_tau_rec_rows"][y]
