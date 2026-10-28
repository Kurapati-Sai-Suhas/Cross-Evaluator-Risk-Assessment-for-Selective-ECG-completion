"""Tests for M4b-0: G1/G2 gate features, leakage guards, reference invariance, window alignment, the M2 comparator, and
(once extracted) the integrity of the feature files. No gate is fitted here."""
from __future__ import annotations

import ast
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.data import missingness as mz
from src.data import preprocess as pp
from src.data import ptbxl_io as io
from src.data.ptbxl_waveforms import LEAD_NAMES, load_record, resolve_waveform_root
from src.eval.d1 import check_feature_columns
from src.gate import comparators as cmp
from src.gate import extract_features as ef
from src.gate import features as gf
from tests.conftest import REPO_ROOT

OUT = REPO_ROOT / "artifacts" / "gate" / "v0_2"
ROLES = ["B_gate", "9a_calibration", "9b_threshold"]
N = gf.WINDOW[1] - gf.WINDOW[0]
T = np.arange(N) / gf.FS


@pytest.fixture(scope="module")
def p2() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "protocol_v0_2.yaml")


@pytest.fixture(scope="module")
def gcfg() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "gate_m4b_v0_2.yaml")


# --------------------------------------------------------------------------- #
# Config and protocol files
# --------------------------------------------------------------------------- #
def test_formulas_match_preregistered_config(gcfg):
    gf.check_config(gcfg)
    broken = json.loads(json.dumps(gcfg))
    broken["g2"]["per_lead"]["flat_frac"] = broken["g2"]["per_lead"]["flat_frac"].replace(">= 50", ">= 40")
    with pytest.raises(RuntimeError, match="differs"):
        gf.check_config(broken)


def test_protocol_v0_2_references_and_preserves_v0_1(p2):
    v1 = io.load_config(REPO_ROOT / "configs" / "protocol_v0_1.yaml")
    assert io.sha256_file(REPO_ROOT / "configs" / "protocol_v0_1.yaml") == p2["base_protocol"]["sha256"]
    assert io.sha256_file(REPO_ROOT / p2["gate_preregistration"]["file"]) == p2["gate_preregistration"]["sha256"]
    for key in ("targets", "roles", "eligibility", "calibration", "seed"):
        assert p2[key] == v1[key], key
    assert set(v1["gate"]["forbidden_inputs"]) <= set(p2["gate"]["forbidden_inputs"])
    assert p2["gate"]["allowed_feature_groups"] == ["G1_mask_geometry", "G2_observed_signal_quality"]
    assert p2["primary_analysis"]["procedure"] == "fixed_sequence" and p2["primary_analysis"]["alpha_one_sided"] == 0.05
    assert "status" not in p2  # approval lives only in the lock file
    lock = REPO_ROOT / "artifacts" / "locks" / "protocol_v0_2.lock.json"
    if lock.exists():  # after pinning, the lock must point at exactly this file
        assert json.loads(lock.read_text(encoding="utf-8"))["sha256"] == io.sha256_file(REPO_ROOT / "configs/protocol_v0_2.yaml")


# --------------------------------------------------------------------------- #
# G1
# --------------------------------------------------------------------------- #
def test_g1_for_known_masks():
    a = gf.g1_features(("I",))
    assert a["g1_n_observed"] == 1 and a["g1_obs_I"] == 1 and a["g1_n_limb_observed"] == 1 and a["g1_n_precordial_observed"] == 0
    assert sum(a[f"g1_obs_{lead}"] for lead in LEAD_NAMES) == 1
    b = gf.g1_features(("II", "V1", "V5"))
    assert (b["g1_n_observed"], b["g1_n_limb_observed"], b["g1_n_precordial_observed"]) == (3, 1, 2)
    assert list(b) == gf.G1_COLUMNS and len(gf.G1_COLUMNS) == 15 and len(gf.G2_COLUMNS) == 15
    with pytest.raises(ValueError):
        gf.g1_features(("aVL",))


# --------------------------------------------------------------------------- #
# G2 formulas on synthetic signals with known answers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq, bw_min, hf_min, hf_max", [(0.25, 0.95, 0.0, 0.01), (60.0, 0.0, 0.99, 1.0), (10.0, 0.0, 0.0, 0.01)])
def test_spectral_fractions_of_pure_tones(freq, bw_min, hf_min, hf_max):
    s = gf.sqi_per_lead(np.sin(2 * np.pi * freq * T) + 0.3)
    assert s["bw_frac"] >= bw_min and hf_min <= s["hf_frac"] <= hf_max
    if freq >= 10:
        assert s["bw_frac"] < 0.01


def test_spectral_fractions_of_white_noise_and_mixtures():
    rng = np.random.default_rng(0)
    s = gf.sqi_per_lead(rng.standard_normal(N))
    assert s["hf_frac"] == pytest.approx((250 - 40) / 250, abs=0.03)   # flat spectrum
    mix = gf.sqi_per_lead(np.sin(2 * np.pi * 10 * T) + np.sin(2 * np.pi * 60 * T))
    assert mix["hf_frac"] == pytest.approx(0.5, abs=0.01)


def test_ptp_kurtosis_and_flat_fraction():
    rng = np.random.default_rng(1)
    g = gf.sqi_per_lead(rng.standard_normal(N))
    assert g["kurtosis"] == pytest.approx(3.0, abs=0.3) and g["flat_frac"] == 0.0
    sine = gf.sqi_per_lead(2.0 * np.sin(2 * np.pi * 5 * T))
    assert sine["ptp_mv"] == pytest.approx(4.0, abs=1e-3) and sine["kurtosis"] == pytest.approx(1.5, abs=0.01)
    x = rng.standard_normal(N)
    x[1000:1050] = 0.25                                   # exactly 50 identical samples -> flat
    assert gf.sqi_per_lead(x)["flat_frac"] == pytest.approx(50 / N)
    x[1000:1050] = rng.standard_normal(50)
    x[2000:2049] = 0.25                                   # 49 samples -> not flat
    assert gf.sqi_per_lead(x)["flat_frac"] == 0.0
    y = rng.standard_normal(N)
    y[3000:3100] = 1.0 + np.tile([0.0, 0.0004], 50)       # varies by < half an ADC step -> flat
    assert gf.sqi_per_lead(y)["flat_frac"] == pytest.approx(100 / N)


def test_constant_lead_and_sentinel_aggregation():
    c = gf.sqi_per_lead(np.full(N, 0.7))
    assert c["ptp_mv"] == 0.0 and c["flat_frac"] == 1.0
    assert all(np.isnan(c[s]) for s in ("bw_frac", "hf_frac", "kurtosis"))
    only_constant = gf.aggregate([c])
    assert only_constant["g2_kurtosis_mean"] == gf.SENTINEL and only_constant["g2_bw_frac_max"] == gf.SENTINEL
    assert only_constant["g2_flat_frac_mean"] == 1.0
    live = gf.sqi_per_lead(np.sin(2 * np.pi * 5 * T))
    mixed = gf.aggregate([c, live])                       # undefined values are skipped, defined ones kept
    assert mixed["g2_kurtosis_mean"] == pytest.approx(live["kurtosis"]) and mixed["g2_flat_frac_mean"] == pytest.approx(0.5)
    assert mixed["g2_ptp_mv_min"] == 0.0 and mixed["g2_ptp_mv_max"] == pytest.approx(live["ptp_mv"])


def test_non_finite_and_wrong_window_are_refused():
    x = np.zeros(N)
    x[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        gf.sqi_per_lead(x)
    with pytest.raises(ValueError, match="window"):
        gf.sqi_per_lead(np.zeros(N - 1))


# --------------------------------------------------------------------------- #
# Reference invariance and determinism (synthetic and real raw records)
# --------------------------------------------------------------------------- #
def _real_raw(role: str = "B_gate", i: int = 0) -> tuple[int, np.ndarray]:
    audit = io.load_config(REPO_ROOT / "configs" / "data_audit.yaml")
    man = pd.read_csv(REPO_ROOT / "artifacts" / "manifests" / "split_manifest_v1_frozen.csv")
    e = int(man.loc[man.role == role, "ecg_id"].sort_values().iloc[i])
    db = pd.read_csv(REPO_ROOT / audit["dataset"]["database_csv"], usecols=["ecg_id", "filename_hr"]).set_index("ecg_id")
    try:
        root = resolve_waveform_root(audit, REPO_ROOT)
        return e, load_record(root, db.loc[e, "filename_hr"], "hr")
    except Exception as exc:  # pragma: no cover - raw waveforms not available
        pytest.skip(f"raw waveforms not available: {exc}")


@pytest.mark.parametrize("source", ["synthetic", "real"])
def test_reference_invariance_missing_leads_never_matter(source):
    rng = np.random.default_rng(2)
    raw = rng.standard_normal((12, 5000)) if source == "synthetic" else _real_raw()[1]
    for e in (5, 77, 4096):
        for observed in mz.evaluation_combinations(e, 20261030, k=8):
            base = gf.row_features(raw, observed)
            missing = [j for j, lead in enumerate(LEAD_NAMES) if lead not in observed]
            for replacement in (rng.standard_normal((len(missing), 5000)) * 50, np.full((len(missing), 5000), np.nan)):
                other = raw.copy()
                other[missing] = replacement
                assert gf.row_features(other, observed) == base     # bit-identical, even with NaN in missing leads
            assert gf.row_features(raw, observed) == base           # deterministic


def test_raw_window_aligns_with_the_250hz_evaluation_window():
    cfg = io.load_config(REPO_ROOT / "configs" / "preprocessing.yaml")
    impulse = np.zeros((12, 5000))
    impulse[:, 2000] = 5.0                               # raw sample 2000 = 4.000 s
    filtered = pp.filter_resample(impulse, 500, cfg)
    assert filtered.shape[1] == 2500 and int(np.argmax(np.abs(filtered[0]))) == 1000   # 4.000 s at 250 Hz, zero lag
    assert gf.WINDOW == (0, 4096) and 4096 // 2 == int(cfg["window"]["length"]) and int(cfg["window"]["evaluation_offset"]) == 0


def test_cache_window_comes_from_the_same_raw_samples():
    e, raw = _real_raw()
    cfg = io.load_config(REPO_ROOT / "configs" / "preprocessing.yaml")
    idx = pd.read_csv("C:/ecg_cache/v0_1/B_gate_index.csv")
    pos = int(np.flatnonzero(idx.ecg_id.to_numpy() == e)[0])
    cached = np.load("C:/ecg_cache/v0_1/B_gate_signals.npy", mmap_mode="r")[pos]
    assert np.array_equal(cached, pp.filter_resample(raw, 500, cfg).astype(np.float32))  # cache = filtered raw, same origin


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def test_feature_code_cannot_open_offline_label_files(tmp_path):
    with pytest.raises(PermissionError):
        gf.assert_not_offline("artifacts/d1/v0_1/offline_labels/B_gate_OFFLINE_d1_labels.csv")
    with pytest.raises(PermissionError):
        gf.assert_not_offline(tmp_path / "x_OFFLINE_e1_complete_by_record.csv")
    assert gf.assert_not_offline("artifacts/d1/v0_1/keys/B_gate_rows.csv").name == "B_gate_rows.csv"
    # static checks: the extractor has no string pointing at offline files; features.py does no file I/O at all
    tree = ast.parse(inspect.getsource(ef))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)}
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    path_like = [s for s in literals if s == "offline_labels" or "offline_labels/" in s or "offline_labels\\" in s
                 or "OFFLINE_" in s or "_OFFLINE" in s]
    assert not path_like, path_like
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
             for n in ast.walk(ast.parse(inspect.getsource(gf))) if isinstance(n, ast.Call)}
    assert not calls & {"open", "read_csv", "read_text", "load", "rdrecord", "load_record"}


def test_feature_columns_pass_the_guard_and_comparators_do_not(p2):
    assert check_feature_columns(gf.G1_COLUMNS + gf.G2_COLUMNS, p2) == []
    assert check_feature_columns(["e1_recon_margin_score", "edir_outputs", "g3_reconstruction_statistics", "lolo_features"], p2) \
        == ["e1_recon_margin_score", "edir_outputs", "g3_reconstruction_statistics", "lolo_features"]


@pytest.mark.parametrize("role", ["test_fold10", "A_fit", "C_dev"])
def test_extractor_refuses_non_gate_roles(role):
    with pytest.raises(RuntimeError):
        ef.check_role(role)


def test_m2_score_formula():
    p = np.array([[0.31639263, 0.9], [0.1, 0.35816982], [0.5, 0.5], [0.0, 1.0]], np.float32)
    tau = np.array([0.31639263, 0.35816982], np.float32)
    s = cmp.m2_scores(p, tau)
    assert s[0] == 0.0 and s[1] == 0.0                    # on the boundary = riskiest
    assert s[2] == pytest.approx(-min(0.5 - 0.31639263, 0.5 - 0.35816982), abs=1e-7)
    assert (s <= 0).all() and s[3] < s[2]


# --------------------------------------------------------------------------- #
# Extracted artifacts (skipped until M4b-0 extraction has run)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def extracted():
    if not (OUT / "extraction_summary.json").is_file():
        pytest.skip("M4b-0 features not extracted yet")
    return json.loads((OUT / "extraction_summary.json").read_text(encoding="utf-8"))


def test_extraction_flags_and_hashes(extracted):
    assert extracted["d1_joined"] is False and extracted["models_fitted"] is False and extracted["fold10_loaded"] is False
    assert extracted["offline_labels_read"] is False and extracted["protocol_v0_2_pinned"] is False
    man = json.loads((OUT / "hash_manifest.json").read_text(encoding="utf-8"))
    for rel, sha in man["files"].items():
        assert io.sha256_file(REPO_ROOT / rel) == sha, rel


@pytest.mark.parametrize("role", ROLES)
def test_feature_files_match_keys_and_reference_implementation(extracted, role):
    feats = pd.read_csv(OUT / "features" / f"{role}_features.csv", float_precision="round_trip")  # exact float64 parse
    keys = pd.read_csv(REPO_ROOT / "artifacts" / "d1" / "v0_1" / "keys" / f"{role}_rows.csv")
    assert list(feats.columns) == ["row_id"] + gf.G1_COLUMNS + gf.G2_COLUMNS
    assert (feats.row_id == keys.row_id).all() and np.isfinite(feats.drop(columns="row_id").to_numpy(float)).all()
    for obs, (_, r) in zip(keys.observed_leads.str.split(";"), feats.iterrows()):  # G1 = mask geometry
        if r["g1_n_observed"] != len(obs):
            raise AssertionError(r["row_id"])
    # batch extraction equals the reference row implementation on raw data (first record's 8 rows)
    e, raw = keys.ecg_id.iloc[0], None
    audit = io.load_config(REPO_ROOT / "configs" / "data_audit.yaml")
    db = pd.read_csv(REPO_ROOT / audit["dataset"]["database_csv"], usecols=["ecg_id", "filename_hr"]).set_index("ecg_id")
    raw = load_record(resolve_waveform_root(audit, REPO_ROOT), db.loc[e, "filename_hr"], "hr")
    for i in np.flatnonzero(keys.ecg_id.to_numpy() == e):
        ref = gf.row_features(raw, keys.observed_leads.iloc[i].split(";"))
        got = feats.iloc[i]
        assert all(float(got[c]) == float(ref[c]) for c in gf.G1_COLUMNS + gf.G2_COLUMNS), keys.row_id.iloc[i]


def test_m2_comparator_files(extracted):
    summ = OUT / "comparators" / "comparators_summary.json"
    if not summ.is_file():
        pytest.skip("M2 comparator scores not written yet")
    s = json.loads(summ.read_text(encoding="utf-8"))
    for role, info in s["per_role"].items():
        assert info["columns_read"] == ["row_id", "p_rec_CLBBB", "p_rec_CRBBB"]
        df = pd.read_csv(OUT / "comparators" / f"{role}_M2_e1_recon_margin.csv")
        assert list(df.columns) == ["row_id", "e1_recon_margin_score"] and (df.e1_recon_margin_score <= 0).all()
        assert io.sha256_file(OUT / "comparators" / f"{role}_M2_e1_recon_margin.csv") == info["output_sha256"]
