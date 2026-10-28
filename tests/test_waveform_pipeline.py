"""Tests for the waveform pipeline: loader, preprocessing, masks, copy-through.

Synthetic tests use signals whose correct answer is known (pure sines, constant leads).
Real-data tests read a few PTB-XL records from the A_fit role of the FROZEN manifest only
(never fold 10). They are skipped with a reason if the waveform folder is not available.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.data import missingness as mz
from src.data import preprocess as pp
from src.data import ptbxl_io as io
from src.data.ptbxl_waveforms import LEAD_NAMES, load_by_ecg_id, resolve_waveform_root
from tests.conftest import REPO_ROOT

FS, N = 500, 5000
T = np.arange(N) / FS


@pytest.fixture
def pcfg() -> dict:
    return io.load_config(REPO_ROOT / "configs" / "preprocessing.yaml")


def _sine(freq: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * T)


def _amplitude(x: np.ndarray, fs: float, freq: float) -> float:
    """Amplitude of one frequency component (signal length is a whole number of periods)."""
    t = np.arange(x.size) / fs
    return float(2 * np.abs(np.mean(x * np.exp(-2j * np.pi * freq * t))))


# --------------------------------------------------------------------------- #
# Settings must not drift silently from the paper (D-014)
# --------------------------------------------------------------------------- #
def test_paper_settings_unchanged(pcfg):
    assert pcfg["lead_order"] == list(LEAD_NAMES)
    assert (pcfg["highpass"]["order"], pcfg["highpass"]["cutoff_hz"]) == (6, 0.5)
    assert pcfg["notch"]["freq_hz"] == 50.0
    assert pcfg["resample"]["target_fs"] == 250
    assert pcfg["window"]["length"] == 2048
    assert pcfg["normalization"]["method"] == "per_lead_minmax"
    assert pcfg["masks"]["missing_fill_value"] == 0.0
    assert pcfg["source"] == {"resolution": "hr", "fs": 500}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_shape_dtype_range_and_rate(pcfg):
    rng = np.random.default_rng(0)
    x = np.stack([_sine(1 + i, 0.5) + 0.05 * rng.standard_normal(N) for i in range(12)])
    out = pp.preprocess(x, FS, pcfg)
    assert out.x.shape == (12, 2048) and out.x.dtype == np.float32 and out.fs == 250
    assert np.isfinite(out.x).all()
    assert (out.x.min(axis=1) == 0.0).all() and (out.x.max(axis=1) == 1.0).all()
    assert not out.flat_leads.any()


def test_highpass_removes_baseline_wander_keeps_qrs_band():
    x = np.stack([_sine(0.1, 1.0) + _sine(10.0, 0.5)] * 12)
    y = pp.highpass(x, FS, 6, 0.5, zero_phase=True)
    mid = slice(1000, 4000)  # ignore the first/last 2 s, where filter edge effects live
    assert _amplitude(y[0], FS, 0.1) < 0.01  # baseline wander (1 mV) removed
    assert abs(_amplitude(y[0], FS, 10.0) - 0.5) < 0.005  # 10 Hz content kept within 1%
    assert np.abs(y[0, mid] - _sine(10.0, 0.5)[mid]).max() < 0.02


def test_notch_removes_mains_keeps_neighbours():
    x = np.stack([_sine(50.0, 1.0) + _sine(10.0, 0.5)] * 12)
    y = pp.notch(x, FS, 50.0, 30.0, zero_phase=True)
    assert _amplitude(y[0, 500:4500], FS, 50.0) < 0.02
    assert abs(_amplitude(y[0, 500:4500], FS, 10.0) - 0.5) < 0.005


def test_resample_halves_length_and_keeps_frequency():
    x = np.stack([_sine(5.0)] * 12)
    y = pp.resample(x, 500, 250)
    t250 = np.arange(2500) / 250
    assert y.shape == (12, 2500)
    assert np.abs(y[0, 100:2400] - np.sin(2 * np.pi * 5.0 * t250[100:2400])).max() < 1e-3


def test_crop_bounds():
    x = np.zeros((12, 2500))
    assert pp.crop(x, 2048, 452).shape == (12, 2048)
    with pytest.raises(ValueError):
        pp.crop(x, 2048, 453)
    with pytest.raises(ValueError):
        pp.crop(x, 2048, -1)


def test_flat_lead_becomes_zero_and_is_flagged(pcfg):
    x = np.stack([_sine(5.0)] * 12)
    x[3] = 0.25  # constant lead (e.g. a disconnected electrode)
    out = pp.preprocess(x, FS, pcfg)
    assert out.flat_leads.tolist() == [i == 3 for i in range(12)]
    assert (out.x[3] == 0).all() and np.isfinite(out.x).all()


@pytest.mark.parametrize("bad", ["fs", "shape", "nan"])
def test_preprocess_rejects_bad_input(pcfg, bad):
    x = np.stack([_sine(5.0)] * 12)
    with pytest.raises(ValueError):
        if bad == "fs":
            pp.preprocess(x[:, ::5], 100, pcfg)
        elif bad == "shape":
            pp.preprocess(x[:11], FS, pcfg)
        else:
            x[0, 10] = np.nan
            pp.preprocess(x, FS, pcfg)


# --------------------------------------------------------------------------- #
# Masks (D-015)
# --------------------------------------------------------------------------- #
def test_l2g_combinations_match_paper():
    combos = mz.l2g_lead_combinations()
    assert len(combos) == 258  # PAPER p. 8: 12C1 + sum_{i=2..7} 8Ci = 258
    assert combos[:12] == [(lead,) for lead in LEAD_NAMES]
    sizes = Counter(len(c) for c in combos[12:])
    assert sizes == {2: 28, 3: 56, 4: 70, 5: 56, 6: 28, 7: 8}
    assert all(set(c) <= set(mz.L2G_MULTI_LEAD_POOL) for c in combos[12:])
    assert len(set(combos)) == 258


def test_lead_and_signal_masks():
    m = mz.signal_mask(("I", "III", "V2"), 2048)
    assert m.shape == (12, 2048) and m.dtype == np.float32
    assert m[:, 0].tolist() == [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert (m == m[:, :1]).all()  # whole leads, constant in time
    for bad in [(), ("aVR",), ("I", "X1")]:
        with pytest.raises(ValueError):
            mz.lead_mask(bad)


def test_apply_mask_and_model_input():
    rng = np.random.default_rng(1)
    x = rng.random((12, 2048)).astype(np.float32)
    m = mz.signal_mask(("II", "V5"), 2048)
    x_obs = mz.apply_mask(x, m)
    assert (x_obs[m == 0] == 0).all() and np.array_equal(x_obs[m == 1], x[m == 1])
    model_in = mz.build_model_input(x_obs, m)
    assert model_in.shape == (24, 2048) and model_in.dtype == np.float32
    assert np.array_equal(model_in[:12], x_obs) and np.array_equal(model_in[12:], m)
    with pytest.raises(ValueError):
        mz.build_model_input(x, m)  # not masked


def test_copy_through_preserves_observed_exactly_for_every_combination():
    rng = np.random.default_rng(2)
    x = rng.random((12, 2048)).astype(np.float32)
    for observed in mz.l2g_lead_combinations():
        m = mz.signal_mask(observed, 2048)
        x_obs = mz.apply_mask(x, m)
        x_gen = rng.random((12, 2048)).astype(np.float32)
        x_gen[m == 1] = np.nan  # garbage at observed positions must not leak through
        x_hat = mz.copy_through(x_obs, x_gen, m)
        assert np.array_equal(x_hat[m == 1].view(np.uint32), x[m == 1].view(np.uint32))  # bit-exact
        assert np.array_equal(x_hat[m == 0], x_gen[m == 0])


def test_copy_through_rejects_bad_inputs():
    x = np.zeros((12, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        mz.copy_through(x, np.zeros((12, 11), dtype=np.float32), np.ones((12, 10)))
    with pytest.raises(ValueError):
        mz.copy_through(x, x, np.full((12, 10), 0.5))


def test_evaluation_masks_are_deterministic_per_record():
    a = mz.evaluation_combinations(ecg_id=123, seed=20261030, k=3)
    assert a == mz.evaluation_combinations(ecg_id=123, seed=20261030, k=3)
    assert len(set(a)) == 3
    others = {tuple(mz.evaluation_combinations(ecg_id=i, seed=20261030)) for i in range(50)}
    assert len(others) > 30  # different records get different masks


def test_training_sampling_covers_all_combinations_uniformly():
    rng = np.random.default_rng(3)
    combos = mz.l2g_lead_combinations()
    counts = Counter(mz.sample_training_combination(rng, combos) for _ in range(258 * 60))
    assert len(counts) == 258
    # Chi-square goodness-of-fit against the uniform distribution (seeded, so deterministic).
    from scipy.stats import chisquare
    assert chisquare([counts[c] for c in combos]).pvalue > 0.001


# --------------------------------------------------------------------------- #
# Real PTB-XL waveforms (A_fit records of the frozen manifest only)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_waveforms(real_data):
    try:
        root = resolve_waveform_root(real_data.config, REPO_ROOT)
    except io.MissingInputError as exc:
        pytest.skip(f"PTB-XL waveforms not available: {exc}")
    from src.data.freeze_split_manifest import load_frozen_manifest
    manifest = load_frozen_manifest(real_data.config, REPO_ROOT)
    a_fit = manifest.loc[manifest["role"] == "A_fit"].sort_values("ecg_id")
    return root, a_fit["ecg_id"].head(5).tolist(), manifest


def test_real_records_have_expected_structure(real_data, real_waveforms):
    root, ecg_ids, manifest = real_waveforms
    assert set(manifest.loc[manifest["ecg_id"].isin(ecg_ids), "strat_fold"]) <= set(range(1, 9))
    for ecg_id in ecg_ids:
        hr = load_by_ecg_id(real_data.db, root, ecg_id, "hr")
        lr = load_by_ecg_id(real_data.db, root, ecg_id, "lr")
        assert hr.shape == (12, 5000) and lr.shape == (12, 1000)
        assert np.isfinite(hr).all() and np.isfinite(lr).all()
        assert 0 < np.abs(hr).max() < 50  # plausible millivolt range


def test_real_end_to_end_smoke(real_data, real_waveforms, pcfg):
    """Load -> preprocess -> mask -> 24-channel input -> stand-in generator -> copy-through."""
    root, ecg_ids, _ = real_waveforms
    rng = np.random.default_rng(4)
    combos = [("I",), ("I", "III", "V2"), ("I", "III", "V1", "V2", "V3", "V4", "V5")]
    for ecg_id in ecg_ids[:3]:
        out = pp.preprocess(load_by_ecg_id(real_data.db, root, ecg_id, "hr"), 500, pcfg)
        assert out.x.shape == (12, 2048) and np.isfinite(out.x).all()
        assert out.x.min() >= 0 and out.x.max() <= 1
        for observed in combos:
            m = mz.signal_mask(observed, 2048)
            x_obs = mz.apply_mask(out.x, m)
            model_in = mz.build_model_input(x_obs, m)
            assert model_in.shape == (24, 2048)
            x_gen = rng.random((12, 2048)).astype(np.float32)  # stand-in for the future generator
            x_hat = mz.copy_through(x_obs, x_gen, m)
            assert np.array_equal(x_hat[m == 1], out.x[m == 1])
            assert np.isfinite(x_hat).all() and x_hat.shape == (12, 2048)
