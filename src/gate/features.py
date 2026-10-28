"""Gate features (protocol v0.2; formulas fixed in configs/gate_m4b_v0_2.yaml before any feature was computed).

G1 mask geometry (15 columns) depends only on the observed-lead set. G2 observed-lead signal quality (15 columns) is computed
from the RAW 500 Hz WFDB signal (mV, unfiltered) of the OBSERVED leads only, over raw samples [0, 4096):
    per lead: ptp_mv, bw_frac (< 0.5 Hz), hf_frac (> 40 Hz), flat_frac, kurtosis; then mean/min/max over observed leads.
Inference safety by construction: ``row_features`` slices the observed leads out of the record before any SQI is computed,
so the missing leads cannot influence a feature (reference-invariance test). This module never reads D1, labels, E1/E_dir
outputs or any file under ``offline_labels``; ``assert_not_offline`` guards every path the feature code opens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import periodogram
from scipy.stats import kurtosis

from src.data.ptbxl_waveforms import LEAD_INDEX, LEAD_NAMES

SQIS = ("ptp_mv", "bw_frac", "hf_frac", "flat_frac", "kurtosis")
AGGS = ("mean", "min", "max")
G1_COLUMNS = ["g1_n_observed", *[f"g1_obs_{lead}" for lead in LEAD_NAMES], "g1_n_limb_observed", "g1_n_precordial_observed"]
G2_COLUMNS = [f"g2_{s}_{a}" for s in SQIS for a in AGGS]
SENTINEL = -1.0

# Constants transcribed from configs/gate_m4b_v0_2.yaml; ``check_config`` refuses to run if the config differs.
FS = 500
WINDOW = (0, 4096)
BW_CUTOFF_HZ = 0.5
HF_CUTOFF_HZ = 40.0
FLAT_TOL_MV = 0.0005
FLAT_MIN_RUN = 50


def check_config(cfg: dict[str, Any]) -> None:
    """The numbers used here must be the pre-registered ones."""
    g1, g2 = cfg["g1"], cfg["g2"]
    expected = {
        "lead_order": (list(g1["lead_order"]), list(LEAD_NAMES)),
        "fs_hz": (int(g2["fs_hz"]), FS),
        "window_samples": (tuple(g2["window_samples"]), WINDOW),
        "limb_leads": (list(g1["limb_leads"]), list(LEAD_NAMES[:6])),
        "precordial_leads": (list(g1["precordial_leads"]), list(LEAD_NAMES[6:])),
    }
    bad = {k: v for k, v in expected.items() if v[0] != v[1]}
    for text, needle in ((g2["per_lead"]["bw_frac"], "0 < f < 0.5 Hz"), (g2["per_lead"]["hf_frac"], "f > 40 Hz"),
                         (g2["per_lead"]["flat_frac"], ">= 50 consecutive samples"), (g2["per_lead"]["flat_frac"], "<= 0.0005 mV"),
                         (g2["per_lead"]["kurtosis"], "fisher=False, bias=True"), (g2["per_lead"]["psd"], "window='hann'"),
                         (g2["aggregation"], "-1.0")):
        if needle not in text:
            bad[needle] = "not found in the pre-registered config text"
    if bad:
        raise RuntimeError(f"gate config differs from the implemented formulas: {bad}")


def assert_not_offline(path: str | Path) -> Path:
    """Feature code may never open offline label files (D1, E1 scores, labels)."""
    p = Path(path)
    if "offline_labels" in p.parts or "OFFLINE" in p.name:
        raise PermissionError(f"feature code must not read offline label files: {p}")
    return p


def g1_features(observed: Sequence[str]) -> dict[str, int]:
    obs = set(observed)
    unknown = obs - set(LEAD_NAMES)
    if unknown or not obs:
        raise ValueError(f"invalid observed leads {list(observed)}")
    out = {"g1_n_observed": len(obs)}
    out.update({f"g1_obs_{lead}": int(lead in obs) for lead in LEAD_NAMES})
    out["g1_n_limb_observed"] = sum(lead in obs for lead in LEAD_NAMES[:6])
    out["g1_n_precordial_observed"] = sum(lead in obs for lead in LEAD_NAMES[6:])
    return out


def flat_fraction(x: np.ndarray, tol: float = FLAT_TOL_MV, min_run: int = FLAT_MIN_RUN) -> float:
    """Share of samples inside a run of >= min_run consecutive samples whose successive |differences| are all <= tol."""
    small = np.abs(np.diff(x)) <= tol                     # small[i]: x[i+1] ~ x[i]
    edges = np.diff(np.r_[0, small.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)  # runs of differences [start, end)
    flat = np.zeros(len(x), dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s) + 1 >= min_run:                          # k differences cover k + 1 samples
            flat[s:e + 1] = True
    return float(flat.mean())


def sqi_per_lead(x: np.ndarray) -> dict[str, float]:
    """The five SQIs of one lead's raw window (mV). Undefined values are NaN (only for a constant lead)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) != WINDOW[1] - WINDOW[0]:
        raise ValueError(f"expected a 1-D window of {WINDOW[1] - WINDOW[0]} samples, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("non-finite raw samples")
    ptp = float(x.max() - x.min())
    out = {"ptp_mv": ptp, "flat_frac": flat_fraction(x), "bw_frac": np.nan, "hf_frac": np.nan, "kurtosis": np.nan}
    if ptp == 0.0:                                          # constant lead: spectral fractions and kurtosis undefined
        return out
    x0 = x - x.mean()
    f, p = periodogram(x0, fs=FS, window="hann", nfft=len(x0), detrend=False, return_onesided=True, scaling="density")
    total = p[(f > 0) & (f <= FS / 2)].sum()
    out["bw_frac"] = float(p[(f > 0) & (f < BW_CUTOFF_HZ)].sum() / total)
    out["hf_frac"] = float(p[f > HF_CUTOFF_HZ].sum() / total)
    out["kurtosis"] = float(kurtosis(x0, fisher=False, bias=True))
    return out


def aggregate(per_lead: Sequence[dict[str, float]]) -> dict[str, float]:
    """mean/min/max of each SQI over the observed leads with a defined value; sentinel -1.0 if none is defined."""
    out: dict[str, float] = {}
    for s in SQIS:
        vals = np.array([d[s] for d in per_lead], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out.update({f"g2_{s}_mean": SENTINEL, f"g2_{s}_min": SENTINEL, f"g2_{s}_max": SENTINEL})
        else:
            out.update({f"g2_{s}_mean": float(vals.mean()), f"g2_{s}_min": float(vals.min()), f"g2_{s}_max": float(vals.max())})
    return out


def observed_window(raw_record: np.ndarray, observed: Sequence[str]) -> np.ndarray:
    """(n_observed, 4096) copy of the observed leads' raw window; the only signal data G2 ever sees."""
    raw_record = np.asarray(raw_record)
    if raw_record.shape[0] != len(LEAD_NAMES) or raw_record.shape[1] < WINDOW[1]:
        raise ValueError(f"expected a (12, >= {WINDOW[1]}) raw record, got {raw_record.shape}")
    idx = [LEAD_INDEX[lead] for lead in observed]
    return np.array(raw_record[idx, WINDOW[0]:WINDOW[1]], dtype=np.float64, copy=True)


def row_features(raw_record: np.ndarray, observed: Sequence[str]) -> dict[str, float]:
    """G1 + G2 for one (record, observed-lead set) row. Reference implementation used by the tests."""
    obs = observed_window(raw_record, observed)
    return {**g1_features(observed), **aggregate([sqi_per_lead(sig) for sig in obs])}
