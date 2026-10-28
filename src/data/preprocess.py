"""Waveform preprocessing following L2G-ECG (paper p. 3), adapted to PTB-XL.

Order (PAPER): [signal-quality assessment] -> high-pass + notch filter -> resample to 250 Hz
-> cut a 2048-sample window -> per-lead min-max normalisation to [0, 1].
Settings come from configs/preprocessing.yaml, where each one is tagged PAPER / CODE / PROJECT.
The signal-quality step is not reproduced (D-017).

Input:  (12, n) array in mV at the source sampling rate (from ptbxl_waveforms.load_record).
Output: PreprocessResult with x of shape (12, 2048), float32, every lead in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
from scipy import signal as sps

N_LEADS = 12


@dataclass(frozen=True)
class PreprocessResult:
    x: np.ndarray            # (12, window) float32, each lead in [0, 1]
    lead_min_mV: np.ndarray  # (12,) minimum of each filtered, resampled, windowed lead (mV)
    lead_max_mV: np.ndarray  # (12,) maximum (mV); kept for offline analysis only
    flat_leads: np.ndarray   # (12,) bool: lead range < epsilon, so the lead was set to 0
    fs: int                  # sampling rate of x (Hz)


def _check_input(x: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[0] != N_LEADS:
        raise ValueError(f"expected a (12, n_samples) array, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input contains NaN or infinite values")


def highpass(x: np.ndarray, fs: float, order: int, cutoff_hz: float, zero_phase: bool) -> np.ndarray:
    """Butterworth high-pass along time (axis 1) to remove baseline wander."""
    sos = sps.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x, axis=1) if zero_phase else sps.sosfilt(sos, x, axis=1)


def notch(x: np.ndarray, fs: float, freq_hz: float, quality_factor: float, zero_phase: bool) -> np.ndarray:
    """IIR notch at the mains frequency."""
    b, a = sps.iirnotch(freq_hz, quality_factor, fs=fs)
    return sps.filtfilt(b, a, x, axis=1) if zero_phase else sps.lfilter(b, a, x, axis=1)


def resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Anti-aliased polyphase resampling (500 -> 250 Hz is up=1, down=2)."""
    if fs_in == fs_out:
        return x
    g = gcd(int(fs_in), int(fs_out))
    return sps.resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=1)


def crop(x: np.ndarray, length: int, offset: int) -> np.ndarray:
    """Samples [offset, offset + length) of every lead. Refuses to pad."""
    if offset < 0 or offset + length > x.shape[1]:
        raise ValueError(f"window [{offset}, {offset + length}) does not fit in {x.shape[1]} samples")
    return x[:, offset:offset + length]


def minmax_normalize(x: np.ndarray, flat_eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-lead min-max to [0, 1] (PAPER eq. 9). Flat leads (range < flat_eps) become 0."""
    lo, hi = x.min(axis=1), x.max(axis=1)
    rng = hi - lo
    flat = rng < flat_eps
    safe = np.where(flat, 1.0, rng)
    out = (x - lo[:, None]) / safe[:, None]
    out[flat] = 0.0
    return np.clip(out, 0.0, 1.0), lo, hi, flat


def filter_resample(x_mV: np.ndarray, fs: int, cfg: dict[str, Any]) -> np.ndarray:
    """Steps before windowing: checks, high-pass, notch, resample. Returns (12, n_target) in mV.
    Split out so the slow part can be cached once per record (Milestone 2c training cache)."""
    _check_input(x_mV)
    if fs != cfg["source"]["fs"]:
        raise ValueError(f"input sampling rate {fs} Hz differs from configured source rate {cfg['source']['fs']} Hz")
    if cfg.get("signal_quality_assessment", "none") != "none":
        raise NotImplementedError("signal-quality assessment is not implemented (decision D-017)")
    hp, nt = cfg["highpass"], cfg["notch"]
    y = highpass(x_mV, fs, hp["order"], hp["cutoff_hz"], hp["zero_phase"])
    y = notch(y, fs, nt["freq_hz"], nt["quality_factor"], nt["zero_phase"])
    return resample(y, fs, int(cfg["resample"]["target_fs"]))


def crop_normalize(y: np.ndarray, cfg: dict[str, Any], offset: int | None = None) -> PreprocessResult:
    """Steps after resampling: cut the 2048-sample window, then per-lead min-max to [0, 1]."""
    win = cfg["window"]
    y = crop(y, int(win["length"]), int(win["evaluation_offset"] if offset is None else offset))
    norm, lo, hi, flat = minmax_normalize(y, float(cfg["normalization"]["flat_lead_epsilon_mV"]))
    out = norm.astype(cfg.get("output_dtype", "float32"))
    if not np.isfinite(out).all():
        raise ValueError("preprocessing produced non-finite values")
    return PreprocessResult(x=out, lead_min_mV=lo, lead_max_mV=hi, flat_leads=flat, fs=int(cfg["resample"]["target_fs"]))


def preprocess(x_mV: np.ndarray, fs: int, cfg: dict[str, Any], offset: int | None = None) -> PreprocessResult:
    """Run the full chain. ``cfg`` is the parsed configs/preprocessing.yaml."""
    return crop_normalize(filter_resample(x_mV, fs, cfg), cfg, offset)
