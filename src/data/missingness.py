"""Missing-lead masks, model-input assembly and copy-through.

Mask convention (PROJECT, design document Rev. 2.0): 1 = observed, 0 = missing.
Missing leads are filled with 0 (PAPER p. 1; the released code does the same).

* ``l2g_lead_combinations()``: the 258 observed-lead sets evaluated in L2G-ECG (PAPER pp. 7-8):
  each of the 12 leads alone, plus every 2..7-lead subset of {I, III, V1, ..., V6}.
* ``build_model_input()``: AD-1 (PROJECT). The 12 masked signal channels plus 12 mask channels = 24.
* ``copy_through()``: AD-2 (PROJECT). Observed samples are returned bit-for-bit unchanged.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from src.data.ptbxl_waveforms import LEAD_INDEX, LEAD_NAMES

# PAPER p. 7: "leads from the set I = {I, III, V1, V2, V3, V4, V5, V6}" for 2+ available leads
# (II, aVR, aVL, aVF are linear combinations of I and III by Einthoven/Goldberger).
L2G_MULTI_LEAD_POOL: tuple[str, ...] = ("I", "III", "V1", "V2", "V3", "V4", "V5", "V6")


def l2g_lead_combinations() -> list[tuple[str, ...]]:
    """The 258 observed-lead sets of L2G-ECG, in a fixed, documented order."""
    singles = [(lead,) for lead in LEAD_NAMES]
    multi = [combo for size in range(2, 8) for combo in combinations(L2G_MULTI_LEAD_POOL, size)]
    return singles + multi


def lead_mask(observed: Sequence[str]) -> np.ndarray:
    """(12,) uint8 vector: 1 for each observed lead."""
    unknown = [lead for lead in observed if lead not in LEAD_INDEX]
    if unknown or not observed:
        raise ValueError(f"observed leads must be a non-empty subset of {LEAD_NAMES}; got {list(observed)}")
    mask = np.zeros(len(LEAD_NAMES), dtype=np.uint8)
    mask[[LEAD_INDEX[lead] for lead in observed]] = 1
    return mask


def signal_mask(observed: Sequence[str], n_samples: int) -> np.ndarray:
    """(12, n_samples) float32 mask, constant over time (whole leads are missing)."""
    return np.repeat(lead_mask(observed).astype(np.float32)[:, None], n_samples, axis=1)


def _check_mask(mask: np.ndarray, shape: tuple[int, ...]) -> None:
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match signal shape {shape}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask must contain only 0 and 1")


def apply_mask(x: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Masked ECG: observed samples kept, missing samples set to ``fill_value``."""
    _check_mask(mask, x.shape)
    return np.where(mask == 1, x, np.asarray(fill_value, dtype=x.dtype))


def build_model_input(x_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(24, n) model input = masked signal channels stacked on mask channels (AD-1)."""
    _check_mask(mask, x_obs.shape)
    if np.any(x_obs[mask == 0] != 0):
        raise ValueError("x_obs has non-zero values at missing positions; call apply_mask first")
    return np.concatenate([x_obs, mask.astype(x_obs.dtype)], axis=0)


def copy_through(x_obs: np.ndarray, x_gen: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """x_hat = m * x_obs + (1 - m) * x_gen, computed with np.where so that observed samples
    are copied exactly (arithmetic blending could change them if x_gen were NaN or inf)."""
    if x_obs.shape != x_gen.shape:
        raise ValueError(f"x_obs {x_obs.shape} and x_gen {x_gen.shape} differ in shape")
    _check_mask(mask, x_obs.shape)
    return np.where(mask == 1, x_obs, x_gen.astype(x_obs.dtype, copy=False))


def sample_training_combination(rng: np.random.Generator,
                                combos: Sequence[tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """Uniform draw over the L2G combinations (AD-4; the paper does not specify sampling)."""
    combos = combos or l2g_lead_combinations()
    return combos[int(rng.integers(len(combos)))]


def evaluation_combinations(ecg_id: int, seed: int, k: int = 1,
                            combos: Sequence[tuple[str, ...]] | None = None) -> list[tuple[str, ...]]:
    """k distinct combinations that depend only on (seed, ecg_id), never on processing order."""
    combos = combos or l2g_lead_combinations()
    if not 1 <= k <= len(combos):
        raise ValueError(f"k must be between 1 and {len(combos)}")
    rng = np.random.default_rng([int(seed), int(ecg_id)])
    return [combos[i] for i in rng.choice(len(combos), size=k, replace=False)]
