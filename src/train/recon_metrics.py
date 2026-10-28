"""Reconstruction metrics and the two pre-declared non-learned baselines (Milestone 2c).

All signals are (B, 12, T) in the normalised [0, 1] space. ``lead_mask`` is (B, 12) with 1 = observed.
"Missing-lead" metrics average only over missing leads, because copy-through makes observed leads
perfect and would inflate full-signal numbers.

Baselines (declared in configs/train_l2g_v0_1.yaml BEFORE training):
  B0  every missing lead = 0.5 (a flat line in the middle of the normalised range)
  B1  every missing lead = the observed lead with the highest mean |Pearson r| to it on A_fit,
      flipped (1 - x) when that correlation is negative.
"""
from __future__ import annotations

import numpy as np
import torch

from src.models.l2g import SSIM1D


def gpu_crop_normalize(signals: torch.Tensor, offsets: torch.Tensor, length: int = 2048,
                       flat_eps: float = 1e-6) -> torch.Tensor:
    """Torch twin of preprocess.crop_normalize for batches of cached (B, 12, 2500) mV signals."""
    b, leads, _ = signals.shape
    idx = offsets.view(b, 1, 1) + torch.arange(length, device=signals.device).view(1, 1, length)
    x = signals.gather(2, idx.expand(b, leads, length))
    lo = x.amin(dim=-1, keepdim=True)
    rng = x.amax(dim=-1, keepdim=True) - lo
    flat = rng < flat_eps
    out = (x - lo) / torch.where(flat, torch.ones_like(rng), rng)
    out = torch.where(flat, torch.zeros_like(out), out)
    return out.clamp(0.0, 1.0)


def per_lead_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean(dim=-1)


def per_lead_pcc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson r per sample and lead; NaN where either signal is constant."""
    ac = a - a.mean(dim=-1, keepdim=True)
    bc = b - b.mean(dim=-1, keepdim=True)
    den = torch.sqrt((ac ** 2).sum(-1) * (bc ** 2).sum(-1))
    pcc = (ac * bc).sum(-1) / den.clamp_min(eps)
    return torch.where(den > eps, pcc, torch.full_like(pcc, float("nan")))


def masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Row-wise mean of values over leads with weight 1 (NaN entries ignored)."""
    ok = torch.isfinite(values) & (weights > 0)
    total = torch.where(ok, values, torch.zeros_like(values)).sum(-1)
    count = ok.sum(-1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float("nan")))


def row_metrics(ssim: SSIM1D, x_hat: torch.Tensor, x: torch.Tensor, lead_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row metrics after copy-through: missing-lead-only and full-signal versions."""
    s, m, p = ssim.per_lead(x_hat, x), per_lead_mse(x_hat, x), per_lead_pcc(x_hat, x)
    missing = 1.0 - lead_mask.float()
    every = torch.ones_like(missing)
    return {
        "ssim_missing": masked_mean(s, missing), "mse_missing": masked_mean(m, missing),
        "pcc_missing": masked_mean(p, missing), "ssim_full": masked_mean(s, every), "mse_full": masked_mean(m, every),
    }


def lead_correlation_matrix(windows: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Mean over records of the 12x12 Pearson correlation between leads (NaN-safe, chunked for memory)."""
    total = np.zeros((windows.shape[1], windows.shape[1]))
    count = np.zeros_like(total)
    for start in range(0, windows.shape[0], chunk):
        x = windows[start:start + chunk].astype(np.float64)
        x = x - x.mean(axis=-1, keepdims=True)
        norm = np.sqrt((x ** 2).sum(axis=-1, keepdims=True))
        z = np.divide(x, norm, out=np.full_like(x, np.nan), where=norm > 1e-8)
        corr = np.einsum("nit,njt->nij", z, z)
        total += np.nansum(corr, axis=0)
        count += np.isfinite(corr).sum(axis=0)
    return total / np.maximum(count, 1)


def baseline_constant(x_obs: torch.Tensor, lead_mask: torch.Tensor, value: float = 0.5) -> torch.Tensor:
    """B0: observed leads kept, missing leads set to a constant."""
    m = lead_mask.bool().unsqueeze(-1)
    return torch.where(m, x_obs, torch.full_like(x_obs, value))


def baseline_best_correlated(x_obs: torch.Tensor, lead_mask: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    """B1: each missing lead copies the observed lead with the highest |r| (flipped if r < 0)."""
    obs = lead_mask.bool()  # (B, 12)
    score = corr.abs().unsqueeze(0).expand(obs.shape[0], -1, -1).clone()  # (B, target j, source o)
    score = score.masked_fill(~obs.unsqueeze(1), float("-inf"))
    src = score.argmax(dim=-1)  # (B, 12)
    copied = torch.gather(x_obs, 1, src.unsqueeze(-1).expand(-1, -1, x_obs.shape[-1]))
    sign = torch.gather(corr.unsqueeze(0).expand(obs.shape[0], -1, -1), 2, src.unsqueeze(-1)).squeeze(-1)
    filled = torch.where((sign < 0).unsqueeze(-1), 1.0 - copied, copied)
    return torch.where(obs.unsqueeze(-1), x_obs, filled)
