"""L2G-ECG reconstruction core for architecture lock v0.1 (docs/architecture_lock_v0_1.md, D-021).

Written from the approved specification, NOT copied from the authors' repository (it has no licence).
Provenance tags: PAPER = L2G-ECG paper (IEEE JBHI 2025, DOI 10.1109/JBHI.2025.3587428);
CODE = authors' released code (commit 92788cd6ca), read for the specification only;
PROJECT = our adaptation (adaptation ledger AD-n in the architecture lock).

Contents
* ECGen  - 1-D CNN encoder-decoder with 5 max-pool levels, index max-unpooling and concatenation skips.
* VTTD   - Visual Turing Test Discriminator (grouped convs, 24 sigmoid outputs).
* channel_shuffle, random_crop - inputs for the VTTD.
* SSIM1D - 1-D SSIM used by the reconstruction loss.
* Loss helpers for the staged objective (SSIM only for epochs 0-9, then SSIM - alpha * BCE).
No training loop lives here (Milestone 2b is smoke tests only).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LEADS = 12
POOL_LEVELS = 5  # PAPER Fig. 3 / CODE: five max-pool stages


def _conv(in_ch: int, out_ch: int, kernel: int) -> nn.Conv1d:
    # 'same' padding, stride 1: CODE (Fig. 3 labels "1p"; only the code reading matches 7.97M parameters)
    return nn.Conv1d(in_ch, out_ch, kernel, stride=1, padding="same")


def _enc_block(specs: list[tuple[int, int, int]]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_ch, out_ch, k in specs:  # Conv1D + BN + ReLU: PAPER (p. 3) + CODE
        layers += [_conv(in_ch, out_ch, k), nn.BatchNorm1d(out_ch), nn.ReLU()]
    return nn.Sequential(*layers)


def _dec_block(specs: list[tuple[int, int, int]]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_ch, out_ch, k in specs:  # no BatchNorm in the decoder: CODE
        layers += [_conv(in_ch, out_ch, k), nn.ReLU()]
    return nn.Sequential(*layers)


class ECGen(nn.Module):
    """Generator. Input (B, in_channels, T) with T divisible by 32; output (B, 12, T), no activation (CODE).

    in_channels = 24 (12 masked leads + 12 mask channels, PROJECT AD-1) or 12 (original L2G input).
    Expected parameters: 7,980,076 (24 inputs) / 7,972,396 (12 inputs, = the paper's 7.97M).
    """

    ENCODER = [  # (in, out, kernel) per conv; first kernel 10 = PAPER + CODE, inner kernel 3 = CODE (AD-5)
        [(None, 64, 10)],
        [(64, 128, 3)],
        [(128, 256, 3), (256, 256, 3)],
        [(256, 512, 3), (512, 512, 3)],
        [(512, 512, 3), (512, 512, 3)],
    ]
    DECODER = [  # level 1 (shallowest) ... level 5 (deepest); input = unpooled + concatenated skip
        [(128, 32, 3)],
        [(256, 64, 3)],
        [(512, 256, 3), (256, 128, 3)],
        [(1024, 512, 3), (512, 256, 3)],
        [(1024, 512, 3), (512, 512, 3)],
    ]

    def __init__(self, in_channels: int = 2 * N_LEADS, out_channels: int = N_LEADS) -> None:
        super().__init__()
        enc = [list(level) for level in self.ENCODER]
        enc[0][0] = (in_channels, 64, 10)
        self.encoder = nn.ModuleList(_enc_block(level) for level in enc)
        self.decoder = nn.ModuleList(_dec_block(level) for level in self.DECODER)
        self.pool = nn.MaxPool1d(2, return_indices=True)  # PAPER + CODE
        self.unpool = nn.MaxUnpool1d(2)  # index unpooling, also at the bottleneck: CODE (AD-5; Fig. 3 says Upsample)
        self.output = nn.Conv1d(32, out_channels, 10, stride=1, padding="same")  # PAPER + CODE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % (2 ** POOL_LEVELS) != 0:
            raise ValueError(f"signal length {x.shape[-1]} must be divisible by {2 ** POOL_LEVELS}")
        skips, indices = [], []
        h = x
        for block in self.encoder:
            h = block(h)
            skips.append(h)  # pre-pool activation is the skip (CODE)
            h, idx = self.pool(h)
            indices.append(idx)
        for level in reversed(range(POOL_LEVELS)):
            h = self.unpool(h, indices[level], output_size=skips[level].shape)
            h = torch.cat([h, skips[level]], dim=1)  # concatenation skip: PAPER + CODE
            h = self.decoder[level](h)
        return self.output(h)


def minmax_per_lead(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample, per-lead min-max of the generator output over time (CODE). A constant lead
    becomes 0 instead of NaN (the eps guard is a PROJECT numerical-safety detail)."""
    lo = y.amin(dim=-1, keepdim=True)
    hi = y.amax(dim=-1, keepdim=True)
    rng = hi - lo
    return torch.where(rng > eps, (y - lo) / rng.clamp_min(eps), torch.zeros_like(y))


class VTTD(nn.Module):
    """Visual Turing Test Discriminator: (B, 24, L) -> (B, 24) probabilities. Expected parameters: 14,184."""

    def __init__(self, n_leads: int = N_LEADS) -> None:
        super().__init__()
        # Fig. 5 (PAPER) + CODE; groups = 12 so each group sees exactly one lead's pair (PAPER p. 4)
        self.conv1 = nn.Conv1d(2 * n_leads, 48, 10, stride=2, padding=1, groups=n_leads)
        self.conv2 = nn.Conv1d(48, 96, 4, stride=2, padding=1, groups=n_leads)
        self.bn2 = nn.BatchNorm1d(96)
        self.conv3 = nn.Conv1d(96, 192, 4, stride=2, padding=1, groups=n_leads)
        self.bn3 = nn.BatchNorm1d(192)
        self.fc = nn.Linear(192, 2 * n_leads)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(d))  # ReLU as executed by the released code (CODE, inference)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = h.mean(dim=-1)  # global average pooling
        return torch.sigmoid(self.fc(h))


def channel_shuffle(x_gen: torch.Tensor, x_real: torch.Tensor,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly order each lead's (generated, real) pair (PAPER eq. 11-12; vectorised, AD-13).

    Returns d of shape (B, 24, L) ordered [lead0_a, lead0_b, lead1_a, lead1_b, ...] and labels (B, 24)
    with 1 = generated, 0 = real (PAPER p. 5: c = 0 means real; the released code uses the opposite
    coding, which is equivalent under binary cross-entropy).
    """
    if x_gen.shape != x_real.shape:
        raise ValueError(f"shape mismatch {tuple(x_gen.shape)} vs {tuple(x_real.shape)}")
    b, leads, _ = x_gen.shape
    gen_first = torch.randint(0, 2, (b, leads, 1), generator=generator,
                              device="cpu").to(x_gen.device).bool()
    first = torch.where(gen_first, x_gen, x_real)
    second = torch.where(gen_first, x_real, x_gen)
    d = torch.stack([first, second], dim=2).reshape(b, 2 * leads, -1)
    flag = gen_first.squeeze(-1).float()
    labels = torch.stack([flag, 1.0 - flag], dim=2).reshape(b, 2 * leads)
    return d, labels


def random_crop(x_gen: torch.Tensor, x_real: torch.Tensor, length: int = 400,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The same random window for the whole batch, as in the released code (VTTD sees 400 samples: CODE)."""
    t = x_gen.shape[-1]
    start = int(torch.randint(0, t - length + 1, (1,), generator=generator).item())
    return x_gen[..., start:start + length], x_real[..., start:start + length]


def gaussian_window(size: int = 50, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian window (size 50, sigma 1.5: CODE), normalised to sum 1."""
    x = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).float()


class SSIM1D(nn.Module):
    """Mean SSIM over batch, leads and time. C1 = 0.01**2, C2 = 0.03**2 (PAPER p. 6 + CODE)."""

    def __init__(self, window_size: int = 50, sigma: float = 1.5, channels: int = N_LEADS) -> None:
        super().__init__()
        self.window_size, self.channels = window_size, channels
        self.register_buffer("window", gaussian_window(window_size, sigma).expand(channels, 1, window_size).clone())
        self.c1, self.c2 = 0.01 ** 2, 0.03 ** 2

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ssim_map(a, b).mean()

    def per_lead(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(B, 12) SSIM per sample and lead (used for missing-lead-only metrics)."""
        return self.ssim_map(a, b).mean(dim=-1)

    def ssim_map(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape or a.shape[1] != self.channels:
            raise ValueError(f"expected two (B, {self.channels}, T) tensors, got {tuple(a.shape)}, {tuple(b.shape)}")
        w, pad, g = self.window.to(a.dtype), self.window_size // 2, self.channels
        mu_a = F.conv1d(a, w, padding=pad, groups=g)
        mu_b = F.conv1d(b, w, padding=pad, groups=g)
        var_a = F.conv1d(a * a, w, padding=pad, groups=g) - mu_a ** 2
        var_b = F.conv1d(b * b, w, padding=pad, groups=g) - mu_b ** 2
        cov = F.conv1d(a * b, w, padding=pad, groups=g) - mu_a * mu_b
        return ((2 * mu_a * mu_b + self.c1) * (2 * cov + self.c2)) / (
            (mu_a ** 2 + mu_b ** 2 + self.c1) * (var_a + var_b + self.c2))


ALPHA = 0.1  # adversarial weight: PAPER p. 5 + CODE
ADVERSARIAL_START_EPOCH = 10  # VTTD starts after 10 generator-only epochs: PAPER + CODE


def reconstruction_loss(ssim: SSIM1D, raw_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM(min-max(output), target) (PAPER Stage 1 + CODE)."""
    return 1.0 - ssim(minmax_per_lead(raw_output), target)


def discriminator_loss(vttd: VTTD, x_gen: torch.Tensor, x_real: torch.Tensor,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """BCE of the VTTD on a shuffled, cropped pair batch (PAPER Stage 2 + CODE). x_gen must be detached by the caller."""
    g, r = random_crop(x_gen, x_real, generator=generator)
    d, labels = channel_shuffle(g, r, generator=generator)
    return F.binary_cross_entropy(vttd(d), labels)


def generator_total_loss(ssim: SSIM1D, vttd: VTTD, raw_output: torch.Tensor, target: torch.Tensor,
                         epoch: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Staged generator objective: 1 - SSIM before epoch 10, then (1 - SSIM) - alpha * BCE(VTTD) (PAPER Stage 3 + CODE)."""
    rec = reconstruction_loss(ssim, raw_output, target)
    if epoch < ADVERSARIAL_START_EPOCH:
        return rec
    adv = discriminator_loss(vttd, minmax_per_lead(raw_output), target, generator=generator)
    return rec - ALPHA * adv


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_input(x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, 24, T): masked signal channels then mask channels (AD-1). Mirrors missingness.build_model_input."""
    if x_obs.shape != mask.shape:
        raise ValueError("x_obs and mask must have the same shape")
    if torch.any(x_obs[mask == 0] != 0):
        raise ValueError("x_obs must be zero at missing positions")
    return torch.cat([x_obs, mask.to(x_obs.dtype)], dim=1)


def expected_parameter_counts() -> dict[str, int]:
    """Values the architecture lock requires (arithmetic in docs/l2g_architecture_status.md)."""
    return {"ecgen_24_inputs": 7_980_076, "ecgen_12_inputs": 7_972_396, "vttd": 14_184}
