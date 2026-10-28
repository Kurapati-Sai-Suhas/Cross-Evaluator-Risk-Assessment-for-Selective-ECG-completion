"""Diagnostic evaluators (project additions, AD-14; architecture lock v0.1 Section 2.4).

E1 = ResNet-family 1-D CNN in the style of the Wang et al. (2017) time-series ResNet, which the PTB-XL benchmark
(Strodthoff et al. 2021, verified in the local PDF) includes as `resnet1d_wang`. The exact layer settings below are
our PROJECT specification, fixed before training (configs/evaluator_e1_v0_1.yaml):
  3 residual blocks with 64, 128, 128 filters; each block = conv(k=8)-BN-ReLU-conv(k=5)-BN-ReLU-conv(k=3)-BN;
  shortcut = 1x1 conv + BN when the channel count changes, else BN; output ReLU(main + shortcut);
  global average pooling; linear layer to 2 logits (CLBBB, CRBBB). Inputs: (B, in_channels, 2048), normalised [0, 1].
E_dir uses the same network with in_channels = 24 (masked signal + mask, AD-1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 8, padding="same"), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 5, padding="same"), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding="same"), nn.BatchNorm1d(out_ch),
        )
        self.shortcut = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
                         if in_ch != out_ch else nn.BatchNorm1d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.main(x) + self.shortcut(x))


class ResNet1dWang(nn.Module):
    """Returns logits of shape (B, n_outputs)."""

    def __init__(self, in_channels: int = 12, n_outputs: int = 2, widths: tuple[int, ...] = (64, 128, 128)) -> None:
        super().__init__()
        blocks, prev = [], in_channels
        for w in widths:
            blocks.append(ResBlock1d(prev, w))
            prev = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x).mean(dim=-1))
