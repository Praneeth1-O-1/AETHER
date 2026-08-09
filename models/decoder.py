"""Decoder — Lightweight progressive upsampling with SE refinement.

Converts the fused feature representation (F_shared) from H/16 × W/16
back to full resolution H × W through 4 upsampling stages.  Each stage
doubles spatial resolution, halves channel count, and applies
Squeeze-and-Excitation channel attention for adaptive refinement.

No encoder skip connections are used — the fusion module is the sole
information bottleneck, preserving the integrity of CrossModalAlphaFusion
as the research contribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """SE channel attention block (decoder-local copy to avoid circular imports).

    Parameters
    ----------
    channels : int
        Number of input / output channels.
    reduction : int
        Bottleneck reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class _DecoderStage(nn.Module):
    """Single decoder upsampling stage.

    Upsample(2×) → Conv3×3 → BN → GELU → SE

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    se_reduction : int
        SE bottleneck reduction ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.se = SqueezeExcitation(out_channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(self.up(x)))


class Decoder(nn.Module):
    """Progressive upsampling decoder with SE channel attention.

    Recovers full spatial resolution from the fused feature map.

    Architecture (4 stages, 16× total upsampling)::

        F_shared (B, 256, H/16, W/16)
          → Stage 1: 256 → 128, H/8
          → Stage 2: 128 →  64, H/4
          → Stage 3:  64 →  32, H/2
          → Stage 4:  32 →  out_channels, H

    Parameters
    ----------
    feature_dim : int
        Input channel dimension from F_shared (default 256).
    out_channels : int
        Output channel dimension fed to task heads (default 16).
    se_reduction : int
        SE reduction ratio at each stage (default 16).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        out_channels: int = 16,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.out_channels = out_channels

        # Channel progression: 256 → 128 → 64 → 32 → out_channels
        channels = [feature_dim, 128, 64, 32, out_channels]

        self.stages = nn.Sequential(
            *[
                _DecoderStage(channels[i], channels[i + 1], se_reduction)
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, f_shared: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        f_shared : torch.Tensor
            Fused features of shape ``(B, feature_dim, H/16, W/16)``.

        Returns
        -------
        torch.Tensor
            Decoded features of shape ``(B, out_channels, H, W)``.
        """
        return self.stages(f_shared)
