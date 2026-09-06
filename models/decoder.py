"""Decoder — progressive upsampling with optional U-Net skip connections.

Recovers full spatial resolution from the fused feature map.

Why skips matter here
---------------------
``f_shared`` arrives at H/16. For a 256x256 tile that is a 16x16 grid, and at
Sentinel-2's 10 m/px a two-lane road is roughly one full-resolution pixel --
about 1/16th of a bottleneck cell. The information is not lost (the 256
channels can encode sub-cell geometry, and the skip-free model does reach
0.30 road IoU, well above the 0.083 a single-channel H/16 map could support),
but the bottleneck has to spend capacity encoding *where* something is on top
of *what* it is, and the decoder then has to synthesize the geometry back.

Skip connections hand the decoder the encoder's high-resolution feature maps
directly, so it can read boundaries instead of reconstructing them, and the
bottleneck is freed to specialize in semantics. This is the standard U-Net
division of labour: deep layers know *what*, shallow layers know *where*.

Skips are optional (``skip_channels=None``) so a model can still be built in
the original skip-free configuration to reproduce earlier checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Recalibrates channel-wise feature responses by modelling
    inter-channel dependencies.

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
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * scale


class _DecoderStage(nn.Module):
    """Single decoder upsampling stage.

    Upsample(2x) -> [concat skip] -> Conv3x3 -> BN -> GELU -> SE

    Parameters
    ----------
    in_channels : int
        Input channel count from the previous stage.
    out_channels : int
        Output channel count.
    skip_channels : int
        Total channels of the skip tensor concatenated after upsampling.
        Zero disables the skip path, which reproduces the original stage
        exactly -- including its ``conv``/``se`` parameter names, so
        skip-free checkpoints still load.
    se_reduction : int
        SE bottleneck reduction ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.skip_channels = skip_channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.se = SqueezeExcitation(out_channels, se_reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None and self.skip_channels:
            # Guard against odd input sizes, where H/16 * 2 may not equal H/8.
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.se(self.conv(x))


class Decoder(nn.Module):
    """Progressive upsampling decoder with SE channel attention and U-Net skips.

    Architecture (4 stages, 16x total upsampling)::

        F_shared (B, feature_dim, H/16, W/16)
          -> Stage 1: -> H/8,  concat skip[h8] -> 128
          -> Stage 2: -> H/4,  concat skip[h4] ->  64
          -> Stage 3: -> H/2,  concat skip[h2] ->  32
          -> Stage 4: -> H                     -> out_channels

    Parameters
    ----------
    feature_dim : int
        Input channel dimension from F_shared (default 256).
    out_channels : int
        Output channel dimension fed to task heads. Widened from the original
        16 because road and building are detail tasks that previously shared a
        16-channel bottleneck feeding a single 3x3 conv per head.
    se_reduction : int
        SE reduction ratio at each stage (default 16).
    skip_channels : dict[str, int] or None
        Channels available at each scale, e.g. ``{"h8": 448, "h4": 224, "h2": 96}``.
        ``None`` builds the original skip-free decoder.
    """

    #: Decoder stage index -> the skip scale consumed after that stage upsamples.
    STAGE_SCALES = ("h8", "h4", "h2", None)

    def __init__(
        self,
        feature_dim: int = 256,
        out_channels: int = 48,
        se_reduction: int = 16,
        skip_channels: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.skip_channels = dict(skip_channels or {})

        # Channel progression: feature_dim -> 128 -> 64 -> 32 -> out_channels
        channels = [feature_dim, 128, 64, 32, out_channels]
        self.stages = nn.ModuleList(
            _DecoderStage(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                skip_channels=self.skip_channels.get(self.STAGE_SCALES[i], 0),
                se_reduction=se_reduction,
            )
            for i in range(len(channels) - 1)
        )

    def forward(
        self,
        f_shared: torch.Tensor,
        skips: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decode to full resolution.

        Parameters
        ----------
        f_shared : torch.Tensor
            Fused features ``(B, feature_dim, H/16, W/16)``.
        skips : dict[str, torch.Tensor] or None
            Concatenated encoder features keyed ``"h8"``, ``"h4"``, ``"h2"``.

        Returns
        -------
        torch.Tensor
            Decoded features ``(B, out_channels, H, W)``.
        """
        skips = skips or {}
        x = f_shared
        for stage, scale in zip(self.stages, self.STAGE_SCALES):
            x = stage(x, skips.get(scale) if scale else None)
        return x
