"""DEM Encoder — Lightweight CNN for Digital Elevation Model data.

A compact 4-layer convolutional encoder with residual blocks designed
for terrain information.  DEM data carries a simpler signal than optical
or SAR imagery, so a heavy backbone is unnecessary.  The encoder uses
strided convolutions for 16× spatial downsampling and outputs 256-channel
features at H/16 × W/16, matching the shared fusion interface.

No pretrained weights — Kaiming initialization throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """Lightweight residual block: Conv3×3 → BN → GELU → Conv3×3 → BN + skip.

    Parameters
    ----------
    channels : int
        Number of input and output channels (no channel change).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class DEMEncoder(nn.Module):
    """Lightweight CNN encoder for Digital Elevation Model data.

    Architecture
    ------------
    Conv(in→32, stride 2)  → H/2,  32ch
    Conv(32→64, stride 2)  → H/4,  64ch
    ResidualBlock(64)
    Conv(64→128, stride 2) → H/8,  128ch
    ResidualBlock(128)
    Conv(128→256, stride 2)→ H/16, 256ch

    Parameters
    ----------
    in_channels : int
        Number of input channels (typically 1 for elevation).
    feature_dim : int
        Output feature dimension (default 256).
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # Stage 1: in_channels → 32, H/2
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Stage 2: 32 → 64, H/4
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Residual refinement at 64 channels
            _ResidualBlock(64),
            # Stage 3: 64 → 128, H/8
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Residual refinement at 128 channels
            _ResidualBlock(128),
            # Stage 4: 128 → 256, H/16
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Optional projection if feature_dim ≠ 256
        if feature_dim != 256:
            self.projection: nn.Module = nn.Conv2d(
                256, feature_dim, kernel_size=1, bias=False,
            )
        else:
            self.projection = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for all convolutional layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="linear"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, feature_dim, H/16, W/16)``.
        """
        features = self.features(x)         # (B, 256, H/16, W/16)
        return self.projection(features)    # (B, feature_dim, H/16, W/16)
