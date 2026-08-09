"""Optical Encoder — ConvNeXt-Tiny backbone for Sentinel-2 imagery.

Extracts rich spectral-semantic features from multispectral optical data.
Uses Stage 3 of ConvNeXt-Tiny (the deepest stage with 9 blocks) and
projects to the shared 256-channel feature space at H/16 × W/16 resolution.

The first convolutional layer is surgically replaced to accept an arbitrary
number of input channels while preserving pretrained ImageNet weights for
the remainder of the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class OpticalEncoder(nn.Module):
    """ConvNeXt-Tiny encoder for multispectral optical imagery.

    Parameters
    ----------
    in_channels : int
        Number of input spectral bands (e.g. 13 for Sentinel-2).
    feature_dim : int
        Output feature dimension.  Must match the shared fusion dimension
        (default 256).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        in_channels: int = 13,
        feature_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # ------------------------------------------------------------------
        # 1. Load backbone
        # ------------------------------------------------------------------
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # ------------------------------------------------------------------
        # 2. First-layer surgery
        # ------------------------------------------------------------------
        # ConvNeXt-Tiny stem: features[0] = Sequential(Conv2d(3,96,4,4), LayerNorm)
        original_stem = backbone.features[0]
        original_conv: nn.Conv2d = original_stem[0]  # type: ignore[assignment]

        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,  # type: ignore[arg-type]
                stride=original_conv.stride,  # type: ignore[arg-type]
                padding=original_conv.padding,  # type: ignore[arg-type]
                bias=original_conv.bias is not None,
            )
            if pretrained:
                self._replicate_weights(original_conv, new_conv, in_channels)
            original_stem[0] = new_conv  # type: ignore[index]

        # ------------------------------------------------------------------
        # 3. Extract up to Stage 3 (index 0–5 in features Sequential)
        #    features[0] = Stem          → H/4,  96ch
        #    features[1] = Stage 1       → H/4,  96ch   (3 blocks)
        #    features[2] = Downsample    → H/8,  192ch
        #    features[3] = Stage 2       → H/8,  192ch  (3 blocks)
        #    features[4] = Downsample    → H/16, 384ch
        #    features[5] = Stage 3       → H/16, 384ch  (9 blocks)
        # ------------------------------------------------------------------
        self.features = nn.Sequential(*list(backbone.features.children())[:6])

        # ------------------------------------------------------------------
        # 4. Projection: 384 → feature_dim
        # ------------------------------------------------------------------
        self.projection = nn.Conv2d(384, feature_dim, kernel_size=1, bias=False)

    # ------------------------------------------------------------------
    # Weight replication for non-RGB inputs
    # ------------------------------------------------------------------
    @staticmethod
    def _replicate_weights(
        src_conv: nn.Conv2d,
        dst_conv: nn.Conv2d,
        target_channels: int,
    ) -> None:
        """Replicate pretrained 3-channel weights to *target_channels*.

        Strategy: tile the mean of the 3-channel kernel along the input
        dimension, then scale so that the expected activation magnitude
        is preserved.  This is standard practice in remote sensing
        (e.g., SatMAE, SSL4EO).
        """
        with torch.no_grad():
            src_weight = src_conv.weight  # (out, 3, kH, kW)
            mean_weight = src_weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
            new_weight = mean_weight.repeat(1, target_channels, 1, 1)
            # Scale so ||new_weight||₂ ≈ ||src_weight||₂  per output filter
            scale = (3 / target_channels) ** 0.5
            dst_conv.weight.copy_(new_weight * scale)
            if src_conv.bias is not None and dst_conv.bias is not None:
                dst_conv.bias.copy_(src_conv.bias)

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
        features = self.features(x)       # (B, 384, H/16, W/16)
        return self.projection(features)   # (B, 256, H/16, W/16)
