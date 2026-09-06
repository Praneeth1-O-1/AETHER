"""SAR Encoder — ResNet18 backbone for Sentinel-1 SAR imagery.

Extracts structural and textural features from SAR backscatter data.
Uses Layer 3 of ResNet18 which outputs 256 channels at H/16 × W/16
resolution, matching the shared fusion dimension directly (no projection
layer needed).

The first convolutional layer is surgically replaced to accept an arbitrary
number of input channels (e.g. 2 for VV+VH polarizations).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SAREncoder(nn.Module):
    """ResNet18 encoder for SAR imagery.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 2 for VV + VH).
    feature_dim : int
        Output feature dimension.  Must match the shared fusion dimension
        (default 256).  ResNet18 Layer 3 already outputs 256 channels,
        so no projection is applied when ``feature_dim == 256``.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        in_channels: int = 2,
        feature_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # ------------------------------------------------------------------
        # 1. Load backbone
        # ------------------------------------------------------------------
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # ------------------------------------------------------------------
        # 2. First-layer surgery
        # ------------------------------------------------------------------
        original_conv: nn.Conv2d = backbone.conv1  # Conv2d(3, 64, 7, 2, 3)

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
            backbone.conv1 = new_conv

        # ------------------------------------------------------------------
        # 3. Extract up to Layer 3
        #    conv1 + bn1 + relu + maxpool  → H/4,  64ch
        #    layer1                        → H/4,  64ch
        #    layer2                        → H/8,  128ch
        #    layer3                        → H/16, 256ch
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        # ------------------------------------------------------------------
        # 4. Projection (only if feature_dim ≠ 256)
        # ------------------------------------------------------------------
        layer3_channels = 256
        if feature_dim != layer3_channels:
            self.projection: nn.Module = nn.Conv2d(
                layer3_channels, feature_dim, kernel_size=1, bias=False,
            )
        else:
            self.projection = nn.Identity()

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

        Same strategy as :class:`OpticalEncoder`: mean-tile + magnitude
        scaling.
        """
        with torch.no_grad():
            src_weight = src_conv.weight  # (64, 3, 7, 7)
            mean_weight = src_weight.mean(dim=1, keepdim=True)
            new_weight = mean_weight.repeat(1, target_channels, 1, 1)
            scale = (3 / target_channels) ** 0.5
            dst_conv.weight.copy_(new_weight * scale)
            if src_conv.bias is not None and dst_conv.bias is not None:
                dst_conv.bias.copy_(src_conv.bias)

    # features = [conv1, bn1, relu, maxpool, layer1, layer2, layer3]
    #             idx 2 -> H/2 64ch, idx 4 -> H/4 64ch, idx 5 -> H/8 128ch
    _SKIP_TAPS = {2: "h2", 4: "h4", 5: "h8"}

    def skip_channels(self) -> dict[str, int]:
        """Channels this encoder contributes at each skip scale."""
        return {"h2": 64, "h4": 64, "h8": 128}

    def forward(self, x: torch.Tensor, return_skips: bool = False):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_channels, H, W)``.
        return_skips : bool
            If ``True``, also return intermediate maps for decoder skips.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, dict[str, torch.Tensor]]
            Feature tensor ``(B, feature_dim, H/16, W/16)``, plus skips
            when requested.
        """
        if not return_skips:
            return self.projection(self.features(x))

        skips: dict[str, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self._SKIP_TAPS:
                skips[self._SKIP_TAPS[i]] = h
        return self.projection(h), skips
