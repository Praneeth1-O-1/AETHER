"""Optical Encoder — Sentinel-2 backbone.

Extracts rich spectral-semantic features from multispectral optical data and
projects them to the shared 256-channel feature space at H/16 × W/16.

Two backbones are available:

- ``convnext_tiny`` (default): ConvNeXt-Tiny up to Stage 3, ImageNet weights.
  The first convolution is surgically replaced to accept an arbitrary number
  of input channels while preserving the pretrained weights downstream.
- ``resnet50_ssl4eo``: ResNet-50 up to layer 3, with weights self-supervised
  on 13-band Sentinel-2 imagery (SSL4EO-S12, MoCo v2; Wang et al., 2023). The
  pretrained first convolution already reads every band we have, so the
  spectral filters are kept rather than averaged into RGB-derived ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from data.dataset import OPTICAL_BANDS, OPTICAL_MEAN, OPTICAL_STD

# SSL4EO-S12 ResNet-50, MoCo, all 13 Sentinel-2 L1C bands (as published by torchgeo
# as ResNet50_Weights.SENTINEL2_ALL_MOCO). Input is reflectance (DN / 10000).
SSL4EO_S2_URL = ("https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/"
                 "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth")
SSL4EO_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
# Bands the archive lacks, folded into their nearest neighbour so the pretrained
# filter keeps its response: aerosol B1 -> blue B2, water vapour B9 -> B8A.
# Cirrus B10 is near-zero over the surface and is dropped.
SSL4EO_FOLD = {"B1": "B2", "B9": "B8A"}

BACKBONES = ("convnext_tiny", "resnet50_ssl4eo")


class OpticalEncoder(nn.Module):
    """Encoder for multispectral optical imagery.

    Parameters
    ----------
    in_channels : int
        Number of input channels (10 Sentinel-2 bands + validity).
    feature_dim : int
        Output feature dimension.  Must match the shared fusion dimension
        (default 256).
    pretrained : bool
        Whether to load pretrained weights.
    detail_channels : int
        ConvNeXt only: width of the H/2 detail branch (0 disables). The ResNet
        stem already runs at H/2 and supplies that skip itself.
    backbone : str
        One of ``BACKBONES``.
    full_res_channels : int
        Width of a small full-resolution branch feeding the decoder's last
        stage (0 disables). Roads are about one pixel wide at 10 m, and without
        it the finest feature the decoder sees is at H/2.
    """

    def __init__(
        self,
        in_channels: int = 13,
        feature_dim: int = 256,
        pretrained: bool = True,
        detail_channels: int = 0,
        backbone: str = "convnext_tiny",
        full_res_channels: int = 0,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.backbone = backbone

        if backbone == "convnext_tiny":
            self._build_convnext(in_channels, pretrained)
            self.detail_channels = detail_channels
        else:
            self._build_resnet50(in_channels, pretrained)
            self.detail_channels = 0

        # ------------------------------------------------------------------
        # Optional H/2 detail branch (ConvNeXt only)
        # ------------------------------------------------------------------
        # ConvNeXt's stem is a 4x4 stride-4 patchify, so this backbone produces
        # NOTHING above H/4 -- yet optical is the dominant modality and roads
        # are ~1 px wide at 10 m/px. This small parallel branch supplies the
        # H/2 detail the patchify discards, at negligible cost. 0 disables it.
        if self.detail_channels:
            self.detail = nn.Sequential(
                nn.Conv2d(in_channels, self.detail_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.detail_channels),
                nn.GELU(),
            )

        # ------------------------------------------------------------------
        # Optional full-resolution branch
        # ------------------------------------------------------------------
        # Two 3x3 convolutions at H: a 5x5 receptive field, enough to see a
        # road's edges without competing with the backbone for semantics.
        self.full_res_channels = full_res_channels
        if full_res_channels:
            self.full_res = nn.Sequential(
                nn.Conv2d(in_channels, full_res_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(full_res_channels),
                nn.GELU(),
                nn.Conv2d(full_res_channels, full_res_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(full_res_channels),
                nn.GELU(),
            )

    # ------------------------------------------------------------------
    # Backbones
    # ------------------------------------------------------------------
    def _build_convnext(self, in_channels: int, pretrained: bool) -> None:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # First-layer surgery.
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

        # Extract up to Stage 3 (index 0–5 in features Sequential)
        #    features[0] = Stem          → H/4,  96ch
        #    features[1] = Stage 1       → H/4,  96ch   (3 blocks)
        #    features[2] = Downsample    → H/8,  192ch
        #    features[3] = Stage 2       → H/8,  192ch  (3 blocks)
        #    features[4] = Downsample    → H/16, 384ch
        #    features[5] = Stage 3       → H/16, 384ch  (9 blocks)
        self.features = nn.Sequential(*list(backbone.features.children())[:6])
        self.projection = nn.Conv2d(384, self.feature_dim, kernel_size=1, bias=False)
        self._skip_taps = {1: "h4", 3: "h8"}
        self._skip_widths = {"h4": 96, "h8": 192}

    def _build_resnet50(self, in_channels: int, pretrained: bool) -> None:
        if in_channels != len(OPTICAL_BANDS) + 1:
            raise ValueError("resnet50_ssl4eo expects the 10 archive bands plus validity, "
                             f"got {in_channels} channels")
        net = models.resnet50(weights=None)
        if pretrained:
            state = torch.hub.load_state_dict_from_url(SSL4EO_S2_URL, map_location="cpu", progress=True)
            net.conv1 = nn.Conv2d(len(SSL4EO_BANDS), 64, 7, stride=2, padding=3, bias=False)
            missing, unexpected = net.load_state_dict(state, strict=False)
            if set(missing) - {"fc.weight", "fc.bias"} or unexpected:
                raise RuntimeError(f"SSL4EO weights do not match ResNet-50: "
                                   f"missing {missing}, unexpected {unexpected}")
            src = net.conv1.weight.detach()
        net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                w = torch.zeros_like(net.conv1.weight)    # validity channel starts at 0
                ours = [b.split("_")[-1] for b in OPTICAL_BANDS]
                for j, band in enumerate(SSL4EO_BANDS):
                    target = SSL4EO_FOLD.get(band, band)
                    if target in ours:
                        w[:, ours.index(target)] += src[:, j]
                net.conv1.weight.copy_(w)

        #    features[0..2] = conv1, bn1, relu → H/2,  64ch
        #    features[3]    = maxpool          → H/4
        #    features[4]    = layer1           → H/4,  256ch
        #    features[5]    = layer2           → H/8,  512ch
        #    features[6]    = layer3           → H/16, 1024ch
        self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3)
        self.projection = nn.Conv2d(1024, self.feature_dim, kernel_size=1, bias=False)
        self._skip_taps = {2: "h2", 4: "h4", 5: "h8"}
        self._skip_widths = {"h2": 64, "h4": 256, "h8": 512}
        # The pretrained filters expect reflectance, so the standardized bands
        # are mapped back to physical units before the stem (see `forward`).
        self.register_buffer("band_mean", torch.from_numpy(OPTICAL_MEAN.copy()).view(1, -1, 1, 1),
                             persistent=False)
        self.register_buffer("band_std", torch.from_numpy(OPTICAL_STD.copy()).view(1, -1, 1, 1),
                             persistent=False)

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

    def skip_channels(self) -> dict[str, int]:
        """Channels this encoder contributes at each skip scale."""
        out = dict(self._skip_widths)
        if self.detail_channels:
            out["h2"] = self.detail_channels
        if self.full_res_channels:
            out["h1"] = self.full_res_channels
        return out

    def _backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone != "resnet50_ssl4eo":
            return x
        bands, valid = x[:, :-1], x[:, -1:]
        reflectance = (bands * self.band_std.to(x.dtype) + self.band_mean.to(x.dtype)) * valid
        return torch.cat([reflectance, valid], dim=1)

    def forward(self, x: torch.Tensor, return_skips: bool = False):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_channels, H, W)``.
        return_skips : bool
            If ``True``, also return the intermediate feature maps keyed by
            scale for the decoder's skip connections.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, dict[str, torch.Tensor]]
            Feature tensor ``(B, feature_dim, H/16, W/16)``, plus skips
            when requested.
        """
        h = self._backbone_input(x)
        if not return_skips:
            return self.projection(self.features(h))

        skips: dict[str, torch.Tensor] = {}
        if self.full_res_channels:
            skips["h1"] = self.full_res(x)
        if self.detail_channels:
            skips["h2"] = self.detail(x)
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self._skip_taps:
                skips[self._skip_taps[i]] = h
        return self.projection(h), skips
