"""Early-fusion baseline: every band stacked into one encoder.

The comparison AETHER has to beat. Optical, SAR and DEM channels are
concatenated at the input and passed through a single ConvNeXt-Tiny encoder,
then the same skip-connected decoder and task heads as AETHER. There is no
fusion module and no per-modality weighting: the network has to learn from the
raw stack how much to trust each sensor.

Absent modalities still arrive as all-zero channels (the dataset zeroes them),
so the baseline sees exactly the same inputs as AETHER under every cloud and
dropout condition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.decoder import Decoder
from models.optical_encoder import OpticalEncoder
from models.task_heads import get_task_head
from utils.config import DotDict


class EarlyFusionModel(nn.Module):
    """Single-encoder baseline with AETHER's decoder and heads."""

    use_skips = True

    def __init__(self, encoder: OpticalEncoder, decoder: Decoder, task_heads: nn.ModuleDict) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.task_heads = task_heads
        self.aux_heads = nn.ModuleDict()

    # train.build_optimizer gives the pretrained backbone its lower LR through
    # these names; the baseline has one backbone and no separate SAR encoder.
    @property
    def optical_encoder(self) -> nn.Module:
        return self.encoder

    @property
    def sar_encoder(self) -> nn.Module:
        return nn.Identity()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor, dem: torch.Tensor,
                presence: torch.Tensor | None = None, return_intermediates: bool = False):
        x = torch.cat([optical, sar, dem], dim=1)
        if presence is None:
            presence = x.new_ones(x.shape[0], 3)
        presence = presence.to(x.dtype)

        f, skips = self.encoder(x, return_skips=True)
        decoded = self.decoder(f, skips)
        outputs = {name: head(decoded) for name, head in self.task_heads.items()}

        # No learned fusion weights. Report the uniform weighting over present
        # sensors that the stack implies, so logging code stays unchanged.
        uniform = presence / presence.sum(dim=1, keepdim=True).clamp(min=1)
        outputs["alpha_maps"] = uniform.view(-1, 3, 1, 1).expand(-1, -1, *f.shape[-2:])
        outputs["f_shared"] = f
        outputs["presence"] = presence
        return outputs

    @classmethod
    def build_from_dict(cls, model_cfg: DotDict) -> "EarlyFusionModel":
        in_channels = (model_cfg.optical_encoder.in_channels + model_cfg.sar_encoder.in_channels
                       + model_cfg.dem_encoder.in_channels)
        encoder = OpticalEncoder(
            in_channels=in_channels,
            feature_dim=model_cfg.fusion.feature_dim,
            pretrained=model_cfg.optical_encoder.pretrained,
            detail_channels=int(getattr(model_cfg.optical_encoder, "detail_channels", 0)),
            backbone="convnext_tiny",
            full_res_channels=int(getattr(model_cfg.optical_encoder, "full_res_channels", 0)),
        )
        decoder = Decoder(
            feature_dim=model_cfg.fusion.feature_dim,
            out_channels=model_cfg.decoder.out_channels,
            se_reduction=getattr(model_cfg.decoder, "se_reduction", 16),
            skip_channels=encoder.skip_channels(),
        )
        heads = nn.ModuleDict()
        for task_name, head_params in dict(model_cfg.task_heads).items():
            kwargs = dict(head_params) if isinstance(head_params, dict) else {}
            heads[task_name] = get_task_head(task_name, in_channels=model_cfg.decoder.out_channels,
                                             **kwargs)
        return cls(encoder, decoder, heads)
