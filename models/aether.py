"""AETHER — Full model assembly.

Composes the three modality-specific encoders, CrossModalAlphaFusion,
progressive decoder, and task heads into a single end-to-end model.

Usage::

    from models.aether import AETHERModel

    model = AETHERModel.build_from_config("configs/model.yaml")
    outputs = model(optical, sar, dem)
    lulc_logits = outputs["lulc"]
    alpha_maps  = outputs["alpha_maps"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from models.optical_encoder import OpticalEncoder
from models.sar_encoder import SAREncoder
from models.dem_encoder import DEMEncoder
from models.crossmodal_fusion import CrossModalAlphaFusion
from models.decoder import Decoder
from models.task_heads import get_task_head, TaskHead
from utils.config import DotDict, load_config


class AETHERModel(nn.Module):
    """AETHER: Adaptive Earth Observation Through Heterogeneous Encoder Representation.

    End-to-end multimodal geospatial fusion model.

    Parameters
    ----------
    optical_encoder : OpticalEncoder
        Encoder for multispectral optical imagery.
    sar_encoder : SAREncoder
        Encoder for SAR imagery.
    dem_encoder : DEMEncoder
        Encoder for DEM / elevation data.
    fusion : CrossModalAlphaFusion
        Cross-modal adaptive fusion module.
    decoder : Decoder
        Progressive upsampling decoder.
    task_heads : nn.ModuleDict
        Named task-specific prediction heads.
    aux_heads : nn.ModuleDict, optional
        Per-modality auxiliary LULC classifiers operating directly on each
        encoder's features, before fusion. Their purpose is not accuracy but
        pressure: an encoder that must predict on its own cannot be quietly
        ignored by the fusion. Modality collapse -- one dominant modality's
        features masking a weaker one's through the shared fusion neurons -- is
        the documented failure mode here, and unimodal supervision is the
        standard remedy. They also yield a free per-modality skill readout,
        which is what drives gradient modulation in `train.py`.

    Notes
    -----
    Modality dropout is deliberately NOT implemented here. It lives in
    `data.dataset`, applied to the raw input bands, so that a synthetically
    dropped modality is byte-identical to a genuinely unacquired one. Dropping
    at the encoded features (as an earlier revision did) produces an exact zero
    that no real tile ever produces, which trains the model for a condition
    that never occurs at test time.
    """

    def __init__(
        self,
        optical_encoder: OpticalEncoder,
        sar_encoder: SAREncoder,
        dem_encoder: DEMEncoder,
        fusion: CrossModalAlphaFusion,
        decoder: Decoder,
        task_heads: nn.ModuleDict,
        use_skips: bool = False,
        aux_heads: nn.ModuleDict | None = None,
    ) -> None:
        super().__init__()
        self.use_skips = use_skips
        self.optical_encoder = optical_encoder
        self.sar_encoder = sar_encoder
        self.dem_encoder = dem_encoder
        self.fusion = fusion
        self.decoder = decoder
        self.task_heads = task_heads
        self.aux_heads = aux_heads if aux_heads is not None else nn.ModuleDict()

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        dem: torch.Tensor,
        presence: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """End-to-end forward pass.

        Parameters
        ----------
        optical : torch.Tensor
            Optical input ``(B, C_opt, H, W)``.
        sar : torch.Tensor
            SAR input ``(B, C_sar, H, W)``.
        dem : torch.Tensor
            DEM input ``(B, C_dem, H, W)``.
        presence : torch.Tensor, optional
            ``(B, 3)`` in {0, 1} marking which modalities are genuinely
            available. Supplied by the dataset, which already knows -- from the
            validity channels -- whether a modality was acquired, dropped, or
            fully occluded by injected cloud. ``None`` means all present, which
            is the correct default for a caller holding complete data.
        return_intermediates : bool
            If ``True``, pass through to the fusion module for ablation.

        Returns
        -------
        dict[str, torch.Tensor]
            One key per active task head (e.g. ``"lulc"``), plus
            ``"alpha_maps"`` and ``"f_shared"`` always, ``"aux_<modality>"``
            when auxiliary heads are built, and fusion intermediates if
            requested.
        """
        # 1. Encode each modality independently
        if self.use_skips:
            f_optical, skip_o = self.optical_encoder(optical, return_skips=True)
            f_sar, skip_s = self.sar_encoder(sar, return_skips=True)
            f_dem, skip_d = self.dem_encoder(dem, return_skips=True)
        else:
            f_optical = self.optical_encoder(optical)   # (B, 256, H/16, W/16)
            f_sar = self.sar_encoder(sar)               # (B, 256, H/16, W/16)
            f_dem = self.dem_encoder(dem)               # (B, 256, H/16, W/16)
            skip_o = skip_s = skip_d = {}

        if presence is None:
            presence = f_optical.new_ones(f_optical.shape[0], 3)
        presence = presence.to(f_optical.dtype)

        # 1b. An absent modality's input is all zeros, but conv bias and norm
        # affine terms mean encoder(0) is a nonzero learned constant, not zero.
        # The fusion handles that for f_*, but the skip path bypasses fusion
        # entirely -- so an ungated skip would smuggle that constant straight
        # into the decoder and quietly undo the masking.
        gates = [presence[:, i].view(-1, 1, 1, 1) for i in range(3)]
        skip_o = {k: v * gates[0] for k, v in skip_o.items()}
        skip_s = {k: v * gates[1] for k, v in skip_s.items()}
        skip_d = {k: v * gates[2] for k, v in skip_d.items()}

        # 2. Cross-modal fusion
        fusion_out = self.fusion(
            f_optical, f_sar, f_dem,
            presence=presence,
            return_intermediates=return_intermediates,
        )
        f_shared = fusion_out["f_shared"]           # (B, 256, H/16, W/16)

        # 3. Decode to full resolution, concatenating each modality's skips
        #    at the matching scale (optical, then SAR, then DEM -- the order the
        #    decoder's input channel count was built from).
        if self.use_skips:
            merged = {
                scale: torch.cat(
                    [d[scale] for d in (skip_o, skip_s, skip_d) if scale in d], dim=1
                )
                for scale in ("h8", "h4", "h2")
                if any(scale in d for d in (skip_o, skip_s, skip_d))
            }
            decoded = self.decoder(f_shared, merged)
        else:
            decoded = self.decoder(f_shared)        # (B, out_channels, H, W)

        # 4. Task heads
        outputs: dict[str, torch.Tensor] = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(decoded)

        # 4b. Auxiliary per-modality predictions, straight off each encoder.
        #     Deliberately a bare 1x1 conv plus bilinear upsampling: the point
        #     is to force each encoder to be independently predictive, not to
        #     build a second decoder. Anything heavier would let the aux head
        #     compensate for a weak encoder, which defeats the purpose.
        if self.aux_heads:
            size = decoded.shape[-2:]
            for name, feat in (("optical", f_optical), ("sar", f_sar), ("dem", f_dem)):
                if name in self.aux_heads:
                    outputs[f"aux_{name}"] = nn.functional.interpolate(
                        self.aux_heads[name](feat), size=size,
                        mode="bilinear", align_corners=False,
                    )

        # Always include alpha maps and f_shared for interpretability/evaluation
        outputs["alpha_maps"] = fusion_out["alpha_maps"]
        outputs["f_shared"] = f_shared
        outputs["presence"] = presence

        # Include intermediates if requested
        if return_intermediates:
            for key in ("f_optical_cross", "f_sar_cross", "f_joint"):
                if key in fusion_out:
                    outputs[key] = fusion_out[key]

        return outputs

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def build_from_config(
        cls,
        config_path: Union[str, Path],
    ) -> "AETHERModel":
        """Construct an AETHER model from a YAML configuration file.

        Parameters
        ----------
        config_path : str or Path
            Path to the model configuration YAML.

        Returns
        -------
        AETHERModel
            Fully constructed model ready for training or inference.
        """
        cfg = load_config(config_path)
        return cls.build_from_dict(cfg.model)

    @classmethod
    def build_from_dict(
        cls,
        model_cfg: DotDict,
    ) -> "AETHERModel":
        """Construct an AETHER model from a parsed config dict.

        Parameters
        ----------
        model_cfg : DotDict
            The ``model`` subtree of the configuration.

        Returns
        -------
        AETHERModel
            Fully constructed model.
        """
        feature_dim: int = model_cfg.fusion.feature_dim

        # --- Encoders ---
        use_skips = bool(model_cfg.get("use_skips", False))
        detail_channels = int(getattr(model_cfg.optical_encoder, "detail_channels", 0)) if use_skips else 0

        optical_encoder = OpticalEncoder(
            in_channels=model_cfg.optical_encoder.in_channels,
            feature_dim=feature_dim,
            pretrained=model_cfg.optical_encoder.pretrained,
            detail_channels=detail_channels,
        )
        sar_encoder = SAREncoder(
            in_channels=model_cfg.sar_encoder.in_channels,
            feature_dim=feature_dim,
            pretrained=model_cfg.sar_encoder.pretrained,
        )
        dem_encoder = DEMEncoder(
            in_channels=model_cfg.dem_encoder.in_channels,
            feature_dim=feature_dim,
        )

        # --- Fusion ---
        use_se = getattr(model_cfg.fusion, "use_se_refinement", False)
        se_reduction = getattr(model_cfg.fusion, "se_reduction", 16)
        fusion = CrossModalAlphaFusion(
            feature_dim=feature_dim,
            num_heads=model_cfg.fusion.num_heads,
            dropout=model_cfg.fusion.dropout,
            use_modality_embeddings=model_cfg.fusion.use_modality_embeddings,
            num_refinement_blocks=model_cfg.fusion.num_refinement_blocks,
            use_se_refinement=use_se,
            se_reduction=se_reduction,
        )

        # --- Decoder ---
        # Sum each encoder's contribution per scale so the stage convs are built
        # with the exact concatenated width they will receive at runtime.
        skip_channels = None
        if use_skips:
            skip_channels = {}
            for enc in (optical_encoder, sar_encoder, dem_encoder):
                for scale, n in enc.skip_channels().items():
                    skip_channels[scale] = skip_channels.get(scale, 0) + n

        decoder_se_reduction = getattr(model_cfg.decoder, "se_reduction", 16)
        decoder = Decoder(
            feature_dim=feature_dim,
            out_channels=model_cfg.decoder.out_channels,
            se_reduction=decoder_se_reduction,
            skip_channels=skip_channels,
        )

        # --- Task Heads ---
        heads = nn.ModuleDict()
        task_cfg: dict = dict(model_cfg.task_heads)
        for task_name, head_params in task_cfg.items():
            if isinstance(head_params, dict):
                heads[task_name] = get_task_head(
                    task_name,
                    in_channels=model_cfg.decoder.out_channels,
                    **head_params,
                )
            else:
                heads[task_name] = get_task_head(
                    task_name,
                    in_channels=model_cfg.decoder.out_channels,
                )

        # --- Auxiliary unimodal heads ---
        # Only meaningful for a dense classification task; built when LULC is
        # active and aux supervision is switched on.
        aux_heads = nn.ModuleDict()
        if bool(model_cfg.get("use_aux_heads", False)) and "lulc" in task_cfg:
            n_classes = model_cfg.task_heads.lulc.num_classes
            for name in ("optical", "sar", "dem"):
                aux_heads[name] = nn.Conv2d(feature_dim, n_classes, kernel_size=1)

        return cls(
            optical_encoder=optical_encoder,
            sar_encoder=sar_encoder,
            dem_encoder=dem_encoder,
            fusion=fusion,
            decoder=decoder,
            task_heads=heads,
            use_skips=use_skips,
            aux_heads=aux_heads,
        )
