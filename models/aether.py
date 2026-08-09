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
    """

    def __init__(
        self,
        optical_encoder: OpticalEncoder,
        sar_encoder: SAREncoder,
        dem_encoder: DEMEncoder,
        fusion: CrossModalAlphaFusion,
        decoder: Decoder,
        task_heads: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.optical_encoder = optical_encoder
        self.sar_encoder = sar_encoder
        self.dem_encoder = dem_encoder
        self.fusion = fusion
        self.decoder = decoder
        self.task_heads = task_heads

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        dem: torch.Tensor,
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
        return_intermediates : bool
            If ``True``, pass through to the fusion module for ablation.

        Returns
        -------
        dict[str, torch.Tensor]
            Contains one key per active task head (e.g. ``"lulc"``),
            plus ``"alpha_maps"`` (always) and intermediate tensors
            if requested.
        """
        # 1. Encode each modality independently
        f_optical = self.optical_encoder(optical)   # (B, 256, H/16, W/16)
        f_sar = self.sar_encoder(sar)               # (B, 256, H/16, W/16)
        f_dem = self.dem_encoder(dem)                # (B, 256, H/16, W/16)

        # 2. Cross-modal fusion
        fusion_out = self.fusion(
            f_optical, f_sar, f_dem,
            return_intermediates=return_intermediates,
        )
        f_shared = fusion_out["f_shared"]           # (B, 256, H/16, W/16)

        # 3. Decode to full resolution
        decoded = self.decoder(f_shared)            # (B, 16, H, W)

        # 4. Task heads
        outputs: dict[str, torch.Tensor] = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(decoded)

        # Always include alpha maps for interpretability
        outputs["alpha_maps"] = fusion_out["alpha_maps"]

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
        optical_encoder = OpticalEncoder(
            in_channels=model_cfg.optical_encoder.in_channels,
            feature_dim=feature_dim,
            pretrained=model_cfg.optical_encoder.pretrained,
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
        decoder_se_reduction = getattr(model_cfg.decoder, "se_reduction", 16)
        decoder = Decoder(
            feature_dim=feature_dim,
            out_channels=model_cfg.decoder.out_channels,
            se_reduction=decoder_se_reduction,
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

        return cls(
            optical_encoder=optical_encoder,
            sar_encoder=sar_encoder,
            dem_encoder=dem_encoder,
            fusion=fusion,
            decoder=decoder,
            task_heads=heads,
        )
