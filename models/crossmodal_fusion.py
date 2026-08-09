"""CrossModalAlphaFusion — Adaptive multimodal feature fusion module.

This is the core research contribution of AETHER.  It fuses optical, SAR,
and DEM features into a shared representation (F_shared) via:

1. **Learnable modality embeddings** — explicit modality identity signals
2. **Bidirectional cross-attention** (SAR ↔ Optical) — mutual enrichment
3. **Joint Feature Embedding** — learned projection into a shared space
4. **Adaptive Spatial Weight Estimator** — per-pixel modality importance
5. **Weighted Adaptive Fusion** — spatially-varying convex combination
6. **Fusion Refinement** — standard residual feature harmonization block

Key property: the spatial alpha maps (α_O, α_S, α_D) are *per-pixel*
weights generated from the joint multimodal representation, NOT global
scalars.  At every spatial location, the network learns which modality
is most informative.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Building Blocks
# =========================================================================


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


class RefinementBlock(nn.Module):
    """Standard residual block for post-fusion feature harmonization.

    Conv3×3 → BN → GELU → Conv3×3 → BN → Residual Skip (+ GELU)

    Parameters
    ----------
    channels : int
        Number of input / output channels.
    use_se : bool
        Whether to append Squeeze-and-Excitation channel attention (default False).
    se_reduction : int
        SE bottleneck reduction ratio (used if use_se is True).
    """

    def __init__(
        self,
        channels: int,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(channels, se_reduction)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        return self.act2(out + residual)


# =========================================================================
# Main Module
# =========================================================================


class CrossModalAlphaFusion(nn.Module):
    """Adaptive cross-modal feature fusion with spatial alpha maps.

    Parameters
    ----------
    feature_dim : int
        Channel dimension of all encoder outputs (default 256).
    num_heads : int
        Number of attention heads for cross-attention (default 8).
    dropout : float
        Dropout probability in attention layers (default 0.1).
    use_modality_embeddings : bool
        Whether to add learnable modality embeddings (default True).
    num_refinement_blocks : int
        Number of stacked RefinementBlocks after fusion (default 1).
    use_se_refinement : bool
        Whether to include SE attention in refinement blocks (default False).
    se_reduction : int
        SE reduction ratio inside RefinementBlocks (default 16).
    """

    NUM_MODALITIES: int = 3  # optical, SAR, DEM

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_modality_embeddings: bool = True,
        num_refinement_blocks: int = 1,
        use_se_refinement: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.use_modality_embeddings = use_modality_embeddings

        # ------------------------------------------------------------------
        # 1. Learnable modality embeddings
        # ------------------------------------------------------------------
        if use_modality_embeddings:
            self.emb_optical = nn.Parameter(
                torch.zeros(1, feature_dim, 1, 1)
            )
            self.emb_sar = nn.Parameter(
                torch.zeros(1, feature_dim, 1, 1)
            )
            self.emb_dem = nn.Parameter(
                torch.zeros(1, feature_dim, 1, 1)
            )
            # Small random init so embeddings are non-degenerate at start
            nn.init.trunc_normal_(self.emb_optical, std=0.02)
            nn.init.trunc_normal_(self.emb_sar, std=0.02)
            nn.init.trunc_normal_(self.emb_dem, std=0.02)

        # ------------------------------------------------------------------
        # 2. Bidirectional cross-attention (SAR ↔ Optical)
        # ------------------------------------------------------------------
        # Optical attends to SAR
        self.cross_attn_opt2sar = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_opt = nn.LayerNorm(feature_dim)

        # SAR attends to Optical
        self.cross_attn_sar2opt = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_sar = nn.LayerNorm(feature_dim)

        # ------------------------------------------------------------------
        # 3. Joint Feature Embedding
        #    Cat(F_O', F_S', F_D) → Conv(3C, C, 1) → BN → GELU
        # ------------------------------------------------------------------
        self.joint_embedding = nn.Sequential(
            nn.Conv2d(
                feature_dim * self.NUM_MODALITIES,
                feature_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # 4. Adaptive Spatial Weight Estimator
        #    Conv(C, C//2, 3) → GELU → Conv(C//2, 3, 1) → Softmax(dim=1)
        # ------------------------------------------------------------------
        self.alpha_estimator = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, self.NUM_MODALITIES, 1),
        )

        # ------------------------------------------------------------------
        # 5. Fusion Refinement Block(s)
        # ------------------------------------------------------------------
        self.refinement = nn.Sequential(
            *[
                RefinementBlock(
                    feature_dim,
                    use_se=use_se_refinement,
                    se_reduction=se_reduction,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_to_seq(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Reshape (B, C, H, W) → (B, H*W, C) for attention."""
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), H, W  # (B, N, C)

    @staticmethod
    def _seq_to_spatial(
        x: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Reshape (B, H*W, C) → (B, C, H, W)."""
        B, _, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        f_optical: torch.Tensor,
        f_sar: torch.Tensor,
        f_dem: torch.Tensor,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Fuse optical, SAR, and DEM feature maps.

        Parameters
        ----------
        f_optical : torch.Tensor
            Optical features ``(B, C, H', W')``.
        f_sar : torch.Tensor
            SAR features ``(B, C, H', W')``.
        f_dem : torch.Tensor
            DEM features ``(B, C, H', W')``.
        return_intermediates : bool
            If ``True``, include intermediate tensors in the output dict
            for ablation studies.

        Returns
        -------
        dict[str, torch.Tensor]
            Always contains:

            - ``"f_shared"`` — fused representation ``(B, C, H', W')``
            - ``"alpha_maps"`` — spatial weights ``(B, 3, H', W')``

            When *return_intermediates* is ``True``, additionally:

            - ``"f_optical_cross"`` — cross-attended optical features
            - ``"f_sar_cross"`` — cross-attended SAR features
            - ``"f_joint"`` — joint embedding before alpha weighting
        """
        # ---- 1. Add modality embeddings ----
        if self.use_modality_embeddings:
            f_optical = f_optical + self.emb_optical
            f_sar = f_sar + self.emb_sar
            f_dem = f_dem + self.emb_dem

        # ---- 2. Bidirectional cross-attention ----
        opt_seq, H, W = self._spatial_to_seq(f_optical)
        sar_seq, _, _ = self._spatial_to_seq(f_sar)

        # Optical attends to SAR (query=opt, key/value=sar)
        opt_cross, _ = self.cross_attn_opt2sar(opt_seq, sar_seq, sar_seq)
        opt_cross = self.norm_opt(opt_cross + opt_seq)  # Residual + LN

        # SAR attends to Optical (query=sar, key/value=opt)
        sar_cross, _ = self.cross_attn_sar2opt(sar_seq, opt_seq, opt_seq)
        sar_cross = self.norm_sar(sar_cross + sar_seq)  # Residual + LN

        f_optical_cross = self._seq_to_spatial(opt_cross, H, W)
        f_sar_cross = self._seq_to_spatial(sar_cross, H, W)

        # ---- 3. Joint Feature Embedding ----
        f_concat = torch.cat([f_optical_cross, f_sar_cross, f_dem], dim=1)
        f_joint = self.joint_embedding(f_concat)  # (B, C, H', W')

        # ---- 4. Adaptive Spatial Weight Estimator ----
        alpha_logits = self.alpha_estimator(f_joint)  # (B, 3, H', W')
        alpha_maps = F.softmax(alpha_logits, dim=1)   # Sum to 1 per pixel

        alpha_o = alpha_maps[:, 0:1, :, :]  # (B, 1, H', W')
        alpha_s = alpha_maps[:, 1:2, :, :]
        alpha_d = alpha_maps[:, 2:3, :, :]

        # ---- 5. Weighted Adaptive Fusion ----
        f_fused = (
            alpha_o * f_optical_cross
            + alpha_s * f_sar_cross
            + alpha_d * f_dem
        )  # (B, C, H', W')

        # ---- 6. Fusion Refinement ----
        f_shared = self.refinement(f_fused)  # (B, C, H', W')

        # ---- Build output ----
        output: dict[str, torch.Tensor] = {
            "f_shared": f_shared,
            "alpha_maps": alpha_maps,
        }

        if return_intermediates:
            output["f_optical_cross"] = f_optical_cross
            output["f_sar_cross"] = f_sar_cross
            output["f_joint"] = f_joint

        return output
