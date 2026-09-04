# AETHER Architecture & CrossModalAlphaFusion Context Guide

> **Document Target**: Technical handoff document for the team member working on the `CrossModalAlphaFusion` module and multimodal integration in **AETHER** (*Adaptive Earth Observation Through Heterogeneous Encoder Representation*).

---

## 1. Project Overview & Current Status

**AETHER** is a multimodal geospatial intelligence framework designed for dense prediction tasks (Land Use / Land Cover Classification, Road Extraction, Building Extraction, and Change Detection).

### Current Implementation Status
- ✅ **Optical Encoder (`OpticalEncoder`)**: Implemented (ConvNeXt-Tiny stage 3 backbone).
- ✅ **SAR Encoder (`SAREncoder`)**: Implemented (ResNet-18 layer 3 backbone).
- ✅ **DEM Encoder (`DEMEncoder`)**: Implemented (Custom lightweight CNN).
- ✅ **Fusion Module (`CrossModalAlphaFusion`)**: Implemented baseline with bidirectional cross-attention, spatial alpha maps, and residual refinement.
- ✅ **Progressive Decoder (`ProgressiveDecoder`)**: Implemented (Upsamples $H/16 \times W/16 \to H \times W$).
- ✅ **Task Heads**: LULC segmentation head implemented.
- ✅ **Validation & Verification**:
  - Full end-to-end forward/backward gradient flow verified (`test_aether.py`).
  - Evaluated on **real satellite imagery** (Sentinel-2, Sentinel-1, Copernicus DEM) from Microsoft Planetary Computer (`evaluate_real_data.py`).
  - Spatial alpha maps verified to strictly satisfy the $\sum \alpha = 1.0$ constraint per pixel.

---

## 2. Input Modalities & Feature Dimensions

All input imagery is processed at native input spatial dimensions $H \times W$ (typically $256 \times 256$). The three independent encoders downsample spatial resolution by a factor of 16 (stride 16) and project to a uniform channel dimension $C = 256$.

### Summary Table

| Modality | Sensor / Source | Input Tensor Shape | Encoder Output Feature Shape |
|---|---|---|---|
| **Optical** | Sentinel-2 (13 bands) | `(B, 13, H, W)` | `(B, 256, H/16, W/16)` |
| **SAR** | Sentinel-1 (VV, VH) | `(B, 2, H, W)` | `(B, 256, H/16, W/16)` |
| **DEM** | Copernicus DEM (Elevation) | `(B, 1, H, W)` | `(B, 256, H/16, W/16)` |
| **Fused Output ($F_{\text{shared}}$)** | CrossModalAlphaFusion | `3x (B, 256, H/16, W/16)` | `(B, 256, H/16, W/16)` |
| **Alpha Maps** | Adaptive Spatial Estimator | `3x (B, 256, H/16, W/16)` | `(B, 3, H/16, W/16)` |
| **Decoded Output** | ProgressiveDecoder | `(B, 256, H/16, W/16)` | `(B, 16, H, W)` |
| **Task Predictions** | e.g. LULC Head | `(B, 16, H, W)` | `(B, num_classes, H, W)` |

---

## 3. Encoder Architecture Breakdown

> 🚨 **Critical Design Rule**: Modalities do **NOT** share weights. Each encoder has an independent feature extractor.

### 3.1 Optical Encoder (`models/optical_encoder.py`)
- **Backbone**: ConvNeXt-Tiny (truncated at Stage 3, deepest stage with 9 blocks).
- **First-Layer Surgery**: Replaces the standard 3-channel stem conv with a 13-channel conv while retaining pretrained ImageNet weights for channels 0..2.
- **Output**: Feature map $F_{\text{optical}} \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}$.

### 3.2 SAR Encoder (`models/sar_encoder.py`)
- **Backbone**: ResNet-18 (truncated at Layer 3).
- **First-Layer Surgery**: Replaces 3-channel `conv1` with a 2-channel conv (`in_channels=2` for VV, VH).
- **Output**: Feature map $F_{\text{sar}} \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}$.

### 3.3 DEM Encoder (`models/dem_encoder.py`)
- **Backbone**: Lightweight custom CNN (No transformer backbone for DEM).
- **Structure**: 4 strided Conv2d blocks with BatchNorm + GELU up to 256 channels.
- **Output**: Feature map $F_{\text{dem}} \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}$.

---

## 4. `CrossModalAlphaFusion` Module Deep-Dive

Located in `models/crossmodal_fusion.py`, this module performs spatially-varying adaptive feature fusion.

```
       f_optical (B, 256, H/16, W/16)    f_sar (B, 256, H/16, W/16)    f_dem (B, 256, H/16, W/16)
                    │                                │                            │
          + emb_optical                    + emb_sar                    + emb_dem
                    │                                │                            │
                    └───────────┬────────────────────┘                            │
                                ▼                                                 │
                   Bidirectional Cross-Attention                                  │
                       (SAR ↔ Optical)                                           │
                                │                                                 │
                    ┌───────────┴────────────────────┐                            │
                    ▼                                ▼                            │
             f_optical_cross                   f_sar_cross                        │
                    │                                │                            │
                    └────────────────────────────────┴──────────────┬─────────────┘
                                                                    │
                                                                    ▼
                                                            Concatenate (3C)
                                                                    │
                                                                    ▼
                                                          Joint Feature Embedding
                                                             (Conv1x1 -> BN -> GELU)
                                                                    │
                                                                    ▼
                                                                f_joint
                                                                /     \
                                                               /       \
                                                              /         \
                                                             ▼           ▼
                                                     Spatial Weight     Adaptive Spatial
                                                       Estimator        Alpha Maps (α_O, α_S, α_D)
                                                      (Conv3x3->GELU     (Softmax along dim=1)
                                                       ->Conv1x1)               │
                                                           │                    │
                                                           └─────────┬──────────┘
                                                                     ▼
                                                         Spatially Weighted Sum
                                                     α_O * f_optical_cross +
                                                     α_S * f_sar_cross +
                                                     α_D * f_dem
                                                                     │
                                                                     ▼
                                                             f_fused (256ch)
                                                                     │
                                                                     ▼
                                                             Fusion Refinement
                                                            (Residual Conv3x3 Block)
                                                                     │
                                                                     ▼
                                                             f_shared (256ch)
```

### Step-by-Step Processing Flow

1. **Modality Embeddings**
   - Add learnable spatial 1D vectors $E_O, E_S, E_D \in \mathbb{R}^{1 \times 256 \times 1 \times 1}$ to explicitely encode modality identity:
     $$\hat{F}_O = F_O + E_O, \quad \hat{F}_S = F_S + E_S, \quad \hat{F}_D = F_D + E_D$$

2. **Bidirectional Cross-Attention (SAR $\leftrightarrow$ Optical)**
   - Reshape features to sequence representation $(B, N, C)$ where $N = (H/16) \times (W/16)$.
   - Optical attends to SAR: $F_{O\to S} = \text{MultiheadAttention}(Q=\hat{F}_O, K=\hat{F}_S, V=\hat{F}_S)$
   - SAR attends to Optical: $F_{S\to O} = \text{MultiheadAttention}(Q=\hat{F}_S, K=\hat{F}_O, V=\hat{F}_O)$
   - Apply Residual connection + LayerNorm, reshape back to spatial $(B, C, H/16, W/16)$.

3. **Joint Feature Embedding**
   - Concatenate $F_{O\to S}$, $F_{S\to O}$, and $\hat{F}_D$ along channel axis: shape $(B, 3C, H/16, W/16)$.
   - Project via $\text{Conv1x1}(768 \to 256) \to \text{BatchNorm2d} \to \text{GELU}$ to produce $F_{\text{joint}} \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}$.

4. **Adaptive Spatial Weight Estimator (Alpha Maps)**
   - Process $F_{\text{joint}}$ via $\text{Conv3x3}(256 \to 128) \to \text{GELU} \to \text{Conv1x1}(128 \to 3)$ to obtain logits $(B, 3, H/16, W/16)$.
   - Apply Softmax over channel dimension `dim=1`:
     $$\alpha_O, \alpha_S, \alpha_D = \text{Softmax}(\text{logits}, \text{dim}=1)$$
   - **Constraint**: At every single pixel $(x, y)$, $\alpha_O(x,y) + \alpha_S(x,y) + \alpha_D(x,y) = 1.0$.

5. **Weighted Adaptive Fusion**
   - Perform element-wise convex combination:
     $$F_{\text{fused}} = \alpha_O \odot F_{O\to S} + \alpha_S \odot F_{S\to O} + \alpha_D \odot \hat{F}_D$$

6. **Fusion Refinement**
   - Pass $F_{\text{fused}}$ through 1 residual block: $\text{Conv3x3} \to \text{BN} \to \text{GELU} \to \text{Conv3x3} \to \text{BN} + \text{Residual} \to \text{GELU}$.
   - Yields final fused feature map $F_{\text{shared}} \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}$.

---

## 5. Python API & Contract

### Initialization Parameters (`CrossModalAlphaFusion`)
```python
CrossModalAlphaFusion(
    feature_dim=256,              # Feature channel depth
    num_heads=8,                  # Cross-attention heads
    dropout=0.1,                  # Attention dropout rate
    use_modality_embeddings=True, # Learnable 1x1 modality tokens
    num_refinement_blocks=1,      # Number of post-fusion residual blocks
    use_se_refinement=False,      # Squeeze-and-Excitation in refinement
    se_reduction=16,
)
```

### Forward Signature
```python
def forward(
    self,
    f_optical: torch.Tensor,       # (B, 256, H/16, W/16)
    f_sar: torch.Tensor,           # (B, 256, H/16, W/16)
    f_dem: torch.Tensor,           # (B, 256, H/16, W/16)
    return_intermediates: bool = False,
) -> dict[str, torch.Tensor]
```

### Return Dictionary Keys

| Key | Tensor Shape | Description | Always Returned? |
|---|---|---|---|
| `"f_shared"` | `(B, 256, H/16, W/16)` | Primary output feature map for decoder | ✅ Yes |
| `"alpha_maps"` | `(B, 3, H/16, W/16)` | Spatial attention weights ($\sum=1.0$) | ✅ Yes |
| `"f_optical_cross"` | `(B, 256, H/16, W/16)` | Cross-attended optical features | Only if `return_intermediates=True` |
| `"f_sar_cross"` | `(B, 256, H/16, W/16)` | Cross-attended SAR features | Only if `return_intermediates=True` |
| `"f_joint"` | `(B, 256, H/16, W/16)` | Pre-alpha joint feature representation | Only if `return_intermediates=True` |

---

## 6. How to Test / Modify the Fusion Module

1. **Configuration**: Architectural options are set in `configs/model.yaml` under `model.fusion`.
2. **Run Unit & Gradient Verification**:
   ```bash
   /Users/praneeth/fall_detection/.venv/bin/python test_aether.py
   ```
3. **Run Real Satellite Data Pipeline Test**:
   ```bash
   /Users/praneeth/fall_detection/.venv/bin/python evaluate_real_data.py
   ```
   Outputs will update in `outputs/` including:
   - `alpha_optical.png`, `alpha_sar.png`, `alpha_dem.png`
   - `fshared_mean.png`, `feature_optical_mean.png`, etc.

---

## 7. Key Focus Areas for Further Fusion Research

If you are expanding or experimenting with the `CrossModalAlphaFusion` module:
- **Spatial vs. Channel Alpha Maps**: Exploring 3D spatial-channel attention $\alpha \in \mathbb{R}^{B \times 3 \times C \times H' \times W'}$.
- **Gated DEM Integration**: Evaluating whether DEM should participate directly in cross-attention or remain an auxiliary spatial prior.
- **Ablation Flags**: Use `return_intermediates=True` to inspect feature maps pre- and post-fusion.
