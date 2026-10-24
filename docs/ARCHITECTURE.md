# AETHER architecture

How data moves through the model, the tensor shapes at each step, and why each piece is built the way it is. Shapes are for one 256 × 256 tile at 10 m (batch size *B*).

![architecture](assets/architecture.svg)

## 1. Inputs (`data/dataset.py`)

| Stream | Channels | Shape | Preprocessing |
|---|---|---|---|
| Optical | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + validity | `(B, 11, 256, 256)` | reflectance clipped at 1.5, per-band z-score |
| SAR | VV, VH, despeckled VV/VH, VV−VH ratio, validity, orbit direction | `(B, 7, 256, 256)` | dB, z-score |
| DEM | elevation, local relief, validity | `(B, 3, 256, 256)` | z-score |
| Presence | one flag per sensor | `(B, 3)` | 0 when the sensor is missing or dropped |

Normalization constants are fixed archive-wide statistics (`scripts/dataset_stats.py`), never per tile, so no test-time information leaks into standardization. Unlabelled pixels are masked out of every loss: LULC nodata → `IGNORE_INDEX`; tiles with no OpenStreetMap road mapping contribute nothing to the road loss rather than being treated as "no road".

**Split.** Locations are held out whole (every year and sub-tile) by a stable hash of the location key (`location_split`). See the README for the known leak and the `--leak-free` evaluation subset.

## 2. Encoders

Each sensor gets an encoder sized to its information content, all ending in the same `(B, 256, 16, 16)` space.

| Encoder | Backbone | Params | Why |
|---|---|---|---|
| Optical | ConvNeXt-Tiny to stage 3, ImageNet init | 12.46 M | richest signal; natural-image features transfer |
| SAR | ResNet-18 to layer 3, ImageNet init | 2.80 M | speckled, colourless; a lighter backbone suffices |
| DEM | 4-stage strided CNN, from scratch | 0.76 M | single smooth signal; ImageNet offers nothing for height |

The first convolution of each pretrained backbone is widened to the real channel count by tiling the mean pretrained kernel and rescaling by √(3/C). Pretrained backbones train at a lower learning rate (5×10⁻⁵) than everything else (3×10⁻⁴), so their features are not overwritten before the new modules learn.

## 3. CrossModalAlphaFusion (`models/crossmodal_fusion.py`)

Input: `f_O, f_S, f_D` each `(B, 256, 16, 16)` plus presence `p`. Output: `f_shared (B, 256, 16, 16)` and `alpha_maps (B, 3, 16, 16)`.

1. **Gated modality embeddings** — `f_m + p_m · e_m`. An absent sensor stays exactly zero instead of becoming its learned embedding.
2. **Bidirectional cross-attention, optical ↔ SAR** (8 heads over 256 tokens). Each attention *update* is multiplied by the presence of the attended modality, and the LayerNorm output by the presence of the attending one. DEM skips attention: it has no reflectance or backscatter to align against.
3. **Joint embedding** — concatenate (768 ch) → 1×1 conv → 256 ch. Used only to decide the weights.
4. **Alpha estimator** — conv3×3 → GELU → conv1×1 → 3 logits per pixel. Logits of absent sensors are set to −10⁴, then softmax over sensors, so weights sum to 1 over the sensors that are present.
5. **Weighted fusion** — `Σ_m α_m · f′_m`, using the post-attention features (not the joint embedding).
6. **Refinement** — one residual block.

α is never supervised. It moves only because task-loss gradients flow back through it, so it is a direct readout of what the model chose to rely on. Without the presence gates (the pre-September version), a "removed" sensor still leaked into the output through its embedding, the other sensor's attention and a non-zero softmax weight, which made every ablation measurement invalid.

## 4. Decoder and heads

The decoder applies four stages of bilinear ×2 upsampling → 3×3 conv → BN → GELU → squeeze-and-excitation: `(B, 256, 16, 16)` → `(B, 48, 256, 256)`. Presence-gated encoder features are concatenated at H/8, H/4 and H/2. An extra 48-channel optical branch supplies the H/2 skip, because ConvNeXt's stride-4 stem has nothing finer than H/4 and roads are often one pixel wide.

| Head | Layers | Output | Loss |
|---|---|---|---|
| LULC | 1×1 conv | `(B, 9, 256, 256)` | class-weighted cross-entropy, masked |
| Road | 3×3 conv (48→64), GELU, 1×1 | `(B, 1, 256, 256)` | masked BCE (pos_weight 5.09) + soft Dice |
| Building | same as road | `(B, 1, 256, 256)` | masked BCE (pos_weight 8.35) + soft Dice |
| Auxiliary (one per encoder) | 1×1 conv on encoder features | LULC logits | presence-masked cross-entropy |

## 5. Training recipe (`train.py`, `data/clouds.py`)

```
L = L_lulc + 0.5·L_road + 0.5·L_building + 0.3·Σ L_aux − λ(t)·H(α)
```

- **Cloud injection.** 50% of samples get cloud from a bank of real masks (harvested from training locations only) plus fractal noise; 60% opaque (validity → 0), the rest semi-transparent haze (validity stays 1); 15% fully occluded; coverage curriculum 0.3 → 1.0 over the first half of training. Labels stay valid because they come from the clear acquisition.
- **Input-level sensor dropout**, 0.5 per sensor, independent — identical to real absence.
- **OGM-GE gradient modulation** driven by the auxiliary losses: the currently dominant encoder's gradient is damped.
- **Entropy floor on α**, weight 0.02 annealed to 0 over the first 40% of training.
- **Model selection** on 0.5 × clear score + 0.5 × 50%-cloud score.

AdamW, weight decay 0.01, 3% warm-up, gradient clip 1.0, batch 12, fp16 on a T4: about 9.4 minutes per epoch.
