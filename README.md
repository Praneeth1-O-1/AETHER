# AETHER

**Adaptive Earth observation through heterogeneous encoder representation.** AETHER maps land cover, roads and buildings from three satellite sources at once — Sentinel-2 optical, Sentinel-1 radar and a Copernicus elevation model — and decides *per pixel* how much to trust each. When cloud hides the optical image, it shifts its weight to radar and elevation instead of failing.

![AETHER architecture](docs/assets/architecture.svg)

## What it does

| | |
|---|---|
| **Input** | A 256 × 256 tile at 10 m: 10 Sentinel-2 bands, Sentinel-1 VV/VH backscatter, Copernicus DEM, each with a validity mask |
| **Output** | A 9-class land-cover map (Dynamic World classes), a road map, a building-coverage map, and a **reliance map** — the per-pixel weight given to optical, radar and elevation |
| **Model** | 19.37 M parameters, 31.4 GFLOPs per tile |

## The key idea

A network trained on mostly clear-sky imagery learns to use optical alone. Our first model did exactly that: its radar weight stayed at about 0.19 whether or not there was cloud. AETHER fixes this in two places.

1. **Presence-aware fusion** (`models/crossmodal_fusion.py`). Each sensor is either fully present or *exactly* absent: an absent sensor gets no modality embedding, no cross-attention contribution, zero fusion weight and no skip connection. A per-pixel softmax then splits the weight among the sensors that are there.
2. **Training that makes optical fail** (`data/clouds.py`, `train.py`). Real cloud masks are injected into clear tiles — whose labels stay correct underneath — with a coverage curriculum, alongside input-level sensor dropout, per-sensor auxiliary heads, gradient modulation and model selection on a cloudy validation score.

## Results

Epoch-41 checkpoint, **validation** split (1,894 tiles from held-out locations):

| Condition | LULC accuracy | mIoU | Weight: optical / radar / elevation |
|---|---|---|---|
| Clear sky | 0.833 | 0.657 | 0.895 / 0.077 / 0.028 |
| 50% cloud | 0.783 | 0.587 | 0.761 / 0.177 / 0.063 |
| 100% cloud | 0.594 | 0.363 | 0.000 / **0.686** / 0.314 |

As cloud increases, the radar weight rises roughly ninefold and the optical weight drops to exactly zero.

![Radar fusion weight during training](docs/assets/fig_alpha.png)

One tile, predicted with every combination of sensors (pixel accuracy under each panel):

![Sensor combinations](docs/assets/fig_combos.png)

> Held-out **test-set** metrics, the full sensor-combination ablation, the cloud-degradation curve and GPU benchmarks are produced by `scripts/run_paper_eval.sh`; see [Reproducing the evaluation](#reproducing-the-evaluation).

## Dataset

19,149 co-registered tiles from 1,372 locations across Africa, Asia and Latin America (2017–2023), including 6,807 deliberately cloud-affected variants. Labels: Dynamic World land cover, OpenStreetMap roads, building-footprint coverage. Unlabelled pixels are masked out of every loss rather than treated as negatives. Train/validation/test are split by **location**, never by tile.

The archive is hosted on Kaggle: [`kirethikb/aether-real-cloud-multimodal-eo`](https://www.kaggle.com/datasets/kirethikb/aether-real-cloud-multimodal-eo).

## Installation

```bash
git clone https://github.com/Praneeth1-O-1/AETHER.git
cd AETHER
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.0+. Training needs a CUDA GPU (developed on an NVIDIA T4).

## Usage

Every setting lives in `settings.py` — set `DATASET_ROOT` and `CHECKPOINT`, then:

```bash
python run_train.py      # train (writes checkpoints_skip/best.pt, history.csv)
python run_predict.py    # predict sample tiles and render PNGs
python run_evaluate.py   # full metric report across data scenarios
python run_compare.py    # ground truth vs. prediction, side by side
```

Single tile:

```bash
python inference.py --checkpoint checkpoints_skip/best.pt --tile-dir <tile_dir> --save-alpha-maps
```

### Evaluation scripts

| Script | What it measures |
|---|---|
| `scripts/tune_thresholds.py` | Road/building thresholds tuned on validation, then test metrics |
| `scripts/evaluate_full.py` | Per-class precision / recall / F1 / IoU and Cohen's κ, per data scenario |
| `scripts/ablate_modalities.py` | Every sensor combination: single, pairs, all three |
| `scripts/cloud_curve.py` | Performance and fusion weights at 0–100% cloud |
| `scripts/benchmark_gpu.py` | Parameters, FLOPs, latency, throughput, memory |
| `scripts/dataset_stats.py` | Re-derives normalization constants and loss weights |

All evaluation scripts accept `--leak-free` (see Limitations).

### Reproducing the evaluation

```bash
bash scripts/run_paper_eval.sh <checkpoint.pt> <dataset_root>
```

writes every result as JSON to `outputs/paper/`.

## Repository layout

```
models/        encoders, CrossModalAlphaFusion, decoder, task heads, losses
data/          dataset loading, location split, cloud injection
scripts/       evaluation, ablation, benchmarking, visualization
utils/         metrics and config helpers
configs/       model architecture (model.yaml)
docs/          design notes, reports, figures
settings.py    every training / evaluation setting in one place
train.py       training loop
inference.py   prediction and evaluation entry point
```

## Limitations

- **Split leak.** Location keys keep a `g_`/`c_` prefix and two spellings of some region names, so a standard tile and its cloud variant at the same coordinates can fall in different splits (185 test / 128 validation tiles in the current split). Re-keying would invalidate the trained checkpoint, so evaluation scripts report a `--leak-free` subset alongside the full split.
- **Single-date radar.** Each tile has one Sentinel-1 acquisition; many classes are better separated by radar time series.
- **Orbit channel unused.** The archive does not record orbit direction, so that SAR input channel is constant.
- Road and building IoU remain below 0.3.

## Team

Group 108 — [names]

## License

[To be decided.] Dataset labels carry their own licenses (OpenStreetMap: ODbL; Dynamic World: CC BY 4.0).
