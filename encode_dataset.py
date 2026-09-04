"""AETHER — Batch encode a tiled dataset into fused (f_shared) features.

Runs every tile under a dataset's `strict` directory through the optical/SAR/DEM
encoders and CrossModalAlphaFusion, saving `f_shared` and `alpha_maps` per tile.
The model is untrained (random init) unless --checkpoint is given, so this
validates the encode+fuse pipeline at scale and produces cacheable features —
it does not produce meaningful LULC predictions until the model is trained.

Usage::

    python encode_dataset.py \
        --dataset-root datasets/AETHER_DATASET/data/strict \
        --output-dir outputs/fshared
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

from models.aether import AETHERModel
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Imputation constants for nodata (-9999 -> NaN -> these), matched to the
# per-band physical ranges documented in the dataset README.
OPTICAL_FILL = 0.0
SAR_FILL = -20.0       # dB floor for missing backscatter
DEM_FILL = 1500.0      # local elevation baseline for Kigali


def load_tile(inputs_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one tile's inputs.tif and split into optical/SAR/DEM arrays."""
    with rasterio.open(inputs_path) as src:
        arr = src.read().astype(np.float32)
        names = list(src.descriptions)
        nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            arr = np.where(arr == nodata, np.nan, arr)

    opt_idx = [i for i, n in enumerate(names) if n.startswith("sentinel2_B")]
    sar_idx = [names.index("sentinel1_VV"), names.index("sentinel1_VH")]
    dem_idx = [names.index("DEM")]

    optical = np.nan_to_num(arr[opt_idx], nan=OPTICAL_FILL)
    sar = np.nan_to_num(arr[sar_idx], nan=SAR_FILL)
    dem = np.nan_to_num(arr[dem_idx], nan=DEM_FILL)
    return optical, sar, dem, len(opt_idx)


def build_model(config_path: str, optical_channels: int, checkpoint: str | None,
                 device: torch.device) -> AETHERModel:
    cfg = load_config(config_path)
    cfg.model.optical_encoder.in_channels = optical_channels
    cfg.model.optical_encoder.pretrained = False
    cfg.model.sar_encoder.pretrained = False
    model = AETHERModel.build_from_dict(cfg.model)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded trained weights from {checkpoint}")
    else:
        logger.warning("No --checkpoint given: encoding with a randomly-initialized "
                        "(untrained) model. f_shared will not be semantically meaningful yet.")

    model = model.to(device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path,
                     default=Path("datasets/AETHER_DATASET/data/strict"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/fshared"))
    ap.add_argument("--config", type=str, default="configs/model.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    tile_dirs = sorted(args.dataset_root.glob("*/*"))
    tile_dirs = [d for d in tile_dirs if (d / "inputs.tif").exists()]
    if not tile_dirs:
        raise FileNotFoundError(f"No tiles with inputs.tif found under {args.dataset_root}")

    # Determine optical channel count from the first tile so the model's first
    # conv layer is built with the right shape before any forward pass.
    _, _, _, n_opt = load_tile(tile_dirs[0] / "inputs.tif")
    model = build_model(args.config, n_opt, args.checkpoint, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha_sums = []
    t0 = time.time()
    with torch.no_grad():
        for i, tile_dir in enumerate(tile_dirs, 1):
            tile_id = tile_dir.name
            optical, sar, dem, _ = load_tile(tile_dir / "inputs.tif")

            t_opt = torch.from_numpy(optical).unsqueeze(0).to(device)
            t_sar = torch.from_numpy(sar).unsqueeze(0).to(device)
            t_dem = torch.from_numpy(dem).unsqueeze(0).to(device)

            outputs = model(t_opt, t_sar, t_dem)
            f_shared = outputs["f_shared"].squeeze(0).cpu().numpy()
            alpha_maps = outputs["alpha_maps"].squeeze(0).cpu().numpy()

            np.savez_compressed(
                args.output_dir / f"{tile_id}.npz",
                f_shared=f_shared,
                alpha_maps=alpha_maps,
            )
            alpha_sums.append(alpha_maps.mean(axis=(1, 2)))  # [alpha_o, alpha_s, alpha_d]

            logger.info(f"[{i}/{len(tile_dirs)}] {tile_id}: "
                        f"f_shared {f_shared.shape}, alpha mean "
                        f"O={alpha_sums[-1][0]:.3f} S={alpha_sums[-1][1]:.3f} D={alpha_sums[-1][2]:.3f}")

    elapsed = time.time() - t0
    alpha_sums = np.stack(alpha_sums)
    logger.info(f"Done: {len(tile_dirs)} tiles encoded in {elapsed:.1f}s "
                f"({elapsed / len(tile_dirs):.2f}s/tile)")
    logger.info(f"Dataset-wide mean alpha [Optical, SAR, DEM]: "
                f"{alpha_sums.mean(axis=0).round(3).tolist()}")
    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
