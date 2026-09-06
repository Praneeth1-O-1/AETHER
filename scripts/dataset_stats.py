"""Derive the dataset constants baked into `data.dataset` and `train.py`.

Every normalization constant and loss weight in this project is measured from
the archive rather than assumed, and this script is how they were measured.
Re-run it if the archive changes, then paste the printed blocks back into
`data/dataset.py` (band statistics) and `train.py` (loss weights).

Band statistics are computed on raw tiles; loss weights are computed *through*
the Dataset, so they see exactly the masking the loss will see -- road
positives are counted only over mapped tiles, and LULC frequencies only over
observed pixels.

Usage::

    python scripts/dataset_stats.py --dataset-root data/strict --sample 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    NUM_LULC_CLASSES,
    OPTICAL_BANDS,
    SAR_BANDS,
    AETHERTileDataset,
    build_manifest,
    location_split,
)

STRIDE = 53  # subsample pixels within a tile; the estimate is stable well before this matters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--sample", type=int, default=500, help="Tiles to sample for band statistics.")
    p.add_argument("--loss-sample", type=int, default=1200, help="Train tiles to sample for loss weights.")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def band_statistics(tiles: list[Path]) -> dict[str, dict[str, float]]:
    """Per-band mean/std over a tile sample, with nodata excluded."""
    bands = [*OPTICAL_BANDS, *SAR_BANDS, "DEM"]
    acc: dict[str, list[np.ndarray]] = {b: [] for b in bands}
    relief: list[np.ndarray] = []

    for tile in tiles:
        with rasterio.open(tile / "inputs.tif") as src:
            arr = src.read().astype(np.float32)
            names = list(src.descriptions)
            nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            arr = np.where(arr == nodata, np.nan, arr)

        for band in bands:
            values = arr[names.index(band)]
            values = values[np.isfinite(values)]
            if values.size:
                acc[band].append(values[::STRIDE])

        dem = arr[names.index("DEM")]
        if np.isfinite(dem).any():
            centered = dem - np.nanmean(dem)
            relief.append(centered[np.isfinite(centered)][::STRIDE])

    stats = {}
    for band in bands:
        values = np.concatenate(acc[band])
        stats[band] = {"mean": float(values.mean()), "std": float(values.std())}
    centered = np.concatenate(relief)
    stats["DEM_relief"] = {"mean": float(centered.mean()), "std": float(centered.std())}
    return stats


def loss_weights(records, n_sample: int, seed: int) -> dict:
    """Class weights and pos_weights, measured through the Dataset's masking."""
    train, _, _ = location_split(records)
    random.seed(seed)
    sample = random.sample(train, min(n_sample, len(train)))
    dataset = AETHERTileDataset(sample, augment=False)

    counts = np.zeros(NUM_LULC_CLASSES, dtype=np.int64)
    road_pos = road_n = building_sum = building_n = 0.0

    for i in range(len(dataset)):
        item = dataset[i]
        lulc = item["lulc"].numpy()
        observed = lulc != IGNORE_INDEX
        if observed.any():
            counts += np.bincount(lulc[observed], minlength=NUM_LULC_CLASSES)

        road_mask = item["road_mask"].numpy() > 0.5
        road_pos += (item["road"].numpy()[road_mask] > 0.5).sum()
        road_n += road_mask.sum()

        building_mask = item["building_mask"].numpy() > 0.5
        building_sum += item["building"].numpy()[building_mask].sum()
        building_n += building_mask.sum()

    freq = counts / counts.sum()
    # Inverse-sqrt frequency, normalized to mean 1: lifts rare classes without
    # the instability of full inverse weighting (flooded_veg is 0.55% of pixels).
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    weights = weights / weights.mean()

    road_rate = road_pos / max(road_n, 1)
    building_rate = building_sum / max(building_n, 1)
    return {
        "freq": freq, "weights": weights,
        "road_rate": road_rate, "building_rate": building_rate,
        # sqrt-damped: full inverse would be 26x / 78x and make BCE over-predict.
        "road_pos_weight": float(np.sqrt((1 - road_rate) / max(road_rate, 1e-9))),
        "building_pos_weight": float(np.sqrt((1 - building_rate) / max(building_rate, 1e-9))),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)

    records = build_manifest(root)
    tiles = [r.path for r in records]
    random.seed(args.seed)
    sample = random.sample(tiles, min(args.sample, len(tiles)))

    print(f"# Band statistics over {len(sample)} tiles of {root}")
    stats = band_statistics(sample)
    print(f"{'band':<16}{'mean':>12}{'std':>12}")
    for band, s in stats.items():
        print(f"{band:<16}{s['mean']:>12.4f}{s['std']:>12.4f}")

    print("\n# Paste into data/dataset.py")
    fmt = lambda key: ", ".join(f"{stats[b][key]:.4f}" for b in OPTICAL_BANDS)  # noqa: E731
    print(f"OPTICAL_MEAN = np.array([{fmt('mean')}], dtype=np.float32).reshape(-1, 1, 1)")
    print(f"OPTICAL_STD  = np.array([{fmt('std')}], dtype=np.float32).reshape(-1, 1, 1)")
    sar = lambda key: ", ".join(f"{stats[b][key]:.4f}" for b in SAR_BANDS)  # noqa: E731
    print(f"SAR_MEAN = np.array([{sar('mean')}], dtype=np.float32).reshape(-1, 1, 1)")
    print(f"SAR_STD  = np.array([{sar('std')}], dtype=np.float32).reshape(-1, 1, 1)")
    print(f"DEM_MEAN = {stats['DEM']['mean']:.2f}")
    print(f"DEM_STD = {stats['DEM']['std']:.2f}")
    print(f"DEM_RELIEF_STD = {stats['DEM_relief']['std']:.2f}")

    print(f"\n# Loss weights over {min(args.loss_sample, len(records))} train tiles (masked)")
    lw = loss_weights(records, args.loss_sample, args.seed)
    for i, name in enumerate(LULC_CLASS_NAMES):
        print(f"  {i} {name:<12} freq={lw['freq'][i] * 100:6.3f}%  weight={lw['weights'][i]:.4f}")
    print("\n# Paste into train.py")
    print(f"LULC_CLASS_WEIGHTS = {[round(float(w), 4) for w in lw['weights']]}")
    print(f"ROAD_POS_WEIGHT = {lw['road_pos_weight']:.2f}      # positive rate {lw['road_rate']:.5f}")
    print(f"BUILDING_POS_WEIGHT = {lw['building_pos_weight']:.2f}  # coverage {lw['building_rate']:.5f}")


if __name__ == "__main__":
    main()
