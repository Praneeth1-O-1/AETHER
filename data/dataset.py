"""AETHER — Dataset loading, geographic split, and augmentation.

Reads tiles from the AETHER_DATASET `strict` archive (inputs.tif + labels.tif
per tile) into (optical, sar, dem, lulc_target) tensors ready for the model.

Key choices, and why:

- **Split is spatial, not random.** The archive is 12 spatial locations x 7
  yearly snapshots. Nearby years of the same location are highly
  autocorrelated, so a random shuffle would leak the same ground truth across
  train/val/test. `spatial_split` instead holds out whole spatial locations
  (all years) for val/test.
- **LULC class 0 (water) is a real class, not "no data".** Unlabeled pixels
  in `labels.tif` are NaN and must map to `IGNORE_INDEX`, never to 0.
- **Optical is already ~[0,1] reflectance** (see README), so it is left
  unscaled. SAR (dB) and DEM (metres) are on very different numeric scales
  and are standardized with fixed, dataset-level constants derived from the
  archive's documented value ranges -- an approximation, not per-tile stats,
  to avoid leaking any per-tile information into standardization.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = 255
NUM_LULC_CLASSES = 9  # Dynamic World: water, trees, grass, flooded_veg, crops, shrub, built, bare, snow_ice

OPTICAL_FILL = 0.0
SAR_FILL = -20.0
DEM_FILL = 1500.0

SAR_MEAN = np.array([-9.0, -15.0], dtype=np.float32).reshape(2, 1, 1)
SAR_STD = np.array([6.0, 6.0], dtype=np.float32).reshape(2, 1, 1)
DEM_MEAN = 1650.0
DEM_STD = 300.0

_TILE_RE = re.compile(r"_r(\d+)_c(\d+)$")


def spatial_id(tile_dir: Path) -> str:
    """Extract the row/col spatial id (e.g. 'r000_c001') from a tile dir name."""
    m = _TILE_RE.search(tile_dir.name)
    if not m:
        raise ValueError(f"Could not parse spatial id from tile dir: {tile_dir.name}")
    return f"r{m.group(1)}_c{m.group(2)}"


def list_tiles(dataset_root: Path) -> list[Path]:
    tiles = sorted(p for p in dataset_root.glob("*/*") if (p / "inputs.tif").exists())
    if not tiles:
        raise FileNotFoundError(f"No tiles found under {dataset_root}")
    return tiles


def spatial_split(
    dataset_root: Path, n_val: int = 2, n_test: int = 2
) -> tuple[list[Path], list[Path], list[Path]]:
    """Hold out whole spatial locations (every year of them) for val/test.

    Deterministic (sorted spatial ids, last n_test held out for test, the
    n_val before that for val) rather than random, so splits are reproducible
    without seeding.
    """
    tiles = list_tiles(dataset_root)
    ids = sorted({spatial_id(t) for t in tiles})
    if n_val + n_test >= len(ids):
        raise ValueError(
            f"Not enough spatial locations ({len(ids)}) for n_val={n_val} + n_test={n_test}"
        )

    test_ids = set(ids[len(ids) - n_test:]) if n_test else set()
    val_ids = set(ids[len(ids) - n_test - n_val: len(ids) - n_test]) if n_val else set()
    train_ids = set(ids) - val_ids - test_ids

    train = [t for t in tiles if spatial_id(t) in train_ids]
    val = [t for t in tiles if spatial_id(t) in val_ids]
    test = [t for t in tiles if spatial_id(t) in test_ids]
    return train, val, test


def _read_stack(path: Path) -> tuple[np.ndarray, list[str]]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        names = list(src.descriptions)
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, names


class AETHERTileDataset(Dataset):
    """Loads (optical, sar, dem, lulc_target, road_target, building_target) tensors.

    ``road`` and ``building_presence`` (unlike ``lulc``) have no nodata pixels
    anywhere in the archive (checked across the dataset), so no ignore-index
    handling is needed for them. ``building_presence`` is a continuous [0,1]
    per-pixel building-coverage fraction, not a hard 0/1 label -- it's used
    as-is as a soft target for ``BCEWithLogitsLoss``, which preserves partial
    building-edge coverage instead of collapsing it to a hard threshold.
    """

    def __init__(self, tile_dirs: list[Path], augment: bool = False):
        self.tile_dirs = tile_dirs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tile_dirs)

    def __getitem__(self, idx: int):
        tile_dir = self.tile_dirs[idx]

        arr, names = _read_stack(tile_dir / "inputs.tif")
        opt_idx = [i for i, n in enumerate(names) if n.startswith("sentinel2_B")]
        sar_idx = [names.index("sentinel1_VV"), names.index("sentinel1_VH")]
        dem_idx = [names.index("DEM")]

        optical = np.nan_to_num(arr[opt_idx], nan=OPTICAL_FILL)

        sar = np.nan_to_num(arr[sar_idx], nan=SAR_FILL)
        sar = (sar - SAR_MEAN) / SAR_STD

        dem = np.nan_to_num(arr[dem_idx], nan=DEM_FILL)
        dem = (dem - DEM_MEAN) / DEM_STD

        lab_arr, lab_names = _read_stack(tile_dir / "labels.tif")
        lulc = lab_arr[lab_names.index("lulc")]
        lulc_target = np.where(np.isfinite(lulc), np.round(lulc), IGNORE_INDEX).astype(np.int64)

        road_target = lab_arr[lab_names.index("road")][None, ...].astype(np.float32)
        building_target = lab_arr[lab_names.index("building_presence")][None, ...].astype(np.float32)

        if self.augment:
            optical, sar, dem, lulc_target, road_target, building_target = self._augment(
                optical, sar, dem, lulc_target, road_target, building_target
            )

        return (
            torch.from_numpy(optical.copy()),
            torch.from_numpy(sar.copy()),
            torch.from_numpy(dem.copy()),
            torch.from_numpy(lulc_target.copy()),
            torch.from_numpy(road_target.copy()),
            torch.from_numpy(building_target.copy()),
        )

    @staticmethod
    def _augment(optical, sar, dem, lulc_target, road_target, building_target):
        arrays = [optical, sar, dem, lulc_target, road_target, building_target]
        if np.random.rand() < 0.5:
            arrays = [np.flip(a, axis=-1) for a in arrays]
        if np.random.rand() < 0.5:
            arrays = [np.flip(a, axis=-2) for a in arrays]
        k = int(np.random.randint(0, 4))
        if k:
            arrays = [np.rot90(a, k, axes=(-2, -1)) for a in arrays]
        return arrays
