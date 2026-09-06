"""AETHER — Dataset loading, geographic split, and augmentation.

Reads tiles from the AETHER_DATASET `strict` archive (inputs.tif + labels.tif
per tile) into model-ready tensors.

Key choices, and why:

- **Split is by geographic location, year-blind.** The archive is 544 AOIs
  across 22 countries, each cut into a 3x2..3x6 grid of 256x256 tiles. The
  `_rXXX_cYYY` suffix is a tile's position *within* its AOI, not a place --
  there are only 16 distinct suffixes archive-wide, so splitting on it puts
  every AOI in every split and leaks neighbouring ground truth. We group on
  the AOI name with the trailing year stripped, so the same lat/lon in 2021
  and 2023 always lands on the same side of the split.

- **Nothing is trained on fabricated data.** A band the pipeline could not
  acquire is written as NaN (or the -9999 nodata sentinel) and flagged in
  meta.json's ``missing`` list. Inputs are filled so the tensors are finite,
  but every fill is accompanied by a validity channel, and every label that
  was never observed is masked out of the loss rather than taught as a zero.

- **LULC class 0 (water) is a real class, not "no data".** Unlabeled pixels
  in `labels.tif` are NaN and map to `IGNORE_INDEX`, never to 0.

- **Normalization constants are measured, not assumed.** Per-band mean/std
  come from a 500-tile sample of this archive (see `scripts/band_stats`).
  They are fixed dataset-level constants, never per-tile statistics, so no
  per-tile information leaks into standardization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset

from data.clouds import CloudInjector

logger = logging.getLogger(__name__)

IGNORE_INDEX = 255
NUM_LULC_CLASSES = 9  # Dynamic World: water, trees, grass, flooded_veg, crops, shrub, built, bare, snow_ice

LULC_CLASS_NAMES = (
    "water", "trees", "grass", "flooded_veg", "crops",
    "shrub", "built", "bare", "snow_ice",
)

# Measured over a 500-tile random sample of data/strict.
OPTICAL_BANDS = (
    "sentinel2_B2", "sentinel2_B3", "sentinel2_B4", "sentinel2_B5", "sentinel2_B6",
    "sentinel2_B7", "sentinel2_B8", "sentinel2_B8A", "sentinel2_B11", "sentinel2_B12",
)
OPTICAL_MEAN = np.array(
    [0.0682, 0.0940, 0.1082, 0.1442, 0.2110, 0.2364, 0.2440, 0.2543, 0.2245, 0.1644],
    dtype=np.float32,
).reshape(-1, 1, 1)
OPTICAL_STD = np.array(
    [0.0849, 0.0870, 0.1094, 0.1071, 0.1016, 0.1062, 0.1110, 0.1086, 0.1238, 0.1211],
    dtype=np.float32,
).reshape(-1, 1, 1)

SAR_BANDS = ("sentinel1_VV", "sentinel1_VH")
SAR_MEAN = np.array([-11.6547, -18.6541], dtype=np.float32).reshape(-1, 1, 1)
SAR_STD = np.array([5.0422, 5.3633], dtype=np.float32).reshape(-1, 1, 1)

# Optical is clipped at OPTICAL_CLIP but SAR was not, so SAR outliers reached
# -51 dB and +15 dB -- i.e. -6.1 to +5.3 sigma -- while optical was held to a
# civilised range. Asymmetric treatment that disadvantaged the modality we are
# trying to make load-bearing. These bounds keep >99% of measured values
# (pooled p01/p99: VV -24.5/+1.6, VH -33.7/-6.6) and truncate the tail.
SAR_CLIP_LO = np.array([-30.0, -35.0], dtype=np.float32).reshape(-1, 1, 1)
SAR_CLIP_HI = np.array([5.0, 0.0], dtype=np.float32).reshape(-1, 1, 1)

# VV - VH in dB is the log of the co/cross-pol ratio, a standard and strongly
# discriminative land-cover feature (surface vs volume vs double-bounce
# scattering). Statistics measured over the same 500-tile sample.
SAR_RATIO_MEAN = np.float32(7.0)
SAR_RATIO_STD = np.float32(3.0)

# Window for the despeckled companion channels. SAR carries 14-18% incoherent
# variance against optical's 11-12%; a 5x5 boxcar in dB is a cheap multilook
# proxy. Kept ALONGSIDE the raw bands, never replacing them -- despeckling
# trades resolution for radiometric stability and the encoder should choose.
SAR_DESPECKLE_WINDOW = 5

DEM_MEAN = 875.29
DEM_STD = 925.65
DEM_RELIEF_STD = 81.30

# Reflectance has a thin tail above 1.0 (specular / snow). Clip in physical
# units before standardizing so a handful of pixels can't dominate a batch.
OPTICAL_CLIP = 1.5

# Number of channels each encoder receives, after validity channels are appended.
OPTICAL_CHANNELS = len(OPTICAL_BANDS) + 1   # + sentinel2_valid
SAR_CHANNELS = 7                            # VV, VH, VV_dspk, VH_dspk, ratio, valid, orbit
DEM_CHANNELS = 3                            # absolute elevation + local relief + validity

# Modality order used by the presence vector and the fusion alpha maps.
MODALITIES = ("optical", "sar", "dem")

# Ascending and descending passes view the same ground from opposite look
# azimuths at different local incidence angles, so identical land cover
# produces systematically different backscatter. The archive mixes 26% ASC /
# 74% DESC across 89 relative orbits, and orbit direction is confounded with
# country in 17 of 22 cases -- under a geographic split that is a large slice
# of SAR variance the network cannot explain, which makes ignoring SAR the
# rational optimum. Telling it which geometry it is looking at costs 1 channel.
ORBIT_CODES = {"ASCENDING": 1.0, "DESCENDING": -1.0, "UNKNOWN": 0.0}

_YEAR_RE = re.compile(r"_(19|20)\d{2}$")


# =========================================================================
# Manifest
# =========================================================================


@dataclass(frozen=True)
class TileRecord:
    """One tile, plus which of its bands are real measurements.

    The ``has_*`` flags come from meta.json's ``missing`` list. They drive
    loss masking: a tile whose ``road`` band is missing has that band written
    as all-zeros, which is *absence of mapping*, not absence of road, and
    must not be backpropagated as a negative.
    """

    path: Path
    location: str
    year: str
    has_optical: bool
    has_sar: bool
    has_lulc: bool
    has_road: bool
    has_building: bool
    orbit: str = "UNKNOWN"


def location_of(aoi_name: str) -> str:
    """AOI name with the trailing year stripped: the geographic location."""
    return _YEAR_RE.sub("", aoi_name)


def year_of(aoi_name: str) -> str:
    m = _YEAR_RE.search(aoi_name)
    return m.group(0)[1:] if m else "unknown"


def _missing_set(meta: dict) -> set[str]:
    """Normalize meta.json's ``missing`` entries to bare band names.

    Entries can be decorated, e.g. ``lulc:low_validity(0.0%)``. A partially
    valid LULC band is still usable -- its invalid pixels are NaN and become
    IGNORE_INDEX -- so only a bare ``lulc`` counts as missing.
    """
    return {str(m) for m in meta.get("missing", [])}


def _orbit_of(meta: dict) -> str:
    """Sentinel-1 pass direction, read from the acquisition note.

    The archive records it only inside a free-text note ("... ASCENDING rel.
    orbit 103, 10 m, AOI coverage 100.0% ..."), so this is a substring test
    rather than a field lookup.
    """
    note = ((meta.get("acquisition") or {}).get("sentinel1") or {}).get("note") or ""
    if "ASCENDING" in note:
        return "ASCENDING"
    if "DESCENDING" in note:
        return "DESCENDING"
    return "UNKNOWN"


def build_manifest(dataset_root: Path, cache: Path | None = None,
                   refresh: bool = False) -> list[TileRecord]:
    """Scan the archive once and cache the per-tile provenance flags.

    Reading 6.4k meta.json files takes a few seconds; the DataLoader workers
    would otherwise repeat that on every run.
    """
    dataset_root = Path(dataset_root)
    if cache is None:
        cache = dataset_root.parent / f"manifest_{dataset_root.name}.json"
    cache = Path(cache)

    if cache.exists() and not refresh:
        raw = json.loads(cache.read_text())
        # A cache written before orbit direction was recorded is missing the
        # field entirely; rebuild rather than silently defaulting every tile to
        # UNKNOWN, which would quietly disable the orbit channel.
        if raw and "orbit" in raw[0]:
            return [
                TileRecord(
                    path=dataset_root / r["rel"], location=r["location"], year=r["year"],
                    has_optical=r["has_optical"], has_sar=r["has_sar"],
                    has_lulc=r["has_lulc"], has_road=r["has_road"],
                    has_building=r["has_building"], orbit=r["orbit"],
                )
                for r in raw
            ]
        logger.info(f"Manifest cache {cache} predates the orbit field -- rebuilding.")

    tiles = sorted(p for p in dataset_root.glob("*/*") if (p / "inputs.tif").exists())
    if not tiles:
        raise FileNotFoundError(f"No tiles found under {dataset_root}")

    records: list[TileRecord] = []
    for t in tiles:
        meta_path = t / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        missing = _missing_set(meta)
        aoi = t.parent.name
        records.append(
            TileRecord(
                path=t,
                location=location_of(aoi),
                year=year_of(aoi),
                has_optical="sentinel2" not in missing,
                has_sar="sentinel1" not in missing,
                has_lulc="lulc" not in missing,
                has_road="road" not in missing,
                has_building="building" not in missing,
                orbit=_orbit_of(meta),
            )
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        [
            {
                "rel": str(r.path.relative_to(dataset_root)), "location": r.location,
                "year": r.year, "has_optical": r.has_optical, "has_sar": r.has_sar,
                "has_lulc": r.has_lulc, "has_road": r.has_road,
                "has_building": r.has_building, "orbit": r.orbit,
            }
            for r in records
        ],
        indent=1,
    ))
    return records


def _bucket(location: str, n_buckets: int = 1000) -> int:
    """Stable hash bucket for a location, independent of directory order."""
    h = hashlib.md5(location.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_buckets


def location_split(
    records: list[TileRecord], val_frac: float = 0.10, test_frac: float = 0.10,
) -> tuple[list[TileRecord], list[TileRecord], list[TileRecord]]:
    """Hold out whole geographic locations (every year and every sub-tile).

    Assignment is by stable hash of the location string, so the split is
    reproducible without seeding and stays fixed if tiles are added later.
    """
    if not 0 <= val_frac + test_frac < 1:
        raise ValueError(f"val_frac + test_frac must be in [0, 1), got {val_frac + test_frac}")

    test_cut = int(round(test_frac * 1000))
    val_cut = test_cut + int(round(val_frac * 1000))

    train, val, test = [], [], []
    for r in records:
        b = _bucket(r.location)
        if b < test_cut:
            test.append(r)
        elif b < val_cut:
            val.append(r)
        else:
            train.append(r)
    return train, val, test


# =========================================================================
# Raster IO
# =========================================================================


def _read_stack(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a GeoTIFF to float32 with every nodata representation as NaN."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        names = list(src.descriptions)
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, names


# =========================================================================
# Dataset
# =========================================================================


class AETHERTileDataset(Dataset):
    """Loads one tile into model inputs plus per-task loss masks.

    Returns a dict with:

    ``optical``   (11, H, W)  10 standardized S2 bands + validity channel
    ``sar``       (7,  H, W)  VV/VH, despeckled VV/VH, VV-VH ratio, validity, orbit
    ``dem``       (3,  H, W)  absolute elevation + local relief + validity
    ``presence``  (3,)        1.0 where {optical, sar, dem} is genuinely present
    ``cloud_cov`` ()          fraction of the tile covered by injected cloud
    ``lulc``      (H, W)      int64 class index, IGNORE_INDEX where unobserved
    ``road``      (1, H, W)   float32 in [0, 1]
    ``road_mask`` (1, H, W)   1 where the road label is a real observation
    ``building``      (1, H, W)  float32 coverage fraction in [0, 1]
    ``building_mask`` (1, H, W)  1 where the building label is a real observation

    The masks are what keep 1,515 road-unmapped tiles and 352 LULC-blank tiles
    from teaching the model that unmapped means empty.
    """

    def __init__(
        self,
        records: list[TileRecord],
        augment: bool = False,
        cloud_injector: CloudInjector | None = None,
        drop_probs: tuple[float, float, float] | None = None,
        deterministic_drop: bool = False,
    ):
        """
        Parameters
        ----------
        cloud_injector : CloudInjector, optional
            Applies synthetic cloud to the optical bands. See `data.clouds`.
        drop_probs : tuple, optional
            Per-modality dropout probability ``(optical, sar, dem)``, applied
            independently. ``None`` disables dropout.
        deterministic_drop : bool
            Seed dropout by tile index. Evaluation only.
        """
        self.records = list(records)
        self.augment = augment
        self.cloud_injector = cloud_injector
        self.drop_probs = drop_probs
        self.deterministic_drop = deterministic_drop

    def __len__(self) -> int:
        return len(self.records)

    # -- inputs ---------------------------------------------------------

    def _build_inputs(self, arr: np.ndarray, names: list[str], rec: TileRecord, index: int):
        idx = {n: i for i, n in enumerate(names)}

        # ---- Optical -------------------------------------------------
        optical = arr[[idx[b] for b in OPTICAL_BANDS]]
        opt_valid = np.isfinite(optical).all(axis=0, keepdims=True).astype(np.float32)
        if "sentinel2_valid" in idx:
            flagged = np.nan_to_num(arr[idx["sentinel2_valid"]], nan=0.0)[None, ...]
            opt_valid = opt_valid * (flagged > 0.5).astype(np.float32)
        # Fill with the band mean so a missing pixel is "average", i.e. contributes
        # ~0 after standardization, rather than a spuriously dark or bright reading.
        optical = np.where(np.isfinite(optical), optical, OPTICAL_MEAN)
        optical = np.clip(optical, 0.0, OPTICAL_CLIP)

        # Cloud injection happens here, on PHYSICAL reflectance before
        # standardization, so a semi-transparent cloud is a real radiometric
        # mix rather than an arbitrary perturbation of a z-score.
        cloud_cov = 0.0
        if self.cloud_injector is not None:
            optical, opt_valid, cloud_cov = self.cloud_injector(optical, opt_valid, index)

        optical = (optical - OPTICAL_MEAN) / OPTICAL_STD
        optical = np.concatenate([optical * opt_valid, opt_valid], axis=0)

        # ---- SAR -----------------------------------------------------
        sar_raw = arr[[idx[b] for b in SAR_BANDS]]
        sar_valid = np.isfinite(sar_raw).all(axis=0, keepdims=True).astype(np.float32)
        if "sentinel1_valid" in idx:
            flagged = np.nan_to_num(arr[idx["sentinel1_valid"]], nan=0.0)[None, ...]
            sar_valid = sar_valid * (flagged > 0.5).astype(np.float32)
        sar_db = np.where(np.isfinite(sar_raw), sar_raw, SAR_MEAN)
        sar_db = np.clip(sar_db, SAR_CLIP_LO, SAR_CLIP_HI)

        sar_z = (sar_db - SAR_MEAN) / SAR_STD
        # Boxcar in dB as a cheap multilook. Kept alongside the raw bands so the
        # encoder can weigh radiometric stability against lost resolution
        # itself, rather than having that trade made for it here.
        sar_dspk = uniform_filter(
            sar_z, size=(1, SAR_DESPECKLE_WINDOW, SAR_DESPECKLE_WINDOW), mode="nearest",
        )
        ratio = ((sar_db[0:1] - sar_db[1:2]) - SAR_RATIO_MEAN) / SAR_RATIO_STD
        orbit = np.full_like(sar_valid, ORBIT_CODES.get(rec.orbit, 0.0))

        sar = np.concatenate(
            [sar_z * sar_valid, sar_dspk * sar_valid, ratio * sar_valid, sar_valid, orbit],
            axis=0,
        )

        # ---- DEM -----------------------------------------------------
        dem_raw = arr[idx["DEM"]]
        dem_valid = np.isfinite(dem_raw)
        # Local relief (elevation minus this tile's mean) is scale-free and
        # transfers across geographies far better than absolute elevation,
        # which is dominated by which continent the tile sits on.
        dem_mean_local = float(dem_raw[dem_valid].mean()) if dem_valid.any() else DEM_MEAN
        dem_filled = np.where(dem_valid, dem_raw, dem_mean_local)
        dem_abs = (dem_filled - DEM_MEAN) / DEM_STD
        dem_relief = (dem_filled - dem_mean_local) / DEM_RELIEF_STD
        # DEM voids reach 42% of a tile in this archive. Without a validity
        # channel a filled void is indistinguishable from genuinely flat ground.
        dem_v = dem_valid.astype(np.float32)
        dem = np.stack([dem_abs * dem_v, dem_relief * dem_v, dem_v], axis=0).astype(np.float32)

        return (
            optical.astype(np.float32),
            sar.astype(np.float32),
            dem,
            float(cloud_cov),
        )

    # -- modality presence ----------------------------------------------

    def _drop_modalities(self, optical, sar, dem, index: int) -> np.ndarray:
        """Zero whole modalities at the INPUT, and report which survive.

        Dropping at the input rather than at the encoded features is the whole
        point: a synthetically dropped modality then becomes byte-identical to
        a genuinely unacquired one (all-zero bands, validity 0), so the two
        share one representation. The previous feature-level dropout produced
        an exact zero that no real tile ever produces, training the model for a
        condition that never occurs at test time while leaving the 304 real
        optical-less tiles out of distribution.

        Draws are INDEPENDENT per modality, not one-of-three, so the
        two-modality-missing case the ablation grid evaluates is finally in
        distribution. Optical is dropped hardest -- it is the modality whose
        absence the project exists to survive.
        """
        presence = np.ones(3, dtype=np.float32)
        # A tile whose bands were never acquired is already all-zero; report it
        # as absent so the fusion is told, instead of having to infer it.
        if not np.any(optical[-1]):
            presence[0] = 0.0
        if not np.any(sar[-2]):          # validity channel, orbit is last
            presence[1] = 0.0
        if not np.any(dem[-1]):
            presence[2] = 0.0

        if self.drop_probs is not None:
            rng = np.random.default_rng() if not self.deterministic_drop else \
                np.random.default_rng(7_919 * index + 13)
            draw = rng.random(3) < np.asarray(self.drop_probs, dtype=np.float64)
            candidate = presence * (~draw).astype(np.float32)
            # Never leave a sample with neither imaging modality: DEM alone
            # carries no land cover (measured DEM-only mIoU is 0.035), so such
            # a sample is noise in the loss, not a useful hard case. The check
            # is on the RESULT, not on the draw -- dropping SAR on a tile whose
            # optical was never acquired strands it just as effectively as
            # drawing both.
            if candidate[0] == 0.0 and candidate[1] == 0.0:
                restorable = [i for i in (0, 1) if presence[i] > 0.0]
                if restorable:
                    candidate[int(rng.choice(restorable))] = 1.0
                # If both were genuinely absent there is nothing to restore.
                # Four train tiles are in that state; their masked losses handle it.
            presence = candidate

        if presence[0] == 0.0:
            optical[:] = 0.0
        if presence[1] == 0.0:
            sar[:] = 0.0
        if presence[2] == 0.0:
            dem[:] = 0.0
        return presence

    # -- labels ---------------------------------------------------------

    def _build_labels(self, rec: TileRecord):
        arr, names = _read_stack(rec.path / "labels.tif")
        idx = {n: i for i, n in enumerate(names)}

        lulc = arr[idx["lulc"]]
        finite = np.isfinite(lulc)
        if not rec.has_lulc:
            finite = np.zeros_like(finite)
        lulc_target = np.where(finite, np.round(lulc), IGNORE_INDEX)
        lulc_target = np.clip(lulc_target, 0, IGNORE_INDEX).astype(np.int64)
        # Guard against an out-of-range class index reaching CrossEntropyLoss.
        lulc_target[(lulc_target >= NUM_LULC_CLASSES) & (lulc_target != IGNORE_INDEX)] = IGNORE_INDEX

        road = arr[idx["road"]][None, ...]
        road_mask = np.isfinite(road).astype(np.float32)
        if not rec.has_road:
            road_mask[:] = 0.0
        road = np.nan_to_num(road, nan=0.0).astype(np.float32)

        building = arr[idx["building_presence"]][None, ...]
        building_mask = np.isfinite(building).astype(np.float32)
        if not rec.has_building:
            building_mask[:] = 0.0
        building = np.clip(np.nan_to_num(building, nan=0.0), 0.0, 1.0).astype(np.float32)

        return lulc_target, road, road_mask, building, building_mask

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rec = self.records[i]
        arr, names = _read_stack(rec.path / "inputs.tif")
        optical, sar, dem, cloud_cov = self._build_inputs(arr, names, rec, i)
        lulc, road, road_mask, building, building_mask = self._build_labels(rec)

        # Dropout AFTER cloud injection: an injected cloud that happens to cover
        # the whole tile should be reported as optical-absent, and this is where
        # that is detected from the validity channel.
        presence = self._drop_modalities(optical, sar, dem, i)

        stack = [optical, sar, dem, lulc, road, road_mask, building, building_mask]
        if self.augment:
            stack = self._augment(stack)

        keys = ("optical", "sar", "dem", "lulc", "road", "road_mask", "building", "building_mask")
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in zip(keys, stack)}
        out["presence"] = torch.from_numpy(presence)
        out["cloud_cov"] = torch.tensor(cloud_cov, dtype=torch.float32)
        return out

    @staticmethod
    def _augment(arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Dihedral (D4) augmentation: the only label-preserving *geometric* transform.

        Arbitrary photometric jitter is still deliberately avoided -- the bands
        are calibrated physical quantities (reflectance, dB) and the LULC labels
        were derived from exactly those values, so perturbing them breaks the
        label's meaning. Cloud injection (see `data.clouds`) is the one
        radiometric perturbation applied, and it is admissible for the opposite
        reason: it simulates a real physical process that occludes the sensor
        without changing what is on the ground, so the label stays true.
        """
        if np.random.rand() < 0.5:
            arrays = [np.flip(a, axis=-1) for a in arrays]
        if np.random.rand() < 0.5:
            arrays = [np.flip(a, axis=-2) for a in arrays]
        k = int(np.random.randint(0, 4))
        if k:
            arrays = [np.rot90(a, k, axes=(-2, -1)) for a in arrays]
        return arrays
