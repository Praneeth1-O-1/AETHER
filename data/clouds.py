"""AETHER — Cloud injection, so SAR has a reason to exist.

The `strict` archive is 95.5% cloud-free (89.4% of tiles have optical
``valid_fraction`` of exactly 1.0). A model trained on it never needs SAR, and
gradient descent duly learns not to use it: ablating SAR costs 0.0026 mIoU.
That is the correct optimum for the training distribution -- so the training
distribution is what has to change.

The trick that makes this free:

    The LULC label was derived from the *clear* anchor granule. It stays
    correct underneath an injected cloud.

So every clear tile can be turned into a supervised ``(occluded optical,
correct label)`` pair without acquiring anything. That pair is the only
gradient signal that can teach the fusion to route around missing optical.

Two cloud types, because they defeat different mechanisms:

- **Opaque**: reflectance is destroyed and ``sentinel2_valid`` goes to 0. The
  model is *told* the pixel is gone. This trains graceful degradation.
- **Semi-transparent**: reflectance is blended toward a bright flat spectrum
  but validity stays 1. The model is *not* told, and must infer from the data
  that optical has become unreliable. This is the harder and more realistic
  case, and the one a validity channel alone cannot express.

Masks come from two sources. Real ``sentinel2_valid`` rasters harvested from
the archive's own ~277 genuinely cloudy tiles carry true cloud morphology --
wispy edges, correlated blobs, holes -- which procedural noise reproduces
badly. A fractal-noise generator supplements coverage fractions the harvested
bank is thin on.

**Split hygiene**: masks are harvested from *train* locations only, and the
evaluation injector uses a disjoint deterministic generator. Cloud shape must
not leak across the split any more than imagery does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# Typical top-of-atmosphere reflectance of an optically thick cloud, in the
# same physical units as OPTICAL_MEAN (i.e. reflectance, not scaled DN).
# Clouds are close to spectrally flat across the S2 bands used here.
CLOUD_REFLECTANCE = 0.85

# Below this, a harvested mask is too sparse to be a useful training signal;
# above it, the tile is effectively blank and teaches nothing beyond "optical
# is gone", which the presence flag already says.
MIN_HARVEST_COVERAGE = 0.03
MAX_HARVEST_COVERAGE = 0.98


# =========================================================================
# Fractal noise masks
# =========================================================================


def _value_noise(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """One octave of smooth value noise on an h x w grid.

    Bilinear upsampling of a small random lattice. Cheap, dependency-free, and
    good enough as an octave primitive -- the fractal sum below is what gives
    the result cloud-like structure at multiple scales.
    """
    cells = max(2, cells)
    lattice = rng.random((cells + 1, cells + 1)).astype(np.float32)

    ys = np.linspace(0, cells, h, dtype=np.float32)
    xs = np.linspace(0, cells, w, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32).clip(0, cells - 1)
    x0 = np.floor(xs).astype(np.int32).clip(0, cells - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]

    # Smoothstep, so octaves blend without visible lattice creases.
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)

    v00 = lattice[np.ix_(y0, x0)]
    v01 = lattice[np.ix_(y0, x0 + 1)]
    v10 = lattice[np.ix_(y0 + 1, x0)]
    v11 = lattice[np.ix_(y0 + 1, x0 + 1)]

    top = v00 * (1 - fx) + v01 * fx
    bot = v10 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bot * fy


def fractal_cloud_mask(
    h: int, w: int, coverage: float, rng: np.random.Generator, octaves: int = 4
) -> np.ndarray:
    """A soft cloud mask in [0, 1] covering approximately *coverage* of the tile.

    Coverage is hit by thresholding at the appropriate quantile rather than at
    a fixed level, so the requested fraction is achieved regardless of how the
    particular noise draw happened to be distributed.
    """
    if coverage <= 0:
        return np.zeros((h, w), dtype=np.float32)
    if coverage >= 1:
        return np.ones((h, w), dtype=np.float32)

    field = np.zeros((h, w), dtype=np.float32)
    amplitude, total = 1.0, 0.0
    for o in range(octaves):
        field += amplitude * _value_noise(h, w, cells=2 * (2 ** o), rng=rng)
        total += amplitude
        amplitude *= 0.5
    field /= max(total, 1e-6)

    # Threshold at the quantile that yields the requested coverage, then keep a
    # soft ramp around it so cloud edges are feathered rather than binary.
    #
    # The ramp is CENTERED on the cut (+0.5), not started at it. Starting at the
    # cut puts the whole feathered band below 0.5, so `mask > 0.5` selects only
    # pixels well above the quantile and the realised coverage undershoots the
    # request by ~7 points at mid-range. That would mislabel the x-axis of the
    # cloud-degradation curve, which is the one artifact that has to be exact.
    cut = float(np.quantile(field, 1.0 - coverage))
    ramp = max(float(field.std()) * 0.7, 1e-3)
    return np.clip((field - cut) / ramp + 0.5, 0.0, 1.0).astype(np.float32)


# =========================================================================
# Harvested real cloud masks
# =========================================================================


class CloudMaskBank:
    """Real cloud shapes taken from the archive's own partially-clouded tiles.

    Stored as packed bits: 512 masks at 256x256 is 4 MB packed, so the whole
    bank sits in every DataLoader worker without meaningful cost.
    """

    def __init__(self, masks: np.ndarray) -> None:
        self.masks = masks  # (N, H, W) uint8 in {0, 1}; 1 = clouded
        self.coverage = masks.reshape(len(masks), -1).mean(axis=1) if len(masks) else np.zeros(0)

    def __len__(self) -> int:
        return len(self.masks)

    # -- construction ---------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "CloudMaskBank":
        with np.load(path) as z:
            packed, shape = z["packed"], tuple(z["shape"])
        masks = np.unpackbits(packed, axis=-1)[..., : shape[2]].reshape(shape)
        return cls(masks.astype(np.uint8))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, packed=np.packbits(self.masks, axis=-1),
            shape=np.array(self.masks.shape),
        )

    @classmethod
    def build(
        cls,
        records,
        cache: Path,
        max_masks: int = 512,
        refresh: bool = False,
    ) -> "CloudMaskBank":
        """Harvest ``sentinel2_valid`` inversions from partially-clouded tiles.

        *records* must be TRAIN records only -- a mask harvested from a val or
        test location and replayed onto a training tile leaks that location's
        cloud structure across the split.
        """
        cache = Path(cache)
        if cache.exists() and not refresh:
            bank = cls.load(cache)
            logger.info(f"Cloud mask bank: {len(bank)} real masks from {cache}")
            return bank

        harvested: list[np.ndarray] = []
        for rec in records:
            if len(harvested) >= max_masks:
                break
            meta_path = rec.path / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            s2 = (meta.get("acquisition") or {}).get("sentinel2") or {}
            vf = s2.get("valid_fraction")
            # Only tiles that are *partially* clouded carry usable morphology.
            if vf is None or not (1 - MAX_HARVEST_COVERAGE <= vf <= 1 - MIN_HARVEST_COVERAGE):
                continue
            mask = _read_valid_band(rec.path / "inputs.tif")
            if mask is not None:
                harvested.append(mask)

        masks = (np.stack(harvested).astype(np.uint8) if harvested
                 else np.zeros((0, 256, 256), dtype=np.uint8))
        bank = cls(masks)
        bank.save(cache)
        logger.info(
            f"Cloud mask bank: harvested {len(bank)} real cloud masks "
            f"from {len(records)} train tiles -> {cache}"
        )
        return bank

    # -- sampling -------------------------------------------------------

    def sample(self, h: int, w: int, coverage: float, rng: np.random.Generator) -> np.ndarray | None:
        """Draw the harvested mask closest in coverage to *coverage*.

        Returns ``None`` when the bank is empty or has nothing within reach, so
        the caller can fall back to fractal noise.
        """
        if not len(self.masks):
            return None
        # Sample among the 16 nearest in coverage rather than the single
        # nearest, so a bank with few masks still yields variety.
        order = np.argsort(np.abs(self.coverage - coverage))
        pool = order[: min(16, len(order))]
        mask = self.masks[int(rng.choice(pool))].astype(np.float32)

        if mask.shape != (h, w):
            return None
        # D4 jitter, so one harvested shape does not become a memorised pattern.
        if rng.random() < 0.5:
            mask = np.flip(mask, axis=-1)
        if rng.random() < 0.5:
            mask = np.flip(mask, axis=-2)
        k = int(rng.integers(0, 4))
        if k:
            mask = np.rot90(mask, k)
        return np.ascontiguousarray(mask, dtype=np.float32)


def _read_valid_band(inputs_tif: Path) -> np.ndarray | None:
    """Return the inverted ``sentinel2_valid`` band, i.e. 1 where clouded."""
    try:
        with rasterio.open(inputs_tif) as src:
            names = list(src.descriptions)
            if "sentinel2_valid" not in names:
                return None
            band = src.read(names.index("sentinel2_valid") + 1).astype(np.float32)
    except (OSError, rasterio.RasterioIOError):
        return None
    band = np.nan_to_num(band, nan=0.0)
    return (band <= 0.5).astype(np.uint8)


# =========================================================================
# Injector
# =========================================================================


@dataclass
class CloudConfig:
    """Cloud-injection policy for one dataset instance.

    Attributes
    ----------
    prob : float
        Fraction of samples that receive any cloud at all. The literature
        recommends 0.4-0.6; below that the clear-sky regime still dominates
        and the fusion has little reason to adapt.
    max_coverage : float
        Upper bound of the coverage draw. Ramped from a low value to 1.0
        across training by :func:`curriculum_coverage`, so the model meets
        light haze before total occlusion.
    opaque_frac : float
        Share of clouded samples that get opaque cloud (validity zeroed)
        rather than semi-transparent haze (validity retained).
    full_occlusion_prob : float
        Share of clouded samples forced to coverage 1.0. Without an explicit
        spike here the fully-occluded case is only ever extrapolated to, and
        that is precisely the case the project exists to serve.
    deterministic : bool
        Seed per tile index instead of randomly. Evaluation only -- makes the
        cloud-degradation curve reproducible across runs and checkpoints.
    fixed_coverage : float | None
        Force every sample to this coverage. Evaluation only.
    """

    prob: float = 0.5
    max_coverage: float = 1.0
    opaque_frac: float = 0.6
    full_occlusion_prob: float = 0.15
    deterministic: bool = False
    fixed_coverage: float | None = None
    seed: int = 0


def curriculum_coverage(epoch: int, total_epochs: int,
                        start: float = 0.3, end: float = 1.0,
                        ramp_frac: float = 0.5) -> float:
    """Max cloud coverage for *epoch*, ramped linearly then held.

    Starting at full occlusion destabilises early training -- the LULC head has
    not yet learned anything from the easy clear-sky case, so a fully-occluded
    batch is pure noise. Ramping reaches the same endpoint without that.
    """
    if total_epochs <= 1:
        return end
    t = min(1.0, (epoch - 1) / max(1.0, ramp_frac * (total_epochs - 1)))
    return start + (end - start) * t


class CloudInjector:
    """Applies :class:`CloudConfig` to one tile's optical bands.

    Operates on *physical* reflectance before standardisation, so the
    semi-transparent blend is a real radiometric mix rather than an arbitrary
    perturbation of a z-score.
    """

    def __init__(self, config: CloudConfig, bank: CloudMaskBank | None = None) -> None:
        self.config = config
        self.bank = bank

    def __call__(
        self, optical: np.ndarray, opt_valid: np.ndarray, index: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(optical, opt_valid, coverage_applied)``.

        Parameters
        ----------
        optical : np.ndarray
            ``(C, H, W)`` physical reflectance, already NaN-filled and clipped.
        opt_valid : np.ndarray
            ``(1, H, W)`` validity in {0, 1}.
        index : int
            Tile index, used only when ``config.deterministic``.
        """
        cfg = self.config
        rng = (np.random.default_rng(cfg.seed * 1_000_003 + index)
               if cfg.deterministic else np.random.default_rng())

        if cfg.fixed_coverage is not None:
            coverage = float(cfg.fixed_coverage)
            if coverage <= 0:
                return optical, opt_valid, 0.0
        else:
            if rng.random() >= cfg.prob:
                return optical, opt_valid, 0.0
            coverage = (1.0 if rng.random() < cfg.full_occlusion_prob
                        else float(rng.uniform(0.05, max(0.05, cfg.max_coverage))))

        _, h, w = optical.shape
        mask = None
        if self.bank is not None and rng.random() < 0.5:
            mask = self.bank.sample(h, w, coverage, rng)
        if mask is None:
            mask = fractal_cloud_mask(h, w, coverage, rng)

        opaque = rng.random() < cfg.opaque_frac
        if opaque:
            # Hard cloud: reflectance destroyed, and the model is TOLD via the
            # validity channel. Byte-identical to a genuinely unacquired pixel.
            hard = (mask > 0.5).astype(np.float32)[None, ...]
            optical = optical * (1.0 - hard) + CLOUD_REFLECTANCE * hard
            opt_valid = opt_valid * (1.0 - hard)
        else:
            # Haze: a real radiometric mix, validity untouched. The model is
            # NOT told, and has to infer unreliability from the data itself.
            alpha = (mask * float(rng.uniform(0.35, 0.85)))[None, ...]
            optical = optical * (1.0 - alpha) + CLOUD_REFLECTANCE * alpha

        return optical, opt_valid, float(mask.mean())
