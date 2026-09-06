"""Render a tile's predictions next to its ground truth as a viewable PNG.

Turns the raw ``.npy`` outputs into something you can actually look at: the
true-colour input, the LULC prediction against its label, road and building
probability against theirs, and the fusion alpha maps.

Two details that matter for reading the picture honestly:

- **Road and building are thresholded at the val-tuned cutoffs** (0.60 / 0.71),
  not 0.5, because both heads are trained with a damped ``pos_weight`` plus a
  Dice term and are deliberately recall-biased. The raw probability panel is
  shown alongside so you can see what the threshold is doing.
- **Where a label was never observed it is drawn in grey, not as zero.** LULC
  is NaN wherever Sentinel-2 was missing, and ~26% of tiles have no OSM road
  mapping. Painting those black would make the model look wrong where there is
  simply no ground truth.

Usage::

    python scripts/visualize_prediction.py --checkpoint checkpoints/best.pt \\
        --tile-dir data/strict/<aoi>/<tile>

    # or let it pick interesting tiles for you
    python scripts/visualize_prediction.py --checkpoint checkpoints/best.pt --auto 4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    OPTICAL_BANDS,
    AETHERTileDataset,
    build_manifest,
)
from inference import load_model  # noqa: E402

# Dynamic World's own palette, so the maps read the way the labels are published.
LULC_COLORS = [
    "#419BDF",  # water
    "#397D49",  # trees
    "#88B053",  # grass
    "#7A87C6",  # flooded_veg
    "#E49635",  # crops
    "#DFC35A",  # shrub
    "#C4281B",  # built
    "#A59B8F",  # bare
    "#B39FE1",  # snow_ice
]
NODATA_COLOR = "#2b2b2b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--tile-dir", type=str, default=None, help="One specific tile.")
    p.add_argument("--auto", type=int, default=0,
                   help="Instead of --tile-dir, auto-pick N tiles that have road "
                        "and building labels (the interesting ones to look at).")
    p.add_argument("--road-threshold", type=float, default=0.60)
    p.add_argument("--building-threshold", type=float, default=0.71)
    p.add_argument("--output-dir", type=str, default="outputs/visualizations")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def true_colour(optical: np.ndarray) -> np.ndarray:
    """Rebuild a viewable RGB image from the standardized optical tensor.

    The dataset hands back z-scored reflectance, so undo that with the same
    constants, then apply a percentile stretch -- raw reflectance is very dark
    and would otherwise render as a near-black square.
    """
    from data.dataset import OPTICAL_MEAN, OPTICAL_STD

    idx = [OPTICAL_BANDS.index(b) for b in ("sentinel2_B4", "sentinel2_B3", "sentinel2_B2")]
    rgb = optical[idx] * OPTICAL_STD[idx] + OPTICAL_MEAN[idx]
    rgb = np.transpose(rgb, (1, 2, 0))
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def lulc_rgb(labels: np.ndarray) -> np.ndarray:
    """Colour a class-index map, drawing unobserved pixels as dark grey."""
    cmap = ListedColormap(LULC_COLORS)
    out = np.zeros((*labels.shape, 3))
    valid = labels != IGNORE_INDEX
    safe = np.where(valid, labels, 0)
    out[valid] = cmap(np.clip(safe[valid], 0, len(LULC_COLORS) - 1))[:, :3]
    out[~valid] = matplotlib.colors.to_rgb(NODATA_COLOR)
    return out


def binary_rgb(mask: np.ndarray, observed: np.ndarray, colour: str) -> np.ndarray:
    """White background, coloured positives, dark grey where unlabelled."""
    out = np.ones((*mask.shape, 3))
    out[mask.astype(bool)] = matplotlib.colors.to_rgb(colour)
    out[~observed.astype(bool)] = matplotlib.colors.to_rgb(NODATA_COLOR)
    return out


def render(tile, model, args, device) -> Path:
    ds = AETHERTileDataset([tile], augment=False)
    item = ds[0]
    inputs = {k: item[k].unsqueeze(0).to(device) for k in ("optical", "sar", "dem")}

    with torch.no_grad():
        out = model(inputs["optical"], inputs["sar"], inputs["dem"])

    lulc_pred = out["lulc"].float().argmax(1)[0].cpu().numpy()
    road_prob = torch.sigmoid(out["road"].float())[0, 0].cpu().numpy()
    bld_prob = torch.sigmoid(out["building"].float())[0, 0].cpu().numpy()
    alpha = out["alpha_maps"].float()[0].cpu().numpy()

    lulc_true = item["lulc"].numpy()
    road_true = item["road"].numpy()[0]
    road_obs = item["road_mask"].numpy()[0]
    bld_true = item["building"].numpy()[0]
    bld_obs = item["building_mask"].numpy()[0]

    # Hide predictions where there is no label, so the two panels are comparable.
    lulc_pred_shown = np.where(lulc_true != IGNORE_INDEX, lulc_pred, IGNORE_INDEX)

    fig, ax = plt.subplots(3, 4, figsize=(19, 14))
    fig.suptitle(f"{tile.path.name}    |    model: {Path(args.checkpoint).name}",
                 fontsize=13, y=0.985)

    ax[0, 0].imshow(true_colour(item["optical"].numpy())); ax[0, 0].set_title("Sentinel-2 true colour")
    sar = item["sar"].numpy()[0]
    ax[0, 1].imshow(sar, cmap="gray"); ax[0, 1].set_title("Sentinel-1 VV (standardized)")
    ax[0, 2].imshow(item["dem"].numpy()[1], cmap="terrain"); ax[0, 2].set_title("DEM local relief")
    ax[0, 3].imshow(lulc_rgb(lulc_true)); ax[0, 3].set_title("LULC — ground truth")

    ax[1, 0].imshow(lulc_rgb(lulc_pred_shown)); ax[1, 0].set_title("LULC — PREDICTION")
    agree = (lulc_pred == lulc_true) | (lulc_true == IGNORE_INDEX)
    ax[1, 1].imshow(np.where(lulc_true == IGNORE_INDEX, 0.5, agree.astype(float)),
                    cmap="RdYlGn", vmin=0, vmax=1)
    valid = lulc_true != IGNORE_INDEX
    acc = (lulc_pred[valid] == lulc_true[valid]).mean() if valid.any() else float("nan")
    ax[1, 1].set_title(f"LULC correct (green) — acc {acc:.3f}")

    ax[1, 2].imshow(binary_rgb(road_true > 0.5, road_obs, "#111111"))
    ax[1, 2].set_title("Road — ground truth" + ("" if road_obs.any() else "  (UNMAPPED)"))
    ax[1, 3].imshow(binary_rgb(road_prob > args.road_threshold, np.ones_like(road_obs), "#C4281B"))
    ax[1, 3].set_title(f"Road — PREDICTION (p > {args.road_threshold})")

    im = ax[2, 0].imshow(road_prob, cmap="magma", vmin=0, vmax=1)
    ax[2, 0].set_title("Road probability (raw)"); fig.colorbar(im, ax=ax[2, 0], fraction=0.046)

    ax[2, 1].imshow(binary_rgb(bld_true > 0.5, bld_obs, "#111111"))
    ax[2, 1].set_title("Building — ground truth")
    ax[2, 2].imshow(binary_rgb(bld_prob > args.building_threshold, np.ones_like(bld_obs), "#E49635"))
    ax[2, 2].set_title(f"Building — PREDICTION (p > {args.building_threshold})")

    ax[2, 3].imshow(np.transpose(alpha, (1, 2, 0)))
    ax[2, 3].set_title(f"Fusion alpha  R=optical G=SAR B=DEM\nmean {alpha.mean(axis=(1,2)).round(2).tolist()}")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    handles = [Patch(facecolor=c, label=n) for c, n in zip(LULC_COLORS, LULC_CLASS_NAMES)]
    handles.append(Patch(facecolor=NODATA_COLOR, label="no label"))
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{tile.path.name}.png"
    fig.savefig(dest, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {dest}   (LULC acc {acc:.3f}, road px {int((road_prob > args.road_threshold).sum())}, "
          f"building px {int((bld_prob > args.building_threshold).sum())})")
    return dest


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    model = load_model(args.config, args.checkpoint, device)

    records = build_manifest(Path(args.dataset_root))
    if args.tile_dir:
        wanted = Path(args.tile_dir).resolve()
        chosen = [r for r in records if r.path.resolve() == wanted]
        if not chosen:
            raise SystemExit(f"Tile not found in manifest: {wanted}")
    else:
        n = args.auto or 3
        # Tiles that actually have road labels and full inputs are the ones
        # worth looking at -- an unmapped-road tile shows an empty truth panel.
        interesting = [r for r in records if r.has_road and r.has_optical and r.has_lulc]
        random.seed(args.seed)
        chosen = random.sample(interesting, min(n, len(interesting)))

    print(f"Rendering {len(chosen)} tile(s):")
    for rec in chosen:
        render(rec, model, args, device)


if __name__ == "__main__":
    main()
