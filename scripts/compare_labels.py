"""Ground truth vs. predicted labels — side by side, nothing else.

Just the comparison: for each of LULC, road, and building, the true label next
to the model's prediction. No input imagery, no probability heatmaps, no alpha
maps — use scripts/visualize_prediction.py for the full diagnostic view.

Where a label was never observed (LULC pixel with no Sentinel-2, or a tile
with no OSM road mapping) it is drawn in dark grey, not black or white, so
"unlabelled" is never mistaken for "predicted empty."

Usage::

    python scripts/compare_labels.py --checkpoint checkpoints/best.pt --auto 4
    python scripts/compare_labels.py --checkpoint checkpoints/best.pt --tile-dir <path>
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

from data.dataset import IGNORE_INDEX, LULC_CLASS_NAMES, AETHERTileDataset, build_manifest  # noqa: E402
from inference import load_model  # noqa: E402

LULC_COLORS = [
    "#419BDF", "#397D49", "#88B053", "#7A87C6", "#E49635",
    "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1",
]
NODATA_COLOR = "#2b2b2b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--tile-dir", type=str, default=None)
    p.add_argument("--auto", type=int, default=0,
                   help="Auto-pick N tiles with road+building labels instead of --tile-dir.")
    p.add_argument("--road-threshold", type=float, default=0.60)
    p.add_argument("--building-threshold", type=float, default=0.71)
    p.add_argument("--output-dir", type=str, default="outputs/comparisons")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lulc_rgb(labels: np.ndarray) -> np.ndarray:
    cmap = ListedColormap(LULC_COLORS)
    out = np.zeros((*labels.shape, 3))
    valid = labels != IGNORE_INDEX
    safe = np.where(valid, labels, 0)
    out[valid] = cmap(np.clip(safe[valid], 0, len(LULC_COLORS) - 1))[:, :3]
    out[~valid] = matplotlib.colors.to_rgb(NODATA_COLOR)
    return out


def binary_rgb(mask: np.ndarray, observed: np.ndarray, colour: str) -> np.ndarray:
    out = np.ones((*mask.shape, 3))
    out[mask.astype(bool)] = matplotlib.colors.to_rgb(colour)
    out[~observed.astype(bool)] = matplotlib.colors.to_rgb(NODATA_COLOR)
    return out


def render(tile, model, args, device) -> Path:
    item = AETHERTileDataset([tile], augment=False)[0]
    inputs = {k: item[k].unsqueeze(0).to(device) for k in ("optical", "sar", "dem")}

    with torch.no_grad():
        out = model(inputs["optical"], inputs["sar"], inputs["dem"])

    lulc_pred = out["lulc"].float().argmax(1)[0].cpu().numpy()
    road_pred = (torch.sigmoid(out["road"].float())[0, 0].cpu().numpy() > args.road_threshold)
    bld_pred = (torch.sigmoid(out["building"].float())[0, 0].cpu().numpy() > args.building_threshold)

    lulc_true = item["lulc"].numpy()
    road_true = item["road"].numpy()[0] > 0.5
    road_obs = item["road_mask"].numpy()[0]
    bld_true = item["building"].numpy()[0] > 0.5
    bld_obs = item["building_mask"].numpy()[0]

    # Hide LULC predictions where there's no label, so the pair is comparable.
    lulc_pred_shown = np.where(lulc_true != IGNORE_INDEX, lulc_pred, IGNORE_INDEX)
    valid = lulc_true != IGNORE_INDEX
    lulc_acc = (lulc_pred[valid] == lulc_true[valid]).mean() if valid.any() else float("nan")

    road_iou = _iou(road_pred, road_true, road_obs)
    bld_iou = _iou(bld_pred, bld_true, bld_obs)

    fig, ax = plt.subplots(3, 2, figsize=(10, 14.5))
    fig.suptitle(tile.path.name, fontsize=13, y=0.995)

    rows = [
        ("LULC", lulc_rgb(lulc_true), lulc_rgb(lulc_pred_shown),
         f"acc {lulc_acc:.3f}" if not np.isnan(lulc_acc) else "no label"),
        ("Road", binary_rgb(road_true, road_obs, "#C4281B"),
         binary_rgb(road_pred, np.ones_like(road_obs), "#C4281B"),
         f"IoU {road_iou:.3f}" if road_obs.any() else "UNMAPPED (no OSM label)"),
        ("Building", binary_rgb(bld_true, bld_obs, "#E49635"),
         binary_rgb(bld_pred, np.ones_like(bld_obs), "#E49635"),
         f"IoU {bld_iou:.3f}" if bld_obs.any() else "no label"),
    ]
    for r, (name, truth_img, pred_img, metric) in enumerate(rows):
        ax[r, 0].imshow(truth_img)
        ax[r, 0].set_title(f"{name} — ground truth", fontsize=11)
        ax[r, 1].imshow(pred_img)
        ax[r, 1].set_title(f"{name} — prediction  ({metric})", fontsize=11)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    handles = [Patch(facecolor=c, label=n) for c, n in zip(LULC_COLORS, LULC_CLASS_NAMES)]
    handles.append(Patch(facecolor=NODATA_COLOR, label="no label"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{tile.path.name}.png"
    fig.savefig(dest, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {dest}   (LULC {lulc_acc:.3f}, road IoU {road_iou:.3f}, building IoU {bld_iou:.3f})")
    return dest


def _iou(pred: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> float:
    keep = observed.astype(bool)
    p, t = pred & keep, truth & keep
    union = (p | t).sum()
    return float((p & t).sum() / union) if union > 0 else 0.0


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
        n = args.auto or 4
        interesting = [r for r in records if r.has_road and r.has_optical and r.has_lulc]
        random.seed(args.seed)
        chosen = random.sample(interesting, min(n, len(interesting)))

    print(f"Rendering {len(chosen)} comparison(s):")
    for rec in chosen:
        render(rec, model, args, device)


if __name__ == "__main__":
    main()
