"""Input/output comparison: one tile predicted by every modality combination.

For each tile, shows the inputs and the ground truth, then the LULC map the
model produces when given each subset of {optical, SAR, DEM}. Absent modalities
are removed through the model's own presence flags (the same mechanism as
scripts/ablate_modalities.py), so this is the trained model genuinely running
without them, not a retrained one.

Usage::

    python scripts/combo_figure.py --checkpoint checkpoints/kaggle_best_epoch41.pt \
        --roots datasets/cloudy_test datasets/downloaded_tile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import IGNORE_INDEX, LULC_CLASS_NAMES, AETHERTileDataset, build_manifest  # noqa: E402
from inference import get_device, load_model, to_device  # noqa: E402
from scripts.ablate_modalities import forward_ablated  # noqa: E402

COMBOS = [("O+S+D", (1, 1, 1)), ("O", (1, 0, 0)), ("S", (0, 1, 0)), ("D", (0, 0, 1)),
          ("O+S", (1, 1, 0)), ("O+D", (1, 0, 1)), ("S+D", (0, 1, 1))]
DW_COLORS = ["#419BDF", "#397D49", "#88B053", "#7A87C6", "#E49635",
             "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="paper/figures/fig_combos.png")
    p.add_argument("--json", default="results/combo_tiles.json")
    return p.parse_args()


def true_colour(tile_dir: Path) -> np.ndarray:
    with rasterio.open(tile_dir / "inputs.tif") as src:
        names = list(src.descriptions)
        rgb = np.stack([src.read(names.index(b) + 1) for b in
                        ("sentinel2_B4", "sentinel2_B3", "sentinel2_B2")], -1).astype(np.float32)
    rgb = np.nan_to_num(rgb)
    return np.clip(rgb / max(np.percentile(rgb, 98), 1e-6), 0, 1)


def main():
    args = parse_args()
    device = get_device(args.device)
    model = load_model(args.config, args.checkpoint, device).eval()
    cmap = ListedColormap(DW_COLORS)

    records = [r for root in args.roots for r in build_manifest(Path(root), refresh=True)]
    ds = AETHERTileDataset(records, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows, summary = [], []
    with torch.no_grad():
        for rec, raw in zip(records, loader):
            batch = to_device(raw, device)
            gt = batch["lulc"][0].cpu().numpy()
            valid = gt != IGNORE_INDEX
            preds, info = {}, {}
            for name, keep in COMBOS:
                out = forward_ablated(model, batch, keep)
                pred = out["lulc"].float().argmax(1)[0].cpu().numpy()
                acc = float((pred[valid] == gt[valid]).mean()) if valid.any() else float("nan")
                alpha = out["alpha_maps"].float().mean(dim=(0, 2, 3)).cpu().tolist()
                preds[name] = (pred, acc)
                info[name] = {"pixel_acc": round(acc, 4), "alpha": [round(a, 3) for a in alpha]}
            rows.append((rec, true_colour(rec.path), gt, preds))
            summary.append({"tile": rec.path.name, "combos": info})
            print(rec.path.name, {k: v["pixel_acc"] for k, v in info.items()})

    ncol = 2 + len(COMBOS)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.0 * ncol, 1.12 * len(rows) + 0.5), dpi=300)
    axes = np.atleast_2d(axes)
    for i, (rec, rgb, gt, preds) in enumerate(rows):
        gt_show = np.where(gt == IGNORE_INDEX, np.nan, gt).astype(float)
        panels = [("Sentinel-2 RGB", rgb, None), ("Ground truth", gt_show, None)]
        panels += [(n, preds[n][0], preds[n][1]) for n, _ in COMBOS]
        for j, (title, img, acc) in enumerate(panels):
            ax = axes[i, j]
            if img.ndim == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=-0.5, vmax=8.5, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(title, fontsize=6.5, family="serif")
            if acc is not None:
                ax.set_xlabel(f"{acc:.2f}", fontsize=6, family="serif", labelpad=1.5)
        axes[i, 0].set_ylabel(rec.path.name.split("_")[1].capitalize() +
                              (" (cloudy)" if "cloudy" in rec.path.name else " (clear)"),
                              fontsize=6.5, family="serif")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DW_COLORS]
    fig.legend(handles, [n.replace("_", " ") for n in LULC_CLASS_NAMES], loc="lower center",
               ncol=9, fontsize=5.5, frameon=False, handlelength=1, prop={"family": "serif", "size": 5.5})
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.3, w_pad=0.2, h_pad=0.6)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(summary, indent=2))
    print("wrote", args.output, args.json)


if __name__ == "__main__":
    main()
