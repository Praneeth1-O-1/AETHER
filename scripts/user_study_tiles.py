"""Render the six user-study tiles (3 clear, 3 cloud-covered) from the held-out test split.

Selection is a fixed rule, not hand-picking: leak-free test tiles sorted by name,
first three standard ("g_") tiles and first three cloud-condition ("cloudy" /
"mid_cloudy") tiles whose labelled area exceeds 80%. For each tile writes the
true-colour image, the predicted land-cover map with legend, and the reliance
map (R = optical, G = radar, B = elevation weight).

Usage::

    python scripts/user_study_tiles.py --checkpoint <ckpt> --dataset-root <root>
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

from data.dataset import (IGNORE_INDEX, LULC_CLASS_NAMES, AETHERTileDataset,  # noqa: E402
                          build_manifest, leak_free, location_split)
from inference import get_device, load_model, to_device  # noqa: E402

DW_COLORS = ["#419BDF", "#397D49", "#88B053", "#7A87C6", "#E49635",
             "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1"]


def true_colour(tile_dir: Path) -> np.ndarray:
    with rasterio.open(tile_dir / "inputs.tif") as src:
        names = list(src.descriptions)
        rgb = np.stack([src.read(names.index(b) + 1) for b in
                        ("sentinel2_B4", "sentinel2_B3", "sentinel2_B2")], -1).astype(np.float32)
    rgb = np.nan_to_num(rgb)
    return np.clip(rgb / max(np.percentile(rgb, 98), 1e-6), 0, 1)


def save_image(img, path: Path, cmap=None, legend=False):
    fig, ax = plt.subplots(figsize=(4.2, 4.6 if legend else 4.2), dpi=150)
    if cmap is None:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(img, cmap=cmap, vmin=-0.5, vmax=8.5, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DW_COLORS]
        fig.legend(handles, [n.replace("_", " ") for n in LULC_CLASS_NAMES], loc="lower center",
                   ncol=5, fontsize=7, frameon=False, handlelength=1)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--out", default="outputs/user_study")
    args = ap.parse_args()

    device = get_device("auto")
    model = load_model(args.config, args.checkpoint, device).eval()
    train, val, test = location_split(build_manifest(Path(args.dataset_root)))
    test = sorted(leak_free(test, train, val), key=lambda r: r.path.name)

    def condition(r):
        aoi = r.path.parent.name
        return "cloudy" if aoi.endswith(("_cloudy", "_mid_cloudy")) else ("clear" if aoi.startswith("g_") else None)

    picked = {"clear": [], "cloudy": []}
    ds = AETHERTileDataset(test, augment=False)
    for rec, raw in zip(test, DataLoader(ds, batch_size=1, shuffle=False)):
        cond = condition(rec)
        if cond is None or len(picked[cond]) >= 3:
            continue
        if (raw["lulc"] != IGNORE_INDEX).float().mean().item() <= 0.8:
            continue
        picked[cond].append((rec, raw))
        if all(len(v) >= 3 for v in picked.values()):
            break

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(DW_COLORS)
    index = []
    with torch.no_grad():
        for cond, items in picked.items():
            for i, (rec, raw) in enumerate(items, 1):
                batch = to_device(raw, device)
                res = model(batch["optical"], batch["sar"], batch["dem"], presence=batch["presence"])
                pred = res["lulc"].float().argmax(1)[0].cpu().numpy()
                alpha = res["alpha_maps"].float()[0].cpu().numpy().transpose(1, 2, 0)
                alpha_up = np.kron(alpha, np.ones((16, 16, 1)))  # 16x16 -> 256x256, blocks kept visible
                tag = f"{cond}_{i}"
                save_image(true_colour(rec.path), out / f"{tag}_image.png")
                save_image(pred, out / f"{tag}_map.png", cmap=cmap, legend=True)
                save_image(np.clip(alpha_up, 0, 1), out / f"{tag}_reliance.png")
                gt = raw["lulc"][0].numpy(); valid = gt != IGNORE_INDEX
                index.append({"tag": tag, "tile": rec.path.name,
                              "pixel_acc_for_team_only": round(float((pred[valid] == gt[valid]).mean()), 4),
                              "mean_alpha_O_S_D": [round(float(a), 3) for a in alpha.mean(axis=(0, 1))]})
                print(index[-1], flush=True)
    (out / "tiles.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index)} tiles to {out}")


if __name__ == "__main__":
    main()
