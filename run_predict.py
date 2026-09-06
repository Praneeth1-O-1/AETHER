#!/home/kirethik/python-envs/general/bin/python
"""Predict and render results you can look at -- no arguments needed.

    python run_predict.py

Reads everything from settings.py. Writes a 12-panel PNG per tile (inputs,
prediction vs ground truth for all three heads, raw road probability, fusion
alpha) to settings.VISUALIZATION_DIR, plus raw .npy arrays if SAVE_ARRAYS.

To change what it predicts, edit settings.py:
    PREDICT_TILE      -- one specific tile, or None to auto-pick
    N_PREDICT_TILES   -- how many to auto-pick
    CHECKPOINT        -- which model
    ROAD_THRESHOLD / BUILDING_THRESHOLD
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import settings as S
from data.dataset import AETHERTileDataset, build_manifest
from inference import load_model
from scripts.visualize_prediction import render


class _Args:
    """Mirrors visualize_prediction's argparse namespace, from settings.py."""
    checkpoint = str(S.CHECKPOINT)
    config = str(S.MODEL_CONFIG)
    dataset_root = str(S.DATASET_ROOT)
    road_threshold = S.ROAD_THRESHOLD
    building_threshold = S.BUILDING_THRESHOLD
    output_dir = str(S.VISUALIZATION_DIR)


def save_arrays(tile, model, device, out_dir: Path) -> None:
    """Raw per-pixel outputs, for anyone wanting the numbers not the picture."""
    item = AETHERTileDataset([tile], augment=False)[0]
    inputs = {k: item[k].unsqueeze(0).to(device) for k in ("optical", "sar", "dem")}
    with torch.no_grad():
        out = model(inputs["optical"], inputs["sar"], inputs["dem"])

    dest = out_dir / tile.path.name
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "lulc_pred.npy", out["lulc"].float().argmax(1)[0].cpu().numpy().astype(np.uint8))
    np.save(dest / "road_prob.npy", torch.sigmoid(out["road"].float())[0, 0].cpu().numpy())
    np.save(dest / "building_prob.npy", torch.sigmoid(out["building"].float())[0, 0].cpu().numpy())
    np.save(dest / "alpha_maps.npy", out["alpha_maps"].float()[0].cpu().numpy())


def main() -> None:
    import random

    device = torch.device("cuda" if (S.DEVICE == "auto" and torch.cuda.is_available())
                          else ("cpu" if S.DEVICE == "auto" else S.DEVICE))
    print(f"Checkpoint : {S.CHECKPOINT}")
    print(f"Device     : {device}")
    print(f"Thresholds : road {S.ROAD_THRESHOLD}  building {S.BUILDING_THRESHOLD}\n")

    if not Path(S.CHECKPOINT).exists():
        raise SystemExit(f"Checkpoint not found: {S.CHECKPOINT}\n"
                         f"Edit CHECKPOINT in settings.py, or train one first "
                         f"with: python run_train.py")

    model = load_model(str(S.MODEL_CONFIG), str(S.CHECKPOINT), device)
    records = build_manifest(Path(S.DATASET_ROOT))

    if S.PREDICT_TILE:
        wanted = Path(S.PREDICT_TILE).resolve()
        chosen = [r for r in records if r.path.resolve() == wanted]
        if not chosen:
            raise SystemExit(f"Tile not in manifest: {wanted}")
    else:
        # Tiles with road + building labels are the informative ones to view.
        pool = [r for r in records if r.has_road and r.has_optical and r.has_lulc]
        random.seed(S.PREDICT_SEED)
        chosen = random.sample(pool, min(S.N_PREDICT_TILES, len(pool)))

    print(f"Rendering {len(chosen)} tile(s) -> {S.VISUALIZATION_DIR}")
    for rec in chosen:
        render(rec, model, _Args, device)
        if S.SAVE_ARRAYS:
            save_arrays(rec, model, device, Path(S.OUTPUT_DIR) / "arrays")

    print(f"\nPNGs   : {S.VISUALIZATION_DIR}")
    if S.SAVE_ARRAYS:
        print(f"Arrays : {Path(S.OUTPUT_DIR) / 'arrays'}")
    print(f"\nOpen them with:  xdg-open {S.VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
