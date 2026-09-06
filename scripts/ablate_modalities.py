"""Measure how much the trained model actually depends on each modality.

The learned alpha maps say what the fusion *weights*; they do not say what the
model *depends on*. A modality can carry a large alpha while contributing
little unique information (because another modality is redundant with it), or
carry a small alpha and still be load-bearing. The causal question -- "what
breaks if this modality disappears?" -- is only answerable by removing it.

Ablation is applied at the **encoded-feature** level, not the input level,
because that is exactly where `AETHERModel` applies modality dropout during
training. The model has therefore already seen zeroed feature maps for ~5% of
samples per modality, so a zeroed modality is in-distribution and the measured
drop reflects lost information rather than a distribution shift artifact.

Usage::

    python scripts/ablate_modalities.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    NUM_LULC_CLASSES,
    AETHERTileDataset,
    build_manifest,
    location_split,
)
from inference import load_model, to_device  # noqa: E402
from utils.metrics import BinaryIoU, ConfusionMatrix  # noqa: E402

# name -> (keep_optical, keep_sar, keep_dem)
ABLATIONS = {
    "all (baseline)": (1, 1, 1),
    "optical only": (1, 0, 0),
    "SAR only": (0, 1, 0),
    "DEM only": (0, 0, 1),
    "no optical (S+D)": (0, 1, 1),
    "no SAR (O+D)": (1, 0, 1),
    "no DEM (O+S)": (1, 1, 0),
    "nothing": (0, 0, 0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--road-threshold", type=float, default=0.60)
    p.add_argument("--building-threshold", type=float, default=0.71)
    p.add_argument("--output", type=str, default="outputs/modality_ablation.json")
    return p.parse_args()


@torch.no_grad()
def forward_ablated(model, batch, keep) -> dict[str, torch.Tensor]:
    """Run the real forward pass with the de-selected modalities marked absent.

    An earlier revision replicated ``AETHERModel.forward`` by hand and zeroed
    the encoder outputs. That did not measure what it claimed to, for two
    reasons, and every ablation number it produced should be discarded:

    - The fusion module added its modality embeddings *after* the zeroing, and
      then cross-attended. A zeroed optical branch became a constant query,
      whose attention output is a function of SAR -- so "optical only" still
      routed optical content through the SAR slot, and vice versa. Nothing was
      actually removed.
    - It called ``model.decoder(f_shared)`` with no skips, so on a
      skip-connected checkpoint it silently evaluated a different architecture
      than the one being ablated.

    Marking absence through the model's own ``presence`` argument fixes both:
    the modality is genuinely gone (verified: perturbing an absent modality's
    input changes the output by exactly zero) and the real decode path runs.
    """
    presence = batch["optical"].new_tensor(keep).view(1, 3).expand(batch["optical"].shape[0], 3)
    return model(batch["optical"], batch["sar"], batch["dem"], presence=presence)


@torch.no_grad()
def run_one(model, loader, device, keep, args) -> dict:
    conf = ConfusionMatrix(NUM_LULC_CLASSES, device=device)
    road = BinaryIoU(threshold=args.road_threshold, device=device)
    building = BinaryIoU(threshold=args.building_threshold, device=device)
    alpha_sum = torch.zeros(3, device=device)
    n = 0

    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = forward_ablated(model, batch, keep)
        conf.update(out["lulc"].float().argmax(1), batch["lulc"], batch["lulc"] != IGNORE_INDEX)
        road.update(out["road"], batch["road"], batch["road_mask"])
        building.update(out["building"], batch["building"], batch["building_mask"])
        alpha_sum += out["alpha_maps"].float().mean(dim=(0, 2, 3))
        n += 1

    per_class = conf.per_class_iou().tolist()
    return {
        "accuracy": conf.accuracy(),
        "miou": conf.miou(),
        "road_iou": road.iou(),
        "building_iou": building.iou(),
        "alpha": [round(a, 4) for a in (alpha_sum / max(n, 1)).tolist()],
        "per_class_iou": {name: (None if np.isnan(v) else round(v, 4))
                          for name, v in zip(LULC_CLASS_NAMES, per_class)},
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.config, args.checkpoint, device)

    records = build_manifest(Path(args.dataset_root))
    _, val_rec, test_rec = location_split(records)
    chosen = {"val": val_rec, "test": test_rec}[args.split]
    loader = DataLoader(
        AETHERTileDataset(chosen, augment=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"\nAblating on '{args.split}': {len(chosen)} tiles / "
          f"{len({r.location for r in chosen})} locations\n")

    results = {}
    for name, keep in ABLATIONS.items():
        results[name] = run_one(model, loader, device, keep, args)

    base = results["all (baseline)"]
    print(f"{'modality set':<20}{'acc':>8}{'mIoU':>8}{'d mIoU':>9}{'road':>8}{'bld':>8}   alpha[O,S,D]")
    print("-" * 84)
    for name, r in results.items():
        delta = r["miou"] - base["miou"]
        mark = "" if name == "all (baseline)" else f"{delta:+.4f}"
        print(f"{name:<20}{r['accuracy']:>8.4f}{r['miou']:>8.4f}{mark:>9}"
              f"{r['road_iou']:>8.4f}{r['building_iou']:>8.4f}   {r['alpha']}")

    print("\n--- unique contribution (mIoU lost when ONLY this modality is removed) ---")
    for mod, key in (("optical", "no optical (S+D)"), ("SAR", "no SAR (O+D)"), ("DEM", "no DEM (O+S)")):
        loss = base["miou"] - results[key]["miou"]
        print(f"  {mod:<8} {loss:+.4f} mIoU  ({loss / base['miou'] * 100:+.1f}% relative)")

    print("\n--- standalone sufficiency (mIoU using ONLY this modality) ---")
    for mod, key in (("optical", "optical only"), ("SAR", "SAR only"), ("DEM", "DEM only")):
        r = results[key]
        print(f"  {mod:<8} {r['miou']:.4f} mIoU  ({r['miou'] / base['miou'] * 100:.1f}% of full model)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
