"""Cloud-degradation curve — the headline evaluation for a cloud-robust model.

A single clear-sky accuracy figure cannot distinguish a model that degrades
gracefully as cloud rises from one that collapses the moment optical becomes
unavailable. Those are entirely different products, and only the second one is
what AETHER was measuring before.

This script reports LULC accuracy / mIoU / kappa at a sweep of optical
occlusion levels, and at each level also reports:

- **mean alpha**, so you can see whether the fusion actually re-weights toward
  SAR as optical degrades. If ``alpha_sar`` is flat across the sweep the fusion
  is not adaptive, however good the aggregate numbers look.
- **the SAR ablation delta**, i.e. how much worse the model gets when SAR is
  additionally removed at that occlusion level. This is the number that says
  whether SAR is load-bearing. In clear sky it was measured at 0.0026 mIoU;
  under occlusion it is the metric the project exists to move.

Occlusion is applied with the same deterministic injector the training loop
uses for validation, seeded by tile index, so the curve is reproducible across
runs and directly comparable between checkpoints.

Usage::

    python scripts/cloud_curve.py --checkpoint checkpoints_cloud/best.pt
    python scripts/cloud_curve.py --checkpoint checkpoints/best.pt --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from data.clouds import CloudConfig, CloudInjector, CloudMaskBank  # noqa: E402
from data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    NUM_LULC_CLASSES,
    AETHERTileDataset,
    build_manifest,
    location_split,
)
from inference import get_device, load_model, to_device  # noqa: E402
from utils.metrics import ConfusionMatrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Which modalities to keep, per ablation row. The model's presence argument
# makes these genuine removals -- verified by the fact that perturbing an
# absent modality's input changes the output by exactly zero.
ABLATIONS = {
    "all": (1.0, 1.0, 1.0),
    "no_sar": (1.0, 0.0, 1.0),
    "no_dem": (1.0, 1.0, 0.0),
    "sar_dem_only": (0.0, 1.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LULC accuracy as a function of cloud cover.")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="outputs/cloud_curve.json")
    return p.parse_args()


@torch.no_grad()
def run(model, loader, device, keep) -> dict:
    conf = ConfusionMatrix(NUM_LULC_CLASSES, device=device)
    alpha_sum = torch.zeros(3, device=device)
    n = 0
    for raw in loader:
        batch = to_device(raw, device)
        B = batch["optical"].shape[0]
        # Combine the tile's own presence (a genuinely unacquired modality) with
        # the ablation mask, so an ablation never resurrects missing data.
        presence = batch["presence"] * batch["optical"].new_tensor(keep).view(1, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch["optical"], batch["sar"], batch["dem"], presence=presence)
        conf.update(out["lulc"].float().argmax(1), batch["lulc"], batch["lulc"] != IGNORE_INDEX)
        alpha_sum += out["alpha_maps"].float().mean(dim=(0, 2, 3))
        n += 1
        del out
    per_class = conf.per_class_iou().tolist()
    return {
        "accuracy": round(conf.accuracy(), 4),
        "miou": round(conf.miou(), 4),
        "kappa": round(conf.kappa(), 4),
        "alpha_mean": [round(a, 4) for a in (alpha_sum / max(n, 1)).tolist()],
        "per_class_iou": {
            name: (None if v != v else round(v, 4))
            for name, v in zip(LULC_CLASS_NAMES, per_class)
        },
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model = load_model(args.config, args.checkpoint, device)

    records = build_manifest(Path(args.dataset_root))
    train_rec, val_rec, test_rec = location_split(records, args.val_frac, args.test_frac)
    chosen = {"val": val_rec, "test": test_rec}[args.split]
    logger.info(f"{args.split}: {len(chosen)} tiles / {len({r.location for r in chosen})} locations")

    # Same bank the training run used, so injected cloud looks the same at
    # evaluation time as it did during training.
    bank_path = Path(args.dataset_root).parent / "cloud_masks.npz"
    bank = CloudMaskBank.load(bank_path) if bank_path.exists() else None
    if bank is None:
        logger.warning(f"No cloud mask bank at {bank_path}; using fractal masks only.")

    results: dict[str, dict] = {}
    for level in args.levels:
        injector = None
        if level > 0:
            injector = CloudInjector(
                CloudConfig(prob=1.0, fixed_coverage=level, opaque_frac=1.0,
                            deterministic=True, seed=1234),
                bank,
            )
        loader = DataLoader(
            AETHERTileDataset(chosen, augment=False, cloud_injector=injector),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        row = {name: run(model, loader, device, keep) for name, keep in ABLATIONS.items()}
        # The load-bearing number: what removing SAR costs at THIS cloud level.
        row["sar_contribution_miou"] = round(
            row["all"]["miou"] - row["no_sar"]["miou"], 4)
        row["dem_contribution_miou"] = round(
            row["all"]["miou"] - row["no_dem"]["miou"], 4)
        results[f"{level:.2f}"] = row

        a = row["all"]
        logger.info(
            f"cloud {level:.0%}: acc {a['accuracy']:.4f} mIoU {a['miou']:.4f} "
            f"alpha[O,S,D] {a['alpha_mean']} | "
            f"SAR worth {row['sar_contribution_miou']:+.4f} mIoU | "
            f"DEM worth {row['dem_contribution_miou']:+.4f} | "
            f"SAR+DEM only: acc {row['sar_dem_only']['accuracy']:.4f} "
            f"mIoU {row['sar_dem_only']['miou']:.4f}"
        )

    out = {"checkpoint": args.checkpoint, "split": args.split,
           "n_tiles": len(chosen), "levels": results}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote {dest}")

    # A compact acceptance summary, so the run states its own verdict.
    clear, full = results.get("0.00"), results.get("1.00")
    if clear and full:
        logger.info("--- acceptance ---")
        logger.info(f"  SAR ablation cost, clear sky : {clear['sar_contribution_miou']:+.4f} "
                    f"(target >= 0.03)")
        logger.info(f"  SAR ablation cost, full cloud: {full['sar_contribution_miou']:+.4f} "
                    f"(target >= 0.20)")
        logger.info(f"  SAR+DEM only under full cloud: acc "
                    f"{full['sar_dem_only']['accuracy']:.4f} (target >= 0.72), mIoU "
                    f"{full['sar_dem_only']['miou']:.4f} (target >= 0.50)")
        logger.info(f"  alpha_sar clear -> full cloud : "
                    f"{clear['all']['alpha_mean'][1]:.4f} -> {full['all']['alpha_mean'][1]:.4f} "
                    f"(target >= 2x)")


if __name__ == "__main__":
    main()
