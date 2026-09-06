"""AETHER — Inference entrypoint.

Loads a trained checkpoint and either evaluates the held-out *test* split --
the locations never seen during training or model selection -- or runs on a
single tile.

Usage::

    python inference.py --checkpoint checkpoints/best.pt --dataset-root data/strict
    python inference.py --checkpoint checkpoints/best.pt --tile-dir data/strict/<aoi>/<tile> --save-alpha-maps

The architecture is reconstructed from the shapes recorded in the checkpoint,
not from configs/model.yaml, so a checkpoint always loads into the model it
was trained as even if the config file has since moved on.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    DEM_CHANNELS,
    _orbit_of,
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    NUM_LULC_CLASSES,
    OPTICAL_CHANNELS,
    SAR_CHANNELS,
    AETHERTileDataset,
    TileRecord,
    build_manifest,
    location_of,
    location_split,
    year_of,
)
from models.aether import AETHERModel
from utils.config import DotDict, load_config
from utils.metrics import BinaryIoU, ConfusionMatrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AETHER inference.")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--tile-dir", type=str, default=None,
                   help="Run on a single tile instead of a split.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--save-alpha-maps", action="store_true")
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> AETHERModel:
    """Rebuild the architecture recorded in the checkpoint, then load weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels = ckpt.get("channels", {})

    if ckpt.get("model_cfg"):
        # Checkpoint carries the architecture it was built with -- always prefer
        # it, so a config file that has since moved on cannot silently rebuild
        # the wrong model and load into it with strict=False.
        cfg_model = DotDict(ckpt["model_cfg"])
    else:
        # Pre-dates model_cfg: reconstruct the original skip-free architecture
        # rather than today's defaults.
        cfg_model = load_config(config_path).model
        cfg_model.use_skips = False
        cfg_model.decoder.out_channels = 16
        cfg_model.optical_encoder.detail_channels = 0
        for head in ("road", "building"):
            if head in cfg_model.task_heads:
                cfg_model.task_heads[head].hidden_channels = 32
        logger.info("Checkpoint has no model_cfg -- rebuilding the legacy "
                    "skip-free architecture (decoder out=16, heads=32).")

    cfg_model.optical_encoder.in_channels = channels.get("optical", OPTICAL_CHANNELS)
    cfg_model.sar_encoder.in_channels = channels.get("sar", SAR_CHANNELS)
    cfg_model.dem_encoder.in_channels = channels.get("dem", DEM_CHANNELS)
    # Weights come from the checkpoint; skip the ImageNet fetch entirely.
    cfg_model.optical_encoder.pretrained = False
    cfg_model.sar_encoder.pretrained = False
    cfg_model.task_heads.lulc.num_classes = ckpt.get("num_lulc_classes", NUM_LULC_CLASSES)

    model = AETHERModel.build_from_dict(cfg_model)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning(f"Checkpoint is missing these modules (kept random init): {missing}")
    if unexpected:
        logger.warning(f"Checkpoint has unused keys (ignored): {unexpected}")

    model = model.to(device, memory_format=torch.channels_last).eval()
    logger.info(
        f"Loaded {checkpoint_path} | epoch {ckpt.get('epoch')} "
        f"| best score {ckpt.get('best_score')} | channels {channels}"
    )
    return model


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        v = v.to(device, non_blocking=True)
        if v.dim() == 4:
            v = v.contiguous(memory_format=torch.channels_last)
        out[k] = v
    return out


@torch.no_grad()
def evaluate_split(model, loader, device, amp_dtype) -> dict:
    """Same metric definitions as training, so numbers are directly comparable."""
    conf = ConfusionMatrix(NUM_LULC_CLASSES, device=device)
    road, building = BinaryIoU(device=device), BinaryIoU(device=device)
    alpha_sum = torch.zeros(3, device=device)
    n_batches = 0

    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch["optical"], batch["sar"], batch["dem"],
                        presence=batch.get("presence"))
        conf.update(out["lulc"].float().argmax(1), batch["lulc"], batch["lulc"] != IGNORE_INDEX)
        road.update(out["road"], batch["road"], batch["road_mask"])
        building.update(out["building"], batch["building"], batch["building_mask"])
        alpha_sum += out["alpha_maps"].float().mean(dim=(0, 2, 3))
        n_batches += 1

    n = max(n_batches, 1)
    per_class = conf.per_class_iou().tolist()
    return {
        "accuracy": conf.accuracy(),
        "miou": conf.miou(),
        "per_class_iou": {
            name: (None if np.isnan(v) else round(v, 4))
            for name, v in zip(LULC_CLASS_NAMES, per_class)
        },
        "road": {"iou": road.iou(), "f1": road.f1(),
                 "precision": road.precision(), "recall": road.recall()},
        "building": {"iou": building.iou(), "f1": building.f1(),
                     "precision": building.precision(), "recall": building.recall()},
        "alpha_mean": [round(a, 4) for a in (alpha_sum / n).tolist()],
    }


def run_single_tile(model, args, device, amp_dtype) -> None:
    tile = Path(args.tile_dir)
    meta_path = tile / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    record = TileRecord(
        path=tile, location=location_of(tile.parent.name), year=year_of(tile.parent.name),
        has_optical=True, has_sar=True, has_lulc=True, has_road=True, has_building=True,
        # Orbit direction feeds an input channel, so a wrong default here is a
        # silent domain shift rather than a crash.
        orbit=_orbit_of(meta),
    )
    batch = AETHERTileDataset([record], augment=False)[0]
    inputs = {k: batch[k].unsqueeze(0).to(device) for k in ("optical", "sar", "dem")}
    # Presence comes from the validity channels, so a tile whose optical was
    # never acquired is reported as absent instead of silently fed as zeros.
    presence = batch["presence"].unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(inputs["optical"], inputs["sar"], inputs["dem"], presence=presence,
                    return_intermediates=args.save_alpha_maps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lulc = out["lulc"].float().argmax(1)[0].cpu().numpy().astype(np.uint8)
    road = torch.sigmoid(out["road"].float())[0, 0].cpu().numpy()
    building = torch.sigmoid(out["building"].float())[0, 0].cpu().numpy()
    alpha = out["alpha_maps"].float()[0].cpu().numpy()

    np.save(out_dir / "lulc_pred.npy", lulc)
    np.save(out_dir / "road_prob.npy", road)
    np.save(out_dir / "building_prob.npy", building)
    if args.save_alpha_maps:
        np.save(out_dir / "alpha_maps.npy", alpha)

    counts = np.bincount(lulc.ravel(), minlength=NUM_LULC_CLASSES)
    top = sorted(zip(LULC_CLASS_NAMES, counts), key=lambda kv: -kv[1])[:3]
    logger.info(f"Wrote predictions -> {out_dir}")
    logger.info(f"  dominant LULC: {[(n, f'{c / lulc.size:.1%}') for n, c in top]}")
    logger.info(f"  road>0.5 {road.mean():.3%} of px | building>0.5 {(building > 0.5).mean():.3%}")
    logger.info(f"  alpha[O,S,D] mean {alpha.mean(axis=(1, 2)).round(3).tolist()}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    if device.type != "cuda":
        amp_dtype = None

    model = load_model(args.config, args.checkpoint, device)

    if args.tile_dir:
        run_single_tile(model, args, device, amp_dtype)
        return

    records = build_manifest(Path(args.dataset_root))
    train, val, test = location_split(records, args.val_frac, args.test_frac)
    chosen = {"train": train, "val": val, "test": test}[args.split]
    logger.info(
        f"Evaluating '{args.split}': {len(chosen)} tiles / "
        f"{len({r.location for r in chosen})} held-out locations"
    )

    loader = DataLoader(
        AETHERTileDataset(chosen, augment=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    metrics = evaluate_split(model, loader, device, amp_dtype)

    logger.info(
        f"{args.split}: acc {metrics['accuracy']:.4f} | mIoU {metrics['miou']:.4f} | "
        f"road IoU {metrics['road']['iou']:.4f} (P{metrics['road']['precision']:.3f}/"
        f"R{metrics['road']['recall']:.3f}) | building IoU {metrics['building']['iou']:.4f} "
        f"(P{metrics['building']['precision']:.3f}/R{metrics['building']['recall']:.3f}) | "
        f"alpha[O,S,D] {metrics['alpha_mean']}"
    )
    logger.info(f"  per-class IoU: {metrics['per_class_iou']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"metrics_{args.split}.json"
    dest.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Wrote {dest}")


if __name__ == "__main__":
    main()
