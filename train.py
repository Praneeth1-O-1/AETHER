"""AETHER — Training entrypoint.

Trains the full encoder -> CrossModalAlphaFusion -> decoder -> {lulc, road,
building} heads jointly, end-to-end, on the AETHER_DATASET tile archive.
Since all three heads share the same trunk (encoders + fusion + decoder),
one combined loss (lulc CE + road BCE + building BCE) is backpropagated
per batch -- there's no per-head training stage. Alpha maps are not
supervised directly -- there is no ground-truth fusion ratio -- they emerge
from backprop through this combined loss.

Usage::

    python train.py --dataset-root datasets/AETHER_DATASET/data/strict
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import AETHERTileDataset, IGNORE_INDEX, NUM_LULC_CLASSES, spatial_split
from models.aether import AETHERModel
from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AETHER model.")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--dataset-root", type=str, default="datasets/AETHER_DATASET/data/strict")
    parser.add_argument("--n-val", type=int, default=2, help="Spatial locations held out for val.")
    parser.add_argument("--n-test", type=int, default=2, help="Spatial locations held out for test.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=3e-4,
                         help="LR for randomly-initialized modules (DEM encoder, fusion, decoder, head).")
    parser.add_argument("--backbone_lr", type=float, default=1e-5,
                         help="LR for the pretrained optical/SAR backbones.")
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--modality_dropout_prob", type=float, default=0.15,
                         help="Prob. of zeroing one random modality per sample during training.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True,
                         help="Disable ImageNet-pretrained backbone weights (avoids a network fetch).")

    parser.add_argument("--lulc_weight", type=float, default=1.0)
    parser.add_argument("--road_weight", type=float, default=0.5,
                         help="Lower than lulc by default -- road/building are auxiliary "
                              "tasks here, kept from dominating the shared trunk's gradient.")
    parser.add_argument("--building_weight", type=float, default=0.5)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint (e.g. checkpoints/last.pt) to resume training from.")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, best_val_loss, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = target * num_classes + pred
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def mean_iou(conf: torch.Tensor) -> float:
    conf = conf.float()
    intersection = torch.diag(conf)
    union = conf.sum(0) + conf.sum(1) - intersection
    valid = union > 0
    if valid.sum() == 0:
        return 0.0
    return (intersection[valid] / union[valid]).mean().item()


def binary_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """IoU for a binary/soft-label head. `target` is thresholded too, so this
    also works for building_presence's continuous [0,1] coverage fraction."""
    pred = (torch.sigmoid(logits) > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred * target_bin).sum().item()
    union = ((pred + target_bin) > 0).float().sum().item()
    return intersection / union if union > 0 else 0.0


def compute_losses(outputs, lulc_target, road_target, building_target,
                    criterion_lulc, criterion_road, criterion_building, args):
    loss_lulc = criterion_lulc(outputs["lulc"], lulc_target)
    loss_road = criterion_road(outputs["road"], road_target)
    loss_building = criterion_building(outputs["building"], building_target)
    total = args.lulc_weight * loss_lulc + args.road_weight * loss_road + args.building_weight * loss_building
    return total, loss_lulc, loss_road, loss_building


@torch.no_grad()
def evaluate(model, loader, criterion_lulc, criterion_road, criterion_building, args, device):
    model.eval()
    n_batches, correct, total = 0, 0, 0
    total_loss = total_lulc = total_road = total_building = 0.0
    conf = torch.zeros(NUM_LULC_CLASSES, NUM_LULC_CLASSES, dtype=torch.int64)
    alpha_sum = torch.zeros(3)
    road_iou_sum, building_iou_sum = 0.0, 0.0

    for optical, sar, dem, lulc_target, road_target, building_target in loader:
        optical, sar, dem = optical.to(device), sar.to(device), dem.to(device)
        lulc_target, road_target, building_target = (
            lulc_target.to(device), road_target.to(device), building_target.to(device)
        )
        outputs = model(optical, sar, dem)
        loss, loss_lulc, loss_road, loss_building = compute_losses(
            outputs, lulc_target, road_target, building_target,
            criterion_lulc, criterion_road, criterion_building, args,
        )

        total_loss += loss.item()
        total_lulc += loss_lulc.item()
        total_road += loss_road.item()
        total_building += loss_building.item()
        n_batches += 1
        alpha_sum += outputs["alpha_maps"].mean(dim=(0, 2, 3)).cpu()
        road_iou_sum += binary_iou(outputs["road"], road_target)
        building_iou_sum += binary_iou(outputs["building"], building_target)

        pred = outputs["lulc"].argmax(dim=1)
        mask = lulc_target != IGNORE_INDEX
        correct += (pred[mask] == lulc_target[mask]).sum().item()
        total += mask.sum().item()
        conf += confusion_matrix(pred[mask].cpu(), lulc_target[mask].cpu(), NUM_LULC_CLASSES)

    n = max(n_batches, 1)
    metrics = {
        "loss": total_loss / n,
        "lulc_loss": total_lulc / n,
        "road_loss": total_road / n,
        "building_loss": total_building / n,
        "acc": correct / max(total, 1),
        "miou": mean_iou(conf),
        "road_iou": road_iou_sum / n,
        "building_iou": building_iou_sum / n,
        "alpha": (alpha_sum / n).tolist(),
    }
    return metrics


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # ---- Data ----
    train_dirs, val_dirs, test_dirs = spatial_split(Path(args.dataset_root), args.n_val, args.n_test)
    logger.info(f"Spatial split: {len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test tiles")

    train_ds = AETHERTileDataset(train_dirs, augment=True)
    val_ds = AETHERTileDataset(val_dirs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # ---- Model ----
    cfg = load_config(args.config)
    cfg.model.optical_encoder.in_channels = 10  # this dataset's 10 usable Sentinel-2 bands
    cfg.model.optical_encoder.pretrained = args.pretrained
    cfg.model.sar_encoder.pretrained = args.pretrained
    cfg.model.task_heads.lulc.num_classes = NUM_LULC_CLASSES
    cfg.model.modality_dropout_prob = args.modality_dropout_prob

    model = AETHERModel.build_from_dict(cfg.model)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    # ---- Optimizer & Scheduler ----
    # Pretrained backbones get a much lower LR than the randomly-initialized
    # DEM encoder / fusion / decoder / head, so fine-tuning doesn't overwrite
    # useful ImageNet features before the fresh modules catch up.
    backbone_params = list(model.optical_encoder.parameters()) + list(model.sar_encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Loss ----
    # ignore_index=255 for unlabeled lulc pixels; class 0 (water) is a real
    # class and must never be confused with "no label". road/building have no
    # nodata pixels anywhere in the archive, so no ignore_index needed there.
    # pos_weight compensates for road/building being minority classes
    # (dataset-wide positive fraction ~0.19 for road, ~0.15 mean coverage for
    # building_presence) -- fixed constants from the archive, not per-batch.
    criterion_lulc = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_road = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0, device=device))
    criterion_building = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.5, device=device))

    # ---- Resume ----
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            # Architecture changed since this checkpoint (e.g. new task heads added).
            # New modules keep random init; optimizer/scheduler state below still
            # matches the *old* param groups, so this is only safe for inspecting
            # weights, not for a seamless training resume across an architecture change.
            logger.warning(f"Resumed checkpoint doesn't match current architecture -- "
                            f"missing: {missing}, unexpected: {unexpected}")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed at epoch {start_epoch}, best_val_loss so far {best_val_loss:.4f}")

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        n_batches = 0
        epoch_loss = epoch_lulc = epoch_road = epoch_building = 0.0
        alpha_sum = torch.zeros(3)

        for optical, sar, dem, lulc_target, road_target, building_target in train_loader:
            optical, sar, dem = optical.to(device), sar.to(device), dem.to(device)
            lulc_target, road_target, building_target = (
                lulc_target.to(device), road_target.to(device), building_target.to(device)
            )

            optimizer.zero_grad()
            outputs = model(optical, sar, dem)
            loss, loss_lulc, loss_road, loss_building = compute_losses(
                outputs, lulc_target, road_target, building_target,
                criterion_lulc, criterion_road, criterion_building, args,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_lulc += loss_lulc.item()
            epoch_road += loss_road.item()
            epoch_building += loss_building.item()
            alpha_sum += outputs["alpha_maps"].detach().mean(dim=(0, 2, 3)).cpu()
            n_batches += 1

        scheduler.step()
        n = max(n_batches, 1)
        avg_loss, avg_lulc, avg_road, avg_building = (
            epoch_loss / n, epoch_lulc / n, epoch_road / n, epoch_building / n
        )
        train_alpha = (alpha_sum / n).tolist()

        val = evaluate(model, val_loader, criterion_lulc, criterion_road, criterion_building, args, device)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss {avg_loss:.4f} (lulc {avg_lulc:.4f} road {avg_road:.4f} building {avg_building:.4f}) | "
            f"val_loss {val['loss']:.4f} val_acc {val['acc']:.3f} val_mIoU {val['miou']:.3f} "
            f"val_road_iou {val['road_iou']:.3f} val_building_iou {val['building_iou']:.3f} | "
            f"alpha[O,S,D] train={[round(a, 3) for a in train_alpha]} "
            f"val={[round(a, 3) for a in val['alpha']]}"
        )

        ckpt_dir = Path(args.checkpoint_dir)
        save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_val_loss, ckpt_dir / "last.pt")
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, val["loss"], best_val_loss, ckpt_dir / "best.pt")
            logger.info(f"  New best val_loss {val['loss']:.4f} -> saved {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
