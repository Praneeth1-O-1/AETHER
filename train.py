"""AETHER — Training entrypoint.

Trains the full encoder -> CrossModalAlphaFusion -> decoder -> lulc head
pipeline end-to-end on the AETHER_DATASET tile archive. Alpha maps are not
supervised directly -- there is no ground-truth fusion ratio -- they emerge
from backprop through the lulc classification loss.

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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches, correct, total = 0.0, 0, 0, 0
    conf = torch.zeros(NUM_LULC_CLASSES, NUM_LULC_CLASSES, dtype=torch.int64)
    alpha_sum = torch.zeros(3)

    for optical, sar, dem, target in loader:
        optical, sar, dem, target = optical.to(device), sar.to(device), dem.to(device), target.to(device)
        outputs = model(optical, sar, dem)
        loss = criterion(outputs["lulc"], target)

        total_loss += loss.item()
        n_batches += 1
        alpha_sum += outputs["alpha_maps"].mean(dim=(0, 2, 3)).cpu()

        pred = outputs["lulc"].argmax(dim=1)
        mask = target != IGNORE_INDEX
        correct += (pred[mask] == target[mask]).sum().item()
        total += mask.sum().item()
        conf += confusion_matrix(pred[mask].cpu(), target[mask].cpu(), NUM_LULC_CLASSES)

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)
    miou = mean_iou(conf)
    avg_alpha = (alpha_sum / max(n_batches, 1)).tolist()
    return avg_loss, acc, miou, avg_alpha


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
    # ignore_index=255 for unlabeled pixels; class 0 (water) is a real class
    # and must never be confused with "no label".
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ---- Resume ----
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed at epoch {start_epoch}, best_val_loss so far {best_val_loss:.4f}")

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        alpha_sum = torch.zeros(3)

        for optical, sar, dem, target in train_loader:
            optical, sar, dem, target = optical.to(device), sar.to(device), dem.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(optical, sar, dem)
            loss = criterion(outputs["lulc"], target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            alpha_sum += outputs["alpha_maps"].detach().mean(dim=(0, 2, 3)).cpu()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_alpha = (alpha_sum / max(n_batches, 1)).tolist()

        val_loss, val_acc, val_miou, val_alpha = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | train_loss {avg_loss:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.3f} val_mIoU {val_miou:.3f} | "
            f"alpha[O,S,D] train={[round(a, 3) for a in train_alpha]} "
            f"val={[round(a, 3) for a in val_alpha]}"
        )

        ckpt_dir = Path(args.checkpoint_dir)
        save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_val_loss, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_loss, ckpt_dir / "best.pt")
            logger.info(f"  New best val_loss {val_loss:.4f} -> saved {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
