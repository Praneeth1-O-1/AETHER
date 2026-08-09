"""AETHER — Training entrypoint.

Skeleton training script.  The training loop is structured but awaits
the dataset pipeline from the teammate.  All model instantiation,
optimizer configuration, and checkpoint logic is ready.

Usage::

    python train.py --config configs/model.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.aether import AETHERModel
from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AETHER model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Resolve device string to a ``torch.device``."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # ---- Model ----
    model = AETHERModel.build_from_config(args.config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Loss ----
    criterion = nn.CrossEntropyLoss()

    # ---- Data ----
    # TODO: Replace with actual dataset and dataloader once dataset
    #       pipeline is completed by teammate.
    #
    # Expected DataLoader output per batch:
    #   optical: (B, C_opt, H, W)
    #   sar:     (B, C_sar, H, W)
    #   dem:     (B, C_dem, H, W)
    #   target:  (B, H, W)  — integer class labels
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, ...)
    # val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, ...)

    logger.warning(
        "Dataset pipeline not yet available. Training loop is structured "
        "but will not execute until DataLoader is provided."
    )

    # ---- Training Loop (skeleton) ----
    # Uncomment when dataset is ready:
    #
    # for epoch in range(args.epochs):
    #     model.train()
    #     epoch_loss = 0.0
    #
    #     for batch_idx, (optical, sar, dem, target) in enumerate(train_loader):
    #         optical = optical.to(device)
    #         sar = sar.to(device)
    #         dem = dem.to(device)
    #         target = target.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = model(optical, sar, dem)
    #         loss = criterion(outputs["lulc"], target)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss += loss.item()
    #
    #     scheduler.step()
    #     avg_loss = epoch_loss / max(len(train_loader), 1)
    #     logger.info(f"Epoch {epoch+1}/{args.epochs} — Loss: {avg_loss:.4f}")
    #
    #     # Save checkpoint
    #     ckpt_path = Path(args.checkpoint_dir) / f"epoch_{epoch+1:03d}.pt"
    #     save_checkpoint(model, optimizer, epoch + 1, avg_loss, ckpt_path)


if __name__ == "__main__":
    main()
