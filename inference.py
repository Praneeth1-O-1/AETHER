"""AETHER — Inference entrypoint.

Loads a trained checkpoint and runs inference, either on the held-out test
split (reporting the same loss/accuracy/mIoU/alpha diagnostics as training)
or, with --tile-dir, on a single tile.

Usage::

    python inference.py --checkpoint checkpoints/best.pt --dataset-root datasets/AETHER_DATASET/data/strict
    python inference.py --checkpoint checkpoints/best.pt --tile-dir path/to/one/tile --save_alpha_maps
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from data.dataset import AETHERTileDataset, IGNORE_INDEX, NUM_LULC_CLASSES, spatial_split
from models.aether import AETHERModel
from utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AETHER inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--optical-channels", type=int, default=10,
                         help="Must match what the checkpoint was trained with.")
    parser.add_argument("--num-classes", type=int, default=NUM_LULC_CLASSES,
                         help="Must match what the checkpoint was trained with.")
    parser.add_argument("--dataset-root", type=str, default="datasets/AETHER_DATASET/data/strict")
    parser.add_argument("--n-val", type=int, default=2)
    parser.add_argument("--n-test", type=int, default=2)
    parser.add_argument("--tile-dir", type=str, default=None,
                         help="Run on a single tile instead of the test split.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--save_alpha_maps",
        action="store_true",
        help="Save alpha maps as numpy arrays for visualization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for inference outputs.",
    )
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


def load_model(
    config_path: str,
    checkpoint_path: str,
    optical_channels: int,
    num_classes: int,
    device: torch.device,
) -> AETHERModel:
    """Load a trained AETHER model from checkpoint.

    Parameters
    ----------
    config_path : str
        Path to model configuration YAML.
    checkpoint_path : str
        Path to saved checkpoint.
    optical_channels, num_classes : int
        Must match the architecture the checkpoint was trained with --
        configs/model.yaml's raw defaults (13 channels, 10 classes) do not
        match this dataset (10 channels, 9 classes), so these are passed
        explicitly rather than read from the config file.
    device : torch.device
        Target device.

    Returns
    -------
    AETHERModel
        Model with loaded weights in eval mode.
    """
    cfg = load_config(config_path)
    cfg.model.optical_encoder.in_channels = optical_channels
    cfg.model.optical_encoder.pretrained = False  # overwritten by checkpoint; skip the network fetch
    cfg.model.sar_encoder.pretrained = False
    cfg.model.task_heads.lulc.num_classes = num_classes

    model = AETHERModel.build_from_dict(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path} (epoch {checkpoint.get('epoch')}, "
                f"best_val_loss {checkpoint.get('best_val_loss')})")
    return model


@torch.no_grad()
def predict(
    model: AETHERModel,
    optical: torch.Tensor,
    sar: torch.Tensor,
    dem: torch.Tensor,
    return_intermediates: bool = False,
) -> dict[str, torch.Tensor]:
    """Run inference on a batch of inputs.

    Parameters
    ----------
    model : AETHERModel
        Trained model in eval mode.
    optical, sar, dem : torch.Tensor
        Input tensors.
    return_intermediates : bool
        Whether to return fusion intermediates.

    Returns
    -------
    dict[str, torch.Tensor]
        Predictions and alpha maps.
    """
    return model(optical, sar, dem, return_intermediates=return_intermediates)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    model = load_model(
        args.config, args.checkpoint, args.optical_channels, args.num_classes, device
    )
    out_dir = Path(args.output_dir)

    if args.tile_dir:
        ds = AETHERTileDataset([Path(args.tile_dir)], augment=False)
        optical, sar, dem, target = ds[0]
        optical, sar, dem = optical.unsqueeze(0).to(device), sar.unsqueeze(0).to(device), dem.unsqueeze(0).to(device)

        outputs = predict(model, optical, sar, dem, return_intermediates=args.save_alpha_maps)
        lulc_pred = outputs["lulc"].argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        alpha_maps = outputs["alpha_maps"].squeeze(0).cpu().numpy()        # (3, H', W')

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "lulc_pred.npy", lulc_pred)
        logger.info(f"Saved prediction -> {out_dir / 'lulc_pred.npy'} "
                    f"| alpha[O,S,D] mean={alpha_maps.mean(axis=(1, 2)).round(3).tolist()}")
        if args.save_alpha_maps:
            np.save(out_dir / "alpha_maps.npy", alpha_maps)
            logger.info(f"Saved alpha maps -> {out_dir / 'alpha_maps.npy'}")
        return

    # No single tile given: evaluate on the held-out spatial test split,
    # using the same split boundary train.py used (same dataset-root/n_val/n_test).
    from train import confusion_matrix, mean_iou  # reuse rather than duplicate

    _, _, test_dirs = spatial_split(Path(args.dataset_root), args.n_val, args.n_test)
    logger.info(f"Evaluating on {len(test_dirs)} held-out test tiles.")
    test_ds = AETHERTileDataset(test_dirs, augment=False)

    correct, total = 0, 0
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.int64)
    alpha_sum = torch.zeros(3)
    all_preds = []

    with torch.no_grad():
        for optical, sar, dem, target in test_ds:
            optical, sar, dem = optical.unsqueeze(0).to(device), sar.unsqueeze(0).to(device), dem.unsqueeze(0).to(device)
            outputs = predict(model, optical, sar, dem)

            pred = outputs["lulc"].argmax(dim=1).squeeze(0).cpu()
            mask = target != IGNORE_INDEX
            correct += (pred[mask] == target[mask]).sum().item()
            total += mask.sum().item()
            conf += confusion_matrix(pred[mask], target[mask], args.num_classes)
            alpha_sum += outputs["alpha_maps"].mean(dim=(0, 2, 3)).cpu()
            all_preds.append(pred.numpy())

    acc = correct / max(total, 1)
    miou = mean_iou(conf)
    avg_alpha = (alpha_sum / max(len(test_ds), 1)).round(decimals=3).tolist()
    logger.info(f"Test set: acc {acc:.3f} mIoU {miou:.3f} | alpha[O,S,D] {avg_alpha}")

    if args.save_alpha_maps:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "test_predictions.npy", np.stack(all_preds))
        logger.info(f"Saved per-tile predictions -> {out_dir / 'test_predictions.npy'}")


if __name__ == "__main__":
    main()
