"""AETHER — Inference entrypoint.

Loads a trained checkpoint and runs inference on input tensors.
Supports alpha map extraction for interpretability analysis.

Usage::

    python inference.py --config configs/model.yaml --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

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
    device: torch.device,
) -> AETHERModel:
    """Load a trained AETHER model from checkpoint.

    Parameters
    ----------
    config_path : str
        Path to model configuration YAML.
    checkpoint_path : str
        Path to saved checkpoint.
    device : torch.device
        Target device.

    Returns
    -------
    AETHERModel
        Model with loaded weights in eval mode.
    """
    model = AETHERModel.build_from_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
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

    model = load_model(args.config, args.checkpoint, device)

    # ------------------------------------------------------------------
    # TODO: Replace with actual data loading once dataset pipeline is ready.
    #
    # Example usage with a DataLoader:
    #
    # for optical, sar, dem in test_loader:
    #     optical = optical.to(device)
    #     sar = sar.to(device)
    #     dem = dem.to(device)
    #
    #     outputs = predict(model, optical, sar, dem)
    #     lulc_preds = outputs["lulc"].argmax(dim=1)  # (B, H, W)
    #     alpha_maps = outputs["alpha_maps"]            # (B, 3, H', W')
    #
    #     if args.save_alpha_maps:
    #         out_dir = Path(args.output_dir)
    #         out_dir.mkdir(parents=True, exist_ok=True)
    #         np.save(out_dir / "alpha_maps.npy", alpha_maps.cpu().numpy())
    # ------------------------------------------------------------------

    logger.warning(
        "Inference pipeline awaiting dataset integration. "
        "Use predict() programmatically or integrate a DataLoader."
    )


if __name__ == "__main__":
    main()
