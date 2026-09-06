"""AETHER — Predict from cached f_shared features (decoder + task head only).

Quick plumbing check: loads the f_shared tensors already cached by
encode_dataset.py and runs them through model.decoder + task_heads["lulc"],
skipping the encoders/fusion (already baked into the cached features).

The model is untrained unless --checkpoint is given, so predictions are
random-init noise, not real land cover -- this only verifies the decoder ->
head shape/plumbing works on real cached features.

Usage::

    python predict_from_fshared.py \
        --fshared-dir outputs/fshared \
        --output-dir outputs/predictions
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from models.aether import AETHERModel
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_model(config_path: str, checkpoint: str | None, device: torch.device) -> AETHERModel:
    cfg = load_config(config_path)
    cfg.model.optical_encoder.pretrained = False
    cfg.model.sar_encoder.pretrained = False
    model = AETHERModel.build_from_dict(cfg.model)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
        if missing:
            logger.warning(f"Checkpoint predates these modules (kept random init): {missing}")
        if unexpected:
            logger.warning(f"Checkpoint has unused keys (ignored): {unexpected}")
        logger.info(f"Loaded trained weights from {checkpoint}")
    else:
        logger.warning("No --checkpoint given: decoder + lulc head are randomly initialized. "
                        "Predictions are a shape/plumbing check only, not real land cover.")

    model = model.to(device)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fshared-dir", type=Path, default=Path("outputs/fshared"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--config", type=str, default="configs/model.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model(args.config, args.checkpoint, device)

    # A skip-connected decoder needs the encoders' intermediate feature maps as
    # well as f_shared, and encode_dataset.py caches only f_shared. Fail with a
    # clear message rather than an opaque channel-count mismatch inside conv1.
    if getattr(model, "use_skips", False):
        raise SystemExit(
            "This model uses skip connections, so the decoder needs encoder features "
            "at H/2, H/4 and H/8 -- cached f_shared alone is not sufficient.\n"
            "Run the full pipeline instead:\n"
            "  python inference.py --checkpoint <ckpt> --tile-dir <tile>"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.fshared_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No cached f_shared .npz files found in {args.fshared_dir}")

    class_counts: Counter[int] = Counter()
    with torch.no_grad():
        for i, npz_path in enumerate(npz_files, 1):
            tile_id = npz_path.stem
            f_shared = torch.from_numpy(np.load(npz_path)["f_shared"]).unsqueeze(0).to(device)

            decoded = model.decoder(f_shared)                     # (1, 16, 256, 256)
            logits = model.task_heads["lulc"](decoded)            # (1, 10, 256, 256)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (256, 256)

            np.save(args.output_dir / f"{tile_id}_lulc_pred.npy", pred.astype(np.uint8))
            counts = Counter(pred.ravel().tolist())
            class_counts.update(counts)

            top_class, top_n = counts.most_common(1)[0]
            logger.info(f"[{i}/{len(npz_files)}] {tile_id}: decoded {tuple(decoded.shape)}, "
                        f"logits {tuple(logits.shape)}, dominant class {top_class} "
                        f"({100 * top_n / pred.size:.1f}% of pixels)")

    total = sum(class_counts.values())
    logger.info(f"Done: {len(npz_files)} tiles decoded -> lulc predictions in {args.output_dir}")
    logger.info("Dataset-wide predicted class distribution (untrained, expect ~uniform noise):")
    for cls in sorted(class_counts):
        pct = 100 * class_counts[cls] / total
        logger.info(f"  class {cls}: {pct:.2f}%")


if __name__ == "__main__":
    main()
