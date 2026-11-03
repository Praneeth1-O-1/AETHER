"""Build the caches a training run would otherwise build on GPU time.

Writes, under ./.manifest_cache/:

- the tile manifest (reused if already present),
- per-tile LULC class counts for every tile, used by repeat-factor sampling,
- the cloud-mask bank harvested from the train split of ``--split-key``.

Reading the archive over a mounted Kaggle input is slow, and none of this
needs a GPU, so run it in a CPU session and hand the caches to training.

Usage::

    python scripts/precompute_caches.py --dataset-root <root> --split-key coordinate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.clouds import CloudMaskBank, default_bank_path  # noqa: E402
from data.dataset import SPLIT_KEYS, build_manifest, class_histograms, location_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split-key", choices=SPLIT_KEYS, default="coordinate")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.dataset_root)

    t0 = time.time()
    records = build_manifest(root)
    logger.info(f"Manifest: {len(records)} tiles ({time.time() - t0:.0f} s)")

    train, val, test = location_split(records, key=args.split_key)
    logger.info(f"{args.split_key} split: train {len(train)} / val {len(val)} / test {len(test)} tiles")

    t0 = time.time()
    CloudMaskBank.build(train, cache=default_bank_path(root, args.split_key))
    logger.info(f"Cloud mask bank done ({time.time() - t0:.0f} s)")

    t0 = time.time()
    hist = class_histograms(records, root, workers=args.workers)
    logger.info(f"Class counts for {len(hist)} tiles ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
