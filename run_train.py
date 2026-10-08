#!/home/kirethik/python-envs/general/bin/python
"""Train -- no arguments needed.

    python run_train.py

Reads every hyperparameter from settings.py. Writes last.pt, best.pt,
history.csv and config.json to settings.CHECKPOINT_DIR.

Set CHECKPOINT_DIR to a NEW directory for each experiment so earlier results
stay intact for comparison. Set RESUME to a checkpoint path to continue an
interrupted run -- optimizer, scheduler and AMP scaler state are all restored.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings as S

argv = [
    "train.py",
    "--config", str(S.MODEL_CONFIG),
    "--dataset-root", str(S.DATASET_ROOT),
    "--val-frac", str(S.VAL_FRACTION),
    "--test-frac", str(S.TEST_FRACTION),
    "--epochs", str(S.EPOCHS),
    "--batch-size", str(S.TRAIN_BATCH_SIZE),
    "--accum-steps", str(S.ACCUM_STEPS),
    "--num-workers", str(S.NUM_WORKERS),
    "--lr", str(S.LR),
    "--backbone-lr", str(S.BACKBONE_LR),
    "--weight-decay", str(S.WEIGHT_DECAY),
    "--warmup-frac", str(S.WARMUP_FRACTION),
    "--grad-clip", str(S.GRAD_CLIP),
    "--tasks", S.TASKS,
    "--drop-optical", str(S.DROP_OPTICAL),
    "--drop-sar", str(S.DROP_SAR),
    "--drop-dem", str(S.DROP_DEM),
    "--cloud-prob", str(S.CLOUD_PROB),
    "--cloud-start-coverage", str(S.CLOUD_START_COVERAGE),
    "--cloud-ramp-frac", str(S.CLOUD_RAMP_FRAC),
    "--cloud-opaque-frac", str(S.CLOUD_OPAQUE_FRAC),
    "--cloud-full-prob", str(S.CLOUD_FULL_PROB),
    "--cloud-select-weight", str(S.CLOUD_SELECT_WEIGHT),
    "--aux-weight", str(S.AUX_WEIGHT),
    "--alpha-entropy-weight", str(S.ALPHA_ENTROPY_WEIGHT),
    "--alpha-entropy-frac", str(S.ALPHA_ENTROPY_FRAC),
    "--ogm-alpha", str(S.OGM_ALPHA),
    "--lulc-weight", str(S.LULC_WEIGHT),
    "--road-weight", str(S.ROAD_WEIGHT),
    "--building-weight", str(S.BUILDING_WEIGHT),
    "--dice-weight", str(S.DICE_WEIGHT),
    "--amp", S.AMP,
    "--device", S.DEVICE,
    "--checkpoint-dir", str(S.CHECKPOINT_DIR),
]
if not S.USE_PRETRAINED:
    argv.append("--no-pretrained")
if S.RESUME:
    argv += ["--resume", str(S.RESUME)]

sys.argv = argv

from train import main  # noqa: E402

if __name__ == "__main__":
    print(f"Checkpoints -> {S.CHECKPOINT_DIR}")
    print(f"Tasks: {S.TASKS} | epochs {S.EPOCHS} | batch {S.TRAIN_BATCH_SIZE} "
          f"| modality dropout O={S.DROP_OPTICAL} S={S.DROP_SAR} D={S.DROP_DEM}\n")
    main()
