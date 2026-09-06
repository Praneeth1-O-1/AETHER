#!/home/kirethik/python-envs/general/bin/python
"""Full metric report -- no arguments needed.

    python run_evaluate.py

Accuracy / precision / recall / F1 / IoU for LULC (per class, macro, weighted)
and for road and building, repeated across every dataset scenario (optical
present vs missing, road mapped vs unmapped, fully complete, ...).

Reads CHECKPOINT, EVAL_SPLIT and the thresholds from settings.py.
Writes settings.OUTPUT_DIR/full_evaluation.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings as S

if not Path(S.CHECKPOINT).exists():
    raise SystemExit(f"Checkpoint not found: {S.CHECKPOINT}\n"
                     f"Edit CHECKPOINT in settings.py, or train one first.")

sys.argv = [
    "evaluate_full.py",
    "--checkpoint", str(S.CHECKPOINT),
    "--config", str(S.MODEL_CONFIG),
    "--dataset-root", str(S.DATASET_ROOT),
    "--split", S.EVAL_SPLIT,
    "--batch-size", str(S.BATCH_SIZE),
    "--num-workers", str(S.NUM_WORKERS),
    "--road-threshold", str(S.ROAD_THRESHOLD),
    "--building-threshold", str(S.BUILDING_THRESHOLD),
    "--output", str(Path(S.OUTPUT_DIR) / "full_evaluation.json"),
]

from scripts.evaluate_full import main  # noqa: E402

if __name__ == "__main__":
    main()
