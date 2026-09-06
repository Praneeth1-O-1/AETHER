#!/home/kirethik/python-envs/general/bin/python
"""Ground truth vs. prediction, side by side -- no arguments needed.

    python run_compare.py

Just the comparison: for LULC, road, and building, the true label next to the
model's prediction. No input imagery, no probability heatmaps, no alpha maps
-- run run_predict.py instead for the full diagnostic view.

Reads CHECKPOINT, PREDICT_TILE / N_PREDICT_TILES / PREDICT_SEED, and the
thresholds from settings.py. Writes PNGs to settings.COMPARISON_DIR.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings as S

if not Path(S.CHECKPOINT).exists():
    raise SystemExit(f"Checkpoint not found: {S.CHECKPOINT}\n"
                     f"Edit CHECKPOINT in settings.py, or train one first "
                     f"with: python run_train.py")

argv = [
    "compare_labels.py",
    "--checkpoint", str(S.CHECKPOINT),
    "--config", str(S.MODEL_CONFIG),
    "--dataset-root", str(S.DATASET_ROOT),
    "--road-threshold", str(S.ROAD_THRESHOLD),
    "--building-threshold", str(S.BUILDING_THRESHOLD),
    "--output-dir", str(S.COMPARISON_DIR),
    "--device", S.DEVICE,
    "--seed", str(S.PREDICT_SEED),
]
if S.PREDICT_TILE:
    argv += ["--tile-dir", str(S.PREDICT_TILE)]
else:
    argv += ["--auto", str(S.N_PREDICT_TILES)]

sys.argv = argv

from scripts.compare_labels import main  # noqa: E402

if __name__ == "__main__":
    print(f"Checkpoint : {S.CHECKPOINT}")
    print(f"Thresholds : road {S.ROAD_THRESHOLD}  building {S.BUILDING_THRESHOLD}\n")
    main()
    print(f"\nOpen them with:  xdg-open {S.COMPARISON_DIR}")
