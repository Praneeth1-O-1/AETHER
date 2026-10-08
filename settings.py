"""AETHER — every knob in one place.

Edit the values here, then run one of the no-argument entry points:

    python run_train.py       # train
    python run_predict.py     # predict + render PNGs you can look at
    python run_evaluate.py    # full metric report, all scenarios

Nothing else needs a command line. Every script under `scripts/` still accepts
CLI flags for one-off overrides, and those flags default to the values below,
so the two paths can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

# =========================================================================
# Paths
# =========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

#: Tile archive to read.
DATASET_ROOT = PROJECT_ROOT / "data" / "strict"

#: Architecture definition.
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"

#: Checkpoint used for prediction and evaluation.
#:   checkpoints/best.pt        -- the trained multi-task model (skip-free)
#:   checkpoints_skip/best.pt   -- once you train the skip-connected version
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"

#: Where training writes last.pt / best.pt / history.csv. Use a NEW directory
#: for a new experiment so previous results stay intact for comparison.
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_skip"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"
LOG_DIR = PROJECT_ROOT / "logs"

# =========================================================================
# Data split -- by geographic location, year-blind (see data/dataset.py)
# =========================================================================

VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# =========================================================================
# Prediction / evaluation
# =========================================================================

#: Decision thresholds, tuned on the VALIDATION split (never on test).
#: Both heads are trained recall-biased, so 0.5 is not the right operating
#: point. Re-derive these with `python scripts/tune_thresholds.py` after
#: training a new model -- they do not transfer between checkpoints.
ROAD_THRESHOLD = 0.60
BUILDING_THRESHOLD = 0.71

#: Which split to evaluate: "test" (unseen locations), "val", or "train".
EVAL_SPLIT = "test"

BATCH_SIZE = 16
NUM_WORKERS = 8
DEVICE = "auto"          # "auto" | "cuda" | "cpu"
AMP = "bf16"             # "bf16" | "fp16" | "off"

# =========================================================================
# Prediction targets
# =========================================================================

#: Set to a tile directory (str or Path) to predict that ONE tile.
#: Leave None to auto-pick N_PREDICT_TILES tiles that have road and building
#: labels -- an unmapped-road tile just shows an empty ground-truth panel.
PREDICT_TILE = None
N_PREDICT_TILES = 4

#: Changes which tiles the auto-picker chooses.
PREDICT_SEED = 0

#: Also write raw .npy arrays (lulc_pred, road_prob, building_prob, alpha_maps)
#: alongside the PNGs.
SAVE_ARRAYS = True

# =========================================================================
# Training
# =========================================================================

EPOCHS = 60

#: 12 fits the 6 GB card comfortably with skip connections on (3.34 GiB peak)
#: and is within 2% of bs=16 throughput, while giving more optimizer steps.
TRAIN_BATCH_SIZE = 12
ACCUM_STEPS = 1

LR = 3e-4                # randomly-initialized modules
BACKBONE_LR = 5e-5       # pretrained optical/SAR backbones
WEIGHT_DECAY = 1e-2
WARMUP_FRACTION = 0.03
GRAD_CLIP = 1.0

#: Heads to train. Dropping a head means its gradient never touches the shared
#: trunk. "road,building" isolates the tasks whose labels are independent of
#: optical -- the only place the fusion contribution can be demonstrated.
TASKS = "lulc,road,building"

#: Per-modality probability of being independently zeroed per sample.
#: train.py exposes these as three separate flags (--drop-optical/-sar/-dem),
#: not one combined probability -- 0.15 each (train.py's own default) was
#: judged too weak to force redundant pathways; 0.5 each is the setting that
#: makes the fusion claim testable.
DROP_OPTICAL = 0.5
DROP_SAR = 0.5
DROP_DEM = 0.5

#: Relative loss weights.
LULC_WEIGHT = 1.0
ROAD_WEIGHT = 0.5
BUILDING_WEIGHT = 0.5
DICE_WEIGHT = 1.0

# =========================================================================
# Cloud injection -- the reason SAR has anything to do (see train.py)
# =========================================================================

CLOUD_PROB = 0.5             # fraction of training samples receiving injected cloud
CLOUD_START_COVERAGE = 0.3   # max cloud coverage at epoch 1; ramps to 1.0
CLOUD_RAMP_FRAC = 0.5        # fraction of training over which coverage reaches 1.0
CLOUD_OPAQUE_FRAC = 0.6      # share of clouded samples with validity zeroed (vs haze)
CLOUD_FULL_PROB = 0.15       # share of clouded samples forced to total occlusion
CLOUD_SELECT_WEIGHT = 0.5    # weight of the 50%-cloud val score in checkpoint selection

# =========================================================================
# Anti-collapse: forces every encoder to be independently predictive
# =========================================================================

AUX_WEIGHT = 0.3             # weight of the per-modality auxiliary LULC losses
ALPHA_ENTROPY_WEIGHT = 0.02  # initial weight of the alpha entropy floor
ALPHA_ENTROPY_FRAC = 0.4     # fraction of training over which it's annealed to 0
OGM_ALPHA = 0.5              # OGM-GE gradient modulation strength; 0 disables

USE_PRETRAINED = True

#: Path to a checkpoint to resume from, or None to start fresh.
RESUME = None


def as_argv(**overrides) -> list[str]:
    """Render selected settings as CLI arguments, for reusing script mains."""
    return [str(x) for pair in overrides.items() for x in pair]
