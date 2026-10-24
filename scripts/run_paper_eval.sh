#!/usr/bin/env bash
# Every number the paper needs, from one trained checkpoint, in one run.
#
#   bash scripts/run_paper_eval.sh <checkpoint.pt> <dataset_root>
#
# Writes JSON to outputs/paper/. Each evaluation runs twice: on the full test
# split, and on the leak-free subset (test tiles whose coordinates never occur
# in train/val -- see data.dataset.leak_free). Thresholds are tuned on
# validation only.
set -uo pipefail
CKPT=${1:?usage: run_paper_eval.sh <checkpoint> <dataset_root>}
ROOT=${2:?usage: run_paper_eval.sh <checkpoint> <dataset_root>}
OUT=outputs/paper
NW=${NUM_WORKERS:-4}
mkdir -p "$OUT"
run() { echo -e "\n=== $* ==="; "$@" || echo "!!! FAILED: $*"; }

run python scripts/benchmark_gpu.py --checkpoint "$CKPT" --output "$OUT/gpu_benchmark.json"

for TAG in full leakfree; do
  LF=""; [ "$TAG" = leakfree ] && LF="--leak-free"
  run python scripts/tune_thresholds.py --checkpoint "$CKPT" --dataset-root "$ROOT" \
      --num-workers "$NW" $LF --output "$OUT/thresholds_$TAG.json"
done

read RT BT < <(python - "$OUT/thresholds_full.json" <<'EOF'
import json, sys
try:
    t = json.load(open(sys.argv[1]))["thresholds_chosen_on_val"]
    print(t["road"], t["building"])
except Exception:
    print(0.5, 0.5)
EOF
)
echo "Using val-tuned thresholds: road=$RT building=$BT"

for TAG in full leakfree; do
  LF=""; [ "$TAG" = leakfree ] && LF="--leak-free"
  run python scripts/evaluate_full.py --checkpoint "$CKPT" --dataset-root "$ROOT" --split test \
      --num-workers "$NW" --road-threshold "$RT" --building-threshold "$BT" $LF \
      --output "$OUT/full_evaluation_$TAG.json"
  run python scripts/ablate_modalities.py --checkpoint "$CKPT" --dataset-root "$ROOT" --split test \
      --num-workers "$NW" --road-threshold "$RT" --building-threshold "$BT" $LF \
      --output "$OUT/modality_ablation_$TAG.json"
  run python scripts/cloud_curve.py --checkpoint "$CKPT" --dataset-root "$ROOT" --split test \
      --num-workers "$NW" $LF --output "$OUT/cloud_curve_$TAG.json"
done

echo -e "\nDone. Results:"; ls -la "$OUT"
