"""Calibrate the road/building decision thresholds, then report held-out test metrics.

The binary heads are trained with a damped ``pos_weight`` plus a Dice term, both
of which push a minority class toward recall. That is deliberate -- it is how a
3.7%-positive road head learns anything at all -- but it leaves the sigmoid
badly calibrated for a hardcoded 0.5 cutoff, so IoU at 0.5 understates the model.

Binary IoU is sharply threshold-sensitive in a way that mIoU and accuracy are
not, so the honest procedure is:

    1. sweep the threshold on the *validation* split,
    2. pick the IoU-maximizing threshold per head,
    3. report the *test* split once, at that threshold.

Choosing on val and reporting on test is what keeps this a real generalization
number rather than fitting the test set. The test split's locations were never
seen in training or in model selection.

Usage::

    python scripts/tune_thresholds.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    NUM_LULC_CLASSES,
    AETHERTileDataset,
    build_manifest,
    location_split,
)
from inference import load_model, to_device  # noqa: E402
from utils.metrics import ConfusionMatrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--output", type=str, default="outputs/final_metrics.json")
    return p.parse_args()


class ThresholdSweep:
    """Accumulate tp/fp/fn at every candidate threshold in one pass.

    Storing raw probabilities for a whole split would be ~5 GB; accumulating
    counts per threshold instead is O(len(grid)) memory and a single pass.
    """

    def __init__(self, grid: torch.Tensor, device: torch.device,
                 label_threshold: float = 0.5) -> None:
        self.grid = grid.to(device).view(-1, 1)
        self.label_threshold = label_threshold
        self.counts = torch.zeros(3, len(grid), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        keep = (mask > 0.5).reshape(-1)
        if not keep.any():
            return
        probs = torch.sigmoid(logits.float()).reshape(-1)[keep]
        truth = (target.reshape(-1)[keep] > self.label_threshold)

        pred = probs.unsqueeze(0) >= self.grid          # (T, N)
        truth_row = truth.unsqueeze(0)
        self.counts[0] += (pred & truth_row).sum(1).double()
        self.counts[1] += (pred & ~truth_row).sum(1).double()
        self.counts[2] += (~pred & truth_row).sum(1).double()

    def curves(self) -> dict[str, np.ndarray]:
        tp, fp, fn = self.counts.cpu().numpy()
        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}

    def best(self) -> tuple[float, dict[str, float]]:
        c = self.curves()
        i = int(np.argmax(c["iou"]))
        return float(self.grid[i].item()), {k: float(v[i]) for k, v in c.items()}


@torch.no_grad()
def sweep_split(model, loader, device, grid) -> tuple[ThresholdSweep, ThresholdSweep, ConfusionMatrix]:
    road = ThresholdSweep(grid, device)
    building = ThresholdSweep(grid, device)
    conf = ConfusionMatrix(NUM_LULC_CLASSES, device=device)

    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch["optical"], batch["sar"], batch["dem"],
                        presence=batch.get("presence"))
        road.update(out["road"], batch["road"], batch["road_mask"])
        building.update(out["building"], batch["building"], batch["building_mask"])
        conf.update(out["lulc"].float().argmax(1), batch["lulc"], batch["lulc"] != IGNORE_INDEX)

    return road, building, conf


def metrics_at(sweep: ThresholdSweep, threshold: float) -> dict[str, float]:
    """Read one threshold's metrics off an already-accumulated sweep."""
    idx = int(torch.argmin((sweep.grid.view(-1) - threshold).abs()).item())
    c = sweep.curves()
    return {k: round(float(v[idx]), 4) for k, v in c.items()}


def make_loader(records, args) -> DataLoader:
    return DataLoader(
        AETHERTileDataset(records, augment=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.config, args.checkpoint, device)
    grid = torch.linspace(0.05, 0.95, 91)

    records = build_manifest(Path(args.dataset_root))
    _, val_rec, test_rec = location_split(records)

    print(f"\n=== VAL sweep ({len(val_rec)} tiles / {len({r.location for r in val_rec})} locations) ===")
    v_road, v_bld, v_conf = sweep_split(model, make_loader(val_rec, args), device, grid)
    road_t, road_val = v_road.best()
    bld_t, bld_val = v_bld.best()

    curves = v_road.curves()
    print(f"{'thr':>6}{'road IoU':>10}{'road P':>9}{'road R':>9}{'bld IoU':>10}{'bld P':>9}{'bld R':>9}")
    bcurves = v_bld.curves()
    for i in range(0, len(grid), 5):
        t = grid[i].item()
        print(f"{t:>6.2f}{curves['iou'][i]:>10.4f}{curves['precision'][i]:>9.3f}{curves['recall'][i]:>9.3f}"
              f"{bcurves['iou'][i]:>10.4f}{bcurves['precision'][i]:>9.3f}{bcurves['recall'][i]:>9.3f}")

    print(f"\nchosen on val -> road {road_t:.2f} (IoU {road_val['iou']:.4f}), "
          f"building {bld_t:.2f} (IoU {bld_val['iou']:.4f})")
    print(f"  vs. threshold 0.50 -> road IoU {metrics_at(v_road, 0.5)['iou']:.4f}, "
          f"building IoU {metrics_at(v_bld, 0.5)['iou']:.4f}")

    print(f"\n=== TEST ({len(test_rec)} tiles / {len({r.location for r in test_rec})} unseen locations) ===")
    t_road, t_bld, t_conf = sweep_split(model, make_loader(test_rec, args), device, grid)

    per_class = t_conf.per_class_iou().tolist()
    result = {
        "checkpoint": args.checkpoint,
        "thresholds_chosen_on_val": {"road": road_t, "building": bld_t},
        "val": {
            "miou": round(v_conf.miou(), 4), "accuracy": round(v_conf.accuracy(), 4),
            "road": {k: round(v, 4) for k, v in road_val.items()},
            "building": {k: round(v, 4) for k, v in bld_val.items()},
        },
        "test": {
            "miou": round(t_conf.miou(), 4),
            "accuracy": round(t_conf.accuracy(), 4),
            "per_class_iou": {n: (None if np.isnan(v) else round(v, 4))
                              for n, v in zip(LULC_CLASS_NAMES, per_class)},
            "road": metrics_at(t_road, road_t),
            "building": metrics_at(t_bld, bld_t),
            "road_at_0.50": metrics_at(t_road, 0.5),
            "building_at_0.50": metrics_at(t_bld, 0.5),
        },
    }

    print(f"  LULC   acc {result['test']['accuracy']:.4f}  mIoU {result['test']['miou']:.4f}")
    print(f"  road      IoU {result['test']['road']['iou']:.4f} "
          f"(P{result['test']['road']['precision']:.3f}/R{result['test']['road']['recall']:.3f}) "
          f"@thr {road_t:.2f}   [was {result['test']['road_at_0.50']['iou']:.4f} @0.50]")
    print(f"  building  IoU {result['test']['building']['iou']:.4f} "
          f"(P{result['test']['building']['precision']:.3f}/R{result['test']['building']['recall']:.3f}) "
          f"@thr {bld_t:.2f}   [was {result['test']['building_at_0.50']['iou']:.4f} @0.50]")
    print(f"  per-class IoU: {result['test']['per_class_iou']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
