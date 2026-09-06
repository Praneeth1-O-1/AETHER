"""Full metric report for every head, across every dataset scenario.

Reports accuracy / precision / recall / F1 / IoU for LULC (per class, macro and
support-weighted) and for the road and building heads, then repeats the whole
report over subsets of the archive defined by what data each tile actually has.

Why the scenario breakdown matters more than a single headline number:

- The archive is heterogeneous. Some tiles have no Sentinel-2, some no
  Sentinel-1, and ~26% have `road` flagged missing in meta.json (absence of OSM
  *mapping*, not absence of road). A single aggregate hides that the model
  behaves completely differently in each regime.
- LULC labels are generated from the anchor Sentinel-2 granule, so wherever
  optical is absent the LULC label is absent too. The optical-missing scenario
  therefore reports road/building only -- LULC is not merely poor there, it is
  unmeasurable.

Binary heads are reported at both the val-tuned threshold and the naive 0.5,
plus a threshold-free average precision, so the numbers cannot be read as an
artifact of one arbitrary cutoff.

Usage::

    python scripts/evaluate_full.py --checkpoint checkpoints/best.pt
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
from utils.metrics import BinaryIoU, ConfusionMatrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--road-threshold", type=float, default=0.60)
    p.add_argument("--building-threshold", type=float, default=0.71)
    p.add_argument("--output", type=str, default="outputs/full_evaluation.json")
    return p.parse_args()


class HeadStats:
    """Binary head at the tuned threshold, at 0.5, and threshold-free (AP)."""

    GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)

    def __init__(self, tuned: float, device: torch.device) -> None:
        self.tuned_threshold = float(tuned)
        self.tuned = BinaryIoU(tuned, device=device)
        self.half = BinaryIoU(0.5, device=device)
        self.grid = torch.tensor(self.GRID, device=device, dtype=torch.float32).view(-1, 1)
        self.pr = torch.zeros(3, len(self.GRID), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits, target, mask) -> None:
        self.tuned.update(logits, target, mask)
        self.half.update(logits, target, mask)
        keep = (mask > 0.5).reshape(-1)
        if not keep.any():
            return
        probs = torch.sigmoid(logits.float()).reshape(-1)[keep]
        truth = target.reshape(-1)[keep] > 0.5   # fixed label definition
        pred = probs.unsqueeze(0) >= self.grid
        t = truth.unsqueeze(0)
        self.pr[0] += (pred & t).sum(1).double()
        self.pr[1] += (pred & ~t).sum(1).double()
        self.pr[2] += (~pred & t).sum(1).double()

    def average_precision(self) -> float:
        """Area under the precision-recall curve, by the rectangle rule over the
        threshold grid. Threshold-free, so it separates ranking quality from
        the choice of operating point."""
        tp, fp, fn = self.pr.cpu().numpy()
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        order = np.argsort(recall)
        r, p = recall[order], precision[order]
        return float(np.sum(np.diff(np.concatenate([[0.0], r])) * p))

    def report(self) -> dict:
        return {
            "tuned_threshold": self.tuned_threshold,
            "at_tuned": {k: round(v, 4) for k, v in self.tuned.all_metrics().items()},
            "at_0.50": {k: round(v, 4) for k, v in self.half.all_metrics().items()},
            "average_precision": round(self.average_precision(), 4),
        }


@torch.no_grad()
def evaluate(model, records, args, device) -> dict | None:
    if not records:
        return None
    loader = DataLoader(
        AETHERTileDataset(records, augment=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    conf = ConfusionMatrix(NUM_LULC_CLASSES, device=device)
    road = HeadStats(args.road_threshold, device)
    building = HeadStats(args.building_threshold, device)
    alpha_sum = torch.zeros(3, device=device)
    n = 0

    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch["optical"], batch["sar"], batch["dem"],
                        presence=batch.get("presence"))
        if "lulc" in out:
            conf.update(out["lulc"].float().argmax(1), batch["lulc"],
                        batch["lulc"] != IGNORE_INDEX)
        if "road" in out:
            road.update(out["road"], batch["road"], batch["road_mask"])
        if "building" in out:
            building.update(out["building"], batch["building"], batch["building_mask"])
        alpha_sum += out["alpha_maps"].float().mean(dim=(0, 2, 3))
        n += 1

    nan = lambda x: None if np.isnan(x) else round(float(x), 4)  # noqa: E731
    per_class = {
        name: {
            "precision": nan(p), "recall": nan(r), "f1": nan(f),
            "iou": nan(i), "support_px": int(s),
        }
        for name, p, r, f, i, s in zip(
            LULC_CLASS_NAMES,
            conf.per_class_precision().tolist(), conf.per_class_recall().tolist(),
            conf.per_class_f1().tolist(), conf.per_class_iou().tolist(),
            conf.support().tolist(),
        )
    }
    return {
        "n_tiles": len(records),
        "n_locations": len({r.location for r in records}),
        "lulc": {
            "labelled_pixels": int(conf.support().sum().item()),
            "accuracy": round(conf.accuracy(), 4),
            "kappa": round(conf.kappa(), 4),
            "macro": {k: round(v, 4) for k, v in conf.macro().items()},
            "weighted": {k: round(v, 4) for k, v in conf.weighted().items()},
            "per_class": per_class,
        },
        "road": road.report(),
        "building": building.report(),
        "alpha_mean": [round(a, 4) for a in (alpha_sum / max(n, 1)).tolist()],
    }


def scenarios(records) -> dict[str, list]:
    """Partition the split by what each tile actually contains."""
    return {
        "ALL": records,
        "optical present": [r for r in records if r.has_optical],
        "optical MISSING": [r for r in records if not r.has_optical],
        "SAR present": [r for r in records if r.has_sar],
        "SAR MISSING": [r for r in records if not r.has_sar],
        "road mapped (OSM)": [r for r in records if r.has_road],
        "road UNMAPPED": [r for r in records if not r.has_road],
        "LULC labelled": [r for r in records if r.has_lulc],
        "fully complete": [r for r in records
                           if r.has_optical and r.has_sar and r.has_lulc and r.has_road],
    }


def print_lulc(tag: str, rep: dict) -> None:
    lulc = rep["lulc"]
    if lulc["labelled_pixels"] == 0:
        print(f"\n### LULC -- {tag}: NO LABELLED PIXELS (label is derived from optical)")
        return
    print(f"\n### LULC -- {tag}  ({rep['n_tiles']} tiles, {lulc['labelled_pixels']:,} labelled px)")
    print(f"  overall accuracy {lulc['accuracy']:.4f}   Cohen's kappa {lulc['kappa']:.4f}")
    print(f"  {'class':<13}{'precision':>10}{'recall':>9}{'F1':>9}{'IoU':>9}{'support':>14}")
    for name, m in lulc["per_class"].items():
        f = lambda v: "   n/a" if v is None else f"{v:9.4f}"  # noqa: E731
        print(f"  {name:<13}{f(m['precision'])}{f(m['recall'])}{f(m['f1'])}{f(m['iou'])}"
              f"{m['support_px']:>14,}")
    for kind in ("macro", "weighted"):
        m = lulc[kind]
        print(f"  {kind + ' avg':<13}{m['precision']:>10.4f}{m['recall']:>9.4f}"
              f"{m['f1']:>9.4f}{m['iou']:>9.4f}")


def print_binary(tag: str, rep: dict) -> None:
    print(f"\n### Road / Building -- {tag}")
    print(f"  {'head':<10}{'thr':>6}{'precision':>11}{'recall':>9}{'F1':>9}{'IoU':>9}"
          f"{'accuracy':>10}{'specif.':>9}{'AP':>8}{'pos rate':>10}")
    for head in ("road", "building"):
        h = rep[head]
        for label, key in ((f"{h['tuned_threshold']:.2f}", "at_tuned"), ("0.50", "at_0.50")):
            m = h[key]
            ap = f"{h['average_precision']:8.4f}" if key == "at_tuned" else " " * 8
            print(f"  {head if key == 'at_tuned' else '':<10}{label:>6}{m['precision']:>11.4f}"
                  f"{m['recall']:>9.4f}{m['f1']:>9.4f}{m['iou']:>9.4f}{m['accuracy']:>10.4f}"
                  f"{m['specificity']:>9.4f}{ap}{m['positive_rate']:>10.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.config, args.checkpoint, device)

    records = build_manifest(Path(args.dataset_root))
    train, val, test = location_split(records)
    chosen = {"train": train, "val": val, "test": test}[args.split]
    print(f"\nSplit '{args.split}': {len(chosen)} tiles / "
          f"{len({r.location for r in chosen})} held-out locations")

    results = {}
    for tag, subset in scenarios(chosen).items():
        rep = evaluate(model, subset, args, device)
        if rep is None:
            print(f"\n=== {tag}: 0 tiles, skipped ===")
            continue
        results[tag] = rep
        print(f"\n{'=' * 96}\n=== SCENARIO: {tag}  "
              f"({rep['n_tiles']} tiles / {rep['n_locations']} locations)  "
              f"alpha[O,S,D] {rep['alpha_mean']}\n{'=' * 96}")
        print_lulc(tag, rep)
        print_binary(tag, rep)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "split": args.split,
                               "scenarios": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
