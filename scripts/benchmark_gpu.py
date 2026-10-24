"""Compute cost of a trained AETHER checkpoint: parameters, FLOPs, latency, throughput, memory.

Accuracy numbers say what the model gets right; these say what it costs to run.
Inputs are random tensors of the checkpoint's real channel counts -- the cost of
a dense CNN does not depend on pixel values, so no dataset is needed.

Usage::

    python scripts/benchmark_gpu.py --checkpoint checkpoints/kaggle_best_epoch41.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import get_device, load_model  # noqa: E402

TILE_PX = 256
TILE_KM2 = (TILE_PX * 10 / 1000) ** 2  # 10 m pixels -> 6.5536 km^2 per tile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--output", default="outputs/gpu_benchmark.json")
    return p.parse_args()


def make_inputs(channels: dict, batch: int, device: torch.device):
    return (torch.randn(batch, channels["optical"], TILE_PX, TILE_PX, device=device),
            torch.randn(batch, channels["sar"], TILE_PX, TILE_PX, device=device),
            torch.randn(batch, channels["dem"], TILE_PX, TILE_PX, device=device),
            torch.ones(batch, 3, device=device))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_params(model) -> dict:
    groups = {
        "optical_encoder": model.optical_encoder,
        "sar_encoder": model.sar_encoder,
        "dem_encoder": model.dem_encoder,
        "fusion": model.fusion,
        "decoder": model.decoder,
        "task_heads": model.task_heads,
    }
    out = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    out["other (aux heads, optical detail branch)"] = (
        sum(p.numel() for p in model.parameters()) - sum(out.values()))
    out["total"] = sum(p.numel() for p in model.parameters())
    return out


@torch.no_grad()
def count_flops(model, channels, device) -> dict:
    """Per-tile FLOPs (multiply and add counted separately), split by top-level module."""
    inputs = make_inputs(channels, 1, device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(*inputs[:3], presence=inputs[3])
    per_module = {}
    for name, ops in counter.get_flop_counts().items():
        # Top-level children only, e.g. "AETHERModel.optical_encoder"
        if name.count(".") == 1:
            per_module[name.split(".", 1)[1]] = round(sum(ops.values()) / 1e9, 3)
    # Task heads live in a ModuleDict that is never called itself, so their
    # FLOPs are only reported one level deeper; account for them as remainder.
    total = counter.get_total_flops() / 1e9
    per_module["task_heads_and_other"] = round(total - sum(per_module.values()), 3)
    return {"gflops_per_tile": round(total, 3),
            "gflops_by_module": per_module}


@torch.no_grad()
def time_batch(model, channels, batch, device, amp_dtype, warmup, iters) -> dict:
    inputs = make_inputs(channels, batch, device)
    autocast = torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with autocast:
        for _ in range(warmup):
            model(*inputs[:3], presence=inputs[3])
        sync(device)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(*inputs[:3], presence=inputs[3])
            sync(device)
            times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    res = {
        "batch": batch,
        "latency_ms_mean": round(mean * 1e3, 2),
        "latency_ms_p50": round(statistics.median(times) * 1e3, 2),
        "latency_ms_p95": round(sorted(times)[int(0.95 * (len(times) - 1))] * 1e3, 2),
        "tiles_per_s": round(batch / mean, 1),
        "km2_per_s": round(batch / mean * TILE_KM2, 1),
    }
    if device.type == "cuda":
        res["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return res


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model = load_model(args.config, args.checkpoint, device).eval()
    channels = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["channels"]

    info = {"device": device.type, "torch": torch.__version__, "platform": platform.platform()}
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, gpu_mem_gib=round(prop.total_memory / 2**30, 1))

    result = {"system": info, "checkpoint": args.checkpoint, "channels": channels,
              "tile": f"{TILE_PX}x{TILE_PX} px @ 10 m = {TILE_KM2:.4f} km^2",
              "params": count_params(model), "flops": count_flops(model, channels, device)}

    precisions = {"fp32": None}
    if device.type == "cuda":
        precisions["fp16"] = torch.float16
    result["timing"] = {}
    for pname, dtype in precisions.items():
        rows = []
        for b in args.batch_sizes:
            try:
                rows.append(time_batch(model, channels, b, device, dtype, args.warmup, args.iters))
            except RuntimeError as e:  # out of memory at large batch
                if "out of memory" not in str(e).lower():
                    raise
                rows.append({"batch": b, "error": "out of memory"})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"{pname} {rows[-1]}")
        result["timing"][pname] = rows

    print(json.dumps({k: result[k] for k in ("system", "params", "flops")}, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
