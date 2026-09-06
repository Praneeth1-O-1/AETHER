"""AETHER — Training entrypoint.

Trains encoders -> CrossModalAlphaFusion -> decoder -> {lulc, road, building}
jointly, end-to-end, on the `strict` tile archive. All three heads share one
trunk, so a single combined loss is backpropagated per batch -- there is no
per-head training stage. The alpha maps are never supervised directly (there
is no ground-truth fusion ratio); they emerge from backprop through that
combined loss.

Usage::

    python train.py --dataset-root data/strict

Design notes specific to this archive:

- **The split is by geographic location, year-blind.** See `data.dataset`.
  Splitting on the `_rXXX_cYYY` tile suffix -- as an earlier revision did --
  puts every AOI in every split and makes validation meaningless.
- **Every loss is masked.** Unobserved LULC, unmapped road, and NaN building
  pixels contribute exactly zero rather than NaN or a false negative.
- **Loss weights are measured, not assumed.** Class weights and pos_weights
  below are derived from the train split; see `scripts/dataset_stats.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.clouds import CloudConfig, CloudInjector, CloudMaskBank, curriculum_coverage
from data.dataset import (
    DEM_CHANNELS,
    IGNORE_INDEX,
    LULC_CLASS_NAMES,
    NUM_LULC_CLASSES,
    OPTICAL_CHANNELS,
    SAR_CHANNELS,
    AETHERTileDataset,
    build_manifest,
    location_split,
)
from models.aether import AETHERModel
from models.losses import (
    alpha_entropy_penalty,
    masked_bce_dice,
    masked_cross_entropy,
    presence_masked_cross_entropy,
)
from utils.config import load_config
from utils.metrics import BinaryIoU, ConfusionMatrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Measured on the train split (see scripts/dataset_stats.py):
#   inverse-sqrt frequency weights, normalized to mean 1.
LULC_CLASS_WEIGHTS = [0.8076, 0.3797, 1.0178, 2.8616, 0.5438, 0.5369, 0.7920, 0.5228, 1.5379]
#   sqrt-damped inverse positive rate: road 3.65% of mapped px, building 1.27%.
ROAD_POS_WEIGHT = 5.14
BUILDING_POS_WEIGHT = 8.81

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AETHER model.")
    p.add_argument("--config", type=str, default="configs/model.yaml")
    p.add_argument("--dataset-root", type=str, default="data/strict")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum-steps", type=int, default=2,
                   help="Gradient accumulation; effective batch = batch_size * accum_steps.")
    p.add_argument("--num-workers", type=int, default=6)

    p.add_argument("--lr", type=float, default=3e-4,
                   help="LR for randomly-initialized modules (DEM encoder, fusion, decoder, heads).")
    p.add_argument("--backbone-lr", type=float, default=5e-5,
                   help="LR for the pretrained optical/SAR backbones.")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-frac", type=float, default=0.03,
                   help="Fraction of total steps spent linearly warming up.")
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--tasks", type=str, default="lulc,road,building",
                   help="Comma-separated heads to train. Heads left out are not built, "
                        "so their gradients never touch the shared trunk.")
    # --- Cloud injection: the reason SAR has anything to do ---------------
    p.add_argument("--cloud-prob", type=float, default=0.5,
                   help="Fraction of training samples receiving injected cloud. "
                        "0 disables and reproduces the clear-sky regime.")
    p.add_argument("--cloud-start-coverage", type=float, default=0.3,
                   help="Max cloud coverage at epoch 1; ramps to 1.0.")
    p.add_argument("--cloud-ramp-frac", type=float, default=0.5,
                   help="Fraction of training over which coverage reaches 1.0.")
    p.add_argument("--cloud-opaque-frac", type=float, default=0.6,
                   help="Share of clouded samples with validity zeroed (vs haze).")
    p.add_argument("--cloud-full-prob", type=float, default=0.15,
                   help="Share of clouded samples forced to total occlusion.")
    p.add_argument("--cloud-select-weight", type=float, default=0.5,
                   help="Weight of the 50%%-cloud validation score in checkpoint "
                        "selection. 0 selects purely on clear sky, which is how "
                        "the optical-only solution wins.")

    # --- Modality dropout: independent per modality, applied at the input --
    p.add_argument("--drop-optical", type=float, default=0.30)
    p.add_argument("--drop-sar", type=float, default=0.15)
    p.add_argument("--drop-dem", type=float, default=0.15)

    # --- Anti-collapse ----------------------------------------------------
    p.add_argument("--aux-weight", type=float, default=0.3,
                   help="Weight of the per-modality auxiliary LULC losses. "
                        "0 disables the auxiliary heads entirely.")
    p.add_argument("--alpha-entropy-weight", type=float, default=0.02,
                   help="Initial weight of the alpha entropy floor; annealed to "
                        "0 over --alpha-entropy-frac of training.")
    p.add_argument("--alpha-entropy-frac", type=float, default=0.4)
    p.add_argument("--ogm-alpha", type=float, default=0.5,
                   help="OGM-GE gradient modulation strength. 0 disables.")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)

    p.add_argument("--lulc-weight", type=float, default=1.0)
    p.add_argument("--road-weight", type=float, default=0.5)
    p.add_argument("--building-weight", type=float, default=0.5)
    p.add_argument("--dice-weight", type=float, default=1.0)

    p.add_argument("--amp", choices=list(AMP_DTYPES), default="bf16")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--limit-train", type=int, default=0, help="Debug: cap train tiles.")
    return p.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================================================================
# Setup
# =========================================================================


def build_loaders(args) -> tuple[DataLoader, dict[str, DataLoader]]:
    """One training loader plus a named validation loader per cloud condition.

    Validation is run at several occlusion levels rather than once in clear
    sky. A single clear-sky number cannot distinguish a model that degrades
    gracefully from one that collapses the moment optical is unavailable --
    which is the entire distinction this project exists to make.
    """
    records = build_manifest(Path(args.dataset_root))
    train_rec, val_rec, test_rec = location_split(records, args.val_frac, args.test_frac)
    # Harvest cloud masks from the FULL train split before --limit-train slices
    # it: the bank is cached to disk, and a debug run must not leave a
    # 14-mask cache behind for the next real run to silently reuse.
    bank_rec = train_rec
    if args.limit_train:
        train_rec = train_rec[: args.limit_train]

    n_loc = lambda rs: len({r.location for r in rs})  # noqa: E731
    logger.info(
        f"Split by location -- train {len(train_rec)} tiles / {n_loc(train_rec)} loc | "
        f"val {len(val_rec)} / {n_loc(val_rec)} | test {len(test_rec)} / {n_loc(test_rec)}"
    )

    # Cloud shapes are harvested from TRAIN locations only. Replaying a val
    # location's cloud structure onto a training tile would leak that location
    # across the split exactly as surely as leaking its imagery would.
    bank = None
    if args.cloud_prob > 0:
        bank = CloudMaskBank.build(
            bank_rec, cache=Path(args.dataset_root).parent / "cloud_masks.npz",
        )

    train_cloud = CloudInjector(
        CloudConfig(
            prob=args.cloud_prob,
            max_coverage=args.cloud_start_coverage,
            opaque_frac=args.cloud_opaque_frac,
            full_occlusion_prob=args.cloud_full_prob,
        ),
        bank,
    ) if args.cloud_prob > 0 else None

    common = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        AETHERTileDataset(
            train_rec, augment=True, cloud_injector=train_cloud,
            drop_probs=(args.drop_optical, args.drop_sar, args.drop_dem),
        ),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        # NOT persistent: the cloud curriculum mutates the injector between
        # epochs, and persistent workers hold a stale copy of the dataset.
        persistent_workers=False, **common,
    )

    def val_loader(coverage: float | None) -> DataLoader:
        injector = None
        if coverage is not None:
            injector = CloudInjector(
                CloudConfig(prob=1.0, fixed_coverage=coverage, opaque_frac=1.0,
                            deterministic=True, seed=1234),
                bank,
            )
        return DataLoader(
            AETHERTileDataset(val_rec, augment=False, cloud_injector=injector),
            batch_size=args.batch_size, shuffle=False, drop_last=False,
            persistent_workers=args.num_workers > 0, **common,
        )

    val_loaders = {"clear": val_loader(None)}
    if args.cloud_prob > 0:
        val_loaders["cloud50"] = val_loader(0.5)
        val_loaders["cloud100"] = val_loader(1.0)
    return train_loader, val_loaders


def build_model(args, device: torch.device) -> AETHERModel:
    cfg = load_config(args.config)
    cfg.model.optical_encoder.in_channels = OPTICAL_CHANNELS
    cfg.model.sar_encoder.in_channels = SAR_CHANNELS
    cfg.model.dem_encoder.in_channels = DEM_CHANNELS
    cfg.model.optical_encoder.pretrained = args.pretrained
    cfg.model.sar_encoder.pretrained = args.pretrained
    cfg.model.task_heads.lulc.num_classes = NUM_LULC_CLASSES
    cfg.model.use_aux_heads = args.aux_weight > 0

    # Drop unselected heads entirely rather than zero-weighting them: an unbuilt
    # head contributes no gradient to the shared trunk at all, which is the point
    # of training a single task.
    active = {t.strip() for t in args.tasks.split(",") if t.strip()}
    unknown = active - set(cfg.model.task_heads)
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}; available: {set(cfg.model.task_heads)}")
    for name in list(cfg.model.task_heads):
        if name not in active:
            del cfg.model.task_heads[name]
    logger.info(f"Active task heads: {sorted(active)}")

    model = AETHERModel.build_from_dict(cfg.model).to(device, memory_format=torch.channels_last)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total:,} parameters | inputs O{OPTICAL_CHANNELS} S{SAR_CHANNELS} D{DEM_CHANNELS} "
                f"| skips {'ON' if model.use_skips else 'off'} "
                f"| decoder out {cfg.model.decoder.out_channels}")
    # Stash the resolved config so the checkpoint can rebuild itself exactly.
    model._build_cfg = json.loads(json.dumps(cfg.model))
    return model


def build_optimizer(model: AETHERModel, args) -> AdamW:
    """Pretrained backbones get a much lower LR than the fresh modules, so
    fine-tuning does not wash out ImageNet features before the randomly
    initialized fusion/decoder/heads have caught up."""
    backbone = list(model.optical_encoder.parameters()) + list(model.sar_encoder.parameters())
    backbone_ids = {id(p) for p in backbone}
    fresh = [p for p in model.parameters() if id(p) not in backbone_ids]
    return AdamW(
        [{"params": backbone, "lr": args.backbone_lr},
         {"params": fresh, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, total_steps: int, warmup_frac: float):
    """Linear warmup then cosine decay to 1% of base LR, stepped per optimizer step."""
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================
# Loss
# =========================================================================


def compute_losses(outputs, batch, criterion_state, args):
    """Combined masked loss over whichever heads are active on the shared trunk."""
    zero = torch.zeros((), device=batch["lulc"].device)

    loss_lulc = zero if "lulc" not in outputs else masked_cross_entropy(
        outputs["lulc"], batch["lulc"],
        weight=criterion_state["lulc_weights"], ignore_index=IGNORE_INDEX,
    )
    loss_road = zero if "road" not in outputs else masked_bce_dice(
        outputs["road"], batch["road"], batch["road_mask"],
        pos_weight=criterion_state["road_pos"], dice_weight=args.dice_weight,
    )
    loss_building = zero if "building" not in outputs else masked_bce_dice(
        outputs["building"], batch["building"], batch["building_mask"],
        pos_weight=criterion_state["building_pos"], dice_weight=args.dice_weight,
    )
    total = (
        args.lulc_weight * loss_lulc
        + args.road_weight * loss_road
        + args.building_weight * loss_building
    )

    # --- Auxiliary unimodal supervision -------------------------------
    # Each encoder must classify from its own features alone. Without this,
    # nothing stops the fusion's shared neurons from entangling a weak
    # modality's features with a strong one's and masking them out entirely --
    # the documented mechanism of modality collapse, and the measured state of
    # this model (ablating SAR costs 0.0026 mIoU).
    aux: dict[str, torch.Tensor] = {}
    if args.aux_weight > 0 and "presence" in outputs:
        presence = outputs["presence"]
        for i, name in enumerate(("optical", "sar", "dem")):
            key = f"aux_{name}"
            if key in outputs:
                aux[name] = presence_masked_cross_entropy(
                    outputs[key], batch["lulc"], presence[:, i],
                    weight=criterion_state["lulc_weights"], ignore_index=IGNORE_INDEX,
                )
                total = total + args.aux_weight * aux[name]

    # --- Alpha entropy floor (annealed) -------------------------------
    if criterion_state.get("alpha_entropy_weight", 0.0) > 0 and "alpha_maps" in outputs:
        total = total + criterion_state["alpha_entropy_weight"] * alpha_entropy_penalty(
            outputs["alpha_maps"], outputs["presence"],
        )

    return total, loss_lulc, loss_road, loss_building, aux


def modulate_encoder_grads(model, aux_losses: dict, ema: dict, alpha: float,
                           momentum: float = 0.9) -> dict[str, float]:
    """OGM-GE style gradient modulation, driven by the auxiliary head losses.

    The imbalance this addresses is not a bug in the optimizer: with optical
    available and sufficient on 95% of tiles, fitting optical really is the
    steepest descent direction, so optical's encoder converges first and the
    fusion learns to lean on it. Once that happens the SAR encoder stops
    receiving useful gradient and can never catch up -- the collapse is
    self-reinforcing.

    So damp the *over-performing* modality's gradient rather than amplifying
    the weak one (amplification destabilizes an encoder that is already poorly
    conditioned). A modality doing better than its peers gets a coefficient
    below 1; a modality doing worse is left untouched at 1.

    Ratios are smoothed with an EMA because per-batch aux losses are noisy
    enough that raw ratios would oscillate the coefficients every step.
    """
    if alpha <= 0 or len(aux_losses) < 2:
        return {}

    encoders = {"optical": model.optical_encoder, "sar": model.sar_encoder,
                "dem": model.dem_encoder}
    # exp(-loss) is a bounded, monotone "how well is this modality doing"
    # score; a raw reciprocal explodes as a loss approaches zero.
    scores = {k: math.exp(-min(v, 20.0)) for k, v in aux_losses.items()}

    coeffs: dict[str, float] = {}
    for name, score in scores.items():
        others = [s for k, s in scores.items() if k != name]
        rho = score / max(sum(others) / len(others), 1e-8)
        ema[name] = momentum * ema.get(name, rho) + (1 - momentum) * rho
        # Only damp; never boost. tanh keeps the coefficient in (0, 1].
        coeffs[name] = 1.0 - math.tanh(alpha * max(ema[name] - 1.0, 0.0))

    for name, coeff in coeffs.items():
        enc = encoders.get(name)
        if enc is None or coeff >= 0.999:
            continue
        for p in enc.parameters():
            if p.grad is not None:
                p.grad.mul_(coeff)
    return coeffs


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        v = v.to(device, non_blocking=True)
        if v.dim() == 4:
            v = v.contiguous(memory_format=torch.channels_last)
        out[k] = v
    return out


# =========================================================================
# Train / eval
# =========================================================================


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion_state,
                    args, device, amp_dtype, epoch) -> dict:
    model.train()
    sums = dict(loss=0.0, lulc=0.0, road=0.0, building=0.0,
                aux_optical=0.0, aux_sar=0.0, aux_dem=0.0)
    alpha_sum = torch.zeros(3, device=device)
    ogm_ema: dict[str, float] = criterion_state.setdefault("ogm_ema", {})
    coeffs: dict[str, float] = {}
    n_batches = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, raw in enumerate(loader):
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(batch["optical"], batch["sar"], batch["dem"],
                            presence=batch["presence"])
        loss, l_lulc, l_road, l_bld, aux = compute_losses(outputs, batch, criterion_state, args)

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at epoch {epoch} step {step}; skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / args.accum_steps).backward()

        if (step + 1) % args.accum_steps == 0:
            scaler.unscale_(optimizer)
            # Modulate before clipping, so the clip sees the gradients that will
            # actually be applied.
            coeffs = modulate_encoder_grads(
                model, {k: v.item() for k, v in aux.items()}, ogm_ema, args.ogm_alpha,
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        sums["loss"] += loss.item()
        sums["lulc"] += l_lulc.item()
        sums["road"] += l_road.item()
        sums["building"] += l_bld.item()
        for name, value in aux.items():
            sums[f"aux_{name}"] += value.item()
        alpha_sum += outputs["alpha_maps"].detach().float().mean(dim=(0, 2, 3))
        n_batches += 1

        if step % 100 == 0:
            rate = (step + 1) / (time.time() - started)
            aux_str = (" aux[" + "/".join(f"{aux[k].item():.2f}" for k in
                                          ("optical", "sar", "dem") if k in aux) + "]") if aux else ""
            ogm_str = (" k[" + "/".join(f"{coeffs.get(k, 1.0):.2f}" for k in
                                        ("optical", "sar", "dem")) + "]") if coeffs else ""
            logger.info(
                f"  e{epoch} [{step + 1}/{len(loader)}] loss {loss.item():.4f} "
                f"(lulc {l_lulc.item():.3f} road {l_road.item():.3f} bld {l_bld.item():.3f})"
                f"{aux_str}{ogm_str} lr {scheduler.get_last_lr()[1]:.2e} {rate:.2f} it/s"
            )

    n = max(n_batches, 1)
    out = {k: v / n for k, v in sums.items()}
    out["alpha"] = (alpha_sum / n).tolist()
    out["ogm"] = coeffs
    out["secs"] = time.time() - started
    return out


@torch.no_grad()
def evaluate(model, loader, criterion_state, args, device, amp_dtype) -> dict:
    model.eval()
    sums = dict(loss=0.0, lulc=0.0, road=0.0, building=0.0)
    alpha_sum = torch.zeros(3, device=device)
    conf = ConfusionMatrix(NUM_LULC_CLASSES, device=device)
    road_iou = BinaryIoU(device=device)
    bld_iou = BinaryIoU(device=device)
    n_batches = 0

    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(batch["optical"], batch["sar"], batch["dem"],
                            presence=batch["presence"])
        loss, l_lulc, l_road, l_bld, _ = compute_losses(outputs, batch, criterion_state, args)

        sums["loss"] += loss.item()
        sums["lulc"] += l_lulc.item()
        sums["road"] += l_road.item()
        sums["building"] += l_bld.item()
        alpha_sum += outputs["alpha_maps"].float().mean(dim=(0, 2, 3))
        n_batches += 1

        if "lulc" in outputs:
            conf.update(outputs["lulc"].float().argmax(dim=1), batch["lulc"],
                        batch["lulc"] != IGNORE_INDEX)
        if "road" in outputs:
            road_iou.update(outputs["road"], batch["road"], batch["road_mask"])
        if "building" in outputs:
            bld_iou.update(outputs["building"], batch["building"], batch["building_mask"])

    n = max(n_batches, 1)
    metrics = {k: v / n for k, v in sums.items()}
    metrics.update(
        acc=conf.accuracy(),
        miou=conf.miou(),
        per_class_iou=[None if math.isnan(v) else round(v, 4)
                       for v in conf.per_class_iou().tolist()],
        road_iou=road_iou.iou(), road_f1=road_iou.f1(),
        road_p=road_iou.precision(), road_r=road_iou.recall(),
        building_iou=bld_iou.iou(), building_f1=bld_iou.f1(),
        building_p=bld_iou.precision(), building_r=bld_iou.recall(),
        alpha=(alpha_sum / n).tolist(),
    )
    # Selection score mirrors the loss weights, so the saved checkpoint is the
    # best at the objective we actually train, not just at the primary head.
    # Average only over ACTIVE heads -- otherwise dropping a head would pin its
    # term at 0 and cap the score, making checkpoints incomparable across runs.
    active = set(model.task_heads)
    terms = [(args.lulc_weight, metrics["miou"], "lulc"),
             (args.road_weight, metrics["road_iou"], "road"),
             (args.building_weight, metrics["building_iou"], "building")]
    used = [(w, v) for w, v, name in terms if name in active]
    denom = sum(w for w, _ in used) or 1.0
    metrics["score"] = sum(w * v for w, v in used) / denom
    return metrics


# =========================================================================
# Checkpointing
# =========================================================================


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, best_score, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score,
            "args": vars(args),
            "channels": {"optical": OPTICAL_CHANNELS, "sar": SAR_CHANNELS, "dem": DEM_CHANNELS},
            "num_lulc_classes": NUM_LULC_CLASSES,
            "model_cfg": getattr(model, "_build_cfg", None),
        },
        path,
    )


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    amp_dtype = AMP_DTYPES[args.amp] if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    logger.info(f"Device: {device} | AMP: {args.amp if amp_dtype else 'off'}")

    train_loader, val_loaders = build_loaders(args)
    model = build_model(args, device)
    optimizer = build_optimizer(model, args)

    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    scheduler = build_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_frac)
    # GradScaler is a no-op for bf16/off; keeping it unconditional keeps one code path.
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))

    criterion_state = {
        "lulc_weights": torch.tensor(LULC_CLASS_WEIGHTS, device=device),
        "road_pos": torch.tensor(ROAD_POS_WEIGHT, device=device),
        "building_pos": torch.tensor(BUILDING_POS_WEIGHT, device=device),
    }

    start_epoch, best_score = 1, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            logger.warning(f"Checkpoint/architecture mismatch -- missing {missing}, unexpected {unexpected}")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_score = ckpt.get("best_score", -1.0)
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch} (best score {best_score:.4f})")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        # Cloud curriculum: light haze first, total occlusion later. Starting at
        # full occlusion is pure noise to a head that has not yet learned the
        # easy clear-sky case.
        if train_loader.dataset.cloud_injector is not None:
            cov = curriculum_coverage(epoch, args.epochs, start=args.cloud_start_coverage,
                                      end=1.0, ramp_frac=args.cloud_ramp_frac)
            train_loader.dataset.cloud_injector.config.max_coverage = cov
        else:
            cov = 0.0

        # Anneal the alpha entropy floor to zero. It exists to keep every branch
        # alive through the early window where collapse happens; leaving it on
        # would forbid the sharply cloud-adaptive alpha that is the actual goal.
        frac = (epoch - 1) / max(1.0, args.alpha_entropy_frac * args.epochs)
        criterion_state["alpha_entropy_weight"] = (
            args.alpha_entropy_weight * max(0.0, 1.0 - frac)
        )

        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                             criterion_state, args, device, amp_dtype, epoch)

        vals = {name: evaluate(model, ldr, criterion_state, args, device, amp_dtype)
                for name, ldr in val_loaders.items()}
        val = vals["clear"]

        # Select on degradation, not on the peak. Selecting on clear sky alone
        # is precisely how an optical-only solution wins -- it is the best
        # clear-sky model by construction.
        w = args.cloud_select_weight if "cloud50" in vals else 0.0
        score = (1 - w) * val["score"] + w * vals.get("cloud50", val)["score"]

        aux_str = " ".join(f"{k.split('_')[1]} {tr[k]:.3f}"
                           for k in ("aux_optical", "aux_sar", "aux_dem") if tr.get(k))
        logger.info(
            f"Epoch {epoch}/{args.epochs} ({tr['secs'] / 60:.1f} min, cloud<={cov:.2f}) | "
            f"train {tr['loss']:.4f} (lulc {tr['lulc']:.3f} road {tr['road']:.3f} bld {tr['building']:.3f}) | "
            f"val {val['loss']:.4f} acc {val['acc']:.3f} mIoU {val['miou']:.3f} "
            f"road_IoU {val['road_iou']:.3f} (P{val['road_p']:.2f}/R{val['road_r']:.2f}) "
            f"bld_IoU {val['building_iou']:.3f} (P{val['building_p']:.2f}/R{val['building_r']:.2f}) "
            f"score {score:.4f} | alpha[O,S,D] {[round(a, 3) for a in val['alpha']]}"
        )
        if aux_str:
            logger.info(f"  aux loss: {aux_str} | ogm k: "
                        f"{ {k: round(v, 3) for k, v in tr['ogm'].items()} }")
        for name in ("cloud50", "cloud100"):
            if name in vals:
                v = vals[name]
                logger.info(
                    f"  {name:9s} acc {v['acc']:.3f} mIoU {v['miou']:.3f} "
                    f"alpha[O,S,D] {[round(a, 3) for a in v['alpha']]}"
                )
        logger.info(f"  per-class IoU: {dict(zip(LULC_CLASS_NAMES, val['per_class_iou']))}")

        row = {
            "epoch": epoch, "cloud_max_cov": round(cov, 3),
            "train_loss": round(tr["loss"], 5),
            "train_lulc": round(tr["lulc"], 5), "train_road": round(tr["road"], 5),
            "train_building": round(tr["building"], 5),
            "aux_optical": round(tr.get("aux_optical", 0.0), 5),
            "aux_sar": round(tr.get("aux_sar", 0.0), 5),
            "aux_dem": round(tr.get("aux_dem", 0.0), 5),
            "val_loss": round(val["loss"], 5), "val_acc": round(val["acc"], 5),
            "val_miou": round(val["miou"], 5), "val_road_iou": round(val["road_iou"], 5),
            "val_building_iou": round(val["building_iou"], 5), "score": round(score, 5),
            "alpha_o": round(val["alpha"][0], 4), "alpha_s": round(val["alpha"][1], 4),
            "alpha_d": round(val["alpha"][2], 4),
            "lr": scheduler.get_last_lr()[1], "minutes": round(tr["secs"] / 60, 2),
        }
        for name in ("cloud50", "cloud100"):
            v = vals.get(name)
            row[f"{name}_acc"] = round(v["acc"], 5) if v else ""
            row[f"{name}_miou"] = round(v["miou"], 5) if v else ""
            # alpha_sar under cloud is the primary acceptance signal: if it does
            # not rise with occlusion, the fusion is still not adaptive however
            # good the aggregate looks.
            row[f"{name}_alpha_s"] = round(v["alpha"][1], 4) if v else ""
        append_history(ckpt_dir / "history.csv", row)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_score, args)
        if score > best_score:
            best_score = score
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_score, args)
            logger.info(f"  New best score {best_score:.4f} -> {ckpt_dir / 'best.pt'}")

    logger.info(f"Done. Best val score {best_score:.4f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
