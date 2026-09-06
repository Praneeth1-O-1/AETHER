"""AETHER — Masked, imbalance-aware losses.

Every loss here is *masked*: it consumes an explicit per-pixel validity mask
and contributes exactly zero for pixels the archive never observed. That is
not a nicety on this dataset -- it is what keeps training alive and honest:

- 352 tiles have no valid LULC pixel at all (Sentinel-2 was never acquired,
  so Dynamic World had nothing to derive from). ``CrossEntropyLoss``'s default
  ``reduction="mean"`` divides by the non-ignored pixel count, so those tiles
  evaluate 0/0 -> NaN, and a single backward pass poisons every weight in the
  model. Summing and dividing by a clamped count yields 0.0 instead.

- 1,684 tiles have ``road`` flagged missing in meta.json and stored as
  all-zeros. That is absence of *mapping*, not absence of road. Backpropagating
  it as a negative teaches the model that unmapped regions are roadless.

- ``building_presence`` carries ~260k NaN pixels archive-wide.

Class imbalance is handled structurally rather than by brute-force weighting.
Road covers 3.7% of mapped pixels and building 1.3%, so a pure inverse-frequency
``pos_weight`` would be 26x and 78x -- enough to make BCE wildly over-predict.
Instead each binary task pairs a *square-root-damped* ``pos_weight`` with a soft
Dice term, which is scale-invariant by construction and optimizes the overlap
these tasks are actually scored on.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Class-weighted cross-entropy that returns 0.0 when nothing is observed.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits ``(B, C, H, W)``.
    target : torch.Tensor
        Class indices ``(B, H, W)``, with ``ignore_index`` for unobserved pixels.
    weight : torch.Tensor, optional
        Per-class weights ``(C,)``.
    ignore_index : int
        Target value marking an unobserved pixel.
    """
    # float() so the reduction is exact under autocast; sum-then-normalize is
    # what makes an all-ignored tile score 0.0 rather than NaN.
    loss = F.cross_entropy(
        logits.float(), target, weight=weight,
        ignore_index=ignore_index, reduction="sum",
    )
    n_valid = (target != ignore_index).sum()
    return loss / n_valid.clamp(min=1)


def presence_masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Cross-entropy for an auxiliary unimodal head, skipping absent samples.

    An auxiliary head exists to force its encoder to be independently
    predictive. Scoring it on a sample whose modality was dropped or fully
    occluded asks it to predict land cover from an all-zero tensor, which is
    not a hard case but an impossible one -- the gradient is pure noise and it
    teaches the encoder to emit a class prior instead of reading its input.

    Parameters
    ----------
    logits : torch.Tensor
        ``(B, C, H, W)``.
    target : torch.Tensor
        ``(B, H, W)`` class indices, ``ignore_index`` where unobserved.
    present : torch.Tensor
        ``(B,)`` in {0, 1} -- 1 where this modality is genuinely available.
    """
    per_px = F.cross_entropy(
        logits.float(), target, weight=weight,
        ignore_index=ignore_index, reduction="none",
    )  # (B, H, W)
    keep = (target != ignore_index).float() * present.float().view(-1, 1, 1)
    return (per_px * keep).sum() / keep.sum().clamp(min=1)


def alpha_entropy_penalty(
    alpha_maps: torch.Tensor, presence: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Negative mean entropy of the fusion weights, over PRESENT modalities.

    Returned as a penalty (lower is more uniform), so adding it with a positive
    weight pushes alpha away from collapse.

    This is a scaffold, not an objective. Collapse onto optical happens early,
    while SAR is still poorly fit and genuinely the worse bet -- once collapsed,
    the SAR encoder stops receiving useful gradient and can never catch up. A
    small entropy floor keeps every branch alive through that window. It must
    be annealed away afterwards: the *goal* is a sharply adaptive alpha that
    peaks on optical in clear sky and on SAR under cloud, and a standing
    uniformity prior would forbid exactly that.

    Normalized by log(n_present) so the penalty does not silently change scale
    when a modality is dropped.
    """
    alpha = alpha_maps.float().clamp_min(eps)
    mask = (presence.float() > 0.5).view(-1, 3, 1, 1)
    n_present = mask.sum(dim=1, keepdim=True).clamp(min=1).float()

    entropy = -(alpha * alpha.log() * mask).sum(dim=1, keepdim=True)
    max_entropy = n_present.log().clamp(min=eps)
    normalized = (entropy / max_entropy)[n_present.expand_as(entropy) > 1.5]
    if normalized.numel() == 0:
        return alpha_maps.new_zeros(())
    return -normalized.mean()


def masked_bce_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    dice_weight: float = 1.0,
    eps: float = 1.0,
) -> torch.Tensor:
    """Masked BCE + soft Dice for an imbalanced binary/soft-label head.

    ``target`` may be continuous -- ``building_presence`` is a per-pixel
    coverage fraction in [0, 1], and both terms treat it as a soft label
    rather than thresholding it, which preserves partial building-edge
    coverage instead of collapsing it.

    Parameters
    ----------
    logits, target, mask : torch.Tensor
        All ``(B, 1, H, W)``. ``mask`` is 1 where the label is a real
        observation, 0 where it was never acquired.
    pos_weight : torch.Tensor
        Scalar tensor scaling the positive term of the BCE.
    dice_weight : float
        Relative weight of the Dice term against the BCE term.
    eps : float
        Dice smoothing, in pixel units. Also defines the loss for an
        all-negative tile the model correctly predicts empty.
    """
    logits = logits.float()
    target = target.float()
    mask = mask.float()

    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none",
    )
    bce = (bce * mask).sum() / mask.sum().clamp(min=1)

    # Per-sample Dice, so one building-dense tile cannot dominate the batch.
    probs = torch.sigmoid(logits) * mask
    truth = target * mask
    dims = (1, 2, 3)
    intersection = (probs * truth).sum(dims)
    cardinality = probs.sum(dims) + truth.sum(dims)
    dice = 1.0 - (2.0 * intersection + eps) / (cardinality + eps)

    observed = (mask.sum(dims) > 0).float()
    dice = (dice * observed).sum() / observed.sum().clamp(min=1)

    return bce + dice_weight * dice
