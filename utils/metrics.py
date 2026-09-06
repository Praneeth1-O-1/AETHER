"""AETHER — Segmentation metric accumulators.

All metrics here accumulate raw counts across the whole evaluation set and
reduce once at the end, rather than averaging a per-batch score. Per-batch
averaging is biased for IoU: a batch holding a single road pixel that the
model happens to find scores 1.0 and is weighted equally against a batch of
dense urban road network. Global counters also make the numbers independent
of batch size, so runs stay comparable.

Counters live on the GPU and are only pulled to the host at reduce time,
which keeps a per-batch device sync out of the eval loop.
"""

from __future__ import annotations

import torch


class ConfusionMatrix:
    """Accumulates a multi-class confusion matrix for mIoU and accuracy."""

    def __init__(self, num_classes: int, device: torch.device | str = "cpu") -> None:
        self.num_classes = num_classes
        self.mat = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        """Add one batch.

        Parameters
        ----------
        pred, target : torch.Tensor
            Class indices, any shape, same shape as each other.
        valid : torch.Tensor
            Boolean mask selecting observed pixels.
        """
        p = pred[valid].reshape(-1)
        t = target[valid].reshape(-1)
        if p.numel() == 0:
            return
        self.mat += torch.bincount(
            t * self.num_classes + p, minlength=self.num_classes ** 2,
        )

    def _matrix(self) -> torch.Tensor:
        return self.mat.reshape(self.num_classes, self.num_classes).float()

    def per_class_iou(self) -> torch.Tensor:
        """IoU per class; NaN for classes absent from both prediction and truth."""
        conf = self._matrix()
        intersection = torch.diag(conf)
        union = conf.sum(0) + conf.sum(1) - intersection
        return torch.where(union > 0, intersection / union, torch.full_like(union, float("nan")))

    def miou(self) -> float:
        """Mean IoU over classes that actually appear in the evaluation set."""
        iou = self.per_class_iou()
        present = ~torch.isnan(iou)
        return iou[present].mean().item() if present.any() else 0.0

    def accuracy(self) -> float:
        conf = self._matrix()
        total = conf.sum()
        return (torch.diag(conf).sum() / total).item() if total > 0 else 0.0

    # -- per-class precision / recall / F1 -------------------------------
    # Rows of the matrix are truth, columns are prediction, so a column sum is
    # the predicted count (precision denominator) and a row sum is the true
    # count (recall denominator).

    def support(self) -> torch.Tensor:
        return self._matrix().sum(1)

    def per_class_precision(self) -> torch.Tensor:
        conf = self._matrix()
        predicted = conf.sum(0)
        return torch.where(predicted > 0, torch.diag(conf) / predicted,
                           torch.full_like(predicted, float("nan")))

    def per_class_recall(self) -> torch.Tensor:
        conf = self._matrix()
        actual = conf.sum(1)
        return torch.where(actual > 0, torch.diag(conf) / actual,
                           torch.full_like(actual, float("nan")))

    def per_class_f1(self) -> torch.Tensor:
        p, r = self.per_class_precision(), self.per_class_recall()
        denom = p + r
        return torch.where(denom > 0, 2 * p * r / denom, torch.zeros_like(denom))

    @staticmethod
    def _nanmean(x: torch.Tensor) -> float:
        present = ~torch.isnan(x)
        return x[present].mean().item() if present.any() else 0.0

    def macro(self) -> dict[str, float]:
        """Unweighted mean over classes present in the data -- every class counts
        equally, so rare classes are not drowned out by frequent ones."""
        return {
            "precision": self._nanmean(self.per_class_precision()),
            "recall": self._nanmean(self.per_class_recall()),
            "f1": self._nanmean(self.per_class_f1()),
            "iou": self.miou(),
        }

    def weighted(self) -> dict[str, float]:
        """Support-weighted mean -- reflects overall pixel-level performance."""
        w = self.support()
        total = w.sum()
        if total == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

        def wavg(x: torch.Tensor) -> float:
            ok = ~torch.isnan(x)
            return ((x[ok] * w[ok]).sum() / w[ok].sum()).item() if ok.any() else 0.0

        return {
            "precision": wavg(self.per_class_precision()),
            "recall": wavg(self.per_class_recall()),
            "f1": wavg(self.per_class_f1()),
            "iou": wavg(self.per_class_iou()),
        }

    def kappa(self) -> float:
        """Cohen's kappa: accuracy corrected for agreement expected by chance.
        On an imbalanced 9-class problem this is a far harder number than
        raw accuracy, which a majority-class predictor can inflate."""
        conf = self._matrix()
        total = conf.sum()
        if total == 0:
            return 0.0
        observed = torch.diag(conf).sum() / total
        expected = (conf.sum(0) * conf.sum(1)).sum() / (total * total)
        return ((observed - expected) / (1 - expected)).item() if expected < 1 else 0.0


class BinaryIoU:
    """Accumulates global intersection/union for a thresholded binary head.

    Also tracks precision/recall counts, because IoU alone hides whether a
    minority-class head is failing by over- or under-predicting -- the most
    common failure mode for road and building at 3.7% and 1.3% positive rate.
    """

    def __init__(self, threshold: float = 0.5, device: torch.device | str = "cpu",
                 label_threshold: float = 0.5) -> None:
        # The LABEL threshold must not move with the PREDICTION threshold.
        # `building_presence` is a continuous coverage fraction, so tying the two
        # together would redefine ground truth at every operating point and make
        # IoU across thresholds incomparable. `road` is already binary 0/1.
        self.threshold = threshold
        self.label_threshold = label_threshold
        self.counts = torch.zeros(3, dtype=torch.float64, device=device)  # tp, fp, fn
        self.observed = torch.zeros((), dtype=torch.float64, device=device)
        self.positives = torch.zeros((), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        keep = mask > 0.5
        pred = (torch.sigmoid(logits.float()) > self.threshold) & keep
        truth = (target > self.label_threshold) & keep
        self.counts += torch.stack([
            (pred & truth).sum(),
            (pred & ~truth & keep).sum(),
            (~pred & truth & keep).sum(),
        ]).double()
        self.observed += keep.sum().double()
        self.positives += truth.sum().double()

    @property
    def total_negatives(self) -> float:
        return (self.observed - self.positives).item()

    def _tp_fp_fn(self) -> tuple[float, float, float]:
        tp, fp, fn = self.counts.tolist()
        return tp, fp, fn

    def iou(self) -> float:
        tp, fp, fn = self._tp_fp_fn()
        denom = tp + fp + fn
        return tp / denom if denom > 0 else 0.0

    def f1(self) -> float:
        tp, fp, fn = self._tp_fp_fn()
        denom = 2 * tp + fp + fn
        return 2 * tp / denom if denom > 0 else 0.0

    def precision(self) -> float:
        tp, fp, _ = self._tp_fp_fn()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self) -> float:
        tp, _, fn = self._tp_fp_fn()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def all_metrics(self) -> dict[str, float]:
        tp, fp, fn = self._tp_fp_fn()
        tn = float(self.total_negatives - fp)
        total = tp + fp + fn + tn
        return {
            "iou": self.iou(), "f1": self.f1(),
            "precision": self.precision(), "recall": self.recall(),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "positive_rate": (tp + fn) / total if total > 0 else 0.0,
        }
