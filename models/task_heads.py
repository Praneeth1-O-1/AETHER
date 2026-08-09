"""Task Heads — Modular prediction heads for downstream tasks.

A registry-based design that makes it trivial to add new task heads
(road extraction, building extraction, change detection) without
modifying the rest of the network.

Currently implements:
- **LULCHead**: Land Use / Land Cover pixel-wise classification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.nn as nn


# =========================================================================
# Registry
# =========================================================================

_TASK_HEAD_REGISTRY: dict[str, Type["TaskHead"]] = {}


def register_task_head(name: str):
    """Decorator to register a task head class under a string key.

    Usage::

        @register_task_head("lulc")
        class LULCHead(TaskHead):
            ...
    """

    def wrapper(cls: Type[TaskHead]) -> Type[TaskHead]:
        if name in _TASK_HEAD_REGISTRY:
            raise ValueError(
                f"Task head '{name}' is already registered "
                f"({_TASK_HEAD_REGISTRY[name].__name__})"
            )
        _TASK_HEAD_REGISTRY[name] = cls
        return cls

    return wrapper


def get_task_head(name: str, **kwargs) -> "TaskHead":
    """Instantiate a registered task head by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"lulc"``).
    **kwargs
        Constructor arguments forwarded to the head class.

    Returns
    -------
    TaskHead
        Instantiated task head module.
    """
    if name not in _TASK_HEAD_REGISTRY:
        available = ", ".join(sorted(_TASK_HEAD_REGISTRY.keys()))
        raise KeyError(
            f"Unknown task head '{name}'. Available: [{available}]"
        )
    return _TASK_HEAD_REGISTRY[name](**kwargs)


def list_task_heads() -> list[str]:
    """Return the names of all registered task heads."""
    return sorted(_TASK_HEAD_REGISTRY.keys())


# =========================================================================
# Base Class
# =========================================================================


class TaskHead(ABC, nn.Module):
    """Abstract base class for all task heads.

    Every task head receives decoded features of shape
    ``(B, in_channels, H, W)`` and produces task-specific predictions.
    """

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Produce predictions from decoder features.

        Parameters
        ----------
        features : torch.Tensor
            Decoded feature map ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Task-specific predictions.
        """
        ...


# =========================================================================
# Implemented Heads
# =========================================================================


@register_task_head("lulc")
class LULCHead(TaskHead):
    """Land Use / Land Cover classification head.

    A single 1×1 convolution mapping decoder features to per-pixel
    class logits.  No activation — outputs raw logits suitable for
    ``nn.CrossEntropyLoss``.

    Parameters
    ----------
    in_channels : int
        Number of input channels from the decoder.
    num_classes : int
        Number of LULC classes.
    """

    def __init__(self, in_channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Produce per-pixel class logits.

        Parameters
        ----------
        features : torch.Tensor
            Decoded features ``(B, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Class logits ``(B, num_classes, H, W)``.
        """
        return self.classifier(features)


# =========================================================================
# Future heads (documented stubs — uncomment and implement when ready)
# =========================================================================
#
# @register_task_head("road")
# class RoadExtractionHead(TaskHead):
#     """Binary segmentation head for road extraction."""
#     ...
#
# @register_task_head("building")
# class BuildingExtractionHead(TaskHead):
#     """Binary segmentation head for building footprint extraction."""
#     ...
#
# @register_task_head("change")
# class ChangeDetectionHead(TaskHead):
#     """Binary change detection head (requires bi-temporal input)."""
#     ...
