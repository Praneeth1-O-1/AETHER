"""Mixed-precision dtype selection."""

from __future__ import annotations

import torch


def autocast_dtype() -> torch.dtype:
    """bfloat16 where the GPU supports it natively (compute capability >= 8.0), else float16.

    A Tesla T4 (7.5) has no bfloat16 convolution kernels in cuDNN, so bf16
    autocast fails there with "unable to find an engine to execute this
    computation". torch.cuda.is_bf16_supported() is not a safe test: by default
    it counts emulation as support and returns True on such GPUs.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16
