"""Configuration loading utilities for AETHER.

Provides a recursive DotDict for attribute-style access to nested YAML
configurations, plus a convenience loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml


class DotDict(dict):
    """Dictionary subclass that supports recursive attribute-style access.

    Nested dictionaries are recursively converted to DotDict instances upon
    initialization, ensuring attribute mutations correctly update the nested structure.

    Examples
    --------
    >>> cfg = DotDict({"model": {"fusion": {"feature_dim": 256}}})
    >>> cfg.model.fusion.feature_dim
    256
    >>> cfg.model.fusion.feature_dim = 512
    >>> cfg["model"]["fusion"]["feature_dim"]
    512
    """

    def __init__(self, mapping: dict | None = None, **kwargs: Any) -> None:
        super().__init__()
        if mapping:
            for k, v in mapping.items():
                self[k] = DotDict(v) if isinstance(v, dict) and not isinstance(v, DotDict) else v
        for k, v in kwargs.items():
            self[k] = DotDict(v) if isinstance(v, dict) and not isinstance(v, DotDict) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = DotDict(value) if isinstance(value, dict) and not isinstance(value, DotDict) else value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(
            key,
            DotDict(value) if isinstance(value, dict) and not isinstance(value, DotDict) else value,
        )

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None

    def to_dict(self) -> dict:
        """Recursively convert back to a plain dictionary."""
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                out[key] = value.to_dict()
            else:
                out[key] = value
        return out


def load_config(path: Union[str, Path]) -> DotDict:
    """Load a YAML configuration file and return it as a ``DotDict``.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the YAML file.

    Returns
    -------
    DotDict
        Parsed configuration with attribute-style access.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        raw: dict = yaml.safe_load(fh)
    return DotDict(raw)
