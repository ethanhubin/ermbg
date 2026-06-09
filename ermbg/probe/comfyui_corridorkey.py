"""Shared result type for the in-process CorridorKey runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ComfyCorridorKeyResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    hint_alpha: np.ndarray
    raw_alpha: np.ndarray
    debug: dict[str, Any]


__all__ = ["ComfyCorridorKeyResult"]
