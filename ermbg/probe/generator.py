"""Probe-generator interface and standard color presets."""

from __future__ import annotations

from typing import Protocol

import numpy as np


# Standard probe colors used in Phase 1 evaluation (sRGB uint8).
# Plan section 6.2 / 6.3 / 33: white + saturated cyan or magenta is a strong default.
PROBE_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (250, 250, 250),
    "black": (8, 8, 8),
    "cyan": (0, 200, 220),
    "magenta": (220, 30, 180),
    "green": (0, 200, 60),
}


class ProbeGenerator(Protocol):
    """Generates a probe image with a known solid background, preserving the subject."""

    name: str

    def generate(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
        background_color: tuple[int, int, int],
        seed: int | None = None,
    ) -> np.ndarray:
        """Returns H x W x 3 sRGB uint8."""
        ...
