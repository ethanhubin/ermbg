"""Synthetic probe generator: paste subject onto a solid background.

This is a *baseline* probe (subject is mathematically guaranteed not to change).
It is the upper-bound reference that the SDXL probe must compete with — and it
also lets us run the rest of the pipeline (validator + later phases) without a
GPU.
"""

from __future__ import annotations

import numpy as np

from .generator import ProbeGenerator


class SyntheticProbeGenerator(ProbeGenerator):
    name = "synthetic"

    def generate(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
        background_color: tuple[int, int, int],
        seed: int | None = None,
    ) -> np.ndarray:
        del seed  # deterministic
        if image.dtype != np.uint8:
            raise ValueError("Synthetic probe expects uint8 sRGB input.")
        if image.shape[:2] != subject_mask.shape[:2]:
            raise ValueError("Mask shape must match image shape.")

        soft = subject_mask.astype(np.float32)
        if soft.max() > 1.5:
            soft /= 255.0
        soft = np.clip(soft, 0.0, 1.0)[..., None]

        bg = np.broadcast_to(np.array(background_color, dtype=np.float32), image.shape).copy()
        fg = image.astype(np.float32)
        out = soft * fg + (1.0 - soft) * bg
        return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)
