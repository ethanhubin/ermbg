"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synth_image():
    """A 128x128 sRGB image with a bright disk on a dark gradient background."""
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Background: dark cyan-ish gradient
    bg = np.stack(
        [
            np.full((h, w), 30, dtype=np.float32),
            60 + 0.4 * yy,
            90 + 0.3 * xx,
        ],
        axis=-1,
    )
    # Foreground disk
    cy, cx = h // 2, w // 2
    r = ((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5
    inside = r < 35
    img = bg.copy()
    img[inside] = np.array([220, 60, 80], dtype=np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    soft_mask = np.zeros((h, w), dtype=np.float32)
    soft_mask[inside] = 1.0
    # Slight feather
    import cv2
    soft_mask = cv2.GaussianBlur(soft_mask, (5, 5), 0)

    return img, soft_mask
