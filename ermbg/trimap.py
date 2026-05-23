"""Trimap construction from a soft mask + known background color.

Plan section 16. We do not just erode/dilate the mask — we additionally use
color distance to B to widen sure_bg into the soft-mask edge band where the
image is colorimetrically background, and tighten sure_fg where pixels are
clearly subject.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .segmenter import _band_radius
from .types import Trimap


def build_trimap(
    image: np.ndarray,
    soft_mask: np.ndarray,
    background_color: np.ndarray,
    band_radius: int | None = None,
    fg_color_threshold: float = 8.0,  # kept for API compat; not currently used
    bg_color_threshold: float = 6.0,  # ΔE; pixels this close to B in the band become sure_bg
    fg_alpha_threshold: float = 0.85,
    bg_alpha_threshold: float = 0.05,
    fg_force_alpha: float = 0.99,
) -> Trimap:
    """Return a Trimap (sure_fg / sure_bg / unknown).

    Rules:
      sure_fg = (inner-eroded AND soft >= fg_alpha_threshold)
                OR (soft >= fg_force_alpha)            # alpha is so high we trust it
      sure_bg = outside dilation
                OR (in band AND color ≈ B AND soft very low)
      unknown = everything else
    """
    if band_radius is None:
        band_radius = _band_radius(image.shape)

    soft = soft_mask.astype(np.float32)
    if soft.max() > 1.5:
        soft /= 255.0

    # Distance to B in OKLab.
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    img_lab = srgb_to_oklab(image)
    d_to_B = oklab_distance(img_lab, bg_lab)  # H x W

    binary = (soft > 0.5).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(binary, k, iterations=band_radius) > 0
    outer = cv2.dilate(binary, k, iterations=band_radius) > 0

    # sure_fg: deep inside the segmentation AND high alpha mask, OR alpha extremely
    # high anywhere (alpha truly close to 1 is reliable on its own).
    sure_fg = (inner & (soft >= fg_alpha_threshold)) | (soft >= fg_force_alpha)

    # sure_bg: outside dilation OR (in band but pixel color is essentially B).
    sure_bg = ~outer | (~inner & (d_to_B < bg_color_threshold) & (soft <= bg_alpha_threshold))
    # Don't let sure_bg eat sure_fg if any disagreement.
    sure_bg = sure_bg & ~sure_fg

    unknown = ~sure_fg & ~sure_bg

    return Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)


def trimap_to_uint8(trimap: Trimap) -> np.ndarray:
    """0 = bg, 128 = unknown, 255 = fg, for visualization/saving."""
    out = np.full(trimap.sure_fg.shape, 128, dtype=np.uint8)
    out[trimap.sure_fg] = 255
    out[trimap.sure_bg] = 0
    return out


__all__ = ["build_trimap", "trimap_to_uint8"]
