"""Light wrap: video-compositing trick to perceptually eliminate edge halos.

After computing α and F_clean, the standard composite is:
    out = α·F + (1-α)·B_new

The "halo" you see on a clean key against a contrasting B_new is partly because
real-world images have light bouncing off the new background onto the subject's
edge. Light wrap fakes that:

    out = α·F + (1-α)·B_new + α·(1-α)·gauss(B_new, σ)·strength

The α·(1-α) gate concentrates the wrap on the soft edge band only. Reference:
Brinkmann, *The Art and Science of Digital Compositing* (2008), ch. 6.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import io


def light_wrap_composite(
    foreground_srgb: np.ndarray,
    alpha: np.ndarray,
    background_color_or_image,
    strength: float = 0.3,
    sigma: float = 3.0,
) -> np.ndarray:
    """Composite F onto B_new with light-wrap halo suppression.

    Operates in linear RGB and returns sRGB uint8.

    Args:
        foreground_srgb: H×W×3 sRGB uint8.
        alpha: H×W float in [0,1].
        background_color_or_image: 3-tuple sRGB or H×W×3 sRGB uint8 image.
        strength: 0..1, how much of the blurred bg leaks into edges.
        sigma: gaussian blur radius (pixels) for the wrap kernel.
    """
    h, w = alpha.shape
    if isinstance(background_color_or_image, np.ndarray):
        bg = background_color_or_image.astype(np.uint8)
        if bg.shape != (h, w, 3):
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    else:
        bg = np.broadcast_to(np.asarray(background_color_or_image, dtype=np.uint8), (h, w, 3)).copy()

    F_lin = io.srgb_to_linear(foreground_srgb).astype(np.float32)
    B_lin = io.srgb_to_linear(bg).astype(np.float32)
    a = alpha[..., None].astype(np.float32)

    if sigma > 0 and strength > 0:
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        if ksize % 2 == 0:
            ksize += 1
        bg_blur = cv2.GaussianBlur(B_lin, (ksize, ksize), sigma)
    else:
        bg_blur = B_lin

    wrap = a * (1.0 - a) * bg_blur * float(strength)
    out_lin = a * F_lin + (1.0 - a) * B_lin + wrap
    return io.linear_to_srgb_u8(np.clip(out_lin, 0.0, 1.0))


__all__ = ["light_wrap_composite"]
