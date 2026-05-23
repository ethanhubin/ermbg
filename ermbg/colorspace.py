"""Color space utilities: sRGB <-> OKLab.

OKLab gives perceptually uniform color distance, used for picking the optimal
probe background color (plan section 12).
"""

from __future__ import annotations

import numpy as np

from .io import linear_to_srgb, srgb_to_linear


# Linear sRGB -> LMS (Bjorn Ottosson, oklab.org)
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float32,
)
# LMS' -> OKLab
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float32,
)
_M1_INV = np.linalg.inv(_M1).astype(np.float32)
_M2_INV = np.linalg.inv(_M2).astype(np.float32)


def linear_rgb_to_oklab(linear_rgb: np.ndarray) -> np.ndarray:
    """linear-RGB float -> OKLab float (same shape, last dim = 3)."""
    shape = linear_rgb.shape
    flat = linear_rgb.reshape(-1, 3).astype(np.float32)
    lms = flat @ _M1.T
    lms_p = np.cbrt(np.maximum(lms, 0.0))
    lab = lms_p @ _M2.T
    return lab.reshape(shape)


def oklab_to_linear_rgb(oklab: np.ndarray) -> np.ndarray:
    shape = oklab.shape
    flat = oklab.reshape(-1, 3).astype(np.float32)
    lms_p = flat @ _M2_INV.T
    lms = lms_p ** 3
    rgb = lms @ _M1_INV.T
    return rgb.reshape(shape)


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 or float 0..1 -> OKLab."""
    return linear_rgb_to_oklab(srgb_to_linear(srgb))


def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    """OKLab -> sRGB float 0..1."""
    return linear_to_srgb(oklab_to_linear_rgb(oklab))


def oklab_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance in OKLab. Inputs broadcast on last dim = 3.

    Distance is scaled by 100 so values are comparable to CIELAB ΔE.
    """
    diff = (a - b).astype(np.float32)
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return d * 100.0
