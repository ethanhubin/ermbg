"""Quality metrics used by the validator and the evaluation script.

Plan sections 8 (validator), 21 (refinement), 26 (multi-bg recheck).
"""

from __future__ import annotations

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Soft mask (float 0..1 or uint8) -> bool mask."""
    if mask.dtype == np.uint8:
        return mask >= int(threshold * 255)
    return mask >= threshold


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Boolean-mask IoU."""
    a_b = binarize(a)
    b_b = binarize(b)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def mask_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Return a bool mask of the outline (boundary pixels) of a binary mask."""
    m = binarize(mask).astype(np.uint8) * 255
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=thickness)
    return (m > 0) & (eroded == 0)


def hausdorff_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between mask boundaries (in pixels).

    Uses 95th-percentile rather than max to be robust to a few outlier pixels.
    """
    boundary_a = mask_boundary(a)
    boundary_b = mask_boundary(b)
    if not boundary_a.any() or not boundary_b.any():
        return 0.0

    # Distance transform: distance from each pixel to the nearest True pixel of the other mask.
    inv_a = (~boundary_a).astype(np.uint8)
    inv_b = (~boundary_b).astype(np.uint8)
    dt_a = cv2.distanceTransform(inv_a, distanceType=cv2.DIST_L2, maskSize=3)
    dt_b = cv2.distanceTransform(inv_b, distanceType=cv2.DIST_L2, maskSize=3)

    d_ab = dt_b[boundary_a]
    d_ba = dt_a[boundary_b]
    if d_ab.size == 0 or d_ba.size == 0:
        return 0.0
    return float(max(np.percentile(d_ab, 95.0), np.percentile(d_ba, 95.0)))


# ---------------------------------------------------------------------------
# Color metrics
# ---------------------------------------------------------------------------


def _bg_region(mask: np.ndarray, dilate_radius: int) -> np.ndarray:
    """Pixels safely outside the subject. Dilates by ``dilate_radius`` (not
    half) plus a small AA safety margin so anti-aliased edges don't pollute
    the bg color/σ measurement."""
    m = binarize(mask).astype(np.uint8) * 255
    iters = max(2, dilate_radius) + 2
    dil = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=iters)
    return dil == 0


def background_purity_sigma(image: np.ndarray, mask: np.ndarray, dilate_radius: int = 16) -> float:
    """Std-dev (in uint8 RGB) of pixels far outside the subject."""
    bg_region = _bg_region(mask, dilate_radius)
    if bg_region.sum() < 16:
        return float("inf")
    pixels = image[bg_region].astype(np.float32)
    return float(np.std(pixels, axis=0).mean())


def measure_background_color(
    image: np.ndarray, mask: np.ndarray, dilate_radius: int = 16
) -> np.ndarray:
    """Median sRGB color of the far-from-subject region. Returns shape (3,) uint8."""
    bg_region = _bg_region(mask, dilate_radius)
    if bg_region.sum() < 16:
        return np.array([0, 0, 0], dtype=np.uint8)
    return np.median(image[bg_region], axis=0).astype(np.uint8)


def internal_color_dE_p95(
    image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray, erode_radius: int = 8
) -> float:
    """OKLab distance P95 inside the eroded subject mask, comparing two images."""
    m = binarize(mask).astype(np.uint8) * 255
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, erode_radius))
    inside = inner > 0
    if inside.sum() < 64:
        return 0.0
    a_lab = srgb_to_oklab(image_a)
    b_lab = srgb_to_oklab(image_b)
    d = oklab_distance(a_lab[inside], b_lab[inside])
    return float(np.percentile(d, 95.0))


def edge_contrast_q10(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: np.ndarray,
    band_radius: int = 8,
    min_alpha: float = 0.1,
) -> float:
    """Q10 of OKLab(edge_pixel, B). Low = bad (some edges look like the bg).

    Only samples band pixels where the soft mask has non-trivial subject content
    (alpha > min_alpha) so that pure-background pixels inside the dilated band
    don't artificially drive the distance to zero.
    """
    soft = mask.astype(np.float32)
    if soft.max() > 1.5:
        soft /= 255.0
    m = (soft > 0.5).astype(np.uint8) * 255
    dil = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=band_radius)
    ero = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=band_radius)
    band = (dil > 0) & (ero == 0) & (soft > min_alpha)
    if band.sum() < 64:
        return 0.0
    edge_lab = srgb_to_oklab(image[band])
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    d = oklab_distance(edge_lab, bg_lab)
    return float(np.percentile(d, 10.0))


# ---------------------------------------------------------------------------
# Recomposition error (for later phases)
# ---------------------------------------------------------------------------


def recomposition_error(
    composite: np.ndarray, alpha: np.ndarray, foreground: np.ndarray, background: np.ndarray
) -> float:
    """Mean L2 distance between observed C and (alpha*F + (1-a)*B). Inputs in linear RGB float."""
    a = alpha[..., None]
    c_hat = a * foreground + (1.0 - a) * background
    diff = composite - c_hat
    return float(np.sqrt(np.mean(diff * diff)))
