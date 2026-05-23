"""Alpha estimation and refinement.

Plan sections 18, 19, 21. Operates entirely in linear RGB.

  - estimate_alpha_projection : C, B, F_ref -> alpha_init  (projection method)
  - estimate_alpha_per_channel: C, B, F_ref -> robust per-channel alpha
  - blend_alpha               : combine the two
  - refine_alpha              : guided-filter smoothing inside unknown band
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Trimap


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return num / np.where(np.abs(den) < eps, eps * np.sign(den + eps), den)


def estimate_alpha_projection(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    f_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan §18:
        α₀(x) = clamp( <C - B, F_ref - B> / ||F_ref - B||² )

    Returns (alpha, contrast) where contrast = ||F_ref - B|| (a confidence proxy).
    """
    C = image_linear
    B = background_linear  # broadcastable to C, typically (3,)
    if B.ndim == 1:
        B = np.broadcast_to(B, C.shape)

    cb = C - B
    fb = f_ref - B
    contrast2 = np.sum(fb * fb, axis=-1)
    dot = np.sum(cb * fb, axis=-1)
    alpha = _safe_div(dot, contrast2)
    alpha = np.clip(alpha, 0.0, 1.0)
    contrast = np.sqrt(contrast2)
    return alpha.astype(np.float32), contrast.astype(np.float32)


def estimate_alpha_per_channel(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    f_ref: np.ndarray,
    min_separation: float = 0.05,
) -> np.ndarray:
    """Plan §19: per-channel α_c = (C_c - B_c) / (F_c - B_c), weighted median across
    channels with sufficient |F-B|.
    """
    C = image_linear
    B = background_linear
    if B.ndim == 1:
        B = np.broadcast_to(B, C.shape)

    sep = f_ref - B
    alpha_c = _safe_div(C - B, sep)
    alpha_c = np.clip(alpha_c, 0.0, 1.0)
    weights = np.clip(np.abs(sep) / min_separation, 0.0, 1.0)
    w_sum = weights.sum(axis=-1, keepdims=True)
    w_sum = np.where(w_sum < 1e-6, 1.0, w_sum)
    weighted_mean = (alpha_c * weights).sum(axis=-1, keepdims=True) / w_sum

    # Use weighted median proxy (mean is fine in practice; medians without
    # sorting per-pixel keep this vectorized).
    return weighted_mean.squeeze(-1).astype(np.float32)


def blend_alpha(
    alpha_proj: np.ndarray, alpha_chan: np.ndarray, contrast: np.ndarray
) -> np.ndarray:
    """When contrast is high, projection is reliable. When low, fall back to
    per-channel which makes use of whichever channel still separates."""
    w_proj = np.clip(contrast / 0.3, 0.0, 1.0)  # 0.3 in linear-RGB ~= moderate sep
    return (w_proj * alpha_proj + (1.0 - w_proj) * alpha_chan).astype(np.float32)


def apply_trimap_constraints(alpha: np.ndarray, trimap: Trimap) -> np.ndarray:
    out = alpha.copy()
    out[trimap.sure_fg] = 1.0
    out[trimap.sure_bg] = 0.0
    return out


def refine_alpha(
    image_linear: np.ndarray,
    alpha: np.ndarray,
    trimap: Trimap,
    radius: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Guided-filter refinement (image as guide).

    OpenCV ximgproc.guidedFilter is the fast path. We fall back to a numpy
    implementation if ximgproc isn't available in this build.
    """
    guide = image_linear.astype(np.float32)
    src = alpha.astype(np.float32)

    refined = _guided_filter(guide, src, radius=radius, eps=eps)

    # Trimap is hard truth.
    refined = apply_trimap_constraints(refined, trimap)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def _guided_filter(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Color guided filter (He et al. 2010). I: HxWx3, p: HxW."""
    # Try OpenCV's ximgproc first (faster, uses integral images).
    try:
        ximgproc = cv2.ximgproc  # type: ignore[attr-defined]
    except AttributeError:
        ximgproc = None

    if ximgproc is not None:
        # ximgproc.guidedFilter accepts 32F guide and 32F src.
        return ximgproc.guidedFilter(guide=I, src=p, radius=radius, eps=eps).astype(np.float32)

    # Fallback: pure numpy implementation, color version.
    h, w, _ = I.shape
    n = (2 * radius + 1) ** 2

    def boxfilter(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REPLICATE)

    mean_I = np.stack([boxfilter(I[..., c]) for c in range(3)], axis=-1)
    mean_p = boxfilter(p)
    mean_Ip = np.stack([boxfilter(I[..., c] * p) for c in range(3)], axis=-1)
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]

    var_I = np.empty((h, w, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            var_I[..., i, j] = boxfilter(I[..., i] * I[..., j]) - mean_I[..., i] * mean_I[..., j]
    var_I += eps * np.eye(3, dtype=np.float32)

    inv = np.linalg.inv(var_I)
    a = np.einsum("...ij,...j->...i", inv, cov_Ip)
    b = mean_p - np.sum(a * mean_I, axis=-1)

    mean_a = np.stack([boxfilter(a[..., c]) for c in range(3)], axis=-1)
    mean_b = boxfilter(b)
    q = np.sum(mean_a * I, axis=-1) + mean_b
    return q.astype(np.float32)


def estimate_alpha_full(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    f_ref: np.ndarray,
    trimap: Trimap,
    refine_radius: int = 8,
    refine_eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: projection -> per-channel -> blend -> trimap snap -> refine.

    Returns (alpha, confidence) where confidence = contrast (||F_ref - B||).
    """
    alpha_proj, contrast = estimate_alpha_projection(image_linear, background_linear, f_ref)
    alpha_chan = estimate_alpha_per_channel(image_linear, background_linear, f_ref)
    alpha_init = blend_alpha(alpha_proj, alpha_chan, contrast)
    alpha_init = apply_trimap_constraints(alpha_init, trimap)
    alpha = refine_alpha(image_linear, alpha_init, trimap, radius=refine_radius, eps=refine_eps)
    return alpha, contrast


__all__ = [
    "estimate_alpha_projection",
    "estimate_alpha_per_channel",
    "blend_alpha",
    "apply_trimap_constraints",
    "refine_alpha",
    "estimate_alpha_full",
]
