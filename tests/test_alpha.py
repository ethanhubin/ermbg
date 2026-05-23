"""Tests for the analytic alpha estimator on a controlled synthetic image."""

from __future__ import annotations

import numpy as np

from ermbg import io
from ermbg.alpha import estimate_alpha_full, estimate_alpha_projection
from ermbg.foreground import estimate_foreground_reference
from ermbg.types import Trimap


def _make_synthetic_case(h=128, w=128, fg=(220, 60, 80), bg=(20, 20, 20)):
    """Build a known scene: solid color disc on solid color bg with feathered edge.

    Returns (image_srgb, soft_mask, alpha_gt, F_gt_lin, B_lin).
    """
    fg_arr = np.array(fg, dtype=np.uint8)
    bg_arr = np.array(bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Smooth alpha falloff between r=35..45
    alpha_gt = np.clip((45.0 - r) / 10.0, 0.0, 1.0).astype(np.float32)

    F_lin = io.srgb_to_linear(np.broadcast_to(fg_arr, (h, w, 3)))
    B_lin = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0]
    a = alpha_gt[..., None]
    C_lin = a * F_lin + (1.0 - a) * B_lin
    image_srgb = io.linear_to_srgb_u8(C_lin)
    return image_srgb, alpha_gt, F_lin, B_lin


def _trimap_from_alpha_gt(alpha_gt: np.ndarray) -> Trimap:
    sure_fg = alpha_gt >= 0.99
    sure_bg = alpha_gt <= 0.01
    unknown = ~sure_fg & ~sure_bg
    return Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)


def test_alpha_full_on_synthetic_case():
    image_srgb, alpha_gt, F_lin, B_lin = _make_synthetic_case()
    trimap = _trimap_from_alpha_gt(alpha_gt)

    C_lin = io.srgb_to_linear(image_srgb)
    f_ref = estimate_foreground_reference(C_lin, trimap)
    alpha, _ = estimate_alpha_full(C_lin, B_lin, f_ref, trimap)

    err = np.abs(alpha - alpha_gt)
    assert err.mean() < 0.02
    assert np.percentile(err, 95) < 0.07


def test_projection_alpha_recovers_pure_pixels():
    """At α=0 -> α≈0, at α=1 -> α≈1."""
    image_srgb, alpha_gt, F_lin, B_lin = _make_synthetic_case()
    C_lin = io.srgb_to_linear(image_srgb)
    f_ref = F_lin  # cheat: use the GT
    alpha_proj, contrast = estimate_alpha_projection(C_lin, B_lin, f_ref)

    pure_fg = alpha_gt >= 0.99
    pure_bg = alpha_gt <= 0.01
    assert alpha_proj[pure_fg].mean() > 0.97
    assert alpha_proj[pure_bg].mean() < 0.03
