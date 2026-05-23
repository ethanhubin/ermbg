"""Tests for despill algorithms."""

from __future__ import annotations

import numpy as np

from ermbg import io
from ermbg.despill import (
    apply_despill,
    chroma_cap,
    has_dominant_screen_channel,
    local_foreground_borrow,
    unmix_foreground,
)


def _make_polluted_pixel(fg_srgb=(200, 80, 60), bg_srgb=(0, 200, 0), alpha=0.5):
    """Build C = α*F + (1-α)*B, return (C_lin, F_lin, B_lin, alpha)."""
    F_lin = io.srgb_to_linear(np.array(fg_srgb, dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    B_lin = io.srgb_to_linear(np.array(bg_srgb, dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    C_lin = alpha * F_lin + (1 - alpha) * B_lin
    return C_lin, F_lin, B_lin, alpha


def test_dominant_channel_detection():
    """Saturated green is detected; black/grey/white are not."""
    green = io.srgb_to_linear(np.array([[[0, 200, 0]]], dtype=np.uint8))[0, 0]
    assert has_dominant_screen_channel(green) == 1

    black = io.srgb_to_linear(np.array([[[0, 0, 0]]], dtype=np.uint8))[0, 0]
    assert has_dominant_screen_channel(black) is None

    grey = io.srgb_to_linear(np.array([[[60, 60, 60]]], dtype=np.uint8))[0, 0]
    assert has_dominant_screen_channel(grey) is None


def test_chroma_cap_reduces_green_channel_only():
    """Chroma cap should pull G down toward max(R, B), not touch R/B."""
    C_lin, _, B_lin, _ = _make_polluted_pixel(fg_srgb=(200, 80, 60))
    C_arr = C_lin.reshape(1, 1, 3)
    out = chroma_cap(C_arr, B_lin)[0, 0]
    # G must be <= max(R, B)
    assert out[1] <= max(out[0], out[2]) + 1e-6
    # R and B unchanged.
    np.testing.assert_allclose(out[0], C_arr[0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(out[2], C_arr[0, 0, 2], atol=1e-6)


def test_chroma_cap_no_op_on_black_bg():
    """Black bg has no dominant channel; chroma cap returns input."""
    C_lin, _, B_lin, _ = _make_polluted_pixel(fg_srgb=(200, 80, 60), bg_srgb=(0, 0, 0))
    C_arr = C_lin.reshape(1, 1, 3)
    out = chroma_cap(C_arr, B_lin)
    np.testing.assert_array_equal(out, C_arr)


def test_local_borrow_fills_band_with_fg_color():
    """Translucent pixels surrounded by sure_fg of color X should become close to X."""
    h, w = 64, 64
    F_color = np.array([180, 60, 60], dtype=np.uint8)
    image_srgb = np.broadcast_to(F_color, (h, w, 3)).copy()
    # Pollute one strip with black (simulating background bleed).
    image_srgb[28:36, :] = 0
    C_lin = io.srgb_to_linear(image_srgb)

    alpha = np.full((h, w), 1.0, dtype=np.float32)
    alpha[28:36, :] = 0.4  # the polluted strip looks translucent

    F_target = io.srgb_to_linear(F_color.reshape(1, 1, 3))[0, 0]
    B_lin = np.array([0, 0, 0], dtype=np.float32)
    out = local_foreground_borrow(C_lin, B_lin, alpha)

    strip = out[28:36]
    # The borrowed color should be much closer to F than the polluted input was.
    diff_in = np.abs(C_lin[28:36] - F_target).mean()
    diff_out = np.abs(strip - F_target).mean()
    assert diff_out < diff_in / 3, f"borrow only reduced from {diff_in:.3f} to {diff_out:.3f}"


def test_apply_despill_dispatches():
    """All three method names should be accepted and produce sensible shapes."""
    h, w = 32, 32
    image = io.srgb_to_linear(np.random.RandomState(0).randint(0, 256, (h, w, 3), dtype=np.uint8))
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[10:20, 10:20] = 0.5
    B = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    for method in ("chroma_cap", "local_borrow", "unmix", "auto"):
        a, F = apply_despill(method, image, B, alpha)
        assert a.shape == alpha.shape
        assert F.shape == image.shape
        assert F.dtype == np.float32


def test_unmix_recovers_original_F():
    """Build a synthetic composite C = α·F + (1-α)·B, then unmix should give F back."""
    F_target = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    B = np.array([0.0, 0.6, 0.0], dtype=np.float32)
    h, w = 16, 16
    alpha = np.full((h, w), 0.5, dtype=np.float32)
    F_img = np.broadcast_to(F_target, (h, w, 3)).copy()
    B_img = np.broadcast_to(B, (h, w, 3))
    C = alpha[..., None] * F_img + (1.0 - alpha[..., None]) * B_img

    F_recovered = unmix_foreground(C, B, alpha)
    np.testing.assert_allclose(F_recovered, F_img, atol=1e-5)


def test_unmix_handles_low_alpha_with_fallback():
    """Pixels with α≈0 should be filled by the fallback, not blow up."""
    F_target = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    B = np.array([0.0, 0.6, 0.0], dtype=np.float32)
    h, w = 32, 32
    alpha = np.full((h, w), 1.0, dtype=np.float32)
    alpha[16:, :] = 0.0   # half the image is fully transparent
    F_img = np.broadcast_to(F_target, (h, w, 3)).copy()
    F_img[16:, :] = B     # transparent half observes only B
    C = alpha[..., None] * F_img + (1.0 - alpha[..., None]) * B

    F_recovered = unmix_foreground(C, B, alpha, fallback_method="background")
    assert np.all(F_recovered <= 1.0) and np.all(F_recovered >= 0.0)
    np.testing.assert_allclose(F_recovered[:16], F_img[:16], atol=1e-5)
