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


def test_chroma_cap_respects_semantic_protect_mask():
    """VLM-marked green subject material should not be capped as screen spill."""
    C_lin, _, B_lin, _ = _make_polluted_pixel(fg_srgb=(30, 150, 80), alpha=1.0)
    C_arr = C_lin.reshape(1, 1, 3)
    out = chroma_cap(C_arr, B_lin, protect_mask=np.ones((1, 1), dtype=np.float32))
    np.testing.assert_allclose(out, C_arr, atol=1e-6)


def test_chroma_cap_scales_with_background_contribution_from_alpha():
    """Near-opaque subject pixels should not be recolored like translucent spill."""
    C_lin, _, B_lin, _ = _make_polluted_pixel(fg_srgb=(170, 230, 170), alpha=1.0)
    C_arr = C_lin.reshape(1, 1, 3)

    opaque = chroma_cap(C_arr, B_lin, alpha=np.ones((1, 1), dtype=np.float32))[0, 0]
    translucent = chroma_cap(C_arr, B_lin, alpha=np.full((1, 1), 0.25, dtype=np.float32))[0, 0]

    np.testing.assert_allclose(opaque, C_arr[0, 0], atol=1e-6)
    assert translucent[1] < opaque[1]


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


def test_local_borrow_falls_back_to_spatial_weights_when_color_weights_collapse():
    """A heavily background-colored edge may be far from every sure-FG color;
    borrowing should still return a nearby foreground color, not black."""
    h, w = 48, 48
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([235, 215, 185], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    image[8:28, 8:40] = fg
    image[30:34, 8:40] = [80, 150, 70]
    C_lin = io.srgb_to_linear(image)
    B_lin = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[8:28, 8:40] = 1.0
    alpha[30:34, 8:40] = 0.35

    out = local_foreground_borrow(C_lin, B_lin, alpha, band_alpha_low=0.0)
    borrowed = io.linear_to_srgb_u8(out[30:34, 10:38])

    assert borrowed.mean(axis=(0, 1))[0] > 180
    assert borrowed.mean(axis=(0, 1))[1] > 160
    assert borrowed.mean(axis=(0, 1))[2] > 130


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


def test_unmix_borrows_unstable_low_alpha_edge_instead_of_clipping_black():
    """If α is underestimated on a known-B edge, inverse compositing can go
    out of gamut and clip to black. Use that out-of-gamut signal to borrow a
    stable foreground color rather than exporting black speckles."""
    h, w = 48, 48
    F_target = np.array([0.75, 0.12, 0.08], dtype=np.float32)
    B = np.array([0.0, 0.58, 0.0], dtype=np.float32)
    C = np.broadcast_to(B, (h, w, 3)).copy()
    alpha_est = np.zeros((h, w), dtype=np.float32)

    C[10:38, 10:38] = F_target
    alpha_est[10:38, 10:38] = 1.0

    true_alpha = 0.50
    estimated_alpha = 0.20
    C[22:26, 10:38] = true_alpha * F_target + (1.0 - true_alpha) * B
    alpha_est[22:26, 10:38] = estimated_alpha

    recovered = unmix_foreground(C, B, alpha_est)
    strip = recovered[22:26, 12:36]

    assert strip[..., 1].mean() > 0.05  # not clipped to black
    np.testing.assert_allclose(strip.mean(axis=(0, 1)), F_target, atol=0.08)


def test_unmix_borrows_background_dominated_in_gamut_edge_colors():
    """Even in-gamut inverse compositing is unreliable when B dominates C."""
    h, w = 48, 48
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([235, 215, 185], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    image[8:28, 8:40] = fg
    image[30:34, 8:40] = [90, 153, 78]
    C = io.srgb_to_linear(image)
    B = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[8:28, 8:40] = 1.0
    alpha[30:34, 8:40] = 0.35

    recovered = io.linear_to_srgb_u8(unmix_foreground(C, B, alpha))
    strip = recovered[30:34, 10:38]

    assert strip[..., 0].mean() > 180
    assert strip[..., 1].mean() > 160
    assert strip[..., 2].mean() > 130


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
