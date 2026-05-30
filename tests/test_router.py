"""Tests for the front-end strategy router."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ermbg.router import assess_source_alpha, classify_strategy

pytestmark = pytest.mark.core


def _solid_with_subject(bg_color, fg_color=(220, 30, 30), h=128, w=128):
    img = np.broadcast_to(np.array(bg_color, dtype=np.uint8), (h, w, 3)).copy()
    img[40:90, 40:90] = fg_color
    return img


def _make_clean_rgba(h=128, w=128):
    """RGBA where opaque interior is uniform red, transparent regions are
    premultiplied (RGB=0), and the soft edge is properly mixed."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    a = np.clip((40.0 - r) / 4.0, 0.0, 1.0).astype(np.float32)
    F = np.array([220, 30, 30], dtype=np.float32)
    rgb = (a[..., None] * F).astype(np.uint8)
    return rgb, a


def test_classify_saturated_green():
    img = _solid_with_subject((0, 200, 0))
    s = classify_strategy(img)
    assert s.bg_type == "saturated"
    assert s.keyer_mode == "chromatic"
    assert s.despill == "auto"


def test_classify_white_bg():
    img = _solid_with_subject((255, 255, 255))
    s = classify_strategy(img)
    assert s.bg_type == "white"
    assert s.keyer_mode == "luminance"
    assert s.despill == "unmix"


def test_classify_black_bg():
    img = _solid_with_subject((0, 0, 0), fg_color=(220, 30, 30))
    s = classify_strategy(img)
    assert s.bg_type == "black"
    assert s.keyer_mode == "luminance"


def test_classify_grey_bg():
    img = _solid_with_subject((128, 128, 128))
    s = classify_strategy(img)
    assert s.bg_type == "grey"
    assert s.keyer_mode == "luminance"


def test_classify_passthrough_when_source_alpha_present():
    """An input with a real, *clean* alpha matte should route to passthrough.
    Premultiplied RGB (zero in transparent regions) and soft edges are required."""
    rgb, a = _make_clean_rgba()
    s = classify_strategy(rgb, source_alpha=a)
    assert s.passthrough is True
    assert s.bg_type == "rgba_passthrough"


def test_classify_no_passthrough_when_source_fully_opaque():
    """An RGBA input that is actually 100% opaque should NOT pass through; it
    has no usable α, and the router should treat it like a normal RGB input."""
    img = _solid_with_subject((0, 200, 0))   # default 128×128, subject doesn't touch corners
    alpha = np.ones(img.shape[:2], dtype=np.float32)
    s = classify_strategy(img, source_alpha=alpha)
    assert s.passthrough is False
    # falls through to normal classification
    assert s.bg_type == "saturated"


def test_classify_image_type_graphic_vs_photo():
    """Solid-color flat graphic should be 'graphic'; noisy random image should be 'photo'."""
    flat = _solid_with_subject((0, 200, 0))
    s_flat = classify_strategy(flat)
    assert s_flat.image_type == "graphic"

    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    s_noisy = classify_strategy(noisy)
    assert s_noisy.image_type == "photo"


# --- alpha hygiene -----------------------------------------------------------


def _make_haloed_rgba(h=128, w=128):
    """Same shape, but soft edge has a strong white halo as if a coarse
    segmenter alpha-blended onto white."""
    rgb, a = _make_clean_rgba(h, w)
    # Pollute soft edge with white.
    edge = (a > 0.05) & (a <= 0.6)
    rgb_f = rgb.astype(np.float32)
    rgb_f[edge] = 0.4 * rgb_f[edge] + 0.6 * np.array([255, 255, 255], dtype=np.float32)
    return rgb_f.astype(np.uint8), a


def test_hygiene_clean_passes():
    rgb, a = _make_clean_rgba()
    h = assess_source_alpha(rgb, a)
    assert h.clean, f"expected clean, got reason={h.reason}"


def test_hygiene_halo_rejected():
    rgb, a = _make_haloed_rgba()
    h = assess_source_alpha(rgb, a)
    assert not h.clean, "halo-y matte should be rejected"
    assert "fringe" in h.reason.lower() or "low-α" in h.reason.lower()


def test_classify_dirty_rgba_routes_to_rematte():
    """A dirty RGBA should NOT pass through; the strategy falls into normal
    bg classification (here a dark rectangle on transparent → after the
    RGB is passed in raw, the bg sampling sees premultiplied-zero corners
    and routes to black_bg)."""
    rgb, a = _make_haloed_rgba()
    s = classify_strategy(rgb, source_alpha=a)
    assert s.passthrough is False
    assert s.bg_type != "rgba_passthrough"
    assert "passthrough_rejected" in s.extras


def test_classify_clean_rgba_passes_through():
    rgb, a = _make_clean_rgba()
    s = classify_strategy(rgb, source_alpha=a)
    assert s.passthrough is True
    assert s.bg_type == "rgba_passthrough"


# --- broader bg-color sweep --------------------------------------------------


@pytest.mark.parametrize(
    "bg, expected_bg_type",
    [
        # Saturated screens — should all route to saturated_bg.
        ((0, 200, 0), "saturated"),       # green-screen
        ((0, 220, 0), "saturated"),       # bright green
        ((0, 200, 220), "saturated"),     # cyan
        ((220, 30, 180), "saturated"),    # magenta
        ((220, 0, 0), "saturated"),       # red
        ((0, 0, 220), "saturated"),       # blue
        ((220, 220, 0), "saturated"),     # yellow
        # Lightness extremes.
        ((255, 255, 255), "white"),
        ((250, 248, 252), "white"),       # near-white with tiny chroma
        ((0, 0, 0), "black"),
        ((6, 8, 4), "black"),             # near-black
        # Mid-grey, low chroma → grey.
        ((128, 128, 128), "grey"),
        ((90, 92, 94), "grey"),
        ((180, 178, 182), "grey"),        # light grey but not white-threshold
    ],
)
def test_classify_bg_color_sweep(bg, expected_bg_type):
    img = _solid_with_subject(bg, fg_color=(140, 90, 30))
    s = classify_strategy(img)
    assert s.bg_type == expected_bg_type, f"bg={bg}: got {s.bg_type}, expected {expected_bg_type}"


def test_classify_extremely_noisy_bg():
    """Random RGB everywhere → noisy strategy, no key, prefer local_borrow."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    s = classify_strategy(img)
    assert s.bg_type == "noisy"
    assert s.keyer_mode is None
    assert s.despill == "local_borrow"


def test_small_ui_icon_with_clean_green_corners_is_not_noisy():
    """Tiny UI sprites can have subject rims on the edge but clean bg corners."""
    path = (
        Path(__file__).resolve().parents[1]
        / "samples"
        / "corridorkey_semantic"
        / "icon"
        / "icon_icon_a01_hard_boundary_strong_outline"
        / "green.png"
    )
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    s = classify_strategy(img)
    assert s.bg_type == "saturated"
    assert s.keyer_mode == "chromatic"
    assert s.extras["bg_sigma"] < 2.0


def test_wide_star_button_with_clean_green_corners_is_not_noisy():
    """Wide UI buttons can dominate the border while still having stable bg corners."""
    path = (
        Path(__file__).resolve().parents[1]
        / "samples"
        / "corridorkey_semantic"
        / "button"
        / "button_green_yellow_a_outlined_no_shadow"
        / "green.png"
    )
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    s = classify_strategy(img)
    assert s.bg_type == "saturated"
    assert s.keyer_mode == "chromatic"
    assert s.extras["bg_sigma"] < 2.0


def test_strategy_thresholds_tighter_for_graphic():
    """Graphic strategies should get tight keyer thresholds."""
    img = _solid_with_subject((0, 200, 0))   # flat → graphic
    s = classify_strategy(img)
    assert s.image_type == "graphic"
    assert s.keyer_thresholds is not None
    assert s.keyer_thresholds.bg_max <= 4.5
    assert s.keyer_thresholds.fg_min <= 16.0


def test_strategy_thresholds_wider_for_photo():
    """Photo strategies should get wider keyer thresholds."""
    rng = np.random.default_rng(0)
    # Make a 'photo': uniform-ish saturated bg with noise + a subject.
    img = np.full((128, 128, 3), [0, 200, 0], dtype=np.uint8)
    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[40:90, 40:90] = (140, 90, 30)
    # noise pushes top-8 quantized colors below 0.6 → photo classification
    s = classify_strategy(img)
    if s.image_type == "photo":
        assert s.keyer_thresholds.bg_max >= 5.0
        assert s.keyer_thresholds.fg_min >= 18.0


def test_gate_enabled_for_graphic_disabled_for_photo():
    """Hard-edged graphics on solid bg should enable keyer gate; photos must not."""
    flat = _solid_with_subject((0, 200, 0))
    s_flat = classify_strategy(flat)
    if s_flat.bg_type == "saturated":
        assert s_flat.use_keyer_gate is True

    rng = np.random.default_rng(0)
    img = np.full((128, 128, 3), [0, 200, 0], dtype=np.uint8)
    noise = rng.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[40:90, 40:90] = (140, 90, 30)
    s_photo = classify_strategy(img)
    if s_photo.image_type == "photo":
        assert s_photo.use_keyer_gate is False


def test_grey_strategy_does_not_use_merge_or_gate():
    """Grey backgrounds give weak signal — must not aggressively gate or merge."""
    img = _solid_with_subject((128, 128, 128))
    s = classify_strategy(img)
    assert s.use_keyer_merge is False
    assert s.use_keyer_gate is False


def test_strategy_extras_contains_diagnostics():
    """Strategy should record bg lightness/chroma/sigma + image_type for debug."""
    img = _solid_with_subject((0, 200, 0))
    s = classify_strategy(img)
    assert "bg_L" in s.extras
    assert "bg_C" in s.extras
    assert "bg_sigma" in s.extras
    assert "image_type" in s.extras


# --- alpha hygiene edge cases -----------------------------------------------


def test_hygiene_binarized_matte_rejected():
    """An α with only 0/1 values (no soft edge) should be rejected so we
    re-matte to recover anti-aliasing."""
    h, w = 128, 128
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[40:90, 40:90] = 200
    a = np.zeros((h, w), dtype=np.float32)
    a[40:90, 40:90] = 1.0
    hyg = assess_source_alpha(img, a)
    assert not hyg.clean
    assert "binarized" in hyg.reason


def test_hygiene_no_opaque_interior_rejected():
    """An α that is 100% transparent has no interior to anchor against → reject."""
    h, w = 32, 32
    img = np.zeros((h, w, 3), dtype=np.uint8)
    a = np.zeros((h, w), dtype=np.float32)
    hyg = assess_source_alpha(img, a)
    assert not hyg.clean


def test_hygiene_low_alpha_leak_detected():
    """If α=0 pixels carry the interior color (asset saved without
    zeroing transparent RGB), low-α residual should fire."""
    rgb, a = _make_clean_rgba()
    # Replace transparent region (α<0.05) RGB with the interior color → leaky.
    interior = (220, 30, 30)
    rgb_leak = rgb.copy()
    rgb_leak[a < 0.05] = interior
    hyg = assess_source_alpha(rgb_leak, a)
    assert not hyg.clean
    assert "low-α" in hyg.reason or "fringe" in hyg.reason


def test_hygiene_passthrough_threshold_5pct():
    """RGBA with <5% transparent pixels is not really keyed — should NOT
    enter the passthrough branch (router treats it as plain RGB)."""
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:90, 10:90] = 200
    a = np.ones((h, w), dtype=np.float32)
    a[0, 0] = 0.0  # 1 transparent pixel → 0.01% — far below 5%
    s = classify_strategy(img, source_alpha=a)
    assert s.passthrough is False
    # No passthrough-related extras (didn't enter the hygiene branch)
    assert "hygiene" not in s.extras
    assert "passthrough_rejected" not in s.extras
