"""Tests for the front-end strategy router."""

from __future__ import annotations

import numpy as np

from ermbg.router import classify_strategy


def _solid_with_subject(bg_color, fg_color=(220, 30, 30), h=128, w=128):
    img = np.broadcast_to(np.array(bg_color, dtype=np.uint8), (h, w, 3)).copy()
    img[40:90, 40:90] = fg_color
    return img


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
    """An input with a real alpha matte (not all-opaque) should route to passthrough."""
    h, w = 64, 64
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0  # center is opaque, 75% transparent
    s = classify_strategy(img, source_alpha=alpha)
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
