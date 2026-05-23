from __future__ import annotations

import numpy as np

from ermbg import colorspace


def test_oklab_roundtrip():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    lab = colorspace.srgb_to_oklab(rgb)
    back = colorspace.oklab_to_srgb(lab)
    diff = np.abs(back - rgb.astype(np.float32) / 255.0)
    # Conversion is lossy through gamma + cbrt; expect well below 2/255.
    assert diff.max() < 0.01


def test_oklab_distance_zero_for_same_color():
    color = np.array([[[200, 50, 80]]], dtype=np.uint8)
    lab = colorspace.srgb_to_oklab(color)
    d = colorspace.oklab_distance(lab, lab)
    assert d.item() < 1e-4


def test_oklab_distance_orders_colors():
    a = colorspace.srgb_to_oklab(np.array([[[0, 0, 0]]], dtype=np.uint8))
    grey = colorspace.srgb_to_oklab(np.array([[[128, 128, 128]]], dtype=np.uint8))
    white = colorspace.srgb_to_oklab(np.array([[[255, 255, 255]]], dtype=np.uint8))
    d_grey = colorspace.oklab_distance(a, grey).item()
    d_white = colorspace.oklab_distance(a, white).item()
    assert d_white > d_grey > 0
