"""Tests for light-wrap composite."""

from __future__ import annotations

import numpy as np

from ermbg.lightwrap import light_wrap_composite


def test_alpha_zero_returns_bg():
    fg = np.full((16, 16, 3), 200, dtype=np.uint8)
    alpha = np.zeros((16, 16), dtype=np.float32)
    out = light_wrap_composite(fg, alpha, (10, 20, 30))
    expected = np.broadcast_to(np.array([10, 20, 30], dtype=np.uint8), (16, 16, 3))
    diff = np.abs(out.astype(int) - expected.astype(int))
    # Allow 1 LSB rounding through linear/sRGB.
    assert diff.max() <= 2


def test_alpha_one_returns_fg():
    fg = np.full((16, 16, 3), 200, dtype=np.uint8)
    alpha = np.ones((16, 16), dtype=np.float32)
    out = light_wrap_composite(fg, alpha, (10, 20, 30))
    diff = np.abs(out.astype(int) - fg.astype(int))
    assert diff.max() <= 2


def test_lightwrap_only_affects_edge_band():
    """For pixels strictly α=0 or α=1, light wrap is mathematically inert
    (because of the α(1-α) gate)."""
    fg = np.full((16, 16, 3), 200, dtype=np.uint8)
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[8:, :] = 1.0
    out = light_wrap_composite(fg, alpha, (10, 20, 30), strength=0.5, sigma=1.0)
    # Top row (α=0) must equal bg; bottom row (α=1) must equal fg.
    expected_top = np.array([10, 20, 30], dtype=np.uint8)
    diff_top = np.abs(out[0].astype(int) - expected_top.astype(int)).max()
    diff_bot = np.abs(out[-1].astype(int) - fg[-1].astype(int)).max()
    assert diff_top <= 2
    assert diff_bot <= 2
