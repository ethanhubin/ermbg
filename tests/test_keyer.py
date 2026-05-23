"""Tests for chromatic-key α + small-component merge."""

from __future__ import annotations

import numpy as np

from ermbg.keyer import (
    chromatic_key_alpha,
    key_alpha,
    luminance_key_alpha,
    merge_alpha_components,
)


def test_chromatic_key_alpha_pure_bg_is_zero():
    """A solid green canvas keyed against green B should produce α≈0."""
    img = np.full((32, 32, 3), [0, 200, 0], dtype=np.uint8)
    a = chromatic_key_alpha(img, (0, 200, 0))
    assert a.max() < 0.05, f"α too high on pure bg: max={a.max()}"


def test_chromatic_key_alpha_distant_color_is_one():
    """Pure red against green B should produce α≈1."""
    img = np.full((32, 32, 3), [220, 0, 0], dtype=np.uint8)
    a = chromatic_key_alpha(img, (0, 200, 0))
    assert a.min() > 0.95, f"α too low on contrasting fg: min={a.min()}"


def test_merge_recovers_missed_component():
    """If matting α misses a small isolated blob that the keyer sees, merge
    should patch it back without altering the main subject."""
    h, w = 64, 64
    matting = np.zeros((h, w), dtype=np.float32)
    matting[10:40, 10:40] = 1.0  # main blob -- already in matting

    chrom = np.zeros((h, w), dtype=np.float32)
    chrom[10:40, 10:40] = 1.0      # keyer also sees main blob
    chrom[50:60, 50:60] = 1.0      # ... plus a small one matting missed

    merged, info = merge_alpha_components(matting, chrom)
    assert info["patched_components"] == 1
    assert merged[10:40, 10:40].mean() > 0.99   # main blob untouched
    assert merged[50:60, 50:60].mean() > 0.5    # small blob recovered


def test_merge_does_not_override_main_subject():
    """If matting already represents the region, the chromatic α (which may
    be coarser) should not replace it."""
    h, w = 64, 64
    matting = np.zeros((h, w), dtype=np.float32)
    matting[10:40, 10:40] = 0.5  # half-α, but represented

    chrom = np.zeros((h, w), dtype=np.float32)
    chrom[10:40, 10:40] = 1.0    # keyer says full

    merged, info = merge_alpha_components(matting, chrom)
    assert info["patched_components"] == 0
    np.testing.assert_array_equal(merged, matting)


def test_luminance_key_white_bg_dark_subject():
    """Dark subject on white bg should get α≈1; pure white pixels α≈0."""
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    img[8:24, 8:24] = 30   # dark square in middle of white canvas
    a = luminance_key_alpha(img, (255, 255, 255))
    assert a[16, 16] > 0.95
    assert a[0, 0] < 0.05


def test_luminance_key_black_bg_bright_subject():
    """Bright subject on black bg should get α≈1; pure black pixels α≈0."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = 220
    a = luminance_key_alpha(img, (0, 0, 0))
    assert a[16, 16] > 0.95
    assert a[0, 0] < 0.05


def test_key_alpha_dispatch():
    img = np.full((16, 16, 3), 255, dtype=np.uint8)
    img[4:12, 4:12] = 30
    a_chrom = key_alpha(img, (255, 255, 255), mode="chromatic")
    a_lum = key_alpha(img, (255, 255, 255), mode="luminance")
    # luminance separates, chromatic does not (white→black both grey-axis)
    assert a_lum[8, 8] > 0.9
    assert a_chrom[8, 8] > 0.9
