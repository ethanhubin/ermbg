"""Tests for chromatic-key α + small-component merge."""

from __future__ import annotations

import numpy as np

from ermbg.keyer import (
    KeyerThresholds,
    chromatic_key_alpha,
    gate_alpha_by_keyer,
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


def test_merge_patches_when_matting_only_has_halo_leak():
    """A small subject the matting net misses, but where its α has a tiny
    halo bleed into the region (e.g. α≈0.05 on a few pixels), should still
    be patched. The 'coverage' rule looks at how much of the component
    matting marks as foreground, not whether *any* pixel is non-zero."""
    h, w = 96, 96
    matting = np.zeros((h, w), dtype=np.float32)
    matting[10:50, 10:50] = 1.0    # main blob, fully captured
    matting[60:70, 60:70] = 0.05   # missed small subject, only halo leak

    chrom = np.zeros((h, w), dtype=np.float32)
    chrom[10:50, 10:50] = 1.0
    chrom[60:70, 60:70] = 1.0      # keyer sees both

    merged, info = merge_alpha_components(matting, chrom)
    assert info["patched_components"] == 1
    assert merged[60:70, 60:70].mean() > 0.5


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


def test_gate_caps_halo_when_keyer_says_bg():
    """If matting α has wide low-α halo on pixels the keyer calls bg, gate
    should cap them to ≈0."""
    h, w = 64, 64
    matting = np.zeros((h, w), dtype=np.float32)
    matting[20:44, 20:44] = 1.0           # solid subject
    matting[10:20, 10:54] = 0.20          # wide halo above
    key = np.zeros((h, w), dtype=np.float32)
    key[20:44, 20:44] = 1.0               # keyer agrees on solid
    # halo region: keyer says α=0 (it's pure bg color)

    gated, info = gate_alpha_by_keyer(matting, key)
    assert info["pixels_gated"] > 0
    # halo region should be near-0 now
    assert gated[10:20, 10:54].max() < 0.1
    # solid region untouched
    np.testing.assert_array_equal(gated[20:44, 20:44], matting[20:44, 20:44])


def test_gate_protects_high_alpha_against_keyer_disagreement():
    """If matting α is high (foreground) but keyer says bg (e.g. a hair
    against bg color), gate must NOT pull it down. fg_protect_threshold
    is the safety net."""
    h, w = 32, 32
    matting = np.full((h, w), 0.92, dtype=np.float32)  # confident fg everywhere
    key = np.zeros((h, w), dtype=np.float32)            # keyer thinks all bg

    gated, info = gate_alpha_by_keyer(matting, key)
    np.testing.assert_array_equal(gated, matting)
    assert info["pixels_gated"] == 0


def test_gate_preserves_genuine_soft_edges():
    """A real soft edge (matting α∈(0.2, 0.8), key α also non-trivial) should
    survive — keyer's bg confidence is too low to gate."""
    h, w = 32, 32
    matting = np.full((h, w), 0.5, dtype=np.float32)
    key = np.full((h, w), 0.4, dtype=np.float32)  # keyer disagrees but isn't bg-confident

    gated, info = gate_alpha_by_keyer(matting, key)
    # nothing gated because key α >= bg_confidence_threshold (0.08 default)
    assert info["pixels_gated"] == 0
    np.testing.assert_array_equal(gated, matting)
