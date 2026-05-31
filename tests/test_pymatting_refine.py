"""Experimental PyMatting known-background alpha refinement tests."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg import io
from ermbg.pymatting_refine import (
    build_known_background_trimap,
    estimate_known_background_alpha_with_pymatting,
)
from ermbg.solid_graphic import analyze_solid_bg_graphic

pytestmark = pytest.mark.core


def _aa_disc_case(size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([220, 40, 30], dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    radius = float(size) * 0.28
    signed = radius - np.sqrt((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2)
    # Mechanism: a hard opaque shape rendered onto known B with only a narrow
    # antialiasing ramp is the class PyMatting should be judged on first.
    alpha = np.clip((signed + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

    f_lin = io.srgb_to_linear(np.broadcast_to(fg, (size, size, 3)))
    b_lin = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    image = io.linear_to_srgb_u8(alpha[..., None] * f_lin + (1.0 - alpha[..., None]) * b_lin)
    return image, alpha, fg, bg


def test_known_background_pymatting_cf_recovers_hard_edge_antialiasing():
    image, alpha_gt, _, bg = _aa_disc_case()

    result = estimate_known_background_alpha_with_pymatting(
        image,
        tuple(int(c) for c in bg),
        method="cf",
        boundary_band_px=2,
    )

    edge = (alpha_gt > 0.001) & (alpha_gt < 0.999)
    err = np.abs(result.alpha - alpha_gt)
    assert result.debug["method"] == "cf"
    assert result.debug["applied"] is True
    assert result.debug["trimap"]["unknown_pixels"] > int(edge.sum() * 0.8)
    assert float(err[edge].mean()) < 0.03
    assert float(err.mean()) < 0.002


def test_known_background_trimap_keeps_only_exterior_band_unknown():
    image, _, _, bg = _aa_disc_case()

    trimap, info = build_known_background_trimap(image, tuple(int(c) for c in bg), boundary_band_px=2)

    assert info["sure_fg_pixels"] > 0
    assert info["sure_bg_pixels"] > 0
    assert info["unknown_pixels"] > 0
    assert not np.any(trimap.sure_fg & trimap.sure_bg)
    assert not np.any(trimap.unknown & (trimap.sure_fg | trimap.sure_bg))
    assert np.all(trimap.sure_fg | trimap.sure_bg | trimap.unknown)


def test_known_background_trimap_marks_enclosed_same_bg_as_sure_background():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((96, 96, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    outer = (xx - 48) ** 2 + (yy - 48) ** 2 <= 34**2
    inner = (xx - 48) ** 2 + (yy - 48) ** 2 <= 17**2
    image[outer] = (230, 210, 20)
    image[inner] = bg

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    # The center is a same-background cutout fully enclosed by subject pixels.
    # Treating it as unknown lets closed-form smooth foreground across the hole.
    assert info["enclosed_bg_pixels"] >= int(inner.sum() * 0.95)
    assert info["largest_enclosed_bg_component"] >= int(inner.sum() * 0.95)
    assert trimap.sure_bg[inner].mean() > 0.95
    assert trimap.unknown[inner].mean() < 0.05


def test_known_background_trimap_allows_broad_ui_antialias_band():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([40, 110, 245], dtype=np.uint8)
    h, w = 128, 256
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dist = np.minimum.reduce([xx - 56, 200 - xx, yy - 42, 82 - yy])
    # Mechanism: generated UI controls can have a several-pixel antialias /
    # contact-shadow transition. If the adaptive cap is too tight, those pixels
    # become hard foreground/background constraints before PyMatting can solve
    # a smooth edge.
    alpha = np.clip((dist + 8.0) / 16.0, 0.0, 1.0).astype(np.float32)
    image = (alpha[..., None] * fg.reshape(1, 1, 3) + (1.0 - alpha[..., None]) * bg.reshape(1, 1, 3)).astype(
        np.uint8
    )

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    assert info["boundary_transition_distance_p90"] >= 6.0
    assert info["boundary_band_px_effective"] >= 6
    assert trimap.unknown.sum() > 0


def test_known_background_trimap_marks_scalar_shadow_as_background():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((128, 200, 3), bg, dtype=np.uint8)
    subject = np.zeros((128, 200), dtype=bool)
    subject[30:86, 36:154] = True
    shadow = np.zeros((128, 200), dtype=bool)
    shadow[90:108, 44:166] = True
    image[subject] = (40, 110, 245)
    image[shadow] = (0, 128, 0)

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    # Mechanism: scalar-darkened known-B near a UI control is shadow behavior,
    # not subject ownership. If PyMatting sees it as unknown/foreground, it can
    # return colored semi-transparent subject pixels that ShadowPatch cannot
    # cleanly remove later.
    assert info["shadow_background"]["pixels"] >= int(shadow.sum() * 0.8)
    assert trimap.sure_bg[shadow].mean() > 0.8
    assert trimap.sure_fg[shadow].mean() == 0.0


def test_solid_graphic_pymatting_refiner_is_explicit_and_debugged():
    image, _, _, _ = _aa_disc_case()

    baseline = analyze_solid_bg_graphic(image)
    refined = analyze_solid_bg_graphic(image, alpha_refiner="pymatting-cf")

    assert baseline.accepted is True
    assert refined.accepted is True
    assert baseline.debug["alpha_refiner"]["used"] is False
    assert refined.debug["alpha_refiner"]["used"] is True
    assert refined.debug["alpha_refiner"]["method"] == "cf"
    assert refined.debug["mask_pixels"] == baseline.debug["mask_pixels"]
