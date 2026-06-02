"""Experimental PyMatting known-background alpha refinement tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ermbg import io
from ermbg.api import matte_image
from ermbg.pymatting_refine import (
    build_known_background_trimap,
    estimate_known_background_alpha_with_pymatting,
    estimate_stable_background_color,
    normalize_known_background_field,
)
from ermbg.solid_graphic import analyze_solid_bg_graphic

pytestmark = pytest.mark.core

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    # Treating the clean center as unknown lets closed-form smooth foreground
    # across the hole, but the enclosed edge still follows the same
    # transition/unknown ownership standard as exterior shadow.
    dist_to_subject = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
    clean_center = inner & (dist_to_subject >= 8.0)
    transition_edge = inner & (dist_to_subject < 8.0)
    assert info["enclosed_bg_pixels"] >= int(inner.sum() * 0.95)
    assert info["largest_enclosed_bg_component"] >= int(inner.sum() * 0.95)
    assert trimap.sure_bg[clean_center].mean() > 0.95
    assert trimap.unknown[transition_edge].mean() > 0.20


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


def test_known_background_trimap_uses_image_adaptive_foreground_threshold():
    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_hard_lite_shadow/green.png"
    )
    image = np.array(Image.open(path).convert("RGB"))

    trimap, info = build_known_background_trimap(
        image,
        (0, 200, 0),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    # Mechanism: fg_threshold is a subject-recall hint, not the edge-residue
    # filter. It should follow the background/subject separation valley so hard
    # UI structure remains anchored while a later local pass handles screen
    # colored pinpricks.
    assert info["fg_threshold_source"] == "histogram_otsu_seed_guard"
    assert info["fg_threshold_effective"] < 30.0
    assert info["fg_threshold_seed_pixels"] >= info["fg_threshold_min_seed_pixels"]
    assert info["fg_threshold_largest_seed_component"] >= info["fg_threshold_min_largest_component"]
    assert info["sure_fg_pixels"] > 0
    assert trimap.sure_fg.sum() == info["sure_fg_pixels"]


def test_known_background_trimap_can_lower_foreground_threshold_for_weak_contrast_ui():
    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_green_yellow_b_unoutlined_hard_lite_shadow/green.png"
    )
    image = np.array(Image.open(path).convert("RGB"))

    trimap, info = build_known_background_trimap(
        image,
        (0, 200, 0),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    # Mechanism: a fixed foreground floor can be too high for weak/unoutlined
    # controls. The adaptive seed guard may lower the threshold when the image
    # distribution proves a coherent foreground anchor below the old default.
    assert info["fg_threshold_effective"] < 30.0
    assert info["fg_threshold_seed_pixels"] >= info["fg_threshold_min_seed_pixels"]
    assert info["fg_threshold_largest_seed_component"] >= info["fg_threshold_min_largest_component"]
    assert trimap.sure_fg.sum() > 0


def test_known_background_trimap_leaves_scalar_shadow_for_shadow_patch():
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
    # not subject ownership. The trimap should expose that full area as unknown
    # repair domain so ShadowPatch can reproject it against the original source.
    assert info["shadow_background"]["pixels"] >= int(shadow.sum() * 0.8)
    assert info["shadow_background"]["unknown_ownership_pixels"] >= int(shadow.sum() * 0.8)
    assert trimap.unknown[shadow].mean() > 0.8
    assert trimap.sure_fg[shadow].mean() == 0.0


def test_known_background_trimap_keeps_weak_known_b_shadow_tail_unknown():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    image[28:68, 38:88] = (230, 210, 20)
    # Mechanism: known-B is now unified, so a coherent near-subject scalar
    # darkening against that B is shadow-tail evidence even when it is only one
    # 8-bit step away from the background. Trimap must leave it unknown for the
    # reconstruction stages instead of pinning it to sure-BG.
    weak_tail = np.zeros((96, 128), dtype=bool)
    weak_tail[69:74, 44:94] = True
    image[weak_tail] = (0, 199, 0)

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    assert info["protected_transition_pixels"] >= int(weak_tail.sum() * 0.95)
    assert trimap.unknown[weak_tail].mean() > 0.95
    assert trimap.sure_bg[weak_tail].mean() == 0.0
    assert trimap.sure_fg[weak_tail].mean() == 0.0


def test_background_normalization_preserves_visible_shadow_tail():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((96, 144, 3), bg, dtype=np.uint8)
    # Mechanism: low-frequency screen drift should be normalized, but a
    # measurable black-screen shadow tail is transferable image content. The
    # normalization gate protects visible display-shadow alpha and fades
    # smoothly through the sub-visible range instead of flattening the tail.
    image[..., 1] = 198
    shadow = np.zeros((96, 144), dtype=bool)
    shadow[48:72, 42:114] = True
    image[shadow] = (0, 184, 0)

    normalized, info = normalize_known_background_field(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )

    assert info["applied"] is True
    assert info["shadow_normalization_gate"]["protected_pixels"] >= int(shadow.sum() * 0.95)
    assert int(np.median(normalized[~shadow, 1])) == 200
    assert int(np.median(normalized[shadow, 1])) <= 186


def test_known_background_color_prefers_boundary_support_near_unknown():
    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_hole_yellow_ring_green/green.png"
    )
    image = np.array(Image.open(path).convert("RGB"))

    bg, info = estimate_stable_background_color(image)

    assert info["accepted"] is True
    assert info["source"] == "sure_bg_mode"
    assert info["seed"]["accepted"] is True
    assert info["sure_bg_pixels"] > 0
    assert info["known_bg_source"] == "boundary_support_quantized_mode"
    assert info["color_support_source"] == "support_boundary_near_unknown"
    assert info["color_support_pixels"] < info["support_pixels"]
    assert bg == tuple(info["background_color"])
    assert bg == (3, 194, 8)


def test_background_normalization_starts_on_any_sure_bg_mismatch():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((64, 64, 3), bg, dtype=np.uint8)
    image[0, 0] = (0, 199, 0)

    normalized, info = normalize_known_background_field(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )

    assert info["applied"] is True
    assert info["changed_bg_pixels"] == 1
    assert tuple(int(c) for c in normalized[0, 0]) == (0, 200, 0)


def test_background_normalization_makes_b055_sure_bg_exact_for_exact_trimap():
    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_hole_yellow_ring_green/green.png"
    )
    image = np.array(Image.open(path).convert("RGB"))
    bg = np.array([0, 200, 0], dtype=np.uint8)

    normalized, normalization = normalize_known_background_field(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )
    trimap, trimap_info = build_known_background_trimap(
        normalized,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        adaptive=True,
    )

    exact_known_bg = np.all(normalized == bg.reshape(1, 1, 3), axis=2)
    assert normalization["applied"] is True
    assert normalization["sure_bg_normalization_pixels"] > 200_000
    assert normalization["protected_transition_pixels"] > 10_000
    assert int(exact_known_bg.sum()) == normalization["sure_bg_normalization_pixels"]
    assert trimap_info["clean_bg_threshold"] == "exact_known_b"
    assert trimap_info["sure_bg_pixels"] > 180_000
    assert trimap_info["unknown_pixels"] < 50_000


def test_b055_hole_shadow_uses_same_unknown_standard_as_exterior_shadow():
    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_hole_yellow_ring_green/green.png"
    )
    image = np.array(Image.open(path).convert("RGB"))
    bg, bg_info = estimate_stable_background_color(image)
    bg_arr = np.asarray(bg, dtype=np.uint8)

    normalized, normalization = normalize_known_background_field(
        image,
        bg,
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )
    trimap, trimap_info = build_known_background_trimap(
        normalized,
        bg,
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        adaptive=True,
    )

    # B055's transparent center is an enclosed background component. The dark
    # inner-wall falloff is still source shadow evidence, so it must not get the
    # enclosed-bg shortcut into sure-BG.
    x1, y1, x2, y2 = 178, 172, 347, 333
    hole = np.zeros(image.shape[:2], dtype=bool)
    hole[y1:y2, x1:x2] = True
    screen_darker = (
        hole
        & (image[..., 1].astype(np.int16) < int(bg_arr[1]) - 1)
        & (image[..., 1] >= image[..., 0])
        & (image[..., 1] >= image[..., 2])
    )

    assert bg_info["accepted"] is True
    assert normalization["ownership"]["enclosed_bg_pixels"] > 20_000
    assert trimap_info["enclosed_bg_pixels"] > 20_000
    assert int(screen_darker.sum()) > 4_000
    assert trimap.unknown[screen_darker].mean() > 0.85
    assert trimap.sure_bg[screen_darker].mean() < 0.15


def test_known_background_trimap_protects_screen_neutral_metal_grooves_from_shadow_growth():
    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_hole_ornate_plate_blue/blue.png"
    )
    image = np.array(Image.open(path).convert("RGB"))

    trimap, info = build_known_background_trimap(
        image,
        (0, 40, 250),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    support_info = info["subject_material_support"]
    shadow_info = info["shadow_background"]
    # Mechanism: the production trimap now protects ornate metal with a local
    # material core instead of letting the shadow-growth mask own foreground.
    assert support_info["policy"] == "local_material_core_extra_inset"
    assert support_info["support_pixels"] > 40000
    assert shadow_info["hard_ownership_pixels"] == 0
    assert shadow_info["unknown_ownership_pixels"] > 20000
    assert not np.any(trimap.sure_fg & trimap.sure_bg)


def test_pymatting_known_b_keeps_protected_metal_grooves_opaque():
    import cv2

    from ermbg.colorspace import oklab_distance, srgb_to_oklab
    from ermbg.pymatting_refine import (
        _flood_from_border,
        _known_background_shadow_like_background_mask,
        _screen_dominant_shadow_pixels,
    )

    path = (
        PROJECT_ROOT
        / "samples/corridorkey_semantic/button/button_hole_ornate_plate_blue/blue.png"
    )
    image = np.array(Image.open(path).convert("RGB"))
    bg = np.array([0, 40, 250], dtype=np.uint8)
    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    lab = srgb_to_oklab(image)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0]
    distance = oklab_distance(lab, bg_lab)
    exterior = _flood_from_border(distance <= info["bg_threshold_effective"])
    dist_to_exterior = cv2.distanceTransform((~exterior).astype(np.uint8), cv2.DIST_L2, 3)
    initial_sure_fg = (distance >= info["fg_threshold_effective"]) & (
        dist_to_exterior > float(info["boundary_band_px_effective"])
    )
    shadow_bg, _ = _known_background_shadow_like_background_mask(image, bg, subject_seed=initial_sure_fg)
    protected = shadow_bg & initial_sure_fg & ~_screen_dominant_shadow_pixels(image, bg)

    result = matte_image(
        path,
        backend="pymatting-known-b",
        shadow_mode="on",
        pymatting_bg_source="custom",
        pymatting_bg_color=tuple(int(c) for c in bg),
        pymatting_fg_threshold=24.0,
    )

    # Mechanism: B056-like ornate metal has near-black grooves connected to a
    # true cast shadow. Those pixels should be available to ShadowPatch as
    # support evidence, but the final subject alpha must remain opaque because
    # same-background reprojection cannot justify eating screen-neutral metal.
    assert int(protected.sum()) > 1000
    assert not np.any(trimap.sure_bg[protected])
    assert float(np.percentile(result.alpha[protected], 10.0)) > 0.98


def test_pymatting_known_b_adaptive_foreground_threshold_removes_dark_screen_edge_residue():
    cases = [
        "button_green_yellow_a_outlined_soft_lite_shadow",
        "button_green_yellow_a_outlined_soft_heavy_shadow",
        "button_hole_yellow_ring_green",
    ]
    for case_id in cases:
        path = PROJECT_ROOT / f"samples/corridorkey_semantic/button/{case_id}/green.png"
        result = matte_image(
            path,
            backend="pymatting-known-b",
            shadow_mode="on",
            pymatting_bg_source="custom",
            pymatting_bg_color=(0, 200, 0),
            pymatting_fg_threshold=24.0,
        )
        rgba = result.rgba
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        dark_screen_edge_residue = (
            (alpha > 0.5)
            & (rgb[..., 1] > np.maximum(rgb[..., 0], rgb[..., 2]) + 8.0)
            & (rgb[..., 1] < 120.0)
        )

        # Mechanism: these pixels used to be pinned as alpha=1 foreground by a
        # fixed trimap threshold, so foreground unmixing exported source-green
        # edge dots. Adaptive seeds leave them for the solver instead.
        residue_budget = max(24, int(round(float(alpha.size) * 0.0012)))
        assert int(dark_screen_edge_residue.sum()) <= residue_budget, case_id


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
