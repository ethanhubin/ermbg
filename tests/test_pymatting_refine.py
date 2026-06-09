"""Experimental PyMatting known-background alpha refinement tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ermbg import io
from ermbg.api import _pymatting_known_b_unknown_domain_shadow_patch, matte_image
from ermbg.pymatting_refine import (
    _build_known_background_ownership,
    _known_background_hard_shadow_subject_evidence_release,
    _same_key_opaque_stroke_core_from_component,
    analyze_same_key_opaque_body_outline,
    build_known_background_hard_edge_boundary_mask,
    build_known_background_trimap,
    build_same_key_opaque_inner_opaque_mask,
    build_same_key_opaque_proxy_subject_mask,
    estimate_known_background_alpha_with_pymatting,
    estimate_stable_background_color,
    normalize_known_background_field,
)
from ermbg.preprocess import repair_known_background_preprocess
from ermbg.solid_graphic import analyze_solid_bg_graphic

pytestmark = pytest.mark.core

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _matte_known_b_after_background_repair(
    image_or_path,
    *,
    bg_color: tuple[int, int, int],
    **kwargs,
):
    image = np.asarray(Image.open(image_or_path).convert("RGB"), dtype=np.uint8) if isinstance(image_or_path, Path) else image_or_path
    return matte_image(
        image,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=bg_color,
        **kwargs,
    )


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


def test_known_background_bg_seed_outline_finds_outer_and_hole_boundaries():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((128, 128, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:128, :128]
    outer = (xx - 64) ** 2 + (yy - 64) ** 2 <= 44**2
    inner = (xx - 64) ** 2 + (yy - 64) ** 2 <= 18**2
    core = outer & ~inner
    image[core] = (240, 190, 20)
    image[inner] = bg

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    outline = info["bg_seed_outline"]
    assert outline["accepted"] is True
    assert outline["exterior_outline_pixels"] > 0
    assert outline["hole_seed_pixels"] > 0
    assert outline["hole_outline_pixels"] > 0
    assert trimap.sure_fg[core].mean() > 0.45
    assert trimap.sure_fg[inner].mean() == 0.0


def test_known_background_bg_seed_outline_uses_complex_hole_surface_gate_only_for_ornate_holes():
    ordinary = np.array(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
        ).convert("RGB")
    )
    ornate = np.array(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_hole_ornate_plate_blue/blue.png"
        ).convert("RGB")
    )
    large_hole = np.array(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_hole_yellow_ring_green/green.png"
        ).convert("RGB")
    )

    ordinary_trimap, ordinary_info = build_known_background_trimap(
        ordinary,
        (0, 200, 0),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        adapt_bg_threshold=True,
        adapt_fg_threshold=True,
        adapt_boundary_band=True,
    )
    ornate_normalized, _normalization = normalize_known_background_field(
        ornate,
        (0, 40, 250),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )
    ornate_trimap, ornate_info = build_known_background_trimap(
        ornate_normalized,
        (0, 40, 250),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        adapt_bg_threshold=True,
        adapt_fg_threshold=True,
        adapt_boundary_band=True,
    )
    large_hole_normalized, _large_hole_normalization = normalize_known_background_field(
        large_hole,
        (3, 194, 8),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )
    large_hole_trimap, large_hole_info = build_known_background_trimap(
        large_hole_normalized,
        (3, 194, 8),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        adapt_bg_threshold=True,
        adapt_fg_threshold=True,
        adapt_boundary_band=True,
    )
    ordinary_outline = ordinary_info["bg_seed_outline"]
    ornate_outline = ornate_info["bg_seed_outline"]
    large_hole_outline = large_hole_info["bg_seed_outline"]
    assert ordinary_outline["accepted"] is True
    assert ordinary_outline["complex_hole_surface"] is False
    assert ordinary_outline["outline_source"] == "front_meets_break_or_non_passable_standard"
    assert ordinary_trimap.unknown.sum() > 0
    assert large_hole_outline["accepted"] is True
    assert large_hole_outline["complex_hole_surface"] is False
    assert large_hole_outline["outline_source"] == "front_meets_break_or_non_passable_standard"
    assert large_hole_outline["outline_component_min_area"] == 0
    assert large_hole_trimap.unknown.sum() > 0
    assert ornate_outline["accepted"] is True
    assert ornate_outline["complex_hole_surface"] is True
    assert ornate_outline["outline_source"] == "front_meets_break_or_non_passable_complex_shadow_open"
    assert ornate_outline["bg_owned_blocked_pixels"] > 0
    assert ornate_outline["outline_component_min_area"] > 0
    assert ornate_outline["exterior_outline_dropped_small_pixels"] > 0
    assert ornate_outline["hole_outline_pixels"] > 0
    assert ornate_trimap.unknown.sum() > 0


def test_known_background_trimap_consumes_enclosed_near_bg_semantic_policy():
    bg = np.array([255, 255, 255], dtype=np.uint8)
    image = np.full((96, 96, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    outer = (xx - 48) ** 2 + (yy - 48) ** 2 <= 34**2
    inner = (xx - 48) ** 2 + (yy - 48) ** 2 <= 17**2
    image[outer] = (230, 30, 30)
    image[inner] = bg

    subject_trimap, subject_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        semantic_decision={"enclosed_near_bg_policy": "subject"},
    )
    hole_trimap, hole_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        semantic_decision={"enclosed_near_bg_policy": "transparent_hole"},
    )

    dist_to_subject = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
    clean_center = inner & (dist_to_subject >= 8.0)
    assert subject_trimap.sure_fg[clean_center].mean() > 0.95
    assert subject_trimap.sure_bg[clean_center].mean() == 0.0
    assert hole_trimap.sure_bg[clean_center].mean() > 0.95
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_edge = inner & ~cv2.erode(inner.astype(np.uint8), edge_kernel, iterations=1).astype(bool)
    subject_side_edge = cv2.dilate(inner.astype(np.uint8), edge_kernel, iterations=1).astype(bool) & ~inner
    assert np.count_nonzero(hole_trimap.unknown[inner_edge]) > 0
    assert np.count_nonzero(hole_trimap.unknown[subject_side_edge]) > 0
    assert subject_info["semantic_decision"]["enclosed_near_bg_policy"] == "subject"
    assert hole_info["semantic_decision"]["enclosed_near_bg_policy"] == "transparent_hole"
    assert subject_info["semantic_decision"]["forced_subject_pixels"] >= int(clean_center.sum())
    assert hole_info["semantic_decision"]["forced_background_pixels"] >= int(clean_center.sum())
    assert hole_info["semantic_decision"]["hole_unknown_release_pixels"] > 0
    assert hole_info["semantic_decision"]["hole_unknown_release"]["components"][0]["release_px"] >= 1


def test_known_background_subject_policy_forces_internal_unknown_to_fg():
    bg = np.array([255, 255, 255], dtype=np.uint8)
    image = np.full((96, 96, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    outer = (xx - 48) ** 2 + (yy - 48) ** 2 <= 34**2
    internal_unknown = (xx - 48) ** 2 + (yy - 48) ** 2 <= 8**2
    image[outer] = (230, 30, 30)
    image[internal_unknown] = (244, 244, 244)

    auto_trimap, _ = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
    )
    subject_trimap, subject_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        semantic_decision={"enclosed_near_bg_policy": "subject"},
    )

    assert auto_trimap.unknown[internal_unknown].mean() > 0.95
    assert subject_trimap.sure_fg[internal_unknown].mean() > 0.95
    assert subject_trimap.unknown[internal_unknown].mean() == 0.0
    assert subject_info["semantic_decision"]["forced_internal_unknown_pixels"] == 0
    assert subject_info["semantic_decision"]["internal_unknown"]["components"] == 0
    assert subject_info["semantic_decision"]["subject_domain"]["method"] == "subject_domain_then_boundary_unknown"
    labels_count, labels = cv2.connectedComponents(subject_trimap.sure_fg.astype(np.uint8), 8)
    center_label = int(labels[48, 48])
    assert center_label > 0
    assert int(np.count_nonzero(labels == center_label)) >= int(internal_unknown.sum() + (outer & ~internal_unknown).sum() * 0.80)
    assert labels_count == 2
    assert subject_trimap.unknown.sum() > 0


def test_known_background_trimap_consumes_user_keep_remove_masks():
    bg = np.array([255, 255, 255], dtype=np.uint8)
    image = np.full((96, 96, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    r = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
    ring = (r <= 34) & (r >= 17)
    inner = r <= 14
    image[ring] = (230, 30, 30)

    empty_trimap, empty_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        semantic_decision={"enclosed_near_bg_policy": "transparent_hole"},
        user_keep_mask=np.zeros(inner.shape, dtype=np.float32),
    )
    keep_trimap, keep_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        semantic_decision={"enclosed_near_bg_policy": "transparent_hole"},
        user_keep_mask=inner.astype(np.float32),
    )
    remove_trimap, remove_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        semantic_decision={"enclosed_near_bg_policy": "subject"},
        user_remove_mask=inner.astype(np.float32),
    )
    conflict_trimap, conflict_info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        semantic_decision={"enclosed_near_bg_policy": "subject"},
        user_keep_mask=inner.astype(np.float32),
        user_remove_mask=inner.astype(np.float32),
    )

    assert empty_trimap.sure_bg[inner].mean() > 0.95
    assert empty_info["user_mask_decision"]["applied"] is False
    assert keep_trimap.sure_fg[inner].mean() > 0.95
    assert keep_trimap.sure_bg[inner].mean() == 0.0
    assert keep_info["user_mask_decision"]["forced_subject_pixels"] == int(inner.sum())
    assert remove_trimap.sure_bg[inner].mean() > 0.95
    assert remove_trimap.sure_fg[inner].mean() == 0.0
    assert remove_info["user_mask_decision"]["forced_background_pixels"] == int(inner.sum())
    assert conflict_trimap.sure_bg[inner].mean() > 0.95
    assert conflict_trimap.sure_fg[inner].mean() == 0.0
    assert conflict_info["user_mask_decision"]["conflict_pixels"] == int(inner.sum())


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


def test_known_background_trimap_promotes_light_neutral_connected_shadow_conflict_to_unknown():
    bg = np.array([253, 253, 253], dtype=np.uint8)
    h = w = 160
    image = np.full((h, w, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = cy = 80
    dist = np.sqrt((xx - (cx + 6)) ** 2 + (yy - (cy + 8)) ** 2)
    strength = np.clip((58.0 - dist) / 18.0, 0.0, 1.0) * 0.55
    cast_shadow = strength > 0.04
    image[cast_shadow] = np.clip(
        image[cast_shadow].astype(np.float32) * (1.0 - strength[cast_shadow, None]),
        0,
        255,
    ).astype(np.uint8)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (r2 <= 45**2) & (r2 >= 38**2)
    body = r2 < 38**2
    image[ring] = (245, 112, 4)
    image[body] = (20, 155, 38)
    visible_shadow = cast_shadow & ~(ring | body)

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
    )

    # Mechanism: on light neutral known-B, a connected scalar cast shadow can be
    # darker than the FG seed threshold. If part of that component is already
    # exterior shadow and it borders a separate colored material core, the whole
    # visible scalar component should feed the unknown domain before outline
    # tracing, not remain sure-FG.
    assert info["neutral_shadow_conflict_unknown_pixels"] > 0
    assert info["shadow_background"]["unknown_ownership_pixels"] == info["shadow_background"]["pixels"]
    assert info["bg_seed_outline"]["shadow_inward_unknown_pixels"] > 0
    assert trimap.unknown[visible_shadow].mean() > 0.98
    assert trimap.sure_fg[visible_shadow].mean() == 0.0


def test_known_background_subject_transition_requires_strong_shadow_anchor():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    image[20:76, 28:100] = (35, 120, 245)
    weak_screen_patch = np.zeros((96, 128), dtype=bool)
    weak_screen_patch[46:50, 60:68] = True
    image[weak_screen_patch] = (0, 170, 20)

    ownership = _build_known_background_ownership(
        image,
        bg,
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
        adapt_bg_threshold=True,
        adapt_fg_threshold=True,
        adapt_boundary_band=True,
        require_exact_bg=True,
    )

    # ``screen_dominant_shadow`` is only weak color-line evidence. It must not
    # start a subject-transition release by itself; otherwise same-channel drift
    # inside a hard opaque button becomes an isolated unknown island unrelated
    # to the exterior/shadow solve. The pixel may still be unknown by ordinary
    # ownership, but it is not allowed to become protected transition evidence
    # unless a real exterior/hole/shadow_unknown anchor reaches it.
    assert ownership.debug["shadow_unknown_pixels"] == 0
    assert ownership.protected_transition[weak_screen_patch].mean() == 0.0


def test_known_background_trimap_follows_connected_weak_shadow_tail_beyond_near_subject_cap():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((192, 160, 3), bg, dtype=np.uint8)
    image[28:68, 44:108] = (30, 120, 245)
    strong_shadow = np.zeros((192, 160), dtype=bool)
    strong_shadow[78:94, 48:112] = True
    weak_tail = np.zeros((192, 160), dtype=bool)
    weak_tail[94:154, 48:112] = True
    image[strong_shadow] = (0, 150, 0)
    image[weak_tail] = (0, 199, 0)

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=30.0,
        boundary_band_px=2,
    )

    shadow_info = info["shadow_background"]
    assert shadow_info["anchor_pixels"] >= int(strong_shadow.sum() * 0.9)
    assert shadow_info["connected_tail_pixels"] >= int(weak_tail.sum() * 0.8)
    assert trimap.unknown[weak_tail].mean() > 0.8
    assert trimap.sure_bg[weak_tail].mean() < 0.2


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


def test_background_normalization_does_not_clean_isolated_screen_colored_residue_without_sure_bg_connectivity():
    bg = np.array([3, 178, 10], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    subject = np.zeros((96, 128), dtype=bool)
    subject[28:58, 40:88] = True
    coherent_shadow = np.zeros((96, 128), dtype=bool)
    coherent_shadow[62:72, 44:84] = True
    isolated_residue = np.zeros((96, 128), dtype=bool)
    isolated_residue[88:90, 70:78] = True
    image[subject] = (230, 210, 20)
    image[coherent_shadow] = (3, 150, 2)
    image[isolated_residue] = (6, 145, 10)

    normalized, info = normalize_known_background_field(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )

    cleanup = info["isolated_bg_residue_cleanup"]
    assert info["applied"] is False
    assert info["reason"] == "sure background already matches known-B"
    assert cleanup["enabled"] is False
    assert cleanup["reason"] == "disabled: background normalization requires connected sure-bg evidence"
    assert cleanup["cleaned_pixels"] == 0
    assert info["isolated_bg_residue_cleanup_pixels"] == 0
    np.testing.assert_array_equal(normalized[isolated_residue], image[isolated_residue])
    np.testing.assert_array_equal(normalized[coherent_shadow], image[coherent_shadow])


def test_background_normalization_does_not_clean_subject_adjacent_dark_screen_material():
    bg = np.array([0, 40, 250], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    subject = np.zeros((96, 128), dtype=bool)
    subject[28:58, 40:88] = True
    dark_groove = np.zeros((96, 128), dtype=bool)
    dark_groove[58:60, 60:68] = True
    image[subject] = (230, 210, 20)
    # This color lies on the same blue-screen darkening line as a shadow, but
    # it is only a few pixels from the eroded subject seed. It models black/blue
    # antialias grooves in hard UI material, which cleanup must not repaint.
    image[dark_groove] = (0, 20, 120)

    normalized, info = normalize_known_background_field(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )

    cleanup = info["isolated_bg_residue_cleanup"]
    assert cleanup["enabled"] is False
    assert cleanup["reason"] == "disabled: background normalization requires connected sure-bg evidence"
    assert info["isolated_bg_residue_cleanup_pixels"] == 0
    np.testing.assert_array_equal(normalized[dark_groove], image[dark_groove])


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


def test_stable_background_refines_route_seed_without_subject_dominant_takeover():
    image = np.full((148, 307, 3), (5, 132, 250), dtype=np.uint8)
    cv2.rectangle(image, (4, 4), (302, 124), (253, 130, 4), -1, cv2.LINE_AA)
    cv2.rectangle(image, (4, 4), (302, 124), (255, 220, 80), 3, cv2.LINE_AA)
    cv2.rectangle(image, (8, 126), (299, 143), (140, 70, 5), -1, cv2.LINE_AA)

    bg, info = estimate_stable_background_color(
        image,
        seed_bg=(5, 132, 250),
        seed_source="route_screen_analysis",
        seed_info={"screen_mode": "blue", "background_confidence": 0.63},
    )

    assert info["accepted"] is True
    assert info["source"] == "sure_bg_mode"
    assert info["seed"]["source"] == "route_screen_analysis"
    assert info["bg_threshold_source"] == "external_seed_cap"
    assert info["bg_threshold_effective"] <= 24.0
    assert bg == tuple(info["background_color"])
    assert max(abs(int(a) - int(b)) for a, b in zip(bg, (5, 132, 250))) <= 4


def test_stable_background_accepts_smooth_low_chroma_corner_drift():
    h = w = 64
    yy = np.linspace(0.0, 16.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 16.0, w, dtype=np.float32)[None, :]
    gray = 154.0 + (xx + yy) * 0.5
    image = np.dstack([gray, gray, gray + 2.0]).astype(np.uint8)
    image[20:46, 22:42] = (220, 40, 30)

    bg, info = estimate_stable_background_color(image)

    assert info["accepted"] is True
    assert info["seed"]["source"] == "corners"
    assert 4.0 < info["seed"]["corner_agreement"] <= 6.0
    assert info["seed"]["sigma"] <= 6.0
    assert bg == tuple(info["background_color"])


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


def test_background_normalization_does_not_repaint_enclosed_subject_white_material():
    bg = np.array([254, 253, 254], dtype=np.uint8)
    image = np.full((128, 128, 3), bg, dtype=np.uint8)
    cv2.circle(image, (64, 64), 42, (36, 24, 18), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 38, (245, 130, 20), -1, cv2.LINE_AA)
    marking_u8 = np.zeros((128, 128), dtype=np.uint8)
    cv2.ellipse(marking_u8, (55, 66), (18, 12), -20, 0, 360, 255, -1, cv2.LINE_AA)
    marking = marking_u8 > 0
    image[marking] = (255, 255, 255)
    # Add a tiny exterior mismatch so the normalization pass has real exterior
    # work to do; the enclosed white marking must still stay untouched.
    image[0, 0] = (253, 253, 254)

    normalized, info = normalize_known_background_field(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )

    assert info["normalization_scope"] == "neutral_exterior_connected_sure_bg_only"
    assert info["enclosed_sure_bg_excluded_pixels"] > 0
    assert tuple(int(c) for c in normalized[0, 0]) == tuple(int(c) for c in bg)
    np.testing.assert_array_equal(normalized[marking], image[marking])


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
        adapt_bg_threshold=True,
        adapt_fg_threshold=True,
        adapt_boundary_band=True,
    )

    exact_known_bg = np.all(normalized == bg.reshape(1, 1, 3), axis=2)
    assert normalization["applied"] is True
    assert normalization["sure_bg_normalization_pixels"] > 200_000
    assert normalization["protected_transition_pixels"] > 10_000
    assert int(exact_known_bg.sum()) == (
        normalization["sure_bg_normalization_pixels"] + normalization["isolated_bg_residue_cleanup_pixels"]
    )
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
        adapt_bg_threshold=True,
        adapt_fg_threshold=True,
        adapt_boundary_band=True,
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
        result = _matte_known_b_after_background_repair(
            path,
            bg_color=(0, 200, 0),
            shadow_mode="on",
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


def test_known_background_trimap_releases_subject_evidence_only_for_hard_shadow_gap():
    cases = {
        "button_green_yellow_a_outlined_hard_lite_shadow": 0,
        "button_green_yellow_a_outlined_hard_heavy_shadow": 1000,
        "button_green_yellow_b_unoutlined_hard_heavy_shadow": 0,
    }
    for case_id, min_release_pixels in cases.items():
        path = PROJECT_ROOT / f"samples/corridorkey_semantic/button/{case_id}/green.png"
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        normalized, _normalization = normalize_known_background_field(
            image,
            (0, 200, 0),
            bg_threshold=3.5,
            fg_threshold=24.0,
            adaptive=True,
        )
        _, info = build_known_background_trimap(
            normalized,
            (0, 200, 0),
            bg_threshold=3.5,
            fg_threshold=24.0,
            boundary_band_px=2,
        )

        released = int(info["hard_shadow_subject_evidence_release_pixels"])
        if min_release_pixels:
            assert released >= min_release_pixels, case_id
            assert info["hard_shadow_subject_evidence"]["components"][0]["keep"] is True
            assert info["hard_shadow_subject_evidence"]["components"][0]["release_fraction_of_adjacent_subject"] < 0.30
        else:
            assert released == 0, case_id


def test_known_background_trimap_releases_neutral_subject_edge_when_shadow_evidence_exists():
    bg = np.array([253, 253, 253], dtype=np.uint8)
    image = np.full((128, 160, 3), bg, dtype=np.uint8)
    subject = np.zeros((128, 160), dtype=bool)
    subject[34:82, 44:108] = True
    shadow = np.zeros((128, 160), dtype=bool)
    shadow[78:96, 50:118] = True
    image[subject] = (18, 105, 246)
    image[shadow] = (230, 230, 230)

    trimap, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
    )

    release = info["neutral_shadow_subject_evidence"]
    released = int(info["neutral_shadow_subject_evidence_release_pixels"])
    assert release["enabled"] is True
    assert release["release_px"] <= 5
    assert released > 0
    assert info["shadow_background"]["unknown_ownership_pixels"] >= int(shadow.sum() * 0.8)
    assert trimap.unknown[shadow].mean() > 0.8
    assert not np.any(trimap.sure_fg & trimap.sure_bg)


def test_known_background_trimap_does_not_release_neutral_subject_edge_without_shadow():
    bg = np.array([253, 253, 253], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    image[28:68, 42:86] = (18, 105, 246)

    _, info = build_known_background_trimap(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
    )

    assert info["neutral_shadow_subject_evidence_release_pixels"] == 0
    assert info["neutral_shadow_subject_evidence"]["reason"] == "missing sure foreground or shadow evidence"


def test_known_background_hard_edge_boundary_crosses_cast_shadow_by_family():
    cases = [
        (
            "button_green_yellow_a_outlined_no_shadow/green.png",
            "button_green_yellow_a_outlined_soft_heavy_shadow/green.png",
            (0, 200, 0),
        ),
        (
            "button_green_yellow_b_unoutlined_no_shadow/green.png",
            "button_green_yellow_b_unoutlined_soft_heavy_shadow/green.png",
            (0, 200, 0),
        ),
        (
            "button_blue_green_a_outlined_no_shadow/blue.png",
            "button_blue_green_a_outlined_soft_heavy_shadow/blue.png",
            (0, 0, 200),
        ),
        (
            "button_blue_green_b_unoutlined_no_shadow/blue.png",
            "button_blue_green_b_unoutlined_soft_heavy_shadow/blue.png",
            (0, 0, 200),
        ),
        (
            "button_green_blue_a_outlined_no_shadow/green.png",
            "button_green_blue_a_outlined_soft_heavy_shadow/green.png",
            (0, 200, 0),
        ),
        (
            "button_green_blue_b_unoutlined_no_shadow/green.png",
            "button_green_blue_b_unoutlined_soft_heavy_shadow/green.png",
            (0, 200, 0),
        ),
    ]
    for clean_rel, shadow_rel, bg in cases:
        clean_image = np.asarray(
            Image.open(PROJECT_ROOT / f"samples/corridorkey_semantic/button/{clean_rel}").convert("RGB")
        )
        shadow_image = np.asarray(
            Image.open(PROJECT_ROOT / f"samples/corridorkey_semantic/button/{shadow_rel}").convert("RGB")
        )

        clean_mask, clean_info = build_known_background_hard_edge_boundary_mask(clean_image, bg)
        shadow_mask, shadow_info = build_known_background_hard_edge_boundary_mask(shadow_image, bg)

        intersection = int((clean_mask & shadow_mask).sum())
        union = int((clean_mask | shadow_mask).sum())
        iou = intersection / float(max(1, union))
        area_delta = abs(int(clean_mask.sum()) - int(shadow_mask.sum())) / float(max(1, int(clean_mask.sum())))

        assert clean_info["accepted"] is True, clean_rel
        assert shadow_info["accepted"] is True, shadow_rel
        assert shadow_info["shadow_bg_pixels"] > clean_info["shadow_bg_pixels"], shadow_rel
        assert iou >= 0.94, (clean_rel, shadow_rel, iou)
        assert area_delta <= 0.08, (clean_rel, shadow_rel, area_delta)


def test_same_key_opaque_body_outline_trimap_uses_measured_outline_evidence():
    bg = (1, 95, 248)
    image = np.full((120, 240, 3), bg, dtype=np.uint8)
    cv2.rectangle(image, (20, 12), (220, 98), (112, 160, 248), -1, cv2.LINE_AA)
    cv2.rectangle(image, (20, 12), (220, 98), (70, 118, 210), 2, cv2.LINE_AA)
    cv2.rectangle(image, (22, 99), (218, 108), (6, 74, 188), -1)
    cv2.line(image, (22, 98), (218, 98), (92, 126, 170), 1, cv2.LINE_AA)

    outline = analyze_same_key_opaque_body_outline(image, bg, bg_threshold=3.5)
    trimap, info = build_known_background_trimap(
        image,
        bg,
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        trimap_mode="same_key_opaque_body_outline",
        unknown_grow_px=2,
    )

    assert outline["accepted"] is True
    assert outline["outline_recipe"] == "lower_perimeter_ridge"
    assert info["method"] == "same_key_opaque_body_outline"
    assert info["same_key_opaque_body_outline"]["accepted"] is True
    assert trimap.sure_fg[40:90, 40:200].mean() > 0.95
    assert trimap.unknown[99:108, 40:200].mean() > 0.80
    assert not np.any(trimap.sure_fg & trimap.sure_bg)


def test_same_key_opaque_body_outline_trimap_supports_closed_plateau_shapes():
    bg = (1, 95, 248)
    image = np.full((128, 128, 3), bg, dtype=np.uint8)
    cv2.circle(image, (64, 64), 39, (112, 160, 248), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 42, (40, 88, 208), 2, cv2.LINE_AA)

    outline = analyze_same_key_opaque_body_outline(image, bg, bg_threshold=3.5)
    trimap, info = build_known_background_trimap(
        image,
        bg,
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        trimap_mode="same_key_opaque_body_outline",
        unknown_grow_px=2,
    )

    yy, xx = np.indices((128, 128))
    center = (xx - 64) ** 2 + (yy - 64) ** 2 <= 30**2
    edge = ((xx - 64) ** 2 + (yy - 64) ** 2 >= 39**2) & ((xx - 64) ** 2 + (yy - 64) ** 2 <= 45**2)
    assert outline["accepted"] is True
    assert outline["outline_recipe"] == "closed_plateau_outline"
    assert info["same_key_opaque_body_outline"]["outline_recipe"] == "closed_plateau_outline"
    assert trimap.sure_fg[center].mean() > 0.95
    assert trimap.unknown[edge].mean() > 0.30
    assert not np.any(trimap.sure_fg & trimap.sure_bg)


def test_same_key_opaque_body_outline_rejects_internal_known_b_holes():
    bg = (0, 200, 0)
    image = np.full((128, 128, 3), bg, dtype=np.uint8)
    cv2.circle(image, (64, 64), 44, (235, 202, 28), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 46, (90, 90, 20), 2, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 18, bg, -1, cv2.LINE_AA)

    outline = analyze_same_key_opaque_body_outline(image, bg, bg_threshold=3.5)

    assert outline["accepted"] is False
    assert outline["reason"] == "body outline contains enclosed known-background holes"
    assert outline["internal_clean_bg_holes"]["pixels"] > 0


def test_same_key_opaque_proxy_subject_mask_expands_antialias_coverage():
    bg = (1, 95, 248)
    image = np.full((128, 128, 3), bg, dtype=np.uint8)
    cv2.circle(image, (64, 64), 39, (112, 160, 248), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 42, (40, 88, 208), 2, cv2.LINE_AA)

    base_mask, base_info = build_same_key_opaque_proxy_subject_mask(
        image,
        bg,
        bg_threshold=3.5,
        expand_px=0,
    )
    expanded_mask, expanded_info = build_same_key_opaque_proxy_subject_mask(
        image,
        bg,
        bg_threshold=3.5,
        expand_px=1,
    )

    assert base_info["accepted"] is True
    assert expanded_info["accepted"] is True
    assert expanded_info["expand_px"] == 1
    assert int(expanded_mask.sum()) > int(base_mask.sum())
    assert int((expanded_mask & ~base_mask).sum()) == expanded_info["expanded_pixels"]


def test_same_key_opaque_proxy_subject_mask_measures_variable_stroke_widths():
    bg = (1, 95, 248)
    measured: list[int] = []
    core_pixels: list[int] = []
    for stroke_px in (2, 5):
        image = np.full((160, 160, 3), bg, dtype=np.uint8)
        component_u8 = np.zeros((160, 160), dtype=np.uint8)
        cv2.circle(component_u8, (80, 80), 58, 1, -1, cv2.LINE_AA)
        cv2.circle(image, (80, 80), 58, (40, 88, 208), -1, cv2.LINE_AA)
        cv2.circle(image, (80, 80), 58 - stroke_px, (112, 160, 248), -1, cv2.LINE_AA)

        core, info = _same_key_opaque_stroke_core_from_component(image, component_u8.astype(bool))
        measured.append(info["stroke_inset_px"])
        core_pixels.append(int(core.sum()))

    assert measured == [2, 5]
    assert core_pixels[1] < core_pixels[0]


def test_same_key_opaque_inner_opaque_mask_keeps_non_proxy_stroke_material():
    bg = (1, 95, 248)
    image = np.full((160, 160, 3), bg, dtype=np.uint8)
    cv2.circle(image, (80, 80), 58, (40, 88, 208), -1, cv2.LINE_AA)
    cv2.circle(image, (80, 80), 53, (112, 160, 248), -1, cv2.LINE_AA)

    proxy_mask, proxy_info = build_same_key_opaque_proxy_subject_mask(
        image,
        bg,
        bg_threshold=3.5,
        expand_px=0,
    )
    inner_mask, inner_info = build_same_key_opaque_inner_opaque_mask(
        image,
        bg,
        bg_threshold=3.5,
        outer_guard_px=1.0,
    )

    extra_inner_material = inner_mask & ~proxy_mask
    assert proxy_info["accepted"] is True
    assert inner_info["enabled"] is True
    assert inner_info["outer_guard_pixels"] > 0
    assert int(inner_mask.sum()) > int(proxy_mask.sum())
    assert int(extra_inner_material.sum()) > 100


def test_same_key_opaque_pymatting_uses_proxy_subject_mask_for_body_outline_solve():
    bg = (1, 95, 248)
    image = np.full((128, 128, 3), bg, dtype=np.uint8)
    cv2.circle(image, (64, 64), 39, (112, 160, 248), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 42, (40, 88, 208), 2, cv2.LINE_AA)

    result = matte_image(
        image,
        backend="pymatting-known-b",
        shadow_mode="off",
        pymatting_bg_source="custom",
        pymatting_bg_color=bg,
        pymatting_trimap_mode="same_key_opaque_body_outline",
    )

    proxy_info = result.debug["pymatting_known_b"]["same_key_proxy_subject"]
    assert proxy_info["enabled"] is True
    assert proxy_info["expand_px"] == 0
    assert proxy_info["proxy_color"] == [254, 160, 7]
    assert proxy_info["proxy_color_source"] == "background_complement"
    assert proxy_info["solver_trimap_mode"] == "same_key_opaque_body_outline"
    assert result.report["strategy"]["extras"]["parameters"]["trimap_mode"] == "same_key_opaque_body_outline"
    assert result.report["strategy"]["extras"]["parameters"]["effective_trimap_mode"] == "same_key_opaque_body_outline"
    assert result.debug["pymatting_known_b"]["trimap"]["method"] == "same_key_opaque_body_outline"
    assert result.debug["proxy_subject_mask"].shape == image.shape[:2]
    assert np.all(result.rgba[..., :3][result.debug["proxy_subject_mask"]] == image[result.debug["proxy_subject_mask"]])
    floor_info = result.debug["pymatting_known_b"]["same_key_inner_opaque_floor"]
    floor_mask = result.debug["same_key_inner_opaque_floor_mask"]
    assert floor_info["enabled"] is True
    assert floor_info["applied_before_shadow_patch"] is True
    assert floor_info["alpha_lift_pixels"] >= 0
    assert floor_mask.shape == image.shape[:2]
    assert np.all(result.alpha[floor_mask] == 1.0)
    assert np.all(result.rgba[..., 3][floor_mask] == 255)
    assert np.all(result.rgba[..., :3][floor_mask] == image[floor_mask])
    assert result.debug["pymatting_subject_alpha_raw"].shape == image.shape[:2]


def test_pymatting_known_b_hard_shadow_evidence_release_prevents_green_foreground_solve():
    path = PROJECT_ROOT / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_hard_heavy_shadow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = _matte_known_b_after_background_repair(
        path,
        bg_color=(0, 200, 0),
        shadow_mode="on",
        pymatting_fg_threshold=24.0,
    )

    other = np.maximum(image[..., 0], image[..., 2]).astype(np.float32)
    green = image[..., 1].astype(np.float32)
    source_shadow = (other < 10.0) & (green < 190.0) & (green > 60.0)

    # B003's failure mode was exporting the hard shadow as dark green foreground.
    # The current path may leave some raw PyMatting foreground green, but the
    # final ShadowPatch layer must own the source-shadow pixels.
    raw_fg_median = np.median(result.debug["pymatting_subject_foreground"][source_shadow], axis=0)
    assert int(source_shadow.sum()) > 1500
    assert float(raw_fg_median[1]) <= 40.0
    result_rgb = result.rgba[..., :3]
    result_alpha = result.rgba[..., 3]
    green_dominant = (
        (result_rgb[..., 1] > result_rgb[..., 0] + 8)
        & (result_rgb[..., 1] > result_rgb[..., 2] + 8)
        & (result_alpha > 0)
    )
    assert int((source_shadow & green_dominant).sum()) == 0
    assert result.debug["shadow"]["shadow_pixels"] > 2000
    assert float(np.median(result.debug["shadow_alpha"][source_shadow])) > 0.30


def test_pymatting_known_b_semantic_shadow_layer_overrides_screen_colored_foreground():
    bg = (0, 40, 250)
    image = np.full((64, 64, 3), bg, dtype=np.uint8)
    shadow = np.zeros((64, 64), dtype=bool)
    shadow[38:50, 12:52] = True
    image[shadow] = (0, 30, 190)
    subject_alpha = np.zeros((64, 64), dtype=np.float32)
    subject_alpha[16:38, 12:52] = 1.0
    subject_alpha[shadow] = 1.0
    foreground = image.copy()

    auto_alpha, auto_rgb, auto_shadow, _, auto_info = _pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=bg,
        repair_domain=shadow,
        force_shadow_layer=False,
    )
    forced_alpha, forced_rgb, forced_shadow, _, forced_info = _pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=bg,
        repair_domain=shadow,
        force_shadow_layer=True,
    )

    assert auto_info["applied"] is False
    assert float(auto_shadow[shadow].mean()) == 0.0
    assert float(auto_alpha[shadow].mean()) == pytest.approx(1.0)
    assert np.asarray(auto_rgb[shadow]).mean(axis=0)[2] > 150.0
    assert forced_info["applied"] is True
    assert forced_info["force_shadow_layer"] is True
    assert forced_info["objective_shadow"]["semantic_forced_shadow_candidate_pixels"] == int(shadow.sum())
    assert forced_info["shadow_pixels"] == int(shadow.sum())
    assert float(forced_shadow[shadow].mean()) > 0.10
    assert float(forced_alpha[shadow].mean()) < 0.50
    assert np.asarray(forced_rgb[shadow]).mean(axis=0).max() < 1.0


def test_hard_shadow_subject_release_stays_local_to_shadow_facing_edge():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((110, 106, 3), bg, dtype=np.uint8)
    sure_fg = np.zeros(image.shape[:2], dtype=bool)
    sure_fg[8:101, 7:100] = True
    image[sure_fg] = (4, 120, 217)

    internal_gap = np.zeros(image.shape[:2], dtype=bool)
    internal_gap[54:85, 40:65] = True
    sure_fg[internal_gap] = False
    image[internal_gap] = (0, 180, 0)

    shadow_unknown = np.zeros(image.shape[:2], dtype=bool)
    shadow_unknown[87:100, 75:96] = True
    sure_fg[shadow_unknown] = False
    image[shadow_unknown] = (0, 100, 0)

    release, info = _known_background_hard_shadow_subject_evidence_release(
        image,
        bg,
        sure_fg=sure_fg,
        shadow_unknown=shadow_unknown,
    )

    assert info["released_pixels"] > 0
    assert int(release[:35].sum()) == 0
    assert int(release[:, :35].sum()) == 0
    assert int((release & shadow_unknown).sum()) == 0
    assert info["components"][0]["shadow_neighborhood_px"] < 15.0


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
