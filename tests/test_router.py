"""Tests for the front-end strategy router."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ermbg.router import assess_source_alpha, build_route_candidates, classify_route, classify_strategy

pytestmark = pytest.mark.core


def _solid_with_subject(bg_color, fg_color=(220, 30, 30), h=128, w=128):
    img = np.broadcast_to(np.array(bg_color, dtype=np.uint8), (h, w, 3)).copy()
    img[40:90, 40:90] = fg_color
    return img


def _hard_glossy_wide_button_gradient_only() -> np.ndarray:
    img = np.full((98, 261, 3), (2, 179, 9), dtype=np.uint8)
    cv2.rectangle(img, (32, 23), (244, 82), (1, 150, 3), -1, cv2.LINE_AA)
    for y in range(18, 75):
        t = (y - 18) / (75 - 18 - 1)
        color = np.array([255, 246, 208]) * (1.0 - t) + np.array([232, 190, 98]) * t
        img[y, 28:242] = np.clip(color, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (28, 18), (242, 75), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(img, (31, 21), (239, 72), (150, 90, 0), 2, cv2.LINE_AA)

    center = (47, 49)
    cv2.circle(img, center, 39, (255, 255, 255), -1, cv2.LINE_AA)
    for radius, color in [
        (35, (255, 146, 0)),
        (29, (255, 213, 19)),
        (23, (255, 156, 0)),
        (18, (255, 195, 16)),
    ]:
        cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, center, 31, (150, 80, 0), 2, cv2.LINE_AA)
    cv2.circle(img, center, 20, (180, 90, 0), 2, cv2.LINE_AA)

    right = (224, 49)
    cv2.circle(img, right, 22, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, right, 18, (95, 206, 26), -1, cv2.LINE_AA)
    cv2.circle(img, right, 17, (20, 88, 0), 2, cv2.LINE_AA)
    cv2.line(img, (224, 38), (224, 60), (255, 255, 255), 6, cv2.LINE_AA)
    cv2.line(img, (213, 49), (235, 49), (255, 255, 255), 6, cv2.LINE_AA)
    cv2.line(img, (224, 38), (224, 60), (0, 80, 0), 2, cv2.LINE_AA)
    cv2.line(img, (213, 49), (235, 49), (0, 80, 0), 2, cv2.LINE_AA)
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


def test_route_clean_rgba_passthrough():
    rgb, a = _make_clean_rgba()
    decision = classify_route(rgb, source_alpha=a)
    assert decision.route == "rgba_passthrough"
    assert decision.backend == "rgba_passthrough"
    assert decision.asset_kind == "rgba"


def test_route_hard_button_uses_pymatting_known_b():
    path = (
        Path(__file__).resolve().parents[1]
        / "samples"
        / "corridorkey_semantic"
        / "button"
        / "button_green_yellow_a_outlined_no_shadow"
        / "green.png"
    )
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    decision = classify_route(img)
    assert decision.route == "pymatting_known_b"
    assert decision.backend == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    assert decision.params["pymatting_trimap_mode"] == "standard"
    assert decision.params["pymatting_bg_color"] == (0, 200, 0)


def test_route_hard_button_uses_known_corridor_screen_color_when_subject_dominates_frame():
    img = np.full((148, 307, 3), (5, 132, 250), dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (302, 124), (253, 130, 4), -1, cv2.LINE_AA)
    cv2.rectangle(img, (4, 4), (302, 124), (255, 220, 80), 3, cv2.LINE_AA)
    cv2.rectangle(img, (8, 126), (299, 143), (140, 70, 5), -1, cv2.LINE_AA)

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    assert decision.params["pymatting_bg_color"] == (5, 132, 250)
    assert decision.params["pymatting_adapt_bg_threshold"] is False
    assert decision.params["pymatting_adapt_fg_threshold"] is True
    assert decision.params["pymatting_adapt_boundary_band"] is True
    assert decision.analysis["stable_background"]["source"] == "sure_bg_mode"
    assert decision.analysis["stable_background"]["seed"]["source"] == "route_screen_analysis"
    assert decision.analysis["stable_background"]["bg_threshold_source"] == "external_seed_cap"


def test_route_translucent_button_uses_corridorkey():
    path = (
        Path(__file__).resolve().parents[1]
        / "samples"
        / "corridorkey_semantic"
        / "button"
        / "button_green_yellow_c_translucent_soft_lite_shadow"
        / "green.png"
    )
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    decision = classify_route(img)
    assert decision.route == "corridorkey"
    assert decision.backend == "corridorkey"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "corridorkey-transparent-button"


def test_route_translucent_button_uses_corridorkey_semialpha_evidence():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    for rel in [
        "button/button_real_glass_blue_bg_green/blue.png",
    ]:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        decision = classify_route(img)
        assert decision.route == "corridorkey", rel
        assert decision.asset_kind == "button", rel
        assert decision.params["execution_profile"] == "corridorkey-transparent-button", rel
        assert "corridorkey_hard_ui_hint_mode" not in decision.params
        button_info = decision.analysis["button_corridorkey_translucency"]
        assert button_info["accepted"] is True, rel
        assert (
            button_info["profile_gate"]
            or button_info["semi_alpha_gate"]
            or button_info["combined_glass_gate"]
            or button_info["interior_material_gate"]
        ), rel


def test_route_gradient_only_glossy_button_stays_known_b():
    decision = classify_route(_hard_glossy_wide_button_gradient_only())

    assert decision.route == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    assert decision.analysis["complex_button_boundary"]["gradient_gate"] is True
    button_info = decision.analysis["button_corridorkey_translucency"]
    assert button_info["accepted"] is False
    assert button_info["gradient_gate_ignored"] is True


def test_route_gradient_only_glass_like_button_stays_known_b():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    img = np.asarray(
        Image.open(root / "button/button_real_glass_blue_bg_yellow/blue.png").convert("RGB"),
        dtype=np.uint8,
    )

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.analysis["complex_button_boundary"]["gradient_gate"] is True
    assert decision.analysis["button_corridorkey_translucency"]["gradient_gate_ignored"] is True


def test_route_opaque_hard_ui_profile_is_not_overridden_by_complex_boundary_gate():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    for rel in [
        "button/button_green_blue_c_translucent_soft_heavy_shadow/green.png",
        "button/button_blue_play_clipped_hard_shadow/blue.png",
    ]:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        decision = classify_route(img)
        assert decision.route == "pymatting_known_b", rel
        assert decision.params["execution_profile"] == "pymatting-hard-button", rel
        assert decision.analysis["corridorkey_analysis"]["parameter_profile"].startswith("opaque_hard_ui"), rel


def test_route_low_transition_translucent_named_button_uses_known_b_without_geometry_gate():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    img = np.asarray(
        Image.open(root / "button/button_blue_green_c_translucent_no_shadow/blue.png").convert("RGB"),
        dtype=np.uint8,
    )

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    ck = decision.analysis["corridorkey_analysis"]
    assert ck["key_transition_fraction"] < 0.03
    assert decision.analysis["complex_button_boundary"]["semi_alpha_fraction"] < 0.04


def test_route_hard_shadow_buttons_do_not_use_corridorkey_semialpha_shadow_gate():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    for rel in [
        "button/button_blue_green_a_outlined_soft_heavy_shadow/blue.png",
        "button/button_blue_green_b_unoutlined_soft_heavy_shadow/blue.png",
        "button/button_green_yellow_d_white_outline_soft_lite_shadow/green.png",
        "button/button_green_yellow_d_white_outline_soft_heavy_shadow/green.png",
    ]:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        decision = classify_route(img)
        assert decision.route == "pymatting_known_b", rel
        assert decision.backend == "pymatting_known_b", rel
        complex_info = decision.analysis["complex_button_boundary"]
        assert complex_info["semi_alpha_gate"] is False, rel


def test_route_square_known_b_hole_buttons_stay_pymatting_not_character():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    for rel in [
        "button/button_hole_yellow_ring_green/green.png",
        "button/button_hole_ornate_plate_blue/blue.png",
    ]:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        decision = classify_route(img)
        assert decision.route == "pymatting_known_b", rel
        assert decision.backend == "pymatting_known_b", rel
        assert decision.asset_kind == "button", rel
        assert decision.params["pymatting_trimap_mode"] == "standard", rel
        assert decision.analysis["character_like_foreground"]["accepted"] is False, rel


def test_route_hard_icon_uses_known_b_and_soft_character_uses_character_corridorkey_profile():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    icon = np.asarray(
        Image.open(root / "icon/icon_icon_a01_hard_boundary_strong_outline/green.png").convert("RGB"),
        dtype=np.uint8,
    )
    character = np.asarray(
        Image.open(root / "character/character_char_a06_pale_hair_translucent_sleeves_white_glow_blue/blue.png").convert("RGB"),
        dtype=np.uint8,
    )
    icon_decision = classify_route(icon)
    character_decision = classify_route(character)
    assert icon_decision.route == "pymatting_known_b"
    assert icon_decision.asset_kind == "button"
    assert icon_decision.params["execution_profile"] == "pymatting-hard-button"
    assert character_decision.route == "corridorkey"
    assert character_decision.asset_kind == "character"
    assert character_decision.params["execution_profile"] == "corridorkey-character"
    assert character_decision.analysis["character_like_foreground"]["accepted"] is True
    assert character_decision.analysis["complex_button_boundary"]["semi_alpha_gate"] is True


def test_route_candidates_c001_c005_expose_known_b_and_corridorkey_models():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "character"
    rels = [
        "character_char_a01_hair_hard_edge_glass_pendant_green/green.png",
        "character_char_a02_hair_armor_glass_visor_blue/blue.png",
        "character_char_a03_hair_fur_hard_blade_translucent_scarf_green/green.png",
        "character_char_a04_hair_transparent_wings_soft_glow_blue/blue.png",
        "character_char_a05_hard_armor_hair_glass_shield_green/green.png",
    ]
    for rel in rels:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        candidates = build_route_candidates(img)
        by_route = {candidate.decision.route: candidate for candidate in candidates}
        default = next(candidate for candidate in candidates if candidate.default)
        assert {"pymatting_known_b", "corridorkey"} <= set(by_route), rel
        assert default.decision.route == "corridorkey", rel
        assert by_route["corridorkey"].decision.asset_kind == "character", rel
        assert by_route["corridorkey"].decision.params["execution_profile"] == "corridorkey-character", rel
        evidence = by_route["corridorkey"].evidence["fine_detail_composite_evidence"]
        assert evidence["accepted"] is True, rel
        assert evidence["fine_boundary_gate"] is True, rel
        assert evidence["fine_edge_components"] >= 240, rel
        assert evidence["boundary_edge_fraction"] >= 0.23, rel
        assert evidence["hard_color_bins"] >= 500, rel


def test_route_candidates_i012_i017_use_corridorkey_soft_effect_composite_default():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "icon"
    rels = [
        "icon_icon_d02_soft_alpha_particle_mist/green.png",
        "icon_icon_d03_soft_alpha_particle_fire_orange/blue.png",
        "icon_icon_d04_soft_alpha_particle_poison_green/blue.png",
        "icon_icon_d05_soft_alpha_particle_arcane_purple/green.png",
        "icon_icon_d06_soft_alpha_particle_golden_stardust/green.png",
        "icon_icon_d07_soft_alpha_particle_ice_white_blue/green.png",
    ]
    for rel in rels:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        candidates = build_route_candidates(img)
        by_route = {candidate.decision.route: candidate for candidate in candidates}
        default = next(candidate for candidate in candidates if candidate.default)
        assert {"pymatting_known_b", "corridorkey"} <= set(by_route), rel
        assert default.decision.route == "corridorkey", rel
        assert by_route["corridorkey"].decision.params["execution_profile"] == "corridorkey-character", rel
        evidence = by_route["corridorkey"].evidence["fine_detail_composite_evidence"]
        assert evidence["accepted"] is True, rel
        assert evidence["soft_effect_gate"] or evidence["fine_boundary_gate"], rel


def test_route_glass_portal_icon_uses_corridorkey_soft_alpha_default():
    path = (
        Path(__file__).resolve().parents[1]
        / "samples/corridorkey_semantic/icon/icon_icon_d11_glass_portal_blue/blue.png"
    )
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    candidates = build_route_candidates(img)
    by_route = {candidate.decision.route: candidate for candidate in candidates}
    default = next(candidate for candidate in candidates if candidate.default)

    assert {"pymatting_known_b", "corridorkey"} <= set(by_route)
    assert default.decision.route == "corridorkey"
    assert by_route["corridorkey"].decision.params["execution_profile"] == "corridorkey-character"
    assert by_route["corridorkey"].decision.analysis["corridorkey_analysis"]["screen_mode"] == "blue"
    assert by_route["corridorkey"].decision.analysis["corridorkey_analysis"]["background_color"] == [0, 37, 252]
    evidence = by_route["corridorkey"].evidence["fine_detail_composite_evidence"]
    assert evidence["accepted"] is True
    assert evidence["fine_boundary_gate"] is True
    assert evidence["semi_alpha_fraction"] >= 0.20
    assert evidence["interior_semi_largest"] >= 7000


def test_route_candidates_i006_i007_use_corridorkey_soft_boundary_detail_default():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "icon"
    rels = [
        "icon_icon_b02_soft_boundary_feathered/blue.png",
        "icon_icon_b03_soft_boundary_fragmented_edge/blue.png",
    ]
    for rel in rels:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        candidates = build_route_candidates(img)
        by_route = {candidate.decision.route: candidate for candidate in candidates}
        default = next(candidate for candidate in candidates if candidate.default)
        assert {"pymatting_known_b", "corridorkey"} <= set(by_route), rel
        assert default.decision.route == "corridorkey", rel
        assert by_route["corridorkey"].decision.asset_kind == "button", rel
        assert by_route["corridorkey"].decision.params["execution_profile"] == "corridorkey-shaped-icon", rel
        evidence = by_route["corridorkey"].evidence["fine_detail_composite_evidence"]
        assert evidence["accepted"] is True, rel
        assert evidence["soft_boundary_detail_gate"] is True, rel
        assert evidence["semi_alpha_fraction"] >= 0.035, rel
        assert evidence["boundary_edge_fraction"] >= 0.24, rel
        assert evidence["interior_semi_largest"] <= 64, rel


def test_route_candidates_hard_or_opaque_icons_do_not_hit_soft_boundary_detail():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "icon"
    rels = [
        "icon_icon_b01_soft_boundary_antialias/green.png",
        "icon_icon_c01_translucent_glass_crystal/green.png",
    ]
    for rel in rels:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        candidates = build_route_candidates(img)
        assert [candidate.decision.route for candidate in candidates] == ["pymatting_known_b"], rel
        evidence = candidates[0].decision.analysis["fine_detail_composite_evidence"]
        assert evidence["soft_boundary_detail_gate"] is False, rel


def test_route_candidates_c006_c009_keep_corridorkey_high_confidence_default():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "character"
    rels = [
        "character_char_a06_pale_hair_translucent_sleeves_white_glow_blue/blue.png",
        "character_char_a07_fur_hair_hard_bow_translucent_cloak_green/green.png",
        "character_char_a08_spiky_hair_hard_mech_translucent_raincoat_blue/blue.png",
        "character_char_a09_hair_hard_costume_translucent_silk_ribbons_green/green.png",
    ]
    for rel in rels:
        img = np.asarray(Image.open(root / rel).convert("RGB"), dtype=np.uint8)
        candidates = build_route_candidates(img)
        default = next(candidate for candidate in candidates if candidate.default)
        assert default.decision.route == "corridorkey", rel
        assert default.decision.asset_kind == "character", rel
        assert default.decision.params["execution_profile"] == "corridorkey-character", rel
        assert default.decision.confidence >= 0.79, rel


def test_route_candidates_b056_keeps_known_b_b057_same_key_gets_counter_candidate():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "button"
    ornate = np.asarray(Image.open(root / "button_hole_ornate_plate_blue/blue.png").convert("RGB"), dtype=np.uint8)
    ornate_candidates = build_route_candidates(ornate)
    assert [candidate.decision.route for candidate in ornate_candidates] == ["pymatting_known_b"]
    assert ornate_candidates[0].default is True

    same_key = np.asarray(
        Image.open(root / "button_blue_play_clipped_hard_shadow/blue.png").convert("RGB"),
        dtype=np.uint8,
    )
    same_key_candidates = build_route_candidates(same_key)
    assert [candidate.id for candidate in same_key_candidates] == [
        "route_pymatting_known_b_same_key_opaque",
        "route_corridorkey_same_key_translucent",
    ]
    assert [candidate.decision.route for candidate in same_key_candidates] == [
        "pymatting_known_b",
        "corridorkey",
    ]
    assert same_key_candidates[0].default is True
    assert same_key_candidates[0].decision.params["pymatting_trimap_mode"] == "same_key_opaque_body_outline"
    assert same_key_candidates[1].decision.params["same_key_button_interpretation"] == "semi_transparent_corridorkey"


def test_route_complex_foreground_is_independent_of_canvas_aspect_ratio():
    root = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"
    image = np.asarray(
        Image.open(root / "character/character_char_a06_pale_hair_translucent_sleeves_white_glow_blue/blue.png").convert("RGB"),
        dtype=np.uint8,
    )
    bg = tuple(int(c) for c in image[0, 0])
    wide = np.full((image.shape[0], image.shape[1] + 600, 3), bg, dtype=np.uint8)
    wide[:, 300 : 300 + image.shape[1]] = image
    tall = np.full((image.shape[0] + 600, image.shape[1], 3), bg, dtype=np.uint8)
    tall[300 : 300 + image.shape[0], :] = image

    for variant in (image, wide, tall):
        decision = classify_route(variant)
        assert decision.route == "corridorkey"
        assert decision.asset_kind == "character"
        assert decision.params["execution_profile"] == "corridorkey-character"
        char_info = decision.analysis["character_like_foreground"]
        assert char_info["accepted"] is True
        assert "foreground_aspect_ratio" not in char_info
        assert "bbox_width_fraction" not in char_info


def test_route_square_canvas_button_uses_known_b_without_geometry_gate():
    img = np.full((1024, 1024, 3), (8, 205, 8), dtype=np.uint8)
    img[420:650, 110:914] = (0, 90, 245)
    img[418:422, 130:894] = (120, 190, 255)
    img[650:690, 150:874] = (0, 85, 40)

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    assert "button_" in decision.reasons[0]
    assert decision.analysis["character_like_foreground"]["accepted"] is False
    ck = decision.analysis["corridorkey_analysis"]
    assert ck["parameter_profile"] != "composite_character_corridor_only"
    assert decision.analysis["complex_button_boundary"]["reason"] == "below complex-boundary and semi-alpha gates"


def test_route_round_same_key_opaque_button_uses_two_simplified_candidates():
    img = np.full((128, 128, 3), (1, 95, 248), dtype=np.uint8)
    cv2.circle(img, (64, 64), 39, (112, 160, 248), -1, cv2.LINE_AA)
    cv2.circle(img, (64, 64), 42, (40, 88, 208), 2, cv2.LINE_AA)

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.backend == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    assert decision.params["pymatting_trimap_mode"] == "same_key_opaque_body_outline"
    candidates = build_route_candidates(img)
    assert [candidate.id for candidate in candidates] == [
        "route_pymatting_known_b_same_key_opaque",
        "route_corridorkey_same_key_translucent",
    ]
    assert candidates[0].default is True
    assert candidates[0].decision.params["parameter_profile"] == "known_b_same_key_opaque_outline"
    assert candidates[1].decision.params["parameter_profile"] == "corridorkey_same_key_translucent_button"
    ck = decision.analysis["corridorkey_analysis"]
    assert ck["foreground_aspect_ratio"] == pytest.approx(1.0)
    assert ck["parameter_profile"] == "opaque_hard_ui_same_key_plateau"
    assert ck["same_key_opaque_plateau_confidence"] >= 0.85


def test_route_same_key_opaque_button_uses_outline_trimap_when_outline_is_measured():
    img = np.full((120, 240, 3), (1, 95, 248), dtype=np.uint8)
    cv2.rectangle(img, (20, 12), (220, 98), (112, 160, 248), -1, cv2.LINE_AA)
    cv2.rectangle(img, (20, 12), (220, 98), (70, 118, 210), 2, cv2.LINE_AA)
    cv2.rectangle(img, (22, 99), (218, 108), (6, 74, 188), -1)
    cv2.line(img, (22, 98), (218, 98), (92, 126, 170), 1, cv2.LINE_AA)

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.params["execution_profile"] == "pymatting-hard-button"
    assert decision.params["pymatting_trimap_mode"] == "same_key_opaque_body_outline"
    assert decision.params["pymatting_unknown_grow_px"] == 0
    assert decision.analysis["corridorkey_analysis"]["parameter_profile"] == "opaque_hard_ui_same_key_plateau"
    assert decision.analysis["same_key_opaque_button_outline"]["accepted"] is True


def test_route_same_key_icon_uses_closed_outline_trimap_near_plateau_threshold():
    img = np.asarray(
        Image.open(
            Path(__file__).resolve().parents[1]
            / "samples/corridorkey_semantic/icon/icon_icon_a03_hard_boundary_weak_contrast/green.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    decision = classify_route(img)

    assert decision.route == "pymatting_known_b"
    assert decision.params["parameter_profile"] == "known_b_same_key_opaque_outline"
    assert decision.params["pymatting_trimap_mode"] == "same_key_opaque_body_outline"
    model = decision.analysis["same_key_button_model"]
    assert model["opaque_plateau"] is False
    assert model["outline_confirmed_plateau"] is True
    assert model["internal_clean_bg_pixels"] == 0
    assert model["translucent_counter_evidence"] is False


def test_route_unknown_unstable_background_uses_pymatting_fallback():
    rng = np.random.default_rng(123)
    img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    img[32:96, 32:96] = (20, 40, 180)
    decision = classify_route(img, fallback_background_color=(0, 200, 0))
    assert decision.route == "pymatting_fallback"
    assert decision.backend == "pymatting_fallback"
    assert decision.asset_kind == "unknown_fallback"
    assert decision.params["execution_profile"] == "pymatting-fallback"
    assert "unknown_or_unstable_background_uses_pymatting_fallback" in decision.reasons
    assert decision.params["pymatting_bg_color"] == (0, 200, 0)
    assert decision.params["pymatting_adapt_bg_threshold"] is False
    assert decision.params["pymatting_adapt_fg_threshold"] is False
    assert decision.params["pymatting_adapt_boundary_band"] is False


def test_route_non_screen_known_b_reports_known_b_parameter_profile():
    img = _solid_with_subject((255, 255, 255))

    decision = classify_route(img)
    payload = decision.to_dict()

    assert decision.route == "pymatting_known_b"
    assert payload["parameter_profile"] == "known_b_background_standard"
    assert payload["parameter_profile"] != decision.analysis["corridorkey_analysis"]["parameter_profile"]


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
