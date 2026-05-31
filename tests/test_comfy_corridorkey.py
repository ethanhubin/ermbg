"""Offline tests for the remote CorridorKey ComfyUI workflow client."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ermbg.probe.comfyui_corridorkey import (
    ComfyUICorridorKeyClient,
    apply_key_color_protection,
    build_corridorkey_hint,
    build_key_color_protection_floor,
)
from ermbg.corridorkey import corridorkey_analyze_asset
from ermbg.keyer import KeyerThresholds

pytestmark = pytest.mark.core

CORRIDORKEY_SEMANTIC_ROOT = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic"


def _selected_candidate(analysis):
    selected = [candidate for candidate in analysis.decision_candidates if candidate.selected]
    assert len(selected) == 1
    assert selected[0].confidence == max(candidate.confidence for candidate in analysis.decision_candidates)
    return selected[0]


def _game_sample_image(case_id: str, variant: str) -> np.ndarray:
    manifest = json.loads((CORRIDORKEY_SEMANTIC_ROOT / "manifest.json").read_text())
    case = next(item for item in manifest["cases"] if item["id"] == case_id)
    path = Path(__file__).resolve().parents[1] / case[variant]
    assert path.exists(), path
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def test_corridorkey_semantic_coverage_catalog_is_actionable():
    catalog_path = CORRIDORKEY_SEMANTIC_ROOT / "manifest.json"
    catalog = json.loads(catalog_path.read_text())

    assert catalog["version"] >= 2
    assert catalog["status"] == "phase1_complete_confirmed_full_test_set"
    assert catalog["phase_status"]["phase1_sample_coverage"] == "complete"
    assert catalog["phase_status"]["phase2_next_goal"]
    assert catalog["backgrounds"]["green"] == [0, 200, 0]
    assert catalog["backgrounds"]["blue"] == [0, 0, 200]
    assert catalog["case_count"] == 83
    assert catalog["category_counts"] == {"button": 54, "icon": 20, "character": 9}
    assert catalog["screen_counts"] == {"green": 57, "blue": 26}

    sample_ids = [case["sample_id"] for case in catalog["cases"]]
    assert len(sample_ids) == len(set(sample_ids))
    assert {"B001", "I001", "C001"} <= set(sample_ids)
    assert all(case["category"] in {"button", "icon", "character"} for case in catalog["cases"])
    blue_green_ids = {f"B{index:03d}" for index in range(16, 31)}
    for case in catalog["cases"]:
        screen = case["screen"]
        assert screen in {"green", "blue"}
        assert screen in case
        assert case["backgrounds"][screen] == catalog["backgrounds"][screen]
        assert case["image_size"] in ([256, 128], [256, 256], [1024, 1024])
        assert case["primary_ambiguity"]
        assert case["target_route"]
        input_path = Path(__file__).resolve().parents[1] / case[screen]
        case_path = input_path.with_name("case.json")
        assert input_path.exists(), input_path
        assert case_path.exists(), case_path
        if case["sample_id"] in blue_green_ids:
            # B016-B030 are the active blue-screen counterexamples for
            # green-subject UI. The retired yellow-on-blue block was useful for
            # diagnosis but no longer represents the failure class we need.
            assert screen == "blue"
            assert "button_blue_green_" in case["id"]
            assert "button_blue_yellow_" not in case["id"]
            assert "button_blue_green_" in case["blue"]
            assert "button_blue_yellow_" not in case["blue"]


def test_corridorkey_hint_is_soft_eroded_known_bg_support():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)

    hint = build_corridorkey_hint(image, (0, 200, 0))

    assert hint.shape == (64, 64)
    assert hint.dtype == np.float32
    assert hint[32, 32] > 0.95
    assert hint[0, 0] == 0.0
    assert hint[18, 32] < hint[32, 32]


def test_key_color_protection_is_color_based_not_region_based():
    image = np.full((8, 8, 3), (0, 200, 0), dtype=np.uint8)
    image[2:6, 2:4] = (250, 188, 24)
    image[2:6, 4:6] = (0, 120, 0)

    floor = build_key_color_protection_floor(image, (0, 200, 0))

    assert floor[3, 3] > 0.95
    assert floor[3, 5] < 0.05
    assert floor[0, 0] == 0.0


def test_key_color_protection_lifts_alpha_and_recovers_input_color():
    image = np.full((4, 4, 3), (0, 200, 0), dtype=np.uint8)
    image[1:3, 1:3] = (250, 188, 24)
    foreground = np.full((4, 4, 3), (80, 20, 10), dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)

    protected_fg, protected_alpha, floor, stats = apply_key_color_protection(
        image_srgb=image,
        foreground_srgb=foreground,
        alpha=alpha,
        background_color=(0, 200, 0),
    )

    assert floor[1, 1] > 0.95
    assert protected_alpha[1, 1] > 0.95
    assert protected_alpha[0, 0] == 0.0
    assert protected_fg[1, 1].tolist() == [250, 188, 24]
    assert stats["lifted_pixels_gt_01"] == 4


def test_key_color_protection_recovers_banded_partial_foreground():
    image = np.full((24, 32, 3), (0, 200, 0), dtype=np.uint8)
    image[6:18, 8:24] = (42, 128, 238)
    foreground = image.copy()
    foreground[6:18, 8:16] = (30, 98, 190)
    foreground[6:18, 16:24] = (74, 160, 242)
    alpha = np.zeros((24, 32), dtype=np.float32)
    alpha[6:18, 8:24] = 0.72

    protected_fg, protected_alpha, floor, _ = apply_key_color_protection(
        image_srgb=image,
        foreground_srgb=foreground,
        alpha=alpha,
        background_color=(0, 200, 0),
    )

    # Mechanism: CorridorKey can produce wavy foreground color inside an opaque
    # UI component while alpha is already partially present. A non-key color
    # floor that survives the shadow/edge gates is material evidence, so RGB
    # should recover from the source image rather than preserving that banding.
    assert floor[12, 12] > 0.95
    assert protected_alpha[12, 12] > 0.95
    assert protected_fg[12, 12].tolist() == [42, 128, 238]
    assert protected_fg[12, 20].tolist() == [42, 128, 238]


def test_key_color_protection_blocks_scalar_shadow_without_losing_highlight():
    image = np.full((20, 24, 3), (0, 0, 200), dtype=np.uint8)
    image[6:12, 7:17] = (250, 188, 24)
    image[8:10, 9:15] = (255, 255, 235)
    image[13:16, 8:18] = (20, 20, 80)
    foreground = np.full_like(image, (30, 20, 10))
    alpha = np.zeros((20, 24), dtype=np.float32)
    alpha[13:16, 8:18] = 0.35

    _, protected_alpha, applied_floor, stats = apply_key_color_protection(
        image_srgb=image,
        foreground_srgb=foreground,
        alpha=alpha,
        background_color=(0, 0, 200),
    )

    # Mechanism guard: subject-colored highlights are not scalar background
    # darkening, but a blue cast shadow is. The applied protection floor should
    # therefore fix the model's highlight hole without converting the shadow
    # into opaque subject ownership.
    assert protected_alpha[8:10, 9:15].mean() > 0.95
    assert applied_floor[13:16, 8:18].max() == 0.0
    assert np.allclose(protected_alpha[13:16, 8:18], 0.35)
    assert stats["floor_shadow_blocked_pixels"] >= 30


def test_key_color_protection_does_not_harden_exterior_antialias_edge():
    image = np.full((24, 32, 3), (0, 0, 200), dtype=np.uint8)
    image[7:17, 9:23] = (250, 188, 24)
    image[7:17, 8] = (120, 100, 126)
    image[11:13, 13:19] = (255, 255, 235)
    foreground = np.full_like(image, (250, 188, 24))
    alpha = np.zeros((24, 32), dtype=np.float32)
    alpha[7:17, 9:23] = 0.98
    alpha[7:17, 8] = 0.55
    alpha[11:13, 13:19] = 0.55

    _, protected_alpha, applied_floor, stats = apply_key_color_protection(
        image_srgb=image,
        foreground_srgb=foreground,
        alpha=alpha,
        background_color=(0, 0, 200),
    )

    # The same color-distance floor has two different ownership meanings:
    # near exterior background it is an antialiased boundary measurement, while
    # inside the subject it is a model hole. Protection should only fill the
    # interior hole.
    assert np.allclose(protected_alpha[7:17, 8], 0.55)
    assert applied_floor[7:17, 8].max() == 0.0
    assert protected_alpha[11:13, 13:19].mean() > 0.95
    assert stats["floor_edge_antialias_blocked_pixels"] >= 10


def test_key_color_protection_keeps_b023_blue_shadow_out_of_protection_floor():
    image = _game_sample_image("button_blue_green_b_unoutlined_hard_heavy_shadow", "blue")
    foreground = np.zeros_like(image)
    alpha = np.zeros(image.shape[:2], dtype=np.float32)
    shadow = (image[..., 0] < 80) & (image[..., 1] < 80) & (image[..., 2] > 40) & (image[..., 2] < 190)
    highlight = (image[..., 0] > 235) & (image[..., 1] > 235) & (image[..., 2] > 180)
    assert shadow.sum() > 200
    assert highlight.sum() > 50
    alpha[shadow] = 0.35

    _, protected_alpha, applied_floor, stats = apply_key_color_protection(
        image_srgb=image,
        foreground_srgb=foreground,
        alpha=alpha,
        background_color=(0, 0, 200),
        thresholds=KeyerThresholds(bg_max=12.0, fg_min=28.0),
    )

    assert float(protected_alpha[highlight].mean()) > 0.95
    assert float(applied_floor[shadow].max()) == 0.0
    assert np.allclose(protected_alpha[shadow], 0.35)
    assert stats["shadow_like_pixels"] > int(shadow.sum())
    assert stats["floor_shadow_blocked_pixels"] > 0


def test_key_color_protection_keeps_b023_exterior_antialias_soft():
    image = _game_sample_image("button_blue_green_b_unoutlined_hard_heavy_shadow", "blue")
    foreground = np.zeros_like(image)
    raw_alpha = np.zeros(image.shape[:2], dtype=np.float32)
    # Representative B023 outer-edge antialias pixels: source is subject color mixed
    # with blue screen/shadow, so CorridorKey's mid alpha is the useful signal.
    edge_points = [(53, 73), (211, 61), (205, 31), (52, 72), (50, 70)]
    for x, y in edge_points:
        raw_alpha[y, x] = 0.60

    _, protected_alpha, applied_floor, stats = apply_key_color_protection(
        image_srgb=image,
        foreground_srgb=foreground,
        alpha=raw_alpha,
        background_color=(0, 0, 200),
        thresholds=KeyerThresholds(bg_max=8.0, fg_min=18.0),
    )

    for x, y in edge_points:
        assert applied_floor[y, x] == 0.0
        assert protected_alpha[y, x] == pytest.approx(0.60)
    assert stats["floor_edge_antialias_blocked_pixels"] >= len(edge_points)


def test_corridorkey_analysis_detects_green_and_keeps_standard_settings():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.screen_mode == "green"
    assert analysis.background_color == (0, 200, 0)
    assert analysis.background_confidence > 0.9
    assert analysis.recommended_settings.despill_strength == 1.0
    assert analysis.recommended_settings.auto_despeckle == "On"
    assert analysis.parameter_profile == "edge_cleanup"
    assert analysis.recommended_settings.refiner_strength == 1.0
    assert analysis.recommended_settings.protection_bg_max == 12.0
    assert analysis.recommended_settings.protection_fg_min == 28.0
    selected = _selected_candidate(analysis)
    assert selected.profile == "edge_cleanup"
    assert any("stage1 selected edge_cleanup semantic path" in note for note in analysis.notes)
    assert any("stage2 edge-cleanup tuning" in note for note in analysis.notes)


def test_corridorkey_analysis_detects_blue_without_green_metadata():
    image = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.screen_mode == "blue"
    assert analysis.background_color == (0, 0, 255)
    assert analysis.border_coverage["blue"] > 0.9
    assert analysis.parameter_profile == "edge_cleanup"
    assert any("blue screen route" in note for note in analysis.notes)


def test_corridorkey_analysis_protects_subject_key_colored_material():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (20, 160, 20)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.subject_key_color_risk > 0.5
    assert analysis.hard_screen_residue_risk == 0.0
    assert analysis.parameter_profile == "key_color_material"
    assert analysis.recommended_settings.despill_strength == 0.45
    assert analysis.recommended_settings.refiner_strength == 0.70
    assert analysis.recommended_settings.protection_bg_max == 4.0
    assert analysis.recommended_settings.protection_fg_min == 10.0
    selected = _selected_candidate(analysis)
    assert selected.profile == "key_color_material"
    assert any("stage1 selected key_color_material semantic path" in note for note in analysis.notes)
    assert any("stage2 key-color material tuning" in note for note in analysis.notes)
    assert {candidate.profile for candidate in analysis.decision_candidates} == {
        "edge_cleanup",
        "screen_tinted_translucency",
        "key_color_material",
    }


def test_corridorkey_analysis_prefers_corridor_for_ambiguous_translucent_key_tint():
    rng = np.random.default_rng(123)
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[12:52, 12:52] = (230, 40, 30)
    ys = rng.integers(14, 50, 200)
    xs = rng.integers(14, 50, 200)
    image[ys, xs] = (40, 150, 40)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.key_color_solid_fraction == 0.0
    assert analysis.key_transition_fraction > 0.04
    assert analysis.parameter_profile == "screen_tinted_translucency"
    selected = _selected_candidate(analysis)
    assert selected.profile == "screen_tinted_translucency"
    assert selected.settings.color_protection is False
    assert {candidate.profile for candidate in analysis.decision_candidates} == {
        "edge_cleanup",
        "screen_tinted_translucency",
        "key_color_material",
    }


def test_corridorkey_analysis_boosts_refiner_for_hard_screen_family_residue():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)
    image[46:52, 24:40] = (20, 100, 20)

    analysis = corridorkey_analyze_asset(image)

    assert 0.04 <= analysis.subject_key_color_risk <= 0.25
    assert analysis.key_transition_fraction < 0.015
    assert analysis.hard_screen_residue_risk >= 0.04
    assert analysis.parameter_profile == "key_color_material"
    assert analysis.recommended_settings.refiner_strength == 1.5
    assert any("stage2 hard screen-family residue tuning" in note for note in analysis.notes)


def test_corridorkey_analysis_keeps_thin_key_residue_out_of_material_route():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)
    image[18:19, 18:46] = (20, 160, 20)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.parameter_profile == "edge_cleanup"
    assert analysis.recommended_settings.protection_bg_max == 12.0
    assert analysis.recommended_settings.protection_fg_min == 28.0
    selected = _selected_candidate(analysis)
    assert selected.profile == "edge_cleanup"
    assert selected.settings.color_protection is True


def test_corridorkey_analysis_interpolates_moderate_blue_family_subject_material():
    image = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)
    image[18:22, 18:46] = (20, 80, 220)

    analysis = corridorkey_analyze_asset(image)

    assert 0.08 < analysis.subject_key_color_risk < 0.45
    assert analysis.parameter_profile == "balanced"
    assert 6.0 < analysis.recommended_settings.protection_bg_max < 12.0
    assert 14.0 < analysis.recommended_settings.protection_fg_min < 28.0


def test_corridorkey_analysis_protects_dominant_blue_family_subject_material():
    image = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
    image[18:46, 18:46] = (20, 80, 220)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.subject_key_color_risk > 0.45
    assert analysis.parameter_profile == "key_color_material"
    assert analysis.recommended_settings.despill_strength == 0.45
    assert analysis.recommended_settings.refiner_strength == 0.70
    assert analysis.recommended_settings.protection_bg_max == 4.0
    assert analysis.recommended_settings.protection_fg_min == 10.0


@pytest.mark.parametrize(
    ("case_id", "variant", "expected_profile"),
    [
        ("button_green_yellow_a_outlined_no_shadow", "green", "opaque_hard_ui_no_shadow"),
        ("button_green_yellow_a_outlined_hard_lite_shadow", "green", "opaque_hard_ui_hard_shadow"),
        ("button_green_yellow_a_outlined_soft_lite_shadow", "green", "opaque_hard_ui_soft_shadow"),
        ("button_green_yellow_c_translucent_no_shadow", "green", "translucent_button"),
        ("button_green_yellow_c_translucent_hard_lite_shadow", "green", "translucent_button"),
        ("button_green_yellow_c_translucent_hard_heavy_shadow", "green", "translucent_button"),
        ("button_green_yellow_c_translucent_soft_lite_shadow", "green", "translucent_button"),
        ("button_green_yellow_c_translucent_soft_heavy_shadow", "green", "translucent_button"),
        ("icon_icon_a01_hard_boundary_strong_outline", "green", "edge_cleanup"),
        ("icon_icon_d10_soft_alpha_smooth_white_glow_blue", "blue", "key_color_material"),
        ("character_char_a04_hair_transparent_wings_soft_glow_blue", "blue", "edge_cleanup"),
    ],
)
def test_corridorkey_game_sample_semantic_path_recognition(case_id, variant, expected_profile):
    image = _game_sample_image(case_id, variant)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.parameter_profile == expected_profile


def test_corridorkey_opaque_hard_ui_uses_assertive_cleanup_parameters():
    image = _game_sample_image("button_green_yellow_a_outlined_no_shadow", "green")

    analysis = corridorkey_analyze_asset(image)

    assert analysis.parameter_profile == "opaque_hard_ui_no_shadow"
    assert analysis.recommended_settings.despill_strength == 1.0
    assert analysis.recommended_settings.refiner_strength == 1.15
    assert analysis.recommended_settings.auto_despeckle == "On"
    assert analysis.recommended_settings.protection_bg_max == 8.0
    assert analysis.recommended_settings.protection_fg_min == 18.0


def test_corridorkey_opaque_hard_ui_hard_shadow_uses_shadow_safe_protection():
    image = _game_sample_image("button_green_yellow_a_outlined_hard_lite_shadow", "green")

    analysis = corridorkey_analyze_asset(image)

    assert analysis.parameter_profile == "opaque_hard_ui_hard_shadow"
    assert analysis.recommended_settings.despill_strength == 1.0
    assert analysis.recommended_settings.refiner_strength == 1.15
    assert analysis.recommended_settings.protection_bg_max == 8.0
    assert analysis.recommended_settings.protection_fg_min == 18.0


def test_corridorkey_translucent_button_disables_color_protection():
    image = _game_sample_image("button_green_yellow_c_translucent_soft_lite_shadow", "green")

    analysis = corridorkey_analyze_asset(image)

    assert analysis.parameter_profile == "translucent_button"
    assert analysis.recommended_settings.color_protection is False
    assert analysis.recommended_settings.refiner_strength == 1.15
    assert analysis.recommended_settings.auto_despeckle == "Off"


@pytest.mark.parametrize(
    ("case_id", "variant", "expected_profile"),
    [
        pytest.param(
            "button_real_glass_green_bg_yellow",
            "green",
            "screen_tinted_translucency",
            marks=pytest.mark.xfail(strict=True, reason="real glass button needs an explicit translucent/glass route"),
        ),
        pytest.param(
            "icon_icon_d09_soft_alpha_smooth_white_glow_green",
            "green",
            "additive_effect_sprite",
            marks=pytest.mark.xfail(strict=True, reason="smooth glow still needs an additive/soft-alpha effect route"),
        ),
    ],
)
def test_corridorkey_game_sample_semantic_path_known_gaps(case_id, variant, expected_profile):
    image = _game_sample_image(case_id, variant)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.parameter_profile == expected_profile


def test_corridorkey_analysis_detail_safe_disables_despeckle():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)

    analysis = corridorkey_analyze_asset(image, preset="detail_safe")

    assert analysis.recommended_settings.auto_despeckle == "Off"
    assert analysis.recommended_settings.despeckle_size == 64


def test_comfy_corridorkey_workflow_renders_inputs():
    client = ComfyUICorridorKeyClient(url="http://example.invalid")

    workflow = client._render_workflow(
        input_image="input.png",
        mask_image="hint.png",
        gamma_space="sRGB",
        screen_color="blue",
        despill_strength=0.75,
        refiner_strength=1.2,
        auto_despeckle="Off",
        despeckle_size=64,
        filename_prefix="case",
    )

    assert "_comment" not in workflow
    assert workflow["12"]["class_type"] == "ImageToMask"
    assert workflow["12"]["inputs"]["image"] == ["11", 0]
    assert workflow["20"]["class_type"] == "CorridorKey"
    assert workflow["20"]["inputs"]["image"] == ["10", 0]
    assert workflow["20"]["inputs"]["mask"] == ["12", 0]
    assert workflow["20"]["inputs"]["screen_color"] == "blue"
    assert workflow["20"]["inputs"]["despill_strength"] == 0.75
    assert workflow["20"]["inputs"]["auto_despeckle"] == "Off"
    assert workflow["50"]["inputs"]["images"] == ["40", 0]


def test_comfy_corridorkey_client_combines_fg_and_alpha(monkeypatch):
    client = ComfyUICorridorKeyClient(url="http://example.invalid")
    uploads: list[tuple[tuple[int, ...], str]] = []

    def fake_upload(image, name):
        uploads.append((image.shape, name))
        return name

    monkeypatch.setattr(client, "_upload", fake_upload)
    monkeypatch.setattr(client, "_queue", lambda workflow: "prompt-1")
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {"outputs": {}})

    def fake_download(history_entry, node_id, mode):
        del history_entry
        if node_id == "30":
            assert mode == "RGB"
            return np.full((6, 8, 3), (10, 20, 30), dtype=np.uint8)
        assert node_id == "50"
        assert mode == "L"
        alpha = np.zeros((6, 8), dtype=np.uint8)
        alpha[2:4, 3:5] = 255
        return alpha

    monkeypatch.setattr(client, "_download_node_image", fake_download)

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    hint = np.ones((6, 8), dtype=np.float32)
    result = client.matte(image, hint_alpha=hint, apply_color_protection=False)

    assert uploads[0] == ((6, 8, 3), uploads[0][1])
    assert uploads[1] == ((6, 8), uploads[1][1])
    assert result.rgba.shape == (6, 8, 4)
    assert result.rgba[3, 4].tolist() == [10, 20, 30, 255]
    assert result.alpha[3, 4] == 1.0
    assert result.raw_alpha[3, 4] == 1.0
    assert result.color_protection_alpha.max() == 0.0
    assert result.debug["backend"] == "comfy-corridorkey"
