"""Offline tests for the remote CorridorKey ComfyUI workflow client."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.probe.comfyui_corridorkey import (
    ComfyUICorridorKeyClient,
    apply_key_color_protection,
    build_corridorkey_hint,
    build_key_color_protection_floor,
)
from ermbg.corridorkey import corridorkey_analyze_asset

pytestmark = pytest.mark.core


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


def test_corridorkey_analysis_detects_green_and_keeps_standard_settings():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.screen_mode == "green"
    assert analysis.background_color == (0, 200, 0)
    assert analysis.background_confidence > 0.9
    assert analysis.recommended_settings.despill_strength == 1.0
    assert analysis.recommended_settings.auto_despeckle == "On"


def test_corridorkey_analysis_detects_blue_without_green_metadata():
    image = np.full((64, 64, 3), (0, 80, 255), dtype=np.uint8)
    image[18:46, 18:46] = (230, 40, 30)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.screen_mode == "blue"
    assert analysis.background_color == (0, 80, 255)
    assert analysis.border_coverage["blue"] > 0.9
    assert any("blue screen route" in note for note in analysis.notes)


def test_corridorkey_analysis_protects_subject_key_colored_material():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    image[18:46, 18:46] = (20, 160, 20)

    analysis = corridorkey_analyze_asset(image)

    assert analysis.subject_key_color_risk > 0.5
    assert analysis.recommended_settings.despill_strength < 1.0
    assert analysis.recommended_settings.refiner_strength < 1.0
    assert analysis.recommended_settings.protection_bg_max == 6.0


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
