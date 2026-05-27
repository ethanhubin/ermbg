"""Offline tests for the full remote ERMBG ComfyUI workflow client."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.probe.comfyui_ermbg_matte import ComfyUIErmbgMatteClient

pytestmark = pytest.mark.core


def test_comfy_ermbg_workflow_renders_automatte_inputs():
    client = ComfyUIErmbgMatteClient(url="http://example.invalid")

    workflow = client._render_workflow(
        input_image="uploaded.png",
        matting_model="ZhengPeng7/BiRefNet-matting",
        bg_color=(0, 200, 0),
        despill=None,
        use_keyer=False,
        shadow_mode="off",
        filename_prefix="case",
    )

    assert "_comment" not in workflow
    assert workflow["20"]["class_type"] == "ErmbgAutoMatte"
    assert workflow["20"]["inputs"]["image"] == ["10", 0]
    assert workflow["20"]["inputs"]["use_keyer"] == "force_off"
    assert workflow["20"]["inputs"]["shadow_mode"] == "off"
    assert workflow["50"]["inputs"]["images"] == ["40", 0]
    assert workflow["60"]["inputs"]["images"] == ["20", 3]


def test_comfy_ermbg_client_combines_foreground_and_alpha(monkeypatch):
    client = ComfyUIErmbgMatteClient(url="http://example.invalid")

    monkeypatch.setattr(client, "_upload", lambda image, name: "uploaded.png")
    monkeypatch.setattr(client, "_queue", lambda workflow: "prompt-1")
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {"outputs": {}})

    def fake_download(history_entry, node_id, mode):
        del history_entry
        if node_id == "30":
            assert mode == "RGB"
            return np.full((6, 8, 3), (10, 20, 30), dtype=np.uint8)
        if node_id == "50":
            assert mode == "L"
            alpha = np.zeros((6, 8), dtype=np.uint8)
            alpha[2:4, 3:5] = 255
            return alpha
        assert node_id == "60"
        assert mode == "RGB"
        return np.full((6, 8, 3), (40, 50, 60), dtype=np.uint8)

    monkeypatch.setattr(client, "_download_node_image", fake_download)

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    result = client.matte(image, shadow_mode="off")

    assert result.rgba.shape == (6, 8, 4)
    assert result.foreground_srgb[0, 0].tolist() == [10, 20, 30]
    assert result.rgba[3, 4].tolist() == [40, 50, 60, 255]
    assert result.alpha[3, 4] == 1.0
    assert result.debug["backend"] == "comfy-ermbg"


def test_comfy_ermbg_client_removes_shadow_mode_for_old_node(monkeypatch):
    client = ComfyUIErmbgMatteClient(url="http://example.invalid")
    queued: list[dict] = []

    monkeypatch.setattr(client, "_upload", lambda image, name: "uploaded.png")
    monkeypatch.setattr(client, "_automatte_supports_input", lambda name: False)

    def fake_queue(workflow):
        queued.append(workflow)
        return "prompt-1"

    monkeypatch.setattr(client, "_queue", fake_queue)
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {"outputs": {}})
    monkeypatch.setattr(
        client,
        "_download_node_image",
        lambda history_entry, node_id, mode: (
            np.zeros((4, 4), dtype=np.uint8)
            if node_id == "50"
            else np.zeros((4, 4, 3), dtype=np.uint8)
        ),
    )

    client.matte(np.zeros((4, 4, 3), dtype=np.uint8), shadow_mode="off")

    assert queued
    assert "shadow_mode" not in queued[0]["20"]["inputs"]


def test_comfy_ermbg_client_skips_object_info_by_default(monkeypatch):
    client = ComfyUIErmbgMatteClient(url="http://example.invalid")
    queued: list[dict] = []

    monkeypatch.setattr(client, "_upload", lambda image, name: "uploaded.png")
    monkeypatch.setattr(client, "_queue", lambda workflow: queued.append(workflow) or "prompt-1")
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {"outputs": {}})
    monkeypatch.setattr(
        client,
        "_download_node_image",
        lambda history_entry, node_id, mode: (
            np.zeros((4, 4), dtype=np.uint8)
            if node_id == "50"
            else np.zeros((4, 4, 3), dtype=np.uint8)
        ),
    )

    def fail_get(*args, **kwargs):
        raise AssertionError("default matte path should not call /object_info")

    monkeypatch.setattr("ermbg.probe.comfyui_ermbg_matte.requests.get", fail_get)

    client.matte(np.zeros((4, 4, 3), dtype=np.uint8), shadow_mode="on")

    assert queued[0]["20"]["inputs"]["shadow_mode"] == "on"
