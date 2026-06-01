"""Offline tests for the remote ColorToMask chroma-key workflow client."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.probe.comfyui_chroma_key import ComfyUIChromaKeyClient

pytestmark = pytest.mark.core


def test_comfy_chroma_key_workflow_renders_range_inputs():
    client = ComfyUIChromaKeyClient(url="http://example.invalid")

    workflow = client._render_workflow(
        input_image="input.png",
        key_color=(0, 200, 0),
        threshold=35,
        filename_prefix="case",
    )

    assert "_comment" not in workflow
    assert workflow["20"]["class_type"] == "ColorToMask"
    assert workflow["20"]["inputs"]["images"] == ["10", 0]
    assert workflow["20"]["inputs"]["invert"] is True
    assert workflow["20"]["inputs"]["red"] == 0
    assert workflow["20"]["inputs"]["green"] == 200
    assert workflow["20"]["inputs"]["blue"] == 0
    assert workflow["20"]["inputs"]["threshold"] == 35
    assert workflow["40"]["inputs"]["images"] == ["30", 0]


def test_comfy_chroma_key_client_combines_input_rgb_and_alpha(monkeypatch):
    client = ComfyUIChromaKeyClient(url="http://example.invalid")
    uploads: list[tuple[tuple[int, ...], str]] = []

    def fake_upload(image, name):
        uploads.append((image.shape, name))
        return name

    monkeypatch.setattr(client, "_upload", fake_upload)
    monkeypatch.setattr(client, "_queue", lambda workflow: "prompt-1")
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {"outputs": {}})

    def fake_download(history_entry, node_id):
        del history_entry
        assert node_id == "40"
        alpha = np.zeros((6, 8), dtype=np.uint8)
        alpha[2:4, 3:5] = 255
        return alpha

    monkeypatch.setattr(client, "_download_node_image", fake_download)

    image = np.full((6, 8, 3), (20, 40, 60), dtype=np.uint8)
    result = client.matte(image, key_color=(0, 200, 0), threshold=35)

    assert uploads[0] == ((6, 8, 3), uploads[0][1])
    assert result.rgba.shape == (6, 8, 4)
    assert result.rgba[3, 4].tolist() == [20, 40, 60, 255]
    assert result.alpha[3, 4] == 1.0
    assert result.debug["backend"] == "comfy-chromakey"
    assert result.debug["threshold"] == 35
