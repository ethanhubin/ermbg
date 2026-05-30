"""Offline tests for the remote SAM3 ComfyUI mask client."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.probe.comfyui_sam3_mask import ComfyUISAM3MaskClient

pytestmark = pytest.mark.core


def test_comfy_sam3_mask_workflow_renders_inputs():
    client = ComfyUISAM3MaskClient(url="http://example.invalid")

    workflow = client._render_workflow(
        input_image="input.png",
        checkpoint="sam3.1_multiplex_fp16.safetensors",
        threshold=0.42,
        refine_iterations=3,
        filename_prefix="case",
        image_width=320,
        image_height=180,
    )

    assert "_comment" not in workflow
    assert workflow["1"]["class_type"] == "CheckpointLoaderSimple"
    assert workflow["1"]["inputs"]["ckpt_name"] == "sam3.1_multiplex_fp16.safetensors"
    assert workflow["20"]["class_type"] == "SAM3_Detect"
    assert workflow["20"]["inputs"]["model"] == ["1", 0]
    assert workflow["20"]["inputs"]["image"] == ["10", 0]
    assert workflow["20"]["inputs"]["threshold"] == 0.42
    assert workflow["20"]["inputs"]["refine_iterations"] == 3
    assert workflow["20"]["inputs"]["individual_masks"] is False
    assert workflow["20"]["inputs"]["bboxes"] == {"x": 0, "y": 0, "width": 320, "height": 180}
    assert workflow["30"]["class_type"] == "MaskToImage"
    assert workflow["40"]["inputs"]["images"] == ["30", 0]


def test_comfy_sam3_mask_client_returns_float_mask(monkeypatch):
    client = ComfyUISAM3MaskClient(url="http://example.invalid")
    uploads: list[tuple[tuple[int, ...], str]] = []

    def fake_upload(image, name):
        uploads.append((image.shape, name))
        return name

    monkeypatch.setattr(client, "_upload", fake_upload)

    def fake_queue(workflow):
        assert workflow["20"]["inputs"]["bboxes"] == {"x": 0, "y": 0, "width": 8, "height": 6}
        return "prompt-1"

    monkeypatch.setattr(client, "_queue", fake_queue)
    monkeypatch.setattr(client, "_wait", lambda prompt_id: {"outputs": {}})

    def fake_download(history_entry, node_id, mode):
        del history_entry
        assert node_id == "40"
        assert mode == "L"
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 255
        return mask

    monkeypatch.setattr(client, "_download_node_image", fake_download)

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    result = client.mask(image, threshold=0.5, refine_iterations=2)

    assert uploads[0] == ((6, 8, 3), uploads[0][1])
    assert result.mask.shape == (6, 8)
    assert result.mask.dtype == np.float32
    assert result.mask[3, 4] == 1.0
    assert result.debug["backend"] == "comfy-sam3"
    assert result.debug["settings"]["threshold"] == 0.5
