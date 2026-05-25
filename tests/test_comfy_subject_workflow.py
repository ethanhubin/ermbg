"""Offline tests for the ComfyUI subject-mask workflow renderer."""

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from ermbg.probe.comfyui_subject_mask import ComfyUISubjectMaskWorkflow, render_clipseg_ermbg_workflow

pytestmark = pytest.mark.core


def test_render_clipseg_ermbg_workflow_wires_subject_mask():
    workflow = render_clipseg_ermbg_workflow(
        input_image="sample_12.png",
        subject_prompt='the entire "framed" green panel',
        filename_prefix="sample12",
    )

    assert "_comment" not in workflow
    assert workflow["10"]["class_type"] == "LoadImage"
    assert workflow["20"]["class_type"] == "CLIPSeg Masking"
    assert workflow["30"]["class_type"] == "ErmbgAutoMatte"
    assert workflow["30"]["inputs"]["subject_mask"] == ["20", 0]
    assert workflow["60"]["inputs"]["images"] == ["50", 0]
    assert workflow["80"]["inputs"]["images"] == ["70", 0]

    # Must remain JSON-serializable after prompt escaping.
    rendered = json.dumps(workflow)
    assert "framed" in rendered


def test_subject_workflow_downloads_named_outputs(monkeypatch, tmp_path):
    runner = ComfyUISubjectMaskWorkflow(url="http://example.invalid")

    history = {
        "outputs": {
            "40": {"images": [{"filename": "fg.png", "subfolder": "", "type": "output"}]},
            "60": {"images": [{"filename": "alpha.png", "subfolder": "", "type": "output"}]},
            "80": {"images": [{"filename": "mask.png", "subfolder": "", "type": "output"}]},
        }
    }

    def fake_get(path, **kwargs):
        assert path == "/view"
        assert kwargs["params"]["type"] == "output"
        buf = BytesIO()
        Image.new("RGB", (8, 6), color=(255, 0, 0)).save(buf, format="PNG")
        return SimpleNamespace(content=buf.getvalue())

    monkeypatch.setattr(runner, "_get", fake_get)

    records = runner.download_images(history, tmp_path)

    assert [r["role"] for r in records] == ["foreground", "alpha", "subject_mask"]
    assert (tmp_path / "foreground.png").exists()
    assert (tmp_path / "alpha.png").exists()
    assert (tmp_path / "subject_mask.png").exists()
    assert records[0]["width"] == 8
    assert records[0]["height"] == 6
