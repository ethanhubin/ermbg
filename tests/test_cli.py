"""CLI smoke tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import ermbg.cli as cli

pytestmark = pytest.mark.core


class _HoleSegmenter:
    def segment(self, image_srgb, object_prompt=None):
        del image_srgb, object_prompt
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[16:48, 16:48] = 1.0
        alpha[28:36, 36:44] = 0.1
        return alpha


def test_matte_cli_accepts_subject_mask(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "build_segmenter", lambda **kwargs: _HoleSegmenter())

    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[16:48, 16:48] = 40
    input_path = tmp_path / "panel.png"
    Image.fromarray(image).save(input_path)

    support = np.zeros((64, 64), dtype=np.uint8)
    support[16:48, 16:48] = 255
    mask_path = tmp_path / "subject.png"
    Image.fromarray(support, mode="L").save(mask_path)

    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        cli.app,
        [
            "matte",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--subject-mask",
            str(mask_path),
            "--no-qa",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((out_dir / "panel.report.json").read_text())
    subject_repair = report["keyer"]["subject_repair"]
    known_repair = report["keyer"].get("known_bg_repair", {})
    assert subject_repair["used"] is True
    assert subject_repair["accepted_pixels"] + known_repair.get("accepted_pixels", 0) > 0


def test_slice_cli_exports_rectangles(tmp_path):
    image = np.full((48, 72, 3), [0, 200, 0], dtype=np.uint8)
    image[8:22, 8:24] = [240, 30, 30]
    image[25:42, 44:64] = [20, 40, 220]
    input_path = tmp_path / "sheet.png"
    Image.fromarray(image, mode="RGB").save(input_path)

    out_dir = tmp_path / "slices"
    result = CliRunner().invoke(
        cli.app,
        [
            "slice",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--min-area",
            "50",
            "--padding",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "sheet_001_rgb.png").exists()
    assert (out_dir / "sheet_002_rgb.png").exists()
    report = json.loads((out_dir / "sheet.slices.json").read_text())
    assert report["count"] == 2
