"""CLI smoke tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

import ermbg.cli as cli

pytestmark = pytest.mark.core


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
