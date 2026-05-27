"""Tests for rectangle slicing on solid-background object sheets."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from ermbg.slicer import SliceBox, classify_ui_slice, crop_slice, save_slices, slice_image


def test_slice_image_finds_separated_subject_rectangles(tmp_path):
    image = np.full((80, 120, 3), [0, 200, 0], dtype=np.uint8)
    image[10:31, 12:34] = [220, 20, 20]
    image[44:72, 70:104] = [30, 40, 230]
    image[5, 60] = [255, 255, 255]

    result = slice_image(image, min_area=80, padding=3)

    assert result.background_color == (0, 200, 0)
    assert [box.bbox for box in result.boxes] == [(9, 7, 28, 27), (67, 41, 40, 34)]

    out_dir = tmp_path / "slices"
    paths = save_slices(image, result, out_dir, stem="sheet")

    assert [path.name for path in paths] == ["sheet_001_rgb.png", "sheet_002_rgb.png"]
    assert Image.open(paths[0]).size == (28, 27)
    report = json.loads((out_dir / "sheet.slices.json").read_text())
    assert report["count"] == 2
    assert (out_dir / "sheet_mask.png").exists()


def test_transparent_crop_uses_detected_foreground_as_alpha():
    image = np.full((32, 48, 3), [255, 255, 255], dtype=np.uint8)
    image[8:20, 10:24] = [10, 120, 230]

    result = slice_image(image, min_area=20, padding=2)
    rgba = crop_slice(image, result.foreground_mask, result.boxes[0], transparent=True)

    assert rgba.shape == (16, 18, 4)
    assert rgba[..., 3].max() == 255
    assert rgba[0, 0, 3] == 0
    assert rgba[4, 4, 3] == 255


def test_classify_ui_slice_uses_geometry_priors():
    sheet_shape = (160, 240)
    button_crop = np.full((40, 150, 3), [30, 180, 80], dtype=np.uint8)
    button_box = SliceBox(id=1, bbox=(0, 0, 150, 40), area=6000)

    icon_crop = np.full((80, 76, 3), [30, 80, 220], dtype=np.uint8)
    icon_box = SliceBox(id=2, bbox=(0, 0, 76, 80), area=6080)

    assert classify_ui_slice(button_crop, button_box, sheet_shape).kind == "button"
    assert classify_ui_slice(icon_crop, icon_box, sheet_shape).kind == "icon"
