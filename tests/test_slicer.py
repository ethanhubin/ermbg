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


def test_slice_image_merges_attached_same_background_shadow():
    image = np.full((96, 120, 3), [0, 200, 0], dtype=np.uint8)
    image[18:44, 36:66] = [230, 40, 40]
    image[50:56, 30:76] = [0, 160, 0]
    image[56:64, 24:82] = [0, 188, 0]
    image[70:86, 92:108] = [40, 40, 230]

    result = slice_image(image, min_area=30, padding=2)

    # Failure mechanism: cast shadow is same-background darkening. The dark
    # contact band may be disconnected from the subject while the softer tail
    # sits below the ordinary foreground threshold, so slicing must attach it
    # to the anchored object rather than export it as an independent crop.
    assert len(result.boxes) == 2
    assert result.boxes[0].bbox == (22, 16, 62, 50)
    assert result.boxes[1].bbox == (90, 68, 20, 20)


def test_slice_image_uses_source_alpha_for_transparent_atlas():
    image = np.zeros((48, 72, 3), dtype=np.uint8)
    yy, xx = np.indices(image.shape[:2])
    checker = ((xx // 4 + yy // 4) % 2) == 0
    image[checker] = [230, 230, 230]
    image[~checker] = [245, 245, 245]
    image[8:22, 8:24] = [240, 30, 30]
    image[26:42, 44:64] = [20, 40, 220]
    alpha = np.zeros(image.shape[:2], dtype=np.float32)
    alpha[8:22, 8:24] = 1.0
    alpha[26:42, 44:64] = 1.0

    result = slice_image(image, min_area=50, padding=1, source_alpha=alpha)

    assert [box.bbox for box in result.boxes] == [(7, 7, 18, 16), (43, 25, 22, 18)]


def test_slice_image_ignores_baked_checkerboard_background():
    yy, xx = np.indices((64, 96))
    image = np.where(((xx // 8 + yy // 8) % 2)[..., None] == 0, 228, 244).astype(np.uint8)
    image = np.repeat(image, 3, axis=2)
    image[10:26, 8:28] = [240, 30, 30]
    image[36:54, 58:82] = [30, 80, 220]

    result = slice_image(image, min_area=80, padding=2)

    assert [box.bbox for box in result.boxes] == [(6, 8, 24, 20), (56, 34, 28, 22)]


def test_classify_ui_slice_uses_geometry_priors():
    sheet_shape = (160, 240)
    button_crop = np.full((40, 150, 3), [30, 180, 80], dtype=np.uint8)
    button_box = SliceBox(id=1, bbox=(0, 0, 150, 40), area=6000)

    icon_crop = np.full((80, 76, 3), [30, 80, 220], dtype=np.uint8)
    icon_box = SliceBox(id=2, bbox=(0, 0, 76, 80), area=6080)

    assert classify_ui_slice(button_crop, button_box, sheet_shape).kind == "button"
    assert classify_ui_slice(icon_crop, icon_box, sheet_shape).kind == "icon"
