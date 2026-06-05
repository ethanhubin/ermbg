"""Tests for rectangle slicing on solid-background object sheets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ermbg.slicer import (
    SliceBox,
    analyze_checkerboard_background,
    classify_ui_slice,
    crop_slice,
    merge_overlapping_slice_boxes,
    normalize_checkerboard_background_to_light_square,
    save_slices,
    slice_image,
)


def test_slice_image_finds_separated_subject_rectangles(tmp_path):
    image = np.full((80, 120, 3), [0, 200, 0], dtype=np.uint8)
    image[10:31, 12:34] = [220, 20, 20]
    image[44:72, 70:104] = [30, 40, 230]
    image[5, 60] = [255, 255, 255]

    result = slice_image(image, min_area=80, padding=3)

    assert result.background_color == (0, 200, 0)
    assert result.padding == 3
    assert [box.bbox for box in result.boxes] == [(12, 10, 22, 21), (70, 44, 34, 28)]

    out_dir = tmp_path / "slices"
    paths = save_slices(image, result, out_dir, stem="sheet")

    assert [path.name for path in paths] == ["sheet_001_rgb.png", "sheet_002_rgb.png"]
    assert Image.open(paths[0]).size == (28, 27)
    report = json.loads((out_dir / "sheet.slices.json").read_text())
    assert report["count"] == 2
    assert (out_dir / "sheet_mask.png").exists()


def test_checkerboard_background_detects_two_value_periodic_border_and_uses_light_square():
    h, w = 160, 224
    tile = 14
    yy, xx = np.indices((h, w))
    parity = ((xx // tile + yy // tile) & 1).astype(bool)
    light = np.array([254, 254, 254], dtype=np.uint8)
    dark = np.array([243, 243, 243], dtype=np.uint8)
    image = np.where(parity[..., None], light, dark).astype(np.uint8)
    cv2.rectangle(image, (48, 46), (176, 106), (120, 60, 210), -1, cv2.LINE_AA)

    info = analyze_checkerboard_background(image)
    normalized, normalization = normalize_checkerboard_background_to_light_square(image)

    assert info["accepted"] is True
    assert info["tile_px"] == pytest.approx(tile, abs=2.0)
    assert info["two_value_tendency"] > 1.0
    assert tuple(info["background_color"]) == tuple(int(c) for c in light)
    assert normalization["applied"] is True
    assert normalization["changed_pixels"] > int((~parity).sum() * 0.5)
    border = np.zeros((h, w), dtype=bool)
    border[:tile, :] = True
    border[-tile:, :] = True
    border[:, :tile] = True
    border[:, -tile:] = True
    assert np.all(normalized[border] == light.reshape(1, 3))


def test_checkerboard_background_detects_shaded_cells_with_weak_mean_split():
    h, w = 180, 240
    tile = 18
    yy, xx = np.indices((h, w))
    parity = ((xx // tile + yy // tile) & 1).astype(bool)
    local_x = xx % tile
    # Keep the actual checker centers two-valued while making each parity group
    # internally uneven. This mirrors generated fake-transparent sheets where
    # compression and soft shading nearly cancel the parity means.
    light_cell = np.where(local_x < int(tile * 0.55), 252, 228)
    dark_cell = np.where(local_x < int(tile * 0.55), 238, 257)
    gray = np.where(parity, light_cell, dark_cell).clip(0, 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    image[58:122, 70:170] = [120, 220, 40]

    info = analyze_checkerboard_background(image)

    assert info["accepted"] is True
    assert info["tile_px"] == pytest.approx(tile, abs=2.0)
    assert info["checker_contrast"] >= 10.0


def test_checkerboard_background_rejects_smooth_low_chroma_drift():
    h, w = 128, 192
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    gray = 236.0 + 12.0 * xx + 5.0 * yy
    image = np.dstack([gray, gray + 1.0, gray]).astype(np.uint8)

    info = analyze_checkerboard_background(image)

    assert info["accepted"] is False


def test_checkerboard_background_rejects_sparse_neutral_card_grid():
    sample = Path(__file__).resolve().parents[1] / "samples" / "corridorkey_semantic" / "sheets" / "full_samples_v1_sheet.jpg"
    image_bgr = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    assert image_bgr is not None
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    info = analyze_checkerboard_background(image)

    assert info["accepted"] is False
    assert info["reason"] == "insufficient bright neutral border samples"


def test_slice_image_merges_overlapping_subject_rectangles():
    image = np.full((80, 120, 3), [0, 200, 0], dtype=np.uint8)
    image[10:13, 10:50] = [230, 40, 40]
    image[47:50, 10:50] = [230, 40, 40]
    image[10:50, 10:13] = [230, 40, 40]
    image[10:50, 47:50] = [230, 40, 40]
    image[25:31, 25:31] = [250, 250, 255]
    image[56:64, 12:42] = [40, 80, 240]
    image[20:36, 75:95] = [240, 220, 30]

    result = slice_image(image, min_area=10, padding=2)

    assert [box.bbox for box in result.boxes] == [(10, 10, 40, 40), (75, 20, 20, 16), (12, 56, 30, 8)]


def test_slice_image_padding_does_not_merge_adjacent_rectangles():
    image = np.full((80, 140, 3), [0, 200, 0], dtype=np.uint8)
    image[20:50, 20:50] = [230, 40, 40]
    image[20:50, 60:90] = [40, 80, 240]
    image[20:50, 100:130] = [240, 220, 30]

    result = slice_image(image, min_area=50, padding=12)

    assert [box.bbox for box in result.boxes] == [(20, 20, 30, 30), (60, 20, 30, 30), (100, 20, 30, 30)]
    padded = [crop_slice(image, result.foreground_mask, box, padding=result.padding).shape[:2] for box in result.boxes]
    assert padded == [(54, 54), (54, 54), (54, 52)]


def test_merge_slice_boxes_joins_aligned_touching_shadow_strip():
    boxes = [
        SliceBox(id=1, bbox=(609, 707, 206, 85), area=16974),
        SliceBox(id=2, bbox=(613, 792, 195, 10), area=963),
        SliceBox(id=3, bbox=(609, 819, 207, 96), area=19497),
    ]

    merged = merge_overlapping_slice_boxes(boxes)

    assert [box.bbox for box in merged] == [(609, 707, 206, 95), (609, 819, 207, 96)]


def test_merge_slice_boxes_joins_detached_lower_glow_strip():
    boxes = [
        SliceBox(id=1, bbox=(11, 12, 239, 192), area=29866),
        SliceBox(id=2, bbox=(37, 249, 194, 7), area=584),
    ]
    exterior_background = np.zeros((268, 261), dtype=bool)

    merged = merge_overlapping_slice_boxes(boxes, exterior_background_mask=exterior_background)

    assert [box.bbox for box in merged] == [(11, 12, 239, 244)]


def test_merge_slice_boxes_keeps_regions_split_across_background_corridor():
    boxes = [
        SliceBox(id=1, bbox=(37, 63, 530, 863), area=455143),
        SliceBox(id=2, bbox=(36, 945, 332, 101), area=33309),
    ]
    exterior_background = np.zeros((1402, 1122), dtype=bool)
    exterior_background[926:945, 37:368] = True

    merged = merge_overlapping_slice_boxes(boxes, exterior_background_mask=exterior_background)

    assert [box.bbox for box in merged] == [(37, 63, 530, 863), (36, 945, 332, 101)]


def test_merge_slice_boxes_does_not_chain_through_union_rectangles():
    boxes = [
        SliceBox(id=1, bbox=(57, 338, 366, 180), area=12759),
        SliceBox(id=2, bbox=(385, 436, 292, 124), area=9539),
        SliceBox(id=3, bbox=(328, 559, 28, 36), area=735),
        SliceBox(id=4, bbox=(198, 562, 25, 32), area=591),
        SliceBox(id=5, bbox=(318, 580, 108, 244), area=8146),
        SliceBox(id=6, bbox=(49, 584, 217, 152), area=8403),
        SliceBox(id=7, bbox=(487, 629, 152, 230), area=7319),
        SliceBox(id=8, bbox=(96, 789, 199, 271), area=18997),
        SliceBox(id=9, bbox=(353, 907, 285, 147), area=9951),
    ]

    merged = merge_overlapping_slice_boxes(boxes)

    assert len(merged) == 6
    assert [box.bbox for box in merged[:3]] == [(57, 338, 620, 222), (318, 559, 108, 265), (49, 562, 217, 174)]


def test_transparent_crop_uses_detected_foreground_as_alpha():
    image = np.full((32, 48, 3), [255, 255, 255], dtype=np.uint8)
    image[8:20, 10:24] = [10, 120, 230]

    result = slice_image(image, min_area=20, padding=2)
    rgba = crop_slice(image, result.foreground_mask, result.boxes[0], padding=result.padding, transparent=True)

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
