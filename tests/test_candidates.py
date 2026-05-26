"""Tests for local candidate generation."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.candidates import generate_matte_candidates

pytestmark = pytest.mark.core


def _white_bg_red_ring_case(h=96, w=96):
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r <= 30) & (r >= 12)
    hole = r < 12
    image[ring] = (230, 0, 0)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[ring, :3] = image[ring]
    rgba[ring, 3] = 255
    # The base matte interprets the white center as transparent background.
    rgba[hole, :3] = 255
    rgba[hole, 3] = 0
    return image, rgba, hole


def test_generate_candidates_for_same_background_color_inner_hole():
    image, rgba, hole = _white_bg_red_ring_case()

    candidates = generate_matte_candidates(image, rgba, (255, 255, 255))

    assert [c.id for c in candidates] == ["transparent_hole", "same_color_marking"]
    assert candidates[0].selected is True
    assert candidates[0].rgba[hole, 3].max() == 0
    assert candidates[0].debug["plan"]["operations"][0]["tool"] == "preserve_hole"
    assert candidates[1].rgba[hole, 3].min() == 255
    assert candidates[1].debug["plan"]["operations"][0]["tool"] == "fill_same_color_region"
    np.testing.assert_array_equal(candidates[1].rgba[hole, :3], image[hole])


def test_generate_candidates_returns_single_auto_when_no_ambiguity():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[16:48, 16:48] = (230, 0, 0)
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[16:48, 16:48, :3] = image[16:48, 16:48]
    rgba[16:48, 16:48, 3] = 255

    candidates = generate_matte_candidates(image, rgba, (255, 255, 255))

    assert [c.id for c in candidates] == ["auto"]
    assert candidates[0].selected is True
