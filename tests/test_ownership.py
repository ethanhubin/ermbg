"""Tests for local ownership scoring."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg import io
from ermbg.ownership import ownership_masks, rank_region_ownership, resolve_execution_masks
from ermbg.planner import RiskRegion

pytestmark = pytest.mark.core


class _StubSegmenter:
    def __init__(self, alpha: np.ndarray):
        self.alpha = alpha.astype(np.float32)

    def segment(self, image, object_prompt=None):
        del image, object_prompt
        return self.alpha


def _top_role(image: np.ndarray, rgba: np.ndarray, bg: tuple[int, int, int], region: RiskRegion) -> str:
    ranked = rank_region_ownership(image, rgba, bg, region)
    assert ranked
    return ranked[0].role


def test_scalar_darkening_region_ranks_as_shadow_like_layer():
    h, w = 64, 80
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[18:38, 24:52, :3] = (220, 40, 40)
    rgba[18:38, 24:52, 3] = 255

    mask = np.zeros((h, w), dtype=bool)
    mask[42:54, 22:58] = True
    dark = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0] * 0.62
    image[mask] = io.linear_to_srgb_u8(dark.reshape(1, 1, 3))[0, 0]
    rgba[mask, 3] = 80

    region = RiskRegion(id="shadow_candidate_0", kind="owned_shadow_candidate", mask=mask)

    assert _top_role(image, rgba, tuple(int(c) for c in bg), region) == "shadow_like_layer"


def test_chroma_shift_mid_alpha_region_ranks_as_subject_soft_layer():
    h, w = 64, 80
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[18:46, 22:58, 3] = 230
    rgba[24:40, 30:50, 3] = 110
    image[24:40, 30:50] = (100, 185, 230)
    mask = np.zeros((h, w), dtype=bool)
    mask[24:40, 30:50] = True

    region = RiskRegion(id="translucent_0", kind="translucent_candidate", mask=mask)

    assert _top_role(image, rgba, tuple(int(c) for c in bg), region) == "subject_soft_layer"


def test_same_background_enclosed_region_ranks_as_hole():
    h, w = 64, 64
    bg = np.array([255, 255, 255], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[16:48, 16:48, 3] = 255
    rgba[24:40, 24:40, 3] = 0
    mask = np.zeros((h, w), dtype=bool)
    mask[24:40, 24:40] = True

    region = RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask)

    assert _top_role(image, rgba, tuple(int(c) for c in bg), region) == "hole"


def test_white_soft_material_avoids_opaque_repair_top_role():
    h, w = 64, 80
    bg = np.array([255, 255, 255], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[18:46, 20:60, 3] = 235
    rgba[24:40, 28:52, 3] = 120
    image[24:40, 28:52] = (205, 225, 245)
    mask = np.zeros((h, w), dtype=bool)
    mask[24:40, 28:52] = True

    region = RiskRegion(id="translucent_0", kind="translucent_candidate", mask=mask)
    ranked = rank_region_ownership(image, rgba, tuple(int(c) for c in bg), region)

    assert ranked[0].role == "subject_soft_layer"
    assert ranked[0].role != "opaque_subject"


def test_ownership_masks_aggregates_private_region_masks():
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:8, 4:9] = True
    rows = [
        {
            "_mask": mask,
            "region": {"id": "translucent_0"},
            "selected": {"role": "subject_soft_layer", "confidence": 0.8},
        }
    ]

    masks = ownership_masks(rows, (12, 12))

    assert masks["subject_soft_layer"][mask].all()
    assert int(masks["subject_soft_layer"].sum()) == int(mask.sum())


def test_resolve_execution_masks_drops_tiny_soft_speckles_but_keeps_shadow():
    masks = {
        "subject_soft_layer": np.zeros((100, 100), dtype=bool),
        "shadow_like_layer": np.zeros((100, 100), dtype=bool),
    }
    masks["subject_soft_layer"][2:7, 2:7] = True
    masks["shadow_like_layer"][30:60, 30:60] = True

    resolved = resolve_execution_masks(masks, (100, 100))

    assert not resolved["subject_soft_layer"].any()
    assert int(resolved["shadow_like_layer"].sum()) == 900


def test_resolve_execution_masks_suppresses_shadow_fragments_inside_soft_layer():
    masks = {
        "subject_soft_layer": np.zeros((100, 100), dtype=bool),
        "shadow_like_layer": np.zeros((100, 100), dtype=bool),
    }
    masks["subject_soft_layer"][10:90, 10:90] = True
    masks["shadow_like_layer"][20:30, 20:30] = True

    resolved = resolve_execution_masks(masks, (100, 100))

    assert int(resolved["subject_soft_layer"].sum()) == 6400
    assert not resolved["shadow_like_layer"].any()

