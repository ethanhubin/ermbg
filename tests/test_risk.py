"""Tests for local risk-region extraction."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.risk import (
    coalesce_risk_regions,
    extract_alpha_keyer_disagreement_regions,
    extract_hard_edge_candidate_regions,
    extract_same_bg_enclosed_regions,
    extract_translucent_candidate_regions,
)
from ermbg.planner import RiskRegion

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
    rgba[hole, :3] = 255
    rgba[hole, 3] = 0
    return image, rgba, hole


def test_extract_same_bg_enclosed_region_returns_prompt_metadata():
    image, rgba, hole = _white_bg_red_ring_case()

    regions, info = extract_same_bg_enclosed_regions(image, rgba, (255, 255, 255))

    assert info["accepted_components"] == 1
    assert [r.kind for r in regions] == ["same_bg_enclosed_region"]
    assert regions[0].mask[hole].all()
    prompt = regions[0].to_prompt_dict()
    assert prompt["id"] == "same_bg_0"
    assert prompt["evidence_kind"] == "same_bg_low_alpha_enclosed"
    assert prompt["area"] == int(regions[0].mask.sum())
    assert prompt["bbox_xyxy"][0] < prompt["bbox_xyxy"][2]


def test_extract_alpha_keyer_disagreement_region_requires_anchor():
    matting = np.zeros((32, 32), dtype=np.float32)
    key = np.zeros((32, 32), dtype=np.float32)
    matting[10:14, 10:14] = 1.0
    key[14:18, 10:14] = 1.0

    regions, info = extract_alpha_keyer_disagreement_regions(matting, key)

    assert info["accepted_components"] == 1
    assert regions[0].kind == "alpha_keyer_disagreement"
    assert regions[0].mask[14:18, 10:14].all()


def test_extract_hard_edge_candidate_region_uses_contrast_and_anchor():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[15:17, 10:20] = 0
    matting = np.zeros((32, 32), dtype=np.float32)
    matting[17:20, 10:20] = 1.0
    key = np.zeros((32, 32), dtype=np.float32)
    key[15:17, 10:20] = 1.0

    regions, info = extract_hard_edge_candidate_regions(image, matting, key, (255, 255, 255))

    assert info["accepted_components"] == 1
    assert regions[0].kind == "hard_edge_candidate"
    assert regions[0].mask[15:17, 10:20].all()


def test_extract_translucent_candidate_prefers_mid_alpha_material_near_subject():
    image = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[20:44, 20:44, :3] = (120, 210, 255)
    rgba[20:44, 20:44, 3] = 115
    rgba[18:20, 18:46, 3] = 220
    rgba[44:46, 18:46, 3] = 220
    rgba[18:46, 18:20, 3] = 220
    rgba[18:46, 44:46, 3] = 220
    image[20:44, 20:44] = (80, 190, 210)

    regions, info = extract_translucent_candidate_regions(image, rgba, (0, 200, 0))

    assert info["accepted_components"] == 1
    assert regions[0].kind == "translucent_candidate"
    assert regions[0].mask[24:40, 24:40].mean() > 0.80
    assert regions[0].evidence["signal"] == "mid_alpha_chroma_shift_near_subject"


def test_extract_translucent_candidate_rejects_plain_background_hole():
    image, rgba, _ = _white_bg_red_ring_case()

    regions, info = extract_translucent_candidate_regions(image, rgba, (255, 255, 255))

    assert info["accepted_components"] == 0
    assert regions == []


def test_coalesce_risk_regions_merges_nearby_same_kind_fragments():
    mask_a = np.zeros((24, 24), dtype=bool)
    mask_b = np.zeros((24, 24), dtype=bool)
    mask_c = np.zeros((24, 24), dtype=bool)
    mask_a[8:10, 8:10] = True
    mask_b[8:10, 13:15] = True
    mask_c[18:20, 18:20] = True
    regions = [
        RiskRegion(id="hard_edge_0", kind="hard_edge_candidate", mask=mask_a),
        RiskRegion(id="hard_edge_1", kind="hard_edge_candidate", mask=mask_b),
        RiskRegion(id="hard_edge_2", kind="hard_edge_candidate", mask=mask_c),
    ]

    grouped = coalesce_risk_regions(regions, merge_distance_px=3)

    assert len(grouped) == 2
    assert grouped[0].evidence["source_region_ids"] == ["hard_edge_0", "hard_edge_1"]
    assert grouped[0].mask.sum() == mask_a.sum() + mask_b.sum()
    assert grouped[1].evidence["source_region_ids"] == ["hard_edge_2"]


def test_coalesce_risk_regions_keeps_unselected_kinds_passthrough():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    region = RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask)

    grouped = coalesce_risk_regions([region])

    assert grouped[0].id == "same_bg_0"
    assert grouped[0].evidence == {}
