"""Tests for local candidate generation."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.candidates import execute_candidate_plans, generate_matte_candidates
from ermbg.planner import CandidatePlan, PlanOperation, PlannerPromptBundle, RiskRegion

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
    assert candidates[1].debug["planner_bundle"]["regions"][0]["kind"] == "same_bg_enclosed_region"
    assert (
        candidates[1].debug["planner_bundle"]["regions"][0]["evidence_kind"]
        == "same_bg_low_alpha_enclosed"
    )
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
    assert candidates[0].debug["planner_bundle"]["regions"] == []


def test_generate_candidates_accepts_injected_planner_client():
    class FillOnlyPlanner:
        bundle: PlannerPromptBundle | None = None

        def plan(self, bundle: PlannerPromptBundle):
            self.bundle = bundle
            region_id = bundle.regions[0]["id"]
            return [
                CandidatePlan(
                    id="custom_fill",
                    label="Custom fill",
                    selected=True,
                    operations=[
                        PlanOperation(
                            tool="fill_same_color_region",
                            region_id=region_id,
                            parameters={"alpha_floor": 1.0},
                        )
                    ],
                )
            ]

    image, rgba, hole = _white_bg_red_ring_case()
    planner = FillOnlyPlanner()

    candidates = generate_matte_candidates(image, rgba, (255, 255, 255), planner_client=planner)

    assert [c.id for c in candidates] == ["custom_fill"]
    assert candidates[0].rgba[hole, 3].min() == 255
    assert planner.bundle is not None
    assert planner.bundle.image["background_color"] == [255, 255, 255]


def test_generate_candidates_supports_multiple_same_bg_regions():
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:96, 0:96]
    centers = [(32, 32), (64, 64)]
    ring = np.zeros((96, 96), dtype=bool)
    holes = []
    for cy, cx in centers:
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ring |= (r <= 16) & (r >= 7)
        holes.append(r < 7)
    image[ring] = (230, 0, 0)

    rgba = np.zeros((96, 96, 4), dtype=np.uint8)
    rgba[ring, :3] = image[ring]
    rgba[ring, 3] = 255
    for hole in holes:
        rgba[hole, :3] = 255
        rgba[hole, 3] = 0

    candidates = generate_matte_candidates(image, rgba, (255, 255, 255))

    assert [c.id for c in candidates] == [
        "transparent_holes",
        "fill_same_bg_0",
        "fill_same_bg_1",
        "fill_all_same_color_regions",
    ]
    assert candidates[0].rgba[holes[0] | holes[1], 3].max() == 0
    assert candidates[-1].rgba[holes[0] | holes[1], 3].min() == 255


def test_execute_repair_opaque_interior_uses_known_bg_repair_tool():
    h, w = 80, 80
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    image[20:60, 20:60] = (180, 230, 180)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[20:60, 20:60, :3] = image[20:60, 20:60]
    rgba[20:60, 20:60, 3] = 255
    rgba[34:46, 42:54, 3] = 26
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[34:46, 42:54] = True
    region = RiskRegion(id="alpha_keyer_0", kind="alpha_keyer_disagreement", mask=region_mask)
    plan = CandidatePlan(
        id="repair",
        label="Repair",
        operations=[
            PlanOperation(
                tool="repair_opaque_interior",
                region_id="alpha_keyer_0",
                parameters={"alpha_floor": 0.9},
            )
        ],
    )

    candidates = execute_candidate_plans([plan], [region], image, rgba, background_color=(255, 255, 255))

    assert candidates[0].rgba[region_mask, 3].min() >= 230
    assert candidates[0].debug["operation_results"][0]["tool"] == "repair_opaque_interior"
    assert candidates[0].debug["operation_results"][0]["repair_info"]["source"] == "known_bg_full_color_key"


def test_execute_snap_hard_edge_uses_hard_edge_repair_tool():
    h, w = 72, 72
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    image[24:52, 20:56] = (230, 0, 0)
    image[23, 20:56] = (20, 20, 20)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[24:52, 20:56, :3] = image[24:52, 20:56]
    rgba[24:52, 20:56, 3] = 255
    rgba[23, 20:56, :3] = image[23, 20:56]
    rgba[23, 20:56, 3] = 64
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[23, 20:56] = True
    region = RiskRegion(id="hard_edge_0", kind="hard_edge_candidate", mask=region_mask)
    plan = CandidatePlan(
        id="snap",
        label="Snap",
        operations=[
            PlanOperation(
                tool="snap_hard_edge",
                region_id="hard_edge_0",
                parameters={"alpha_floor": 0.95},
            )
        ],
    )

    candidates = execute_candidate_plans([plan], [region], image, rgba, background_color=(255, 255, 255))

    assert candidates[0].rgba[23, 22:54, 3].min() >= 242
    assert candidates[0].debug["operation_results"][0]["tool"] == "snap_hard_edge"
    assert candidates[0].debug["operation_results"][0]["repair_info"]["accepted_components"] == 1
