"""Tests for planner schema and candidate-plan validation."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.candidates import execute_candidate_plans
from ermbg.planner import (
    CandidatePlan,
    PlanOperation,
    PlanValidationError,
    RiskRegion,
    build_planner_prompt_bundle,
    default_tool_catalog,
    plan_candidates_from_regions,
    validate_candidate_plans,
)

pytestmark = pytest.mark.core


def test_default_tool_catalog_exposes_finite_planner_tools():
    catalog = default_tool_catalog()

    assert set(catalog) == {
        "preserve_hole",
        "fill_same_color_region",
        "repair_opaque_interior",
        "snap_hard_edge",
        "preserve_soft_alpha",
        "mark_translucent",
    }
    assert catalog["fill_same_color_region"].parameter_ranges["alpha_floor"] == (0.0, 1.0)


def test_build_planner_prompt_bundle_is_json_safe_context():
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:8, 5:11] = True
    region = RiskRegion(
        id="same_bg_0",
        kind="same_bg_enclosed_region",
        mask=mask,
        evidence={"alpha_max": 0.2},
    )

    bundle = build_planner_prompt_bundle(
        image_shape=(12, 16, 3),
        regions=[region],
        background_color=(255, 255, 255),
        strategy={"name": "white_bg", "image_type": "graphic"},
        instructions=["Return CandidatePlan JSON only."],
    )
    payload = bundle.to_dict()

    assert payload["image"] == {
        "height": 12,
        "width": 16,
        "channels": 3,
        "background_color": [255, 255, 255],
        "strategy": {"name": "white_bg", "image_type": "graphic"},
    }
    assert payload["regions"][0]["id"] == "same_bg_0"
    assert payload["regions"][0]["kind"] == "same_bg_enclosed_region"
    assert payload["regions"][0]["evidence_kind"] == "same_bg_low_alpha_enclosed"
    assert payload["regions"][0]["bbox_xyxy"] == [5, 3, 11, 8]
    assert payload["regions"][0]["area"] == 30
    assert payload["tools"][0]["name"] == "preserve_hole"
    assert payload["tools"][0]["allowed_evidence_kinds"] == [
        "same_bg_low_alpha_enclosed",
        "intentional_hole",
    ]
    assert payload["tools"][1]["parameter_ranges"]["alpha_floor"] == [0.0, 1.0]
    assert payload["instructions"] == ["Return CandidatePlan JSON only."]


def test_validate_candidate_plan_rejects_wrong_region_kind():
    region = RiskRegion(
        id="r1",
        kind="soft_edge_band",
        mask=np.ones((8, 8), dtype=bool),
    )
    plan = CandidatePlan(
        id="bad",
        label="Bad",
        operations=[PlanOperation(tool="fill_same_color_region", region_id="r1")],
    )

    with pytest.raises(PlanValidationError, match="cannot run on region kind"):
        validate_candidate_plans([plan], [region])


def test_validate_candidate_plan_rejects_parameter_out_of_range():
    region = RiskRegion(
        id="r1",
        kind="same_bg_enclosed_region",
        mask=np.ones((8, 8), dtype=bool),
    )
    plan = CandidatePlan(
        id="bad_alpha",
        label="Bad alpha",
        operations=[
            PlanOperation(
                tool="fill_same_color_region",
                region_id="r1",
                parameters={"alpha_floor": 1.2},
            )
        ],
    )

    with pytest.raises(PlanValidationError, match="outside allowed range"):
        validate_candidate_plans([plan], [region])


def test_execute_candidate_plans_supports_variable_length_candidate_list():
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[4:12, 4:12] = (10, 20, 30)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    region = RiskRegion(id="r1", kind="same_bg_enclosed_region", mask=mask)
    plans = [
        CandidatePlan(
            id="keep_transparent",
            label="Keep transparent",
            selected=True,
            operations=[PlanOperation(tool="preserve_hole", region_id="r1")],
        ),
        CandidatePlan(
            id="fill_opaque",
            label="Fill opaque",
            operations=[
                PlanOperation(
                    tool="fill_same_color_region",
                    region_id="r1",
                    parameters={"alpha_floor": 1.0},
                )
            ],
        ),
        CandidatePlan(
            id="fill_half",
            label="Fill half",
            operations=[
                PlanOperation(
                    tool="fill_same_color_region",
                    region_id="r1",
                    parameters={"alpha_floor": 0.5},
                )
            ],
        ),
    ]

    candidates = execute_candidate_plans(plans, [region], image, rgba)

    assert [c.id for c in candidates] == ["keep_transparent", "fill_opaque", "fill_half"]
    assert candidates[0].rgba[mask, 3].max() == 0
    assert candidates[1].rgba[mask, 3].min() == 255
    assert candidates[2].rgba[mask, 3].min() == 128


def test_rule_planner_generates_variable_candidates_for_multiple_same_bg_regions():
    mask_a = np.zeros((16, 16), dtype=bool)
    mask_b = np.zeros((16, 16), dtype=bool)
    mask_a[2:6, 2:6] = True
    mask_b[10:14, 10:14] = True
    regions = [
        RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask_a),
        RiskRegion(id="same_bg_1", kind="same_bg_enclosed_region", mask=mask_b),
    ]

    plans = plan_candidates_from_regions(regions)

    assert [p.id for p in plans] == [
        "transparent_holes",
        "fill_same_bg_0",
        "fill_same_bg_1",
        "fill_all_same_color_regions",
    ]
    assert len(plans[-1].operations) == 2


def test_rule_planner_generates_repair_plan_for_alpha_keyer_disagreement():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    region = RiskRegion(id="alpha_keyer_0", kind="alpha_keyer_disagreement", mask=mask)

    plans = plan_candidates_from_regions([region])

    assert [p.id for p in plans] == ["repair_opaque_interior"]
    assert plans[0].operations[0].tool == "repair_opaque_interior"
