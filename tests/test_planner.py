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
    default_tool_catalog,
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
