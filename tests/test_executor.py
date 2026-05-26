"""Tests for deterministic CandidatePlan execution."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.executor import execute_plan, execute_plans
from ermbg.planner import CandidatePlan, PlanOperation, RiskRegion

pytestmark = pytest.mark.core


def test_execute_plan_returns_debug_metadata_for_fill_tool():
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[4:12, 4:12] = (10, 20, 30)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    region = RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask)
    plan = CandidatePlan(
        id="fill",
        label="Fill",
        operations=[
            PlanOperation(
                tool="fill_same_color_region",
                region_id="same_bg_0",
                parameters={"alpha_floor": 0.5},
            )
        ],
    )

    result = execute_plan(plan, [region], image, rgba)

    assert result.rgba[mask, 3].min() == 128
    assert result.operation_results == [
        {"tool": "fill_same_color_region", "region_id": "same_bg_0", "applied_pixels": 64}
    ]
    assert result.debug_dict()["regions"][0]["id"] == "same_bg_0"


def test_execute_plans_requires_background_for_known_bg_tools():
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    region = RiskRegion(id="alpha_keyer_0", kind="alpha_keyer_disagreement", mask=mask)
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

    with pytest.raises(ValueError, match="requires background_color"):
        execute_plans([plan], [region], image, rgba)
