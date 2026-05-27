"""Tests for deterministic CandidatePlan execution."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.executor import execute_plan, execute_plans
from ermbg.planner import CandidatePlan, PlanOperation, RiskRegion, default_tool_catalog

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
    assert result.operation_results[0] == {
        "tool": "fill_same_color_region",
        "region_id": "same_bg_0",
        "applied_pixels": 64,
        "protected_pixels": 0,
    }
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


def test_executor_recognizes_every_default_catalog_tool():
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    kind_by_tool = {
        "preserve_hole": "same_bg_enclosed_region",
        "fill_same_color_region": "same_bg_enclosed_region",
        "repair_opaque_interior": "alpha_keyer_disagreement",
        "snap_hard_edge": "hard_edge_candidate",
        "preserve_soft_alpha": "soft_edge_band",
        "mark_translucent": "translucent_candidate",
    }

    for tool in default_tool_catalog():
        region = RiskRegion(id="r0", kind=kind_by_tool[tool], mask=mask)
        plan = CandidatePlan(
            id=f"plan_{tool}",
            label=tool,
            operations=[PlanOperation(tool=tool, region_id="r0")],
        )

        result = execute_plan(plan, [region], image, rgba, background_color=(255, 255, 255))

        assert result.operation_results[0]["tool"] == tool


def test_preserve_soft_alpha_protects_overlap_from_fill():
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    image[2:10, 2:10] = (20, 40, 60)
    rgba = np.zeros((12, 12, 4), dtype=np.uint8)
    fill_mask = np.zeros((12, 12), dtype=bool)
    fill_mask[2:10, 2:10] = True
    soft_mask = np.zeros((12, 12), dtype=bool)
    soft_mask[4:8, 4:8] = True
    regions = [
        RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=fill_mask),
        RiskRegion(id="soft_0", kind="soft_edge_band", mask=soft_mask),
    ]
    plan = CandidatePlan(
        id="fill_except_soft",
        label="Fill except soft",
        operations=[
            PlanOperation(tool="preserve_soft_alpha", region_id="soft_0"),
            PlanOperation(
                tool="fill_same_color_region",
                region_id="same_bg_0",
                parameters={"alpha_floor": 1.0},
            ),
        ],
    )

    result = execute_plan(plan, regions, image, rgba)

    assert result.rgba[soft_mask, 3].max() == 0
    assert result.rgba[fill_mask & ~soft_mask, 3].min() == 255
    assert result.operation_results[1]["protected_pixels"] == int(soft_mask.sum())


def test_external_protected_mask_blocks_alpha_repair():
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    image[3:9, 3:9] = (80, 80, 80)
    rgba = np.zeros((12, 12, 4), dtype=np.uint8)
    rgba[3:9, 3:9, :3] = (0, 0, 0)
    rgba[3:9, 3:9, 3] = 96
    repair_mask = np.zeros((12, 12), dtype=bool)
    repair_mask[3:9, 3:9] = True
    protected_mask = np.zeros((12, 12), dtype=bool)
    protected_mask[4:8, 4:8] = True
    region = RiskRegion(id="alpha_keyer_0", kind="alpha_keyer_disagreement", mask=repair_mask)
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

    result = execute_plan(
        plan,
        [region],
        image,
        rgba,
        background_color=(255, 255, 255),
        protected_mask=protected_mask,
    )

    np.testing.assert_array_equal(result.rgba[protected_mask], rgba[protected_mask])
    assert result.operation_results[0]["protected_pixels"] == int(protected_mask.sum())


def test_mark_translucent_protects_overlap_from_hard_edge_snap():
    h, w = 24, 24
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    image[11, 6:18] = (20, 20, 20)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[12:16, 6:18, 3] = 255
    rgba[11, 6:18, 3] = 64
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[11, 6:18] = True
    regions = [
        RiskRegion(id="translucent_0", kind="translucent_candidate", mask=edge_mask),
        RiskRegion(id="hard_edge_0", kind="hard_edge_candidate", mask=edge_mask),
    ]
    plan = CandidatePlan(
        id="protect_translucent",
        label="Protect translucent",
        operations=[
            PlanOperation(tool="mark_translucent", region_id="translucent_0"),
            PlanOperation(
                tool="snap_hard_edge",
                region_id="hard_edge_0",
                parameters={"alpha_floor": 0.95},
            ),
        ],
    )

    result = execute_plan(plan, regions, image, rgba, background_color=(255, 255, 255))

    np.testing.assert_array_equal(result.rgba[..., 3], rgba[..., 3])
    assert result.operation_results[1]["applied_pixels"] == 0
    assert result.operation_results[1]["protected_pixels"] == int(edge_mask.sum())
