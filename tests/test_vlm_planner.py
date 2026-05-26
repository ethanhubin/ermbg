"""Tests for planner client adapters and model JSON parsing."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.planner import RiskRegion, build_planner_prompt_bundle, validate_candidate_plans
from ermbg.vlm_planner import RulePlannerClient, parse_candidate_plans

pytestmark = pytest.mark.core


def test_rule_planner_client_plans_from_prompt_bundle():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    region = RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask)
    bundle = build_planner_prompt_bundle(
        image_shape=(16, 16, 3),
        regions=[region],
        background_color=(255, 255, 255),
    )

    plans = RulePlannerClient().plan(bundle)

    assert [plan.id for plan in plans] == ["transparent_hole", "same_color_marking"]
    validate_candidate_plans(plans, [region])


def test_parse_candidate_plans_accepts_nested_and_flat_parameters():
    plans = parse_candidate_plans(
        {
            "candidates": [
                {
                    "id": "repair",
                    "label": "Repair",
                    "confidence": 0.7,
                    "selected": True,
                    "operations": [
                        {
                            "tool": "repair_opaque_interior",
                            "region_id": "alpha_keyer_0",
                            "parameters": {"alpha_floor": 0.9},
                        },
                        {
                            "tool": "snap_hard_edge",
                            "region_id": "hard_edge_0",
                            "alpha_floor": 0.95,
                        },
                    ],
                    "reason": "Use local repair tools.",
                }
            ]
        }
    )

    assert len(plans) == 1
    assert plans[0].id == "repair"
    assert plans[0].selected is True
    assert plans[0].operations[0].parameters == {"alpha_floor": 0.9}
    assert plans[0].operations[1].parameters == {"alpha_floor": 0.95}


def test_parse_candidate_plans_rejects_malformed_payload():
    with pytest.raises(ValueError, match="candidates list"):
        parse_candidate_plans({"candidates": {"id": "bad"}})
