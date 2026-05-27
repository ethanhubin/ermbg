"""Unit checks for the Qwen G/W game-eval batch runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ermbg.planner import RiskRegion


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "09_game_eval_qwen_batch.py"
    spec = importlib.util.spec_from_file_location("game_eval_qwen_batch", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_expected_hit_flags_conflicting_hole_fill():
    module = _load_script_module()
    expected = [["snap_hard_edge"], ["preserve_hole"]]
    selected = ["snap_hard_edge", "fill_same_color_region"]

    assert module._expected_hit(selected, expected) is True
    assert module._harmful_tools(selected, expected) == ["fill_same_color_region"]


def test_budget_regions_keeps_shadow_candidates_in_front():
    module = _load_script_module()
    mask = np.ones((4, 4), dtype=bool)
    regions = [
        RiskRegion(id="hard_edge_group_0", kind="hard_edge_candidate", mask=mask),
        RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask),
        RiskRegion(id="translucent_0", kind="translucent_candidate", mask=mask),
        RiskRegion(id="shadow_candidate_0", kind="owned_shadow_candidate", mask=mask),
    ]

    budgeted = module._budget_regions(regions, max_regions=2)

    assert [region.id for region in budgeted] == ["shadow_candidate_0", "translucent_0"]
    assert module._region_counts(budgeted) == {
        "same_bg_enclosed_region": 0,
        "alpha_keyer_disagreement": 0,
        "hard_edge_candidate": 0,
        "translucent_candidate": 1,
        "glow_soft_alpha_candidate": 0,
        "owned_shadow_candidate": 1,
    }


def test_shadow_policy_eval_requires_vlm_owned_shadow_acceptance():
    module = _load_script_module()

    miss = module._shadow_policy_eval(
        target_policy=["shadow_or_contact"],
        semantic_regions=[{"region_id": "shadow_candidate_0", "role": "background", "confidence": 0.9}],
        shadow_candidate_count=1,
    )
    hit = module._shadow_policy_eval(
        target_policy=["shadow_or_contact"],
        semantic_regions=[{"region_id": "shadow_candidate_0", "role": "shadow", "confidence": 0.9}],
        shadow_candidate_count=1,
    )

    assert miss["shadow_policy_hit"] is False
    assert hit["shadow_policy_hit"] is True
