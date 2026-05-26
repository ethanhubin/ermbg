"""Candidate generation for genuine matting ambiguities.

The first supported ambiguity is a same-background-color enclosed region:
observed pixels match the known background, the base matte makes them
transparent, and local evidence cannot decide whether this is an intentional
hole or a same-color foreground marking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .executor import PlanExecutionResult, execute_plans
from .planner import CandidatePlan, RiskRegion, build_planner_prompt_bundle
from .risk import extract_same_bg_enclosed_regions
from .vlm_planner import PlannerClient, RulePlannerClient


@dataclass
class MatteCandidate:
    """A selectable RGBA candidate derived from the same base matte."""

    id: str
    label: str
    rgba: np.ndarray
    kind: str = "RGBA PNG"
    selected: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def _candidate_from_execution(result: PlanExecutionResult) -> MatteCandidate:
    plan = result.plan
    return MatteCandidate(
        id=plan.id,
        label=plan.label,
        rgba=result.rgba,
        selected=plan.selected,
        debug=result.debug_dict(),
    )


def generate_matte_candidates(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    planner_client: PlannerClient | None = None,
) -> list[MatteCandidate]:
    """Return selectable matte candidates for ambiguous local regions.

    Always returns at least the base result. If a same-B enclosed region is
    detected, the base result is labeled as the transparent-hole interpretation,
    and a second candidate fills the region as a same-color foreground marking.
    """
    bg = tuple(int(c) for c in background_color)
    regions, info = extract_same_bg_enclosed_regions(image_srgb, base_rgba, bg)
    bundle = build_planner_prompt_bundle(
        image_shape=image_srgb.shape,
        regions=regions,
        background_color=bg,
        instructions=[
            "Return CandidatePlan JSON only.",
            "Use only registered tools and existing region_id values.",
            "Do not output alpha, RGBA, masks, or image-processing code.",
        ],
    )
    client = planner_client if planner_client is not None else RulePlannerClient()
    plans = client.plan(bundle)
    if not plans:
        return [
            MatteCandidate(
                id="auto",
                label="自动结果",
                rgba=base_rgba.copy(),
                selected=True,
                debug={"same_bg_ambiguity": info, "planner_bundle": bundle.to_dict()},
            )
        ]

    executions = execute_plans(plans, regions, image_srgb, base_rgba, background_color=bg)
    candidates = [_candidate_from_execution(result) for result in executions]
    for candidate in candidates:
        candidate.debug["same_bg_ambiguity"] = info
        candidate.debug["planner_bundle"] = bundle.to_dict()
    return candidates


def execute_candidate_plans(
    plans: list[CandidatePlan],
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int] | None = None,
) -> list[MatteCandidate]:
    """Validate and execute a planner-provided variable-length candidate list.

    This is the first local executor used by rule/mock planners. The supported
    operation subset is intentionally small and will expand as tools are wrapped.
    """
    executions = execute_plans(plans, regions, image_srgb, base_rgba, background_color=background_color)
    return [_candidate_from_execution(result) for result in executions]


__all__ = ["MatteCandidate", "execute_candidate_plans", "generate_matte_candidates"]
