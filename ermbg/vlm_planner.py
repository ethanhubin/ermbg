"""Planner client adapters for rule-based and future VLM planning.

No remote model is called here. The module defines the boundary: planner clients
consume a ``PlannerPromptBundle`` and return constrained ``CandidatePlan``
objects that the local executor can validate and run.
"""

from __future__ import annotations

from typing import Any, Protocol

from .planner import CandidatePlan, PlanOperation, PlannerPromptBundle


class PlannerClient(Protocol):
    """Interface implemented by rule, mock, or future VLM planner clients."""

    def plan(self, bundle: PlannerPromptBundle) -> list[CandidatePlan]:
        """Return candidate plans for the supplied context."""


class RulePlannerClient:
    """Deterministic planner client that mirrors the current local policy."""

    def plan(self, bundle: PlannerPromptBundle) -> list[CandidatePlan]:
        regions = bundle.regions
        same_bg = [r for r in regions if r.get("kind") == "same_bg_enclosed_region"]
        alpha_keyer = [r for r in regions if r.get("kind") == "alpha_keyer_disagreement"]
        hard_edges = [r for r in regions if r.get("kind") == "hard_edge_candidate"]
        plans: list[CandidatePlan] = []

        if len(same_bg) == 1:
            region_id = str(same_bg[0]["id"])
            plans.extend(
                [
                    CandidatePlan(
                        id="transparent_hole",
                        label="透明内洞",
                        selected=True,
                        confidence=0.5,
                        operations=[PlanOperation(tool="preserve_hole", region_id=region_id)],
                        reason="Enclosed low-alpha region matches the known background color.",
                    ),
                    CandidatePlan(
                        id="same_color_marking",
                        label="保留同色内区",
                        confidence=0.5,
                        operations=[
                            PlanOperation(
                                tool="fill_same_color_region",
                                region_id=region_id,
                                parameters={"alpha_floor": 1.0},
                            )
                        ],
                        reason="The same pixels may be a foreground marking whose color equals the background.",
                    ),
                ]
            )
        elif len(same_bg) > 1:
            plans.append(
                CandidatePlan(
                    id="transparent_holes",
                    label="透明内洞",
                    selected=True,
                    confidence=0.5,
                    operations=[
                        PlanOperation(tool="preserve_hole", region_id=str(region["id"]))
                        for region in same_bg
                    ],
                    reason="All enclosed same-background regions remain transparent.",
                )
            )
            for region in same_bg:
                region_id = str(region["id"])
                plans.append(
                    CandidatePlan(
                        id=f"fill_{region_id}",
                        label=f"保留同色内区 {region_id}",
                        confidence=0.35,
                        operations=[
                            PlanOperation(
                                tool="fill_same_color_region",
                                region_id=region_id,
                                parameters={"alpha_floor": 1.0},
                            )
                        ],
                        reason="This enclosed same-background region may be a foreground marking.",
                    )
                )
            plans.append(
                CandidatePlan(
                    id="fill_all_same_color_regions",
                    label="保留全部同色内区",
                    confidence=0.3,
                    operations=[
                        PlanOperation(
                            tool="fill_same_color_region",
                            region_id=str(region["id"]),
                            parameters={"alpha_floor": 1.0},
                        )
                        for region in same_bg
                    ],
                    reason="All enclosed same-background regions may belong to the subject.",
                )
            )

        if not same_bg and alpha_keyer:
            plans.append(
                CandidatePlan(
                    id="repair_opaque_interior",
                    label="修复主体内部低 alpha",
                    selected=True,
                    confidence=0.7,
                    operations=[
                        PlanOperation(
                            tool="repair_opaque_interior",
                            region_id=str(region["id"]),
                            parameters={"alpha_floor": 0.9},
                        )
                        for region in alpha_keyer
                    ],
                    reason="Keyer evidence supports foreground where matting alpha is low.",
                )
            )

        if not same_bg and not alpha_keyer and hard_edges:
            plans.append(
                CandidatePlan(
                    id="snap_hard_edges",
                    label="修复硬边描边",
                    selected=True,
                    confidence=0.75,
                    operations=[
                        PlanOperation(
                            tool="snap_hard_edge",
                            region_id=str(region["id"]),
                            parameters={"alpha_floor": 0.95},
                        )
                        for region in hard_edges
                    ],
                    reason="Small high-contrast components look like graphic hard edges.",
                )
            )

        return plans


def parse_candidate_plans(payload: dict[str, Any] | list[dict[str, Any]]) -> list[CandidatePlan]:
    """Parse model/mock JSON into ``CandidatePlan`` objects.

    Accepts either ``{"candidates": [...]}`` or the candidate list directly.
    Operation parameters may be flattened next to ``tool``/``region_id``.
    """
    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(raw_candidates, list):
        raise ValueError("planner payload must contain a candidates list")

    plans: list[CandidatePlan] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("candidate entries must be objects")
        raw_ops = raw.get("operations", [])
        if not isinstance(raw_ops, list):
            raise ValueError("candidate operations must be a list")
        operations: list[PlanOperation] = []
        for raw_op in raw_ops:
            if not isinstance(raw_op, dict):
                raise ValueError("operation entries must be objects")
            tool = raw_op.get("tool")
            region_id = raw_op.get("region_id")
            if not isinstance(tool, str) or not isinstance(region_id, str):
                raise ValueError("operation requires string tool and region_id")
            nested_parameters = raw_op.get("parameters", {})
            if nested_parameters is not None and not isinstance(nested_parameters, dict):
                raise ValueError("operation parameters must be an object")
            parameters = dict(nested_parameters or {})
            parameters.update(
                {
                    key: value
                    for key, value in raw_op.items()
                    if key not in {"tool", "region_id", "parameters"}
                }
            )
            operations.append(PlanOperation(tool=tool, region_id=region_id, parameters=parameters))

        plan_id = raw.get("id")
        label = raw.get("label")
        if not isinstance(plan_id, str) or not isinstance(label, str):
            raise ValueError("candidate requires string id and label")
        plans.append(
            CandidatePlan(
                id=plan_id,
                label=label,
                operations=operations,
                confidence=float(raw.get("confidence", 1.0)),
                reason=str(raw.get("reason", "")),
                selected=bool(raw.get("selected", False)),
            )
        )
    return plans


__all__ = ["PlannerClient", "RulePlannerClient", "parse_candidate_plans"]
