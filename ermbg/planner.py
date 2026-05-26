"""Schema and validation for evidence-driven matting plans.

This module is intentionally small: it defines the contract that a rule planner,
mock planner, or future VLM planner must satisfy before local ERMBG tools touch
pixels. The planner chooses regions and tools; deterministic code executes them.

``RiskRegion`` is kept as an implementation name, but architecturally these are
EvidenceRegions: local measurements that still need semantic interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RegionKind = Literal[
    "same_bg_enclosed_region",
    "alpha_keyer_disagreement",
    "hard_edge_candidate",
    "soft_edge_band",
    "opaque_interior",
    "translucent_candidate",
    "intentional_hole",
    "unknown",
]

EvidenceKind = Literal[
    "same_bg_low_alpha_enclosed",
    "keyer_fg_matting_low",
    "high_contrast_keyer_fg_matting_low",
    "soft_edge_band",
    "opaque_interior",
    "translucent_candidate",
    "intentional_hole",
    "unknown",
]

EVIDENCE_KIND_ALIASES: dict[str, EvidenceKind] = {
    "same_bg_enclosed_region": "same_bg_low_alpha_enclosed",
    "alpha_keyer_disagreement": "keyer_fg_matting_low",
    "hard_edge_candidate": "high_contrast_keyer_fg_matting_low",
    "soft_edge_band": "soft_edge_band",
    "opaque_interior": "opaque_interior",
    "translucent_candidate": "translucent_candidate",
    "intentional_hole": "intentional_hole",
    "unknown": "unknown",
}

ToolName = Literal[
    "preserve_hole",
    "fill_same_color_region",
    "repair_opaque_interior",
    "snap_hard_edge",
    "preserve_soft_alpha",
    "mark_translucent",
]


def evidence_kind_for(kind: str) -> EvidenceKind:
    """Return the planner-facing evidence label for a legacy region kind."""
    return EVIDENCE_KIND_ALIASES.get(kind, "unknown")


@dataclass
class RiskRegion:
    """A local evidence region that may need policy-level interpretation."""

    id: str
    kind: RegionKind
    mask: np.ndarray
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("RiskRegion.mask must be HxW")
        self.mask = self.mask.astype(bool, copy=False)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return metadata safe to send to a planner without embedding pixels."""
        ys, xs = np.where(self.mask)
        if ys.size:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        else:
            bbox = [0, 0, 0, 0]
        return {
            "id": self.id,
            "kind": self.kind,
            "evidence_kind": evidence_kind_for(self.kind),
            "confidence": self.confidence,
            "area": int(self.mask.sum()),
            "bbox_xyxy": bbox,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ToolSpec:
    """Machine-readable contract for a local matting tool."""

    name: ToolName
    purpose: str
    allowed_region_kinds: tuple[RegionKind, ...]
    parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    risks: tuple[str, ...] = ()

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "allowed_region_kinds": list(self.allowed_region_kinds),
            "allowed_evidence_kinds": [
                evidence_kind_for(kind) for kind in self.allowed_region_kinds
            ],
            "parameter_ranges": {k: list(v) for k, v in self.parameter_ranges.items()},
            "risks": list(self.risks),
        }


@dataclass(frozen=True)
class PlanOperation:
    """One tool invocation inside a candidate matte plan."""

    tool: ToolName
    region_id: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "region_id": self.region_id, **self.parameters}


@dataclass
class CandidatePlan:
    """A selectable interpretation composed from local tool operations."""

    id: str
    label: str
    operations: list[PlanOperation]
    confidence: float = 1.0
    reason: str = ""
    selected: bool = False

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "selected": self.selected,
            "operations": [op.to_dict() for op in self.operations],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlannerPromptBundle:
    """Serializable context passed to a rule planner or future VLM planner."""

    image: dict[str, Any]
    regions: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    instructions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "regions": self.regions,
            "tools": self.tools,
            "instructions": list(self.instructions),
        }


class PlanValidationError(ValueError):
    """Raised when a planner asks for an unsafe or unknown tool operation."""


def default_tool_catalog() -> dict[str, ToolSpec]:
    """Return the finite tool set exposed to rule/VLM planners."""
    return {
        "preserve_hole": ToolSpec(
            name="preserve_hole",
            purpose="Keep an enclosed or intentional hole transparent.",
            allowed_region_kinds=("same_bg_enclosed_region", "intentional_hole"),
            risks=("Can incorrectly remove a same-color foreground marking.",),
        ),
        "fill_same_color_region": ToolSpec(
            name="fill_same_color_region",
            purpose="Interpret an enclosed same-background-color region as foreground marking.",
            allowed_region_kinds=("same_bg_enclosed_region",),
            parameter_ranges={"alpha_floor": (0.0, 1.0)},
            risks=("Can incorrectly fill an intentional transparent opening.",),
        ),
        "repair_opaque_interior": ToolSpec(
            name="repair_opaque_interior",
            purpose="Repair low-alpha holes inside owned opaque subject support.",
            allowed_region_kinds=("alpha_keyer_disagreement", "opaque_interior"),
            parameter_ranges={"alpha_floor": (0.0, 1.0)},
            risks=("Can lift exterior fringe if topology guards are bypassed.",),
        ),
        "snap_hard_edge": ToolSpec(
            name="snap_hard_edge",
            purpose="Raise alpha on small high-contrast graphic strokes and outlines.",
            allowed_region_kinds=("hard_edge_candidate",),
            parameter_ranges={"alpha_floor": (0.0, 1.0)},
            risks=("Can make hair or soft antialiasing too hard.",),
        ),
        "preserve_soft_alpha": ToolSpec(
            name="preserve_soft_alpha",
            purpose="Protect hair, fur, smoke, or other soft boundaries from hard fills.",
            allowed_region_kinds=("soft_edge_band",),
            risks=("Can preserve existing matte errors if over-applied.",),
        ),
        "mark_translucent": ToolSpec(
            name="mark_translucent",
            purpose="Mark glass, veil, or other translucent material as partial-alpha territory.",
            allowed_region_kinds=("translucent_candidate", "unknown"),
            risks=("Does not solve foreground recovery by itself.",),
        ),
    }


def build_planner_prompt_bundle(
    *,
    image_shape: tuple[int, int] | tuple[int, int, int],
    regions: list[RiskRegion],
    background_color: tuple[int, int, int] | None = None,
    strategy: dict[str, Any] | None = None,
    instructions: list[str] | None = None,
    catalog: dict[str, ToolSpec] | None = None,
) -> PlannerPromptBundle:
    """Build a JSON-safe planner context without embedding image pixels."""
    h, w = int(image_shape[0]), int(image_shape[1])
    tool_catalog = catalog if catalog is not None else default_tool_catalog()
    image: dict[str, Any] = {
        "height": h,
        "width": w,
    }
    if len(image_shape) >= 3:
        image["channels"] = int(image_shape[2])
    if background_color is not None:
        image["background_color"] = [int(c) for c in background_color]
    if strategy is not None:
        image["strategy"] = strategy

    return PlannerPromptBundle(
        image=image,
        regions=[region.to_prompt_dict() for region in regions],
        tools=[tool.to_prompt_dict() for tool in tool_catalog.values()],
        instructions=list(instructions or []),
    )


def validate_candidate_plans(
    plans: list[CandidatePlan],
    regions: list[RiskRegion],
    catalog: dict[str, ToolSpec] | None = None,
) -> None:
    """Validate planner output before any pixel operation is executed."""
    tool_catalog = catalog if catalog is not None else default_tool_catalog()
    region_by_id = {region.id: region for region in regions}
    plan_ids: set[str] = set()

    for plan in plans:
        if not plan.id:
            raise PlanValidationError("candidate plan id must be non-empty")
        if plan.id in plan_ids:
            raise PlanValidationError(f"duplicate candidate plan id: {plan.id}")
        plan_ids.add(plan.id)
        for op in plan.operations:
            spec = tool_catalog.get(op.tool)
            if spec is None:
                raise PlanValidationError(f"unknown tool: {op.tool}")
            region = region_by_id.get(op.region_id)
            if region is None:
                raise PlanValidationError(f"unknown region_id: {op.region_id}")
            if region.kind not in spec.allowed_region_kinds:
                raise PlanValidationError(
                    f"tool {op.tool} cannot run on region kind {region.kind}"
                )
            for name, value in op.parameters.items():
                allowed = spec.parameter_ranges.get(name)
                if allowed is None:
                    raise PlanValidationError(f"tool {op.tool} does not accept parameter {name}")
                lo, hi = allowed
                try:
                    v = float(value)
                except (TypeError, ValueError) as e:
                    raise PlanValidationError(f"parameter {name} must be numeric") from e
                if not (lo <= v <= hi):
                    raise PlanValidationError(
                        f"parameter {name}={v} outside allowed range [{lo}, {hi}]"
                    )


def plan_candidates_from_regions(regions: list[RiskRegion]) -> list[CandidatePlan]:
    """Build a deterministic local CandidatePlan list from risk regions.

    This is the rule-planner stand-in for a future VLM planner. It keeps the
    same schema and validation path, so replacing it later does not affect the
    executor.
    """
    same_bg = [r for r in regions if r.kind == "same_bg_enclosed_region"]
    alpha_keyer = [r for r in regions if r.kind == "alpha_keyer_disagreement"]
    hard_edges = [r for r in regions if r.kind == "hard_edge_candidate"]
    plans: list[CandidatePlan] = []

    if len(same_bg) == 1:
        region = same_bg[0]
        plans.extend(
            [
                CandidatePlan(
                    id="transparent_hole",
                    label="透明内洞",
                    selected=True,
                    confidence=0.5,
                    operations=[PlanOperation(tool="preserve_hole", region_id=region.id)],
                    reason="Enclosed low-alpha region matches the known background color.",
                ),
                CandidatePlan(
                    id="same_color_marking",
                    label="保留同色内区",
                    selected=False,
                    confidence=0.5,
                    operations=[
                        PlanOperation(
                            tool="fill_same_color_region",
                            region_id=region.id,
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
                operations=[PlanOperation(tool="preserve_hole", region_id=r.id) for r in same_bg],
                reason="All enclosed same-background regions remain transparent.",
            )
        )
        for region in same_bg:
            plans.append(
                CandidatePlan(
                    id=f"fill_{region.id}",
                    label=f"保留同色内区 {region.id}",
                    confidence=0.35,
                    operations=[
                        PlanOperation(
                            tool="fill_same_color_region",
                            region_id=region.id,
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
                        region_id=r.id,
                        parameters={"alpha_floor": 1.0},
                    )
                    for r in same_bg
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
                        region_id=r.id,
                        parameters={"alpha_floor": 0.9},
                    )
                    for r in alpha_keyer
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
                        region_id=r.id,
                        parameters={"alpha_floor": 0.95},
                    )
                    for r in hard_edges
                ],
                reason="Small high-contrast components look like graphic hard edges.",
            )
        )

    if plans:
        validate_candidate_plans(plans, regions)
    return plans


__all__ = [
    "CandidatePlan",
    "EVIDENCE_KIND_ALIASES",
    "EvidenceKind",
    "PlanOperation",
    "PlanValidationError",
    "PlannerPromptBundle",
    "RegionKind",
    "RiskRegion",
    "ToolName",
    "ToolSpec",
    "build_planner_prompt_bundle",
    "default_tool_catalog",
    "evidence_kind_for",
    "plan_candidates_from_regions",
    "validate_candidate_plans",
]
