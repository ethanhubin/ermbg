"""Candidate generation for genuine matting ambiguities.

The first supported ambiguity is a same-background-color enclosed region:
observed pixels match the known background, the base matte makes them
transparent, and local evidence cannot decide whether this is an intentional
hole or a same-color foreground marking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .planner import CandidatePlan, PlanOperation, RiskRegion, validate_candidate_plans


@dataclass
class MatteCandidate:
    """A selectable RGBA candidate derived from the same base matte."""

    id: str
    label: str
    rgba: np.ndarray
    kind: str = "RGBA PNG"
    selected: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def _touches_border(component: np.ndarray) -> bool:
    return bool(
        component[0, :].any()
        or component[-1, :].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )


def _same_bg_ambiguity_mask(
    image_srgb: np.ndarray,
    rgba: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    bg_distance_max: float = 3.0,
    alpha_max: float = 0.20,
    fg_anchor_threshold: float = 0.85,
    anchor_dilate_px: int = 2,
    min_area_ratio: float = 0.0003,
    max_area_ratio: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find enclosed low-alpha regions whose observed color equals known B."""
    if image_srgb.shape[:2] != rgba.shape[:2]:
        raise ValueError("image_srgb and rgba must share HxW")

    h, w = image_srgb.shape[:2]
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    close_to_bg = oklab_distance(lab, bg_lab) <= float(bg_distance_max)
    candidate = close_to_bg & (alpha <= float(alpha_max))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    min_area = max(1.0, min_area_ratio * float(h * w))
    max_area = max(min_area, max_area_ratio * float(h * w))
    confident_fg = alpha >= float(fg_anchor_threshold)
    kernel = np.ones((3, 3), np.uint8)

    accepted = np.zeros((h, w), dtype=bool)
    accepted_areas: list[int] = []
    rejected = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = labels == i
        if area < min_area or area > max_area or _touches_border(comp):
            rejected += 1
            continue
        near_comp = cv2.dilate(
            comp.astype(np.uint8),
            kernel,
            iterations=max(1, int(anchor_dilate_px)),
        ).astype(bool)
        if not (near_comp & confident_fg).any():
            rejected += 1
            continue
        accepted |= comp
        accepted_areas.append(area)

    return accepted, {
        "accepted_components": len(accepted_areas),
        "accepted_pixels": int(accepted.sum()),
        "component_areas": accepted_areas,
        "rejected_components": rejected,
        "bg_distance_max": bg_distance_max,
        "alpha_max": alpha_max,
    }


def _same_bg_candidate_plans(region_id: str) -> list[CandidatePlan]:
    return [
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
            selected=False,
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


def _execute_candidate_plan(
    plan: CandidatePlan,
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
) -> MatteCandidate:
    region_by_id = {region.id: region for region in regions}
    rgba = base_rgba.copy()

    for op in plan.operations:
        mask = region_by_id[op.region_id].mask
        if op.tool == "preserve_hole":
            continue
        if op.tool == "fill_same_color_region":
            alpha_floor = float(op.parameters.get("alpha_floor", 1.0))
            alpha_u8 = int(np.clip(alpha_floor, 0.0, 1.0) * 255 + 0.5)
            rgba[mask, :3] = image_srgb[mask]
            rgba[mask, 3] = np.maximum(rgba[mask, 3], alpha_u8)
            continue
        raise ValueError(f"candidate executor does not implement tool {op.tool!r}")

    return MatteCandidate(
        id=plan.id,
        label=plan.label,
        rgba=rgba,
        selected=plan.selected,
        debug={
            "plan": plan.to_dict(),
            "regions": [region.to_prompt_dict() for region in regions],
        },
    )


def generate_matte_candidates(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
) -> list[MatteCandidate]:
    """Return selectable matte candidates for ambiguous local regions.

    Always returns at least the base result. If a same-B enclosed region is
    detected, the base result is labeled as the transparent-hole interpretation,
    and a second candidate fills the region as a same-color foreground marking.
    """
    bg = tuple(int(c) for c in background_color)
    ambiguity, info = _same_bg_ambiguity_mask(image_srgb, base_rgba, bg)
    if not ambiguity.any():
        return [
            MatteCandidate(
                id="auto",
                label="自动结果",
                rgba=base_rgba.copy(),
                selected=True,
                debug={"same_bg_ambiguity": info},
            )
        ]

    regions = [
        RiskRegion(
            id="same_bg_0",
            kind="same_bg_enclosed_region",
            mask=ambiguity,
            confidence=1.0,
            evidence={"same_bg_ambiguity": info},
        )
    ]
    plans = _same_bg_candidate_plans(region_id="same_bg_0")
    validate_candidate_plans(plans, regions)
    candidates = [
        _execute_candidate_plan(plan, regions, image_srgb, base_rgba)
        for plan in plans
    ]
    for candidate in candidates:
        candidate.debug["same_bg_ambiguity"] = info
    return candidates


def execute_candidate_plans(
    plans: list[CandidatePlan],
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
) -> list[MatteCandidate]:
    """Validate and execute a planner-provided variable-length candidate list.

    This is the first local executor used by rule/mock planners. The supported
    operation subset is intentionally small and will expand as tools are wrapped.
    """
    validate_candidate_plans(plans, regions)
    return [
        _execute_candidate_plan(plan, regions, image_srgb, base_rgba)
        for plan in plans
    ]


__all__ = ["MatteCandidate", "execute_candidate_plans", "generate_matte_candidates"]
