"""Deterministic execution of planner-selected matting tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .keyer import key_alpha, repair_alpha_with_known_bg_key, repair_hard_edge_alpha
from .planner import CandidatePlan, RiskRegion, validate_candidate_plans


@dataclass
class PlanExecutionResult:
    """RGBA output and debug metadata produced by one candidate plan."""

    plan: CandidatePlan
    rgba: np.ndarray
    regions: list[dict[str, Any]]
    operation_results: list[dict[str, Any]] = field(default_factory=list)

    def debug_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "regions": self.regions,
            "operation_results": self.operation_results,
        }


def execute_plan(
    plan: CandidatePlan,
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    *,
    background_color: tuple[int, int, int] | None = None,
) -> PlanExecutionResult:
    """Execute one validated candidate plan using registered local tools."""
    validate_candidate_plans([plan], regions)
    region_by_id = {region.id: region for region in regions}
    rgba = base_rgba.copy()
    operation_results: list[dict[str, Any]] = []

    for op in plan.operations:
        mask = region_by_id[op.region_id].mask
        if op.tool == "preserve_hole":
            operation_results.append({"tool": op.tool, "region_id": op.region_id, "applied_pixels": 0})
            continue
        if op.tool == "fill_same_color_region":
            alpha_floor = float(op.parameters.get("alpha_floor", 1.0))
            alpha_u8 = int(np.clip(alpha_floor, 0.0, 1.0) * 255 + 0.5)
            rgba[mask, :3] = image_srgb[mask]
            rgba[mask, 3] = np.maximum(rgba[mask, 3], alpha_u8)
            operation_results.append(
                {"tool": op.tool, "region_id": op.region_id, "applied_pixels": int(mask.sum())}
            )
            continue
        if op.tool == "repair_opaque_interior":
            if background_color is None:
                raise ValueError("repair_opaque_interior requires background_color")
            alpha_floor = float(op.parameters.get("alpha_floor", 0.9))
            bg = tuple(int(c) for c in background_color)
            alpha = rgba[..., 3].astype(np.float32) / 255.0
            full_color_key = key_alpha(image_srgb, bg, mode="chromatic")
            repaired, info = repair_alpha_with_known_bg_key(
                alpha,
                full_color_key,
                target_alpha_floor=alpha_floor,
            )
            repaired_u8 = (np.clip(repaired, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
            before = rgba[..., 3].copy()
            rgba[mask, 3] = np.maximum(rgba[mask, 3], repaired_u8[mask])
            changed = mask & (rgba[..., 3] > before)
            rgba[changed, :3] = image_srgb[changed]
            operation_results.append(
                {
                    "tool": op.tool,
                    "region_id": op.region_id,
                    "applied_pixels": int(changed.sum()),
                    "repair_info": info,
                }
            )
            continue
        if op.tool == "snap_hard_edge":
            if background_color is None:
                raise ValueError("snap_hard_edge requires background_color")
            alpha_floor = float(op.parameters.get("alpha_floor", 0.95))
            bg = tuple(int(c) for c in background_color)
            alpha = rgba[..., 3].astype(np.float32) / 255.0
            edge_key = key_alpha(image_srgb, bg, mode="luminance")
            repaired, info = repair_hard_edge_alpha(
                image_srgb,
                alpha,
                edge_key,
                bg,
                target_alpha_floor=alpha_floor,
            )
            repaired_u8 = (np.clip(repaired, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
            before = rgba[..., 3].copy()
            rgba[mask, 3] = np.maximum(rgba[mask, 3], repaired_u8[mask])
            changed = mask & (rgba[..., 3] > before)
            rgba[changed, :3] = image_srgb[changed]
            operation_results.append(
                {
                    "tool": op.tool,
                    "region_id": op.region_id,
                    "applied_pixels": int(changed.sum()),
                    "repair_info": info,
                }
            )
            continue
        raise ValueError(f"plan executor does not implement tool {op.tool!r}")

    return PlanExecutionResult(
        plan=plan,
        rgba=rgba,
        regions=[region.to_prompt_dict() for region in regions],
        operation_results=operation_results,
    )


def execute_plans(
    plans: list[CandidatePlan],
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    *,
    background_color: tuple[int, int, int] | None = None,
) -> list[PlanExecutionResult]:
    """Validate and execute a variable-length candidate plan list."""
    validate_candidate_plans(plans, regions)
    return [
        execute_plan(
            plan,
            regions,
            image_srgb,
            base_rgba,
            background_color=background_color,
        )
        for plan in plans
    ]


__all__ = ["PlanExecutionResult", "execute_plan", "execute_plans"]
