"""Local ownership execution helpers.

This module turns deterministic ownership scores into optional protected matte
candidates. It is the production-facing counterpart of the local ownership eval
script: callers provide the already computed base matte, and this module only
reruns matting when a coherent subject-owned soft layer needs protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidates import MatteCandidate
from .keyer import key_alpha
from .ownership import rank_regions_ownership, resolve_execution_masks
from .planner import RiskRegion
from .risk import (
    coalesce_risk_regions,
    extract_alpha_keyer_disagreement_regions,
    extract_hard_edge_candidate_regions,
    extract_same_bg_enclosed_regions,
    extract_translucent_candidate_regions,
)


@dataclass(frozen=True)
class LocalOwnershipAnalysis:
    regions: list[RiskRegion]
    ownership: list[dict[str, Any]]
    role_masks: dict[str, np.ndarray]
    raw_role_masks: dict[str, np.ndarray]
    evidence_info: dict[str, Any]

    @property
    def role_mask_pixels(self) -> dict[str, int]:
        return {role: int(mask.sum()) for role, mask in self.role_masks.items()}


def extract_local_ownership_regions(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    coalesce: bool = True,
    merge_distance_px: int = 3,
) -> tuple[list[RiskRegion], dict[str, Any]]:
    """Extract local evidence families used by ownership scoring."""
    alpha = base_rgba[..., 3].astype(np.float32) / 255.0
    same_regions, same_info = extract_same_bg_enclosed_regions(image_srgb, base_rgba, background_color)
    chroma_key = key_alpha(image_srgb, background_color, mode="chromatic")
    alpha_keyer_regions, alpha_keyer_info = extract_alpha_keyer_disagreement_regions(alpha, chroma_key)
    lum_key = key_alpha(image_srgb, background_color, mode="luminance")
    hard_edge_regions, hard_edge_info = extract_hard_edge_candidate_regions(
        image_srgb,
        alpha,
        lum_key,
        background_color,
    )
    translucent_regions, translucent_info = extract_translucent_candidate_regions(
        image_srgb,
        base_rgba,
        background_color,
    )
    raw_regions = same_regions + alpha_keyer_regions + hard_edge_regions + translucent_regions
    regions = (
        coalesce_risk_regions(raw_regions, merge_distance_px=merge_distance_px)
        if coalesce
        else raw_regions
    )
    return regions, {
        "raw_counts": _counts(raw_regions),
        "counts": _counts(regions),
        "coalesced": coalesce,
        "merge_distance_px": merge_distance_px if coalesce else 0,
        "extraction_info": {
            "same_bg_enclosed_region": same_info,
            "alpha_keyer_disagreement": alpha_keyer_info,
            "hard_edge_candidate": hard_edge_info,
            "translucent_candidate": translucent_info,
        },
    }


def analyze_local_ownership(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    confidence_min: float = 0.45,
) -> LocalOwnershipAnalysis:
    """Score local ownership roles and resolve execution masks."""
    bg = tuple(int(c) for c in background_color)
    regions, evidence_info = extract_local_ownership_regions(image_srgb, base_rgba, bg)
    ownership = rank_regions_ownership(image_srgb, base_rgba, bg, regions)
    raw_masks = selected_role_masks(ownership, regions, image_srgb.shape[:2], confidence_min=confidence_min)
    role_masks = resolve_execution_masks(raw_masks, image_srgb.shape[:2])
    return LocalOwnershipAnalysis(
        regions=regions,
        ownership=ownership,
        role_masks=role_masks,
        raw_role_masks=raw_masks,
        evidence_info=evidence_info,
    )


def selected_role_masks(
    ownership: list[dict[str, Any]],
    regions: list[RiskRegion],
    shape: tuple[int, int],
    *,
    confidence_min: float = 0.45,
) -> dict[str, np.ndarray]:
    """Aggregate selected ownership roles into exact region masks."""
    masks = {
        "hole": np.zeros(shape, dtype=bool),
        "opaque_subject": np.zeros(shape, dtype=bool),
        "subject_soft_layer": np.zeros(shape, dtype=bool),
        "shadow_like_layer": np.zeros(shape, dtype=bool),
        "conservative_unknown": np.zeros(shape, dtype=bool),
    }
    region_by_id = {str(region.id): region for region in regions}
    for item in ownership:
        selected = item.get("selected") if isinstance(item, dict) else None
        region_meta = item.get("region") if isinstance(item, dict) else None
        if not isinstance(selected, dict) or not isinstance(region_meta, dict):
            continue
        role = selected.get("role")
        if role not in masks:
            continue
        try:
            confidence = float(selected.get("confidence", selected.get("score", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < confidence_min:
            continue
        region = region_by_id.get(str(region_meta.get("id", "")))
        if region is not None:
            masks[role] |= np.asarray(region.mask, dtype=bool)
    return masks


def generate_local_ownership_candidate(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    backend: str = "auto",
    segmenter: Any | None = None,
    soft_mask: np.ndarray | None = None,
    shadow_mode: str = "on",
) -> MatteCandidate | None:
    """Legacy protected rerun candidate.

    The old implementation re-entered the removed ERMBG full-matting pipeline
    with semantic protection masks. Keep the public hook as a no-op so Web
    candidate rendering and old callers remain stable while local ownership
    evidence can still be inspected through ``analyze_local_ownership``.
    """
    del backend, segmenter, soft_mask, shadow_mode
    analysis = analyze_local_ownership(image_srgb, base_rgba, background_color)
    return None


def _counts(regions: list[RiskRegion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        counts[str(region.kind)] = counts.get(str(region.kind), 0) + 1
    return {
        "same_bg_enclosed_region": counts.get("same_bg_enclosed_region", 0),
        "alpha_keyer_disagreement": counts.get("alpha_keyer_disagreement", 0),
        "hard_edge_candidate": counts.get("hard_edge_candidate", 0),
        "translucent_candidate": counts.get("translucent_candidate", 0),
    }


__all__ = [
    "LocalOwnershipAnalysis",
    "analyze_local_ownership",
    "extract_local_ownership_regions",
    "generate_local_ownership_candidate",
    "selected_role_masks",
]
