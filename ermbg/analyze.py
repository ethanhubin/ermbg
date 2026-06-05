"""Lightweight Analyze stage for semantic candidate planning."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .pipeline_contracts import (
    AmbiguityRegion,
    AnalyzeResult,
    PreprocessDecision,
    SemanticCandidate,
)
from .preprocess import normalize_known_background_preprocess
from .router import classify_route


def _route_payload(route_decision: Any) -> dict[str, Any]:
    payload = route_decision.to_dict()
    return {
        "algorithm": payload.get("algorithm") or payload.get("route"),
        "route": payload.get("route"),
        "backend": payload.get("algorithm") or payload.get("backend") or payload.get("route"),
        "asset_kind": payload.get("asset_kind"),
        "parameter_profile": payload.get("parameter_profile"),
        "execution_profile": payload.get("execution_profile"),
        "confidence": payload.get("confidence"),
        "reasons": payload.get("reasons"),
        "params": payload.get("params"),
        "analysis": payload.get("analysis"),
        "corridorkey_analysis": payload.get("corridorkey_analysis"),
    }


def _background_from_route(route: dict[str, Any]) -> tuple[int, int, int] | None:
    params = route.get("params")
    if not isinstance(params, dict):
        return None
    color = params.get("pymatting_bg_color")
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(int(np.clip(c, 0, 255)) for c in color)
    return None


def _known_b_thresholds_from_route(route: dict[str, Any]) -> dict[str, Any]:
    params = route.get("params")
    params = params if isinstance(params, dict) else {}
    return {
        "bg_threshold": float(params.get("pymatting_bg_threshold", 3.5)),
        "fg_threshold": float(params.get("pymatting_fg_threshold", 24.0)),
        "adaptive": bool(params.get("pymatting_auto_adapt", True)),
    }


def _merge_preprocess_decisions(
    current: PreprocessDecision | None,
    addition: PreprocessDecision,
) -> PreprocessDecision:
    selected = list(current.selected) if current is not None else []
    applied = list(current.applied) if current is not None else []
    metadata = dict(current.metadata) if current is not None else {}
    for item in addition.selected:
        if item not in selected:
            selected.append(item)
    for item in addition.applied:
        if item not in applied:
            applied.append(item)
    metadata.update(addition.metadata)
    return PreprocessDecision(
        selected=selected,
        applied=applied,
        metadata=metadata,
        background_model=addition.background_model or (current.background_model if current is not None else None),
    )


def _exterior_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    work = mask.astype(np.uint8).copy()
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seeds: list[tuple[int, int]] = []
    _, xs = np.nonzero(work[0:1, :])
    seeds.extend((int(x), 0) for x in xs)
    _, xs = np.nonzero(work[-1:, :])
    seeds.extend((int(x), h - 1) for x in xs)
    ys, _ = np.nonzero(work[:, 0:1])
    seeds.extend((0, int(y)) for y in ys)
    ys, _ = np.nonzero(work[:, -1:])
    seeds.extend((w - 1, int(y)) for y in ys)
    for x, y in seeds:
        if work[y, x]:
            cv2.floodFill(work, flood, (x, y), 2)
    return work == 2


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _enclosed_near_background_regions(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    bg_distance_max: float = 3.5,
    subject_distance_min: float = 18.0,
    anchor_dilate_px: int = 2,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.25,
) -> list[AmbiguityRegion]:
    """Find near-background regions enclosed by measurable subject support.

    The gates intentionally use observable signals: distance to the route
    background color, exterior connectivity, nearby non-background material,
    and component area. This is Analyze-only evidence, not a final alpha.
    """

    h, w = image_srgb.shape[:2]
    lab = srgb_to_oklab(image_srgb)
    bg = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg_lab = srgb_to_oklab(bg).reshape(3)
    distance = oklab_distance(lab, bg_lab).astype(np.float32)
    near_bg = distance <= float(bg_distance_max)
    exterior = _exterior_mask(near_bg)
    enclosed = near_bg & ~exterior
    subject_support = distance >= float(subject_distance_min)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * int(anchor_dilate_px) + 1, 2 * int(anchor_dilate_px) + 1),
    )
    near_subject = cv2.dilate(subject_support.astype(np.uint8), kernel, iterations=1).astype(bool)
    candidate = enclosed & near_subject

    min_area = max(1, int(round(float(h * w) * float(min_area_ratio))))
    max_area = max(min_area, int(round(float(h * w) * float(max_area_ratio))))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    regions: list[AmbiguityRegion] = []
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = labels == label_idx
        region_id = f"ambiguous_enclosed_bg_{len(regions)}"
        regions.append(
            AmbiguityRegion(
                id=region_id,
                type="enclosed_near_background",
                bbox_xyxy=_bbox(comp),
                area_px=area,
                mask_ref=region_id,
                evidence={
                    "background_color": [int(c) for c in background_color],
                    "touches_exterior_background": False,
                    "enclosed_by_subject_support": True,
                    "near_background_fraction": float(near_bg[comp].mean()) if area else 0.0,
                    "bg_distance_max": float(bg_distance_max),
                    "subject_distance_min": float(subject_distance_min),
                    "anchor_dilate_px": int(anchor_dilate_px),
                },
                ambiguity={
                    "transparent_hole_score": 0.5,
                    "subject_material_score": 0.5,
                    "reason": "single_image_semantic_ambiguity",
                },
            )
        )
    return regions


def _ready_candidate() -> SemanticCandidate:
    return SemanticCandidate(
        id="auto_default",
        label="Auto default",
        intent="Use the current route/profile default interpretation.",
        default=True,
        confidence=1.0,
        risk_level="low",
        decision={"policy": "auto_default"},
        reasons=["no high-impact semantic ambiguity detected"],
    )


def _enclosed_near_bg_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
    region_ids = [region.id for region in regions]
    total_area = sum(region.area_px for region in regions)
    preview = {
        "regions": region_ids,
        "area_px": int(total_area),
        "bbox_xyxy": regions[0].bbox_xyxy if regions else [0, 0, 0, 0],
    }
    return [
        SemanticCandidate(
            id="auto_default",
            label="Auto default",
            intent="Use the current route/profile default interpretation.",
            default=True,
            confidence=0.5,
            risk_level="medium",
            decision={"policy": "auto_default"},
            regions=region_ids,
            preview=preview,
            reasons=["current pipeline executes immediately without semantic confirmation"],
        ),
        SemanticCandidate(
            id="protect_near_bg_subject",
            label="Keep internal light material",
            intent="Treat enclosed near-background pixels as subject-owned material.",
            default=False,
            confidence=0.5,
            risk_level="medium",
            decision={"enclosed_near_bg_policy": "subject"},
            regions=region_ids,
            preview=preview,
            reasons=[
                "near-background region does not touch exterior background",
                "region is adjacent to strong subject-color support",
                "single-image evidence cannot prove this is a transparent hole",
            ],
        ),
        SemanticCandidate(
            id="cut_enclosed_holes",
            label="Transparent internal holes",
            intent="Treat enclosed near-background pixels as transparent holes.",
            default=False,
            confidence=0.5,
            risk_level="medium",
            decision={"enclosed_near_bg_policy": "transparent_hole"},
            regions=region_ids,
            preview=preview,
            reasons=[
                "pixels match the known background color",
                "enclosed same-background components can be true holes in UI or icons",
            ],
        ),
    ]


def analyze_candidates(
    image_srgb: np.ndarray,
    *,
    preprocess: PreprocessDecision | None = None,
    screen_mode: str = "auto",
    preset: str = "auto",
    fallback_background_color: tuple[int, int, int] = (0, 200, 0),
) -> AnalyzeResult:
    """Run lightweight route/profile and semantic candidate analysis."""

    route_decision = classify_route(
        image_srgb,
        screen_mode=screen_mode,
        preset=preset,
        fallback_background_color=fallback_background_color,
    )
    route = _route_payload(route_decision)
    regions: list[AmbiguityRegion] = []
    background = _background_from_route(route)
    semantic_input = image_srgb
    effective_preprocess = preprocess
    if route.get("algorithm") == "pymatting_known_b" and background is not None:
        normalized, known_b_preprocess = normalize_known_background_preprocess(
            image_srgb,
            background,
            **_known_b_thresholds_from_route(route),
        )
        effective_preprocess = _merge_preprocess_decisions(preprocess, known_b_preprocess)
        semantic_input = normalized
        regions = _enclosed_near_background_regions(semantic_input, background)

    if regions:
        return AnalyzeResult(
            status="needs_decision",
            analysis_id=None,
            preprocess=effective_preprocess,
            route=route,
            ambiguity_regions=regions,
            candidates=_enclosed_near_bg_candidates(regions),
            default_candidate_id="auto_default",
            preview_assets={},
        )
    return AnalyzeResult(
        status="ready",
        analysis_id=None,
        preprocess=effective_preprocess,
        route=route,
        candidates=[_ready_candidate()],
        default_candidate_id="auto_default",
        preview_assets={},
    )


__all__ = ["analyze_candidates"]
