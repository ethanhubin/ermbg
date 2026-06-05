"""Lightweight Analyze stage for semantic candidate planning."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
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
from .preprocess import BACKGROUND_REPAIR, repair_known_background_preprocess
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


def _known_b_trimap_kwargs_from_route(route: dict[str, Any]) -> dict[str, Any]:
    params = route.get("params")
    params = params if isinstance(params, dict) else {}
    background = _background_from_route(route)
    return {
        "background_color": background,
        "bg_threshold": float(params.get("pymatting_bg_threshold", 3.5)),
        "fg_threshold": float(params.get("pymatting_fg_threshold", 24.0)),
        "boundary_band_px": int(params.get("pymatting_boundary_band_px", 2)),
        "adaptive": bool(params.get("pymatting_auto_adapt", True)),
        "trimap_mode": str(params.get("pymatting_trimap_mode") or "standard"),
        "unknown_grow_px": int(params.get("pymatting_unknown_grow_px", 0)),
    }


def _known_b_semantic_region_settings(route: dict[str, Any]) -> dict[str, Any]:
    profile = str(route.get("parameter_profile") or "")
    if profile == "translucent_button":
        return {
            # Translucent/glossy known-B buttons contain white or near-white
            # internal highlights that are not literal holes but can still be
            # mis-owned as background. Use a wider near-background evidence
            # band only for this measured profile; ordinary hard graphics keep
            # the tight background match below so true cutout holes remain
            # precise and do not balloon into subject material.
            "bg_distance_max": 20.0,
            "subject_distance_min": 18.0,
            "anchor_dilate_px": 3,
            "min_area_ratio": 0.00035,
            "max_area_ratio": 0.18,
            "evidence_mode": "translucent_known_b_material_band",
        }
    return {
        "bg_distance_max": 3.5,
        "subject_distance_min": 18.0,
        "anchor_dilate_px": 2,
        "min_area_ratio": 0.0005,
        "max_area_ratio": 0.25,
        "evidence_mode": "tight_enclosed_background_match",
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


def _png_data_url_rgb(rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to encode Analyze preview PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _png_data_url_rgba(rgba: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise RuntimeError("failed to encode Analyze preview PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _analysis_id(
    image_srgb: np.ndarray,
    *,
    route: dict[str, Any],
    preprocess: PreprocessDecision | None,
    regions: list[AmbiguityRegion],
) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(image_srgb)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    digest.update(
        _stable_json(
            {
                "route": route,
                "preprocess": preprocess.to_dict() if preprocess is not None else None,
                "regions": [region.to_dict() for region in regions],
            }
        ).encode("utf-8")
    )
    return f"analysis_{digest.hexdigest()[:20]}"


def _with_mask_refs(regions: list[AmbiguityRegion], analysis_id: str) -> list[AmbiguityRegion]:
    return [
        replace(region, mask_ref=f"{analysis_id}:region_mask:{region.id}")
        for region in regions
    ]


def _candidate_mask(candidate: SemanticCandidate, region_masks: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for region_id in candidate.regions:
        region_mask = region_masks.get(region_id)
        if region_mask is not None:
            mask |= region_mask
    return mask


def _overlay_preview(image_srgb: np.ndarray, mask: np.ndarray, rgba: tuple[int, int, int, int]) -> np.ndarray:
    rgba_image = np.dstack(
        [
            image_srgb,
            np.full(image_srgb.shape[:2], 255, dtype=np.uint8),
        ]
    ).astype(np.float32)
    if bool(mask.any()):
        color = np.asarray(rgba, dtype=np.float32)
        alpha = color[3] / 255.0
        rgba_image[mask, :3] = rgba_image[mask, :3] * (1.0 - alpha) + color[:3] * alpha
        rgba_image[mask, 3] = 255.0
    return np.clip(rgba_image + 0.5, 0, 255).astype(np.uint8)


def _trimap_preview_from_masks(sure_bg: np.ndarray, unknown: np.ndarray, sure_fg: np.ndarray) -> np.ndarray:
    trimap = np.full((*sure_bg.shape, 3), 128, dtype=np.uint8)
    trimap[sure_bg] = (0, 0, 0)
    trimap[unknown] = (128, 128, 128)
    trimap[sure_fg] = (255, 255, 255)
    return trimap


def _known_b_trimap_preview(
    image_srgb: np.ndarray,
    route: dict[str, Any],
    candidate: SemanticCandidate,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    kwargs = _known_b_trimap_kwargs_from_route(route)
    background = kwargs.pop("background_color")
    if background is None:
        return None
    from .pymatting_refine import build_known_background_trimap

    trimap, info = build_known_background_trimap(
        image_srgb,
        background,
        semantic_decision=candidate.decision,
        **kwargs,
    )
    preview = _trimap_preview_from_masks(trimap.sure_bg, trimap.unknown, trimap.sure_fg)
    metadata = {
        "states": {
            "sure_bg": {"value": 0, "color": [0, 0, 0], "pixels": int(trimap.sure_bg.sum())},
            "unknown": {"value": 128, "color": [128, 128, 128], "pixels": int(trimap.unknown.sum())},
            "sure_fg": {"value": 255, "color": [255, 255, 255], "pixels": int(trimap.sure_fg.sum())},
        },
        "source": "build_known_background_trimap",
        "method": info.get("method"),
        "semantic_decision": info.get("semantic_decision"),
    }
    return preview, metadata


def _hint_preview(mask: np.ndarray, shape: tuple[int, int], *, keep: bool) -> np.ndarray:
    hint = np.full((*shape, 4), (23, 28, 26, 255), dtype=np.uint8)
    if bool(mask.any()):
        hint[mask] = (0, 190, 255, 190) if keep else (255, 86, 72, 190)
    return hint


def _attach_preview_assets(
    image_srgb: np.ndarray,
    *,
    route: dict[str, Any],
    route_algorithm: str,
    analysis_id: str,
    regions: list[AmbiguityRegion],
    region_masks: dict[str, np.ndarray],
    candidates: list[SemanticCandidate],
) -> tuple[list[SemanticCandidate], dict[str, Any]]:
    """Generate cheap Analyze previews without running a matting backend."""

    h, w = image_srgb.shape[:2]
    preview_assets: dict[str, Any] = {
        "schema": "ermbg.analysis_preview_assets.v1",
        "image_space": "analyze_semantic_input",
    }
    for region in regions:
        mask = region_masks.get(region.id)
        if mask is None:
            continue
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask] = (255, 255, 255, 255)
        key = f"region_mask:{region.id}"
        preview_assets[key] = {
            "kind": "region_mask",
            "region_id": region.id,
            "media_type": "image/png",
            "encoding": "data_url",
            "data_url": _png_data_url_rgba(rgba),
        }

    if route_algorithm == "pymatting_known_b":
        preview_modes = ("overlay", "trimap")
    elif route_algorithm == "corridorkey":
        preview_modes = ("overlay", "hint")
    else:
        preview_modes = ("overlay",)

    updated: list[SemanticCandidate] = []
    for candidate in candidates:
        decision = candidate.decision or {}
        policy = str(decision.get("enclosed_near_bg_policy") or decision.get("screen_material_policy") or "")
        cut = policy in {"transparent_hole", "background", "remove"}
        keep = policy in {"subject", "foreground", "keep", "preserve"} or not cut
        color = (0, 190, 255, 96) if keep else (255, 86, 72, 112)
        mask = _candidate_mask(candidate, region_masks, (h, w))
        refs: dict[str, str] = {}
        trimap_preview = _known_b_trimap_preview(image_srgb, route, candidate) if route_algorithm == "pymatting_known_b" else None
        trimap_image = trimap_preview[0] if trimap_preview is not None else _trimap_preview_from_masks(
            np.zeros((h, w), dtype=bool),
            np.ones((h, w), dtype=bool),
            np.zeros((h, w), dtype=bool),
        )
        trimap_metadata = trimap_preview[1] if trimap_preview is not None else {
            "states": {
                "sure_bg": {"value": 0, "color": [0, 0, 0], "pixels": 0},
                "unknown": {"value": 128, "color": [128, 128, 128], "pixels": int(h * w)},
                "sure_fg": {"value": 255, "color": [255, 255, 255], "pixels": 0},
            },
            "source": "fallback_all_unknown",
        }
        mode_images = {
            "overlay": _png_data_url_rgba(_overlay_preview(image_srgb, mask, color)),
            "trimap": _png_data_url_rgb(trimap_image),
            "hint": _png_data_url_rgba(_hint_preview(mask, (h, w), keep=keep)),
        }
        for mode in preview_modes:
            data_url = mode_images[mode]
            key = f"candidate:{candidate.id}:{mode}"
            refs[mode] = key
            preview_assets[key] = {
                "kind": mode,
                "candidate_id": candidate.id,
                "region_ids": list(candidate.regions),
                "media_type": "image/png",
                "encoding": "data_url",
                "data_url": data_url,
            }
            if mode == "trimap":
                preview_assets[key]["execution_role"] = "pymatting_explicit_trimap"
                preview_assets[key]["states"] = {"sure_bg": 0, "unknown": 128, "sure_fg": 255}
                preview_assets[key]["metadata"] = trimap_metadata
        preview = {
            **candidate.preview,
            "assets": refs,
            "analysis_id": analysis_id,
        }
        updated.append(replace(candidate, preview=preview))
    return updated, preview_assets


def _enclosed_near_background_regions(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    bg_distance_max: float = 3.5,
    subject_distance_min: float = 18.0,
    anchor_dilate_px: int = 2,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.25,
    evidence_mode: str = "tight_enclosed_background_match",
) -> tuple[list[AmbiguityRegion], dict[str, np.ndarray]]:
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
    masks: dict[str, np.ndarray] = {}
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = labels == label_idx
        region_id = f"ambiguous_enclosed_bg_{len(regions)}"
        masks[region_id] = comp.copy()
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
                    "evidence_mode": evidence_mode,
                },
                ambiguity={
                    "transparent_hole_score": 0.5,
                    "subject_material_score": 0.5,
                    "reason": "single_image_semantic_ambiguity",
                },
            )
        )
    return regions, masks


def _screen_material_regions(
    image_srgb: np.ndarray,
    route: dict[str, Any],
    *,
    min_area_ratio: float = 0.003,
) -> tuple[list[AmbiguityRegion], dict[str, np.ndarray]]:
    """Find same-screen/translucent material risk for lightweight choices.

    This is intentionally route-analysis driven: CorridorKey has already
    measured a known green/blue screen and semantic profile. The mask is a
    cheap chromatic-distance support proxy so Analyze can preview ownership
    risk without pre-running alternate mattes.
    """

    analysis = route.get("analysis")
    if not isinstance(analysis, dict):
        return [], {}
    ck = analysis.get("corridorkey_analysis")
    if not isinstance(ck, dict):
        return [], {}
    if route.get("algorithm") != "corridorkey":
        return [], {}
    profile = str(route.get("parameter_profile") or "")
    execution_profile = str(route.get("execution_profile") or "")
    high_risk_profile = profile in {
        "key_color_material",
        "screen_tinted_translucency",
        "translucent_button",
    } or execution_profile in {
        "corridorkey-transparent-button",
        "corridorkey-effect-icon",
    }
    if not high_risk_profile:
        return [], {}
    bg = ck.get("background_color")
    if not (isinstance(bg, list) and len(bg) == 3):
        return [], {}
    background = tuple(int(np.clip(c, 0, 255)) for c in bg)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    distance = oklab_distance(lab, bg_lab).astype(np.float32)
    # These gates describe a semantic risk zone: pixels are clearly not plain
    # background, but still close enough to the screen that ownership can flip
    # between transparent material, glow, and same-key foreground.
    material = (distance >= 7.0) & (distance <= 32.0)
    material = cv2.morphologyEx(material.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    min_area = max(1, int(round(float(image_srgb.shape[0] * image_srgb.shape[1]) * float(min_area_ratio))))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(material.astype(np.uint8), connectivity=8)
    regions: list[AmbiguityRegion] = []
    masks: dict[str, np.ndarray] = {}
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label_idx
        region_id = f"ambiguous_screen_material_{len(regions)}"
        masks[region_id] = comp.copy()
        regions.append(
            AmbiguityRegion(
                id=region_id,
                type="screen_material_or_translucency",
                bbox_xyxy=_bbox(comp),
                area_px=area,
                mask_ref=region_id,
                evidence={
                    "background_color": [int(c) for c in background],
                    "parameter_profile": profile,
                    "execution_profile": execution_profile,
                    "distance_min": 7.0,
                    "distance_max": 32.0,
                },
                ambiguity={
                    "foreground_material_score": 0.5,
                    "transparent_material_score": 0.5,
                    "reason": "same_screen_or_translucent_material_risk",
                },
            )
        )
        break
    return regions, masks


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


def _screen_material_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
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
            confidence=0.55,
            risk_level="medium",
            decision={"policy": "auto_default"},
            regions=region_ids,
            preview=preview,
            reasons=["route detected screen-material or translucent ownership risk"],
        ),
        SemanticCandidate(
            id="preserve_screen_material",
            label="Keep screen-color material",
            intent="Treat same-screen or translucent pixels as subject-owned material/glow.",
            default=False,
            confidence=0.5,
            risk_level="medium",
            decision={"screen_material_policy": "preserve"},
            regions=region_ids,
            preview=preview,
            reasons=[
                "pixels are near the known screen color but not connected plain background",
                "transparent buttons and glow can be damaged by treating this band as background",
            ],
        ),
        SemanticCandidate(
            id="remove_screen_tint",
            label="Remove screen tint",
            intent="Treat same-screen or translucent pixels as removable screen contamination.",
            default=False,
            confidence=0.45,
            risk_level="medium",
            decision={"screen_material_policy": "background"},
            regions=region_ids,
            preview=preview,
            reasons=[
                "near-screen material may be spill or residual screen tint",
                "single-image evidence cannot prove user intent for this translucent band",
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
    region_masks: dict[str, np.ndarray] = {}
    background = _background_from_route(route)
    semantic_input = image_srgb
    effective_preprocess = preprocess
    if route.get("algorithm") == "pymatting_known_b" and background is not None:
        selected_preprocess = set(preprocess.selected) if preprocess is not None else set()
        if BACKGROUND_REPAIR in selected_preprocess:
            normalized, known_b_preprocess = repair_known_background_preprocess(
                image_srgb,
                background,
                **_known_b_thresholds_from_route(route),
            )
            effective_preprocess = _merge_preprocess_decisions(preprocess, known_b_preprocess)
            semantic_input = normalized
        regions, region_masks = _enclosed_near_background_regions(
            semantic_input,
            background,
            **_known_b_semantic_region_settings(route),
        )
    elif route.get("algorithm") == "corridorkey":
        regions, region_masks = _screen_material_regions(semantic_input, route)

    analysis_id = _analysis_id(
        semantic_input,
        route=route,
        preprocess=effective_preprocess,
        regions=regions,
    )
    regions = _with_mask_refs(regions, analysis_id)

    if regions:
        region_types = {region.type for region in regions}
        if "screen_material_or_translucency" in region_types:
            candidates = _screen_material_candidates(regions)
        else:
            candidates = _enclosed_near_bg_candidates(regions)
        candidates, preview_assets = _attach_preview_assets(
            semantic_input,
            route=route,
            route_algorithm=str(route.get("algorithm") or ""),
            analysis_id=analysis_id,
            regions=regions,
            region_masks=region_masks,
            candidates=candidates,
        )
        return AnalyzeResult(
            status="needs_decision",
            analysis_id=analysis_id,
            preprocess=effective_preprocess,
            route=route,
            ambiguity_regions=regions,
            candidates=candidates,
            default_candidate_id="auto_default",
            preview_assets=preview_assets,
        )
    candidates, preview_assets = _attach_preview_assets(
        semantic_input,
        route=route,
        route_algorithm=str(route.get("algorithm") or ""),
        analysis_id=analysis_id,
        regions=[],
        region_masks={},
        candidates=[_ready_candidate()],
    )
    return AnalyzeResult(
        status="ready",
        analysis_id=analysis_id,
        preprocess=effective_preprocess,
        route=route,
        candidates=candidates,
        default_candidate_id="auto_default",
        preview_assets=preview_assets,
    )


__all__ = ["analyze_candidates"]
