"""Lightweight Analyze stage for semantic candidate planning."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .corridorkey_hint import corridorkey_hint_strengths
from .keyer import KeyerThresholds, chromatic_key_alpha
from .pipeline_contracts import (
    AmbiguityRegion,
    AnalyzeResult,
    PreprocessDecision,
    SemanticCandidate,
)
from .router import build_route_candidates, select_default_route_candidate
from .types import Trimap


@dataclass(frozen=True)
class _KnownBTrimapPreviewBase:
    trimap: Trimap
    assembly_info: dict[str, Any]
    preprocess_info: dict[str, Any]
    kwargs: dict[str, Any]


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


def _route_candidate_payload(route_candidate: Any) -> dict[str, Any]:
    payload = route_candidate.to_dict()
    route = _route_payload(route_candidate)
    return {
        **route,
        "id": payload.get("id"),
        "default": bool(payload.get("default", False)),
        "route_candidate_id": payload.get("route_candidate_id") or payload.get("id"),
        "evidence": payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
        "risks": payload.get("risks") if isinstance(payload.get("risks"), list) else [],
    }


def _background_from_route(route: dict[str, Any]) -> tuple[int, int, int] | None:
    params = route.get("params")
    if not isinstance(params, dict):
        return None
    color = params.get("pymatting_bg_color")
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(int(np.clip(c, 0, 255)) for c in color)
    return None


def _corridorkey_background_from_route(route: dict[str, Any]) -> tuple[int, int, int] | None:
    analysis = route.get("analysis")
    if isinstance(analysis, dict):
        ck = analysis.get("corridorkey_analysis")
    else:
        ck = None
    if not isinstance(ck, dict):
        ck = route.get("corridorkey_analysis") if isinstance(route.get("corridorkey_analysis"), dict) else None
    if not isinstance(ck, dict):
        return None
    color = ck.get("background_color")
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(int(np.clip(c, 0, 255)) for c in color)
    return None


def _known_b_thresholds_from_route(route: dict[str, Any]) -> dict[str, Any]:
    params = route.get("params")
    params = params if isinstance(params, dict) else {}
    return {
        "bg_threshold": float(params.get("pymatting_bg_threshold", 3.5)),
        "fg_threshold": float(params.get("pymatting_fg_threshold", 24.0)),
        "adapt_bg_threshold": bool(params.get("pymatting_adapt_bg_threshold", False)),
        "adapt_fg_threshold": bool(params.get("pymatting_adapt_fg_threshold", True)),
        "adapt_boundary_band": bool(params.get("pymatting_adapt_boundary_band", True)),
    }


def _known_b_trimap_kwargs_from_route(route: dict[str, Any]) -> dict[str, Any]:
    params = route.get("params")
    params = params if isinstance(params, dict) else {}
    background = _background_from_route(route)
    return {
        "background_color": background,
        "bg_source": str(params.get("pymatting_bg_source") or "custom"),
        "bg_threshold": float(params.get("pymatting_bg_threshold", 3.5)),
        "fg_threshold": float(params.get("pymatting_fg_threshold", 24.0)),
        "boundary_band_px": int(params.get("pymatting_boundary_band_px", 2)),
        "adapt_bg_threshold": bool(params.get("pymatting_adapt_bg_threshold", False)),
        "adapt_fg_threshold": bool(params.get("pymatting_adapt_fg_threshold", True)),
        "adapt_boundary_band": bool(params.get("pymatting_adapt_boundary_band", True)),
        "trimap_mode": str(params.get("pymatting_trimap_mode") or "standard"),
        "unknown_grow_px": int(params.get("pymatting_unknown_grow_px", 0)),
    }


def _known_b_semantic_region_settings(route: dict[str, Any]) -> dict[str, Any]:
    profile = _route_semantic_parameter_profile(route)
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


def _route_semantic_parameter_profile(route: dict[str, Any]) -> str:
    """Return the measured semantic profile used for Analyze ambiguity logic."""

    analysis = route.get("analysis")
    if isinstance(analysis, dict):
        ck = analysis.get("corridorkey_analysis")
        if isinstance(ck, dict) and isinstance(ck.get("parameter_profile"), str):
            return str(ck.get("parameter_profile") or "")
    return str(route.get("parameter_profile") or "")


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


def _components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    seed_bool = np.asarray(seed, dtype=bool) & mask_bool
    if not bool(mask_bool.any()) or not bool(seed_bool.any()):
        return np.zeros_like(mask_bool, dtype=bool)
    labels_count, labels, _stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    touching = np.unique(labels[seed_bool])
    touching = touching[touching != 0]
    if touching.size == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    return np.isin(labels, touching)


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _dark_outline_mask(image_srgb: np.ndarray, subject_support: np.ndarray) -> np.ndarray:
    rgb = image_srgb.astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    max_channel = np.max(rgb, axis=2)
    # Cartoon/UI outlines are dark strokes, not simply saturated material such as
    # a red ring or orange fur. The max-channel gate keeps saturated body color
    # from being mistaken for an outline.
    return np.asarray(subject_support, dtype=bool) & (max_channel <= 145.0) & (luma <= 130.0)


def _outline_color_similarity(
    image_srgb: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> float:
    if int(first_mask.sum()) < 4 or int(second_mask.sum()) < 4:
        return 0.0
    first = np.median(image_srgb[first_mask].astype(np.float32), axis=0)
    second = np.median(image_srgb[second_mask].astype(np.float32), axis=0)
    mean_abs_delta = float(np.mean(np.abs(first - second)))
    return float(np.clip(1.0 - max(0.0, mean_abs_delta - 24.0) / 96.0, 0.0, 1.0))


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


def _semantic_hole_bg_core_and_unknown(
    mask: np.ndarray,
    *,
    sure_fg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split transparent-hole masks into sure-BG core plus a local solve ring."""

    mask_bool = np.asarray(mask, dtype=bool)
    shape = mask_bool.shape
    bg_core = np.zeros(shape, dtype=bool)
    unknown_release = np.zeros(shape, dtype=bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    components: list[dict[str, Any]] = []
    image_release_cap = max(1, min(4, int(round(float(min(shape)) * 0.004))))
    for label_idx in range(1, n_labels):
        comp = labels == label_idx
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        hole_release_cap = max(1, min(4, int(round(float(max(1, min(width, height))) * 0.20))))
        release_px = max(1, min(image_release_cap, hole_release_cap))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (release_px * 2 + 1, release_px * 2 + 1))
        eroded = cv2.erode(comp.astype(np.uint8), kernel, iterations=1).astype(bool)
        if bool(eroded.any()):
            comp_core = eroded
            inner_unknown = comp & ~eroded
        else:
            comp_core = comp
            inner_unknown = np.zeros(shape, dtype=bool)
        outer_unknown = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(bool) & ~comp & sure_fg
        release = inner_unknown | outer_unknown
        bg_core |= comp_core
        unknown_release |= release
        components.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(stats[label_idx, cv2.CC_STAT_LEFT]),
                    int(stats[label_idx, cv2.CC_STAT_TOP]),
                    int(stats[label_idx, cv2.CC_STAT_LEFT] + width),
                    int(stats[label_idx, cv2.CC_STAT_TOP] + height),
                ],
                "release_px": int(release_px),
                "bg_core_pixels": int(comp_core.sum()),
                "inner_unknown_pixels": int(inner_unknown.sum()),
                "outer_subject_unknown_pixels": int(outer_unknown.sum()),
            }
        )
    return bg_core, unknown_release, {
        "method": "adaptive_hole_edge_unknown_release",
        "image_release_cap_px": int(image_release_cap),
        "components": components[:24],
        "omitted_components": max(0, len(components) - 24),
        "bg_core_pixels": int(bg_core.sum()),
        "unknown_release_pixels": int(unknown_release.sum()),
    }


def _known_b_trimap_preview_base(
    image_srgb: np.ndarray,
    route: dict[str, Any],
) -> _KnownBTrimapPreviewBase | None:
    kwargs = _known_b_trimap_kwargs_from_route(route)
    background = kwargs.pop("background_color")
    if background is None:
        return None
    bg_source = str(kwargs.pop("bg_source", "custom"))
    try:
        from .api import prepare_known_b_preprocessed_input
        from .pymatting_refine import build_known_background_trimap

        trimap_image_srgb, trimap_background, _bg_info, preprocess_info = prepare_known_b_preprocessed_input(
            image_srgb,
            bg_source=bg_source,
            bg_color=background,
            bg_threshold=float(kwargs["bg_threshold"]),
            fg_threshold=float(kwargs["fg_threshold"]),
            adaptive=False,
        )
        trimap, assembly_info = build_known_background_trimap(
            trimap_image_srgb,
            trimap_background,
            bg_threshold=float(kwargs["bg_threshold"]),
            fg_threshold=float(kwargs["fg_threshold"]),
            boundary_band_px=int(kwargs["boundary_band_px"]),
            adapt_bg_threshold=bool(kwargs["adapt_bg_threshold"]),
            adapt_fg_threshold=bool(kwargs["adapt_fg_threshold"]),
            adapt_boundary_band=bool(kwargs["adapt_boundary_band"]),
            trimap_mode=str(kwargs.get("trimap_mode") or "standard"),
            unknown_grow_px=int(kwargs.get("unknown_grow_px") or 0),
        )
    except Exception:
        return None
    return _KnownBTrimapPreviewBase(
        trimap=trimap,
        assembly_info=assembly_info,
        preprocess_info=preprocess_info,
        kwargs=kwargs,
    )


def _known_b_trimap_preview(
    image_srgb: np.ndarray,
    route: dict[str, Any],
    candidate: SemanticCandidate,
    *,
    candidate_mask: np.ndarray | None = None,
    region_masks: dict[str, np.ndarray] | None = None,
    regions: list[AmbiguityRegion] | None = None,
    base: _KnownBTrimapPreviewBase | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Build the explicit trimap that Analyze will hand to Execute.

    The current Known-B contract is deliberately simple: normalize the measured
    background field, let the BG-seed outline builder decide BG/unknown/FG, then
    apply only semantic hole decisions as an overlay. Shadow evidence is handled
    inside the outline builder as boundary expansion; it is not a separate
    candidate branch here.
    """

    decision = candidate.decision or {}
    region_masks = region_masks or {}
    hole_policies = decision.get("enclosed_near_bg_region_policies")
    shadow_policies = decision.get("known_b_shadow_region_policies")
    summary_policy = str(decision.get("enclosed_near_bg_policy") or "")
    can_reuse_base = (
        isinstance(hole_policies, dict)
        or isinstance(shadow_policies, dict)
        or summary_policy in {"", "auto_default"}
    )
    if base is None or not can_reuse_base:
        try:
            from .api import prepare_known_b_preprocessed_input
            from .pymatting_refine import build_known_background_trimap

            kwargs = _known_b_trimap_kwargs_from_route(route)
            background = kwargs.pop("background_color")
            if background is None:
                return None
            bg_source = str(kwargs.pop("bg_source", "custom"))
            trimap_image_srgb, trimap_background, _bg_info, preprocess_info = prepare_known_b_preprocessed_input(
                image_srgb,
                bg_source=bg_source,
                bg_color=background,
                bg_threshold=float(kwargs["bg_threshold"]),
                fg_threshold=float(kwargs["fg_threshold"]),
                adaptive=False,
            )
            trimap, assembly_info = build_known_background_trimap(
                trimap_image_srgb,
                trimap_background,
                bg_threshold=float(kwargs["bg_threshold"]),
                fg_threshold=float(kwargs["fg_threshold"]),
                boundary_band_px=int(kwargs["boundary_band_px"]),
                adapt_bg_threshold=bool(kwargs["adapt_bg_threshold"]),
                adapt_fg_threshold=bool(kwargs["adapt_fg_threshold"]),
                adapt_boundary_band=bool(kwargs["adapt_boundary_band"]),
                trimap_mode=str(kwargs.get("trimap_mode") or "standard"),
                unknown_grow_px=int(kwargs.get("unknown_grow_px") or 0),
                semantic_decision=dict(decision),
            )
        except Exception:
            return None
        base_reused = False
    else:
        trimap = base.trimap
        assembly_info = base.assembly_info
        preprocess_info = base.preprocess_info
        kwargs = base.kwargs
        base_reused = True
    region_by_id = {region.id: region for region in (regions or [])}
    semantic_forced_bg = np.zeros(trimap.sure_bg.shape, dtype=bool)
    semantic_forced_fg = np.zeros(trimap.sure_fg.shape, dtype=bool)
    semantic_hole_unknown = np.zeros(trimap.unknown.shape, dtype=bool)
    semantic_shadow_unknown = np.zeros(trimap.unknown.shape, dtype=bool)
    semantic_forced_fg_exterior_unknown_overlap = np.zeros(trimap.unknown.shape, dtype=bool)
    semantic_hole_unknown_info: dict[str, Any] = {"method": "not_applied", "unknown_release_pixels": 0}
    flat_opaque_internal_unknown = np.zeros(trimap.unknown.shape, dtype=bool)
    flat_opaque_release_enabled = False
    flat_opaque_info: dict[str, Any] = {
        "applied": False,
        "released_pixels": 0,
        "reason": "not_applicable",
    }
    if isinstance(hole_policies, dict):
        for region_id, policy_value in hole_policies.items():
            mask = region_masks.get(str(region_id))
            if mask is None or mask.shape != trimap.sure_bg.shape:
                continue
            if str(policy_value) == "transparent_hole":
                semantic_forced_bg |= np.asarray(mask, dtype=bool)
            elif str(policy_value) == "subject":
                semantic_forced_fg |= np.asarray(mask, dtype=bool)
    else:
        summary_policy = str(decision.get("enclosed_near_bg_policy") or "")
        if summary_policy == "transparent_hole" and candidate_mask is not None:
            semantic_forced_bg |= np.asarray(candidate_mask, dtype=bool)
        elif summary_policy == "subject" and candidate_mask is not None:
            semantic_forced_fg |= np.asarray(candidate_mask, dtype=bool)

    if isinstance(shadow_policies, dict):
        for region_id, policy_value in shadow_policies.items():
            mask = region_masks.get(str(region_id))
            if mask is None or mask.shape != trimap.sure_bg.shape:
                continue
            if str(policy_value) in {"shadow_unknown", "shadow"}:
                semantic_shadow_unknown |= np.asarray(mask, dtype=bool)
    elif str(decision.get("known_b_shadow_policy") or "") in {"shadow_unknown", "shadow"} and candidate_mask is not None:
        semantic_shadow_unknown |= np.asarray(candidate_mask, dtype=bool)

    if bool(semantic_forced_bg.any()):
        semantic_forced_bg, semantic_hole_unknown, semantic_hole_unknown_info = _semantic_hole_bg_core_and_unknown(
            semantic_forced_bg,
            sure_fg=trimap.sure_fg,
        )
    if isinstance(hole_policies, dict) and any(str(value) == "subject" for value in hole_policies.values()):
        subject_regions = [
            region_by_id.get(str(region_id))
            for region_id, value in hole_policies.items()
            if str(value) == "subject"
        ]
        subject_regions = [region for region in subject_regions if region is not None]
        max_subject_outline = max(
            (float(region.evidence.get("subject_outline_confidence") or 0.0) for region in subject_regions),
            default=0.0,
        )
        max_hole_outline = max(
            (float(region.evidence.get("hole_outline_confidence") or 0.0) for region in subject_regions),
            default=0.0,
        )
        if max_subject_outline >= 0.45 and max_hole_outline < 0.65:
            flat_opaque_release_enabled = True
            flat_opaque_info = {
                "applied": False,
                "released_pixels": 0,
                "reason": "pending post-semantic topology pass",
                "subject_outline_confidence": float(max_subject_outline),
                "hole_outline_confidence": float(max_hole_outline),
                "policy": "topological_subject_internal_unknown_to_sure_fg",
            }
        else:
            flat_opaque_info = {
                "applied": False,
                "released_pixels": 0,
                "reason": "missing flat subject outline or matching hole outline is present",
                "subject_outline_confidence": float(max_subject_outline),
                "hole_outline_confidence": float(max_hole_outline),
            }

    if bool(
        semantic_forced_bg.any()
        or semantic_forced_fg.any()
        or semantic_hole_unknown.any()
        or semantic_shadow_unknown.any()
        or flat_opaque_release_enabled
    ):
        sure_fg = trimap.sure_fg.copy()
        sure_bg = trimap.sure_bg.copy()
        unknown = trimap.unknown.copy()
        if bool(semantic_forced_fg.any()):
            sure_fg[semantic_forced_fg] = True
            sure_bg[semantic_forced_fg] = False
            unknown[semantic_forced_fg] = False
        if bool(semantic_forced_bg.any()):
            sure_bg[semantic_forced_bg] = True
            sure_fg[semantic_forced_bg] = False
            unknown[semantic_forced_bg] = False
        if bool(flat_opaque_release_enabled):
            # Run after subject/FG overlays: semantic opaque regions can cut
            # thin connections that made interior marks look exterior-connected
            # in the raw builder trimap. Transparent-hole and shadow unknown
            # overlays are applied afterward so they can still restore their
            # own explicit solve regions.
            exterior_bg = _exterior_mask(sure_bg)
            unknown_seed = unknown & cv2.dilate(
                exterior_bg.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=1,
            ).astype(bool)
            unknown_seed[0, :] |= unknown[0, :]
            unknown_seed[-1, :] |= unknown[-1, :]
            unknown_seed[:, 0] |= unknown[:, 0]
            unknown_seed[:, -1] |= unknown[:, -1]
            exterior_connected_unknown = _components_touching_seed(unknown, unknown_seed)
            flat_opaque_internal_unknown = unknown & ~exterior_connected_unknown
            semantic_forced_fg_exterior_unknown_overlap = semantic_forced_fg & exterior_connected_unknown
            if bool(flat_opaque_internal_unknown.any()):
                sure_fg[flat_opaque_internal_unknown] = True
                sure_bg[flat_opaque_internal_unknown] = False
                unknown[flat_opaque_internal_unknown] = False
            flat_opaque_info.update(
                {
                    "applied": bool(flat_opaque_internal_unknown.any()),
                    "released_pixels": int(flat_opaque_internal_unknown.sum()),
                    "reason": ""
                    if bool(flat_opaque_internal_unknown.any())
                    else "no post-semantic topologically internal unknown components",
                    "exterior_connected_unknown_pixels": int(exterior_connected_unknown.sum()),
                    "internal_unknown_pixels": int(flat_opaque_internal_unknown.sum()),
                    "unknown_seed_pixels": int(unknown_seed.sum()),
                    "semantic_forced_fg_exterior_unknown_overlap_pixels": int(
                        semantic_forced_fg_exterior_unknown_overlap.sum()
                    ),
                    "topology_stage": "after_subject_fg_and_bg_overlays",
                }
            )
        if bool(semantic_hole_unknown.any()):
            unknown[semantic_hole_unknown] = True
            sure_bg[semantic_hole_unknown] = False
            sure_fg[semantic_hole_unknown] = False
        if bool(semantic_shadow_unknown.any()):
            unknown[semantic_shadow_unknown] = True
            sure_bg[semantic_shadow_unknown] = False
            sure_fg[semantic_shadow_unknown] = False
        trimap = Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)

    preview = _trimap_preview_from_masks(trimap.sure_bg, trimap.unknown, trimap.sure_fg)
    metadata = {
        "states": {
            "sure_bg": {"value": 0, "color": [0, 0, 0], "pixels": int(trimap.sure_bg.sum())},
            "unknown": {"value": 128, "color": [128, 128, 128], "pixels": int(trimap.unknown.sum())},
            "sure_fg": {"value": 255, "color": [255, 255, 255], "pixels": int(trimap.sure_fg.sum())},
        },
        "source": "known_b_bg_seed_outline_candidate_trimap",
        "method": assembly_info.get("method"),
        "semantic_decision": dict(decision),
        "pymatting_trimap_mode": kwargs.get("trimap_mode"),
        "pymatting_unknown_grow_px": kwargs.get("unknown_grow_px"),
        "enclosed_near_bg_region_policies": hole_policies if isinstance(hole_policies, dict) else None,
        "known_b_shadow_region_policies": shadow_policies if isinstance(shadow_policies, dict) else None,
        "region_policy_application": "bg_seed_outline_region_overlay_applied"
        if isinstance(hole_policies, dict) or isinstance(shadow_policies, dict)
        else None,
        "semantic_forced_bg_pixels": int(semantic_forced_bg.sum()),
        "semantic_forced_fg_pixels": int(semantic_forced_fg.sum()),
        "semantic_hole_unknown_pixels": int(semantic_hole_unknown.sum()),
        "semantic_shadow_unknown_pixels": int(semantic_shadow_unknown.sum()),
        "semantic_forced_fg_exterior_unknown_overlap_pixels": int(
            semantic_forced_fg_exterior_unknown_overlap.sum()
        ),
        "semantic_hole_unknown": semantic_hole_unknown_info,
        "flat_opaque_internal_unknown": flat_opaque_info,
        "preprocess": preprocess_info,
        "candidate_assembly": assembly_info,
        "base_trimap_reused": bool(base_reused),
    }
    return preview, metadata


def _hint_preview(mask: np.ndarray, shape: tuple[int, int], *, keep: bool) -> np.ndarray:
    hint = np.full((*shape, 4), (23, 28, 26, 255), dtype=np.uint8)
    if bool(mask.any()):
        hint[mask] = (0, 190, 255, 190) if keep else (255, 86, 72, 190)
    return hint


def _grayscale_hint_preview(hint: np.ndarray) -> np.ndarray:
    gray = np.clip(hint.astype(np.float32), 0.0, 1.0)
    u8 = (gray * 255.0 + 0.5).astype(np.uint8)
    return np.dstack([u8, u8, u8, np.full(u8.shape, 255, dtype=np.uint8)])


def _attach_preview_assets(
    image_srgb: np.ndarray,
    *,
    route: dict[str, Any],
    route_algorithm: str,
    analysis_id: str,
    regions: list[AmbiguityRegion],
    region_masks: dict[str, np.ndarray],
    candidates: list[SemanticCandidate],
    eager_trimap: bool = True,
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

    if route_algorithm == "pymatting_known_b" and eager_trimap:
        preview_modes = ("overlay", "trimap")
    elif route_algorithm == "pymatting_known_b":
        preview_modes = ("overlay",)
    elif route_algorithm == "corridorkey":
        preview_modes = ("overlay", "hint")
    else:
        preview_modes = ("overlay",)

    updated: list[SemanticCandidate] = []
    known_b_base = (
        _known_b_trimap_preview_base(image_srgb, route)
        if route_algorithm == "pymatting_known_b" and "trimap" in preview_modes
        else None
    )
    for candidate in candidates:
        decision = candidate.decision or {}
        policy = str(
            decision.get("enclosed_near_bg_policy")
            or decision.get("screen_material_policy")
            or decision.get("button_body_policy")
            or decision.get("known_b_shadow_policy")
            or ""
        )
        cut = policy in {"transparent_hole", "background", "remove", "shadow_unknown", "shadow"}
        keep = policy in {"subject", "foreground", "keep", "preserve", "opaque_subject"} or not cut
        color = (0, 190, 255, 96) if keep else (255, 86, 72, 112)
        mask = _candidate_mask(candidate, region_masks, (h, w))
        refs: dict[str, str] = {}
        trimap_preview = (
            _known_b_trimap_preview(
                image_srgb,
                route,
                candidate,
                candidate_mask=mask,
                region_masks=region_masks,
                regions=regions,
                base=known_b_base,
            )
            if route_algorithm == "pymatting_known_b" and "trimap" in preview_modes
            else None
        )
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
        mode_images: dict[str, str] = {}
        if "overlay" in preview_modes:
            mode_images["overlay"] = _png_data_url_rgba(_overlay_preview(image_srgb, mask, color))
        if "trimap" in preview_modes:
            mode_images["trimap"] = _png_data_url_rgb(trimap_image)
        if "hint" in preview_modes:
            mode_images["hint"] = _png_data_url_rgba(_hint_preview(mask, (h, w), keep=keep))
        hint_metadata: dict[str, Any] | None = None
        if route_algorithm == "corridorkey":
            hint_value = decision.get("corridorkey_hint_value")
            if isinstance(hint_value, (int, float)):
                value = float(np.clip(float(hint_value), 0.0, 1.0))
                hint = np.full((h, w), value, dtype=np.float32)
                mode_images["hint"] = _png_data_url_rgba(_grayscale_hint_preview(hint))
                hint_metadata = {
                    "schema": "ermbg.corridorkey_constant_hint.v1",
                    "source": "semantic_corridorkey_hint_value",
                    "kind": "full_frame_constant",
                    "value": value,
                    "min": value,
                    "max": value,
                    "mean": value,
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
            if mode == "hint" and hint_metadata is not None:
                preview_assets[key]["execution_role"] = "corridorkey_hint_mask"
                preview_assets[key]["metadata"] = hint_metadata
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
    tight_min_area_ratio: float = 0.00002,
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
    outline_support = _dark_outline_mask(image_srgb, subject_support)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * int(anchor_dilate_px) + 1, 2 * int(anchor_dilate_px) + 1),
    )
    outline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    exterior_subject_ring = cv2.dilate(exterior.astype(np.uint8), outline_kernel, iterations=1).astype(bool)
    exterior_subject_ring &= subject_support
    exterior_outline = exterior_subject_ring & outline_support
    exterior_ring_pixels = int(exterior_subject_ring.sum())
    exterior_outline_pixels = int(exterior_outline.sum())
    exterior_outline_fraction = float(exterior_outline_pixels) / max(float(exterior_ring_pixels), 1.0)
    subject_outline_confidence = float(np.clip((exterior_outline_fraction - 0.08) / 0.42, 0.0, 1.0))
    near_subject = cv2.dilate(subject_support.astype(np.uint8), kernel, iterations=1).astype(bool)
    support = enclosed & near_subject

    image_area = float(h * w)
    loose_min_area_ratio = float(min_area_ratio)
    tight_area_ratio = min(float(tight_min_area_ratio), loose_min_area_ratio)
    max_area = max(1, int(round(image_area * float(max_area_ratio))))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), connectivity=8)
    regions: list[AmbiguityRegion] = []
    masks: dict[str, np.ndarray] = {}
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > max_area:
            continue
        comp = labels == label_idx
        comp_distance = distance[comp].astype(np.float32)
        distance_p50 = float(np.percentile(comp_distance, 50.0)) if comp_distance.size else 0.0
        distance_p95 = float(np.percentile(comp_distance, 95.0)) if comp_distance.size else 0.0
        # Use the component core rather than the edge tail for area confidence:
        # antialiasing around real holes can make p95 look loose even when the
        # enclosed island has a clean known-B center.
        distance_fraction = float(np.clip(distance_p50 / max(float(bg_distance_max), 1e-6), 0.0, 1.0))
        effective_min_area_ratio = tight_area_ratio + (loose_min_area_ratio - tight_area_ratio) * (
            distance_fraction * distance_fraction
        )
        effective_min_area = max(1, int(round(image_area * effective_min_area_ratio)))
        if area < effective_min_area:
            continue
        support_pixels = int((comp & support).sum())
        support_fraction = float(support_pixels) / max(float(area), 1.0)
        # The semantic unit is the whole enclosed same-B component. Nearby
        # subject support is only evidence that the component is an internal
        # ownership ambiguity, not the mask to preserve/cut. Using only the
        # support ring makes keep-hole candidates preserve a thin outline while
        # the actual same-B island remains governed by the generic trimap.
        min_support_pixels = max(4, min(24, int(round(float(area) * 0.03))))
        if support_pixels < min_support_pixels:
            continue
        outer_ring = cv2.dilate(comp.astype(np.uint8), outline_kernel, iterations=1).astype(bool) & ~comp
        inner_edge = comp & ~cv2.erode(comp.astype(np.uint8), edge_kernel, iterations=1).astype(bool)
        hole_outline = outer_ring & outline_support
        hole_outline_pixels = int(hole_outline.sum())
        hole_ring_pixels = int(outer_ring.sum())
        hole_edge_pixels = int(inner_edge.sum())
        hole_outline_fraction = float(hole_outline_pixels) / max(float(hole_ring_pixels), 1.0)
        hole_outline_near = cv2.dilate(hole_outline.astype(np.uint8), edge_kernel, iterations=1).astype(bool)
        hole_outline_continuity = float((inner_edge & hole_outline_near).sum()) / max(float(hole_edge_pixels), 1.0)
        outline_similarity = _outline_color_similarity(image_srgb, exterior_outline, hole_outline)
        hole_outline_confidence = float(
            np.clip(
                min(hole_outline_fraction / 0.35, hole_outline_continuity / 0.60) * outline_similarity,
                0.0,
                1.0,
            )
        )
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
                    "subject_support_pixels": int(support_pixels),
                    "subject_support_fraction": float(support_fraction),
                    "subject_outline_pixels": int(exterior_outline_pixels),
                    "subject_outline_fraction": float(exterior_outline_fraction),
                    "subject_outline_confidence": float(subject_outline_confidence),
                    "hole_outline_pixels": int(hole_outline_pixels),
                    "hole_outline_fraction": float(hole_outline_fraction),
                    "hole_outline_continuity": float(hole_outline_continuity),
                    "hole_outline_color_similarity": float(outline_similarity),
                    "hole_outline_confidence": float(hole_outline_confidence),
                    "semantic_mask_scope": "full_enclosed_component",
                    "bg_distance_max": float(bg_distance_max),
                    "bg_distance_p50": distance_p50,
                    "bg_distance_p95": distance_p95,
                    "subject_distance_min": float(subject_distance_min),
                    "anchor_dilate_px": int(anchor_dilate_px),
                    "min_area_ratio_loose": float(loose_min_area_ratio),
                    "min_area_ratio_tight": float(tight_area_ratio),
                    "min_area_ratio_effective": float(effective_min_area_ratio),
                    "min_area_px_effective": int(effective_min_area),
                    "area_gate_source": "background_distance_confidence",
                    "area_gate_color_stat": "bg_distance_p50",
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
    profile = _route_semantic_parameter_profile(route)
    execution_profile = str(route.get("execution_profile") or "")
    high_risk_profile = profile in {
        "key_color_material",
        "screen_tinted_translucency",
        "translucent_button",
    } or execution_profile in {
        "corridorkey-character",
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


def _known_b_connected_shadow_regions(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    route: dict[str, Any],
) -> tuple[list[AmbiguityRegion], dict[str, np.ndarray]]:
    """Expose connected scalar-shadow evidence that currently overlaps FG.

    This detector only creates a semantic ambiguity region. The default
    candidate keeps the current route behavior; a separate candidate can release
    the disputed connected shadow field into trimap unknown.
    """

    bg = np.asarray(background_color, dtype=np.uint8)
    bg_float = bg.astype(np.float32)
    if float(bg_float.mean()) < 180.0 or float(bg_float.max() - bg_float.min()) > 28.0:
        return [], {}

    try:
        from .pymatting_refine import (
            _known_background_shadow_like_background_mask,
            build_known_background_trimap,
        )

        kwargs = _known_b_trimap_kwargs_from_route(route)
        kwargs.pop("background_color", None)
        trimap, trimap_info = build_known_background_trimap(
            image_srgb,
            background_color,
            bg_threshold=float(kwargs["bg_threshold"]),
            fg_threshold=float(kwargs["fg_threshold"]),
            boundary_band_px=int(kwargs["boundary_band_px"]),
            adapt_bg_threshold=bool(kwargs["adapt_bg_threshold"]),
            adapt_fg_threshold=bool(kwargs["adapt_fg_threshold"]),
            adapt_boundary_band=bool(kwargs["adapt_boundary_band"]),
            trimap_mode=str(kwargs.get("trimap_mode") or "standard"),
            unknown_grow_px=int(kwargs.get("unknown_grow_px") or 0),
        )
        shadow_mask, shadow_info = _known_background_shadow_like_background_mask(
            image_srgb,
            bg,
            subject_seed=trimap.sure_fg,
        )
    except Exception:
        return [], {}

    shadow_mask = np.asarray(shadow_mask, dtype=bool)
    conflict = shadow_mask & np.asarray(trimap.sure_fg, dtype=bool)
    if not bool(conflict.any()):
        return [], {}

    h, w = conflict.shape
    image_area = float(max(1, h * w))
    min_conflict_area = int(max(16, round(image_area * 0.0008)))
    min_component_area = int(max(96, round(image_area * 0.004)))
    exterior_bg = _exterior_mask(trimap.sure_bg)
    material_core = np.asarray(trimap.sure_fg, dtype=bool) & ~shadow_mask
    touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(shadow_mask.astype(np.uint8), 8)
    regions: list[AmbiguityRegion] = []
    masks: dict[str, np.ndarray] = {}
    for label in range(1, labels_count):
        comp = labels == label
        comp_area = int(stats[label, cv2.CC_STAT_AREA])
        if comp_area < min_component_area:
            continue
        comp_conflict = comp & conflict
        conflict_area = int(comp_conflict.sum())
        if conflict_area < min_conflict_area:
            continue
        comp_near = cv2.dilate(comp.astype(np.uint8), touch_kernel, iterations=1).astype(bool)
        unknown_contact = int((comp_near & trimap.unknown).sum())
        exterior_contact = int((comp_near & exterior_bg).sum())
        material_contact = int((comp_near & material_core).sum())
        material_core_pixels = int(material_core.sum())
        if unknown_contact <= 0 and exterior_contact <= 0:
            continue
        if material_contact < max(24, int(round(conflict_area * 0.10))):
            continue
        if material_core_pixels < max(64, int(round(conflict_area * 1.5))):
            continue

        region_id = f"ambiguous_known_b_shadow_{len(regions)}"
        review_mask = comp & ~material_core
        if not bool(review_mask.any()):
            continue
        masks[region_id] = review_mask.copy()
        regions.append(
            AmbiguityRegion(
                id=region_id,
                type="connected_known_b_shadow_ownership",
                bbox_xyxy=[
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                area_px=int(review_mask.sum()),
                mask_ref=region_id,
                evidence={
                    "background_color": [int(c) for c in background_color],
                    "background_profile": "bright_neutral_known_b",
                    "component_area_px": comp_area,
                    "sure_fg_conflict_pixels": conflict_area,
                    "unknown_contact_pixels": unknown_contact,
                    "exterior_contact_pixels": exterior_contact,
                    "material_contact_pixels": material_contact,
                    "material_core_pixels": material_core_pixels,
                    "shadow_detector": shadow_info,
                    "trimap_method": trimap_info.get("method"),
                    "bg_seed_outline": trimap_info.get("bg_seed_outline"),
                    "min_conflict_area_px": int(min_conflict_area),
                    "min_component_area_px": int(min_component_area),
                },
                ambiguity={
                    "kind": "known_b_connected_shadow_vs_subject",
                    "reason": "scalar known-background darkening is connected to the exterior solve band but currently overlaps sure-FG",
                    "default_policy": "current_route_default",
                    "alternative_policy": "shadow_unknown",
                },
            )
        )
    return regions, masks


def _corridorkey_alpha_structure_regions(
    image_srgb: np.ndarray,
    route: dict[str, Any],
    *,
    min_area_ratio: float = 0.001,
) -> tuple[list[AmbiguityRegion], dict[str, np.ndarray]]:
    analysis = route.get("analysis")
    if not isinstance(analysis, dict) or route.get("algorithm") != "corridorkey":
        return [], {}
    ck = analysis.get("corridorkey_analysis")
    if not isinstance(ck, dict):
        return [], {}
    execution_profile = str(route.get("execution_profile") or "")
    if execution_profile not in {
        "corridorkey-character",
        "corridorkey-transparent-button",
        "corridorkey-effect-icon",
        "corridorkey-shaped-icon",
    }:
        return [], {}
    bg = ck.get("background_color")
    if not (isinstance(bg, list) and len(bg) == 3):
        return [], {}
    background = tuple(int(np.clip(c, 0, 255)) for c in bg)
    key = chromatic_key_alpha(image_srgb, background, KeyerThresholds(bg_max=5.5, fg_min=18.0))
    support = key >= 0.16
    if not bool(support.any()):
        return [], {}
    distance_to_exterior = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    core = (key >= 0.20) & (key <= 0.85) & (distance_to_exterior >= 10.0)
    gradient = (key >= 0.03) & (key <= 0.55) & (distance_to_exterior <= 14.0)
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    gradient = cv2.morphologyEx(gradient.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    min_area = max(16, int(round(float(image_srgb.shape[0] * image_srgb.shape[1]) * float(min_area_ratio))))
    region_specs = [
        (
            "ambiguous_glass_core_transparency",
            "glass_core_transparency",
            core,
            {
                "alpha_min": 0.20,
                "alpha_max": 0.85,
                "distance_to_exterior_min": 10.0,
            },
            {
                "transparent_glass_score": 0.5,
                "solid_subject_score": 0.5,
                "reason": "central_mid_alpha_glass_or_energy_core",
            },
        ),
        (
            "ambiguous_soft_alpha_gradient",
            "soft_alpha_gradient",
            gradient,
            {
                "alpha_min": 0.03,
                "alpha_max": 0.55,
                "distance_to_exterior_max": 14.0,
            },
            {
                "preserve_gradient_score": 0.55,
                "remove_residue_score": 0.45,
                "reason": "outer_soft_alpha_gradient_or_screen_residue",
            },
        ),
    ]
    regions: list[AmbiguityRegion] = []
    masks: dict[str, np.ndarray] = {}
    for prefix, region_type, mask, evidence, ambiguity in region_specs:
        if int(mask.sum()) < min_area:
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        kept = np.zeros(mask.shape, dtype=bool)
        for label_idx in range(1, n_labels):
            if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_area:
                kept |= labels == label_idx
        if not bool(kept.any()):
            continue
        region_id = f"{prefix}_{len(regions)}"
        masks[region_id] = kept
        regions.append(
            AmbiguityRegion(
                id=region_id,
                type=region_type,
                bbox_xyxy=_bbox(kept),
                area_px=int(kept.sum()),
                mask_ref=region_id,
                evidence={
                    "background_color": [int(c) for c in background],
                    "execution_profile": execution_profile,
                    **evidence,
                },
                ambiguity=ambiguity,
            )
        )
    return regions, masks


def _known_b_button_body_regions(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    route: dict[str, Any],
    *,
    shadow_anchor_masks: dict[str, np.ndarray] | None = None,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.55,
) -> tuple[list[AmbiguityRegion], dict[str, np.ndarray]]:
    """Find same-key opaque body ownership ambiguity.

    This is not the ordinary hard-button shadow path. Standard Known-B should
    solve normal button shadows by shaping the shadow-facing unknown band. The
    opaque-body recipe is a same-key special case: the subject color is close
    enough to the known background that standard body evidence may fail to
    measure the opaque material domain at all. Only that route/profile may add
    an opaque-body candidate dimension.
    """
    if route.get("asset_kind") != "button":
        return [], {}
    profile = _route_semantic_parameter_profile(route)
    if profile != "opaque_hard_ui_same_key_plateau":
        return [], {}

    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    try:
        from .pymatting_refine import build_known_background_trimap

        standard, standard_info = build_known_background_trimap(
            image_srgb,
            background_color,
            bg_threshold=float(params.get("pymatting_bg_threshold", 3.5)),
            fg_threshold=float(params.get("pymatting_fg_threshold", 24.0)),
            boundary_band_px=int(params.get("pymatting_boundary_band_px", 2)),
            adapt_bg_threshold=bool(params.get("pymatting_adapt_bg_threshold", False)),
            adapt_fg_threshold=bool(params.get("pymatting_adapt_fg_threshold", True)),
            adapt_boundary_band=bool(params.get("pymatting_adapt_boundary_band", True)),
            trimap_mode="standard",
            unknown_grow_px=0,
        )
        body, body_info = build_known_background_trimap(
            image_srgb,
            background_color,
            bg_threshold=float(params.get("pymatting_bg_threshold", 3.5)),
            fg_threshold=float(params.get("pymatting_fg_threshold", 24.0)),
            boundary_band_px=int(params.get("pymatting_boundary_band_px", 2)),
            adapt_bg_threshold=bool(params.get("pymatting_adapt_bg_threshold", False)),
            adapt_fg_threshold=bool(params.get("pymatting_adapt_fg_threshold", True)),
            adapt_boundary_band=bool(params.get("pymatting_adapt_boundary_band", True)),
            trimap_mode="same_key_opaque_body_outline",
            unknown_grow_px=2,
        )
    except Exception:
        return [], {}

    invaded_subject = standard.unknown & body.sure_fg
    h, w = invaded_subject.shape
    min_area = max(24, int(round(float(h * w) * min_area_ratio)))
    max_area = max(min_area, int(round(float(h * w) * max_area_ratio)))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(invaded_subject.astype(np.uint8), connectivity=8)

    components: list[tuple[int, int]] = []
    for label_idx in range(1, labels_count):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            components.append((area, label_idx))
    components.sort(reverse=True)

    if not components:
        return [], {}

    shadow_anchor = np.zeros((h, w), dtype=bool)
    if shadow_anchor_masks:
        for mask in shadow_anchor_masks.values():
            if mask.shape == shadow_anchor.shape:
                shadow_anchor |= np.asarray(mask, dtype=bool)
    if not bool(shadow_anchor.any()) or not bool(body.sure_fg.any()):
        return [], {}

    body_dist = cv2.distanceTransform(body.sure_fg.astype(np.uint8), cv2.DIST_L2, 3)
    shadow_labels_count, shadow_labels, shadow_stats, _ = cv2.connectedComponentsWithStats(
        shadow_anchor.astype(np.uint8),
        connectivity=8,
    )
    largest_shadow_area = max(
        (int(shadow_stats[label_idx, cv2.CC_STAT_AREA]) for label_idx in range(1, shadow_labels_count)),
        default=0,
    )
    min_body_anchor_area = max(24, int(round(float(largest_shadow_area) * 0.25)))
    body_evidence = np.zeros((h, w), dtype=bool)
    body_components: list[dict[str, Any]] = []
    omitted_shadow_components = 0
    image_short = float(max(1, min(h, w)))
    for label_idx in range(1, shadow_labels_count):
        comp = shadow_labels == label_idx
        area = int(shadow_stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        # Body evidence release should be driven by the dominant shadow-facing
        # boundary. Tiny dark edge fragments are ignored as anchors so they do
        # not cause unrelated subject-side exposure.
        if area < min_body_anchor_area:
            omitted_shadow_components += 1
            continue
        comp_w = int(shadow_stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(shadow_stats[label_idx, cv2.CC_STAT_HEIGHT])
        comp_short = float(max(1, min(comp_w, comp_h)))
        comp_long = float(max(1, max(comp_w, comp_h)))
        # The exposure width is tied to the measured shadow component and image
        # size. It should be a shallow subject-side strip along the shadow
        # boundary: wide enough for the unknown band to contain real subject
        # color, but not so wide that a small button turns into scattered
        # interior unknowns unrelated to the shadow solve.
        expose_px = float(max(1.0, min(6.0, image_short * 0.055, max(2.0, comp_short * 0.35))))
        search_px = float(max(expose_px + 2.0, min(12.0, image_short * 0.10, max(expose_px + 2.0, comp_long * 0.45))))
        radius = max(1, int(round(search_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        near_shadow = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(bool)
        local_body = body.sure_fg & near_shadow & (body_dist <= expose_px)
        if bool(local_body.any()):
            body_evidence |= local_body
        body_components.append(
            {
                "shadow_area_px": area,
                "shadow_bbox_xyxy": [
                    int(shadow_stats[label_idx, cv2.CC_STAT_LEFT]),
                    int(shadow_stats[label_idx, cv2.CC_STAT_TOP]),
                    int(shadow_stats[label_idx, cv2.CC_STAT_LEFT] + shadow_stats[label_idx, cv2.CC_STAT_WIDTH]),
                    int(shadow_stats[label_idx, cv2.CC_STAT_TOP] + shadow_stats[label_idx, cv2.CC_STAT_HEIGHT]),
                ],
                "expose_px": float(expose_px),
                "search_px": float(search_px),
                "released_body_pixels": int(local_body.sum()),
            }
        )

    if not bool(body_evidence.any()):
        return [], {}
    # Close tiny raster breaks along the shadow-facing strip, while keeping the
    # mask anchored to body.sure_fg so it cannot spill into the shadow band.
    body_evidence = cv2.morphologyEx(body_evidence.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    body_evidence &= body.sure_fg

    body_outline = body_info.get("same_key_opaque_body_outline")
    if not isinstance(body_outline, dict):
        body_outline = {}

    region_id = "ambiguous_button_body_shadow_facing_0"
    largest_invaded = int(components[0][0]) if components else 0
    region = AmbiguityRegion(
        id=region_id,
        type="button_body_subject_ownership",
        bbox_xyxy=_bbox(body_evidence),
        area_px=int(body_evidence.sum()),
        mask_ref=region_id,
        evidence={
            "background_color": [int(c) for c in background_color],
            "standard_trimap_method": standard_info.get("method"),
            "body_outline_method": body_info.get("method"),
            "outline_recipe": body_outline.get("outline_recipe"),
            "standard_unknown_pixels": int(standard.unknown.sum()),
            "body_outline_sure_fg_pixels": int(body.sure_fg.sum()),
            "invaded_subject_pixels": int(invaded_subject.sum()),
            "invaded_subject_components": int(labels_count - 1),
            "largest_invaded_subject_component_px": largest_invaded,
            "shadow_facing_body_pixels": int(body_evidence.sum()),
            "shadow_components": body_components,
            "omitted_small_shadow_components": int(omitted_shadow_components),
            "min_body_anchor_shadow_area_px": int(min_body_anchor_area),
            "evidence_mode": "shadow_facing_subject_evidence_for_button_shadow",
        },
        ambiguity={
            "standard_unknown_score": 0.5,
            "opaque_subject_score": 0.5,
            "reason": "standard_trimap_lacks_subject_color_evidence_next_to_button_shadow",
        },
    )
    return [region], {region_id: body_evidence.copy()}


def _known_b_button_shadow_anchor_masks(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    route: dict[str, Any],
    *,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.18,
) -> dict[str, np.ndarray]:
    """Find shadow-like boundary anchors for same-key body analysis.

    These masks are not semantic ambiguity regions and never produce candidates.
    They are only a direction hint for the rare same-key button-body case: when
    the subject color is close to B, a shadow-facing strip can reveal where the
    body outline should release more foreground evidence.
    """
    if route.get("asset_kind") != "button":
        return {}

    h, w = image_srgb.shape[:2]
    bg = np.asarray(background_color, dtype=np.uint8).reshape(3)
    bgf = bg.astype(np.float32)
    dominant = int(np.argmax(bgf))
    sorted_bg = np.sort(bgf)
    if float(bgf[dominant]) < 64.0 or float(sorted_bg[-1] - sorted_bg[-2]) < 48.0:
        return {}

    profile = _route_semantic_parameter_profile(route)
    if profile == "opaque_hard_ui_no_shadow":
        # Route/CorridorKey has already measured a hard UI button without cast
        # shadow. Dark antialiasing, outline ridges, and body shading can satisfy
        # the known-B darkening equation locally, but without shadow-profile
        # support that color evidence is not enough to create a semantic shadow
        # ownership candidate.
        return {}
    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    bg_distance_max = float(params.get("pymatting_bg_threshold", 3.5))
    subject_distance_min = float(params.get("pymatting_fg_threshold", 24.0))
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3)).reshape(3)
    distance = oklab_distance(lab, bg_lab).astype(np.float32)
    near_bg = distance <= bg_distance_max
    exterior_bg = _exterior_mask(near_bg)
    enclosed_bg = near_bg & ~exterior_bg
    # Distance from B is only "not proven background". Hard shadows and dark
    # screen-material can satisfy this test too, so it must not be used as a
    # subject semantic anchor directly.
    non_bg_support = distance >= subject_distance_min
    img = image_srgb.astype(np.float32)
    other_max = np.max(np.delete(img, dominant, axis=2), axis=2)
    non_dominant_delta = np.max(np.delete(np.abs(img - bgf.reshape(1, 1, 3)), dominant, axis=2), axis=2)
    channel_drop = bgf[dominant] - img[..., dominant]
    bg_image = bgf.reshape(1, 1, 3)
    usable_bg = bg_image >= 8.0
    shadow_weights = np.where(usable_bg, bg_image * bg_image, 0.0).astype(np.float32)
    shadow_weight_sum = max(float(shadow_weights.sum()), 1e-6)
    display_shadow = np.clip(
        ((1.0 - img / np.maximum(bg_image, 1.0)) * shadow_weights).sum(axis=2) / shadow_weight_sum,
        0.0,
        1.0,
    ).astype(np.float32)
    shadow_replay = (1.0 - display_shadow[..., None]) * bg_image
    shadow_replay_error = np.mean(np.abs(shadow_replay - img), axis=2).astype(np.float32)
    exact_known_bg = np.all(image_srgb == bg.reshape(1, 1, 3), axis=2)

    max_channel = np.max(img, axis=2)
    scalar_shadow_for_anchor = (
        (display_shadow >= 0.025)
        & (shadow_replay_error <= 18.0)
        & ~exact_known_bg
    )
    bright_subject_floor = max(72.0, float(bgf[dominant]) * 0.55)
    shadow_subject_anchor = (
        non_bg_support
        & (max_channel >= bright_subject_floor)
        & ~scalar_shadow_for_anchor
    )
    shadow_subject_anchor = cv2.morphologyEx(
        shadow_subject_anchor.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    if int(shadow_subject_anchor.sum()) < max(24, int(round(float(non_bg_support.sum()) * 0.12))):
        shadow_subject_anchor = non_bg_support & ~scalar_shadow_for_anchor

    image_short = float(max(1, min(h, w)))
    near_subject_radius = max(1, int(round(image_short * 0.030)))
    near_subject = cv2.dilate(
        shadow_subject_anchor.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_subject_radius * 2 + 1, near_subject_radius * 2 + 1)),
        iterations=1,
    ).astype(bool)
    boundary_anchor = shadow_subject_anchor | enclosed_bg
    shadow_invade_px = float(image_short * 0.010)
    shadow_expand_px = float(image_short * 0.125)
    if bool(boundary_anchor.any()):
        dist_outside_anchor = cv2.distanceTransform((~boundary_anchor).astype(np.uint8), cv2.DIST_L2, 3)
        dist_inside_anchor = cv2.distanceTransform(boundary_anchor.astype(np.uint8), cv2.DIST_L2, 3)
        subject_invasion_corridor = boundary_anchor & (dist_inside_anchor <= shadow_invade_px)
        background_expansion_corridor = (~boundary_anchor) & (dist_outside_anchor <= shadow_expand_px)
        shadow_boundary_corridor = (
            background_expansion_corridor
            | subject_invasion_corridor
        )
    else:
        subject_invasion_corridor = np.zeros((h, w), dtype=bool)
        background_expansion_corridor = np.zeros((h, w), dtype=bool)
        shadow_boundary_corridor = np.zeros((h, w), dtype=bool)
    background_shadow_solvable = (
        (display_shadow >= 0.025)
        & (shadow_replay_error <= 10.0)
        & (background_expansion_corridor | (subject_invasion_corridor & near_subject))
        & ~exterior_bg
    )
    screen_channel_darkening = (
        (channel_drop >= max(10.0, float(bgf[dominant]) * 0.06))
        & (img[..., dominant] > other_max + max(6.0, float(bgf[dominant]) * 0.03))
        & (non_dominant_delta <= 52.0)
        & (background_expansion_corridor | (subject_invasion_corridor & near_subject))
        & ~exterior_bg
    )
    shadow_core = (background_shadow_solvable | screen_channel_darkening) & shadow_boundary_corridor & ~enclosed_bg
    outward_shadow_tail = (
        (display_shadow >= 0.006)
        & (shadow_replay_error <= 12.0)
        & background_expansion_corridor
        & ~exact_known_bg
        & ~enclosed_bg
    )
    shadow_direction_info: dict[str, Any] = {
        "enabled": False,
        "reason": "shadow_region_is_not_direction_clipped",
    }
    shadow_like = _components_touching_seed(shadow_core | outward_shadow_tail, shadow_core)
    shadow_like = cv2.morphologyEx(shadow_like.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)

    min_area = max(12, int(round(float(h * w) * min_area_ratio)))
    max_area = max(min_area, int(round(float(h * w) * max_area_ratio)))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(shadow_like.astype(np.uint8), connectivity=8)
    components: list[tuple[int, int]] = []
    adjacency_radius = max(1, int(round(image_short * 0.006)))
    adjacency_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (adjacency_radius * 2 + 1, adjacency_radius * 2 + 1),
    )
    for label_idx in range(1, labels_count):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = labels == label_idx
        ring = cv2.dilate(comp.astype(np.uint8), adjacency_kernel, iterations=1).astype(bool) & ~comp
        ring_pixels = int(ring.sum())
        exterior_bg_pixels = int((ring & exterior_bg).sum()) if ring_pixels else 0
        near_bg_pixels = int((ring & near_bg).sum()) if ring_pixels else 0
        subject_ring_pixels = int((ring & shadow_subject_anchor).sum()) if ring_pixels else 0
        exterior_bg_fraction = float(exterior_bg_pixels) / max(float(ring_pixels), 1.0)
        # Color alone cannot distinguish black/dark subject decoration from a
        # black shadow over the measured screen. As an anchor it must expose a
        # background-side boundary; subject-only dark details would pull the body
        # detector inward, which is exactly the B056 failure mode we avoid.
        min_exterior_bg_fraction = 0.015
        if exterior_bg_fraction < min_exterior_bg_fraction:
            continue
        components.append((area, label_idx))
    components.sort(reverse=True)
    if components:
        dominant_area = int(components[0][0])
        min_relative_area = max(min_area, int(round(float(dominant_area) * 0.05)))
        components = [
            (area, label_idx)
            for area, label_idx in components
            if int(area) >= min_relative_area
        ]

    masks: dict[str, np.ndarray] = {}
    for area, label_idx in components[:3]:
        comp = labels == label_idx
        region_id = f"button_shadow_anchor_{len(masks)}"
        masks[region_id] = comp.copy()
    return masks


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


def _same_key_opaque_outline_candidate() -> SemanticCandidate:
    return SemanticCandidate(
        id="opaque_outline",
        label="Opaque outline",
        intent="Treat the same-key button body as opaque material inside the measured outline.",
        default=True,
        confidence=0.9,
        risk_level="medium",
        decision={
            "policy": "same_key_opaque_outline",
            "button_body_policy": "opaque_subject",
            "pymatting_trimap_mode": "same_key_opaque_body_outline",
            "pymatting_unknown_grow_px": 0,
        },
        reasons=[
            "same-key button outline was measured before execution",
            "opaque interpretation ignores enclosed near-background hole choices",
        ],
    )


def _same_key_corridorkey_translucent_candidate() -> SemanticCandidate:
    hint_value = 0.32
    return SemanticCandidate(
        id="semi_transparent_corridorkey",
        label="Semi-transparent CorridorKey",
        intent="Treat the same-key button interior as semi-transparent screen material and run CorridorKey with a light prior.",
        default=True,
        confidence=0.64,
        risk_level="medium",
        decision={
            "policy": "same_key_semi_transparent_corridorkey",
            "corridorkey_hint_value": hint_value,
        },
        reasons=[
            "same-key green/blue button can also be interpreted as translucent screen material",
            "near-background material uses a stronger foreground prior so it is not erased too aggressively",
        ],
    )


def _route_uses_same_key_opaque_outline(route: dict[str, Any]) -> bool:
    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    return (
        route.get("algorithm") == "pymatting_known_b"
        and str(params.get("pymatting_trimap_mode") or "") == "same_key_opaque_body_outline"
    )


def _route_uses_same_key_corridorkey(route: dict[str, Any]) -> bool:
    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    return (
        route.get("algorithm") == "corridorkey"
        and str(params.get("same_key_button_interpretation") or "") == "semi_transparent_corridorkey"
    )


def _default_candidate_id(candidates: list[SemanticCandidate]) -> str:
    for candidate in candidates:
        if candidate.default:
            return candidate.id
    return candidates[0].id if candidates else "auto_default"


def _enclosed_near_bg_region_scores(region: AmbiguityRegion) -> tuple[float, float, list[str]]:
    evidence = region.evidence
    evidence_mode = str(evidence.get("evidence_mode") or "")
    bg_distance_max = max(float(evidence.get("bg_distance_max") or 1.0), 1e-6)
    bg_distance_p50 = float(evidence.get("bg_distance_p50") or 0.0)
    distance_confidence = 1.0 - float(np.clip(bg_distance_p50 / bg_distance_max, 0.0, 1.0))
    support_fraction = float(evidence.get("subject_support_fraction") or 0.0)
    support_confidence = float(np.clip(support_fraction / 0.25, 0.0, 1.0))
    subject_outline_confidence = float(np.clip(float(evidence.get("subject_outline_confidence") or 0.0), 0.0, 1.0))
    hole_outline_confidence = float(np.clip(float(evidence.get("hole_outline_confidence") or 0.0), 0.0, 1.0))

    reasons = [
        "enclosed region does not touch exterior background",
        "candidate policy is scored per region before execution",
    ]
    if evidence_mode == "tight_enclosed_background_match":
        transparent_score = (
            0.42
            + 0.17 * distance_confidence
            + 0.28 * hole_outline_confidence
            + 0.04 * support_confidence
        )
        subject_score = (
            0.50
            - 0.04 * distance_confidence
            + 0.45 * subject_outline_confidence * (1.0 - hole_outline_confidence)
            + 0.04 * (1.0 - support_confidence)
        )
        if hole_outline_confidence >= 0.65:
            reasons.append("matching enclosed outline raises transparent-hole confidence")
        elif subject_outline_confidence >= 0.45:
            reasons.append("flat outlined subject without matching hole outline raises opaque-material confidence")
        else:
            reasons.append("tight known-background color match remains ambiguous without hole-outline evidence")
    elif evidence_mode == "translucent_known_b_material_band":
        transparent_score = 0.30 + 0.08 * distance_confidence
        subject_score = 0.66 + 0.10 * support_confidence
        reasons.append("wide near-background band is more likely retained translucent material")
    else:
        transparent_score = 0.48 + 0.12 * distance_confidence
        subject_score = 0.52 + 0.06 * support_confidence
        reasons.append("single-image evidence keeps this region semantically ambiguous")

    return (
        float(np.clip(transparent_score, 0.05, 0.95)),
        float(np.clip(subject_score, 0.05, 0.95)),
        reasons,
    )


def _enclosed_near_bg_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
    region_ids = [region.id for region in regions]
    total_area = sum(region.area_px for region in regions)
    preview = {
        "regions": region_ids,
        "area_px": int(total_area),
        "bbox_xyxy": regions[0].bbox_xyxy if regions else [0, 0, 0, 0],
    }
    transparent_scores: dict[str, float] = {}
    subject_scores: dict[str, float] = {}
    auto_policies: dict[str, str] = {}
    auto_units: list[dict[str, Any]] = []
    auto_reasons: list[str] = [
        "Auto is a recommendation strategy; Execute receives concrete per-region hole policies",
    ]
    for region in regions:
        transparent_score, subject_score, region_reasons = _enclosed_near_bg_region_scores(region)
        transparent_scores[region.id] = transparent_score
        subject_scores[region.id] = subject_score
        policy = "transparent_hole" if transparent_score >= subject_score else "subject"
        score = transparent_score if policy == "transparent_hole" else subject_score
        auto_policies[region.id] = policy
        auto_units.append(
            {
                "option_id": f"auto_{region.id}_{policy}",
                "label": "Transparent hole" if policy == "transparent_hole" else "Subject material",
                "score": float(score),
                "regions": [region.id],
                "evidence_mode": str(region.evidence.get("evidence_mode") or ""),
            }
        )
        for reason in region_reasons:
            if reason not in auto_reasons:
                auto_reasons.append(reason)

    auto_score = float(
        sum(
            transparent_scores[region_id] if policy == "transparent_hole" else subject_scores[region_id]
            for region_id, policy in auto_policies.items()
        )
        / max(float(len(auto_policies)), 1.0)
    )
    auto_decision: dict[str, Any] = {
        "candidate_strategy": "auto_region_recommendation",
        "enclosed_near_bg_region_policies": auto_policies,
        "candidate_score": auto_score,
        "candidate_rank": 0,
        "candidate_units": auto_units,
    }
    unique_auto_policies = set(auto_policies.values())
    if len(unique_auto_policies) == 1:
        auto_decision["enclosed_near_bg_policy"] = next(iter(unique_auto_policies))

    cut_score = float(sum(transparent_scores.values()) / max(float(len(transparent_scores)), 1.0))
    keep_score = float(sum(subject_scores.values()) / max(float(len(subject_scores)), 1.0))
    return [
        SemanticCandidate(
            id="auto_recommended_holes",
            label="Auto recommended",
            intent="Apply the highest-confidence concrete hole/material ownership policy per region.",
            default=True,
            confidence=auto_score,
            risk_level="medium",
            decision=auto_decision,
            regions=region_ids,
            preview={**preview, "score": auto_score, "rank": 0, "strategy": "auto_region_recommendation"},
            reasons=auto_reasons,
        ),
        SemanticCandidate(
            id="protect_near_bg_subject",
            label="Keep internal light material",
            intent="Treat enclosed near-background pixels as subject-owned material.",
            default=False,
            confidence=keep_score,
            risk_level="medium",
            decision={
                "enclosed_near_bg_policy": "subject",
                "enclosed_near_bg_region_policies": {region_id: "subject" for region_id in region_ids},
            },
            regions=region_ids,
            preview={**preview, "score": keep_score},
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
            confidence=cut_score,
            risk_level="medium",
            decision={
                "enclosed_near_bg_policy": "transparent_hole",
                "enclosed_near_bg_region_policies": {region_id: "transparent_hole" for region_id in region_ids},
            },
            regions=region_ids,
            preview={**preview, "score": cut_score},
            reasons=[
                "pixels match the known background color",
                "enclosed same-background components can be true holes in UI or icons",
            ],
        ),
    ]


def _screen_material_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
    return _corridorkey_translucency_candidates(regions)


def _known_b_connected_shadow_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
    shadow_regions = [region for region in regions if region.type == "connected_known_b_shadow_ownership"]
    shadow_ids = [region.id for region in shadow_regions]
    area_px = int(sum(region.area_px for region in shadow_regions))
    bbox = shadow_regions[0].bbox_xyxy if shadow_regions else [0, 0, 0, 0]
    policies = {region_id: "shadow_unknown" for region_id in shadow_ids}
    return [
        SemanticCandidate(
            id="auto_default",
            label="Auto default",
            intent="Use the current route/profile default interpretation.",
            default=True,
            confidence=0.56,
            risk_level="medium",
            decision={"policy": "auto_default"},
            regions=shadow_ids,
            preview={
                "regions": shadow_ids,
                "area_px": area_px,
                "bbox_xyxy": bbox,
                "strategy": "current_route_default",
            },
            reasons=[
                "connected scalar-shadow evidence overlaps sure-FG, but default preserves current route behavior",
            ],
        ),
        SemanticCandidate(
            id="solve_connected_shadow",
            label="Solve connected shadow",
            intent="Release the connected scalar-darkening region to trimap unknown so ShadowPatch can solve it as known-background shadow.",
            default=False,
            confidence=0.68,
            risk_level="medium",
            decision={
                "known_b_shadow_policy": "shadow_unknown",
                "known_b_shadow_region_policies": policies,
                "shadow_mode": "on",
            },
            regions=shadow_ids,
            preview={
                "regions": shadow_ids,
                "area_px": area_px,
                "bbox_xyxy": bbox,
                "strategy": "known_b_shadow_unknown_release",
            },
            reasons=[
                "region follows scalar known-background darkening",
                "region is connected to the exterior solve band",
                "region is adjacent to non-scalar subject material, so it is ambiguous shadow rather than isolated dark material",
            ],
        ),
    ]


def _corridorkey_translucency_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
    all_ids = [region.id for region in regions]
    screen_ids = [region.id for region in regions if region.type == "screen_material_or_translucency"]
    core_ids = [region.id for region in regions if region.type == "glass_core_transparency"]
    gradient_ids = [region.id for region in regions if region.type == "soft_alpha_gradient"]
    preview = {
        "regions": all_ids,
        "area_px": int(sum(region.area_px for region in regions)),
        "bbox_xyxy": regions[0].bbox_xyxy if regions else [0, 0, 0, 0],
    }
    region_types = [
        region_type
        for region_type, ids in (
            ("glass_core_transparency", core_ids),
            ("soft_alpha_gradient", gradient_ids),
            ("screen_material_or_translucency", screen_ids),
        )
        if ids
    ]
    def spec_for(value: float) -> tuple[str, str, str, float, bool, float]:
        value = float(value)
        suffix = f"{int(round(value * 100)):03d}"
        candidate_id = "auto_default" if np.isclose(value, 0.32) else f"corridorkey_hint_{suffix}"
        label = f"CorridorKey hint {value:.2f}"
        if np.isclose(value, 0.0):
            intent = "Run CorridorKey with no foreground prior."
            confidence = 0.42
        elif np.isclose(value, 0.16):
            intent = "Run CorridorKey with a light full-frame foreground prior."
            confidence = 0.5
        elif np.isclose(value, 0.32):
            intent = "Run CorridorKey with the default full-frame soft prior."
            confidence = 0.62
        elif value < 0.7:
            intent = "Run CorridorKey with a stronger full-frame foreground prior."
            confidence = 0.5
        else:
            intent = "Run CorridorKey with a high full-frame foreground prior."
            confidence = 0.46
        return candidate_id, label, intent, value, bool(np.isclose(value, 0.32)), confidence

    specs = [spec_for(value) for value in corridorkey_hint_strengths()]
    return [
        SemanticCandidate(
            id=candidate_id,
            label=label,
            intent=intent,
            default=default,
            confidence=confidence,
            risk_level="medium",
            decision={
                "policy": "corridorkey_constant_hint",
                "corridorkey_hint_value": value,
                "review_region_types": region_types,
            },
            regions=all_ids,
            preview=preview,
            reasons=[
                "CorridorKey ambiguity is controlled by full-frame hint strength, not post-alpha hard constraints",
                "single-image evidence cannot choose one stable hint strength for every screen asset",
            ],
        )
        for candidate_id, label, intent, value, default, confidence in specs
    ]


@dataclass(frozen=True)
class _CandidateOption:
    id: str
    label: str
    decision: dict[str, Any]
    regions: list[str]
    score: float
    weight: float
    reasons: list[str]


@dataclass(frozen=True)
class _CandidateUnit:
    id: str
    kind: str
    options: list[_CandidateOption]


def _merge_candidate_decisions(options: list[_CandidateOption]) -> dict[str, Any]:
    decision: dict[str, Any] = {}
    hole_policies: dict[str, str] = {}
    shadow_policies: dict[str, str] = {}
    units: list[dict[str, Any]] = []
    for option in options:
        option_decision = dict(option.decision)
        option_holes = option_decision.pop("enclosed_near_bg_region_policies", None)
        if isinstance(option_holes, dict):
            for region_id, policy in option_holes.items():
                hole_policies[str(region_id)] = str(policy)
        option_shadows = option_decision.pop("known_b_shadow_region_policies", None)
        if isinstance(option_shadows, dict):
            for region_id, policy in option_shadows.items():
                shadow_policies[str(region_id)] = str(policy)
        decision.update(option_decision)
        units.append(
            {
                "option_id": option.id,
                "label": option.label,
                "score": float(option.score),
                "regions": list(option.regions),
            }
        )
    if hole_policies:
        decision["enclosed_near_bg_region_policies"] = hole_policies
        unique_values = set(hole_policies.values())
        if len(unique_values) == 1:
            decision["enclosed_near_bg_policy"] = next(iter(unique_values))
    if shadow_policies:
        decision["known_b_shadow_region_policies"] = shadow_policies
        unique_values = set(shadow_policies.values())
        if len(unique_values) == 1:
            decision["known_b_shadow_policy"] = next(iter(unique_values))
    decision["candidate_units"] = units
    return decision


def _rank_candidate_option_combos(
    units: list[_CandidateUnit],
    *,
    beam_width: int = 24,
    max_candidates: int = 8,
) -> list[tuple[float, list[_CandidateOption]]]:
    """Score and prune semantic candidate combinations.

    Button candidates are a product of the semantic units that still require a
    user choice: enclosed known-B holes and the rare same-key opaque-body case.
    Shadow is not a unit here; it is boundary evidence inside the trimap builder.
    The score is a weighted mean of observable local evidence and is exposed as
    candidate confidence rather than hidden as a router default.
    """
    combos: list[tuple[float, float, list[_CandidateOption]]] = [(0.0, 0.0, [])]
    for unit in units:
        expanded: list[tuple[float, float, list[_CandidateOption]]] = []
        for score_sum, weight_sum, chosen in combos:
            for option in unit.options:
                expanded.append(
                    (
                        score_sum + option.score * option.weight,
                        weight_sum + option.weight,
                        chosen + [option],
                    )
                )
        expanded.sort(key=lambda item: (item[0] / max(item[1], 1e-6), item[0]), reverse=True)
        combos = expanded[:beam_width]
    ranked = [
        (score_sum / max(weight_sum, 1e-6), chosen)
        for score_sum, weight_sum, chosen in combos
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:max_candidates]


def _candidate_id_from_options(options: list[_CandidateOption]) -> str:
    if not options:
        return "auto_default"
    return "use_" + "_".join(option.id for option in options)


def _button_candidate_units(regions: list[AmbiguityRegion]) -> list[_CandidateUnit]:
    hole_regions = [region for region in regions if region.type == "enclosed_near_background"]
    body_regions = [region for region in regions if region.type == "button_body_subject_ownership"]
    shadow_regions = [region for region in regions if region.type == "connected_known_b_shadow_ownership"]
    units: list[_CandidateUnit] = []

    if shadow_regions:
        shadow_ids = [region.id for region in shadow_regions]
        units.append(
            _CandidateUnit(
                id="connected_shadow",
                kind="connected_known_b_shadow_ownership",
                options=[
                    _CandidateOption(
                        id="solve_shadow",
                        label="Solve shadow",
                        decision={
                            "known_b_shadow_policy": "shadow_unknown",
                            "known_b_shadow_region_policies": {
                                region_id: "shadow_unknown" for region_id in shadow_ids
                            },
                            "shadow_mode": "on",
                        },
                        regions=shadow_ids,
                        score=0.68,
                        weight=1.0,
                        reasons=[
                            "connected scalar known-B darkening overlaps sure-FG",
                            "release the disputed component into unknown for shadow solving",
                        ],
                    ),
                    _CandidateOption(
                        id="keep_shadow_default",
                        label="Keep default shadow ownership",
                        decision={"known_b_shadow_policy": "current_route_default"},
                        regions=shadow_ids,
                        score=0.72,
                        weight=1.0,
                        reasons=[
                            "current route interpretation remains available as a counterfactual",
                        ],
                    ),
                ],
            )
        )

    if body_regions:
        body_ids = [region.id for region in body_regions]
        # Opaque body is a same-key special-case dimension, not the ordinary
        # hard-button shadow path. Standard body remains the default route for
        # normal buttons; this unit exists only when route/profile evidence says
        # same-key material can collapse under the standard trimap.
        invaded = sum(int(region.evidence.get("invaded_subject_pixels") or 0) for region in body_regions)
        body_score_boost = min(0.08, invaded / max(1.0, sum(region.area_px for region in body_regions)) * 0.02)
        units.append(
            _CandidateUnit(
                id="button_body",
                kind="button_body_subject_ownership",
                options=[
                    _CandidateOption(
                        id="opaque_body",
                        label="Opaque body",
                        decision={
                            "button_body_policy": "opaque_subject",
                            "pymatting_trimap_mode": "same_key_opaque_body_outline",
                            "pymatting_unknown_grow_px": 2,
                        },
                        regions=body_ids,
                        score=min(0.94, 0.84 + body_score_boost),
                        weight=1.15,
                        reasons=[
                            "measured button outline gives a better shadow-facing unknown distribution",
                            "standard trimap leaves the disputed shadow band with too little subject color evidence",
                        ],
                    ),
                    _CandidateOption(
                        id="standard_body",
                        label="Standard body",
                        decision={
                            "button_body_policy": "standard_subject_evidence",
                            "pymatting_trimap_mode": "standard",
                            "pymatting_unknown_grow_px": 0,
                        },
                        regions=body_ids,
                        score=0.42,
                        weight=1.15,
                        reasons=[
                            "standard trimap is retained as a counterfactual for uncertain button-body tracing",
                        ],
                    ),
                ],
            )
        )

    if len(hole_regions) > 1:
        hole_ids = [region.id for region in hole_regions]
        region_scores = [_enclosed_near_bg_region_scores(region) for region in hole_regions]
        cut_score = float(sum(score[0] for score in region_scores) / max(float(len(region_scores)), 1.0))
        keep_score = float(sum(score[1] for score in region_scores) / max(float(len(region_scores)), 1.0))
        units.append(
            _CandidateUnit(
                id="holes",
                kind="enclosed_near_background_group",
                options=[
                    _CandidateOption(
                        id="cut_all_holes",
                        label="Cut all holes",
                        decision={
                            "enclosed_near_bg_region_policies": {
                                region_id: "transparent_hole" for region_id in hole_ids
                            }
                        },
                        regions=hole_ids,
                        score=cut_score,
                        weight=0.9,
                        reasons=[
                            "multiple enclosed same-background components are grouped as one hole ownership choice",
                        ],
                    ),
                    _CandidateOption(
                        id="keep_all_holes",
                        label="Keep all holes",
                        decision={
                            "enclosed_near_bg_region_policies": {
                                region_id: "subject" for region_id in hole_ids
                            }
                        },
                        regions=hole_ids,
                        score=keep_score,
                        weight=0.9,
                        reasons=[
                            "single-image evidence cannot prove that repeated enclosed same-background details are transparent cutouts",
                        ],
                    ),
                ],
            )
        )
    else:
        for index, region in enumerate(hole_regions):
            cut_score, keep_score, _reasons = _enclosed_near_bg_region_scores(region)
            units.append(
                _CandidateUnit(
                    id=f"hole_{index}",
                    kind="enclosed_near_background",
                    options=[
                        _CandidateOption(
                            id=f"cut_hole_{index}",
                            label=f"Cut hole {index + 1}",
                            decision={"enclosed_near_bg_region_policies": {region.id: "transparent_hole"}},
                            regions=[region.id],
                            score=cut_score,
                            weight=0.9,
                            reasons=[
                                "enclosed pixels match the known background color",
                                "hole decisions are scored per region before candidate pruning",
                            ],
                        ),
                        _CandidateOption(
                            id=f"keep_hole_{index}",
                            label=f"Keep hole {index + 1}",
                            decision={"enclosed_near_bg_region_policies": {region.id: "subject"}},
                            regions=[region.id],
                            score=keep_score,
                            weight=0.9,
                            reasons=[
                                "single-image evidence cannot prove every enclosed same-background region is transparent",
                            ],
                        ),
                    ],
                )
            )

    return units


def _button_known_b_candidates(regions: list[AmbiguityRegion]) -> list[SemanticCandidate]:
    units = _button_candidate_units(regions)
    if not units:
        return [_ready_candidate()]

    ranked = _rank_candidate_option_combos(units)
    candidates: list[SemanticCandidate] = []
    for rank, (score, options) in enumerate(ranked):
        decision = _merge_candidate_decisions(options)
        decision["candidate_score"] = float(score)
        decision["candidate_rank"] = int(rank)
        region_ids = sorted({region_id for option in options for region_id in option.regions})
        preview = {
            "regions": region_ids,
            "area_px": int(sum(region.area_px for region in regions if region.id in region_ids)),
            "bbox_xyxy": regions[0].bbox_xyxy if regions else [0, 0, 0, 0],
            "score": float(score),
            "rank": int(rank),
            "beam_width": 24,
            "max_candidates": 8,
            "unit_count": len(units),
        }
        candidates.append(
            SemanticCandidate(
                id=_candidate_id_from_options(options),
                label=" + ".join(option.label for option in options),
                intent="Apply the ranked ownership decisions for the measured Known-B button ambiguities.",
                default=rank == 0,
                confidence=float(score),
                risk_level="medium",
                decision=decision,
                regions=region_ids,
                preview=preview,
                reasons=[reason for option in options for reason in option.reasons],
            )
        )
    return candidates


def _candidate_with_route(
    candidate: SemanticCandidate,
    *,
    route_candidate_id: str,
    id_prefix: str | None = None,
    region_id_map: dict[str, str] | None = None,
) -> SemanticCandidate:
    region_id_map = region_id_map or {}
    candidate_id = f"{id_prefix}__{candidate.id}" if id_prefix else candidate.id
    regions = [region_id_map.get(region_id, region_id) for region_id in candidate.regions]
    preview = dict(candidate.preview)
    if isinstance(preview.get("regions"), list):
        preview["regions"] = [region_id_map.get(str(region_id), str(region_id)) for region_id in preview["regions"]]
    decision = _remap_decision_region_ids(candidate.decision, region_id_map)
    return replace(
        candidate,
        id=candidate_id,
        route_candidate_id=route_candidate_id,
        regions=regions,
        preview=preview,
        decision=decision,
    )


def _remap_decision_region_ids(decision: dict[str, Any], region_id_map: dict[str, str]) -> dict[str, Any]:
    if not region_id_map:
        return dict(decision)
    remapped = dict(decision)
    hole_policies = remapped.get("enclosed_near_bg_region_policies")
    if isinstance(hole_policies, dict):
        remapped["enclosed_near_bg_region_policies"] = {
            region_id_map.get(str(region_id), str(region_id)): policy
            for region_id, policy in hole_policies.items()
        }
    shadow_policies = remapped.get("known_b_shadow_region_policies")
    if isinstance(shadow_policies, dict):
        remapped["known_b_shadow_region_policies"] = {
            region_id_map.get(str(region_id), str(region_id)): policy
            for region_id, policy in shadow_policies.items()
        }
    units = remapped.get("candidate_units")
    if isinstance(units, list):
        updated_units = []
        for unit in units:
            if not isinstance(unit, dict):
                updated_units.append(unit)
                continue
            unit_copy = dict(unit)
            if isinstance(unit_copy.get("regions"), list):
                unit_copy["regions"] = [
                    region_id_map.get(str(region_id), str(region_id))
                    for region_id in unit_copy["regions"]
                ]
            updated_units.append(unit_copy)
        remapped["candidate_units"] = updated_units
    return remapped


def _prefix_route_regions(
    regions: list[AmbiguityRegion],
    masks: dict[str, np.ndarray],
    *,
    prefix: str | None,
) -> tuple[list[AmbiguityRegion], dict[str, np.ndarray], dict[str, str]]:
    if not prefix:
        return regions, masks, {}
    region_id_map = {region.id: f"{prefix}__{region.id}" for region in regions}
    updated_regions = [
        replace(region, id=region_id_map[region.id], mask_ref=region_id_map[region.id])
        for region in regions
    ]
    updated_masks = {
        region_id_map.get(region_id, region_id): mask
        for region_id, mask in masks.items()
    }
    return updated_regions, updated_masks, region_id_map


def _semantic_plan_for_route(
    image_srgb: np.ndarray,
    *,
    route: dict[str, Any],
    preprocess: PreprocessDecision | None,
) -> tuple[np.ndarray, PreprocessDecision | None, list[AmbiguityRegion], dict[str, np.ndarray], list[SemanticCandidate]]:
    regions: list[AmbiguityRegion] = []
    region_masks: dict[str, np.ndarray] = {}
    background = _background_from_route(route)
    semantic_input = image_srgb
    effective_preprocess = preprocess
    if route.get("algorithm") == "pymatting_known_b" and background is not None:
        if _route_uses_same_key_opaque_outline(route):
            return semantic_input, effective_preprocess, [], {}, [_same_key_opaque_outline_candidate()]
        hole_regions, hole_masks = _enclosed_near_background_regions(
            semantic_input,
            background,
            **_known_b_semantic_region_settings(route),
        )
        regions = hole_regions
        region_masks = dict(hole_masks)
        shadow_regions, shadow_masks = _known_b_connected_shadow_regions(
            semantic_input,
            background,
            route,
        )
        if shadow_regions:
            regions = regions + shadow_regions
            region_masks.update(shadow_masks)
        if route.get("asset_kind") == "button":
            # Shadow-like darkening is collected only as a directional anchor for
            # same-key body tracing. It is deliberately not merged into regions:
            # ordinary button shadows should be handled by the BG-seed outline
            # trimap, while only enclosed known-B islands remain as hole choices.
            shadow_anchor_masks = _known_b_button_shadow_anchor_masks(
                semantic_input,
                background,
                route,
            )
            body_regions, body_masks = _known_b_button_body_regions(
                semantic_input,
                background,
                route,
                shadow_anchor_masks=shadow_anchor_masks,
            )
            regions = hole_regions + shadow_regions + body_regions
            region_masks = {**hole_masks, **shadow_masks, **body_masks}
    elif route.get("algorithm") == "corridorkey":
        if _route_uses_same_key_corridorkey(route):
            return semantic_input, effective_preprocess, [], {}, [_same_key_corridorkey_translucent_candidate()]
        # CorridorKey is steered only by full-frame constant hint strength here.
        # The feature-hint experiments remain available as helpers, but are not
        # wired into Analyze or Execute.
        regions = []
        region_masks = {}

    if regions:
        region_types = {region.type for region in regions}
        if "glass_core_transparency" in region_types or "soft_alpha_gradient" in region_types:
            candidates = _corridorkey_translucency_candidates(regions)
        elif "screen_material_or_translucency" in region_types:
            candidates = _screen_material_candidates(regions)
        elif (
            "button_body_subject_ownership" in region_types
            or (route.get("asset_kind") == "button" and "enclosed_near_background" in region_types)
        ):
            candidates = _button_known_b_candidates(regions)
        elif "connected_known_b_shadow_ownership" in region_types:
            candidates = _known_b_connected_shadow_candidates(regions)
        else:
            candidates = _enclosed_near_bg_candidates(regions)
    elif route.get("algorithm") == "corridorkey":
        candidates = _corridorkey_translucency_candidates([])
    else:
        candidates = [_ready_candidate()]
    return semantic_input, effective_preprocess, regions, region_masks, candidates


def _should_expand_route_semantics(route: dict[str, Any], *, default_route: dict[str, Any]) -> bool:
    """Return whether Analyze should materialize semantic candidates for a route."""

    if route.get("algorithm") == default_route.get("algorithm"):
        return True
    if default_route.get("algorithm") == "corridorkey" and route.get("algorithm") == "pymatting_known_b":
        return False
    return True


def analyze_candidates(
    image_srgb: np.ndarray,
    *,
    preprocess: PreprocessDecision | None = None,
    screen_mode: str = "auto",
    preset: str = "auto",
    fallback_background_color: tuple[int, int, int] = (0, 200, 0),
) -> AnalyzeResult:
    """Run lightweight route/profile and semantic candidate analysis."""

    route_candidate_models = build_route_candidates(
        image_srgb,
        screen_mode=screen_mode,
        preset=preset,
        fallback_background_color=fallback_background_color,
    )
    default_route_candidate = select_default_route_candidate(route_candidate_models)
    route = _route_payload(default_route_candidate)
    route_candidate_payloads = [_route_candidate_payload(candidate) for candidate in route_candidate_models]
    route_candidate_models_for_semantics = sorted(
        route_candidate_models,
        key=lambda candidate: (candidate.id != default_route_candidate.id, candidate.id),
    )
    multiple_routes = len(route_candidate_models) > 1

    all_regions: list[AmbiguityRegion] = []
    all_region_masks: dict[str, np.ndarray] = {}
    route_plans: list[dict[str, Any]] = []
    effective_preprocess = preprocess
    for route_candidate in route_candidate_models_for_semantics:
        route_candidate_id = route_candidate.id
        route_payload = _route_payload(route_candidate)
        if not _should_expand_route_semantics(route_payload, default_route=route):
            continue
        semantic_input, route_preprocess, regions, region_masks, candidates = _semantic_plan_for_route(
            image_srgb,
            route=route_payload,
            preprocess=preprocess,
        )
        if route_preprocess is not None:
            effective_preprocess = _merge_preprocess_decisions(effective_preprocess, route_preprocess)
        prefix = route_candidate_id if multiple_routes else None
        regions, region_masks, region_id_map = _prefix_route_regions(regions, region_masks, prefix=prefix)
        candidates = [
            _candidate_with_route(
                candidate,
                route_candidate_id=route_candidate_id,
                id_prefix=prefix,
                region_id_map=region_id_map,
            )
            for candidate in candidates
        ]
        all_regions.extend(regions)
        all_region_masks.update(region_masks)
        route_plans.append(
            {
                "route": route_payload,
                "route_candidate_id": route_candidate_id,
                "semantic_input": semantic_input,
                "regions": regions,
                "region_masks": region_masks,
                "candidates": candidates,
            }
        )

    analysis_id = _analysis_id(
        image_srgb,
        route={"default": route, "route_candidates": route_candidate_payloads},
        preprocess=effective_preprocess,
        regions=all_regions,
    )
    region_refs = _with_mask_refs(all_regions, analysis_id)
    region_by_id = {region.id: region for region in region_refs}
    all_regions = region_refs

    all_candidates: list[SemanticCandidate] = []
    preview_assets: dict[str, Any] = {
        "schema": "ermbg.analysis_preview_assets.v1",
        "image_space": "analyze_semantic_input",
    }
    default_route_candidate_id = default_route_candidate.id
    for plan in route_plans:
        plan_regions = [region_by_id.get(region.id, region) for region in plan["regions"]]
        candidates, assets = _attach_preview_assets(
            plan["semantic_input"],
            route=plan["route"],
            route_algorithm=str(plan["route"].get("algorithm") or ""),
            analysis_id=analysis_id,
            regions=plan_regions,
            region_masks=plan["region_masks"],
            candidates=plan["candidates"],
            eager_trimap=(plan["route_candidate_id"] == default_route_candidate_id),
        )
        preview_assets.update(assets)
        all_candidates.extend(candidates)

    default_route_candidates = [
        candidate
        for candidate in all_candidates
        if candidate.route_candidate_id == default_route_candidate_id
    ]
    default_candidate_id = _default_candidate_id(default_route_candidates or all_candidates)
    status = "needs_decision" if all_regions or len(route_candidate_models) > 1 else "ready"
    return AnalyzeResult(
        status=status,
        analysis_id=analysis_id,
        preprocess=effective_preprocess,
        route=route,
        route_candidates=route_candidate_payloads,
        default_route_candidate_id=default_route_candidate_id,
        ambiguity_regions=all_regions,
        candidates=all_candidates,
        default_candidate_id=default_candidate_id,
        preview_assets=preview_assets,
    )


__all__ = ["analyze_candidates"]
