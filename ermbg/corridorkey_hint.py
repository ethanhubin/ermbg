"""Feature-driven CorridorKey hint generation.

These helpers only build model-input hints. They do not post-clamp alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from .keyer import KeyerThresholds, chromatic_key_alpha

CorridorKeyHintVariant = Literal[
    "current_default_prior",
    "feature_conservative",
    "feature_balanced",
    "feature_internal_opaque",
    "feature_translucent",
    "full_frame_zero",
]


@dataclass(frozen=True)
class CorridorKeyHintFeatures:
    key_alpha: np.ndarray
    outline_mask: np.ndarray
    outline_inner_mask: np.ndarray
    control_outline_mask: np.ndarray
    control_outline_inner_mask: np.ndarray
    subject_support: np.ndarray
    hard_subject: np.ndarray
    translucent_candidate: np.ndarray
    internal_transparency_candidate: np.ndarray
    soft_boundary_candidate: np.ndarray
    background: np.ndarray
    bbox_xyxy: list[int]
    bbox_plus_2_xyxy: list[int]
    bbox_plus_2_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CorridorKeyHintPlan:
    variant: CorridorKeyHintVariant
    hint: np.ndarray
    features: CorridorKeyHintFeatures
    metadata: dict[str, Any]


def corridorkey_full_frame_prior_value(
    *,
    execution_profile: str,
    screen_mode: str,
) -> tuple[float, str]:
    """Return the default full-frame CorridorKey soft-prior hint value."""

    _ = (execution_profile, screen_mode)
    return 0.32, "soft_prior"


def _smooth_mask(mask: np.ndarray, *, sigma: float = 3.0) -> np.ndarray:
    values = mask.astype(np.float32)
    if not bool(values.any()):
        return values
    radius = max(1, int(round(float(sigma) * 3.0)))
    kernel = radius * 2 + 1
    return np.clip(cv2.GaussianBlur(values, (kernel, kernel), float(sigma)), 0.0, 1.0)


def _morph_mask(mask: np.ndarray, *, radius: int, operation: str) -> np.ndarray:
    if radius <= 0 or not bool(mask.any()):
        return mask.astype(bool)
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    values = mask.astype(np.uint8)
    if operation == "dilate":
        return cv2.dilate(values, kernel, iterations=1).astype(bool)
    if operation == "erode":
        return cv2.erode(values, kernel, iterations=1).astype(bool)
    raise ValueError(f"unknown morphology operation: {operation}")


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    if not bool(mask.any()):
        return mask.astype(bool)
    values = mask.astype(np.uint8)
    h, w = values.shape
    flood = values.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = flood == 0
    return (values.astype(bool) | holes).astype(bool)


def _keep_large_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros(mask.shape, dtype=bool)
    for label_idx in range(1, n_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= int(min_area):
            kept |= labels == label_idx
    return kept


def _largest_component(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    if int(stats[largest_idx, cv2.CC_STAT_AREA]) < int(min_area):
        return np.zeros(mask.shape, dtype=bool)
    return labels == largest_idx


def _closed_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0 or not bool(mask.any()):
        return mask.astype(bool)
    return _morph_mask(_morph_mask(mask, radius=radius, operation="dilate"), radius=radius, operation="erode")


def _control_outline_from_internal_candidate(
    internal: np.ndarray,
    *,
    limit: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """Return the domain where internal transparency candidates may steer hint.

    This is deliberately more conservative than the subject outline. It is
    derived from enclosed near-screen evidence and clipped to the known subject
    interior, so possibly-opaque outer edges remain a CorridorKey solve.
    """

    if not bool(internal.any()):
        return np.zeros(internal.shape, dtype=bool)
    h, w = internal.shape
    close_radius = int(np.clip(round(min(h, w) * 0.047), 8, 40))
    inset_radius = int(np.clip(round(min(h, w) * 0.004), 1, 4))
    closed = _closed_mask(internal, radius=close_radius)
    closed = _fill_holes(closed)
    outline = _largest_component(closed & limit.astype(bool), min_area=min_area)
    if not bool(outline.any()):
        return np.zeros(internal.shape, dtype=bool)
    outline = _morph_mask(outline, radius=inset_radius, operation="erode")
    return _largest_component(outline & limit.astype(bool), min_area=min_area)


def _bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _expand_bbox(bbox: list[int], *, pad: int, shape: tuple[int, int]) -> list[int]:
    h, w = shape
    x0, y0, x1, y1 = bbox
    return [
        int(max(0, x0 - pad)),
        int(max(0, y0 - pad)),
        int(min(w, x1 + pad)),
        int(min(h, y1 + pad)),
    ]


def _bbox_mask(bbox: list[int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = bbox
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _build_soft_key_hint(
    features: CorridorKeyHintFeatures,
    *,
    expand_px: int,
    blur_sigma: float,
    unknown_ring_value: float,
    boundary_floor: float,
    internal_floor: float,
    hard_floor: float,
) -> np.ndarray:
    """Build a coarse soft outline hint from deterministic chroma-key evidence.

    CorridorKey should solve edge detail itself. The hint therefore supplies a
    clear subject outline with a soft transition, while internal transparency
    controls are allowed only inside the already-established outline.
    """

    control_outline = features.control_outline_mask
    if not bool(control_outline.any()):
        return np.zeros_like(features.key_alpha, dtype=np.float32)

    expanded = _morph_mask(control_outline, radius=expand_px, operation="dilate")
    inner = _morph_mask(control_outline, radius=max(1, expand_px + 2), operation="erode")
    outline_soft = _smooth_mask(expanded, sigma=blur_sigma)
    hint = np.clip(outline_soft * float(boundary_floor), 0.0, 1.0).astype(np.float32)
    hint[inner] = np.maximum(hint[inner], float(hard_floor))
    edge_band = expanded & ~inner
    hint[edge_band] = np.maximum(hint[edge_band], outline_soft[edge_band] * float(unknown_ring_value))

    if bool(features.internal_transparency_candidate.any()):
        internal_region = features.internal_transparency_candidate & features.control_outline_inner_mask
        internal_soft = _smooth_mask(
            internal_region,
            sigma=max(2.5, blur_sigma * 1.4),
        )
        internal_core = _morph_mask(internal_region, radius=3, operation="erode")
        if bool(internal_core.any()):
            internal_soft = np.maximum(internal_soft, _smooth_mask(internal_core, sigma=1.5))
        hint = np.maximum(hint, internal_soft * float(internal_floor))
    if bool(features.soft_boundary_candidate.any()):
        boundary_soft = _smooth_mask(
            features.soft_boundary_candidate & expanded,
            sigma=max(3.0, blur_sigma * 2.5),
        )
        hint = np.maximum(hint, boundary_soft * float(boundary_floor))

    # Internal controls must never leak beyond the explicit outline envelope.
    hint[~expanded] = 0.0
    return np.clip(hint, 0.0, 1.0).astype(np.float32)


def detect_corridorkey_hint_features(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    thresholds: KeyerThresholds | None = None,
    min_area_ratio: float = 0.0008,
) -> CorridorKeyHintFeatures:
    """Detect feature regions that can drive CorridorKey hint variants.

    The detector is intentionally position-agnostic: it does not assume that
    translucent material is central. Regions are described by key-alpha strength,
    support topology, and distance to exterior background.
    """

    if image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must have shape HxWx3")
    thresholds = thresholds or KeyerThresholds(bg_max=5.5, fg_min=18.0)
    key = chromatic_key_alpha(image_srgb, background_color, thresholds).astype(np.float32)
    area = int(key.size)
    min_area = max(16, int(round(area * float(min_area_ratio))))

    subject_support = _keep_large_components(key >= 0.08, min_area=min_area)
    outline_seed = _keep_large_components(key >= 0.06, min_area=min_area)
    outline_mask = _morph_mask(outline_seed, radius=10, operation="dilate")
    outline_mask = _morph_mask(outline_mask, radius=10, operation="erode")
    outline_mask = _fill_holes(_keep_large_components(outline_mask, min_area=min_area))
    outline_inner_mask = _morph_mask(outline_mask, radius=4, operation="erode")
    hard_subject = _keep_large_components(key >= 0.78, min_area=min_area)
    ambiguous = (key >= 0.12) & (key <= 0.78) & outline_mask
    distance_to_exterior = cv2.distanceTransform(outline_mask.astype(np.uint8), cv2.DIST_L2, 3)
    soft_boundary = ambiguous & (distance_to_exterior <= 18.0)
    translucent = ambiguous & (distance_to_exterior > 4.0)
    internal_transparency = ambiguous & (distance_to_exterior > 18.0)
    translucent = _keep_large_components(translucent, min_area=min_area)
    internal_transparency = _keep_large_components(internal_transparency, min_area=min_area)
    soft_boundary = _keep_large_components(soft_boundary, min_area=min_area)
    control_outline = _control_outline_from_internal_candidate(
        internal_transparency,
        limit=outline_inner_mask,
        min_area=min_area,
    )
    control_outline_inner = _morph_mask(control_outline, radius=4, operation="erode")
    background = key <= 0.03
    bbox = _bbox_xyxy(subject_support)
    bbox_plus_2 = _expand_bbox(bbox, pad=2, shape=key.shape)
    bbox_plus_2_mask = _bbox_mask(bbox_plus_2, key.shape)

    metadata = {
        "schema": "ermbg.corridorkey_hint_features.v1",
        "background_color": [int(c) for c in background_color],
        "thresholds": {"bg_max": float(thresholds.bg_max), "fg_min": float(thresholds.fg_min)},
        "min_area_px": int(min_area),
        "pixels": {
            "subject_support": int(subject_support.sum()),
            "outline_mask": int(outline_mask.sum()),
            "outline_inner_mask": int(outline_inner_mask.sum()),
            "control_outline_mask": int(control_outline.sum()),
            "control_outline_inner_mask": int(control_outline_inner.sum()),
            "hard_subject": int(hard_subject.sum()),
            "translucent_candidate": int(translucent.sum()),
            "internal_transparency_candidate": int(internal_transparency.sum()),
            "soft_boundary_candidate": int(soft_boundary.sum()),
            "background": int(background.sum()),
            "bbox_plus_2": int(bbox_plus_2_mask.sum()),
        },
        "bbox_xyxy": bbox,
        "bbox_plus_2_xyxy": bbox_plus_2,
        "key_alpha": {
            "min": float(key.min()) if key.size else 0.0,
            "max": float(key.max()) if key.size else 0.0,
            "mean": float(key.mean()) if key.size else 0.0,
        },
    }
    return CorridorKeyHintFeatures(
        key_alpha=key,
        outline_mask=outline_mask,
        outline_inner_mask=outline_inner_mask,
        control_outline_mask=control_outline,
        control_outline_inner_mask=control_outline_inner,
        subject_support=subject_support,
        hard_subject=hard_subject,
        translucent_candidate=translucent,
        internal_transparency_candidate=internal_transparency,
        soft_boundary_candidate=soft_boundary,
        background=background,
        bbox_xyxy=bbox,
        bbox_plus_2_xyxy=bbox_plus_2,
        bbox_plus_2_mask=bbox_plus_2_mask,
        metadata=metadata,
    )


def build_corridorkey_hint_plan(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    variant: CorridorKeyHintVariant,
) -> CorridorKeyHintPlan:
    features = detect_corridorkey_hint_features(image_srgb, background_color)
    shape = image_srgb.shape[:2]
    if variant == "current_default_prior":
        hint = np.full(shape, 0.32, dtype=np.float32)
        policy = {
            "hard_subject": 0.32,
            "subject_support": 0.32,
            "translucent_candidate": 0.32,
            "soft_boundary_candidate": 0.32,
            "outside_bbox_plus_2": 0.32,
            "note": "emulates current corridorkey-character full-frame soft prior",
        }
    elif variant == "full_frame_zero":
        hint = np.zeros(shape, dtype=np.float32)
        policy = {
            "diagnostic": True,
            "hard_subject": 0.0,
            "subject_support": 0.0,
            "translucent_candidate": 0.0,
            "soft_boundary_candidate": 0.0,
            "outside_bbox_plus_2": 0.0,
            "note": "full-frame zero CorridorKey hint is a diagnostic, not a candidate",
        }
    else:
        if variant == "feature_conservative":
            expand_px = 4
            blur_sigma = 4.0
            unknown_ring_value = 0.58
            boundary_floor = 0.30
            internal_floor = 0.66
            hard_floor = 0.58
        elif variant == "feature_balanced":
            expand_px = 3
            blur_sigma = 3.5
            unknown_ring_value = 0.46
            boundary_floor = 0.22
            internal_floor = 0.48
            hard_floor = 0.42
        elif variant == "feature_internal_opaque":
            expand_px = 3
            blur_sigma = 3.5
            unknown_ring_value = 0.64
            boundary_floor = 0.34
            internal_floor = 0.96
            hard_floor = 0.88
        elif variant == "feature_translucent":
            expand_px = 2
            blur_sigma = 3.0
            unknown_ring_value = 0.34
            boundary_floor = 0.16
            internal_floor = 0.30
            hard_floor = 0.26
        else:
            raise ValueError(f"unknown CorridorKey hint variant: {variant}")
        hint = _build_soft_key_hint(
            features,
            expand_px=expand_px,
            blur_sigma=blur_sigma,
            unknown_ring_value=unknown_ring_value,
            boundary_floor=boundary_floor,
            internal_floor=internal_floor,
            hard_floor=hard_floor,
        )
        policy = {
            "base": "soft_control_outline",
            "support_expand_px": expand_px,
            "soft_blur_sigma": blur_sigma,
            "unknown_ring_value": unknown_ring_value,
            "outline_core_floor": hard_floor,
            "internal_transparency_candidate_floor": internal_floor,
            "soft_boundary_candidate_floor": boundary_floor,
            "control_outline": "enclosed_internal_candidate_clipped_to_subject_interior",
            "outside_expanded_control_outline": 0.0,
        }

    metadata = {
        "schema": "ermbg.corridorkey_hint_plan.v1",
        "variant": variant,
        "policy": policy,
        "hint": {
            "min": float(hint.min()) if hint.size else 0.0,
            "max": float(hint.max()) if hint.size else 0.0,
            "mean": float(hint.mean()) if hint.size else 0.0,
            "nonzero_pixels": int((hint > 0.001).sum()),
        },
        "features": features.metadata,
    }
    return CorridorKeyHintPlan(variant=variant, hint=hint, features=features, metadata=metadata)


def corridorkey_hint_variants() -> tuple[CorridorKeyHintVariant, ...]:
    return (
        "current_default_prior",
        "feature_balanced",
        "feature_conservative",
        "feature_internal_opaque",
        "feature_translucent",
    )


def corridorkey_hint_diagnostic_variants() -> tuple[CorridorKeyHintVariant, ...]:
    return (
        "full_frame_zero",
    )


__all__ = [
    "CorridorKeyHintFeatures",
    "CorridorKeyHintPlan",
    "CorridorKeyHintVariant",
    "build_corridorkey_hint_plan",
    "corridorkey_full_frame_prior_value",
    "corridorkey_hint_diagnostic_variants",
    "corridorkey_hint_variants",
    "detect_corridorkey_hint_features",
]
