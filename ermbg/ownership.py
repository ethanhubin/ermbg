"""Local ownership hypotheses for known-background evidence regions.

This module is intentionally VLM-free. It scores several deterministic
interpretations for each local evidence region so the pipeline can prefer
measurable explanations before asking any semantic model to break ties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from . import io
from .colorspace import oklab_distance, srgb_to_oklab
from .planner import RiskRegion

OwnershipRole = Literal[
    "hole",
    "opaque_subject",
    "subject_soft_layer",
    "shadow_like_layer",
    "conservative_unknown",
]


@dataclass(frozen=True)
class OwnershipSignals:
    """Measurable local signals used to rank ownership hypotheses."""

    area: int
    alpha_mean: float
    alpha_mid_fraction: float
    alpha_low_fraction: float
    alpha_high_fraction: float
    bg_distance_mean: float
    bg_distance_p25: float
    saturation_mean: float
    scalar_shadow_error_mean: float
    scalar_shadow_strength_mean: float
    scalar_shadow_like_fraction: float
    near_subject_fraction: float
    exterior_fraction: float
    touches_border: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "alpha_mean": self.alpha_mean,
            "alpha_mid_fraction": self.alpha_mid_fraction,
            "alpha_low_fraction": self.alpha_low_fraction,
            "alpha_high_fraction": self.alpha_high_fraction,
            "bg_distance_mean": self.bg_distance_mean,
            "bg_distance_p25": self.bg_distance_p25,
            "saturation_mean": self.saturation_mean,
            "scalar_shadow_error_mean": self.scalar_shadow_error_mean,
            "scalar_shadow_strength_mean": self.scalar_shadow_strength_mean,
            "scalar_shadow_like_fraction": self.scalar_shadow_like_fraction,
            "near_subject_fraction": self.near_subject_fraction,
            "exterior_fraction": self.exterior_fraction,
            "touches_border": self.touches_border,
        }


@dataclass(frozen=True)
class OwnershipCandidate:
    """One local interpretation of an evidence region."""

    region_id: str
    region_kind: str
    role: OwnershipRole
    score: float
    confidence: float
    operations: tuple[str, ...]
    reason: str
    signals: OwnershipSignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_kind": self.region_kind,
            "role": self.role,
            "score": self.score,
            "confidence": self.confidence,
            "operations": list(self.operations),
            "reason": self.reason,
            "signals": self.signals.to_dict(),
        }


def rank_region_ownership(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    region: RiskRegion,
) -> list[OwnershipCandidate]:
    """Return ranked local ownership interpretations for one evidence region."""
    signals = measure_ownership_signals(image_srgb, base_rgba, background_color, region.mask)
    kind = str(region.kind)
    role_scores = _score_roles(kind, signals)
    candidates = [
        _candidate_from_role(region, role, score, signals)
        for role, score in role_scores.items()
        if score > 0.0
    ]
    candidates.sort(key=lambda item: (-item.score, item.role))
    return candidates


def rank_regions_ownership(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    regions: list[RiskRegion],
) -> list[dict[str, Any]]:
    """Rank ownership candidates for many evidence regions."""
    rows: list[dict[str, Any]] = []
    for region in regions:
        candidates = rank_region_ownership(image_srgb, base_rgba, background_color, region)
        rows.append(
            {
                "region": region.to_prompt_dict(),
                "candidates": [candidate.to_dict() for candidate in candidates],
                "selected": candidates[0].to_dict() if candidates else None,
            }
        )
    return rows


def ownership_masks(
    ownership_rows: list[dict[str, Any]],
    shape: tuple[int, int],
    *,
    min_confidence: float = 0.45,
) -> dict[str, np.ndarray]:
    """Aggregate selected ownership rows into boolean role masks.

    This converts local region interpretations into execution masks. The
    confidence floor is deliberately modest because the masks are protective:
    they mostly prevent destructive repairs, and later physical estimators still
    decide actual opacity/color.
    """
    masks = {
        "hole": np.zeros(shape, dtype=bool),
        "opaque_subject": np.zeros(shape, dtype=bool),
        "subject_soft_layer": np.zeros(shape, dtype=bool),
        "shadow_like_layer": np.zeros(shape, dtype=bool),
        "conservative_unknown": np.zeros(shape, dtype=bool),
    }
    for item in ownership_rows:
        if not isinstance(item, dict):
            continue
        selected = item.get("selected")
        region = item.get("region")
        if not isinstance(selected, dict) or not isinstance(region, dict):
            continue
        role = selected.get("role")
        if role not in masks:
            continue
        try:
            confidence = float(selected.get("confidence", selected.get("score", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue
        bbox = region.get("bbox_xyxy")
        # The report JSON is intentionally mask-free, so this helper is mainly
        # useful when callers pass enriched rows with a private `_mask` field.
        mask = item.get("_mask")
        if isinstance(mask, np.ndarray):
            if mask.shape != shape:
                raise ValueError("ownership row _mask must match shape")
            masks[role] |= mask.astype(bool)
        elif isinstance(bbox, list) and len(bbox) == 4:
            # BBox fallback is coarse but safe for protective execution masks.
            x0, y0, x1, y1 = (int(v) for v in bbox)
            masks[role][max(0, y0) : min(shape[0], y1), max(0, x0) : min(shape[1], x1)] = True
    return masks


def resolve_execution_masks(
    masks: dict[str, np.ndarray],
    shape: tuple[int, int],
    *,
    subject_soft_min_area_ratio: float = 0.0005,
    subject_soft_min_pixels: int = 512,
    shadow_fragment_vs_soft_ratio: float = 0.08,
) -> dict[str, np.ndarray]:
    """Apply global arbitration before local roles become execution masks.

    Local candidate scoring is intentionally permissive so it can surface weak
    evidence. Execution masks need a higher bar: tiny soft-material speckles
    should not protect a shadow sample, and scalar-looking fragments inside a
    dominant glow/glass layer should not reopen broad shadow recovery.
    """
    h, w = shape
    min_soft_area = max(int(subject_soft_min_pixels), int(round(h * w * subject_soft_min_area_ratio)))
    resolved = {
        role: np.asarray(mask, dtype=bool).copy()
        for role, mask in masks.items()
    }
    soft = resolved.get("subject_soft_layer")
    shadow = resolved.get("shadow_like_layer")
    if soft is None or shadow is None:
        return resolved

    soft_area = int(soft.sum())
    shadow_area = int(shadow.sum())
    # The minimum is empirical but feature-based: on megapixel-ish assets it
    # removes isolated candidate noise, while preserving coherent glass/glow
    # and smoke layers large enough to be visually meaningful.
    if soft_area < min_soft_area:
        soft.fill(False)
        soft_area = 0

    # Glow/glass often contains small scalar-darkening islands from antialiasing
    # or color recovery. When a coherent soft layer dominates the explanation,
    # keep those islands under material ownership unless shadow evidence forms
    # a comparable separate layer.
    if soft_area > 0 and 0 < shadow_area < soft_area * float(shadow_fragment_vs_soft_ratio):
        shadow.fill(False)
    return resolved


def measure_ownership_signals(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    mask: np.ndarray,
) -> OwnershipSignals:
    """Measure local mixture/topology signals for a boolean region mask."""
    if image_srgb.dtype != np.uint8:
        raise ValueError("image_srgb must be uint8")
    if image_srgb.shape[:2] != base_rgba.shape[:2]:
        raise ValueError("image_srgb and base_rgba must share HxW")
    if mask.shape != image_srgb.shape[:2]:
        raise ValueError("mask must share HxW with image_srgb")

    region = np.asarray(mask, dtype=bool)
    if not region.any():
        return OwnershipSignals(
            area=0,
            alpha_mean=0.0,
            alpha_mid_fraction=0.0,
            alpha_low_fraction=0.0,
            alpha_high_fraction=0.0,
            bg_distance_mean=0.0,
            bg_distance_p25=0.0,
            saturation_mean=0.0,
            scalar_shadow_error_mean=1.0,
            scalar_shadow_strength_mean=0.0,
            scalar_shadow_like_fraction=0.0,
            near_subject_fraction=0.0,
            exterior_fraction=0.0,
            touches_border=False,
        )

    alpha = base_rgba[..., 3].astype(np.float32) / 255.0
    a = alpha[region]
    rgb = image_srgb.astype(np.float32) / 255.0
    saturation = rgb.max(axis=2) - rgb.min(axis=2)

    lab = srgb_to_oklab(image_srgb)
    bg = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg_lab = srgb_to_oklab(bg).reshape(3)
    bg_distance = oklab_distance(lab, bg_lab).astype(np.float32)

    B_lin = io.srgb_to_linear(bg)[0, 0].astype(np.float32)
    C_lin = io.srgb_to_linear(image_srgb).astype(np.float32)
    denom = float(np.dot(B_lin, B_lin))
    if denom > 1e-6:
        scale = np.tensordot(C_lin, B_lin, axes=([-1], [0])) / denom
        recon = scale[..., None] * B_lin
        scalar_err = np.sqrt(np.mean((C_lin - recon) * (C_lin - recon), axis=-1))
        scalar_strength = np.clip(1.0 - scale, 0.0, 1.0).astype(np.float32)
    else:
        scalar_err = np.ones(alpha.shape, dtype=np.float32)
        scalar_strength = np.zeros(alpha.shape, dtype=np.float32)

    subject_anchor = alpha >= 0.70
    near_subject = _near_mask(subject_anchor, radius_px=10)
    exterior = alpha <= 0.75
    scalar_like = (scalar_strength >= 0.02) & (scalar_err <= 0.070)

    return OwnershipSignals(
        area=int(region.sum()),
        alpha_mean=float(a.mean()),
        alpha_mid_fraction=float(((a >= 0.05) & (a <= 0.95)).mean()),
        alpha_low_fraction=float((a <= 0.20).mean()),
        alpha_high_fraction=float((a >= 0.85).mean()),
        bg_distance_mean=float(bg_distance[region].mean()),
        bg_distance_p25=float(np.percentile(bg_distance[region], 25.0)),
        saturation_mean=float(saturation[region].mean()),
        scalar_shadow_error_mean=float(scalar_err[region].mean()),
        scalar_shadow_strength_mean=float(scalar_strength[region].mean()),
        scalar_shadow_like_fraction=float(scalar_like[region].mean()),
        near_subject_fraction=float(near_subject[region].mean()),
        exterior_fraction=float(exterior[region].mean()),
        touches_border=_touches_border(region),
    )


def _score_roles(kind: str, s: OwnershipSignals) -> dict[OwnershipRole, float]:
    scalar_shadow = s.scalar_shadow_like_fraction
    chroma_soft = s.alpha_mid_fraction * (1.0 - scalar_shadow)
    near_subject = s.near_subject_fraction
    enclosed_bonus = 0.15 if not s.touches_border else -0.15

    hole = (
        0.55 * s.alpha_low_fraction
        + 0.25 * _inv_smoothstep(3.0, 14.0, s.bg_distance_p25)
        + (0.25 if kind == "same_bg_enclosed_region" else 0.0)
        + enclosed_bonus
    )
    soft_subject = (
        0.50 * chroma_soft
        + 0.25 * near_subject
        + 0.15 * _smoothstep(0.05, 0.16, s.saturation_mean)
        + (0.30 if kind in {"translucent_candidate", "glow_soft_alpha_candidate"} else 0.0)
        - 0.30 * scalar_shadow
    )
    shadow = (
        0.60 * scalar_shadow
        + 0.20 * s.exterior_fraction
        + 0.20 * _smoothstep(0.04, 0.18, s.scalar_shadow_strength_mean)
        + (0.35 if kind == "owned_shadow_candidate" else 0.0)
        - 0.25 * (1.0 - near_subject)
    )
    opaque = (
        0.45 * s.alpha_high_fraction
        + (0.35 if kind == "hard_edge_candidate" else 0.0)
        + (0.25 if kind == "alpha_keyer_disagreement" else 0.0)
        - 0.35 * s.alpha_mid_fraction
    )
    conservative = 0.25 + 0.20 * min(chroma_soft, scalar_shadow)

    scores: dict[OwnershipRole, float] = {
        "hole": float(np.clip(hole, 0.0, 1.0)),
        "opaque_subject": float(np.clip(opaque, 0.0, 1.0)),
        "subject_soft_layer": float(np.clip(soft_subject, 0.0, 1.0)),
        "shadow_like_layer": float(np.clip(shadow, 0.0, 1.0)),
        "conservative_unknown": float(np.clip(conservative, 0.0, 1.0)),
    }
    return scores


def _candidate_from_role(
    region: RiskRegion,
    role: OwnershipRole,
    score: float,
    signals: OwnershipSignals,
) -> OwnershipCandidate:
    operations_by_role: dict[OwnershipRole, tuple[str, ...]] = {
        "hole": ("preserve_hole",),
        "opaque_subject": (
            ("snap_hard_edge",)
            if region.kind == "hard_edge_candidate"
            else ("repair_opaque_interior",)
        ),
        "subject_soft_layer": ("preserve_soft_alpha", "mark_translucent"),
        "shadow_like_layer": ("preserve_shadow_layer",),
        "conservative_unknown": ("preserve_current_alpha",),
    }
    reasons: dict[OwnershipRole, str] = {
        "hole": "Low-alpha region close to the known background.",
        "opaque_subject": "Local key/edge evidence supports raising alpha.",
        "subject_soft_layer": "Mid-alpha, non-scalar material/glow signal near subject support.",
        "shadow_like_layer": "Region is well explained by scalar darkening of the known background.",
        "conservative_unknown": "Ambiguous evidence; keep current alpha rather than hard repair.",
    }
    return OwnershipCandidate(
        region_id=region.id,
        region_kind=str(region.kind),
        role=role,
        score=float(score),
        confidence=float(np.clip(score, 0.0, 1.0)),
        operations=operations_by_role[role],
        reason=reasons[role],
        signals=signals,
    )


def _near_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    ksize = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = float(np.clip((float(value) - edge0) / (edge1 - edge0), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _inv_smoothstep(edge0: float, edge1: float, value: float) -> float:
    return 1.0 - _smoothstep(edge0, edge1, value)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


__all__ = [
    "OwnershipCandidate",
    "OwnershipRole",
    "OwnershipSignals",
    "measure_ownership_signals",
    "ownership_masks",
    "rank_region_ownership",
    "rank_regions_ownership",
    "resolve_execution_masks",
]
