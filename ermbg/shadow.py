"""Contact/drop shadow detection for known-background assets.

The main matte alpha represents subject ownership. A cast shadow is different:
it is usually the known background darkened by a scalar factor. For solid-color
backgrounds we can recover a conservative shadow matte from

    C_linear ~= s * B_linear,  shadow_alpha ~= 1 - s

and keep it as a separate layer before compositing it into the final RGBA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from . import io

DISPLAY_SAFE_SHADOW_OPACITY_SCALE = 0.75


@dataclass(frozen=True)
class ShadowThresholds:
    """Tunable gates for known-background shadow extraction."""

    min_strength: float = 0.08
    loose_min_strength: float = 0.002
    max_strength: float = 0.92
    max_reconstruction_error: float = 0.070
    loose_error_multiplier: float = 1.45
    subject_alpha_max: float = 0.75
    fg_anchor_alpha: float = 0.50
    max_distance_ratio: float = 0.14
    max_distance_px: int = 180
    loose_distance_multiplier: float = 1.85
    min_component_area_ratio: float = 0.00045
    max_component_area_ratio: float = 0.22
    min_total_area_ratio: float = 0.0025
    boundary_falloff_px: float = 28.0
    hard_boundary_falloff_px: float = 2.0
    hard_boundary_alpha_min: float = 0.16
    hard_boundary_ratio_min: float = 0.40
    # Hard-shadow rescue thresholds are intentionally feature-based, not
    # sample-specific. They cover the case where strict scalar-darkening seeds
    # contain a clear high-alpha platform, but the loose support grows into a
    # broad weak tail or non-pure background noise; in that case a large soft
    # falloff erases the platform edge. These empirical gates require a strong
    # seed plateau, a still-dark seed boundary, and significant support
    # expansion before overriding the default soft-boundary decision.
    hard_plateau_alpha_min: float = 0.30
    hard_plateau_fraction_min: float = 0.10
    hard_seed_alpha_p75_min: float = 0.45
    hard_support_expansion_min: float = 2.50
    field_blur_sigma: float = 5.0
    contact_distance_ratio: float = 0.035
    contact_distance_px: int = 42
    contact_blur_sigma: float = 1.25
    contact_outer_feather_px: float = 10.0
    subject_occlusion_blur_sigma: float = 2.0
    reject_border_components: bool = True


@dataclass(frozen=True)
class ShadowPrior:
    """Semantic constraints supplied before pixel-level shadow extraction.

    VLM/planner output should land here as broad masks or regions. It does not
    set shadow opacity; the measured known-background darkening still does that.
    """

    subject_mask: np.ndarray | None = None
    shadow_search_mask: np.ndarray | None = None
    shadow_ownership_mask: np.ndarray | None = None
    shadow_allowed: bool = True
    source: str = ""


def estimate_shadow_alpha(
    image_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    thresholds: ShadowThresholds | None = None,
    prior: ShadowPrior | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a conservative shadow alpha matte.

    Returns ``(shadow_alpha, info)`` where ``shadow_alpha`` is HxW float32 in
    [0, 1]. If the image does not contain enough coherent shadow evidence, the
    alpha is all zeros and ``info["detected"]`` is false.
    """
    t = thresholds or ShadowThresholds()
    if image_srgb.dtype != np.uint8:
        raise ValueError("estimate_shadow_alpha expects sRGB uint8 image")
    if image_srgb.shape[:2] != subject_alpha.shape:
        raise ValueError("image_srgb and subject_alpha must share HxW")

    h, w = subject_alpha.shape
    img_area = float(h * w)
    alpha = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    if prior is not None and not prior.shadow_allowed:
        out = np.zeros((h, w), dtype=np.float32)
        return out, _shadow_info(out, [], 0, "shadow disallowed by semantic prior", prior=prior)

    subject_prior = _coerce_optional_mask(
        prior.subject_mask if prior is not None else None,
        (h, w),
        "ShadowPrior.subject_mask",
    )
    shadow_search = _coerce_optional_mask(
        prior.shadow_search_mask if prior is not None else None,
        (h, w),
        "ShadowPrior.shadow_search_mask",
    )
    shadow_ownership = _coerce_optional_mask(
        prior.shadow_ownership_mask if prior is not None else None,
        (h, w),
        "ShadowPrior.shadow_ownership_mask",
    )
    subject_ownership = np.maximum(alpha, subject_prior) if subject_prior is not None else alpha

    bg = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    B = io.srgb_to_linear(bg)[0, 0].astype(np.float32)
    denom = float(np.dot(B, B))
    if denom < 1e-6:
        out = np.zeros((h, w), dtype=np.float32)
        return out, _shadow_info(out, [], 0, "background too dark for scalar shadow model", prior=prior)

    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    scale = np.tensordot(C, B, axes=([-1], [0])) / denom
    recon = scale[..., None] * B
    err = np.sqrt(np.mean((C - recon) * (C - recon), axis=-1))
    strength = np.clip(1.0 - scale, 0.0, 1.0).astype(np.float32)

    fg = subject_ownership >= float(t.fg_anchor_alpha)
    if not fg.any():
        out = np.zeros((h, w), dtype=np.float32)
        return out, _shadow_info(out, [], 0, "no foreground anchor", prior=prior)

    inv_fg = (~fg).astype(np.uint8)
    dist = cv2.distanceTransform(inv_fg, cv2.DIST_L2, 3)
    max_dist = min(float(t.max_distance_px), float(t.max_distance_ratio) * min(h, w))
    near_subject = dist <= max_dist
    loose_near_subject = dist <= max_dist * float(t.loose_distance_multiplier)
    exterior = subject_ownership <= float(t.subject_alpha_max)

    if shadow_search is not None or shadow_ownership is not None:
        search_seed = np.zeros((h, w), dtype=bool)
        if shadow_search is not None:
            search_seed |= shadow_search > 0.0
        if shadow_ownership is not None:
            search_seed |= shadow_ownership > 0.0
        search_domain = search_seed
        loose_search_domain = _dilate_mask(
            search_seed,
            max(1, int(round(max_dist * 0.35))),
        )
    else:
        search_domain = near_subject
        loose_search_domain = loose_near_subject

    candidate = (
        exterior
        & search_domain
        & (strength >= float(t.min_strength))
        & (strength <= float(t.max_strength))
        & (err <= float(t.max_reconstruction_error))
    )
    soft_support = (
        exterior
        & loose_search_domain
        & (strength >= float(t.loose_min_strength))
        & (strength <= float(t.max_strength))
        & (err <= float(t.max_reconstruction_error) * float(t.loose_error_multiplier))
    )

    min_area = max(8.0, float(t.min_component_area_ratio) * img_area)
    max_area = max(min_area, float(t.max_component_area_ratio) * img_area)
    accepted, rejected = _filter_components(
        candidate,
        min_area=min_area,
        max_area=max_area,
        reject_border=t.reject_border_components,
    )

    shadow_alpha, boundary_info = _soft_shadow_alpha_from_seeds(
        accepted,
        soft_support,
        strength,
        err,
        subject_ownership,
        dist,
        thresholds=t,
    )

    min_total = max(8.0, float(t.min_total_area_ratio) * img_area)
    if float((shadow_alpha > 0).sum()) < min_total:
        shadow_alpha.fill(0.0)
        rejected += len(accepted)
        accepted = []
        reason = "below minimum total shadow area"
    else:
        reason = ""

    return shadow_alpha, _shadow_info(
        shadow_alpha,
        accepted,
        rejected,
        reason,
        prior=prior,
        boundary_info=boundary_info,
    )


def shadow_prior_from_regions(
    regions: list[Any],
    shape: tuple[int, int],
    *,
    source: str = "planner",
    shadow_allowed: bool = True,
) -> ShadowPrior:
    """Build a ``ShadowPrior`` from planner/VLM EvidenceRegion objects.

    Regions are semantic constraints only. The resulting prior narrows where
    scalar-darkening evidence may be interpreted as an owned shadow.
    """
    h, w = int(shape[0]), int(shape[1])
    subject = np.zeros((h, w), dtype=np.float32)
    search = np.zeros((h, w), dtype=np.float32)
    ownership = np.zeros((h, w), dtype=np.float32)
    seen_subject = seen_search = seen_ownership = False
    for region in regions:
        kind = str(getattr(region, "kind", ""))
        mask = _coerce_optional_mask(getattr(region, "mask", None), (h, w), f"{kind}.mask")
        if mask is None:
            continue
        if kind in {"subject_owned_region", "subject_region", "owned_region"}:
            subject = np.maximum(subject, mask)
            seen_subject = True
        elif kind in {"shadow_search_region", "shadow_search"}:
            search = np.maximum(search, mask)
            seen_search = True
        elif kind in {"owned_shadow_candidate", "shadow_or_contact"}:
            ownership = np.maximum(ownership, mask)
            seen_ownership = True

    return ShadowPrior(
        subject_mask=subject if seen_subject else None,
        shadow_search_mask=search if seen_search else None,
        shadow_ownership_mask=ownership if seen_ownership else None,
        shadow_allowed=shadow_allowed,
        source=source,
    )


def shadow_alpha_to_display_alpha(
    shadow_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    opacity_scale: float = 1.0,
) -> np.ndarray:
    """Map linear scalar-darkening strength to black-alpha for sRGB viewers.

    ``estimate_shadow_alpha`` measures known-background darkening in linear RGB:
    ``C_linear ~= (1 - strength) * B_linear``. A PNG viewer normally composites
    black RGBA in sRGB space, where using ``strength`` directly as alpha makes
    the shadow look too dark. This conversion chooses the black alpha that best
    reproduces the same darkened known background under sRGB alpha compositing.

    Keep this as an export/compositing conversion only. The physical shadow
    alpha is still the right representation for detection, component filtering,
    and VLM shadow-candidate evidence.
    """
    alpha = np.clip(shadow_alpha.astype(np.float32), 0.0, 1.0)
    bg_u8 = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg_srgb = bg_u8.astype(np.float32) / 255.0
    weights = bg_srgb * bg_srgb
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6:
        return alpha

    bg_linear = io.srgb_to_linear(bg_u8)[0, 0].astype(np.float32).reshape(1, 1, 3)
    shadowed_linear = (1.0 - alpha[..., None]) * bg_linear
    shadowed_srgb = io.linear_to_srgb(shadowed_linear)
    channel_alpha = 1.0 - shadowed_srgb / np.maximum(bg_srgb, 1e-6)
    display_alpha = (channel_alpha * weights).sum(axis=-1) / weight_sum
    return np.clip(display_alpha * float(opacity_scale), 0.0, 1.0).astype(np.float32)


def composite_subject_with_shadow(
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
    shadow_alpha: np.ndarray,
    shadow_color_linear: tuple[float, float, float] = (0.0, 0.0, 0.0),
    subject_occlusion_blur_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite a black shadow layer behind the subject on transparent bg."""
    fg = foreground_linear.astype(np.float32)
    a_subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    if subject_occlusion_blur_sigma > 0.0:
        occluder = cv2.GaussianBlur(
            a_subject,
            (0, 0),
            sigmaX=float(subject_occlusion_blur_sigma),
            sigmaY=float(subject_occlusion_blur_sigma),
        )
        occluder = np.maximum(a_subject, np.clip(occluder, 0.0, 1.0))
    else:
        occluder = a_subject
    a_shadow = np.clip(shadow_alpha.astype(np.float32), 0.0, 1.0) * (1.0 - occluder)
    a_out = np.clip(a_subject + a_shadow, 0.0, 1.0)

    shadow_color = np.asarray(shadow_color_linear, dtype=np.float32).reshape(1, 1, 3)
    premul = fg * a_subject[..., None] + shadow_color * a_shadow[..., None]
    out_fg = fg.copy()
    nonzero = a_out > 1e-6
    out_fg[nonzero] = premul[nonzero] / a_out[nonzero, None]
    out_fg[~nonzero] = 0.0
    return a_out.astype(np.float32), np.clip(out_fg, 0.0, 1.0).astype(np.float32)


def _filter_components(
    candidate: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    reject_border: bool,
) -> tuple[list[np.ndarray], int]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        connectivity=8,
    )
    accepted: list[np.ndarray] = []
    rejected = 0
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        comp = labels == label_idx
        if area < min_area or area > max_area:
            rejected += 1
            continue
        if reject_border and _touches_border(comp):
            rejected += 1
            continue
        accepted.append(comp)
    return accepted, rejected


def _soft_shadow_alpha_from_seeds(
    seeds: list[np.ndarray],
    soft_support: np.ndarray,
    strength: np.ndarray,
    err: np.ndarray,
    subject_alpha: np.ndarray,
    dist_to_subject: np.ndarray,
    *,
    thresholds: ShadowThresholds,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow hard seeds into connected support and preserve measured strength."""
    h, w = strength.shape
    if not seeds:
        return np.zeros((h, w), dtype=np.float32), _boundary_info(
            "none",
            0.0,
            0.0,
            float(thresholds.boundary_falloff_px),
        )

    seed_union = np.zeros((h, w), dtype=bool)
    for seed in seeds:
        seed_union |= seed

    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        soft_support.astype(np.uint8),
        connectivity=8,
    )
    support = np.zeros((h, w), dtype=bool)
    for label_idx in range(1, n_labels):
        comp = labels == label_idx
        if (comp & seed_union).any():
            support |= comp

    if not support.any():
        support = seed_union

    loose_err = float(thresholds.max_reconstruction_error) * float(thresholds.loose_error_multiplier)
    err_conf = 1.0 - _smoothstep(float(thresholds.max_reconstruction_error), loose_err, err)
    subject_conf = 1.0 - _smoothstep(
        float(thresholds.subject_alpha_max),
        min(1.0, float(thresholds.subject_alpha_max) + 0.18),
        subject_alpha,
    )
    base_alpha = strength * err_conf * subject_conf
    base_alpha = np.where(support, base_alpha, 0.0).astype(np.float32)
    if not base_alpha.any():
        return base_alpha, _boundary_info(
            "none",
            0.0,
            0.0,
            float(thresholds.boundary_falloff_px),
        )

    support_values = base_alpha[support & (base_alpha > 0.0)]
    support_p75 = float(np.percentile(support_values, 75.0)) if support_values.size else 0.0
    plateau_threshold = max(float(thresholds.hard_boundary_alpha_min), float(thresholds.hard_plateau_alpha_min))
    plateau_fraction = (
        float((support_values >= plateau_threshold).mean()) if support_values.size else 0.0
    )
    seed_area = int((seed_union & support).sum())
    support_area = int(support.sum())
    support_expansion_ratio = float(support_area) / max(float(seed_area), 1.0)
    seed_values = base_alpha[seed_union & support & (base_alpha > 0.0)]
    seed_p75 = float(np.percentile(seed_values, 75.0)) if seed_values.size else 0.0
    seed_eroded = cv2.erode(
        (seed_union & support).astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)
    seed_boundary = (seed_union & support) & ~seed_eroded
    seed_boundary_values = base_alpha[seed_boundary & (base_alpha > 0.0)]
    seed_boundary_p25 = (
        float(np.percentile(seed_boundary_values, 25.0)) if seed_boundary_values.size else 0.0
    )

    falloff_px = min(
        float(thresholds.boundary_falloff_px),
        max(3.0, 0.04 * float(min(h, w))),
    )
    contact_px = min(
        float(thresholds.contact_distance_px),
        max(4.0, float(thresholds.contact_distance_ratio) * float(min(h, w))),
    )
    contact_side = (dist_to_subject <= contact_px) & support
    eroded = cv2.erode(
        support.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)
    inner_boundary = support & ~eroded
    open_boundary = inner_boundary & ~contact_side

    if open_boundary.any():
        boundary_values = base_alpha[open_boundary & (base_alpha > 0.0)]
        boundary_p75 = float(np.percentile(boundary_values, 75.0)) if boundary_values.size else 0.0
        boundary_ratio = boundary_p75 / max(support_p75, 1e-6)
        # A very broad low-alpha support can make a hard UI shadow look soft at
        # the outer edge. If the strict seed itself has a strong high-alpha
        # platform and the loose support expanded far beyond it, keep the
        # measured hard boundary instead of applying a wide falloff.
        plateau_hard_boundary = (
            support_expansion_ratio >= float(thresholds.hard_support_expansion_min)
            and seed_p75 >= float(thresholds.hard_seed_alpha_p75_min)
            and seed_boundary_p25 >= float(thresholds.hard_boundary_alpha_min)
            and plateau_fraction >= float(thresholds.hard_plateau_fraction_min)
        )
        hard_boundary = (
            (
                boundary_p75 >= float(thresholds.hard_boundary_alpha_min)
                and boundary_ratio >= float(thresholds.hard_boundary_ratio_min)
            )
            or plateau_hard_boundary
        )
        if hard_boundary:
            falloff_px = min(falloff_px, float(thresholds.hard_boundary_falloff_px))
            boundary_mode = "hard"
        else:
            boundary_mode = "soft"
        dist_to_open_boundary = cv2.distanceTransform(
            (~open_boundary).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        open_falloff = _smoothstep(0.0, falloff_px, dist_to_open_boundary).astype(np.float32)
    else:
        boundary_p75 = 0.0
        boundary_ratio = 0.0
        boundary_mode = "closed"
        open_falloff = np.ones((h, w), dtype=np.float32)

    alpha = np.where(support, base_alpha * open_falloff, 0.0).astype(np.float32)

    boundary_info = _boundary_info(
        boundary_mode,
        boundary_p75,
        boundary_ratio,
        falloff_px,
    )
    boundary_info.update(
        {
            "support_alpha_p75": support_p75,
            "support_plateau_fraction": plateau_fraction,
            "support_expansion_ratio": support_expansion_ratio,
            "seed_alpha_p75": seed_p75,
            "seed_boundary_alpha_p25": seed_boundary_p25,
        }
    )
    return np.clip(alpha, 0.0, 1.0).astype(np.float32), boundary_info


def _boundary_info(
    mode: str,
    boundary_alpha_p75: float,
    boundary_to_support_alpha_ratio: float,
    falloff_px: float,
) -> dict[str, Any]:
    return {
        "boundary_mode": mode,
        "boundary_alpha_p75": float(boundary_alpha_p75),
        "boundary_to_support_alpha_ratio": float(boundary_to_support_alpha_ratio),
        "boundary_falloff_px": float(falloff_px),
    }


def _coerce_optional_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape HxW matching image")
    if arr.dtype == bool:
        return arr.astype(np.float32)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    ksize = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _prior_info(prior: ShadowPrior | None) -> dict[str, Any]:
    if prior is None:
        return {
            "source": "",
            "shadow_allowed": True,
            "has_subject_mask": False,
            "has_shadow_search_mask": False,
            "has_shadow_ownership_mask": False,
        }
    return {
        "source": prior.source,
        "shadow_allowed": bool(prior.shadow_allowed),
        "has_subject_mask": prior.subject_mask is not None,
        "has_shadow_search_mask": prior.shadow_search_mask is not None,
        "has_shadow_ownership_mask": prior.shadow_ownership_mask is not None,
    }


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (value >= edge1).astype(np.float32)
    x = np.clip((value.astype(np.float32) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _shadow_info(
    shadow_alpha: np.ndarray,
    accepted_components: list[np.ndarray],
    rejected_components: int,
    reason: str,
    *,
    prior: ShadowPrior | None = None,
    boundary_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mask = shadow_alpha > 0
    pixels = int(mask.sum())
    if pixels:
        ys, xs = np.where(mask)
        strengths = shadow_alpha[mask]
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        mean_alpha = float(np.mean(strengths))
        p95_alpha = float(np.percentile(strengths, 95.0))
        max_alpha = float(np.max(strengths))
    else:
        bbox = [0, 0, 0, 0]
        mean_alpha = 0.0
        p95_alpha = 0.0
        max_alpha = 0.0
    return {
        "method": "known_bg_scalar_darkening",
        "detected": bool(pixels > 0),
        "applied": bool(pixels > 0),
        "pixels": pixels,
        "bbox_xyxy": bbox,
        "mean_alpha": mean_alpha,
        "p95_alpha": p95_alpha,
        "max_alpha": max_alpha,
        "accepted_components": len(accepted_components),
        "component_areas": [int(comp.sum()) for comp in accepted_components],
        "rejected_components": int(rejected_components),
        "reason": reason,
        "prior": _prior_info(prior),
        "boundary": dict(boundary_info or {}),
    }


__all__ = [
    "ShadowThresholds",
    "ShadowPrior",
    "DISPLAY_SAFE_SHADOW_OPACITY_SCALE",
    "composite_subject_with_shadow",
    "estimate_shadow_alpha",
    "shadow_alpha_to_display_alpha",
    "shadow_prior_from_regions",
]
