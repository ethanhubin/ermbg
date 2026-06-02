"""Experimental PyMatting-backed alpha refinement for known-background bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from . import io
from .colorspace import oklab_distance, srgb_to_oklab
from .types import Trimap


@dataclass(frozen=True)
class PyMattingAlphaResult:
    alpha: np.ndarray
    debug: dict[str, Any]


@dataclass(frozen=True)
class KnownBOwnership:
    sure_fg: np.ndarray
    sure_bg: np.ndarray
    unknown: np.ndarray
    bg_candidate: np.ndarray
    protected_transition: np.ndarray
    shadow_unknown: np.ndarray
    enclosed_bg: np.ndarray
    thresholds: dict[str, Any]
    boundary_info: dict[str, Any]
    foreground_seed_inset_px: int
    foreground_seed_inset_info: dict[str, Any]
    subject_support_info: dict[str, Any]
    enclosed_info: dict[str, Any]
    shadow_info: dict[str, Any]
    debug: dict[str, Any]


def normalize_known_background_field(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    bg_threshold: float = 3.5,
    fg_threshold: float = 30.0,
    adaptive: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continuously normalize a mildly uneven known background.

    Known-B PyMatting and ShadowPatch assume a single background color. Some
    generated green/blue-screen assets have low-frequency background drift that
    is not subject or shadow, especially in alpha tails. This prepass builds a
    continuous normalization weight from high-confidence background evidence
    and near-background screen-colored tail pixels, then blends those pixels
    toward the measured background color. The field is smooth and applied
    before trimap construction so sure-BG/unknown borders do not get a hard
    discontinuity.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    bg = np.asarray(background_color, dtype=np.uint8)
    ownership = _build_known_background_ownership(
        image_srgb,
        bg,
        bg_threshold=float(bg_threshold),
        fg_threshold=float(fg_threshold),
        boundary_band_px=2,
        adaptive=bool(adaptive),
        require_exact_bg=False,
    )
    thresholds = ownership.thresholds
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0]
    distance = oklab_distance(lab, bg_lab)
    effective_bg_threshold = float(thresholds["bg_threshold_effective"])
    effective_fg_threshold = float(thresholds["fg_threshold_effective"])
    bg_candidate = ownership.bg_candidate
    sure_bg_normalization = ownership.sure_bg
    if int(bg_candidate.sum()) < max(32, int(round(float(bg_candidate.size) * 0.01))):
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient high-confidence background evidence",
            "high_conf_bg_pixels": int(bg_candidate.sum()),
            "sure_bg_normalization_pixels": int(sure_bg_normalization.sum()),
            **thresholds,
        }

    h, w = distance.shape
    image = image_srgb.astype(np.float32)
    bgf = bg.astype(np.float32).reshape(3)
    source_shadow_alpha = _known_background_display_shadow_alpha(image_srgb, bg)
    # Background normalization is allowed only where the source pixel implies
    # essentially no transferable screen darkening. Values above the weak
    # visible-shadow floor are protected so broad soft tails remain available
    # for ShadowPatch; the smooth interval prevents a hard normalization seam.
    normalize_shadow_full_alpha = 1.5 / 255.0
    normalize_shadow_zero_alpha = 8.0 / 255.0
    shadow_normalization_gate = 1.0 - _smoothstep_array(
        normalize_shadow_full_alpha,
        normalize_shadow_zero_alpha,
        source_shadow_alpha,
    )
    residual = image_srgb.astype(np.float32) - bg.astype(np.float32).reshape(1, 1, 3)
    border = _border_mask(distance.shape)
    drift_probe = sure_bg_normalization & border
    if int(drift_probe.sum()) < max(32, int(round(float(bg_candidate.size) * 0.002))):
        drift_probe = sure_bg_normalization
    bg_residual = residual[drift_probe]
    residual_abs = np.abs(bg_residual)
    residual_p95 = float(np.percentile(residual_abs, 95.0)) if bg_residual.size else 0.0
    residual_std = float(np.std(bg_residual.astype(np.float32), axis=0).mean()) if bg_residual.size else 0.0
    changed_bg_pixels = sure_bg_normalization & np.any(image_srgb != bg.reshape(1, 1, 3), axis=2)
    if not bool(changed_bg_pixels.any()):
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "sure background already matches known-B",
            "high_conf_bg_pixels": int(bg_candidate.sum()),
            "sure_bg_normalization_pixels": int(sure_bg_normalization.sum()),
            "drift_probe_pixels": int(drift_probe.sum()),
            "residual_abs_p95_u8": residual_p95,
            "residual_std_u8": residual_std,
            "changed_bg_pixels": 0,
            "ownership": ownership.debug,
            **thresholds,
        }

    dominant = int(np.argmax(bgf))
    others = [idx for idx in range(3) if idx != dominant]
    screen_like_tail = np.zeros((h, w), dtype=bool)
    tail_weight = np.zeros((h, w), dtype=np.float32)
    if float(bgf[dominant]) >= 64.0 and float(bgf[dominant] - np.max(bgf[others])) >= 48.0:
        other_max = np.max(image[..., others], axis=2)
        # Tail normalization is intentionally limited to screen-colored pixels:
        # quiet off-channels and a dominant screen channel. This keeps yellow
        # ring material, highlights, and dark outlines out of the prepass.
        screen_like_tail = (
            (image[..., dominant] > other_max + max(6.0, float(bgf[dominant]) * 0.03))
            & (other_max <= max(8.0, float(np.max(bgf[others])) + 8.0))
        )
        strength = np.clip(1.0 - image[..., dominant] / max(float(bgf[dominant]), 1.0), -0.20, 1.0)
        # This gate is no longer a shadow-tail normalizer. It only lets very
        # weak screen-color drift contribute to the background field; visible
        # darkening is excluded by ``shadow_normalization_gate`` below.
        tail_max_strength = float(normalize_shadow_zero_alpha)
        strength_gate = np.clip((tail_max_strength - np.maximum(strength, 0.0)) / tail_max_strength, 0.0, 1.0)
        strength_gate = strength_gate * strength_gate * (3.0 - 2.0 * strength_gate)
        fg_span = max(effective_fg_threshold - effective_bg_threshold, 1e-6)
        bg_color_gate = np.clip(1.0 - (distance - effective_bg_threshold) / fg_span, 0.0, 1.0)
        tail_weight = np.where(
            screen_like_tail,
            strength_gate * bg_color_gate * shadow_normalization_gate,
            0.0,
        ).astype(np.float32)

        strong_fg = distance >= effective_fg_threshold
        subject_support, _support_info = _known_background_subject_material_support(
            image_srgb,
            bg,
            strong_fg=strong_fg,
            screen_dominant_shadow=_screen_dominant_shadow_pixels(image_srgb, bg),
        )
        if bool(subject_support.any()):
            dist_to_subject = cv2.distanceTransform((~subject_support).astype(np.uint8), cv2.DIST_L2, 3)
            subject_clearance = np.clip((dist_to_subject - 2.0) / 10.0, 0.0, 1.0).astype(np.float32)
            tail_weight *= subject_clearance

    raw_weight = np.zeros((h, w), dtype=np.float32)
    raw_weight[sure_bg_normalization] = 1.0
    raw_weight = np.maximum(raw_weight, tail_weight)
    if float(raw_weight.max()) <= 0.0:
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "empty normalization support",
            "high_conf_bg_pixels": int(bg_candidate.sum()),
            "sure_bg_normalization_pixels": int(sure_bg_normalization.sum()),
            "drift_probe_pixels": int(drift_probe.sum()),
            "residual_abs_p95_u8": residual_p95,
            "residual_std_u8": residual_std,
            "ownership": ownership.debug,
            **thresholds,
        }

    sigma = float(max(2.0, min(8.0, round(float(min(h, w)) * 0.025))))
    ksize = int(round(sigma * 6.0)) | 1
    weight = cv2.GaussianBlur(raw_weight, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    weight = np.minimum(weight, shadow_normalization_gate).astype(np.float32)
    weight[sure_bg_normalization] = 1.0
    # Do not let smoothing spill into obvious material. Any residual line here
    # is less harmful than pre-normalizing real subject color.
    obvious_material = distance >= max(effective_fg_threshold, effective_bg_threshold + 8.0)
    weight[obvious_material & ~screen_like_tail] = 0.0

    normalized = image * (1.0 - weight[..., None]) + bgf.reshape(1, 1, 3) * weight[..., None]
    normalized_u8 = np.clip(normalized + 0.5, 0, 255).astype(np.uint8)
    normalized_u8[sure_bg_normalization] = bg.reshape(1, 3)
    changed = np.abs(normalized_u8.astype(np.int16) - image_srgb.astype(np.int16)).mean(axis=2) > 0
    return normalized_u8, {
        "enabled": True,
        "applied": True,
        "reason": "background drift normalized",
        "background_color": [int(c) for c in bg],
        "high_conf_bg_pixels": int(bg_candidate.sum()),
        "sure_bg_normalization_pixels": int(sure_bg_normalization.sum()),
        "protected_transition_pixels": int(ownership.protected_transition.sum()),
        "shadow_unknown_pixels": int(ownership.shadow_unknown.sum()),
        "changed_bg_pixels": int(changed_bg_pixels.sum()),
        "drift_probe_pixels": int(drift_probe.sum()),
        "enclosed_bg_pixels": int(ownership.enclosed_bg.sum()),
        "enclosed_bg_component_min_area": int(ownership.enclosed_info.get("enclosed_bg_component_min_area", 0)),
        "residual_abs_p95_u8": residual_p95,
        "residual_std_u8": residual_std,
        "screen_like_tail_pixels": int(screen_like_tail.sum()),
        "tail_weight_pixels": int((tail_weight > 1.0 / 255.0).sum()),
        "shadow_normalization_gate": {
            "full_alpha": float(normalize_shadow_full_alpha),
            "zero_alpha": float(normalize_shadow_zero_alpha),
            "protected_pixels": int((shadow_normalization_gate <= 1.0 / 255.0).sum()),
            "transition_pixels": int(
                ((shadow_normalization_gate > 1.0 / 255.0) & (shadow_normalization_gate < 1.0 - 1.0 / 255.0)).sum()
            ),
            "full_pixels": int((shadow_normalization_gate >= 1.0 - 1.0 / 255.0).sum()),
        },
        "weight_nonzero_pixels": int((weight > 1.0 / 255.0).sum()),
        "weight_mean": float(weight.mean()),
        "weight_p95": float(np.percentile(weight, 95.0)),
        "changed_pixels": int(changed.sum()),
        "sigma_px": sigma,
        "ownership": ownership.debug,
        **thresholds,
    }


def _known_background_display_shadow_alpha(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
) -> np.ndarray:
    """Return display-space black alpha implied by known-background darkening."""
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    image = image_srgb.astype(np.float32)
    usable = bg >= 8.0
    if not bool(usable.any()):
        return np.zeros(image_srgb.shape[:2], dtype=np.float32)
    weights = np.where(usable, bg * bg, 0.0).astype(np.float32)
    weight_sum = np.maximum(float(weights.sum()), 1e-6)
    channel_alpha = 1.0 - image / np.maximum(bg, 1.0)
    alpha = (channel_alpha * weights).sum(axis=-1) / weight_sum
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _smoothstep_array(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (value >= edge1).astype(np.float32)
    x = np.clip((value.astype(np.float32) - float(edge0)) / (float(edge1) - float(edge0)), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _known_background_hard_shadow_subject_evidence_release(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    sure_fg: np.ndarray,
    shadow_unknown: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Release subject-side evidence when a hard shadow lacks subject color.

    PyMatting consumes trimap evidence; it does not know that a smooth hard
    shadow should become a separate shadow layer. When a high-alpha known-B
    shadow touches an outlined UI subject, and the current unknown band contains
    shadow/fringe but almost no true subject color, the shadow can be solved as
    green/blue foreground. This pass locally releases a shallow sure-FG boundary
    beside that shadow component so PyMatting sees enough subject-side evidence.
    """
    h, w = sure_fg.shape
    release = np.zeros((h, w), dtype=bool)
    bg = np.asarray(background_color, dtype=np.uint8)
    bgf = bg.astype(np.float32).reshape(3)
    dominant = int(np.argmax(bgf))
    sorted_bg = np.sort(bgf)
    info: dict[str, Any] = {
        "enabled": True,
        "released_pixels": 0,
        "components": [],
        "omitted_components": 0,
        "reason": "",
    }
    if float(bgf[dominant]) < 64.0 or float(sorted_bg[-1] - sorted_bg[-2]) < 48.0:
        info["reason"] = "background is not screen-dominant"
        return release, info
    if not bool(sure_fg.any()) or not bool(shadow_unknown.any()):
        info["reason"] = "missing sure foreground or shadow evidence"
        return release, info

    img = image_srgb.astype(np.float32)
    source_shadow_alpha = _known_background_display_shadow_alpha(image_srgb, bg)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(shadow_unknown.astype(np.uint8), 8)
    min_area = max(32, int(round(float(h * w) * 0.003)))
    release_px = 8.0
    shadow_neighborhood_px = 15
    fg_neighborhood_px = 7
    dist_inside_fg = cv2.distanceTransform(sure_fg.astype(np.uint8), cv2.DIST_L2, 3)
    _, fg_labels, _, _ = cv2.connectedComponentsWithStats(sure_fg.astype(np.uint8), 8)
    shadow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (shadow_neighborhood_px * 2 + 1, shadow_neighborhood_px * 2 + 1),
    )
    fg_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (fg_neighborhood_px * 2 + 1, fg_neighborhood_px * 2 + 1),
    )
    exact_bg = np.all(image_srgb == bg.reshape(1, 1, 3), axis=2)
    components: list[dict[str, Any]] = []

    for label in range(1, labels_count):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        values = source_shadow_alpha[comp]
        alpha_p50 = float(np.percentile(values, 50.0)) if values.size else 0.0
        alpha_p90 = float(np.percentile(values, 90.0)) if values.size else 0.0
        # This pass is intentionally narrow: it is for high-alpha hard shadows
        # that PyMatting can mistake for opaque subject. Lite shadows and soft
        # ramps are left alone because their lower scalar alpha already gives
        # the solver enough background evidence.
        high_alpha_hard_shadow = bool(alpha_p50 >= 0.42 and alpha_p90 >= 0.46)

        dilated_shadow = cv2.dilate(comp.astype(np.uint8), shadow_kernel, iterations=1).astype(bool)
        local_sure_fg = sure_fg & dilated_shadow
        near_fg = cv2.dilate(sure_fg.astype(np.uint8), fg_kernel, iterations=1).astype(bool)
        # For evidence sufficiency we inspect the whole non-clean band beside
        # the subject, not just pixels already protected as transition. B008
        # has usable yellow subject evidence in this wider band; looking only
        # at protected shadow/fringe pixels falsely classifies it as B003-like.
        local_unknown = (~sure_fg) & ~exact_bg & dilated_shadow & near_fg
        local_sure_pixels = int(local_sure_fg.sum())
        local_unknown_pixels = int(local_unknown.sum())
        component_info: dict[str, Any] = {
            "area": area,
            "bbox_xyxy": [
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
            ],
            "shadow_alpha_p50": alpha_p50,
            "shadow_alpha_p90": alpha_p90,
            "local_sure_fg_pixels": local_sure_pixels,
            "local_unknown_pixels": local_unknown_pixels,
            "keep": False,
        }
        if not high_alpha_hard_shadow or local_sure_pixels < 20 or local_unknown_pixels < 20:
            component_info["reason"] = "insufficient hard-shadow or local evidence"
            components.append(component_info)
            continue

        subject_color = np.median(image_srgb[local_sure_fg], axis=0).astype(np.float32)
        color_dist = np.sqrt(np.sum((img - subject_color.reshape(1, 1, 3)) ** 2, axis=2))
        subject_like = local_unknown & (color_dist < 60.0)
        other_max = np.max(np.delete(img, dominant, axis=2), axis=2)
        screen_like = local_unknown & (img[..., dominant] > other_max + max(8.0, float(bgf[dominant]) * 0.04))
        dark_outline = local_unknown & (np.max(img, axis=2) < 80.0)
        subject_like_ratio = float(subject_like.sum()) / float(max(1, local_unknown_pixels))
        screen_like_ratio = float(screen_like.sum()) / float(max(1, local_unknown_pixels))
        dark_outline_ratio = float(dark_outline.sum()) / float(max(1, local_unknown_pixels))
        # Evidence-poor means the local unknown is mostly screen spill or dark
        # outline, while true subject-color pixels are scarce. That combination
        # is the observed B003/B018/B033 failure mode; unoutlined hard shadows
        # such as B008 retain enough subject-like pixels and are protected.
        evidence_poor = bool(
            subject_like_ratio < 0.12
            and (screen_like_ratio >= 0.25 or dark_outline_ratio >= 0.05)
        )
        adjacent_fg_labels = np.unique(fg_labels[local_sure_fg])
        adjacent_fg_labels = adjacent_fg_labels[adjacent_fg_labels > 0]
        adjacent_subject = np.isin(fg_labels, adjacent_fg_labels) if adjacent_fg_labels.size else np.zeros_like(sure_fg)
        # Once the evidence gap is proven by a hard-shadow component, release
        # the adjacent subject component boundary, not only the shadow-facing
        # edge. PyMatting needs a balanced local color manifold; a one-sided
        # strip still lets the hard shadow dominate the unknown solve.
        local_release = adjacent_subject & (dist_inside_fg <= release_px)
        keep = bool(evidence_poor and local_release.any())
        if keep:
            release |= local_release
        component_info.update(
            {
                "subject_color": [float(c) for c in subject_color],
                "subject_like_ratio": subject_like_ratio,
                "screen_like_ratio": screen_like_ratio,
                "dark_outline_ratio": dark_outline_ratio,
                "released_pixels": int(local_release.sum()) if keep else 0,
                "release_px": float(release_px),
                "keep": keep,
                "reason": "" if keep else "subject evidence is sufficient",
            }
        )
        components.append(component_info)

    components.sort(key=lambda item: (item.get("keep", False), item.get("area", 0)), reverse=True)
    info.update(
        {
            "released_pixels": int(release.sum()),
            "component_min_area": int(min_area),
            "release_px": float(release_px),
            "components": components[:12],
            "omitted_components": max(0, len(components) - 12),
            "reason": "" if bool(release.any()) else "no hard-shadow subject evidence gap",
        }
    )
    return release, info


def _build_known_background_ownership(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    bg_threshold: float,
    fg_threshold: float,
    boundary_band_px: int,
    adaptive: bool,
    require_exact_bg: bool,
) -> KnownBOwnership:
    """Classify known-B ownership once, before normalization or trimap use.

    The order is intentional: first find reliable subject material seeds, then
    reserve subject-adjacent transition/shadow pixels, and only then declare
    remaining background-like support as sure-BG. This keeps background
    normalization and trimap construction from using two incompatible region
    definitions.
    """
    bg = np.asarray(background_color, dtype=np.uint8)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0]
    d = oklab_distance(lab, bg_lab)
    thresholds = (
        _adaptive_known_background_thresholds(d, image_srgb, bg, bg_threshold, fg_threshold)
        if adaptive
        else _fixed_known_background_thresholds(bg_threshold, fg_threshold)
    )
    effective_bg_threshold = float(thresholds["bg_threshold_effective"])

    bg_close = d <= effective_bg_threshold
    exterior_bg = _flood_from_border(bg_close)
    enclosed_bg_raw = bg_close & ~exterior_bg
    enclosed_bg, enclosed_info = _filter_enclosed_background_components(enclosed_bg_raw)
    bg_candidate = (exterior_bg | enclosed_bg) & bg_close

    not_exterior = ~exterior_bg
    dist_to_exterior = cv2.distanceTransform(not_exterior.astype(np.uint8), cv2.DIST_L2, 3)
    if adaptive:
        thresholds = {
            **thresholds,
            **_adaptive_foreground_seed_threshold(
                d,
                not_exterior,
                dist_to_exterior,
                bg_threshold=effective_bg_threshold,
                base_fg_threshold=float(thresholds["fg_threshold_effective"]),
                base_fg_source=str(thresholds["fg_threshold_source"]),
                requested_fg_threshold=float(fg_threshold),
                background_noise_mad=float(thresholds["background_noise_mad"]),
                boundary_band_px=int(boundary_band_px),
            ),
        }
    effective_fg_threshold = float(thresholds["fg_threshold_effective"])
    boundary_info = (
        _adaptive_boundary_band(
            d,
            not_exterior,
            dist_to_exterior,
            effective_bg_threshold,
            effective_fg_threshold,
            boundary_band_px,
        )
        if adaptive
        else _fixed_boundary_band(boundary_band_px)
    )
    strong_fg = d >= effective_fg_threshold
    screen_dominant_shadow = _screen_dominant_shadow_pixels(image_srgb, bg)
    subject_support, subject_support_info = _known_background_local_material_core_support(
        image_srgb,
        bg,
        strong_fg=strong_fg,
        screen_dominant_shadow=screen_dominant_shadow,
        boundary_info=boundary_info,
    )
    if bool(subject_support.any()):
        sure_fg = subject_support.copy()
        foreground_seed_inset_px = int(round(float(subject_support_info.get("effective_fg_core_inset_px", 0.0))))
        foreground_seed_inset_info = {
            "source": "local_material_core_with_extra_fg_inset",
            "local_core_px": float(subject_support_info.get("local_core_px", 0.0)),
            "extra_inset_px": float(subject_support_info.get("extra_inset_px", 0.0)),
            "effective_fg_core_inset_px": float(subject_support_info.get("effective_fg_core_inset_px", 0.0)),
        }
    else:
        foreground_seed_inset_px, foreground_seed_inset_info = _foreground_seed_inset_px(
            image_srgb.shape[:2],
            requested_boundary_band_px=boundary_band_px,
            boundary_info=boundary_info,
            support_distance=None,
        )
        sure_fg = strong_fg & (dist_to_exterior > float(foreground_seed_inset_px))

    shadow_bg, shadow_info = _known_background_shadow_like_background_mask(
        image_srgb,
        bg,
        subject_seed=sure_fg,
    )
    shadow_unknown = shadow_bg & (~sure_fg | screen_dominant_shadow)

    if bool(sure_fg.any()):
        dist_to_subject = cv2.distanceTransform((~sure_fg).astype(np.uint8), cv2.DIST_L2, 3)
        # This is a subject-derived transition guard, not an image-id constant.
        # It reserves the immediate subject neighborhood so background fill does
        # not flatten AA/shadow pixels before PyMatting sees them.
        transition_px = float(max(3.0, min(12.0, float(boundary_info["boundary_band_px_effective"]) + 5.0)))
        subject_transition = (
            (dist_to_subject <= transition_px)
            & ~sure_fg
            & (bg_candidate | screen_dominant_shadow | ((d > effective_bg_threshold) & (d < effective_fg_threshold)))
        )
    else:
        transition_px = 0.0
        subject_transition = np.zeros(image_srgb.shape[:2], dtype=bool)

    protected_transition = shadow_unknown | subject_transition
    subject_evidence_release, subject_evidence_info = _known_background_hard_shadow_subject_evidence_release(
        image_srgb,
        bg,
        sure_fg=sure_fg,
        shadow_unknown=shadow_unknown,
    )
    if bool(subject_evidence_release.any()):
        sure_fg = sure_fg & ~subject_evidence_release
        protected_transition |= subject_evidence_release

    if require_exact_bg:
        exact_known_bg = np.all(image_srgb == bg.reshape(1, 1, 3), axis=2)
        clean_bg = exact_known_bg
        clean_bg_threshold: float | str = "exact_known_b"
        clean_bg_policy = "normalized_exact_known_background"
    else:
        exact_known_bg = np.zeros(image_srgb.shape[:2], dtype=bool)
        clean_bg = bg_candidate
        clean_bg_threshold = "ownership_bg_candidate"
        clean_bg_policy = "pre_normalization_ownership_background"

    dist_to_non_clean = cv2.distanceTransform(clean_bg.astype(np.uint8), cv2.DIST_L2, 3)
    sure_bg = ((exterior_bg & clean_bg & (dist_to_non_clean >= 2.0)) | (enclosed_bg & clean_bg))
    # Enclosed holes and exterior background use the same ownership standard:
    # enclosed_bg only says a region may be background, not that it may bypass
    # source shadow or subject-transition evidence. Clean hole centers remain
    # sure-BG through the clean inset; hole-edge shadow/AA stays unknown for the
    # later same-background reconstruction stages.
    sure_bg &= ~protected_transition
    sure_fg = sure_fg & ~(enclosed_bg | shadow_unknown)
    unknown = ~(sure_fg | sure_bg)

    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(enclosed_bg.astype(np.uint8), 8)
    enclosed_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, labels_count)]
    debug = {
        "bg_candidate_pixels": int(bg_candidate.sum()),
        "sure_fg_pixels": int(sure_fg.sum()),
        "sure_bg_pixels": int(sure_bg.sum()),
        "unknown_pixels": int(unknown.sum()),
        "protected_transition_pixels": int(protected_transition.sum()),
        "subject_transition_pixels": int(subject_transition.sum()),
        "subject_transition_px": float(transition_px),
        "hard_shadow_subject_evidence_release_pixels": int(subject_evidence_release.sum()),
        "hard_shadow_subject_evidence": subject_evidence_info,
        "shadow_unknown_pixels": int(shadow_unknown.sum()),
        "exterior_bg_pixels": int(exterior_bg.sum()),
        "enclosed_bg_pixels": int(enclosed_bg.sum()),
        "clean_bg_policy": clean_bg_policy,
        "clean_bg_threshold": clean_bg_threshold,
        "clean_exterior_bg_pixels": int((exterior_bg & clean_bg).sum()),
        "exact_known_bg_pixels": int(exact_known_bg.sum()) if require_exact_bg else None,
        "sure_bg_clean_inset_px": 2.0,
        "enclosed_bg_pixels_raw": int(enclosed_bg_raw.sum()),
        **enclosed_info,
        "enclosed_bg_components": int(labels_count - 1),
        "largest_enclosed_bg_component": int(max(enclosed_areas, default=0)),
    }
    return KnownBOwnership(
        sure_fg=sure_fg,
        sure_bg=sure_bg,
        unknown=unknown,
        bg_candidate=bg_candidate,
        protected_transition=protected_transition,
        shadow_unknown=shadow_unknown,
        enclosed_bg=enclosed_bg,
        thresholds=thresholds,
        boundary_info=boundary_info,
        foreground_seed_inset_px=int(foreground_seed_inset_px),
        foreground_seed_inset_info=foreground_seed_inset_info,
        subject_support_info=subject_support_info,
        enclosed_info=enclosed_info,
        shadow_info=shadow_info,
        debug=debug,
    )


def build_known_background_trimap(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    bg_threshold: float = 3.5,
    fg_threshold: float = 24.0,
    boundary_band_px: int = 2,
    adaptive: bool = True,
) -> tuple[Trimap, dict[str, Any]]:
    """Build a conservative known-B trimap for hard-edged solid graphics.

    The mechanism assumption is narrow: exterior pixels close to the measured
    background are true background, strongly separated pixels away from that
    exterior are true opaque foreground, and the uncertain contour between them
    is the antialiasing band for PyMatting to solve. This is intentionally not a
    general object segmenter.

    Ownership is split into three passes: this trimap should recall hard
    subject anchors, edge-residue cleanup handles tiny screen-colored pinpricks,
    and ShadowPatch accepts only source-reprojection-consistent shadow repairs.
    A single global foreground threshold must not try to do all three jobs.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    bg = np.asarray(background_color, dtype=np.uint8)
    ownership = _build_known_background_ownership(
        image_srgb,
        bg,
        bg_threshold=float(bg_threshold),
        fg_threshold=float(fg_threshold),
        boundary_band_px=int(boundary_band_px),
        adaptive=bool(adaptive),
        require_exact_bg=True,
    )

    # If a source has no clear foreground core, PyMatting has nothing stable to
    # propagate from. Keep the trimap valid but report the weak support.
    trimap = Trimap(sure_fg=ownership.sure_fg, sure_bg=ownership.sure_bg, unknown=ownership.unknown)
    return trimap, {
        "method": "known_background_exterior_band",
        "adaptive": bool(adaptive),
        "background_color": [int(c) for c in bg],
        "bg_threshold": float(bg_threshold),
        "fg_threshold": float(fg_threshold),
        **ownership.thresholds,
        "boundary_band_px": int(boundary_band_px),
        **ownership.boundary_info,
        "foreground_seed_inset_px": int(ownership.foreground_seed_inset_px),
        "foreground_seed_inset": ownership.foreground_seed_inset_info,
        "subject_material_support": ownership.subject_support_info,
        **ownership.debug,
        "shadow_background": {
            **ownership.shadow_info,
            "unknown_ownership_pixels": int(ownership.shadow_unknown.sum()),
            "hard_ownership_pixels": 0,
            "screen_dominant_overlap_pixels": 0,
            "protected_foreground_overlap_pixels": 0,
        },
    }


def _known_background_local_material_core_support(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    strong_fg: np.ndarray,
    screen_dominant_shadow: np.ndarray,
    boundary_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find high-confidence material cores without using one global component radius.

    PyMatting should solve edge/AA/shadow transitions, not the interior of hard
    UI materials. A single support-radius inset works for simple buttons but
    fails on ornate controls: a large center panel makes thin metal ornaments
    look too close to a boundary, so they remain unknown and can be solved as
    semi-transparent. This helper keeps every coherent material island, builds
    a local core a few pixels inside its own edge, then pulls that core back by
    an extra 2px. The extra inset preserves the yellow/metal transition pixels
    that PyMatting needs to keep adjacent green/blue shadow from turning into
    foreground.
    """
    h, w = strong_fg.shape
    bg = background_color.astype(np.float32).reshape(3)
    dominant = int(np.argmax(bg))
    others = [idx for idx in range(3) if idx != dominant]
    if float(bg[dominant]) < 64.0 or float(bg[dominant] - np.max(bg[others])) < 48.0:
        return np.zeros((h, w), dtype=bool), {
            "used": False,
            "reason": "background is not a saturated single-channel screen",
            "candidate_pixels": 0,
            "support_pixels": 0,
        }

    img = image_srgb.astype(np.float32)
    off_energy = np.max(img[..., others], axis=2)
    material_floor = max(10.0, min(48.0, float(bg[dominant]) * 0.08))
    candidate = strong_fg & (off_energy >= material_floor) & ~screen_dominant_shadow
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    min_area = int(max(8.0, round(float(h * w) * 0.00003)))
    material = np.zeros((h, w), dtype=bool)
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep = bool(area >= min_area)
        if keep:
            material |= labels == label
        components.append(
            {
                "area": area,
                "keep": keep,
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
            }
        )
    components.sort(key=lambda item: (item["keep"], item["area"]), reverse=True)
    if not bool(material.any()):
        return np.zeros((h, w), dtype=bool), {
            "used": True,
            "reason": "no material support components",
            "candidate_pixels": int(candidate.sum()),
            "support_pixels": 0,
            "component_min_area": int(min_area),
            "material_floor_u8": float(material_floor),
            "screen_dominant_rejected_pixels": int((strong_fg & screen_dominant_shadow).sum()),
            "components": components[:20],
            "omitted_components": max(0, len(components) - 20),
        }

    # This is a local edge clearance, not a sample-specific radius. The boundary
    # measurement provides a floor, while the cap keeps thin ornaments seeded.
    local_core_px = float(max(1.5, min(3.0, float(boundary_info["boundary_band_px_effective"]) + 0.5)))
    material_dist = cv2.distanceTransform(material.astype(np.uint8), cv2.DIST_L2, 3)
    core = material & (material_dist >= local_core_px)

    labels_count2, labels2, stats2, _ = cv2.connectedComponentsWithStats(material.astype(np.uint8), 8)
    recovered_pixels = 0
    for label in range(1, labels_count2):
        comp = labels2 == label
        if bool((core & comp).any()):
            continue
        values = material_dist[comp]
        if values.size and float(values.max()) >= 1.8 and int(stats2[label, cv2.CC_STAT_AREA]) >= min_area * 4:
            add = comp & (material_dist >= min(1.8, float(values.max())))
            core |= add
            recovered_pixels += int(add.sum())

    # Keep foreground anchors strong but not edge-touching. The extra guard band
    # leaves transition pixels in unknown while preserving thin metal anchors.
    extra_inset_px = 2.0
    core_dist = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 3)
    support = core & (core_dist > extra_inset_px)
    released_pixels = int((core & ~support).sum())
    return support, {
        "used": True,
        "reason": "",
        "policy": "local_material_core_extra_inset",
        "candidate_pixels": int(candidate.sum()),
        "material_pixels": int(material.sum()),
        "support_pixels": int(support.sum()),
        "component_min_area": int(min_area),
        "material_floor_u8": float(material_floor),
        "local_core_px": float(local_core_px),
        "extra_inset_px": float(extra_inset_px),
        "effective_fg_core_inset_px": float(local_core_px + extra_inset_px),
        "core_pixels_before_extra_inset": int(core.sum()),
        "released_core_pixels": int(released_pixels),
        "recovered_core_pixels": int(recovered_pixels),
        "screen_dominant_rejected_pixels": int((strong_fg & screen_dominant_shadow).sum()),
        "components": components[:20],
        "omitted_components": max(0, len(components) - 20),
    }


def _known_background_subject_material_support(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    strong_fg: np.ndarray,
    screen_dominant_shadow: np.ndarray,
    fill_holes: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find a connected foreground material support without shadow pixels.

    Saturated screen shadows keep the screen channel dominant. Subject material
    has energy in at least one non-screen channel; using that evidence before
    the distance-transform inset gives PyMatting a clean interior seed while
    leaving outline, AA, and shadow falloff unknown.
    """
    h, w = strong_fg.shape
    bg = background_color.astype(np.float32).reshape(3)
    dominant = int(np.argmax(bg))
    others = [idx for idx in range(3) if idx != dominant]
    if float(bg[dominant]) < 64.0 or float(bg[dominant] - np.max(bg[others])) < 48.0:
        return np.zeros((h, w), dtype=bool), {
            "used": False,
            "reason": "background is not a saturated single-channel screen",
            "candidate_pixels": 0,
            "support_pixels": 0,
        }

    img = image_srgb.astype(np.float32)
    off_energy = np.max(img[..., others], axis=2)
    material_floor = max(10.0, min(48.0, float(bg[dominant]) * 0.08))
    candidate = strong_fg & (off_energy >= material_floor) & ~screen_dominant_shadow
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    if labels_count <= 1:
        return np.zeros((h, w), dtype=bool), {
            "used": True,
            "reason": "no material support components",
            "candidate_pixels": int(candidate.sum()),
            "support_pixels": 0,
            "material_floor_u8": float(material_floor),
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas) + 1)
    support = labels == label
    support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    support = support.astype(bool)
    if fill_holes:
        support = _fill_mask_holes(support)
    return support, {
        "used": True,
        "reason": "",
        "fill_holes": bool(fill_holes),
        "candidate_pixels": int(candidate.sum()),
        "support_pixels": int(support.sum()),
        "largest_component_area": int(areas.max()),
        "material_floor_u8": float(material_floor),
        "screen_dominant_rejected_pixels": int((strong_fg & screen_dominant_shadow).sum()),
    }


def _foreground_seed_inset_px(
    shape: tuple[int, int],
    *,
    requested_boundary_band_px: int | float,
    boundary_info: dict[str, Any],
    support_distance: np.ndarray | None,
) -> tuple[int, dict[str, Any]]:
    """Return a conservative interior-only foreground seed distance.

    The seed is a color anchor, not an ownership mask. Instead of baking in a
    fixed 10px/11px inset, derive the clearance from two measured signals:
    the observed source transition width and the material support's own
    thickness. The transition width keeps AA/outline pixels out of sure-FG; the
    thickness cap keeps small or thin controls from losing every seed.
    """
    h, w = shape
    floor_px = max(2.0, float(requested_boundary_band_px) + 2.0)
    transition_p90 = boundary_info.get("boundary_transition_distance_p90")
    transition_px = float(transition_p90) if transition_p90 is not None else float(
        boundary_info.get("boundary_band_px_effective", requested_boundary_band_px)
    )
    transition_inset = max(floor_px, transition_px + 0.5)

    if support_distance is not None and bool((support_distance > 0.0).any()):
        support_values = support_distance[support_distance > 0.0].astype(np.float32)
        support_radius = float(support_values.max())
        p90_radius = float(np.percentile(support_values, 90.0))
        # The 0.45 radius fraction keeps seeds comfortably inside rounded
        # button material without depending on a pixel constant. Blending with
        # p90 avoids a single very fat lobe forcing an excessive inset on
        # tapered or beveled UI controls.
        thickness_cap = max(floor_px, min(support_radius * 0.45, p90_radius * 0.80))
        measured = min(transition_inset, thickness_cap)
        source = "transition_width_limited_by_material_thickness"
    else:
        support_radius = 0.0
        p90_radius = 0.0
        scale_cap = max(floor_px, float(min(h, w)) * 0.12)
        measured = min(transition_inset, scale_cap)
        thickness_cap = scale_cap
        source = "transition_width_limited_by_image_scale"

    inset = int(max(floor_px, round(measured)))
    return inset, {
        "source": source,
        "floor_px": float(floor_px),
        "transition_inset_px": float(transition_inset),
        "material_thickness_cap_px": float(thickness_cap),
        "support_radius_max_px": float(support_radius),
        "support_radius_p90_px": float(p90_radius),
    }


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inv = (~mask).astype(np.uint8)
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    work = inv.copy()
    for x in range(w):
        if work[0, x]:
            cv2.floodFill(work, flood, (x, 0), 2)
        if work[h - 1, x]:
            cv2.floodFill(work, flood, (x, h - 1), 2)
    for y in range(h):
        if work[y, 0]:
            cv2.floodFill(work, flood, (0, y), 2)
        if work[y, w - 1]:
            cv2.floodFill(work, flood, (w - 1, y), 2)
    holes = work == 1
    return mask | holes


def _filter_enclosed_background_components(enclosed_bg: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep real known-B cutouts while dropping tiny same-screen speckles.

    Known-B holes are important: forcing them unknown lets PyMatting smear
    subject across transparent openings. But generator pinpricks inside ornate
    hard UI can have the exact screen color and become black dots if promoted
    to sure background. Component area separates real cutouts from speckles
    without encoding a sample id.
    """
    h, w = enclosed_bg.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed_bg.astype(np.uint8), 8)
    min_area = int(max(8.0, round(float(h * w) * 0.00005)))
    kept = np.zeros((h, w), dtype=bool)
    dropped_pixels = 0
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        keep = bool(area >= min_area)
        if keep:
            kept |= labels == label
        else:
            dropped_pixels += area
        components.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "keep": keep,
            }
        )
    components.sort(key=lambda item: (item["keep"], item["area"]), reverse=True)
    return kept, {
        "enclosed_bg_component_min_area": int(min_area),
        "enclosed_bg_dropped_pixels": int(dropped_pixels),
        "enclosed_bg_components_debug": components[:12],
        "enclosed_bg_omitted_components": max(0, len(components) - 12),
    }


def _screen_dominant_shadow_pixels(image_srgb: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    """Return pixels whose source color still points at the screen channel.

    Empirical rule, mechanism-driven: true green/blue-screen shadows remain
    dominated by the screen channel even when darkened. Subject-owned dark
    grooves often become near-black or cross-channel material; those must not
    let shadow evidence override a strong foreground seed.
    """
    h, w = image_srgb.shape[:2]
    bg = background_color.astype(np.float32).reshape(3)
    dominant = int(np.argmax(bg))
    sorted_bg = np.sort(bg)
    if float(bg[dominant]) < 64.0 or float(sorted_bg[-1] - sorted_bg[-2]) < 48.0:
        return np.zeros((h, w), dtype=bool)
    img = image_srgb.astype(np.float32)
    other = np.max(np.delete(img, dominant, axis=2), axis=2)
    margin = max(8.0, float(bg[dominant]) * 0.08)
    return (img[..., dominant] > other + margin) & (img[..., dominant] <= float(bg[dominant]) * 0.98)


def _known_background_shadow_like_background_mask(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    subject_seed: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect scalar-darkened known-B support near a subject.

    This is support evidence, not final subject ownership. The caller may pin
    weak/non-seed scalar support to trimap background to keep screen residue out
    of the foreground, but strong foreground seeds are overridden only when the
    source color is still screen-dominant. Screen-neutral dark material, such
    as ornate metal grooves connected to a cast shadow, must remain subject
    owned and let ShadowPatch prove any actual shadow by same-background
    reprojection later.
    """
    h, w = subject_seed.shape
    if not bool(subject_seed.any()):
        return np.zeros((h, w), dtype=bool), {
            "enabled": True,
            "reason": "no subject seed",
            "pixels": 0,
        }

    bg = background_color.astype(np.float32).reshape(3)
    img = image_srgb.astype(np.float32)
    informative = bg >= max(8.0, float(bg.max()) * 0.12)
    if not bool(informative.any()):
        informative = bg == float(bg.max())
    off_background = bg < max(8.0, float(bg.max()) * 0.12)

    denom = max(float((bg[informative] * bg[informative]).sum()), 1e-6)
    scale = (img[..., informative] * bg[informative]).sum(axis=-1) / denom
    strength = np.clip(1.0 - scale, 0.0, 1.0).astype(np.float32)
    recon = scale[..., None] * bg.reshape(1, 1, 3)
    err = np.sqrt(np.mean((img - recon) * (img - recon), axis=-1)).astype(np.float32)
    off_excess = (
        np.where(off_background.reshape(1, 1, 3), img - bg.reshape(1, 1, 3), 0.0).max(axis=-1)
        if bool(off_background.any())
        else np.zeros((h, w), dtype=np.float32)
    )

    dist_to_seed = cv2.distanceTransform((~subject_seed).astype(np.uint8), cv2.DIST_L2, 3)
    near_px = float(max(8.0, min(80.0, round(min(h, w) * 0.34))))
    near_seed = dist_to_seed <= near_px
    border = _border_mask((h, w))
    border_strength = strength[border].astype(np.float32)
    strength_floor = float(np.percentile(border_strength, 99.5)) if border_strength.size else 0.0
    # Use the border as the background-drift floor. True shadow evidence must
    # be stronger than ordinary generator drift; otherwise large noisy green
    # fields become protected transition and never normalize to exact known-B.
    strength_min = max(2.0 / 255.0, strength_floor + 1.0 / 255.0)

    # These guards are 8-bit reconstruction tolerances. The values are broad
    # enough for generated shadow texture but still reject colorful subject
    # material because off-background channels would have to stay quiet.
    err_max = max(18.0, float(np.percentile(err[border], 99.5)) + 6.0) if border.any() else 18.0
    off_excess_max = max(28.0, float(np.percentile(off_excess[border], 99.5)) + 8.0) if border.any() else 28.0
    candidate = near_seed & (strength >= strength_min) & (err <= err_max) & (off_excess <= off_excess_max)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    min_area = int(max(4.0, float(h * w) * 0.00003))
    coherent_anchor_area = int(max(float(min_area), float(h * w) * 0.005))
    kept = np.zeros((h, w), dtype=bool)
    largest_kept_area = 0
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        values = strength[comp]
        p90 = float(np.percentile(values, 90.0)) if values.size else 0.0
        keep = bool(area >= min_area and p90 >= max(strength_min * 1.5, 0.02))
        if keep:
            kept |= comp
            largest_kept_area = max(largest_kept_area, area)
        components.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "strength_p90": p90,
                "keep": keep,
            }
        )
    components.sort(key=lambda item: (item["keep"], item["area"]), reverse=True)
    if largest_kept_area < coherent_anchor_area:
        return np.zeros((h, w), dtype=bool), {
            "enabled": True,
            "reason": "no coherent shadow anchor component",
            "pixels": 0,
            "candidate_pixels": int(candidate.sum()),
            "near_subject_px": near_px,
            "strength_min": float(strength_min),
            "err_max_u8": float(err_max),
            "off_excess_max_u8": float(off_excess_max),
            "component_min_area": int(min_area),
            # Profile gate: no coherent shadow anchor means no shadow path.
            # Fragmented dark outlines can look shadow-like, but should stay
            # with the subject instead of creating a false shadow layer.
            "coherent_anchor_min_area": int(coherent_anchor_area),
            "largest_kept_component_area": int(largest_kept_area),
            "components": components[:12],
            "omitted_components": max(0, len(components) - 12),
        }
    return kept, {
        "enabled": True,
        "reason": "" if bool(kept.any()) else "no scalar-darkening background near subject",
        "pixels": int(kept.sum()),
        "candidate_pixels": int(candidate.sum()),
        "near_subject_px": near_px,
        "strength_min": float(strength_min),
        "err_max_u8": float(err_max),
        "off_excess_max_u8": float(off_excess_max),
        "component_min_area": int(min_area),
        "coherent_anchor_min_area": int(coherent_anchor_area),
        "largest_kept_component_area": int(largest_kept_area),
        "components": components[:12],
        "omitted_components": max(0, len(components) - 12),
    }


def _fixed_known_background_thresholds(bg_threshold: float, fg_threshold: float) -> dict[str, Any]:
    return {
        "bg_threshold_effective": float(bg_threshold),
        "fg_threshold_effective": float(fg_threshold),
        "fg_threshold_requested": float(fg_threshold),
        "fg_threshold_source": "fixed",
        "fg_threshold_percentile": None,
        "fg_threshold_raise_cap": None,
        "fg_threshold_seed_pixels": None,
        "fg_threshold_largest_seed_component": None,
        "fg_threshold_candidate_pixels": None,
        "fg_threshold_min_seed_pixels": None,
        "fg_threshold_min_largest_component": None,
        "background_noise_median": None,
        "background_noise_mad": None,
        "background_noise_q99": None,
        "histogram_otsu_threshold": None,
    }


def _fixed_boundary_band(boundary_band_px: int) -> dict[str, Any]:
    return {
        "boundary_band_px_effective": int(boundary_band_px),
        "boundary_band_px_measured": int(boundary_band_px),
        "boundary_transition_pixels": 0,
        "boundary_transition_distance_p50": None,
        "boundary_transition_distance_p90": None,
    }


def _adaptive_boundary_band(
    distance: np.ndarray,
    not_exterior: np.ndarray,
    dist_to_exterior: np.ndarray,
    bg_threshold: float,
    fg_threshold: float,
    boundary_band_px: int,
) -> dict[str, Any]:
    """Choose an unknown-band width from the observed edge transition.

    When the source is a generated UI asset, the antialiasing/rim width can be
    one pixel on icons or several pixels on glossy 3D objects. The transition
    pixels between the effective background and foreground thresholds provide a
    direct measurement, so the manual setting acts as a floor rather than a
    sample-specific constant.
    """
    transition = not_exterior & (distance > bg_threshold) & (distance < fg_threshold)
    if bool(transition.any()):
        transition_dist = dist_to_exterior[transition].astype(np.float32)
        p50 = float(np.percentile(transition_dist, 50.0))
        p90 = float(np.percentile(transition_dist, 90.0))
        # A true antialiasing band is a narrow contour. If the transition
        # distances have a long tail, they include shadows/interior gradients;
        # widening the trimap for those pixels lets the solver eat the subject.
        measured_base = (
            int(boundary_band_px)
            if p50 > (float(boundary_band_px) * 4.0 + 1.0) or p90 > (p50 * 3.0 + 2.0)
            else int(np.ceil(p90))
        )
        if measured_base > int(boundary_band_px) * 2:
            # Once the measured edge is genuinely broad, PyMatting benefits
            # from a little context on both sides of the visible ramp. This
            # protects UI edge antialiasing from being pinned by nearby
            # sure-fg/sure-bg labels while the long-tail guard still rejects
            # shadow/interior gradients masquerading as an edge.
            measured = measured_base + int(np.ceil(float(min(distance.shape)) * 0.03))
        else:
            measured = measured_base
    else:
        p50 = 0.0
        p90 = 0.0
        measured = int(boundary_band_px)
    # Broad generated UI edges can have a real 6-12px antialias/shadow
    # transition even on 128px-high buttons. The long-tail guard above keeps
    # interior gradients from widening the trimap; this cap should therefore
    # allow measured UI edge widths instead of forcing them back to a 3-4px
    # band, which hardens subject edges before PyMatting can solve them.
    scale_cap = max(int(boundary_band_px), int(np.ceil(float(min(distance.shape)) * 0.10)))
    effective = int(max(int(boundary_band_px), min(measured, scale_cap)))
    return {
        "boundary_band_px_effective": effective,
        "boundary_band_px_measured": int(measured),
        "boundary_transition_pixels": int(transition.sum()),
        "boundary_transition_distance_p50": float(p50),
        "boundary_transition_distance_p90": float(p90),
    }


def _adaptive_known_background_thresholds(
    distance: np.ndarray,
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    bg_threshold: float,
    fg_threshold: float,
) -> dict[str, Any]:
    """Estimate trimap distance thresholds from the current known background.

    The user-facing thresholds remain useful as conservative defaults, but PNGs
    from generators often have compression/generation noise around the nominal
    background. The effective background threshold is therefore calibrated from
    border pixels that already match the known-B mode. Foreground separation is
    then chosen from the distance histogram when a clear valley exists; this
    keeps UI holes/background noise out of the foreground core without baking in
    one sample's absolute OKLab distances.
    """
    border_mask = _border_mask(distance.shape)
    border_dist = distance[border_mask].astype(np.float32)
    if border_dist.size == 0:
        border_dist = distance.reshape(-1).astype(np.float32)
    seed_limit = max(float(bg_threshold), float(np.percentile(border_dist, 50.0)) + 1.0)
    bg_seed = border_dist[border_dist <= seed_limit]
    if bg_seed.size < max(16, int(border_dist.size * 0.05)):
        bg_seed = border_dist[np.argsort(border_dist)[: max(1, int(border_dist.size * 0.25))]]
    seed_median = float(np.median(bg_seed)) if bg_seed.size else 0.0
    seed_mad = _mad(bg_seed)
    seed_q99 = float(np.percentile(bg_seed, 99.0)) if bg_seed.size else float(bg_threshold)
    # The additive floor is an OKLab quantization/noise allowance; the image
    # statistic dominates on noisy or compressed backgrounds.
    bg_effective = max(float(bg_threshold), seed_q99 + max(0.5, 3.0 * seed_mad))

    all_dist = distance.reshape(-1).astype(np.float32)
    otsu = _otsu_float_threshold(all_dist)
    min_gap = max(2.0, 6.0 * max(seed_mad, 0.25))
    if otsu is not None and otsu > bg_effective + min_gap:
        fg_effective = max(bg_effective + min_gap, float(otsu))
        source = "histogram_otsu"
    else:
        fg_effective = bg_effective + min_gap
        source = "background_noise_gap"
    return {
        "bg_threshold_effective": float(bg_effective),
        "fg_threshold_effective": float(fg_effective),
        "fg_threshold_requested": float(fg_threshold),
        "fg_threshold_source": source,
        "fg_threshold_percentile": None,
        "fg_threshold_raise_cap": None,
        "fg_threshold_seed_pixels": None,
        "fg_threshold_largest_seed_component": None,
        "fg_threshold_candidate_pixels": None,
        "fg_threshold_min_seed_pixels": None,
        "fg_threshold_min_largest_component": None,
        "background_noise_median": seed_median,
        "background_noise_mad": float(seed_mad),
        "background_noise_q99": seed_q99,
        "histogram_otsu_threshold": float(otsu) if otsu is not None else None,
    }


def _adaptive_foreground_seed_threshold(
    distance: np.ndarray,
    not_exterior: np.ndarray,
    dist_to_exterior: np.ndarray,
    *,
    bg_threshold: float,
    base_fg_threshold: float,
    base_fg_source: str,
    requested_fg_threshold: float,
    background_noise_mad: float,
    boundary_band_px: int,
) -> dict[str, Any]:
    """Choose high-confidence foreground seeds from this image's distance field.

    Foreground thresholding is a recall hint for hard subject structure, not
    the screen-residue removal mechanism. Pick the background/foreground valley
    estimated from this image, then lower it only if needed for a coherent
    anchor. Edge screen-color residue is handled by local edge passes and
    ShadowPatch reprojection; raising this global threshold to suppress residue
    would drop B056-like hard silver/metal edges into PyMatting unknown.
    """
    h, w = distance.shape
    image_area = int(h * w)
    min_gap = max(2.0, 6.0 * max(float(background_noise_mad), 0.25))
    candidate_floor = float(bg_threshold) + min_gap
    candidate_domain = not_exterior & (distance > candidate_floor)
    values = distance[candidate_domain].astype(np.float32)
    min_seed_pixels = int(max(16, round(float(image_area) * 0.0008)))
    min_largest_component = int(max(8, round(float(image_area) * 0.0004)))
    # Keep a broad safety cap so noisy histograms cannot push the foreground
    # recall threshold into highlight-only territory. Residue cleanup is local,
    # so the global seed threshold should bias toward preserving hard subject
    # structure rather than excluding every weak screen-colored edge pixel.
    raise_cap = max(candidate_floor, float(requested_fg_threshold) + 12.0)
    if values.size < min_seed_pixels:
        return {
            "fg_threshold_effective": float(min(max(candidate_floor, requested_fg_threshold), raise_cap)),
            "fg_threshold_source": "fallback_requested_insufficient_non_bg",
            "fg_threshold_percentile": None,
            "fg_threshold_raise_cap": float(raise_cap),
            "fg_threshold_seed_pixels": 0,
            "fg_threshold_largest_seed_component": 0,
            "fg_threshold_candidate_pixels": int(values.size),
            "fg_threshold_min_seed_pixels": int(min_seed_pixels),
            "fg_threshold_min_largest_component": int(min_largest_component),
        }

    seed_domain = not_exterior & (dist_to_exterior > float(max(1, boundary_band_px)))
    base = min(max(candidate_floor, float(base_fg_threshold)), raise_cap)
    best: dict[str, Any] | None = None
    candidates: list[tuple[float, float | None, str]] = [(base, None, f"{base_fg_source}_seed_guard")]
    for percentile in (50.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0):
        candidates.append(
            (
                min(max(candidate_floor, float(np.percentile(values, percentile))), raise_cap),
                float(percentile),
                "foreground_recall_percentile_seed_guard",
            )
        )

    seen: set[float] = set()
    for threshold, percentile, source in candidates:
        key = round(float(threshold), 4)
        if key in seen:
            continue
        seen.add(key)
        seed = seed_domain & (distance >= threshold)
        seed_pixels = int(seed.sum())
        largest = _largest_component_area(seed)
        info = {
            "fg_threshold_effective": float(threshold),
            "fg_threshold_source": source,
            "fg_threshold_percentile": percentile,
            "fg_threshold_raise_cap": float(raise_cap),
            "fg_threshold_seed_pixels": int(seed_pixels),
            "fg_threshold_largest_seed_component": int(largest),
            "fg_threshold_candidate_pixels": int(values.size),
            "fg_threshold_min_seed_pixels": int(min_seed_pixels),
            "fg_threshold_min_largest_component": int(min_largest_component),
        }
        best = info
        if seed_pixels >= min_seed_pixels and largest >= min_largest_component:
            return info

    assert best is not None
    # If even the recall-oriented candidates are fragmented, prefer the
    # requested threshold when it gives a coherent anchor; otherwise use the
    # least-bad candidate. This is a guard, not a sample override.
    requested = min(max(candidate_floor, float(requested_fg_threshold)), raise_cap)
    requested_seed = seed_domain & (distance >= requested)
    requested_seed_pixels = int(requested_seed.sum())
    requested_largest = _largest_component_area(requested_seed)
    if requested_seed_pixels >= min_seed_pixels and requested_largest >= min_largest_component:
        return {
            "fg_threshold_effective": float(requested),
            "fg_threshold_source": "requested_threshold_seed_guard",
            "fg_threshold_percentile": None,
            "fg_threshold_raise_cap": float(raise_cap),
            "fg_threshold_seed_pixels": int(requested_seed_pixels),
            "fg_threshold_largest_seed_component": int(requested_largest),
            "fg_threshold_candidate_pixels": int(values.size),
            "fg_threshold_min_seed_pixels": int(min_seed_pixels),
            "fg_threshold_min_largest_component": int(min_largest_component),
        }
    return {**best, "fg_threshold_source": "fallback_fragmented_foreground_recall"}


def _largest_component_area(mask: np.ndarray) -> int:
    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if labels_count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    band = max(2, min(10, int(round(min(h, w) * 0.06))))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values.astype(np.float32) - med)))


def _otsu_float_threshold(values: np.ndarray) -> float | None:
    vals = values[np.isfinite(values)]
    if vals.size < 16:
        return None
    hi = float(np.percentile(vals, 99.5))
    lo = float(np.percentile(vals, 0.5))
    if not hi > lo:
        return None
    hist, edges = np.histogram(vals, bins=128, range=(lo, hi))
    total = float(hist.sum())
    if total <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not bool(valid.any()):
        return None
    sum_bg = np.cumsum(hist * centers)
    sum_total = float(sum_bg[-1])
    mean_bg = np.zeros_like(sum_bg, dtype=np.float64)
    mean_fg = np.zeros_like(sum_bg, dtype=np.float64)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = np.zeros_like(mean_bg)
    between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    idx = int(np.argmax(between))
    return float(centers[idx])


def estimate_stable_background_color(image_srgb: np.ndarray) -> tuple[tuple[int, int, int], dict[str, Any]]:
    """Estimate the single known-B/sure-background color for routing.

    The corners/border mode are only seed evidence: they find a plausible
    background color family without looking at the subject. The final known-B is
    computed from structurally sure background support grown from that seed, so
    route params, normalization, trimap, unmix, and ShadowPatch all use one
    color contract.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    seed_bg, seed_info = _estimate_known_background_seed(image_srgb)
    if not seed_info.get("accepted", False):
        return seed_bg, {
            **seed_info,
            "accepted": False,
            "reason": seed_info.get("reason", "corner/background border is unstable"),
            "source": "sure_bg_mode",
            "seed": seed_info,
        }

    support, support_info = _known_background_support_from_seed(image_srgb, seed_bg)
    min_support = max(32, int(round(float(image_srgb.shape[0] * image_srgb.shape[1]) * 0.01)))
    if int(support.sum()) < min_support:
        return seed_bg, {
            "accepted": False,
            "reason": "insufficient sure background support",
            "source": "sure_bg_mode",
            "seed": seed_info,
            **support_info,
        }

    known_bg, color_info = _known_background_color_from_support(image_srgb, support)
    return known_bg, {
        "accepted": True,
        "reason": "accepted",
        "source": "sure_bg_mode",
        "seed": seed_info,
        **support_info,
        **color_info,
    }


def _estimate_known_background_seed(image_srgb: np.ndarray) -> tuple[tuple[int, int, int], dict[str, Any]]:
    h, w = image_srgb.shape[:2]
    size = max(2, min(10, int(round(min(h, w) * 0.06))))
    patches = [
        image_srgb[:size, :size],
        image_srgb[:size, -size:],
        image_srgb[-size:, :size],
        image_srgb[-size:, -size:],
    ]
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    medians = np.asarray([np.median(p.reshape(-1, 3), axis=0) for p in patches], dtype=np.float32)
    corner_agreement = float(np.std(medians, axis=0).mean())
    sigma = float(np.std(pixels.astype(np.float32), axis=0).mean())
    bg_arr = np.median(pixels, axis=0).astype(np.uint8)
    if corner_agreement <= 4.0 and sigma <= 6.0:
        return tuple(int(c) for c in bg_arr), {
            "accepted": True,
            "reason": "accepted",
            "source": "corners",
            "corner_agreement": corner_agreement,
            "sigma": sigma,
        }

    border = _border_pixels(image_srgb)
    q = (border >> 3).astype(np.int32)
    keys = q[:, 0] * 32 * 32 + q[:, 1] * 32 + q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    dominant = keys == values[dominant_idx]
    dominant_fraction = float(counts[dominant_idx]) / float(len(keys))
    dominant_pixels = border[dominant]
    dominant_sigma = float(np.std(dominant_pixels.astype(np.float32), axis=0).mean())
    dominant_bg = np.median(dominant_pixels, axis=0).astype(np.uint8)
    # Same mechanism as the solid-graphic prepass: a low-variance border mode is
    # stable known-B evidence when one or more corners are touched by subject.
    accepted = dominant_fraction >= 0.45 and dominant_sigma <= 8.0
    reason = "accepted" if accepted else "corner/background border is unstable"
    return tuple(int(c) for c in dominant_bg), {
        "accepted": accepted,
        "reason": reason,
        "source": "border_mode",
        "corner_agreement": corner_agreement,
        "corner_sigma": sigma,
        "dominant_border_fraction": dominant_fraction,
        "sigma": dominant_sigma,
    }


def _known_background_support_from_seed(
    image_srgb: np.ndarray,
    seed_bg: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    seed = np.asarray(seed_bg, dtype=np.uint8)
    lab = srgb_to_oklab(image_srgb)
    seed_lab = srgb_to_oklab(seed.reshape(1, 1, 3))[0, 0]
    distance = oklab_distance(lab, seed_lab)
    thresholds = _adaptive_known_background_thresholds(distance, image_srgb, seed, 3.5, 24.0)
    bg_close = distance <= float(thresholds["bg_threshold_effective"])
    exterior_bg = _flood_from_border(bg_close)
    enclosed_bg, enclosed_info = _filter_enclosed_background_components(bg_close & ~exterior_bg)
    support = exterior_bg | enclosed_bg
    return support, {
        "sure_bg_pixels": int(support.sum()),
        "sure_bg_fraction": float(support.mean()),
        "exterior_bg_pixels": int(exterior_bg.sum()),
        "enclosed_bg_pixels": int(enclosed_bg.sum()),
        "enclosed_bg_component_min_area": int(enclosed_info.get("enclosed_bg_component_min_area", 0)),
        **thresholds,
    }


def _known_background_color_from_support(
    image_srgb: np.ndarray,
    support: np.ndarray,
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    if bool((~support).any()):
        dist_to_non_support = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
        band_px = float(max(2.0, min(16.0, round(float(min(support.shape)) * 0.035))))
        color_support = support & (dist_to_non_support <= band_px)
        min_pixels = max(32, int(round(float(support.sum()) * 0.01)))
        if int(color_support.sum()) < min_pixels:
            color_support = support
        color_support_source = "support_boundary_near_unknown" if bool(color_support is not support) else "support_all"
    else:
        band_px = 0.0
        color_support = support
        color_support_source = "support_all"

    pixels = image_srgb[color_support]
    if pixels.size == 0:
        return (0, 0, 0), {
            "known_bg_source": "empty_support",
            "dominant_support_fraction": 0.0,
        }
    q = (pixels >> 3).astype(np.int32)
    keys = q[:, 0] * 32 * 32 + q[:, 1] * 32 + q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    dominant_idx = int(np.argmax(counts))
    dominant = keys == values[dominant_idx]
    dominant_pixels = pixels[dominant]
    mode_bg_arr = np.median(dominant_pixels, axis=0).astype(np.uint8)
    return tuple(int(c) for c in mode_bg_arr), {
        "known_bg_source": "boundary_support_quantized_mode",
        "support_pixels": int(support.sum()),
        "color_support_source": color_support_source,
        "color_support_pixels": int(pixels.shape[0]),
        "color_support_boundary_px": float(band_px),
        "dominant_support_pixels": int(counts[dominant_idx]),
        "dominant_support_fraction": float(counts[dominant_idx]) / float(max(1, pixels.shape[0])),
        "background_color": [int(c) for c in mode_bg_arr],
    }


def _border_pixels(image_srgb: np.ndarray) -> np.ndarray:
    h, w = image_srgb.shape[:2]
    band = max(2, min(10, int(round(min(h, w) * 0.06))))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return image_srgb[mask]


def _flood_from_border(mask: np.ndarray) -> np.ndarray:
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


def estimate_known_background_alpha_with_pymatting(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    method: str = "cf",
    image_space: str = "linear",
    bg_threshold: float = 3.5,
    fg_threshold: float = 24.0,
    boundary_band_px: int = 2,
    adaptive: bool = True,
) -> PyMattingAlphaResult:
    """Convenience path: build a known-B trimap, then solve with PyMatting."""
    trimap, trimap_info = build_known_background_trimap(
        image_srgb,
        background_color,
        bg_threshold=bg_threshold,
        fg_threshold=fg_threshold,
        boundary_band_px=boundary_band_px,
        adaptive=adaptive,
    )
    result = estimate_alpha_with_pymatting(
        image_srgb,
        trimap,
        method=method,
        image_space=image_space,
    )
    debug = dict(result.debug)
    debug["trimap"] = trimap_info
    return PyMattingAlphaResult(alpha=result.alpha, debug=debug)


def estimate_alpha_with_pymatting(
    image_srgb: np.ndarray,
    trimap: Trimap,
    *,
    method: str = "cf",
    image_space: str = "linear",
    cg_maxiter: int = 1000,
    cg_rtol: float = 1e-6,
) -> PyMattingAlphaResult:
    """Estimate alpha in ``trimap.unknown`` with PyMatting.

    PyMatting is trimap-driven and can propagate a bad trimap very confidently,
    so this helper deliberately does not construct ownership masks. Callers must
    pass a mechanism-proved trimap; the result is snapped back to sure fg/bg.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")
    if trimap.shape != image_srgb.shape[:2]:
        raise ValueError("trimap shape must match image")

    method_key = method.removeprefix("pymatting-").lower().replace("_", "-")
    if method_key not in {"cf", "knn", "lbdm", "lkm", "rw", "sm"}:
        raise ValueError(f"Unsupported PyMatting method: {method!r}")
    if image_space not in {"linear", "sRGB"}:
        raise ValueError("image_space must be 'linear' or 'sRGB'")

    unknown_pixels = int(trimap.unknown.sum())
    alpha = np.zeros(trimap.shape, dtype=np.float32)
    alpha[trimap.sure_fg] = 1.0
    if unknown_pixels == 0:
        return PyMattingAlphaResult(
            alpha=alpha,
            debug={
                "used": True,
                "method": method_key,
                "applied": False,
                "reason": "no unknown pixels",
                "unknown_pixels": 0,
                "image_space": image_space,
            },
        )

    if image_space == "linear":
        image = io.srgb_to_linear(image_srgb).astype(np.float64)
    else:
        image = (image_srgb.astype(np.float64) / 255.0).clip(0.0, 1.0)

    pm_trimap = np.full(trimap.shape, 0.5, dtype=np.float64)
    pm_trimap[trimap.sure_bg] = 0.0
    pm_trimap[trimap.sure_fg] = 1.0

    import time

    started = time.perf_counter()
    if method_key == "cf":
        from pymatting import estimate_alpha_cf

        solved = estimate_alpha_cf(
            image,
            pm_trimap,
            laplacian_kwargs={"epsilon": 1e-6},
            cg_kwargs={"maxiter": int(cg_maxiter), "rtol": float(cg_rtol)},
        )
    elif method_key == "knn":
        from pymatting import estimate_alpha_knn

        solved = estimate_alpha_knn(
            image,
            pm_trimap,
            laplacian_kwargs={"n_neighbors": [10, 5]},
            cg_kwargs={"maxiter": int(cg_maxiter), "rtol": float(cg_rtol)},
        )
    elif method_key == "lbdm":
        from pymatting import estimate_alpha_lbdm

        solved = estimate_alpha_lbdm(
            image,
            pm_trimap,
            laplacian_kwargs={"epsilon": 1e-6},
            cg_kwargs={"maxiter": int(cg_maxiter), "rtol": float(cg_rtol)},
        )
    elif method_key == "lkm":
        from pymatting import estimate_alpha_lkm

        solved = estimate_alpha_lkm(
            image,
            pm_trimap,
            laplacian_kwargs={"epsilon": 1e-6, "radius": 10},
            cg_kwargs={"maxiter": int(cg_maxiter), "rtol": float(cg_rtol)},
        )
    elif method_key == "rw":
        from pymatting import estimate_alpha_rw

        solved = estimate_alpha_rw(
            image,
            pm_trimap,
            laplacian_kwargs={"sigma": 0.03},
            cg_kwargs={"maxiter": int(cg_maxiter), "rtol": float(cg_rtol)},
        )
    else:
        from pymatting import estimate_alpha_sm

        solved = estimate_alpha_sm(image, pm_trimap)
    elapsed = time.perf_counter() - started

    solved = np.clip(np.asarray(solved, dtype=np.float32), 0.0, 1.0)
    alpha[trimap.unknown] = solved[trimap.unknown]
    alpha[trimap.sure_fg] = 1.0
    alpha[trimap.sure_bg] = 0.0
    return PyMattingAlphaResult(
        alpha=alpha,
        debug={
            "used": True,
            "method": method_key,
            "applied": True,
            "unknown_pixels": unknown_pixels,
            "sure_fg_pixels": int(trimap.sure_fg.sum()),
            "sure_bg_pixels": int(trimap.sure_bg.sum()),
            "image_space": image_space,
            "elapsed_sec": elapsed,
            "alpha_unknown_mean": float(alpha[trimap.unknown].mean()),
            "alpha_unknown_min": float(alpha[trimap.unknown].min()),
            "alpha_unknown_max": float(alpha[trimap.unknown].max()),
        },
    )


__all__ = [
    "PyMattingAlphaResult",
    "build_known_background_trimap",
    "estimate_stable_background_color",
    "estimate_alpha_with_pymatting",
    "estimate_known_background_alpha_with_pymatting",
    "normalize_known_background_field",
]
