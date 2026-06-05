"""Experimental PyMatting-backed alpha refinement for known-background bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

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
    isolated_residue_cleanup, isolated_residue_info = _known_background_isolated_bg_residue_cleanup_mask(
        image_srgb,
        bg,
        subject_seed=ownership.sure_fg,
        shadow_unknown=ownership.shadow_unknown,
        source_shadow_alpha=source_shadow_alpha,
    )
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
    if not bool(changed_bg_pixels.any()) and not bool(isolated_residue_cleanup.any()):
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
            "isolated_bg_residue_cleanup_pixels": 0,
            "isolated_bg_residue_cleanup": isolated_residue_info,
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
    normalized_u8[isolated_residue_cleanup] = bg.reshape(1, 3)
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
        "isolated_bg_residue_cleanup_pixels": int(isolated_residue_cleanup.sum()),
        "isolated_bg_residue_cleanup": isolated_residue_info,
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


def _known_background_isolated_bg_residue_cleanup_mask(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    subject_seed: np.ndarray,
    shadow_unknown: np.ndarray,
    source_shadow_alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find isolated screen-colored background dirt that should become known-B.

    Coherent scalar darkening tied to a subject is shadow evidence and stays
    unknown. Detached components that are close to the known-B color line and
    do not touch subject seeds are treated as dirty background residue before
    PyMatting sees them.
    """
    h, w = image_srgb.shape[:2]
    bg = np.asarray(background_color, dtype=np.float32).reshape(3)
    dominant = int(np.argmax(bg))
    others = [idx for idx in range(3) if idx != dominant]
    if float(bg[dominant]) < 64.0 or float(bg[dominant] - np.max(bg[others])) < 48.0:
        empty = np.zeros((h, w), dtype=bool)
        return empty, {
            "enabled": False,
            "reason": "background is not screen-dominant",
            "candidate_pixels": 0,
            "cleaned_pixels": 0,
        }

    image = image_srgb.astype(np.float32)
    other_max = np.max(image[..., others], axis=2)
    screen_like = (
        (image[..., dominant] > other_max + max(4.0, float(bg[dominant]) * 0.02))
        & (image[..., dominant] < float(bg[dominant]) * 0.90)
        & (other_max <= max(12.0, float(np.max(bg[others])) + 12.0))
    )
    shadow_replay = (1.0 - source_shadow_alpha[..., None]) * bg.reshape(1, 1, 3)
    shadow_error = np.mean(np.abs(shadow_replay - image), axis=2)
    candidate = (
        screen_like
        & (source_shadow_alpha > 0.035)
        & (shadow_error < 8.0)
        & ~np.asarray(subject_seed, dtype=bool)
        & ~np.asarray(shadow_unknown, dtype=bool)
    )
    if not bool(candidate.any()):
        return np.zeros((h, w), dtype=bool), {
            "enabled": True,
            "reason": "no isolated background residue candidates",
            "candidate_pixels": 0,
            "cleaned_pixels": 0,
        }

    subject_touch = cv2.dilate(subject_seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    shadow_touch = cv2.dilate(shadow_unknown.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    # Detached screen-darkened strips can be wider than pinpricks, but they are
    # still weak background residue when they do not touch the subject. The
    # absolute cap keeps this from flattening a large detached cast shadow.
    max_area = int(max(32, min(768, round(float(h * w) * 0.025))))
    cleanup = np.zeros((h, w), dtype=bool)
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_subject = bool((comp & subject_touch).any())
        touches_shadow = bool((comp & shadow_touch).any())
        keep = bool(area <= max_area and not touches_subject)
        if keep:
            cleanup |= comp
        components.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "touches_subject": touches_subject,
                "touches_shadow": touches_shadow,
                "keep": keep,
            }
        )
    components.sort(key=lambda item: (item["keep"], item["area"]), reverse=True)
    return cleanup, {
        "enabled": True,
        "reason": "" if bool(cleanup.any()) else "no disconnected small residue components",
        "candidate_pixels": int(candidate.sum()),
        "cleaned_pixels": int(cleanup.sum()),
        "component_max_area": int(max_area),
        "components": components[:12],
        "omitted_components": max(0, len(components) - 12),
    }


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


def _known_background_neutral_shadow_subject_evidence_release(
    background_color: np.ndarray,
    *,
    sure_fg: np.ndarray,
    shadow_unknown: np.ndarray,
    boundary_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Release neutral-background subject edges when shadow evidence is present.

    White/gray known-B assets do not have a saturated screen channel, so
    PyMatting can solve a strong neutral shadow as foreground unless the
    unknown band contains enough true subject-side color. This pass is a trimap
    evidence correction: it only runs when a coherent near-background shadow
    has already been found, then moves a shallow sure-FG rim into unknown.
    """
    h, w = sure_fg.shape
    release = np.zeros((h, w), dtype=bool)
    bg = np.asarray(background_color, dtype=np.float32).reshape(3)
    bg_chroma = float(bg.max() - bg.min())
    info: dict[str, Any] = {
        "enabled": True,
        "released_pixels": 0,
        "release_px": 0,
        "reason": "",
    }
    if float(bg.min()) < 180.0 or bg_chroma > 12.0:
        info["reason"] = "background is not light neutral"
        return release, info
    if not bool(sure_fg.any()) or not bool(shadow_unknown.any()):
        info["reason"] = "missing sure foreground or shadow evidence"
        return release, info

    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(shadow_unknown.astype(np.uint8), 8)
    largest_shadow = int(stats[1:, cv2.CC_STAT_AREA].max()) if labels_count > 1 else 0
    min_coherent_shadow = int(max(32.0, round(float(h * w) * 0.003)))
    if largest_shadow < min_coherent_shadow:
        info.update(
            {
                "reason": "no coherent neutral shadow component",
                "largest_shadow_component_area": largest_shadow,
                "coherent_shadow_min_area": min_coherent_shadow,
            }
        )
        return release, info

    fg_distance = cv2.distanceTransform(sure_fg.astype(np.uint8), cv2.DIST_L2, 3)
    fg_values = fg_distance[fg_distance > 0.0].astype(np.float32)
    if not fg_values.size:
        info["reason"] = "empty foreground distance field"
        return release, info
    fg_radius_p50 = float(np.percentile(fg_values, 50.0))
    boundary_px = float(boundary_info.get("boundary_band_px_effective", 2.0))

    dist_to_shadow = cv2.distanceTransform((~shadow_unknown).astype(np.uint8), cv2.DIST_L2, 3)
    # Keep the neutral-background pass narrow. It is a simple evidence release
    # for clean white/gray assets like B017, not a general solution for thick
    # outlined shadow cases such as B015.
    measured_px = 5.0
    thickness_cap = max(1.0, min(5.0, fg_radius_p50 * 0.45))
    release_px = int(max(1, round(min(measured_px, thickness_cap))))
    shadow_neighborhood_px = max(float(release_px) + 8.0, min(24.0, boundary_px + 6.0))
    near_shadow = dist_to_shadow <= shadow_neighborhood_px
    local_release = sure_fg & near_shadow & (fg_distance <= float(release_px))
    if bool(local_release.any()):
        release |= local_release

    info.update(
        {
            "released_pixels": int(release.sum()),
            "release_px": int(release_px),
            "measured_release_px": float(measured_px),
            "shadow_neighborhood_px": float(shadow_neighborhood_px),
            "boundary_band_px_effective": boundary_px,
            "foreground_radius_p50_px": fg_radius_p50,
            "largest_shadow_component_area": largest_shadow,
            "coherent_shadow_min_area": min_coherent_shadow,
            "reason": "" if bool(release.any()) else "no adjacent sure foreground rim",
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
    neutral_subject_evidence_release, neutral_subject_evidence_info = (
        _known_background_neutral_shadow_subject_evidence_release(
            bg,
            sure_fg=sure_fg,
            shadow_unknown=shadow_unknown,
            boundary_info=boundary_info,
        )
    )
    if bool(neutral_subject_evidence_release.any()):
        sure_fg = sure_fg & ~neutral_subject_evidence_release

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

    protected_transition = shadow_unknown | subject_transition | neutral_subject_evidence_release
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
    clean_bg_core = clean_bg & (dist_to_non_clean >= 2.0)
    sure_bg = (exterior_bg | enclosed_bg) & clean_bg_core
    if bool(enclosed_bg.any()):
        enclosed_dist = cv2.distanceTransform(enclosed_bg.astype(np.uint8), cv2.DIST_L2, 3)
        # Enclosed same-B cutouts should have a real clean interior. Thin
        # subject-adjacent same-screen islands, usually caused by contact shadow
        # or AA closing around a background-colored strip, are not reliable
        # sure-BG evidence and must stay unknown for PyMatting.
        enclosed_core_min_px = max(4.0, float(boundary_info["boundary_band_px_effective"]) + 2.0)
        enclosed_core = enclosed_bg & (enclosed_dist >= enclosed_core_min_px)
        sure_bg = (sure_bg & ~enclosed_bg) | (enclosed_core & clean_bg_core)
    else:
        enclosed_core_min_px = max(4.0, float(boundary_info["boundary_band_px_effective"]) + 2.0)
        enclosed_core = enclosed_bg
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
        "neutral_shadow_subject_evidence_release_pixels": int(neutral_subject_evidence_release.sum()),
        "neutral_shadow_subject_evidence": neutral_subject_evidence_info,
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
        "enclosed_bg_core_min_px": float(enclosed_core_min_px),
        "enclosed_bg_core_pixels": int(enclosed_core.sum()),
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
    trimap_mode: str = "standard",
    unknown_grow_px: int = 0,
    semantic_decision: dict[str, Any] | None = None,
    user_keep_mask: np.ndarray | None = None,
    user_remove_mask: np.ndarray | None = None,
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
    semantic_policy = str((semantic_decision or {}).get("enclosed_near_bg_policy") or "auto_default")
    mask_shape = image_srgb.shape[:2]
    # Brush masks are semantic constraints, not alpha mattes. Thresholding at
    # 0.5 makes antialiased PNG uploads behave like binary user intent while
    # keeping empty masks inert; remove wins on overlap to prevent accidental
    # opaque islands inside an explicitly erased region.
    keep_constraint = None if user_keep_mask is None else np.asarray(user_keep_mask, dtype=np.float32) > 0.5
    remove_constraint = None if user_remove_mask is None else np.asarray(user_remove_mask, dtype=np.float32) > 0.5
    if keep_constraint is not None and keep_constraint.shape != mask_shape:
        raise ValueError(f"user_keep_mask must have shape {mask_shape}, got {keep_constraint.shape}")
    if remove_constraint is not None and remove_constraint.shape != mask_shape:
        raise ValueError(f"user_remove_mask must have shape {mask_shape}, got {remove_constraint.shape}")
    keep_mask = np.zeros(mask_shape, dtype=bool) if keep_constraint is None else keep_constraint
    remove_mask = np.zeros(mask_shape, dtype=bool) if remove_constraint is None else remove_constraint
    user_mask_info: dict[str, Any] = {
        "source": "execute_user_mask",
        "keep_pixels": int(keep_mask.sum()),
        "remove_pixels": int(remove_mask.sum()),
        "conflict_pixels": int((keep_mask & remove_mask).sum()),
        "forced_subject_pixels": 0,
        "forced_background_pixels": 0,
        "applied": bool(keep_mask.any() or remove_mask.any()),
        "conflict_policy": "remove_overrides_keep",
    }
    ownership = _build_known_background_ownership(
        image_srgb,
        bg,
        bg_threshold=float(bg_threshold),
        fg_threshold=float(fg_threshold),
        boundary_band_px=int(boundary_band_px),
        adaptive=bool(adaptive),
        require_exact_bg=True,
    )

    # Route selection owns the semantic decision. If it requests the same-key
    # opaque body-outline trimap, execution consumes that contract directly
    # instead of reclassifying the asset here.
    body_outline_info: dict[str, Any] = {"enabled": trimap_mode == "same_key_opaque_body_outline"}
    semantic_info: dict[str, Any] = {
        "enclosed_near_bg_policy": semantic_policy,
        "forced_subject_pixels": 0,
        "forced_internal_unknown_pixels": 0,
        "internal_unknown": {
            "components": 0,
            "pixels": 0,
            "largest_component_pixels": 0,
            "method": "unknown_components_not_adjacent_to_exterior_sure_bg",
        },
        "forced_background_pixels": 0,
        "applied": False,
    }
    if trimap_mode == "same_key_opaque_body_outline":
        body_trimap, body_outline_info = _build_same_key_opaque_body_outline_trimap(
            image_srgb,
            bg,
            bg_threshold=float(bg_threshold),
            unknown_grow_px=int(unknown_grow_px),
        )
        body_sure_fg = body_trimap.sure_fg.copy()
        body_sure_bg = body_trimap.sure_bg.copy()
        body_unknown = body_trimap.unknown.copy()
        keep_only = keep_mask & ~remove_mask
        if keep_only.any():
            body_sure_fg[keep_only] = True
            body_sure_bg[keep_only] = False
            body_unknown[keep_only] = False
            user_mask_info["forced_subject_pixels"] = int(keep_only.sum())
        if remove_mask.any():
            body_sure_bg[remove_mask] = True
            body_sure_fg[remove_mask] = False
            body_unknown[remove_mask] = False
            user_mask_info["forced_background_pixels"] = int(remove_mask.sum())
        body_trimap = Trimap(sure_fg=body_sure_fg, sure_bg=body_sure_bg, unknown=body_unknown)
        return body_trimap, {
            "method": "same_key_opaque_body_outline",
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
            "same_key_opaque_body_outline": body_outline_info,
            "semantic_decision": semantic_info,
            "user_mask_decision": user_mask_info,
            "shadow_background": {
                **ownership.shadow_info,
                "unknown_ownership_pixels": int(ownership.shadow_unknown.sum()),
                "hard_ownership_pixels": 0,
                "screen_dominant_overlap_pixels": 0,
                "protected_foreground_overlap_pixels": 0,
            },
            "sure_fg_pixels": int(body_trimap.sure_fg.sum()),
            "sure_bg_pixels": int(body_trimap.sure_bg.sum()),
            "unknown_pixels": int(body_trimap.unknown.sum()),
        }

    # If a source has no clear foreground core, PyMatting has nothing stable to
    # propagate from. Keep the trimap valid but report the weak support.
    sure_fg = ownership.sure_fg.copy()
    sure_bg = ownership.sure_bg.copy()
    unknown = ownership.unknown.copy()
    if semantic_policy in {"subject", "transparent_hole"} and bool(ownership.enclosed_bg.any()):
        if semantic_policy == "subject":
            semantic_trimap, subject_info = _known_background_subject_semantic_trimap(
                ownership,
                boundary_band_px=int(boundary_band_px),
            )
            sure_fg = semantic_trimap.sure_fg
            sure_bg = semantic_trimap.sure_bg
            unknown = semantic_trimap.unknown
            semantic_info = {
                **semantic_info,
                "applied": True,
                "forced_subject_pixels": int(ownership.enclosed_bg.sum()),
                "forced_internal_unknown_pixels": int(subject_info["internal_unknown"]["pixels"]),
                "internal_unknown": subject_info["internal_unknown"],
                "subject_domain": subject_info,
            }
        else:
            forced_background = ownership.enclosed_bg & ownership.bg_candidate
            sure_bg[forced_background] = True
            sure_fg[forced_background] = False
            unknown[forced_background] = False
            semantic_info = {
                **semantic_info,
                "applied": True,
                "forced_background_pixels": int(forced_background.sum()),
            }
    keep_only = keep_mask & ~remove_mask
    if keep_only.any():
        sure_fg[keep_only] = True
        sure_bg[keep_only] = False
        unknown[keep_only] = False
        user_mask_info["forced_subject_pixels"] = int(keep_only.sum())
    if remove_mask.any():
        sure_bg[remove_mask] = True
        sure_fg[remove_mask] = False
        unknown[remove_mask] = False
        user_mask_info["forced_background_pixels"] = int(remove_mask.sum())
    trimap = Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)
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
        "semantic_decision": semantic_info,
        "user_mask_decision": user_mask_info,
        "sure_fg_pixels": int(trimap.sure_fg.sum()),
        "sure_bg_pixels": int(trimap.sure_bg.sum()),
        "unknown_pixels": int(trimap.unknown.sum()),
        "same_key_opaque_body_outline": body_outline_info,
        "shadow_background": {
            **ownership.shadow_info,
            "unknown_ownership_pixels": int(ownership.shadow_unknown.sum()),
            "hard_ownership_pixels": 0,
            "screen_dominant_overlap_pixels": 0,
            "protected_foreground_overlap_pixels": 0,
        },
    }


def _known_background_subject_semantic_trimap(
    ownership: KnownBOwnership,
    *,
    boundary_band_px: int,
) -> tuple[Trimap, dict[str, Any]]:
    """Build a trimap after accepting enclosed near-B pixels as subject.

    The subject semantic candidate changes the topology first: every pixel not
    connected to the exterior known background becomes one subject domain. The
    unknown band is then derived from that domain boundary, so opaque internal
    light material cannot remain as disconnected PyMatting unknown islands.
    """

    bg_candidate = np.asarray(ownership.bg_candidate, dtype=bool)
    exterior_bg = _flood_from_border(bg_candidate & ~np.asarray(ownership.enclosed_bg, dtype=bool))
    subject_domain = ~exterior_bg
    if bool(ownership.sure_fg.any()):
        subject_domain |= ownership.sure_fg
    subject_u8 = subject_domain.astype(np.uint8)
    boundary_px = max(1, int(round(float(boundary_band_px))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_px * 2 + 1, boundary_px * 2 + 1))
    eroded_subject = cv2.erode(subject_u8, kernel, iterations=1).astype(bool)
    dilated_subject = cv2.dilate(subject_u8, kernel, iterations=1).astype(bool)
    boundary_unknown = (dilated_subject & ~eroded_subject) & ~ownership.enclosed_bg
    unknown = (boundary_unknown | ownership.shadow_unknown | ownership.protected_transition) & ~eroded_subject
    sure_fg = subject_domain & ~unknown
    sure_bg = ~(sure_fg | unknown)

    internal_unknown, internal_unknown_info = _internal_unknown_components(unknown, sure_bg)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(subject_domain.astype(np.uint8), 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if labels_count > 1 else 0
    return Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown), {
        "method": "subject_domain_then_boundary_unknown",
        "pixels": int(subject_domain.sum()),
        "components": max(0, int(labels_count) - 1),
        "largest_component_pixels": largest,
        "boundary_band_px": int(boundary_px),
        "unknown_pixels": int(unknown.sum()),
        "internal_unknown": internal_unknown_info,
    }


def _internal_unknown_components(
    unknown: np.ndarray,
    sure_bg: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return unknown components that are not part of the exterior boundary band."""

    unknown_mask = np.asarray(unknown, dtype=bool)
    if not bool(unknown_mask.any()):
        return np.zeros_like(unknown_mask, dtype=bool), {
            "components": 0,
            "pixels": 0,
            "largest_component_pixels": 0,
            "method": "unknown_components_not_adjacent_to_exterior_sure_bg",
        }
    exterior_bg = _flood_exterior(np.asarray(sure_bg, dtype=bool))
    exterior_contact = cv2.dilate(
        exterior_bg.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(unknown_mask.astype(np.uint8), 8)
    internal = np.zeros_like(unknown_mask, dtype=bool)
    largest = 0
    components = 0
    for label in range(1, labels_count):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if bool((comp & exterior_contact).any()):
            continue
        internal |= comp
        components += 1
        largest = max(largest, area)
    return internal, {
        "components": int(components),
        "pixels": int(internal.sum()),
        "largest_component_pixels": int(largest),
        "method": "unknown_components_not_adjacent_to_exterior_sure_bg",
    }


def _flood_exterior(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    work = mask.astype(np.uint8).copy()
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
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
    return work == 2


def _largest_filled_component(mask: np.ndarray, *, close_px: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    h, w = mask_bool.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), 8)
    if labels_count <= 1:
        return np.zeros((h, w), dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == label
    component = np.asarray(ndimage.binary_fill_holes(component), dtype=bool)
    close = max(0, int(close_px))
    if close > 0:
        component = cv2.morphologyEx(
            component.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=close,
        ).astype(bool)
        component = np.asarray(ndimage.binary_fill_holes(component), dtype=bool)
    return component


def build_known_background_hard_edge_boundary_mask(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    bg_threshold: float = 7.0,
    fg_threshold: float = 24.0,
    shadow_close_px: int = 1,
    subject_close_px: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find a hard opaque subject boundary across known-background shadow.

    This is the production form of the base-close experiment: flood only
    high-confidence exterior known background, identify scalar-darkened
    known-B shadow as background support, close tiny gaps in that shadow
    support, then take the largest remaining filled component as the hard
    subject boundary. It deliberately returns a binary hard-edge support mask;
    it does not solve antialiasing or shadow alpha.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    bg = np.asarray(background_color, dtype=np.uint8).reshape(3)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3)).reshape(3)
    distance = oklab_distance(lab, bg_lab).astype(np.float32)
    rgb_distance = np.sqrt(
        np.sum((image_srgb.astype(np.float32) - bg.astype(np.float32).reshape(1, 1, 3)) ** 2, axis=2)
    ).astype(np.float32)
    effective_bg_threshold = float(max(bg_threshold, 0.0))
    bg_close = (distance <= effective_bg_threshold) | (rgb_distance <= max(5.0, effective_bg_threshold + 2.0))
    exterior_bg = _flood_from_border(bg_close)

    broad_support = _largest_filled_component(~exterior_bg, close_px=1)
    if not bool(broad_support.any()):
        empty = np.zeros(image_srgb.shape[:2], dtype=bool)
        return empty, {
            "enabled": True,
            "method": "known_background_hard_edge_boundary_base_close",
            "accepted": False,
            "reason": "missing non-background component",
            "background_color": [int(c) for c in bg],
            "bg_threshold": float(effective_bg_threshold),
            "fg_threshold": float(fg_threshold),
            "mask_pixels": 0,
        }

    dist_to_broad_edge = cv2.distanceTransform(broad_support.astype(np.uint8), cv2.DIST_L2, 3)
    strong_material = broad_support & (distance >= float(fg_threshold)) & (dist_to_broad_edge > 2.0)
    min_seed_pixels = int(max(16, round(float(image_srgb.shape[0] * image_srgb.shape[1]) * 0.0005)))
    if int(strong_material.sum()) >= min_seed_pixels:
        subject_seed = _largest_filled_component(strong_material, close_px=1)
        seed_source = "strong_material_core"
    else:
        subject_seed = broad_support & (dist_to_broad_edge > 4.0)
        subject_seed = _largest_filled_component(subject_seed, close_px=1)
        seed_source = "broad_support_inset_fallback"

    shadow_bg_raw, shadow_info = _known_background_shadow_like_background_mask(
        image_srgb,
        bg,
        subject_seed=subject_seed,
    )
    shadow_close = max(0, int(shadow_close_px))
    if shadow_close > 0 and bool(shadow_bg_raw.any()):
        shadow_bg = cv2.morphologyEx(
            shadow_bg_raw.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=shadow_close,
        ).astype(bool)
    else:
        shadow_bg = shadow_bg_raw

    hard_bg = exterior_bg | shadow_bg
    subject_mask = _largest_filled_component(~hard_bg, close_px=int(subject_close_px))
    accepted = bool(subject_mask.any())
    return subject_mask, {
        "enabled": True,
        "method": "known_background_hard_edge_boundary_base_close",
        "accepted": accepted,
        "reason": "" if accepted else "empty subject after exterior and shadow background removal",
        "background_color": [int(c) for c in bg],
        "bg_threshold": float(effective_bg_threshold),
        "fg_threshold": float(fg_threshold),
        "shadow_close_px": int(shadow_close),
        "subject_close_px": int(max(0, subject_close_px)),
        "bg_close_pixels": int(bg_close.sum()),
        "exterior_bg_pixels": int(exterior_bg.sum()),
        "broad_support_pixels": int(broad_support.sum()),
        "subject_seed_source": seed_source,
        "subject_seed_pixels": int(subject_seed.sum()),
        "shadow_bg_raw_pixels": int(shadow_bg_raw.sum()),
        "shadow_bg_pixels": int(shadow_bg.sum()),
        "shadow_bg_closed_pixels": int((shadow_bg & ~shadow_bg_raw).sum()),
        "hard_bg_pixels": int(hard_bg.sum()),
        "mask_pixels": int(subject_mask.sum()),
        "shadow_background": shadow_info,
    }


def _median_smooth_1d(values: np.ndarray, k: int = 9) -> np.ndarray:
    radius = k // 2
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    return np.asarray([np.median(padded[i : i + k]) for i in range(len(values))], dtype=np.float32)


def _same_key_opaque_body_outline_trace(
    image_srgb: np.ndarray,
    background_color: np.ndarray | tuple[int, int, int],
    *,
    bg_threshold: float,
) -> dict[str, Any]:
    lower = _same_key_opaque_lower_perimeter_ridge_trace(
        image_srgb,
        background_color,
        bg_threshold=float(bg_threshold),
    )
    if lower.get("accepted", False):
        return lower
    closed = _same_key_opaque_closed_plateau_outline_trace(
        image_srgb,
        background_color,
        bg_threshold=float(bg_threshold),
    )
    if closed.get("accepted", False):
        closed["fallback_from"] = lower
        return closed
    return lower


def _same_key_opaque_lower_perimeter_ridge_trace(
    image_srgb: np.ndarray,
    background_color: np.ndarray | tuple[int, int, int],
    *,
    bg_threshold: float,
) -> dict[str, Any]:
    h, w = image_srgb.shape[:2]
    bg = np.asarray(background_color, dtype=np.uint8).reshape(3)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3)).reshape(3)
    dE = oklab_distance(lab, bg_lab).astype(np.float32)
    # Same-key lower-ridge shape tracing uses a relaxed exterior threshold.
    # The strict Known-B threshold is still used later for matting, but shape
    # extraction needs to flood smooth same-screen exterior drift; otherwise
    # darkened blue/green background strips attach to the body component and
    # become proxy-painted subject residue.
    shape_bg_threshold = max(float(bg_threshold), 7.0)
    exterior = _flood_exterior(dE <= shape_bg_threshold)
    support = (~exterior).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    if labels_count <= 1:
        return {"accepted": False, "reason": "missing non-background component"}

    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8)
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
    aspect = float(comp_w / max(1, comp_h))

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "accepted": False,
            "reason": "missing component contour",
            "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
            "component_aspect_ratio": aspect,
        }
    broad_fill = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(broad_fill, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)

    gray = cv2.cvtColor(image_srgb, cv2.COLOR_RGB2GRAY)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    y0 = int(np.clip(round(y + comp_h * 0.90), 0, h - 1))
    y1 = int(np.clip(round(y + comp_h * 0.985), y0 + 1, h))
    band = gy[y0:y1, :]
    if band.size == 0:
        return {
            "accepted": False,
            "reason": "empty lower ridge band",
            "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
            "component_aspect_ratio": aspect,
        }

    strength = band.max(axis=0)
    ridge_y = (band.argmax(axis=0) + y0).astype(np.float32)
    columns = np.arange(w)
    col_support = broad_fill.any(axis=0)
    trusted = col_support & (strength >= 18.0)
    support_columns = int(col_support.sum())
    trusted_columns = int(trusted.sum())
    trusted_fraction = float(trusted_columns / max(1, support_columns))
    if support_columns <= 0:
        return {
            "accepted": False,
            "reason": "missing support columns",
            "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
            "component_aspect_ratio": aspect,
        }

    if trusted_columns > 0:
        interpolated = np.interp(columns, columns[trusted], ridge_y[trusted])
    else:
        interpolated = np.full(w, float(y1 - 1), dtype=np.float32)
    line = np.rint(_median_smooth_1d(interpolated, 9)).astype(np.int32)
    line = np.clip(line, y0, y1 - 1)
    line_values = line[col_support]
    line_range = int(line_values.max() - line_values.min()) if line_values.size else 0
    # Same-key opaque plateau proves material opacity; this extra route signal
    # proves that the current outline recipe has enough measured perimeter-ridge
    # evidence to close the foreground body before PyMatting solves AA/shadow.
    # Lower-ridge buttons have a sustained horizontal perimeter signal near the
    # bottom. Closed shapes such as circles can produce a short edge in the same
    # band, but their trusted-column fraction stays lower; 0.58 keeps that
    # fallback path available while accepting measured button ridges.
    accepted = bool(trusted_fraction >= 0.58 and line_range <= max(6, int(round(comp_h * 0.08))))
    reason = "" if accepted else "body outline ridge is not sufficiently continuous for this recipe"
    yy, xx = np.indices((h, w))
    body_fill = broad_fill.astype(bool) & (yy <= line[xx])
    return {
        "accepted": accepted,
        "reason": reason,
        "outline_recipe": "lower_perimeter_ridge",
        "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
        "component_aspect_ratio": aspect,
        "shape_bg_threshold": float(shape_bg_threshold),
        "matte_bg_threshold": float(bg_threshold),
        "ridge_band_y": [int(y0), int(y1 - 1)],
        "ridge_strength_min": 18.0,
        "support_columns": support_columns,
        "trusted_columns": trusted_columns,
        "trusted_fraction": trusted_fraction,
        "line_y_min": int(line_values.min()),
        "line_y_max": int(line_values.max()),
        "line_y_median": float(np.median(line_values)),
        "line_y_range": line_range,
        "line": line,
        "broad_fill": broad_fill.astype(bool),
        "body_fill": body_fill,
        "unknown_domain": broad_fill.astype(bool),
    }


def _same_key_opaque_closed_plateau_outline_trace(
    image_srgb: np.ndarray,
    background_color: np.ndarray | tuple[int, int, int],
    *,
    bg_threshold: float,
) -> dict[str, Any]:
    del bg_threshold
    from .keyer import KeyerThresholds, chromatic_key_alpha

    h, w = image_srgb.shape[:2]
    bg = np.asarray(background_color, dtype=np.uint8).reshape(3)
    key_alpha = chromatic_key_alpha(image_srgb, bg, KeyerThresholds(bg_max=8.0, fg_min=18.0))
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3)).reshape(3)
    delta_ab = lab[..., 1:] - bg_lab[1:]
    ab_distance = np.sqrt(np.sum(delta_ab * delta_ab, axis=-1)).astype(np.float32) * 100.0
    near_plateau = (ab_distance <= 12.0) & (key_alpha >= 0.16)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(near_plateau.astype(np.uint8), 8)
    if labels_count <= 1:
        return {"accepted": False, "reason": "missing same-key opaque plateau component"}

    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8)
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
    area = int(stats[label, cv2.CC_STAT_AREA])
    min_area = max(32, int(round(float(h * w) * 0.01)))
    if area < min_area:
        return {
            "accepted": False,
            "reason": "same-key opaque plateau component is too small",
            "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
            "component_area": area,
            "component_min_area": min_area,
        }

    closed = cv2.morphologyEx(component, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "accepted": False,
            "reason": "same-key opaque plateau has no closable contour",
            "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
            "component_area": area,
            "component_min_area": min_area,
        }
    contour = max(contours, key=cv2.contourArea)
    body_fill_u8 = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(body_fill_u8, [contour], -1, 1, thickness=cv2.FILLED)
    body_fill = body_fill_u8.astype(bool)
    fill_pixels = int(body_fill.sum())
    plateau_fill_fraction = float((near_plateau & body_fill).sum() / max(1, fill_pixels))

    gray = cv2.cvtColor(image_srgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    kernel = np.ones((3, 3), dtype=np.uint8)
    perimeter_ring = (
        cv2.dilate(body_fill_u8, kernel, iterations=2).astype(bool)
        & ~cv2.erode(body_fill_u8, kernel, iterations=2).astype(bool)
    )
    perimeter_pixels = int(perimeter_ring.sum())
    perimeter_edge_pixels = int((perimeter_ring & (mag >= 18.0)).sum())
    perimeter_edge_fraction = float(perimeter_edge_pixels / max(1, perimeter_pixels))
    exterior = _flood_exterior(oklab_distance(lab, bg_lab).astype(np.float32) <= 3.5)
    support = ~exterior
    unknown_domain = support | cv2.dilate(body_fill_u8, kernel, iterations=1).astype(bool)
    accepted = bool(plateau_fill_fraction >= 0.55 and perimeter_edge_fraction >= 0.18)
    reason = "" if accepted else "same-key plateau outline is not sufficiently closed or edged"
    return {
        "accepted": accepted,
        "reason": reason,
        "outline_recipe": "closed_plateau_outline",
        "component_bbox_xyxy": [x, y, x + comp_w, y + comp_h],
        "component_area": area,
        "component_min_area": min_area,
        "fill_pixels": fill_pixels,
        "plateau_fill_fraction": plateau_fill_fraction,
        "perimeter_edge_threshold": 18.0,
        "perimeter_pixels": perimeter_pixels,
        "perimeter_edge_pixels": perimeter_edge_pixels,
        "perimeter_edge_fraction": perimeter_edge_fraction,
        "body_fill": body_fill,
        "support": support,
        "unknown_domain": unknown_domain,
    }


def analyze_same_key_opaque_body_outline(
    image_srgb: np.ndarray,
    background_color: np.ndarray | tuple[int, int, int],
    *,
    bg_threshold: float = 3.5,
) -> dict[str, Any]:
    trace = _same_key_opaque_body_outline_trace(
        image_srgb,
        background_color,
        bg_threshold=float(bg_threshold),
    )
    return _public_same_key_outline_trace(trace)


def _public_same_key_outline_trace(trace: dict[str, Any]) -> dict[str, Any]:
    hidden = {"line", "broad_fill", "body_fill", "unknown_domain", "support"}
    public: dict[str, Any] = {}
    for key, value in trace.items():
        if key in hidden:
            continue
        if key == "fallback_from" and isinstance(value, dict):
            public[key] = _public_same_key_outline_trace(value)
        else:
            public[key] = value
    return public


def _same_key_opaque_stroke_core_from_component(
    image_srgb: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a proxy seed whose edge lands near the measured outline middle."""
    component = np.asarray(component, dtype=bool)
    if not component.any():
        return component, {
            "enabled": False,
            "reason": "empty component",
            "stroke_inset_px": 0,
            "proxy_inset_px": 0,
            "stroke_pixels": 0,
        }

    ys, xs = np.where(component)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    min_dim = max(1, min(bbox_w, bbox_h))
    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
    max_dist = float(dist[component].max())
    if max_dist < 3.0:
        core = component & (dist > 1.0)
        if core.any():
            component_for_return = core
        else:
            component_for_return = component
        return component_for_return, {
            "enabled": False,
            "reason": "component too thin to measure outline stroke",
            "stroke_inset_px": 0,
            "proxy_inset_px": 1 if core.any() else 0,
            "stroke_pixels": int((component & ~component_for_return).sum()),
        }

    # Probe only the first few percent of the measured component radius. The
    # signal is the abrupt Oklab color falloff from the hard outline into the
    # button body; limiting the search prevents interior gradients/highlights
    # from being mistaken for extra stroke.
    max_probe_px = int(max(4, min(18, round(float(min_dim) * 0.08), np.floor(max_dist * 0.45))))
    core_floor = min(float(max_probe_px + 2), max(3.0, max_dist * 0.45))
    interior_seed = component & (dist >= core_floor)
    if int(interior_seed.sum()) < 32:
        interior_seed = component & (dist >= max_dist * 0.35)
    if int(interior_seed.sum()) < 32:
        proxy_inset_px = 1
        core = component & (dist > float(proxy_inset_px))
        core = np.asarray(ndimage.binary_fill_holes(core), dtype=bool)
        if not core.any():
            core = component
            proxy_inset_px = 0
        return core, {
            "enabled": False,
            "reason": "not enough interior pixels to measure outline stroke",
            "stroke_inset_px": 0,
            "proxy_inset_px": int(proxy_inset_px),
            "stroke_pixels": int((component & ~core).sum()),
            "max_probe_px": int(max_probe_px),
        }

    lab = srgb_to_oklab(image_srgb)
    interior_lab = np.median(lab[interior_seed], axis=0)
    body_delta = oklab_distance(lab, interior_lab).astype(np.float32)
    ring_medians: list[float] = []
    for radius in range(1, max_probe_px + 1):
        ring = component & (dist > float(radius - 1)) & (dist <= float(radius))
        ring_medians.append(float(np.median(body_delta[ring])) if ring.any() else 0.0)

    if len(ring_medians) < 2:
        stroke_inset_px = 0
        strongest_drop = 0.0
    else:
        drops = np.asarray(ring_medians[:-1], dtype=np.float32) - np.asarray(ring_medians[1:], dtype=np.float32)
        search_limit = max(2, min(len(drops), int(round(float(max_probe_px) * 0.65))))
        limited_drops = drops[:search_limit]
        best_index = int(np.argmax(limited_drops))
        strongest_drop = float(limited_drops[best_index])
        core_delta_floor = float(np.percentile(body_delta[interior_seed], 75) + 6.0)
        before = ring_medians[best_index]
        after = ring_medians[best_index + 1]
        # A valid outline boundary has a visible perceptual drop and lands near
        # the measured body-color spread. This protects soft body gradients from
        # being over-eroded while still excluding thick dark strokes.
        if strongest_drop >= 4.0 and before >= core_delta_floor and after <= before - 4.0:
            stroke_inset_px = best_index + 1
        else:
            stroke_inset_px = 0

    # Proxy painting only needs to stay off the outer AA boundary. Placing the
    # proxy edge just inside the measured stroke midpoint keeps exterior AA in
    # the solve while avoiding an over-large inset that would let same-key body
    # pixels be solved as translucent. If the stroke signal is weak, default to
    # a minimal 1px inset instead of trusting the outer component edge.
    proxy_inset_px = max(1, int(stroke_inset_px // 2) + 1) if stroke_inset_px > 0 else 1
    core = component & (dist > float(proxy_inset_px))
    if core.any():
        core = np.asarray(ndimage.binary_fill_holes(core), dtype=bool)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
        if labels_count <= 1:
            core = component
            proxy_inset_px = 0
        else:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            core = labels == label
    else:
        core = component
        proxy_inset_px = 0

    stroke_mask = component & ~core
    return core, {
        "enabled": bool(stroke_inset_px > 0),
        "method": "oklab_radial_drop",
        "stroke_inset_px": int(stroke_inset_px),
        "proxy_inset_px": int(proxy_inset_px),
        "stroke_pixels": int(stroke_mask.sum()),
        "max_probe_px": int(max_probe_px),
        "interior_seed_pixels": int(interior_seed.sum()),
        "ring_median_body_delta": [round(float(v), 3) for v in ring_medians],
        "strongest_drop": round(float(strongest_drop), 3),
    }


def _build_same_key_opaque_body_outline_trimap(
    image_srgb: np.ndarray,
    background_color: np.ndarray,
    *,
    bg_threshold: float,
    unknown_grow_px: int,
) -> tuple[Trimap, dict[str, Any]]:
    """Build a body-outline trimap for a same-key opaque outline recipe.

    This profile has already established that the button is not transparent.
    Route analysis has also established that this image has the measured outline
    evidence required by the current recipe. Pixels inside the outline are
    sure-FG, the outline plus exterior shadow/AA are unknown, and clean exterior
    known-B remains sure-BG. This helper is an execution recipe, not an asset
    classifier; failures are explicit so callers do not silently get a different
    trimap mode from the route contract.
    """
    h, w = image_srgb.shape[:2]
    trace = _same_key_opaque_body_outline_trace(
        image_srgb,
        background_color,
        bg_threshold=float(bg_threshold),
    )
    if not trace.get("accepted", False):
        raise ValueError(f"same-key body-outline trimap route contract failed: {trace.get('reason', 'unknown')}")
    body_fill = trace["body_fill"]
    body_labels_count, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body_fill.astype(np.uint8), 8)
    if body_labels_count <= 1:
        raise ValueError("same-key body-outline trimap body fill is empty after ridge clip")
    body_label = 1 + int(np.argmax(body_stats[1:, cv2.CC_STAT_AREA]))
    body_fill = body_labels == body_label

    dist = cv2.distanceTransform(body_fill.astype(np.uint8), cv2.DIST_L2, 3)
    sure_fg = body_fill & (dist >= 2.0)
    unknown_domain = np.asarray(trace.get("unknown_domain", body_fill), dtype=bool)
    unknown = unknown_domain & ~sure_fg
    if int(unknown_grow_px) > 0:
        grow_kernel = np.ones((3, 3), dtype=np.uint8)
        grown = cv2.dilate((sure_fg | unknown).astype(np.uint8), grow_kernel, iterations=int(unknown_grow_px)).astype(bool)
        unknown = grown & ~sure_fg
    sure_bg = ~(sure_fg | unknown)
    trimap = Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)
    return trimap, {
        "enabled": True,
        **_public_same_key_outline_trace(trace),
        "unknown_grow_px": int(unknown_grow_px),
    }


def build_same_key_opaque_proxy_subject_mask(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    bg_threshold: float,
    expand_px: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the subject-domain mask used for same-key proxy color replacement.

    Same-key opaque UI can have body pixels that are almost identical to the
    screen color. The proxy edge is measured from the component and usually
    placed near the outline midpoint. Callers can request a small expansion for
    legacy recipes, but the same final mask is always reused for proxy painting
    and source-color restoration so the two phases cannot disagree.
    """
    bg = np.asarray(background_color, dtype=np.uint8)
    trace = _same_key_opaque_body_outline_trace(
        image_srgb,
        bg,
        bg_threshold=float(bg_threshold),
    )
    if not trace.get("accepted", False):
        raise ValueError(f"same-key proxy subject route contract failed: {trace.get('reason', 'unknown')}")

    stroke_info: dict[str, Any] = {
        "enabled": False,
        "stroke_inset_px": 0,
        "stroke_pixels": 0,
    }
    if trace.get("outline_recipe") == "lower_perimeter_ridge" and "broad_fill" in trace:
        # For same-key lower-ridge assets the relaxed dE7 component is the
        # measured subject extent. Before proxy painting, place the proxy edge
        # near the measured stroke midpoint; the outer AA must stay in PyMatting
        # while the proxy still covers enough same-key body to avoid translucency.
        measured_component = np.asarray(trace["broad_fill"], dtype=bool)
        body_fill, stroke_info = _same_key_opaque_stroke_core_from_component(image_srgb, measured_component)
        proxy_source = "relaxed_component_stroke_core"
    else:
        measured_component = np.asarray(trace["body_fill"], dtype=bool)
        body_fill, stroke_info = _same_key_opaque_stroke_core_from_component(image_srgb, measured_component)
        proxy_source = "body_fill_stroke_core" if stroke_info.get("enabled", False) else "body_fill"
    body_labels_count, body_labels, body_stats, _ = cv2.connectedComponentsWithStats(body_fill.astype(np.uint8), 8)
    if body_labels_count <= 1:
        raise ValueError("same-key proxy subject mask body fill is empty after outline trace")
    body_label = 1 + int(np.argmax(body_stats[1:, cv2.CC_STAT_AREA]))
    body_fill = body_labels == body_label

    proxy_mask = body_fill
    expand = max(0, int(expand_px))
    if expand > 0:
        # The expansion covers one-pixel antialias stair steps that otherwise
        # remain same-key blue/green and are accepted as sure background by the
        # standard trimap. It is deliberately small; broader growth would start
        # painting outline/shadow evidence as solid subject.
        kernel = np.ones((3, 3), dtype=np.uint8)
        proxy_mask = cv2.dilate(body_fill.astype(np.uint8), kernel, iterations=expand).astype(bool)
        proxy_mask = np.asarray(ndimage.binary_fill_holes(proxy_mask), dtype=bool)

    return proxy_mask, {
        "enabled": True,
        **_public_same_key_outline_trace(trace),
        "method": "same_key_opaque_proxy_subject_mask",
        "proxy_source": proxy_source,
        "stroke_outline": stroke_info,
        "expand_px": int(expand),
        "body_pixels": int(body_fill.sum()),
        "mask_pixels": int(proxy_mask.sum()),
        "expanded_pixels": int((proxy_mask & ~body_fill).sum()),
    }


def build_same_key_opaque_inner_opaque_mask(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    bg_threshold: float,
    outer_guard_px: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the same-key opaque material region protected from alpha collapse.

    Proxy painting changes the near-background body color, but the body/stroke
    transition is still opaque UI material. This mask is deliberately derived
    from the original outline support, then pulled back from only the exterior
    edge. The outer guard leaves true silhouette AA and shadow to the existing
    PyMatting/ShadowPatch path while allowing inner material AA to be floored to
    alpha=1 after PyMatting has produced its raw solve.
    """
    bg = np.asarray(background_color, dtype=np.uint8)
    trace = _same_key_opaque_body_outline_trace(
        image_srgb,
        bg,
        bg_threshold=float(bg_threshold),
    )
    if not trace.get("accepted", False):
        raise ValueError(f"same-key inner opaque mask route contract failed: {trace.get('reason', 'unknown')}")

    if trace.get("outline_recipe") == "lower_perimeter_ridge" and "broad_fill" in trace:
        support = np.asarray(trace["broad_fill"], dtype=bool)
        support_source = "relaxed_component"
    elif "support" in trace:
        support = np.asarray(trace["support"], dtype=bool)
        support_source = "closed_outline_support"
    else:
        support = np.asarray(trace["body_fill"], dtype=bool)
        support_source = "body_fill"

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    if labels_count <= 1:
        raise ValueError("same-key inner opaque mask support is empty after outline trace")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    support = labels == label
    support = np.asarray(ndimage.binary_fill_holes(support), dtype=bool)

    guard = max(0.0, float(outer_guard_px))
    dist_to_exterior = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    mask = support & (dist_to_exterior > guard)
    guarded = support & ~mask
    return mask, {
        "enabled": True,
        **_public_same_key_outline_trace(trace),
        "method": "same_key_opaque_inner_opaque_mask",
        "support_source": support_source,
        "outer_guard_px": float(guard),
        "support_pixels": int(support.sum()),
        "mask_pixels": int(mask.sum()),
        "outer_guard_pixels": int(guarded.sum()),
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
    weak_tail_min = max(0.75 / 255.0, strength_floor + 0.25 / 255.0)
    weak_tail_candidate = (strength >= weak_tail_min) & (err <= err_max) & (off_excess <= off_excess_max)
    if bool(weak_tail_candidate.any()) and bool(kept.any()):
        tail_labels_count, tail_labels, tail_stats, _ = cv2.connectedComponentsWithStats(
            weak_tail_candidate.astype(np.uint8),
            8,
        )
        seeded_labels = np.unique(tail_labels[kept])
        seeded_labels = seeded_labels[seeded_labels > 0]
        connected_tail = (
            np.isin(tail_labels, seeded_labels) & weak_tail_candidate
            if seeded_labels.size
            else np.zeros_like(kept, dtype=bool)
        )
        tail_component_areas = [
            int(tail_stats[label, cv2.CC_STAT_AREA])
            for label in seeded_labels
            if 0 < label < tail_labels_count
        ]
    else:
        connected_tail = np.zeros_like(kept, dtype=bool)
        tail_component_areas = []
    # The near-subject pass provides a coherent anchor; this connected pass
    # follows only scalar known-B darkening in the same component so soft shadow
    # tails can fade to zero instead of being clipped by the near-subject cap.
    shadow_mask = kept | connected_tail
    return shadow_mask, {
        "enabled": True,
        "reason": "" if bool(shadow_mask.any()) else "no scalar-darkening background near subject",
        "pixels": int(shadow_mask.sum()),
        "candidate_pixels": int(candidate.sum()),
        "anchor_pixels": int(kept.sum()),
        "weak_tail_candidate_pixels": int(weak_tail_candidate.sum()),
        "connected_tail_pixels": int((connected_tail & ~kept).sum()),
        "weak_tail_strength_min": float(weak_tail_min),
        "connected_tail_component_areas": tail_component_areas[:12],
        "connected_tail_omitted_components": max(0, len(tail_component_areas) - 12),
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


def estimate_stable_background_color(
    image_srgb: np.ndarray,
    *,
    seed_bg: tuple[int, int, int] | np.ndarray | None = None,
    seed_source: str = "external",
    seed_info: dict[str, Any] | None = None,
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    """Estimate the single known-B/sure-background color for routing.

    The corners/border mode are only seed evidence: they find a plausible
    background color family without looking at the subject. The final known-B is
    computed from structurally sure background support grown from that seed, so
    route params, normalization, trimap, unmix, and ShadowPatch all use one
    color contract.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    if seed_bg is None:
        selected_seed_bg, selected_seed_info = _estimate_known_background_seed(image_srgb)
        support_bg_threshold_cap = None
    else:
        selected_seed_bg = tuple(int(np.clip(c, 0, 255)) for c in np.asarray(seed_bg).reshape(3))
        selected_seed_info = {
            "accepted": True,
            "reason": "accepted_external_seed",
            "source": str(seed_source),
            "background_color": [int(c) for c in selected_seed_bg],
            **(seed_info or {}),
        }
        # External route-layer seeds are anchors, not permission to grow the
        # background family across the subject. The stable estimator still
        # refines the final color from support, but support must stay inside
        # the requested known-B foreground threshold so dominant subject colors
        # cannot take over when UI fills most of the frame.
        support_bg_threshold_cap = 24.0
    if not selected_seed_info.get("accepted", False):
        return selected_seed_bg, {
            **selected_seed_info,
            "accepted": False,
            "reason": selected_seed_info.get("reason", "corner/background border is unstable"),
            "source": "sure_bg_mode",
            "seed": selected_seed_info,
        }

    support, support_info = _known_background_support_from_seed(
        image_srgb,
        selected_seed_bg,
        bg_threshold_cap=support_bg_threshold_cap,
    )
    min_support = max(32, int(round(float(image_srgb.shape[0] * image_srgb.shape[1]) * 0.01)))
    if int(support.sum()) < min_support:
        return selected_seed_bg, {
            "accepted": False,
            "reason": "insufficient sure background support",
            "source": "sure_bg_mode",
            "seed": selected_seed_info,
            **support_info,
        }

    known_bg, color_info = _known_background_color_from_support(image_srgb, support)
    return known_bg, {
        "accepted": True,
        "reason": "accepted",
        "source": "sure_bg_mode",
        "seed": selected_seed_info,
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
    # Smooth generated/studio backgrounds can drift a few RGB units from one
    # corner to another while still being a valid single known-B after the
    # normalization prepass. Keep the internal-variance gate tight so textured
    # or noisy photo borders do not pass as deterministic known background.
    if corner_agreement <= 6.0 and sigma <= 6.0:
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
    *,
    bg_threshold_cap: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed = np.asarray(seed_bg, dtype=np.uint8)
    lab = srgb_to_oklab(image_srgb)
    seed_lab = srgb_to_oklab(seed.reshape(1, 1, 3))[0, 0]
    distance = oklab_distance(lab, seed_lab)
    thresholds = _adaptive_known_background_thresholds(distance, image_srgb, seed, 3.5, 24.0)
    if bg_threshold_cap is not None:
        capped_bg_threshold = min(float(thresholds["bg_threshold_effective"]), float(bg_threshold_cap))
        thresholds = {
            **thresholds,
            "bg_threshold_uncapped": float(thresholds["bg_threshold_effective"]),
            "bg_threshold_effective": float(capped_bg_threshold),
            "bg_threshold_cap": float(bg_threshold_cap),
            "bg_threshold_source": "external_seed_cap",
        }
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
    "analyze_same_key_opaque_body_outline",
    "build_known_background_hard_edge_boundary_mask",
    "build_same_key_opaque_inner_opaque_mask",
    "build_same_key_opaque_proxy_subject_mask",
    "build_known_background_trimap",
    "estimate_stable_background_color",
    "estimate_alpha_with_pymatting",
    "estimate_known_background_alpha_with_pymatting",
    "normalize_known_background_field",
]
