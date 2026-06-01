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
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0]
    distance = oklab_distance(lab, bg_lab)
    thresholds = (
        _adaptive_known_background_thresholds(distance, image_srgb, bg, bg_threshold, fg_threshold)
        if adaptive
        else _fixed_known_background_thresholds(bg_threshold, fg_threshold)
    )
    effective_bg_threshold = float(thresholds["bg_threshold_effective"])
    effective_fg_threshold = float(thresholds["fg_threshold_effective"])

    bg_close = distance <= effective_bg_threshold
    exterior_bg = _flood_from_border(bg_close)
    enclosed_bg, enclosed_info = _filter_enclosed_background_components(bg_close & ~exterior_bg)
    high_conf_bg = (exterior_bg | enclosed_bg) & bg_close
    if int(high_conf_bg.sum()) < max(32, int(round(float(bg_close.size) * 0.01))):
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient high-confidence background evidence",
            "high_conf_bg_pixels": int(high_conf_bg.sum()),
            **thresholds,
        }

    residual = image_srgb.astype(np.float32) - bg.astype(np.float32).reshape(1, 1, 3)
    border = _border_mask(distance.shape)
    drift_probe = high_conf_bg & border
    if int(drift_probe.sum()) < max(32, int(round(float(bg_close.size) * 0.002))):
        drift_probe = high_conf_bg
    bg_residual = residual[drift_probe]
    residual_abs = np.abs(bg_residual)
    residual_p95 = float(np.percentile(residual_abs, 95.0))
    residual_std = float(np.std(bg_residual.astype(np.float32), axis=0).mean())
    # These are 8-bit drift gates, not shadow thresholds. They are low enough
    # to catch generator background mottling but leave true flat known screens
    # like B010 as an exact no-op.
    if residual_p95 < 2.0 and residual_std < 0.75:
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "background already uniform",
            "high_conf_bg_pixels": int(high_conf_bg.sum()),
            "drift_probe_pixels": int(drift_probe.sum()),
            "residual_abs_p95_u8": residual_p95,
            "residual_std_u8": residual_std,
            **thresholds,
        }

    h, w = distance.shape
    image = image_srgb.astype(np.float32)
    bgf = bg.astype(np.float32).reshape(3)
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
        tail_max_strength = 0.12
        strength_gate = np.clip((tail_max_strength - np.maximum(strength, 0.0)) / tail_max_strength, 0.0, 1.0)
        strength_gate = strength_gate * strength_gate * (3.0 - 2.0 * strength_gate)
        fg_span = max(effective_fg_threshold - effective_bg_threshold, 1e-6)
        bg_color_gate = np.clip(1.0 - (distance - effective_bg_threshold) / fg_span, 0.0, 1.0)
        tail_weight = np.where(screen_like_tail, strength_gate * bg_color_gate, 0.0).astype(np.float32)

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
    raw_weight[high_conf_bg] = 1.0
    raw_weight = np.maximum(raw_weight, tail_weight)
    if float(raw_weight.max()) <= 0.0:
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "empty normalization support",
            "high_conf_bg_pixels": int(high_conf_bg.sum()),
            "drift_probe_pixels": int(drift_probe.sum()),
            "residual_abs_p95_u8": residual_p95,
            "residual_std_u8": residual_std,
            **thresholds,
        }

    sigma = float(max(2.0, min(8.0, round(float(min(h, w)) * 0.025))))
    ksize = int(round(sigma * 6.0)) | 1
    weight = cv2.GaussianBlur(raw_weight, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    weight[high_conf_bg] = 1.0
    # Do not let smoothing spill into obvious material. Any residual line here
    # is less harmful than pre-normalizing real subject color.
    obvious_material = distance >= max(effective_fg_threshold, effective_bg_threshold + 8.0)
    weight[obvious_material & ~screen_like_tail] = 0.0

    normalized = image * (1.0 - weight[..., None]) + bgf.reshape(1, 1, 3) * weight[..., None]
    normalized_u8 = np.clip(normalized + 0.5, 0, 255).astype(np.uint8)
    changed = np.abs(normalized_u8.astype(np.int16) - image_srgb.astype(np.int16)).mean(axis=2) > 0
    return normalized_u8, {
        "enabled": True,
        "applied": True,
        "reason": "background drift normalized",
        "background_color": [int(c) for c in bg],
        "high_conf_bg_pixels": int(high_conf_bg.sum()),
        "drift_probe_pixels": int(drift_probe.sum()),
        "enclosed_bg_pixels": int(enclosed_bg.sum()),
        "enclosed_bg_component_min_area": int(enclosed_info.get("enclosed_bg_component_min_area", 0)),
        "residual_abs_p95_u8": residual_p95,
        "residual_std_u8": residual_std,
        "screen_like_tail_pixels": int(screen_like_tail.sum()),
        "tail_weight_pixels": int((tail_weight > 1.0 / 255.0).sum()),
        "weight_nonzero_pixels": int((weight > 1.0 / 255.0).sum()),
        "weight_mean": float(weight.mean()),
        "weight_p95": float(np.percentile(weight, 95.0)),
        "changed_pixels": int(changed.sum()),
        "sigma_px": sigma,
        **thresholds,
    }


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
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0]
    d = oklab_distance(lab, bg_lab)
    thresholds = (
        _adaptive_known_background_thresholds(d, image_srgb, bg, bg_threshold, fg_threshold)
        if adaptive
        else _fixed_known_background_thresholds(bg_threshold, fg_threshold)
    )
    effective_bg_threshold = thresholds["bg_threshold_effective"]

    rgb_distance = np.sqrt(
        np.sum((image_srgb.astype(np.float32) - bg.astype(np.float32).reshape(1, 1, 3)) ** 2, axis=2)
    )
    bg_close = d <= effective_bg_threshold
    exterior_bg = _flood_from_border(bg_close)
    enclosed_bg_raw = bg_close & ~exterior_bg
    enclosed_bg, enclosed_info = _filter_enclosed_background_components(enclosed_bg_raw)
    not_exterior = ~exterior_bg
    dist_to_exterior = cv2.distanceTransform(not_exterior.astype(np.uint8), cv2.DIST_L2, 3)
    if adaptive:
        thresholds = {
            **thresholds,
            **_adaptive_foreground_seed_threshold(
                d,
                not_exterior,
                dist_to_exterior,
                bg_threshold=float(effective_bg_threshold),
                base_fg_threshold=float(thresholds["fg_threshold_effective"]),
                base_fg_source=str(thresholds["fg_threshold_source"]),
                requested_fg_threshold=float(fg_threshold),
                background_noise_mad=float(thresholds["background_noise_mad"]),
                boundary_band_px=int(boundary_band_px),
            ),
        }
    effective_fg_threshold = thresholds["fg_threshold_effective"]
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
    effective_boundary_band_px = boundary_info["boundary_band_px_effective"]
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
        sure_fg = subject_support
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
    # Interior pixels that still match the known background are transparent
    # cutouts, not antialiasing unknowns. Leaving them unknown lets local
    # smoothness solvers propagate surrounding foreground through UI holes.
    if bool(enclosed_bg.any()):
        clean_bg = bg_close
        clean_bg_threshold: float | str = "perceptual_bg_close"
        clean_bg_policy = "hole_aware_preserve_enclosed_background"
    else:
        clean_bg_threshold_u8 = max(2.5, min(4.0, float(bg_threshold)))
        clean_bg = rgb_distance <= clean_bg_threshold_u8
        clean_bg_threshold = float(clean_bg_threshold_u8)
        clean_bg_policy = "solid_subject_strict_known_background"
    dist_to_non_clean = cv2.distanceTransform(clean_bg.astype(np.uint8), cv2.DIST_L2, 3)
    # Sure background must be boring, unchanged screen. Pixels close to the
    # subject or to a faint shadow falloff stay unknown so the later
    # same-background reconstruction can decide their opacity.
    sure_bg = (exterior_bg & clean_bg & (dist_to_non_clean >= 2.0)) | enclosed_bg
    shadow_bg, shadow_info = _known_background_shadow_like_background_mask(
        image_srgb,
        bg,
        subject_seed=sure_fg,
    )
    shadow_unknown = shadow_bg & (~sure_fg | screen_dominant_shadow)
    # ShadowPatch is the ownership stage for scalar-darkened known background.
    # The trimap should therefore not pin contact/drop shadows to foreground or
    # background. This carve-out is especially important for the very faint
    # outer falloff: those pixels can be close enough to the screen color to
    # pass the sure-BG threshold, but they are still part of the transferable
    # shadow and must remain in the repair domain. Screen-neutral dark
    # material/grooves that already have a strong interior foreground seed stay
    # foreground-protected.
    sure_bg &= ~shadow_unknown
    sure_fg &= ~(enclosed_bg | shadow_unknown)
    unknown = ~(sure_fg | sure_bg)
    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(enclosed_bg.astype(np.uint8), 8)
    enclosed_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, labels_count)]

    # If a source has no clear foreground core, PyMatting has nothing stable to
    # propagate from. Keep the trimap valid but report the weak support.
    trimap = Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)
    return trimap, {
        "method": "known_background_exterior_band",
        "adaptive": bool(adaptive),
        "background_color": [int(c) for c in bg],
        "bg_threshold": float(bg_threshold),
        "fg_threshold": float(fg_threshold),
        **thresholds,
        "boundary_band_px": int(boundary_band_px),
        **boundary_info,
        "foreground_seed_inset_px": int(foreground_seed_inset_px),
        "foreground_seed_inset": foreground_seed_inset_info,
        "subject_material_support": subject_support_info,
        "sure_fg_pixels": int(sure_fg.sum()),
        "sure_bg_pixels": int(sure_bg.sum()),
        "unknown_pixels": int(unknown.sum()),
        "exterior_bg_pixels": int(exterior_bg.sum()),
        "enclosed_bg_pixels": int(enclosed_bg.sum()),
        "clean_bg_policy": clean_bg_policy,
        "clean_bg_threshold": clean_bg_threshold,
        "clean_exterior_bg_pixels": int((exterior_bg & clean_bg).sum()),
        "sure_bg_clean_inset_px": 2.0,
        "enclosed_bg_pixels_raw": int(enclosed_bg_raw.sum()),
        **enclosed_info,
        "enclosed_bg_components": int(labels_count - 1),
        "largest_enclosed_bg_component": int(max(enclosed_areas, default=0)),
        "shadow_background": {
            **shadow_info,
            "unknown_ownership_pixels": int(shadow_unknown.sum()),
            "hard_ownership_pixels": 0,
            "screen_dominant_overlap_pixels": int((shadow_bg & sure_fg & screen_dominant_shadow).sum()),
            "protected_foreground_overlap_pixels": int((shadow_bg & sure_fg & ~screen_dominant_shadow).sum()),
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
    """Estimate the known flat background color from corners or border modes."""
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

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
