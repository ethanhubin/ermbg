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
    sure_fg = strong_fg & (dist_to_exterior > float(effective_boundary_band_px))
    # Interior pixels that still match the known background are transparent
    # cutouts, not antialiasing unknowns. Leaving them unknown lets local
    # smoothness solvers propagate surrounding foreground through UI holes.
    sure_bg = exterior_bg | enclosed_bg
    shadow_bg, shadow_info = _known_background_shadow_like_background_mask(
        image_srgb,
        bg,
        subject_seed=sure_fg,
    )
    screen_dominant_shadow = _screen_dominant_shadow_pixels(image_srgb, bg)
    hard_shadow_bg = shadow_bg & (~sure_fg | screen_dominant_shadow)
    # ShadowPatch is a reconstruction patch, not a trimap ownership oracle.
    # The grown shadow-like support is useful debug evidence for the later
    # source-reprojection patch, but it must not freely overturn strong
    # foreground seeds: connected metal grooves can look like scalar-darkened
    # screen when attached to a real cast shadow. Only weak/non-seed support, or
    # strong seeds that still have screen-dominant shadow color, are pinned to
    # background. Near-black/cross-channel subject grooves stay with PyMatting.
    sure_bg |= hard_shadow_bg
    sure_fg &= ~(enclosed_bg | hard_shadow_bg)
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
        "sure_fg_pixels": int(sure_fg.sum()),
        "sure_bg_pixels": int(sure_bg.sum()),
        "unknown_pixels": int(unknown.sum()),
        "exterior_bg_pixels": int(exterior_bg.sum()),
        "enclosed_bg_pixels": int(enclosed_bg.sum()),
        "enclosed_bg_pixels_raw": int(enclosed_bg_raw.sum()),
        **enclosed_info,
        "enclosed_bg_components": int(labels_count - 1),
        "largest_enclosed_bg_component": int(max(enclosed_areas, default=0)),
        "shadow_background": {
            **shadow_info,
            "hard_ownership_pixels": int(hard_shadow_bg.sum()),
            "screen_dominant_overlap_pixels": int((shadow_bg & sure_fg & screen_dominant_shadow).sum()),
            "protected_foreground_overlap_pixels": int((shadow_bg & sure_fg & ~screen_dominant_shadow).sum()),
        },
    }


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
]
