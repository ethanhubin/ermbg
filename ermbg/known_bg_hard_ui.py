"""Analytic known-background solver for hard UI plus cast shadow.

This path is intentionally narrow. It handles high-confidence game/UI assets
on a known flat screen where the subject is hard-edged and the shadow is a
physical scalar darkening of the known background:

    C_linear ~= scale * B_linear

Complex hair, glass, smoke, and mixed translucency should stay on the neural
CorridorKey/ERMBG paths. The goal here is to avoid asking a general matting
model to hallucinate a shadow alpha when the known background gives us a direct
measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from . import io
from .keyer import KeyerThresholds, chromatic_key_alpha
from .shadow import shadow_alpha_to_display_alpha


@dataclass(frozen=True)
class KnownBgHardUiShadowResult:
    accepted: bool
    reason: str
    background_color: tuple[int, int, int]
    alpha: np.ndarray
    subject_alpha: np.ndarray
    foreground_srgb: np.ndarray
    rgba_rgb_srgb: np.ndarray
    shadow_alpha: np.ndarray
    shadow_alpha_physical: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("alpha", None)
        payload.pop("subject_alpha", None)
        payload.pop("foreground_srgb", None)
        payload.pop("rgba_rgb_srgb", None)
        payload.pop("shadow_alpha", None)
        payload.pop("shadow_alpha_physical", None)
        return payload


def solve_known_bg_hard_ui_shadow(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
) -> KnownBgHardUiShadowResult:
    """Solve a hard UI subject and scalar known-background shadow."""
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("solve_known_bg_hard_ui_shadow() expects HxWx3 uint8 sRGB")

    h, w = image_srgb.shape[:2]
    bg = tuple(int(c) for c in background_color)
    empty = np.zeros((h, w), dtype=np.float32)
    empty_rgb = np.zeros_like(image_srgb)
    key_alpha = chromatic_key_alpha(image_srgb, bg, KeyerThresholds(bg_max=3.5, fg_min=18.0))
    physical_shadow, shadow_model = _scalar_darkening_alpha(image_srgb, bg)

    # Strong black/dark outline can be a near-zero scale of the background
    # equation. It is subject material, not a reusable cast shadow. The upper
    # strength bound keeps those hard outlines in the subject seed.
    scalar_shadow = (
        (physical_shadow >= 0.045)
        & (physical_shadow <= 0.86)
        & (shadow_model["error"] <= 0.055)
    )
    subject_seed = (key_alpha >= 0.92) & ~scalar_shadow
    subject_seed |= (key_alpha >= 0.35) & (physical_shadow > 0.86)
    subject_seed = _keep_large_components(subject_seed, min_area=max(24, int(round(h * w * 0.002))))
    if not subject_seed.any():
        return KnownBgHardUiShadowResult(False, "no hard subject seed", bg, empty, empty, empty_rgb, empty_rgb, empty, empty)

    # Dark antialiasing on a black outline can satisfy the same scalar
    # darkening equation as a cast shadow. If that evidence is a narrow band
    # glued to the hard subject seed, it is subject edge coverage; detached
    # scalar components remain available to the shadow layer.
    subject_edge_band = cv2.dilate(subject_seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=2) > 0
    subject_contour_scalar = scalar_shadow & subject_edge_band & (key_alpha > 0.018)
    subject_possible = (key_alpha > 0.018) & (~scalar_shadow | subject_contour_scalar)
    subject_domain = _keep_components_touching_seed(subject_possible, subject_seed)
    opaque_color_seed = cv2.erode(subject_seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    if not opaque_color_seed.any():
        opaque_color_seed = subject_seed
    subject_alpha = _subject_alpha_from_nearest_opaque(image_srgb, bg, subject_domain, opaque_color_seed, key_alpha)
    subject_alpha = np.clip(subject_alpha, 0.0, 1.0).astype(np.float32)

    subject_core = subject_alpha >= 0.65
    if not subject_core.any():
        return KnownBgHardUiShadowResult(False, "no subject core", bg, empty, empty, empty_rgb, empty_rgb, empty, empty)

    dist_to_subject = cv2.distanceTransform((~subject_core).astype(np.uint8), cv2.DIST_L2, 3)
    max_shadow_distance = min(float(max(42, min(h, w) * 0.45)), 96.0)
    shadow_domain = (
        scalar_shadow
        & ~subject_contour_scalar
        & (dist_to_subject <= max_shadow_distance)
        & (subject_alpha <= 0.25)
    )
    shadow_domain = _keep_shadow_like_components(
        shadow_domain,
        min_area=max(16, int(round(h * w * 0.0015))),
        image_shape=(h, w),
    )
    if not shadow_domain.any():
        return KnownBgHardUiShadowResult(
            False,
            "no exterior scalar shadow component",
            bg,
            empty,
            subject_alpha,
            _foreground_from_known_bg(image_srgb, bg, subject_alpha),
            _foreground_from_known_bg(image_srgb, bg, subject_alpha),
            empty,
            empty,
            debug={"subject_pixels": int((subject_alpha > 0.0).sum())},
        )

    shadow_physical = np.where(shadow_domain, physical_shadow, 0.0).astype(np.float32)
    shadow_display = shadow_alpha_to_display_alpha(shadow_physical, bg)
    shadow_display, level_info = _level_hard_shadow_components(shadow_display, shadow_domain)

    foreground_srgb = _foreground_from_known_bg(image_srgb, bg, subject_alpha)
    rgba_rgb_srgb = _compose_subject_and_shadow_rgb(foreground_srgb, subject_alpha, shadow_display)
    alpha = np.maximum(subject_alpha, shadow_display).astype(np.float32)

    subject_fraction = float((subject_alpha > 0.02).sum() / max(1, h * w))
    shadow_fraction = float((shadow_display > 0.0).sum() / max(1, h * w))
    accepted = bool(subject_fraction >= 0.03 and shadow_fraction >= 0.0015)
    reason = "accepted" if accepted else "insufficient subject/shadow support"

    debug = {
        "method": "known_bg_hard_ui_shadow",
        "subject_fraction": subject_fraction,
        "shadow_fraction": shadow_fraction,
        "subject_pixels": int((subject_alpha > 0.02).sum()),
        "shadow_pixels": int((shadow_display > 0.0).sum()),
        "key_alpha_mean": float(key_alpha.mean()),
        "scalar_shadow_pixels": int(scalar_shadow.sum()),
        "subject_contour_scalar_pixels": int(subject_contour_scalar.sum()),
        "shadow_model": {
            "strength_mean": float(physical_shadow[shadow_domain].mean()) if shadow_domain.any() else 0.0,
            "strength_p75": float(np.percentile(physical_shadow[shadow_domain], 75.0)) if shadow_domain.any() else 0.0,
            "error_mean": float(shadow_model["error"][shadow_domain].mean()) if shadow_domain.any() else 0.0,
            "max_shadow_distance": max_shadow_distance,
        },
        "hard_shadow_leveling": level_info,
    }
    return KnownBgHardUiShadowResult(
        accepted=accepted,
        reason=reason,
        background_color=bg,
        alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
        subject_alpha=subject_alpha,
        foreground_srgb=foreground_srgb,
        rgba_rgb_srgb=rgba_rgb_srgb,
        shadow_alpha=np.clip(shadow_display, 0.0, 1.0).astype(np.float32),
        shadow_alpha_physical=np.clip(shadow_physical, 0.0, 1.0).astype(np.float32),
        debug=debug,
    )


def _scalar_darkening_alpha(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    denom = max(float(np.dot(B, B)), 1e-6)
    scale = np.clip(np.sum(C * B.reshape(1, 1, 3), axis=-1) / denom, 0.0, 2.0)
    recon = scale[..., None] * B.reshape(1, 1, 3)
    error = np.sqrt(np.mean((C - recon) ** 2, axis=-1)).astype(np.float32)
    strength = np.clip(1.0 - scale, 0.0, 1.0).astype(np.float32)
    return strength, {"scale": scale.astype(np.float32), "error": error}


def _keep_large_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out |= labels == label
    return out


def _keep_components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    seed_labels = np.unique(labels[seed])
    for label in seed_labels:
        if int(label) != 0:
            out |= labels == int(label)
    return out


def _subject_alpha_from_nearest_opaque(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    subject_domain: np.ndarray,
    opaque_seed: np.ndarray,
    key_alpha: np.ndarray,
) -> np.ndarray:
    """Estimate hard-subject AA alpha from nearby opaque foreground color.

    Hard UI edges are not semantically translucent, but their source pixels are
    antialias blends of foreground color over the known screen. A plain
    distance-to-background key often snaps those pixels to 1 and makes jagged
    corners. Borrow a nearby opaque subject color and solve
    ``C = a F + (1-a) B`` for the boundary alpha instead.
    """
    alpha = np.zeros(subject_domain.shape, dtype=np.float32)
    if not opaque_seed.any():
        alpha[subject_domain] = key_alpha[subject_domain]
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    _, indices = ndimage.distance_transform_edt(~opaque_seed, return_indices=True)
    nearest_rgb = image_srgb[indices[0], indices[1]]
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    F = io.srgb_to_linear(nearest_rgb.astype(np.uint8)).astype(np.float32)
    B = io.srgb_to_linear(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    direction = F - B.reshape(1, 1, 3)
    denom = np.sum(direction * direction, axis=-1)
    solved = np.sum((C - B.reshape(1, 1, 3)) * direction, axis=-1) / np.maximum(denom, 1e-6)
    solved = np.clip(solved, 0.0, 1.0).astype(np.float32)

    weak_foreground_color = denom <= 1e-5
    subject_values = np.where(weak_foreground_color, key_alpha, solved)
    # High-confidence opaque seeds remain opaque; surrounding AA keeps the
    # source edge smooth. Low alpha tails are clipped out so background specks
    # connected through one-pixel noise do not become subject.
    alpha[subject_domain] = subject_values[subject_domain]
    alpha[opaque_seed] = 1.0
    alpha[(alpha < 0.02) & ~opaque_seed] = 0.0
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _keep_shadow_like_components(mask: np.ndarray, *, min_area: int, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        if left <= 1 or top <= 1 or left + width >= w - 1 or top + height >= h - 1:
            continue
        if width < max(6, int(round(height * 1.15))):
            continue
        out |= labels == label
    return out


def _level_hard_shadow_components(shadow_display: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(domain.astype(np.uint8), connectivity=8)
    out = shadow_display.astype(np.float32).copy()
    components: list[dict[str, Any]] = []
    for label in range(1, count):
        comp = labels == label
        values = shadow_display[comp]
        if values.size == 0:
            continue
        p75 = float(np.percentile(values, 75.0))
        floor = max(0.045, p75 * 0.35)
        core = comp & (shadow_display >= floor)
        out[core] = np.maximum(out[core], p75)
        components.append(
            {
                "area": int(stats[label, cv2.CC_STAT_AREA]),
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "display_p75": p75,
                "level_floor": float(floor),
                "core_pixels": int(core.sum()),
            }
        )
    return np.clip(out, 0.0, 1.0).astype(np.float32), {"components": components}


def _foreground_from_known_bg(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    alpha: np.ndarray,
) -> np.ndarray:
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    F = C.copy()
    solve = a > 1e-4
    F[solve] = (C[solve] - (1.0 - a[solve, None]) * B.reshape(1, 3)) / np.maximum(a[solve, None], 1e-4)
    F = np.clip(F, 0.0, 1.0)
    F[~solve] = 0.0
    return io.linear_to_srgb_u8(F)


def _compose_subject_and_shadow_rgb(
    foreground_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    shadow_alpha: np.ndarray,
) -> np.ndarray:
    fg_linear = io.srgb_to_linear(foreground_srgb).astype(np.float32)
    a_subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    a_shadow = np.clip(shadow_alpha.astype(np.float32), 0.0, 1.0) * (1.0 - a_subject)
    a_out = np.clip(a_subject + a_shadow, 0.0, 1.0)
    premul = fg_linear * a_subject[..., None]
    out = np.zeros_like(fg_linear)
    nonzero = a_out > 1e-6
    out[nonzero] = premul[nonzero] / a_out[nonzero, None]
    return io.linear_to_srgb_u8(np.clip(out, 0.0, 1.0))
