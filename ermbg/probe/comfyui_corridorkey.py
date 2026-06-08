"""CorridorKey hint builders and key-color protection shared by the executors.

This module no longer hosts a remote ComfyUI client; the live path runs through
the in-process CorridorKey runner. Only the deterministic hint/protection
helpers remain and are consumed by the direct worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ermbg.colorspace import oklab_distance, srgb_to_oklab
from ermbg.keyer import KeyerThresholds, chromatic_key_alpha
from ermbg.shadow import ShadowThresholds, exterior_scalar_darkening_mask


@dataclass(frozen=True)
class ComfyCorridorKeyResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    hint_alpha: np.ndarray
    raw_alpha: np.ndarray
    color_protection_alpha: np.ndarray
    debug: dict[str, Any]


def build_corridorkey_hint(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Build CorridorKey's coarse alpha hint from known green-screen evidence.

    CorridorKey is not a generic segmenter; it expects a rough foreground
    ownership hint. For high-confidence AI green-screen assets, direct known-B
    chroma distance gives that hint without running a second neural model.
    The support is slightly eroded and blurred because the model's own docs
    prefer a soft, under-expanded hint over an exact or over-grown mask.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_corridorkey_hint() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    )
    h, w = raw.shape
    if not np.any(raw > 0.18):
        return raw.astype(np.float32)

    # Empirical, signal-based values: the threshold accepts pixels with clear
    # non-background chroma evidence; the one-pixel erosion protects against
    # known-B fringe leakage becoming model-owned foreground in the hint.
    support = (raw >= 0.18).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    if min(h, w) >= 24:
        support = cv2.erode(support, kernel, iterations=1)
    if not support.any():
        support = (raw >= 0.35).astype(np.uint8)
    if not support.any():
        return raw.astype(np.float32)

    blur_ksize = 7 if min(h, w) >= 96 else 5
    hint = cv2.GaussianBlur(support.astype(np.float32), (blur_ksize, blur_ksize), 0)
    return np.clip(hint, 0.0, 1.0).astype(np.float32)


def build_hard_ui_corridorkey_hint(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Build a CorridorKey bbox hint for opaque hard UI.

    Failure mode this tests against: pixel-exact known-B hints can import mask
    defects into CorridorKey, while full-frame priors are too coarse to support
    crisp UI outlines. For hard UI, use known-B evidence only to locate the
    subject's rectangle, then give CorridorKey a simple bbox hint. The 2 px
    expansion is deliberate for hard UI antialias/outline pixels: it avoids
    clipping the visible border without turning the whole screen into foreground.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_hard_ui_corridorkey_hint() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    ).astype(np.float32)
    h, w = raw.shape
    if not np.any(raw > 0.18):
        return raw.astype(np.float32)

    # The support threshold is only for locating a stable hard-UI component, not
    # for transferring edge alpha. A bbox hint intentionally gives CorridorKey
    # room to solve antialiasing, outline alpha, and possible nearby shadow.
    support = raw >= 0.18
    yy, xx = np.nonzero(support)
    if yy.size == 0:
        return raw.astype(np.float32)

    pad_px = 2
    y0 = max(0, int(yy.min()) - pad_px)
    y1 = min(h, int(yy.max()) + pad_px + 1)
    x0 = max(0, int(xx.min()) - pad_px)
    x1 = min(w, int(xx.max()) + pad_px + 1)
    hint = np.zeros((h, w), dtype=np.float32)
    hint[y0:y1, x0:x1] = 1.0
    return hint


def build_hard_ui_boundary_corridorkey_hint(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Build a boundary-only CorridorKey hint for opaque hard UI experiments.

    The current hard-UI path remains bbox-based. This separate experiment sends
    only a narrow boundary band to CorridorKey, so solid interiors can be owned
    by local known-background evidence while the model spends its capacity on
    antialiasing, outlines, and nearby shadow.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_hard_ui_boundary_corridorkey_hint() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    ).astype(np.float32)
    if not np.any(raw > 0.18):
        return raw.astype(np.float32)

    # The support threshold locates the hard UI silhouette; the ring sends only
    # the uncertain edge to CorridorKey. The two-pixel band is empirical and
    # keyed to crisp UI AA/outline widths observed in the button samples.
    support = raw >= 0.18
    if not support.any():
        return raw.astype(np.float32)

    band_px = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    outer = cv2.dilate(support.astype(np.uint8), kernel, iterations=1).astype(bool)
    inner = cv2.erode(support.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = outer & ~inner
    return ring.astype(np.float32)


def build_hard_ui_solid_interior_mask(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Return high-confidence hard-UI interior that should bypass CorridorKey.

    Broad rule: for opaque hard UI, strong known-background evidence well away
    from the edge is already a solid subject decision. CorridorKey is still used
    for the boundary band, but this interior is restored to alpha 1 and source
    RGB so learned keyer uncertainty cannot add waves inside a button.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_hard_ui_solid_interior_mask() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    ).astype(np.float32)
    support = raw >= 0.70
    if not support.any():
        return np.zeros(raw.shape, dtype=bool)

    # Three pixels keeps AA/outline/shadow ownership with CorridorKey while
    # leaving broad, high-confidence hard UI material to the local solid restore.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    interior = cv2.erode(support.astype(np.uint8), kernel, iterations=1).astype(bool)
    return interior


def build_hard_ui_shadow_safe_solid_interior_mask(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return hard-UI interior while leaving measured shadow to shadow repair.

    Failure mode this protects against: a boundary-only CorridorKey hint needs
    local interior restoration, but screen-colored contact shadows can satisfy
    the same high-alpha chroma evidence as opaque UI material. Excluding
    scalar-darkening connected to exterior background keeps those pixels out of
    subject ownership so the later known-background shadow patch can own them.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_hard_ui_shadow_safe_solid_interior_mask() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    ).astype(np.float32)
    support = raw >= 0.70
    if not support.any():
        empty = np.zeros(raw.shape, dtype=bool)
        return empty, {
            "shadow_safe_enabled": True,
            "base_interior_pixels": 0,
            "shadow_like_pixels": 0,
            "shadow_candidate_pixels": 0,
            "shadow_exclusion_pixels": 0,
            "shadow_excluded_interior_pixels": 0,
            "solid_interior_pixels": 0,
        }

    # Match the existing interior path first so this mode changes only the
    # ownership split near measured shadow, not the broad hard-UI material rule.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    base_interior = cv2.erode(support.astype(np.uint8), kernel, iterations=1).astype(bool)

    known_bg = raw <= 0.20
    shadow_like, shadow_info = exterior_scalar_darkening_mask(
        image_srgb,
        background_color,
        known_bg,
        ShadowThresholds(
            # Interior restore runs before the dedicated shadow patch. These
            # gates intentionally mirror the color-protection shadow guard:
            # scalar reconstruction and exterior connectivity are the real
            # ownership signals, while the low strength catches light contact
            # shadow AA that should not become opaque subject material.
            min_strength=0.01,
            max_reconstruction_error=0.07,
            reject_border_components=False,
        ),
    )
    if shadow_like.any():
        # A small cushion keeps contact-shadow antialiasing out of the forced
        # alpha=1 restore. It is empirical but signal-bound: it only grows from
        # measured scalar-darkening support, not from arbitrary geometry.
        shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_exclusion = cv2.dilate(shadow_like.astype(np.uint8), shadow_kernel, iterations=1).astype(bool)
    else:
        shadow_exclusion = shadow_like

    interior = base_interior & ~shadow_exclusion
    excluded = base_interior & shadow_exclusion
    info = {
        "shadow_safe_enabled": True,
        "base_interior_pixels": int(base_interior.sum()),
        "shadow_like_pixels": int(shadow_like.sum()),
        "shadow_candidate_pixels": int(shadow_info.get("candidate_pixels", 0)),
        "shadow_exclusion_pixels": int(shadow_exclusion.sum()),
        "shadow_excluded_interior_pixels": int(excluded.sum()),
        "solid_interior_pixels": int(interior.sum()),
    }
    return interior.astype(bool), info


def build_hard_ui_shadow_safe_material_alpha_floor(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a local material alpha floor for unoutlined hard UI.

    Failure mode this protects against: boundary-only CorridorKey can leave a
    blank band on opaque UI that has no outline, because the model has no dark
    edge anchor and the local solid restore intentionally erodes the interior.
    Known-background chroma alpha is reliable for that hard material edge, but
    scalar screen-darkening must still be excluded so shadows remain owned by
    the dedicated shadow patch.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_hard_ui_shadow_safe_material_alpha_floor() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    ).astype(np.float32)
    material_support = raw >= 0.18
    if not material_support.any():
        empty = np.zeros(raw.shape, dtype=np.float32)
        return empty, {
            "material_floor_enabled": True,
            "base_interior_pixels": 0,
            "material_floor_pixels": 0,
            "shadow_like_pixels": 0,
            "shadow_candidate_pixels": 0,
            "shadow_exclusion_pixels": 0,
            "shadow_excluded_floor_pixels": 0,
            "solid_interior_pixels": 0,
        }

    solid_support = raw >= 0.70
    if solid_support.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        base_interior = cv2.erode(solid_support.astype(np.uint8), kernel, iterations=1).astype(bool)
    else:
        base_interior = np.zeros(raw.shape, dtype=bool)

    known_bg = raw <= 0.20
    shadow_like, shadow_info = exterior_scalar_darkening_mask(
        image_srgb,
        background_color,
        known_bg,
        ShadowThresholds(
            # Keep this matched to the shadow-safe solid interior path: the
            # only difference is that surviving material edge pixels receive a
            # soft alpha floor instead of being left entirely to CorridorKey.
            min_strength=0.01,
            max_reconstruction_error=0.07,
            reject_border_components=False,
        ),
    )
    if shadow_like.any():
        shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_exclusion = cv2.dilate(shadow_like.astype(np.uint8), shadow_kernel, iterations=1).astype(bool)
    else:
        shadow_exclusion = shadow_like

    allowed_material = material_support & ~shadow_exclusion
    floor = np.where(allowed_material, raw, 0.0).astype(np.float32)
    solid_interior = base_interior & ~shadow_exclusion
    floor[solid_interior] = 1.0
    excluded = material_support & shadow_exclusion
    info = {
        "material_floor_enabled": True,
        "base_interior_pixels": int(base_interior.sum()),
        "material_floor_pixels": int((floor > 0.0).sum()),
        "shadow_like_pixels": int(shadow_like.sum()),
        "shadow_candidate_pixels": int(shadow_info.get("candidate_pixels", 0)),
        "shadow_exclusion_pixels": int(shadow_exclusion.sum()),
        "shadow_excluded_floor_pixels": int(excluded.sum()),
        "solid_interior_pixels": int(solid_interior.sum()),
        "material_floor_mean": float(floor[floor > 0.0].mean()) if (floor > 0.0).any() else 0.0,
    }
    return np.clip(floor, 0.0, 1.0).astype(np.float32), info


def build_key_color_protection_floor(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Build a soft alpha floor from key-color distance, not region ownership.

    Failure mode this protects against: CorridorKey can treat saturated UI
    colors such as yellow/orange as spill-like transparency even when the hint
    says foreground. The rule is color based: pixels far outside the key
    color family should not be driven transparent by a learned green-screen
    prior. For saturated key colors we measure OKLab a/b distance only, so
    darker same-hue screen shadows remain key-colored instead of becoming an
    opaque protected component. Neutral screens fall back to full OKLab distance
    because their useful signal is mostly lightness.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_key_color_protection_floor() expects HxWx3 sRGB uint8")

    t = thresholds or KeyerThresholds(bg_max=8.0, fg_min=16.0)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    bg_chroma = float(np.linalg.norm(bg_lab[1:]))
    if bg_chroma >= 0.04:
        # Empirical signal split: saturated screen shadows often differ mainly
        # in L while keeping the same a/b family. Protecting by chroma distance
        # preserves yellow/red/white UI material without turning dark green
        # screen shading into opaque subject.
        delta = lab[..., 1:] - bg_lab[1:]
        d = np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32) * 100.0
        mode = "oklab_ab"
    else:
        d = oklab_distance(lab, bg_lab).astype(np.float32)
        mode = "oklab_full"
    x = np.clip((d - t.bg_max) / max(t.fg_min - t.bg_max, 1e-6), 0.0, 1.0)
    floor = x * x * (3.0 - 2.0 * x)
    return np.clip(floor, 0.0, 1.0).astype(np.float32)


def _shadow_safe_color_protection_floor(
    *,
    image_srgb: np.ndarray,
    raw_alpha: np.ndarray,
    background_color: tuple[int, int, int],
    floor: np.ndarray,
    trusted_material_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Suppress color protection where pixels are measured screen darkening.

    Broad rule: color distance alone is not ownership. A cast shadow on a
    saturated known background can move far enough from the key color to look
    like protected subject, while the pixel still satisfies the physical model
    ``C ~= scale * B``. The empirical gates below are intentionally loose and
    evidence-based: they only disable protection for scalar darkening connected
    to exterior background, and only where CorridorKey has not already assigned
    strong subject ownership.
    """
    known_bg = (floor <= 0.05) & (raw_alpha <= 0.20)
    shadow_like, shadow_info = exterior_scalar_darkening_mask(
        image_srgb,
        background_color,
        known_bg,
        ShadowThresholds(
            # Color protection runs before the dedicated shadow pass, so this
            # guard must also catch very light hard-shadow antialiasing. The
            # scalar reconstruction error and exterior flood are the stronger
            # counter-signals against suppressing real subject material.
            min_strength=0.01,
            max_reconstruction_error=0.07,
            reject_border_components=False,
        ),
    )
    subject_owned = raw_alpha >= 0.80
    trusted_material = np.zeros_like(floor, dtype=bool)
    if trusted_material_alpha is not None:
        trusted = np.clip(trusted_material_alpha.astype(np.float32), 0.0, 1.0)
        if trusted.shape != floor.shape:
            raise ValueError("trusted_material_alpha must have shape HxW matching floor")
        # General ownership rule for small/icon-like solid material: color
        # distance can say "this is material", while scalar-darkening can still
        # say "this could be a shadow" when the material shares the screen hue.
        # Resolve that conflict by topology rather than by a sample color: only
        # components supported by both the coarse CorridorKey hint and the
        # color floor, with interior pixels separated from known exterior
        # background, are trusted as material. Exterior-connected shadows and
        # outer antialias pixels remain available to the shadow-safe suppressor.
        material_candidate = (trusted > 0.0) & (floor > 0.0)
        if material_candidate.any():
            distance_to_known_bg = cv2.distanceTransform((~known_bg).astype(np.uint8), cv2.DIST_L2, 3)
            interior_candidate = material_candidate & (distance_to_known_bg > 2.0)
            num_labels, labels = cv2.connectedComponents(material_candidate.astype(np.uint8), connectivity=8)
            if num_labels > 1 and interior_candidate.any():
                component_has_interior = np.bincount(
                    labels.ravel(),
                    weights=interior_candidate.ravel().astype(np.float32),
                    minlength=num_labels,
                ) > 0
                trusted_material = component_has_interior[labels] & material_candidate & (distance_to_known_bg > 1.0)
    blocked = shadow_like & (~subject_owned) & (floor > 0.0) & (~trusted_material)
    applied_floor = floor.copy()
    applied_floor[blocked] = 0.0
    exterior_domain = known_bg | shadow_like
    distance_to_exterior = cv2.distanceTransform((~exterior_domain).astype(np.uint8), cv2.DIST_L2, 3)
    edge_antialias = (
        (raw_alpha >= 0.20)
        & (raw_alpha <= 0.88)
        & (floor > raw_alpha + 0.05)
        & (distance_to_exterior <= 2.0)
        & (~trusted_material)
    )
    # Color protection is allowed to fill interior holes, but not to convert
    # CorridorKey's measured outer-edge antialiasing into full opacity. B023's
    # blue-screen hard UI edge exposed this: mixed yellow+screen pixels were
    # lifted to alpha 1.0 and became discrete dark edge dots on recomposite.
    applied_floor[edge_antialias] = 0.0
    stats = {
        "shadow_safe_enabled": True,
        "shadow_like_pixels": int(shadow_like.sum()),
        "shadow_known_background_pixels": int(known_bg.sum()),
        "shadow_candidate_pixels": int(shadow_info.get("candidate_pixels", 0)),
        "floor_shadow_blocked_pixels": int(blocked.sum()),
        "floor_shadow_blocked_mean": float(floor[blocked].mean()) if blocked.any() else 0.0,
        "floor_edge_antialias_blocked_pixels": int(edge_antialias.sum()),
        "trusted_material_pixels": int(trusted_material.sum()),
        "floor_applied_mean": float(applied_floor.mean()),
    }
    return applied_floor.astype(np.float32), stats


def apply_key_color_protection(
    *,
    image_srgb: np.ndarray,
    foreground_srgb: np.ndarray,
    alpha: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    thresholds: KeyerThresholds | None = None,
    trusted_material_alpha: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Lift model alpha where non-key colors prove the pixel is not screen.

    This deliberately avoids geometric protection masks. The floor comes only
    from color distance to the key color, so anti-aliased edge pixels blended
    toward the screen naturally get a lower floor instead of a hard region
    boundary.
    """
    t = thresholds or KeyerThresholds(bg_max=8.0, fg_min=16.0)
    raw_alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    floor = build_key_color_protection_floor(image_srgb, background_color, thresholds=t)
    applied_floor, shadow_safe_stats = _shadow_safe_color_protection_floor(
        image_srgb=image_srgb,
        raw_alpha=raw_alpha,
        background_color=background_color,
        floor=floor,
        trusted_material_alpha=trusted_material_alpha,
    )
    protected_alpha = np.maximum(raw_alpha, applied_floor).astype(np.float32)
    lift = np.clip(protected_alpha - raw_alpha, 0.0, 1.0)
    alpha_lift_recovery = lift / np.maximum(protected_alpha, 1e-6)
    # The surviving floor is material evidence, not only an alpha lower bound.
    # Opaque hard UI can receive a visibly banded CorridorKey foreground even
    # when raw alpha is already nonzero; once shadow and outer-edge gates have
    # accepted a non-key color as protected subject material, recover that RGB
    # from the source image instead of preserving learned transparency waves.
    blend = np.maximum(alpha_lift_recovery, applied_floor)
    protected_fg = (
        foreground_srgb.astype(np.float32) * (1.0 - blend[..., None])
        + image_srgb.astype(np.float32) * blend[..., None]
    )
    stats = {
        "source": "key_color_distance_floor",
        "mode": "auto",
        "thresholds": {
            "bg_max": float(t.bg_max),
            "fg_min": float(t.fg_min),
        },
        "floor_min": float(floor.min()),
        "floor_max": float(floor.max()),
        "floor_mean": float(floor.mean()),
        "floor_applied_min": float(applied_floor.min()),
        "floor_applied_max": float(applied_floor.max()),
        "floor_applied_mean": float(applied_floor.mean()),
        "lifted_pixels_gt_01": int((lift > 0.01).sum()),
        "alpha_mean_before": float(raw_alpha.mean()),
        "alpha_mean_after": float(protected_alpha.mean()),
        **shadow_safe_stats,
    }
    return (
        np.clip(protected_fg + 0.5, 0, 255).astype(np.uint8),
        np.clip(protected_alpha, 0.0, 1.0).astype(np.float32),
        applied_floor,
        stats,
    )


__all__ = [
    "ComfyCorridorKeyResult",
    "apply_key_color_protection",
    "build_key_color_protection_floor",
    "build_corridorkey_hint",
    "build_hard_ui_corridorkey_hint",
    "build_hard_ui_boundary_corridorkey_hint",
    "build_hard_ui_solid_interior_mask",
    "build_hard_ui_shadow_safe_solid_interior_mask",
    "build_hard_ui_shadow_safe_material_alpha_floor",
]
