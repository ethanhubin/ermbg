"""Analytic matting for high-confidence solid-background graphics.

This module implements the analytic prepass for the production ``matte()``
router. The invariant is different from the BiRefNet repair path: first prove
which topology owns each region, then assign alpha/RGB semantics for that role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from . import io
from .colorspace import oklab_distance, srgb_to_oklab


@dataclass
class SolidGraphicResult:
    """Outputs from the isolated solid-background graphic engine."""

    accepted: bool
    reason: str
    confidence: float
    background_color: tuple[int, int, int]
    alpha: np.ndarray
    subject_alpha: np.ndarray
    foreground_linear: np.ndarray
    rgba_rgb_linear: np.ndarray
    ownership_masks: dict[str, np.ndarray]
    debug: dict[str, Any] = field(default_factory=dict)


def analyze_solid_bg_graphic(image_srgb: np.ndarray) -> SolidGraphicResult:
    """Return an ownership-first matte for a solid-background graphic candidate.

    The initial implementation handles the deterministic class documented in
    ``docs/solid-bg-graphic-plan.md``: stable flat background, exterior
    topology, enclosed holes, simple subject-owned soft pixels, and exterior
    scalar darkening. Ambiguous photographic/gradient inputs are rejected so
    callers can fall back to the existing matting path.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("analyze_solid_bg_graphic() expects HxWx3 uint8 sRGB")

    h, w = image_srgb.shape[:2]
    bg, bg_info = _estimate_stable_background(image_srgb)
    empty = _empty_result(image_srgb, bg, accepted=False, reason=bg_info["reason"])
    if not bg_info["accepted"]:
        return empty

    bg_close = _known_background_mask(image_srgb, bg)
    scalar_shadow = _scalar_darkening_mask(image_srgb, bg)
    hole_darkening = _strict_background_family_darkening_mask(image_srgb, bg)
    exterior_bg = _flood_from_border(bg_close)
    exterior_hole_passable = _flood_from_border(bg_close | hole_darkening)
    scalar_exterior = _flood_from_border(bg_close | scalar_shadow) & scalar_shadow
    strong_non_scalar = ~(bg_close | scalar_shadow | exterior_bg)
    # A saturated-screen antialiasing rim can satisfy C ~= scale*B, the same
    # equation as a cast shadow. If that scalar evidence is only a thin contour
    # glued to strong subject material, it is subject-owned softness. Detached
    # scalar fields remain exterior shadow candidates.
    subject_contour_scalar = scalar_exterior & ndimage.binary_dilation(strong_non_scalar, iterations=2)
    shadow = scalar_exterior & ~subject_contour_scalar
    exterior = exterior_bg | shadow

    exterior_fraction = float(exterior.sum()) / float(h * w)
    if exterior_fraction < 0.03:
        out = _empty_result(image_srgb, bg, accepted=False, reason="no dominant exterior background component")
        out.debug.update({"background": bg_info, "exterior_fraction": exterior_fraction})
        return out

    # G02/G03 distinction: same-background pixels inside the subject are not
    # decided by color alone. First prove an enclosed local-background basin by
    # topology; then split it into exact-B transparent hole pixels and strict
    # background-family darkening that belongs to the hole-side shadow layer.
    # This makes inner holes follow the same known-B rule as the outer edge,
    # while connected same-hue subject material remains protected later by the
    # internal-material component guard.
    hole_region = _enclosed_hole_mask((bg_close | hole_darkening) & ~exterior_hole_passable)
    hole = hole_region & bg_close
    internal_hole_shadow = _scalar_darkening_reachable_from_holes(
        bg_close,
        hole_darkening,
        exterior_hole_passable,
        hole_region,
    )
    shadow |= internal_hole_shadow
    exterior = exterior_bg | shadow
    subject_candidate = ~(exterior | hole | shadow)
    subject_candidate = _keep_subject_anchored_components(subject_candidate)
    subject_distance = _background_distance(image_srgb, bg)
    exterior_distance = ndimage.distance_transform_edt(~exterior)
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) >= 40 and bg_info["sigma"] <= 2.5:
        pixels = image_srgb.astype(np.float32)
        dominant_margin = pixels[..., dominant] - np.maximum(
            pixels[..., other_channels[0]],
            pixels[..., other_channels[1]],
        )
        # Experience-driven gate: wide exterior haze/glow on generated assets
        # appears with an extremely stable solid screen. When corner/background
        # variance is higher, this same rule can soften ordinary UI edges, so
        # those cases stay with the narrower antialiasing classifier.
        exterior_translucent_soft = (
            subject_candidate
            & ~bg_close
            & (subject_distance < 26.0)
            & (exterior_distance <= 35.0)
            & (dominant_margin < 40.0)
        )
    else:
        exterior_translucent_soft = np.zeros((h, w), dtype=bool)
    # Strongly separated material is opaque. Connected, weaker non-B material
    # is subject-owned softness; the 18 dE boundary keys on observable color
    # separation so saturated glow stays soft while flat icon fills on neutral
    # screens do not become translucent. Exact-B pixels that survived the hole
    # proof are same-color decoration and stay opaque rather than becoming alpha.
    base_soft_subject = subject_candidate & ~bg_close & (subject_distance < 18.0)
    base_soft_fraction = float(base_soft_subject.sum()) / float(h * w)
    if base_soft_fraction < 0.015:
        exterior_translucent_soft[:] = False
    soft_subject = base_soft_subject | exterior_translucent_soft
    opaque_subject = subject_candidate & ~soft_subject
    unknown = ~(exterior | hole | shadow | opaque_subject | soft_subject)

    subject_alpha = np.zeros((h, w), dtype=np.float32)
    subject_alpha[opaque_subject] = 1.0
    soft_alpha = _estimate_soft_alpha(image_srgb, bg, soft_subject)
    subject_alpha[soft_subject] = soft_alpha[soft_subject]
    # Exterior smoke/glow can be visibly semi-transparent even when its color is
    # farther from B than a hard antialiasing pixel. Cap that background-facing
    # soft band so known-B foreground solve can remove screen color instead of
    # exporting an opaque green/cyan haze.
    subject_alpha[exterior_translucent_soft] = np.minimum(
        subject_alpha[exterior_translucent_soft],
        0.72,
    )
    soft_background_leak = _soft_background_family_leak_mask(
        image_srgb,
        bg,
        soft_subject,
        opaque_subject,
        subject_alpha,
    )
    if soft_background_leak.any():
        soft_subject &= ~soft_background_leak
        subject_alpha[soft_background_leak] = 0.0
        exterior_bg |= soft_background_leak

    glass_internal_shadow, glass_internal_shadow_info = _reclassify_glass_internal_shadow_as_hole(
        bg,
        hole,
        soft_subject,
        opaque_subject,
        shadow,
    )
    if glass_internal_shadow.any():
        # In saturated-screen glass, mild scalar darkening inside a broad
        # transparent basin is often the old background seen through glass, not
        # a reusable black shadow layer. Reclassify only components embedded in
        # the proved glass interior; exterior/contact-shadow components stay in
        # shadow so G02/G03-style cast shadows are preserved.
        shadow &= ~glass_internal_shadow
        hole |= glass_internal_shadow

    # Background-colored enclosed openings are true holes only after the broad
    # topology/shape proof in _enclosed_hole_mask; thin same-color markings stay
    # in opaque_subject as subject-owned decoration.
    alpha = subject_alpha.copy()
    alpha[hole | exterior_bg] = 0.0
    shadow_alpha_seed = _shadow_display_alpha(image_srgb, bg, shadow)
    shadow_alpha, shadow_rgb_linear, shadow_reconstruct_info = _luminance_shadow_rgba_from_known_bg(
        image_srgb,
        bg,
        shadow,
        shadow_alpha_seed,
    )
    shadow_alpha, shadow_feather, shadow_feather_info = _feather_broad_exterior_shadow_alpha(
        image_srgb,
        bg,
        shadow_alpha,
        shadow & ~internal_hole_shadow,
        exterior_bg,
        subject_alpha > 0.0,
    )
    if shadow_feather.any():
        shadow |= shadow_feather
        exterior_bg &= ~shadow_feather
    alpha[shadow] = shadow_alpha[shadow]

    C_lin = io.srgb_to_linear(image_srgb).astype(np.float32)
    B_lin = io.srgb_to_linear(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    foreground_linear = _foreground_from_known_bg(C_lin, B_lin, subject_alpha)
    internal_bg_material = _internal_background_colored_material_mask(image_srgb, bg, opaque_subject, soft_subject)
    opaque_glass_leak = _opaque_background_family_glass_leak_mask(
        image_srgb,
        bg,
        opaque_subject,
        soft_subject,
        hole,
    )
    opaque_glass_leak &= ~internal_bg_material
    if opaque_glass_leak.any():
        opaque_subject &= ~opaque_glass_leak
        subject_alpha[opaque_glass_leak] = 0.0
        alpha[opaque_glass_leak] = 0.0
        foreground_linear[opaque_glass_leak] = 0.0
        exterior_bg |= opaque_glass_leak
        internal_bg_material &= ~opaque_glass_leak
    known_bg_projection_strength = _known_background_projection_strength(
        bg,
        foreground_linear,
        subject_alpha,
        opaque_subject | soft_subject,
        image_srgb=image_srgb,
        subject_hole=hole,
        soft_subject=soft_subject,
    )
    known_bg_projection_strength[internal_bg_material] = 0.0
    known_bg_projection = known_bg_projection_strength > 1e-4
    if known_bg_projection.any():
        foreground_linear = _project_known_background_foreground(
            foreground_linear,
            B_lin,
            known_bg_projection_strength,
        )
    solved_soft_background_leak = _solved_soft_background_leak_mask(
        image_srgb,
        foreground_linear,
        bg,
        soft_subject,
        opaque_subject,
        hole,
        subject_alpha,
    )
    if solved_soft_background_leak.any():
        soft_background_leak |= solved_soft_background_leak
        soft_subject &= ~solved_soft_background_leak
        subject_alpha[solved_soft_background_leak] = 0.0
        alpha[solved_soft_background_leak] = 0.0
        foreground_linear[solved_soft_background_leak] = 0.0
        exterior_bg |= solved_soft_background_leak
    glass_color_gap, glass_color_gap_info = _restore_saturated_glass_color_shifted_gaps(
        image_srgb,
        bg,
        subject_alpha,
        alpha,
        soft_subject,
        opaque_subject,
        hole,
        shadow,
        exterior_bg,
    )
    if glass_color_gap.any():
        hole &= ~glass_color_gap
        exterior_bg &= ~glass_color_gap
        soft_subject |= glass_color_gap
        foreground_from_bg = _foreground_from_known_bg(C_lin, B_lin, subject_alpha)
        foreground_linear[glass_color_gap] = foreground_from_bg[glass_color_gap]
    foreground_linear, glass_foreground_info = _stabilize_saturated_glass_soft_foreground(
        image_srgb,
        bg,
        foreground_linear,
        subject_alpha,
        soft_subject,
        opaque_subject,
        hole,
    )
    foreground_linear, subject_alpha, alpha, thin_glass_repair_info = _repair_thin_unstable_glass_foreground_ridges(
        bg,
        foreground_linear,
        subject_alpha,
        alpha,
        soft_subject,
        opaque_subject,
        hole,
        shadow,
    )
    foreground_linear, subject_alpha, alpha, glass_continuous_field_info = _solve_saturated_glass_continuous_field(
        image_srgb,
        bg,
        foreground_linear,
        subject_alpha,
        alpha,
        soft_subject,
        opaque_subject,
        hole,
        shadow,
    )
    foreground_linear, source_preserving_glass_info = _source_preserving_saturated_glass_foreground(
        image_srgb,
        bg,
        foreground_linear,
        alpha,
        soft_subject,
        opaque_subject,
        hole,
        shadow,
    )
    rgba_rgb_linear = foreground_linear.copy()
    rgba_rgb_linear[shadow] = shadow_rgb_linear[shadow]

    masks = {
        "external_background": exterior_bg,
        "opaque_subject": opaque_subject,
        "subject_hole": hole,
        "soft_subject_layer": soft_subject,
        "shadow_layer": shadow,
        "unknown_fallback": unknown,
    }
    unknown_fraction = float(unknown.sum()) / float(h * w)
    palette_info = _subject_palette_info(image_srgb, opaque_subject | soft_subject)
    confidence = float(
        np.clip(
            1.0
            - unknown_fraction * 4.0
            - bg_info["sigma"] / 30.0
            - max(0.0, 0.55 - palette_info["top32_3bit_fraction"]),
            0.0,
            1.0,
        )
    )
    # Solid-background photos can have a perfectly deterministic exterior too.
    # Require the subject colors to remain compressible in a coarse palette:
    # graphics/icons cluster into a small number of quantized colors, while
    # photo-like texture remains spread across many bins.
    graphic_like_subject = palette_info["top32_3bit_fraction"] >= 0.55
    accepted = bool(
        confidence >= 0.70
        and unknown_fraction <= 0.05
        and subject_candidate.any()
        and graphic_like_subject
    )
    if accepted:
        reason = "accepted"
    elif not graphic_like_subject:
        reason = "subject palette is photo-like"
    else:
        reason = "ownership ambiguity too high"

    return SolidGraphicResult(
        accepted=accepted,
        reason=reason,
        confidence=confidence,
        background_color=bg,
        alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
        subject_alpha=np.clip(subject_alpha, 0.0, 1.0).astype(np.float32),
        foreground_linear=foreground_linear,
        rgba_rgb_linear=np.clip(rgba_rgb_linear, 0.0, 1.0).astype(np.float32),
        ownership_masks=masks,
        debug={
            "background": bg_info,
            "exterior_fraction": exterior_fraction,
            "unknown_fraction": unknown_fraction,
            "subject_palette": palette_info,
            "known_background_projection_pixels": int(known_bg_projection.sum()),
            "edge_background_residual_pixels": int(known_bg_projection.sum()),
            "internal_background_material_pixels": int(internal_bg_material.sum()),
            "opaque_glass_leak_pixels": int(opaque_glass_leak.sum()),
            "glass_internal_shadow_reclassified_pixels": int(glass_internal_shadow.sum()),
            "glass_internal_shadow_reclassification": glass_internal_shadow_info,
            "glass_color_shifted_gap_restore": glass_color_gap_info,
            "glass_soft_foreground_stabilization": glass_foreground_info,
            "thin_glass_foreground_ridge_repair": thin_glass_repair_info,
            "glass_continuous_field": glass_continuous_field_info,
            "source_preserving_glass_foreground": source_preserving_glass_info,
            "internal_hole_shadow_pixels": int(internal_hole_shadow.sum()),
            "exterior_shadow_feather": shadow_feather_info,
            "soft_background_leak_pixels": int(soft_background_leak.sum()),
            "exterior_translucent_soft_pixels": int(exterior_translucent_soft.sum()),
            "shadow_luminance_reconstruction": shadow_reconstruct_info,
            "mask_pixels": {name: int(mask.sum()) for name, mask in masks.items()},
        },
    )


def _estimate_stable_background(image_srgb: np.ndarray) -> tuple[tuple[int, int, int], dict[str, Any]]:
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
    # Subject can touch a corner or side in small sprites, so all-corner
    # agreement is not required. A dominant low-variance border color is still
    # deterministic known-background evidence; noisy/photo borders do not form
    # this kind of large quantized mode.
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


def _known_background_mask(image_srgb: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    # The tight gate is deliberate for graphic assets: exact-B topology should
    # decide holes/exterior, while antialiasing is handled as soft ownership.
    return _background_distance(image_srgb, bg) <= 3.5


def _background_distance(image_srgb: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    return oklab_distance(lab, bg_lab)


def _scalar_darkening_mask(image_srgb: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    denom = float(np.dot(B, B))
    if denom <= 1e-5:
        return np.zeros(image_srgb.shape[:2], dtype=bool)
    scale = np.clip(np.sum(C * B.reshape(1, 1, 3), axis=-1) / denom, 0.0, 2.0)
    recon = scale[..., None] * B.reshape(1, 1, 3)
    err = np.sqrt(np.mean((C - recon) ** 2, axis=-1))
    darker = np.mean(C, axis=-1) < np.mean(B) - 0.01
    # Experience-driven shadow gate: scalar darkening must be both chromatically
    # close to scaled B and visibly dimmer, protecting dark subject interiors.
    return (scale >= 0.18) & (scale <= 0.98) & (err <= 0.035) & darker


def _strict_background_family_darkening_mask(image_srgb: np.ndarray, bg: tuple[int, int, int]) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(image_srgb.shape[:2], dtype=bool)

    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0].astype(np.float32)
    denom = max(float(np.dot(B, B)), 1e-5)
    scale = np.clip(np.sum(C * B.reshape(1, 1, 3), axis=-1) / denom, 0.0, 2.0)
    recon = scale[..., None] * B.reshape(1, 1, 3)
    err = np.sqrt(np.mean((C - recon) ** 2, axis=-1))
    darker = np.mean(C, axis=-1) < np.mean(B) - 0.01

    pixels = image_srgb.astype(np.float32)
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    # Hole-side darkening uses a stricter chroma test than exterior shadows.
    # G03-like interior holes are near-perfect scaled-B colors; G02-like green
    # subject material can be dark and green-dominant but has a different
    # channel ratio, especially in the secondary channels.
    return (scale >= 0.03) & (scale <= 0.98) & (err <= 0.008) & darker & (norm_delta <= 0.10)


def _reclassify_glass_internal_shadow_as_hole(
    bg: tuple[int, int, int],
    subject_hole: np.ndarray,
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    shadow: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move saturated-screen glass-interior darkening out of shadow export.

    For G04-like green glass, generated source pixels inside the transparent
    basin can satisfy the scalar-darkening equation ``C ~= scale * B``. The
    exterior shadow detector is physically correct for cast/contact shadows,
    but exporting these interior basin fragments as neutral black shadow makes
    grey dirt on new backgrounds. The guard is deliberately topology-based:
    only broad glass basins with both same-background holes and soft material
    are eligible, and components must live mostly in that interior context.
    """
    out = np.zeros(shadow.shape, dtype=bool)
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return out, {"method": "glass_internal_shadow_reclassification", "applied": False, "reason": "background is not saturated"}

    img_area = float(shadow.size)
    hole_fraction = float(subject_hole.sum()) / img_area
    soft_fraction = float(soft_subject.sum()) / img_area
    if hole_fraction < 0.08 or soft_fraction < 0.04:
        return out, {
            "method": "glass_internal_shadow_reclassification",
            "applied": False,
            "reason": "no broad glass basin",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
    basin_context = hole_density >= 0.08
    anchored_context = ndimage.binary_dilation(subject_hole, iterations=8) & ndimage.binary_dilation(
        soft_subject | opaque_subject,
        iterations=8,
    )
    interior_context = basin_context | anchored_context

    labels, n = ndimage.label(shadow)
    component_infos: list[dict[str, Any]] = []
    rejected = 0
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        if area < 8:
            rejected += 1
            continue
        interior_pixels = int((comp & interior_context).sum())
        interior_fraction = interior_pixels / float(area)
        if interior_fraction < 0.65:
            rejected += 1
            continue
        out |= comp
        ys, xs = np.nonzero(comp)
        component_infos.append(
            {
                "area": area,
                "interior_fraction": interior_fraction,
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            }
        )

    return out, {
        "method": "glass_internal_shadow_reclassification",
        "applied": bool(out.any()),
        "pixels": int(out.sum()),
        "components": component_infos,
        "rejected_components": rejected,
        "hole_fraction": hole_fraction,
        "soft_fraction": soft_fraction,
        "interior_fraction_min": 0.65,
    }


def _soft_background_family_leak_mask(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_alpha: np.ndarray,
) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(soft_subject.shape, dtype=bool)

    pixels = image_srgb.astype(np.float32)
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    dominant_margin = pixels[..., dominant] - np.maximum(
        pixels[..., other_channels[0]],
        pixels[..., other_channels[1]],
    )
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0].astype(np.float32)
    not_darker_than_bg = np.mean(C, axis=-1) >= (float(np.mean(B)) - 0.002)
    # Deleting alpha is only allowed when there is broad soft-region support.
    # Isolated same-B pixels on a hard contour are better handled by foreground
    # projection so the edge remains antialiased instead of becoming chipped.
    soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=21)
    broad_soft_context = soft_density >= 0.18
    far_from_opaque = ~ndimage.binary_dilation(opaque_subject, iterations=3)
    # Low-alpha soft pixels that are still nearly pure background-green are not
    # useful foreground evidence; inverse-compositing turns them into visible
    # green wisps on dark/new backgrounds. High-alpha same-hue regions are left
    # to component ownership (for example G02's protected internal material).
    pure_background_leak = (
        soft_subject
        & (subject_alpha <= 0.35)
        & (broad_soft_context | far_from_opaque)
        & not_darker_than_bg
        & (dominant_margin >= 80.0)
        & (norm_delta <= 0.12)
    )
    # Translucent glass can lift the secondary channels while still being
    # dominated by the known screen color. Keep this broader than the pure-leak
    # gate only for low-alpha pixels; otherwise background-green subject
    # material would be erased before component-level protection can arbitrate.
    glass_mixed_background_leak = (
        soft_subject
        & (subject_alpha <= 0.45)
        & broad_soft_context
        & not_darker_than_bg
        & (dominant_margin >= 20.0)
        & (norm_delta <= 0.26)
    )
    candidates = pure_background_leak | glass_mixed_background_leak
    return _remove_small_components(candidates, min_area=3)


def _restore_saturated_glass_color_shifted_gaps(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    subject_alpha: np.ndarray,
    alpha: np.ndarray,
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_hole: np.ndarray,
    shadow_layer: np.ndarray,
    exterior_bg: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore small color-shifted glass fragments that leak into holes.

    On saturated screens, true transparent holes keep the screen channel ratios.
    Refractive glass gaps often remain green-dominant but lift red/blue enough
    to form a local color-shifted fragment. If that small fragment is embedded
    in a broad glass basin, deleting its alpha creates a grey notch on neutral
    backgrounds. The component-size guard keeps large transparent centers and
    exterior screen regions as holes.
    """
    out = np.zeros(subject_alpha.shape, dtype=bool)
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return out, {
            "method": "saturated_glass_color_shifted_gap_restore",
            "applied": False,
            "reason": "background is not saturated",
        }

    img_area = float(subject_alpha.size)
    hole_fraction = float(subject_hole.sum()) / img_area
    soft_fraction = float(soft_subject.sum()) / img_area
    if hole_fraction <= 0.08 or soft_fraction <= 0.04:
        return out, {
            "method": "saturated_glass_color_shifted_gap_restore",
            "applied": False,
            "reason": "no broad glass basin",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    subject_region = soft_subject | opaque_subject
    if not subject_region.any():
        return out, {
            "method": "saturated_glass_color_shifted_gap_restore",
            "applied": False,
            "reason": "no subject support",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    pixels = image_srgb.astype(np.float32)
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    secondary_shift = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    secondary_value = np.maximum(pixels[..., other_channels[0]], pixels[..., other_channels[1]])
    bg_close = _known_background_mask(image_srgb, bg)

    dist_to_subject = ndimage.distance_transform_edt(~subject_region)
    subject_density = ndimage.uniform_filter(subject_region.astype(np.float32), size=21)
    hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
    local_glass_context = (dist_to_subject <= 14.0) & ((subject_density >= 0.05) | (hole_density >= 0.08))
    color_shifted = (secondary_shift >= 0.28) | ((secondary_shift >= 0.20) & (secondary_value >= 52.0))
    candidates = (
        (subject_hole | exterior_bg)
        & ~shadow_layer
        & ~bg_close
        & local_glass_context
        & color_shifted
    )

    labels, n = ndimage.label(candidates)
    components: list[dict[str, Any]] = []
    restored_component_count = 0
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        if area < 3 or area > 1600:
            continue
        ys, xs = np.nonzero(comp)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        if min(width, height) > 36 and area > 360:
            continue
        out |= comp
        restored_component_count += 1
        if len(components) < 80:
            components.append(
                {
                    "area": area,
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                    "mean_secondary_shift": float(secondary_shift[comp].mean()),
                }
            )

    if not out.any():
        return out, {
            "method": "saturated_glass_color_shifted_gap_restore",
            "applied": False,
            "reason": "no small color-shifted gaps",
            "candidate_pixels": int(candidates.sum()),
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    support = subject_region & (subject_alpha > 0.03)
    support_weight = support.astype(np.float32) * np.clip(subject_alpha / 0.45, 0.20, 1.0)
    support_density = ndimage.gaussian_filter(support_weight, sigma=2.0)
    borrowed_alpha = np.zeros_like(subject_alpha)
    if float(support_density.max(initial=0.0)) > 1e-5:
        borrowed_alpha = (
            ndimage.gaussian_filter(subject_alpha * support_weight, sigma=2.0)
            / np.maximum(support_density, 1e-6)
        )
    estimated_alpha = _estimate_soft_alpha(image_srgb, bg, out)
    restored_alpha = np.maximum(estimated_alpha, borrowed_alpha * 0.72)
    # Empirical alpha bounds: restored pixels are missing glass fragments, not
    # hard opaque artwork. The lower bound keeps repaired chroma out of the
    # fragile ultra-low-alpha range where straight-F hue becomes visible dirt;
    # the upper bound avoids filling true transparent centers as solid material.
    restored_alpha = np.clip(restored_alpha, 0.26, 0.72)
    subject_alpha[out] = restored_alpha[out]
    alpha[out] = restored_alpha[out]

    return out, {
        "method": "saturated_glass_color_shifted_gap_restore",
        "applied": True,
        "pixels": int(out.sum()),
        "candidate_pixels": int(candidates.sum()),
        "component_count": restored_component_count,
        "components": components,
        "components_truncated": max(0, restored_component_count - len(components)),
        "mean_alpha": float(subject_alpha[out].mean()),
        "hole_fraction": hole_fraction,
        "soft_fraction": soft_fraction,
    }


def _solved_soft_background_leak_mask(
    image_srgb: np.ndarray,
    foreground_linear: np.ndarray,
    bg: tuple[int, int, int],
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_hole: np.ndarray,
    subject_alpha: np.ndarray,
) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(soft_subject.shape, dtype=bool)

    fg = io.linear_to_srgb_u8(foreground_linear)
    dominant_margin = fg[..., dominant].astype(np.int16) - np.maximum(
        fg[..., other_channels[0]].astype(np.int16),
        fg[..., other_channels[1]].astype(np.int16),
    )
    far_from_opaque_leak = (
        soft_subject
        & (subject_alpha <= 0.35)
        & ~ndimage.binary_dilation(opaque_subject, iterations=3)
        & (dominant_margin > 15)
    )
    pixels = image_srgb.astype(np.float32)
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    source_margin = pixels[..., dominant] - np.maximum(
        pixels[..., other_channels[0]],
        pixels[..., other_channels[1]],
    )
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0].astype(np.float32)
    source_not_darker_than_bg = np.mean(C, axis=-1) >= (float(np.mean(B)) - 0.002)
    soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=21)
    broad_soft_context = soft_density >= 0.18
    # A glass interior can be fragmented by nearby high-alpha reflection
    # texture, so the normal "far from opaque" guard would leave low-alpha
    # screen-colored speckles. Source-color evidence keeps this narrow: true
    # hard-edge antialiasing is darker than B, while glass leak is a lifted
    # background-family color.
    near_opaque_glass_leak = (
        soft_subject
        & (subject_alpha <= 0.35)
        & broad_soft_context
        & (dominant_margin > 15)
        & source_not_darker_than_bg
        & (source_margin >= 20.0)
        & (norm_delta <= 0.55)
    )
    fg_mean = fg.astype(np.float32).mean(axis=-1)
    fg_chroma = fg.astype(np.int16).max(axis=-1) - fg.astype(np.int16).min(axis=-1)
    # If a background-family soft pixel solves to a dark low-chroma foreground,
    # the solve has manufactured dirt rather than recovered glass color. This
    # is the soft counterpart of _opaque_background_family_glass_leak_mask and
    # is limited to broad soft regions so real hard outlines are not erased.
    dark_solved_glass_leak = (
        soft_subject
        & (subject_alpha <= 0.58)
        & broad_soft_context
        & (source_margin >= 55.0)
        & (norm_delta <= 0.60)
        & (fg_mean < 90.0)
        & (fg_chroma < 85)
    )
    perceptual_bg_delta = oklab_distance(
        srgb_to_oklab(image_srgb),
        srgb_to_oklab(bg_arr.reshape(1, 1, 3))[0, 0],
    )
    local_bg_family = _local_background_family_continuity_mask(
        image_srgb,
        bg,
        soft_subject,
        broad_soft_context | (ndimage.uniform_filter(subject_hole.astype(np.float32), size=31) >= 0.08),
    )
    other_max = np.maximum(
        fg[..., other_channels[0]].astype(np.int16),
        fg[..., other_channels[1]].astype(np.int16),
    )
    # Broad-glass context is an empirical guard from G04: large transparent
    # basins contain many near-background pixels, while ordinary icons only
    # have thin antialiasing rims. Keep false-hue deletion in that broad basin
    # so real colored soft materials (G06/G10-style glow/gradients) are not
    # globally punched out just because some pixels are low-alpha.
    global_glass_context = float(subject_hole.mean()) > 0.10 and float(soft_subject.mean()) > 0.05
    # Projection must not invent hue. In a broad glass basin, if the source
    # pixel is still perceptually near the green screen but the solved foreground
    # flips to red/blue/purple dominance, that color is an inverse-solve
    # artifact. True colored glass without this large-hole context is left to
    # the local color-manifold path instead of being globally deleted.
    false_hue_glass_leak = (
        soft_subject
        & global_glass_context
        & (subject_alpha <= 0.25)
        # OKLab <= 8 is the direct perceptual anchor; local_bg_family is the
        # local-diffusion escape hatch for pixels the eye reads as continuous
        # with neighboring screen-colored glass even when they miss the flat-B
        # threshold by a small amount.
        & ((perceptual_bg_delta <= 8.0) | local_bg_family)
        & ((other_max - fg[..., dominant].astype(np.int16)) > 25)
    )
    return (
        far_from_opaque_leak
        | near_opaque_glass_leak
        | dark_solved_glass_leak
        | false_hue_glass_leak
    )


def _opaque_background_family_glass_leak_mask(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    opaque_subject: np.ndarray,
    soft_subject: np.ndarray,
    subject_hole: np.ndarray,
) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(opaque_subject.shape, dtype=bool)

    pixels = image_srgb.astype(np.float32)
    dominant_margin = pixels[..., dominant] - np.maximum(
        pixels[..., other_channels[0]],
        pixels[..., other_channels[1]],
    )
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0].astype(np.float32)
    not_darker_than_bg = np.mean(C, axis=-1) >= (float(np.mean(B)) - 0.002)
    perceptual_bg_delta = oklab_distance(
        srgb_to_oklab(image_srgb),
        srgb_to_oklab(bg_arr.reshape(1, 1, 3))[0, 0],
    )
    perceptually_near_bg = perceptual_bg_delta <= 8.0
    soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=31)
    hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
    local_bg_family = _local_background_family_continuity_mask(
        image_srgb,
        bg,
        opaque_subject,
        (soft_density >= 0.12) | (hole_density >= 0.08),
    )
    # G04-like glass interiors can contain small opaque-classified background
    # flecks. If those pixels are projected as opaque foreground they turn into
    # black/grey dirt. Require both background-family source color and local
    # glass context (soft layer or hole density), so real dark outlines and G02
    # same-hue subject material are not deleted. The perceptual OKLab gate is
    # intentionally anchored to visibility: tiny hue/luma shifts that are barely
    # noticeable on the original green screen should not become opaque exported
    # foreground just because the inverse solve can magnify them.
    contextual_candidates = (
        opaque_subject
        & (not_darker_than_bg | perceptually_near_bg | local_bg_family)
        & (dominant_margin >= 20.0)
        & (norm_delta <= 0.55)
        & ((soft_density >= 0.12) | (hole_density >= 0.08))
    )
    # Same broad-glass guard as the soft false-hue path: this branch deletes
    # tiny opaque-classified dust only inside a large glass/hole basin, where
    # source-space local continuity proves it belongs to the background family.
    global_glass_context = float(subject_hole.mean()) > 0.10 and float(soft_subject.mean()) > 0.05
    # In a large glass basin, isolated one- or two-pixel specks are often the
    # most visible exported dirt. Keep this extra gate global-context-only so
    # normal hard-edge green antialiasing dust still keeps alpha and is handled
    # by color projection instead of being punched out.
    tiny_perceptual_dust = (
        opaque_subject
        & global_glass_context
        & (perceptually_near_bg | local_bg_family)
        & (dominant_margin >= 20.0)
        & (norm_delta <= 0.55)
    )
    return _remove_small_components(contextual_candidates, min_area=3) | tiny_perceptual_dust


def _local_background_family_continuity_mask(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    candidate_mask: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray:
    """Return candidates that are locally indistinguishable from screen color.

    This is not a replacement for known-background solving. It is a perceptual
    gate used after ownership context is already known: if a candidate pixel is
    background-family in channel direction and lies among proven screen-colored
    neighbors, a slightly wider local OKLab threshold is allowed. That matches
    how G04's glass speckles disappear on the original green screen but become
    visible after inverse compositing magnifies their residual color.
    """
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(candidate_mask.shape, dtype=bool)

    pixels = image_srgb.astype(np.float32)
    dominant_margin = pixels[..., dominant] - np.maximum(
        pixels[..., other_channels[0]],
        pixels[..., other_channels[1]],
    )
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(bg_arr.reshape(1, 1, 3))[0, 0]
    bg_delta = oklab_distance(lab, bg_lab)
    # These channel-direction gates are broad by design but still require the
    # dominant screen channel to lead. They prevent local diffusion from walking
    # into true blue/purple glass gradients; the empirical widths were chosen
    # from G04 speckles and checked against G06/G10 soft colored material.
    bg_family = (dominant_margin >= 18.0) & (norm_delta <= 0.62)
    # Seeds use the stricter global perceptual threshold. Nearby pixels can
    # inherit tolerance only from already-proved background-family evidence.
    seed = bg_family & (bg_delta <= 8.0) & context_mask
    seed_weight = seed.astype(np.float32)
    # A 15px window is a local perceptual neighborhood for large UI/glass
    # assets: large enough to bridge dither-like speckles, small enough that it
    # does not diffuse across whole gradients or unrelated colored regions.
    seed_density = ndimage.uniform_filter(seed_weight, size=15)
    denom = np.maximum(ndimage.uniform_filter(seed_weight, size=15), 1e-6)
    local_lab = np.empty_like(lab, dtype=np.float32)
    for channel in range(3):
        local_lab[..., channel] = ndimage.uniform_filter(lab[..., channel] * seed_weight, size=15) / denom
    local_delta = oklab_distance(lab, local_lab)
    # Human-visible continuity is local: in soft glass, a pixel that is very
    # close to nearby proven background-family pixels should remain removable
    # even when its absolute distance from the flat screen color is slightly
    # larger. seed_density >= 0.08 requires a real local neighborhood instead of
    # one lucky seed; local_delta <= 4.5 is tighter than the global 8 dE anchor
    # because this branch is expanding reach through neighbors. The global B
    # gate still supplies the physical anchor, while this local gate prevents
    # invisible speckle differences from becoming exported foreground after
    # inverse compositing.
    return (
        candidate_mask
        & context_mask
        & bg_family
        & (seed_density >= 0.08)
        & (local_delta <= 4.5)
    )


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    seed = np.zeros(passable.shape, dtype=bool)
    seed[0, :] = passable[0, :]
    seed[-1, :] = passable[-1, :]
    seed[:, 0] = passable[:, 0]
    seed[:, -1] = passable[:, -1]
    return ndimage.binary_propagation(seed, mask=passable)


def _enclosed_hole_mask(candidates: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(candidates)
    out = np.zeros(candidates.shape, dtype=bool)
    if n == 0:
        return out
    dist = ndimage.distance_transform_edt(candidates)
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        # Broad enclosed same-B basins are holes. Thin same-color strokes or
        # highlights are usually subject decoration, so they stay foreground.
        # The area/depth gates are empirical topology guards from G02/G03: they
        # require a real opening, not a one-pixel green marking or reflection.
        if area >= 16 and float(dist[comp].max()) >= 3.0:
            out |= comp
    return out


def _scalar_darkening_reachable_from_holes(
    bg_close: np.ndarray,
    hole_darkening: np.ndarray,
    exterior_bg: np.ndarray,
    hole_region: np.ndarray,
) -> np.ndarray:
    if not hole_region.any():
        return np.zeros(hole_region.shape, dtype=bool)
    # Treat proved interior holes as local background seeds. Dark pixels that
    # are connected to that seed and satisfy the same known-B scalar equation as
    # exterior shadows are hole-side shadow, not subject material. This is the
    # topology mirror of the outer-edge rule and prevents G03-like green-screen
    # darkening inside holes from being exported as foreground RGB. Excluding
    # exterior_bg keeps outer/drop shadows from being pulled into the interior
    # hole proof.
    return hole_region & hole_darkening & ~exterior_bg


def _keep_subject_anchored_components(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    min_area = max(4, int(mask.size * 0.0005))
    out = np.zeros(mask.shape, dtype=bool)
    for i, area in enumerate(areas, start=1):
        if int(area) >= min_area:
            out |= labels == i
    return out


def _remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    out = np.zeros(mask.shape, dtype=bool)
    areas = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    for label_id, area in enumerate(areas, start=1):
        if int(area) >= min_area:
            out |= labels == label_id
    return out


def _subject_palette_info(image_srgb: np.ndarray, subject_mask: np.ndarray) -> dict[str, Any]:
    pixels = image_srgb[subject_mask]
    if len(pixels) == 0:
        return {"pixels": 0, "top32_3bit_fraction": 0.0}
    q = (pixels >> 5).astype(np.int32)
    keys = q[:, 0] * 8 * 8 + q[:, 1] * 8 + q[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    counts.sort()
    top32 = float(counts[-32:].sum()) / float(counts.sum())
    return {"pixels": int(len(pixels)), "top32_3bit_fraction": top32}


def _estimate_soft_alpha(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros(image_srgb.shape[:2], dtype=np.float32)
    if not mask.any():
        return out
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    d = oklab_distance(lab, bg_lab)
    # Empirical graphic-edge ramp: background-adjacent pixels below 3.5 dE are
    # exterior; by about 24 dE they are visually owned by foreground material.
    out[mask] = np.clip((d[mask] - 3.5) / (24.0 - 3.5), 0.05, 0.95)
    return out


def _shadow_display_alpha(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros(image_srgb.shape[:2], dtype=np.float32)
    if not mask.any():
        return out
    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    denom = max(float(np.dot(B, B)), 1e-5)
    scale = np.clip(np.sum(C * B.reshape(1, 1, 3), axis=-1) / denom, 0.0, 1.0)
    out[mask] = np.clip(1.0 - scale[mask], 0.0, 1.0)
    return out


def _luminance_shadow_rgba_from_known_bg(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    mask: np.ndarray,
    preferred_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Export shadow as neutral darkening while matching source luminance.

    Exact RGB reconstruction over a saturated known background is possible, but
    it encodes part of that background color into the shadow layer. That looks
    wrong when the RGBA is placed on a new background. The visual invariant for
    reusable assets is therefore luminance, not chroma: solve a neutral black
    shadow alpha so ``Y((1-a)B)`` follows the observed ``Y(C)`` on the original
    background, while the RGBA shadow RGB remains green-free.
    """
    alpha = np.zeros(image_srgb.shape[:2], dtype=np.float32)
    rgb_linear = np.zeros(image_srgb.shape, dtype=np.float32)
    pixels = int(mask.sum())
    if pixels == 0:
        return alpha, rgb_linear, {
            "method": "known_bg_luminance_neutral_shadow",
            "pixels": 0,
            "alpha_changed_pixels": 0,
            "mean_alpha_before": 0.0,
            "mean_alpha_after": 0.0,
            "p95_alpha_after": 0.0,
            "max_luminance_abs_error": 0.0,
        }

    target_u8 = image_srgb[mask]
    bg_u8 = np.asarray(bg, dtype=np.uint8)
    target_lin = io.srgb_to_linear(target_u8)
    bg_lin = io.srgb_to_linear(bg_u8.reshape(1, 3))[0]
    preferred = np.clip(preferred_alpha[mask].astype(np.float32), 0.0, 1.0)

    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    target_y = np.sum(target_lin * weights.reshape(1, 3), axis=1)
    bg_y = float(np.sum(bg_lin * weights))
    if bg_y <= 1e-6:
        alpha_values = preferred
    else:
        alpha_values = np.clip(1.0 - (target_y / bg_y), 0.0, 1.0)

    # Quantize the alpha deliberately because the exported PNG is 8-bit. Try
    # the nearest neighboring alpha codes and keep the one with lowest linear
    # luminance error over the source background.
    alpha_center = np.clip((alpha_values * 255.0 + 0.5).astype(np.int16), 0, 255)
    candidates = np.stack(
        [
            np.maximum(alpha_center - 1, 0),
            alpha_center,
            np.minimum(alpha_center + 1, 255),
        ],
        axis=1,
    )
    candidate_alpha = candidates.astype(np.float32) / 255.0
    candidate_y = (1.0 - candidate_alpha) * bg_y
    best = np.argmin(np.abs(candidate_y - target_y[:, None]), axis=1)
    alpha_u8 = candidates[np.arange(pixels), best]
    original_alpha_u8 = np.clip((preferred * 255.0 + 0.5).astype(np.int16), 0, 255)
    alpha_values = alpha_u8.astype(np.float32) / 255.0

    alpha[mask] = alpha_values
    rgb_linear[mask] = 0.0
    final_y_error = np.abs((1.0 - alpha_values) * bg_y - target_y)

    return alpha, rgb_linear, {
        "method": "known_bg_luminance_neutral_shadow",
        "pixels": pixels,
        "alpha_changed_pixels": int((alpha_u8 != original_alpha_u8).sum()),
        "mean_alpha_before": float(preferred.mean()),
        "mean_alpha_after": float(alpha_values.mean()),
        "p95_alpha_after": float(np.percentile(alpha_values, 95.0)),
        "max_alpha_after": float(alpha_values.max(initial=0.0)),
        "mean_luminance_abs_error": float(final_y_error.mean()),
        "p95_luminance_abs_error": float(np.percentile(final_y_error, 95.0)),
        "max_luminance_abs_error": float(final_y_error.max(initial=0.0)),
    }


def _feather_broad_exterior_shadow_alpha(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    alpha: np.ndarray,
    exterior_shadow: np.ndarray,
    exterior_bg: np.ndarray,
    subject_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve broad soft cast-shadow tails as a continuous known-B field.

    Once a broad exterior region is proven to be shadow, the remaining problem
    is not classification but field recovery: on a known background
    ``C ~= scale * B`` and the reusable black shadow is ``alpha = 1 - scale``.
    Generated green-screen tails contain measurement noise and faint pixels
    below the strict seed mask, so solve a weighted continuous field from the
    source luminance and constrain its outside boundary to zero. Hard UI
    shadows and hole-side internal darkening stay exact by component gates.
    """
    out = alpha.astype(np.float32).copy()
    added = np.zeros(alpha.shape, dtype=bool)
    if not exterior_shadow.any():
        return out, added, {
            "method": "broad_exterior_shadow_feather",
            "applied": False,
            "reason": "no exterior shadow",
        }

    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    B = io.srgb_to_linear(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
    weights_y = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    bg_y = max(float(np.sum(B * weights_y)), 1e-6)
    target_y = np.sum(C * weights_y.reshape(1, 1, 3), axis=-1)
    raw_alpha = np.clip(1.0 - target_y / bg_y, 0.0, 1.0).astype(np.float32)
    denom = max(float(np.dot(B, B)), 1e-6)
    scale = np.clip(np.sum(C * B.reshape(1, 1, 3), axis=-1) / denom, 0.0, 1.2)
    recon = scale[..., None] * B.reshape(1, 1, 3)
    scalar_err = np.sqrt(np.mean((C - recon) ** 2, axis=-1))
    # Soft evidence, not ownership: it says whether source pixels still obey a
    # scaled-known-background model. The broad component already supplied the
    # semantic "this is shadow" anchor.
    scalar_conf = np.exp(-((scalar_err / 0.045) ** 2)).astype(np.float32)
    dark_conf = np.clip(raw_alpha / 0.09, 0.0, 1.0).astype(np.float32)
    source_shadow_like = (raw_alpha > (0.7 / 255.0)) & (scalar_conf > 0.08)

    labels, n = ndimage.label(exterior_shadow)
    selected = np.zeros(alpha.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    image_area = float(alpha.size)
    min_area = max(1200.0, image_area * 0.015)
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        if area < min_area:
            continue
        values = alpha[comp]
        p20 = float(np.percentile(values, 20.0))
        p50 = float(np.percentile(values, 50.0))
        low_tail_fraction = float((values < 0.18).mean())
        # Broad generated cast shadows have a substantial low-opacity tail.
        # A hard rectangle/contact shadow can be large too, but lacks this
        # low-alpha distribution, so it keeps the luminance-exact export.
        if p20 >= 0.22 or low_tail_fraction < 0.18:
            continue
        selected |= comp
        ys, xs = np.nonzero(comp)
        components.append(
            {
                "area": area,
                "p20_alpha": p20,
                "p50_alpha": p50,
                "low_tail_fraction": low_tail_fraction,
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            }
        )

    if not selected.any():
        return out, added, {
            "method": "broad_exterior_shadow_feather",
            "applied": False,
            "reason": "no broad low-alpha exterior shadow components",
            "component_count": int(n),
            "min_area": float(min_area),
        }

    selected_f = selected.astype(np.float32)
    dist = ndimage.distance_transform_edt(~selected)
    max_feather_px = 24.0
    # Generated soft shadows often have small holes and horizontal missing
    # strips inside the accepted component. Close only a few pixels so we mend
    # local support discontinuities without changing the broad shadow shape.
    closed_support = ndimage.binary_closing(selected, structure=np.ones((5, 5), dtype=bool), iterations=2)
    gap_zone = closed_support & ~selected & ~subject_mask
    feather_zone = (
        (dist > 0.0)
        & (dist <= max_feather_px)
        & ~subject_mask
        & (source_shadow_like | exterior_bg)
    )
    solve_domain = selected | gap_zone | feather_zone
    confidence = np.zeros(alpha.shape, dtype=np.float32)
    confidence[selected] = np.maximum(0.45, scalar_conf[selected])
    continuity_conf = np.clip(scalar_conf * dark_conf, 0.0, 1.0)
    confidence[gap_zone | feather_zone] = np.maximum(
        confidence[gap_zone | feather_zone],
        continuity_conf[gap_zone | feather_zone],
    )
    confidence[~solve_domain] = 0.0

    sigma_field = 7.0
    # Include a zero-valued boundary in the denominator. This makes the solved
    # field naturally decay to transparent instead of ending where the hard
    # seed mask stopped.
    boundary = ndimage.binary_dilation(solve_domain, iterations=12) & ~solve_domain & ~subject_mask
    boundary_weight = boundary.astype(np.float32) * 0.75
    numerator = ndimage.gaussian_filter(raw_alpha * confidence, sigma=sigma_field)
    denominator = ndimage.gaussian_filter(confidence + boundary_weight, sigma=sigma_field)
    field = numerator / np.maximum(denominator, 1e-6)
    field = np.clip(field, 0.0, 1.0).astype(np.float32)

    # In soft tails, trust the continuous known-B field more than the quantized
    # per-pixel inverse solve. Near contact-dark regions, keep more of the
    # measured alpha so the subject does not visually detach from its shadow.
    blend = 0.96 * np.clip((0.70 - raw_alpha) / 0.70, 0.0, 1.0)
    out[selected] = out[selected] * (1.0 - blend[selected]) + field[selected] * blend[selected]
    write = (gap_zone | feather_zone) & (field > (1.0 / 255.0))
    out[write] = np.maximum(out[write], np.minimum(field[write], 0.22))
    added |= write

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out, added, {
        "method": "broad_exterior_shadow_continuous_field",
        "applied": True,
        "pixels": int(selected.sum()),
        "added_pixels": int(added.sum()),
        "component_count": len(components),
        "components": components,
        "max_feather_px": max_feather_px,
        "field_sigma": sigma_field,
        "closed_gap_pixels": int(gap_zone.sum()),
        "mean_scalar_confidence": float(confidence[solve_domain].mean()) if solve_domain.any() else 0.0,
    }


def _known_background_projection_strength(
    bg: tuple[int, int, int],
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
    subject_mask: np.ndarray,
    *,
    image_srgb: np.ndarray | None = None,
    subject_hole: np.ndarray | None = None,
    soft_subject: np.ndarray | None = None,
) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(subject_mask.shape, dtype=np.float32)

    foreground_srgb = io.linear_to_srgb_u8(foreground_linear)
    fg = foreground_srgb.astype(np.int16)
    fg_f32 = foreground_srgb.astype(np.float32)
    bg_f32 = bg_arr.astype(np.float32)
    dominant_margin = fg[..., dominant] - np.maximum(fg[..., other_channels[0]], fg[..., other_channels[1]])
    rgb_dist = np.sqrt(np.sum((fg_f32 - bg_f32.reshape(1, 1, 3)) ** 2, axis=-1))
    edge_band = subject_mask & ndimage.binary_dilation(~subject_mask, iterations=3)
    # Core known-B color recovery: every subject-owned soft layer can be
    # unmixed against the known background. For opaque pixels, limit projection
    # to the topology boundary so real same-hue subject interiors are preserved.
    projection_region = subject_mask & (edge_band | (subject_alpha < 0.98))
    # Glass interiors can solve to pale green/cyan foreground values whose RGB
    # distance from B is larger than a hard edge, even though the green-dominant
    # direction still proves leftover known-background contribution. Use the
    # wider distance only for non-opaque pixels; opaque same-hue interiors stay
    # constrained to the narrower boundary/material-protection path.
    soft_region = projection_region & (subject_alpha > 1e-3) & (subject_alpha < 0.98)
    soft_seed = soft_region & (dominant_margin >= 38) & (rgb_dist <= 240.0)
    soft_contour_projection = soft_seed & edge_band
    opaque_edge_projection = (
        projection_region
        & edge_band
        & (subject_alpha >= 0.98)
        & (dominant_margin >= 38)
        & (rgb_dist <= 120.0)
    )
    strength = np.zeros(subject_mask.shape, dtype=np.float32)
    strength[opaque_edge_projection] = 1.0
    # Contour soft pixels should keep their alpha for antialiasing, but their
    # color can be projected firmly because the neighboring hard subject edge
    # supplies the ownership proof. Broad glass interiors are handled below by
    # a distance-weighted field instead of this crisp contour rule.
    strength[soft_contour_projection] = 1.0
    if soft_seed.any():
        # Soft glass/antialiasing should not flip abruptly at a global channel
        # threshold. High-confidence known-B residuals seed the correction, then
        # nearby pixels in the same soft layer inherit a decayed weight. This
        # keeps transitions smooth and avoids turning green removal into a hard
        # cyan/blue edge inside translucent material.
        dist_to_seed = ndimage.distance_transform_edt(~soft_seed)
        diffusion = np.exp(-((dist_to_seed / 5.0) ** 2)).astype(np.float32)
        local_evidence = np.clip(
            (dominant_margin.astype(np.float32) - 12.0) / (38.0 - 12.0),
            0.0,
            1.0,
        )
        local_evidence *= np.clip((320.0 - rgb_dist.astype(np.float32)) / (320.0 - 220.0), 0.20, 1.0)
        soft_strength = np.clip(diffusion * local_evidence, 0.0, 1.0)
        soft_strength[soft_seed] = np.maximum(soft_strength[soft_seed], 0.65)
        soft_strength = ndimage.gaussian_filter(soft_strength, sigma=1.0)
        soft_strength[~soft_region] = 0.0
        strength = np.maximum(strength, np.clip(soft_strength, 0.0, 1.0).astype(np.float32))
    if image_srgb is not None and subject_hole is not None and soft_subject is not None:
        hole_fraction = float(subject_hole.mean())
        soft_fraction = float(soft_subject.mean())
        # Broad saturated-screen glass contains large same-background basins
        # plus soft refractive rims. Residual green patches there are known-B
        # transmission, not subject hue. Use source-space background-family
        # evidence to extend projection into those rims; the broad-basin guard
        # prevents ordinary green subject material or thin antialiasing from
        # being globally de-greened.
        if hole_fraction > 0.08 and soft_fraction > 0.04:
            pixels = image_srgb.astype(np.float32)
            source_margin = pixels[..., dominant] - np.maximum(
                pixels[..., other_channels[0]],
                pixels[..., other_channels[1]],
            )
            bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
            pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
            norm_delta = np.maximum(
                np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
                np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
            )
            hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
            soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=21)
            glass_context = (hole_density >= 0.08) | (soft_density >= 0.18)
            source_bg_family = (source_margin >= 8.0) & (norm_delta <= 0.90)
            solved_bg_family = dominant_margin >= 8
            glass_seed = soft_region & glass_context & (source_bg_family | solved_bg_family)
            if glass_seed.any():
                dist_to_seed = ndimage.distance_transform_edt(~glass_seed)
                diffusion = np.exp(-((dist_to_seed / 9.0) ** 2)).astype(np.float32)
                source_evidence = np.clip((source_margin - 2.0) / (28.0 - 2.0), 0.0, 1.0)
                source_evidence *= np.clip((0.95 - norm_delta) / (0.95 - 0.45), 0.15, 1.0)
                solved_evidence = np.clip((dominant_margin.astype(np.float32) - 4.0) / (24.0 - 4.0), 0.0, 1.0)
                glass_strength = diffusion * np.maximum(source_evidence, solved_evidence)
                glass_strength[glass_seed] = np.maximum(glass_strength[glass_seed], 0.55)
                glass_strength = ndimage.gaussian_filter(glass_strength, sigma=1.2)
                glass_strength[~(soft_region & glass_context)] = 0.0
                strength = np.maximum(strength, np.clip(glass_strength, 0.0, 0.92).astype(np.float32))
    return strength


def _internal_background_colored_material_mask(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    opaque_subject: np.ndarray,
    soft_subject: np.ndarray,
) -> np.ndarray:
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return np.zeros(opaque_subject.shape, dtype=bool)

    subject = opaque_subject | soft_subject
    edge_band = subject & ndimage.binary_dilation(~subject, iterations=3)
    pixels = image_srgb.astype(np.int16)
    bg_i16 = bg_arr.astype(np.int16)
    dominant_margin = pixels[..., dominant] - np.maximum(pixels[..., other_channels[0]], pixels[..., other_channels[1]])
    rgb_dist = np.sqrt(np.sum((image_srgb.astype(np.float32) - bg_arr.astype(np.float32).reshape(1, 1, 3)) ** 2, axis=-1))
    internal_candidates = subject & ~edge_band & (dominant_margin >= 38) & (rgb_dist <= 80.0)

    labels, n = ndimage.label(internal_candidates)
    if n == 0:
        return internal_candidates
    min_area = max(128, int(internal_candidates.size * 0.00015))
    out = np.zeros(internal_candidates.shape, dtype=bool)
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        if area < min_area:
            continue
        soft_pixels = int((comp & soft_subject).sum())
        opaque_pixels = int((comp & opaque_subject).sum())
        if soft_pixels <= 0 or opaque_pixels <= 0:
            continue
        opaque_fraction = opaque_pixels / float(area)
        # Component-level arbitration: a sizeable internal same-B-family region
        # that mixes soft and opaque ownership is likely subject material, not a
        # per-pixel spill field. Protect the whole component from known-B
        # projection; otherwise one contiguous in-subject green region gets
        # partially de-greened and partially preserved.
        if 0.10 <= opaque_fraction <= 0.95:
            out |= comp
    return out


def _project_known_background_foreground(
    foreground_linear: np.ndarray,
    bg_linear: np.ndarray,
    projection_strength: np.ndarray,
) -> np.ndarray:
    bg = np.clip(bg_linear.astype(np.float32), 0.0, 1.0)
    dominant = int(np.argmax(bg))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if float(bg[dominant]) <= 1e-4:
        return foreground_linear

    repaired = foreground_linear.copy()
    projection_mask = projection_strength > 1e-4
    values = repaired[projection_mask]
    other_max = np.maximum(values[:, other_channels[0]], values[:, other_channels[1]])
    # Preserve the solved soft alpha: these pixels are on the antialiased edge,
    # and deleting alpha creates stair-stepping. Because the background is known,
    # remove only the residual component along B from the solved foreground
    # color until the background-dominant channel is no longer dominant. This is
    # a constrained unmix/projection, not neighbor color borrowing.
    target_dominant = other_max
    excess = np.maximum(values[:, dominant] - target_dominant, 0.0)
    strength = np.clip(projection_strength[projection_mask].astype(np.float32), 0.0, 1.0)
    subtract_scale = (excess / float(bg[dominant])) * strength
    projected = np.clip(values - subtract_scale[:, None] * bg.reshape(1, 3), 0.0, 1.0)
    repaired[projection_mask] = projected
    return repaired


def _stabilize_saturated_glass_soft_foreground(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_hole: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extend stable glass color over underconstrained soft foreground RGB.

    In a broad saturated-screen glass basin, soft pixels that are still
    source-background-family can have a valid alpha but an unstable straight
    foreground solve: the division by small/medium alpha amplifies tiny known-B
    residuals into dark lines or green flecks. Alpha remains the transparency
    signal; only RGB is borrowed from nearby stable subject/glass material.
    """
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return foreground_linear, {
            "method": "saturated_glass_soft_foreground_stabilization",
            "applied": False,
            "reason": "background is not saturated",
        }

    hole_fraction = float(subject_hole.mean())
    soft_fraction = float(soft_subject.mean())
    if hole_fraction <= 0.08 or soft_fraction <= 0.04:
        return foreground_linear, {
            "method": "saturated_glass_soft_foreground_stabilization",
            "applied": False,
            "reason": "no broad glass basin",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    pixels = image_srgb.astype(np.float32)
    source_margin = pixels[..., dominant] - np.maximum(
        pixels[..., other_channels[0]],
        pixels[..., other_channels[1]],
    )
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    source_bg_family = (source_margin >= 6.0) & (norm_delta <= 0.95)

    fg_srgb = io.linear_to_srgb_u8(foreground_linear)
    fg = fg_srgb.astype(np.float32)
    fg_mean = fg.mean(axis=-1)
    fg_chroma = fg.max(axis=-1) - fg.min(axis=-1)
    fg_green_margin = fg[..., dominant] - np.maximum(fg[..., other_channels[0]], fg[..., other_channels[1]])

    hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
    soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=21)
    glass_context = hole_density >= 0.08
    mid_soft = (subject_alpha >= 0.12) & (subject_alpha <= 0.86)
    unstable_dark = (fg_mean < 112.0) & (fg_chroma < 170.0)
    unstable_green = fg_green_margin > 10.0
    invalid = soft_subject & glass_context & mid_soft & source_bg_family & (unstable_dark | unstable_green)
    if not invalid.any():
        return foreground_linear, {
            "method": "saturated_glass_soft_foreground_stabilization",
            "applied": False,
            "reason": "no unstable soft foreground pixels",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    stable_soft = (
        soft_subject
        & glass_context
        & ~invalid
        & (subject_alpha >= 0.18)
        & (fg_mean >= 112.0)
        & (fg_green_margin <= 10.0)
    )
    stable_opaque = opaque_subject & (fg_mean >= 90.0) & (fg_green_margin <= 16.0)
    seeds = stable_soft | stable_opaque
    if not seeds.any():
        return foreground_linear, {
            "method": "saturated_glass_soft_foreground_stabilization",
            "applied": False,
            "reason": "no stable foreground color seeds",
            "invalid_pixels": int(invalid.sum()),
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    _, nearest = ndimage.distance_transform_edt(~seeds, return_indices=True)
    repaired = foreground_linear.copy()
    repaired[invalid] = foreground_linear[nearest[0][invalid], nearest[1][invalid]]
    return repaired, {
        "method": "saturated_glass_soft_foreground_stabilization",
        "applied": True,
        "invalid_pixels": int(invalid.sum()),
        "seed_pixels": int(seeds.sum()),
        "hole_fraction": hole_fraction,
        "soft_fraction": soft_fraction,
        "mean_invalid_alpha": float(subject_alpha[invalid].mean()),
        "mean_invalid_fg_green_margin_before": float(fg_green_margin[invalid].mean()),
        "mean_invalid_fg_luma_before": float(fg_mean[invalid].mean()),
    }


def _repair_thin_unstable_glass_foreground_ridges(
    bg: tuple[int, int, int],
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
    alpha: np.ndarray,
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_hole: np.ndarray,
    shadow_layer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Repair thin dark foreground-solve ridges in broad glass basins."""
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return foreground_linear, subject_alpha, alpha, {
            "method": "thin_unstable_glass_foreground_ridge_repair",
            "applied": False,
            "reason": "background is not saturated",
        }

    hole_fraction = float(subject_hole.mean())
    soft_fraction = float(soft_subject.mean())
    if hole_fraction <= 0.08 or soft_fraction <= 0.04:
        return foreground_linear, subject_alpha, alpha, {
            "method": "thin_unstable_glass_foreground_ridge_repair",
            "applied": False,
            "reason": "no broad glass basin",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    fg = io.linear_to_srgb_u8(foreground_linear).astype(np.float32)
    fg_mean = fg.mean(axis=-1)
    subject_region = soft_subject | opaque_subject
    # Experience-driven luma/alpha gates: the failure is a visibly dark
    # straight-F solve ridge on transparent glass, not a saturated/cyan glass
    # highlight or a barely visible alpha fringe.
    dark_subject = subject_region & (subject_alpha >= 0.22) & (fg_mean < 92.0)
    labels, n = ndimage.label(dark_subject)
    cap_mask = np.zeros_like(dark_subject, dtype=bool)
    high_alpha_black_mask = np.zeros_like(dark_subject, dtype=bool)
    components: list[dict[str, Any]] = []
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        if area < 3:
            continue
        ys, xs = np.nonzero(comp)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        # In broad saturated glass, one- or few-pixel dark ridges can be
        # promoted to soft/opaque subject by local evidence even though they
        # are unstable foreground-solve residue. The 14px empirical thinness
        # allowance covers high-resolution rasterized glass ridges; the area
        # cap keeps broad contact shadows and large glass gradients untouched.
        if min(width, height) > 14 or area > 3000:
            continue
        cap_mask |= comp
        components.append(
            {
                "area": area,
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            }
        )

    high_alpha_black = subject_region & ~shadow_layer & (subject_alpha >= 0.60) & (fg_mean < 45.0)
    labels, n = ndimage.label(high_alpha_black)
    for label_id in range(1, n + 1):
        comp = labels == label_id
        area = int(comp.sum())
        if area < 2:
            continue
        ys, xs = np.nonzero(comp)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        # The bottom-left failure is a short near-black opaque arc embedded in
        # a glass rim. Treat only small/local high-alpha black strokes here;
        # broad shadows remain excluded by size and by the shadow mask.
        if area > 600 or max(width, height) > 96 or min(width, height) > 32:
            continue
        high_alpha_black_mask |= comp
        components.append(
            {
                "area": area,
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                "kind": "high_alpha_black_ridge",
            }
        )
    cap_mask |= high_alpha_black_mask

    if not cap_mask.any():
        return foreground_linear, subject_alpha, alpha, {
            "method": "thin_unstable_glass_foreground_ridge_repair",
            "applied": False,
            "reason": "no thin unstable soft components",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    # A lower alpha alone still leaves visible dark strokes when straight-F is
    # near black. Borrow RGB from nearby stable glass/material pixels, while
    # keeping the repaired pixels' alpha soft. This targets foreground-solve
    # residue; broad shadow components are excluded by the thinness/area gate.
    fg_chroma = fg.max(axis=-1) - fg.min(axis=-1)
    stable_color_seed = (
        subject_region
        & ~cap_mask
        & (subject_alpha >= 0.18)
        & (fg_mean >= 100.0)
        & (fg_chroma >= 25.0)
    )
    repaired_foreground = foreground_linear.copy()
    seed_pixels = int(stable_color_seed.sum())
    if seed_pixels > 0:
        _, nearest = ndimage.distance_transform_edt(~stable_color_seed, return_indices=True)
        repaired_foreground[cap_mask] = foreground_linear[nearest[0][cap_mask], nearest[1][cap_mask]]

    capped_subject_alpha = subject_alpha.copy()
    capped_alpha = alpha.copy()
    cap_value = 0.30
    capped_subject_alpha[cap_mask] = np.minimum(capped_subject_alpha[cap_mask], cap_value)
    capped_alpha[cap_mask] = np.minimum(capped_alpha[cap_mask], cap_value)
    return repaired_foreground, capped_subject_alpha, capped_alpha, {
        "method": "thin_unstable_glass_foreground_ridge_repair",
        "applied": True,
        "pixels": int(cap_mask.sum()),
        "high_alpha_black_pixels": int(high_alpha_black_mask.sum()),
        "rgb_repaired": seed_pixels > 0,
        "stable_color_seed_pixels": seed_pixels,
        "components": components,
        "cap_value": cap_value,
        "mean_alpha_before": float(subject_alpha[cap_mask].mean()),
        "mean_alpha_after": float(capped_subject_alpha[cap_mask].mean()),
        "mean_fg_luma_before": float(fg_mean[cap_mask].mean()),
        "hole_fraction": hole_fraction,
        "soft_fraction": soft_fraction,
    }


def _solve_saturated_glass_continuous_field(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
    alpha: np.ndarray,
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_hole: np.ndarray,
    shadow_layer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover broad green-screen glass as continuous alpha/premul fields.

    Glass is not a one-dimensional solve like shadow: both opacity and
    foreground color are unknown. Once topology has proved a broad glass basin,
    however, the source still gives a continuous premultiplied signal
    ``P = C - (1 - alpha)B``. Treat alpha and P as noisy fields in that basin,
    with holes/exterior acting as zero-alpha boundary constraints. This avoids
    making every local defect a separate mask repair.
    """
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return foreground_linear, subject_alpha, alpha, {
            "method": "saturated_glass_continuous_field",
            "applied": False,
            "reason": "background is not saturated",
        }

    hole_fraction = float(subject_hole.mean())
    soft_fraction = float(soft_subject.mean())
    if hole_fraction <= 0.08 or soft_fraction <= 0.04:
        return foreground_linear, subject_alpha, alpha, {
            "method": "saturated_glass_continuous_field",
            "applied": False,
            "reason": "no broad glass basin",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    subject_region = soft_subject | opaque_subject
    hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
    soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=21)
    glass_context = (hole_density >= 0.08) | (soft_density >= 0.16)
    domain = subject_region & glass_context & ~shadow_layer & (alpha > 1e-3)
    if not domain.any():
        return foreground_linear, subject_alpha, alpha, {
            "method": "saturated_glass_continuous_field",
            "applied": False,
            "reason": "no glass field domain",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    C_lin = io.srgb_to_linear(image_srgb).astype(np.float32)
    B_lin = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0].astype(np.float32)
    pixels = image_srgb.astype(np.float32)
    source_luma = pixels.mean(axis=-1)
    source_chroma = pixels.max(axis=-1) - pixels.min(axis=-1)

    # Alpha confidence is semantic, not a target opacity. High-alpha specular
    # highlights are preserved by reducing blend, while soft glass ramps are
    # regularized against nearby hole boundaries so they can fade continuously.
    domain_f = domain.astype(np.float32)
    boundary = ndimage.binary_dilation(domain, iterations=7) & ~domain & subject_hole
    boundary_weight = boundary.astype(np.float32) * 0.55
    alpha_conf = domain_f * np.clip(alpha / 0.48, 0.22, 1.0)
    sigma_alpha = 2.4
    alpha_num = ndimage.gaussian_filter(alpha * alpha_conf, sigma=sigma_alpha)
    alpha_den = ndimage.gaussian_filter(alpha_conf + boundary_weight, sigma=sigma_alpha)
    alpha_field = alpha_num / np.maximum(alpha_den, 1e-6)
    alpha_field = np.clip(alpha_field, 0.0, 1.0).astype(np.float32)

    # Bright chromatic ridges are glass evidence, not dirty color defects. The
    # broad-field solve may smooth low-contrast gaps, but it must not average
    # away saturated cyan/white refraction lines that already exist in source.
    alpha_highlight_protect = (
        ((source_luma >= 224.0) & (source_chroma <= 70.0))
        | ((source_luma >= 190.0) & (source_chroma >= 45.0))
        | (alpha >= 0.92)
    )
    alpha_blend = 0.34 * np.clip((0.88 - alpha) / 0.88, 0.0, 1.0)
    alpha_blend[alpha_highlight_protect] *= 0.20
    alpha_write = domain & (alpha_blend > 1e-4)

    alpha_out = alpha.copy()
    subject_alpha_out = subject_alpha.copy()
    alpha_out[alpha_write] = (
        alpha[alpha_write] * (1.0 - alpha_blend[alpha_write])
        + alpha_field[alpha_write] * alpha_blend[alpha_write]
    )
    subject_alpha_out[alpha_write] = alpha_out[alpha_write]

    source_premultiplied = C_lin - (1.0 - alpha_out[..., None]) * B_lin.reshape(1, 1, 3)
    source_premultiplied = np.clip(source_premultiplied, 0.0, np.maximum(alpha_out[..., None], 1e-6))
    premultiplied = foreground_linear * alpha_out[..., None]
    premultiplied[domain] = source_premultiplied[domain]

    color_seed = (
        domain
        & (alpha_out >= 0.025)
        & ((source_chroma >= 28.0) | (source_luma >= 150.0))
    )
    color_weight = color_seed.astype(np.float32) * np.clip(alpha_out / 0.42, 0.18, 1.0)
    sigma_color = 2.2
    color_den = ndimage.gaussian_filter(color_weight, sigma=sigma_color)
    if float(color_den.max(initial=0.0)) <= 1e-6:
        return foreground_linear, subject_alpha_out, alpha_out, {
            "method": "saturated_glass_continuous_field",
            "applied": False,
            "reason": "no glass color seeds",
            "pixels": int(domain.sum()),
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    premul_field = np.empty_like(premultiplied)
    for channel in range(3):
        premul_field[..., channel] = (
            ndimage.gaussian_filter(premultiplied[..., channel] * color_weight, sigma=sigma_color)
            / np.maximum(color_den, 1e-6)
        )
    premul_field = np.clip(premul_field, 0.0, 1.0)

    straight = np.zeros_like(premultiplied)
    visible = domain & (alpha_out > 1e-3)
    straight[visible] = np.clip(
        premultiplied[visible] / np.maximum(alpha_out[visible, None], 1e-3),
        0.0,
        1.0,
    )
    straight_srgb = io.linear_to_srgb_u8(straight).astype(np.float32)
    straight_chroma = straight_srgb.max(axis=-1) - straight_srgb.min(axis=-1)
    straight_luma = straight_srgb.mean(axis=-1)
    color_highlight_protect = alpha_highlight_protect | (
        (straight_luma >= 175.0) & (straight_chroma >= 55.0)
    )
    local_luma_delta = np.abs(ndimage.gaussian_filter(straight_luma, sigma=1.0) - straight_luma)
    dirty_mid_luma_jump = local_luma_delta > 18.0
    unstable_color = (
        domain
        & (alpha_out > 0.015)
        & (color_den >= 0.025)
        & ~color_highlight_protect
        & (
            (straight_chroma < 42.0)
            | (dirty_mid_luma_jump & (straight_luma < 170.0))
        )
    )
    color_blend = 0.38 * np.clip(color_den / 0.20, 0.0, 1.0)
    color_blend[color_highlight_protect] *= 0.05
    color_write = unstable_color & (color_blend > 1e-4)
    premultiplied[color_write] = (
        premultiplied[color_write] * (1.0 - color_blend[color_write, None])
        + premul_field[color_write] * color_blend[color_write, None]
    )

    repaired = foreground_linear.copy()
    final_write = domain & (alpha_out > 1e-3)
    repaired[final_write] = np.clip(
        premultiplied[final_write] / np.maximum(alpha_out[final_write, None], 1e-3),
        0.0,
        1.0,
    )
    return repaired, subject_alpha_out, alpha_out, {
        "method": "saturated_glass_continuous_field",
        "applied": True,
        "pixels": int(domain.sum()),
        "alpha_pixels": int(alpha_write.sum()),
        "color_pixels": int(color_write.sum()),
        "color_seed_pixels": int(color_seed.sum()),
        "highlight_protected_pixels": int((domain & color_highlight_protect).sum()),
        "mean_alpha_delta": float(np.abs(alpha_out[alpha_write] - alpha[alpha_write]).mean()) if alpha_write.any() else 0.0,
        "hole_fraction": hole_fraction,
        "soft_fraction": soft_fraction,
    }


def _source_preserving_saturated_glass_foreground(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    foreground_linear: np.ndarray,
    alpha: np.ndarray,
    soft_subject: np.ndarray,
    opaque_subject: np.ndarray,
    subject_hole: np.ndarray,
    shadow_layer: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stabilize broad saturated-glass RGB from source premultiplied evidence.

    Straight foreground solving divides by alpha, so tiny screen-color
    residuals in translucent glass can become discontinuous green flecks, dark
    ridges, or false cyan/purple patches. In broad proved glass basins the
    source image already carries the continuous premultiplied signal:
    ``P = C - (1 - alpha) * B``. Work in that premultiplied domain, remove only
    the screen-direction excess, and apply a small local regularizer to the
    same source-owned field before converting back to straight foreground RGB.
    """
    bg_arr = np.asarray(bg, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    if int(bg_arr[dominant]) - int(max(bg_arr[other_channels])) < 40:
        return foreground_linear, {
            "method": "source_preserving_saturated_glass_foreground",
            "applied": False,
            "reason": "background is not saturated",
        }

    hole_fraction = float(subject_hole.mean())
    soft_fraction = float(soft_subject.mean())
    if hole_fraction <= 0.08 or soft_fraction <= 0.04:
        return foreground_linear, {
            "method": "source_preserving_saturated_glass_foreground",
            "applied": False,
            "reason": "no broad glass basin",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    subject_region = soft_subject | opaque_subject
    hole_density = ndimage.uniform_filter(subject_hole.astype(np.float32), size=31)
    soft_density = ndimage.uniform_filter(soft_subject.astype(np.float32), size=21)
    glass_context = (hole_density >= 0.08) | (soft_density >= 0.18)
    write_mask = subject_region & glass_context & ~shadow_layer & (alpha > 1e-3)
    if not write_mask.any():
        return foreground_linear, {
            "method": "source_preserving_saturated_glass_foreground",
            "applied": False,
            "reason": "no writable glass pixels",
            "hole_fraction": hole_fraction,
            "soft_fraction": soft_fraction,
        }

    pixels = image_srgb.astype(np.float32)
    source_luma = pixels.mean(axis=-1)
    source_chroma = pixels.max(axis=-1) - pixels.min(axis=-1)
    source_margin = pixels[..., dominant] - np.maximum(
        pixels[..., other_channels[0]],
        pixels[..., other_channels[1]],
    )
    bg_norm = bg_arr.astype(np.float32) / max(float(bg_arr[dominant]), 1.0)
    pixel_norm = pixels / np.maximum(pixels[..., dominant : dominant + 1], 1.0)
    norm_delta = np.maximum(
        np.abs(pixel_norm[..., other_channels[0]] - bg_norm[other_channels[0]]),
        np.abs(pixel_norm[..., other_channels[1]] - bg_norm[other_channels[1]]),
    )
    source_bg_family = (source_margin >= 4.0) & (norm_delta <= 0.95)
    seed = write_mask & source_bg_family & ((source_margin >= 18.0) | (alpha < 0.45))
    if seed.any():
        dist_to_seed = ndimage.distance_transform_edt(~seed)
        diffusion = np.exp(-((dist_to_seed / 8.0) ** 2)).astype(np.float32)
    else:
        diffusion = np.zeros(alpha.shape, dtype=np.float32)
    # The empirical ranges below are evidence gates, not target colors. They
    # say when the source pixel still points in the saturated screen direction;
    # the actual RGB remains source-derived premultiplied color.
    source_evidence = np.clip((source_margin - 1.0) / (26.0 - 1.0), 0.0, 1.0)
    source_evidence *= np.clip((1.0 - norm_delta) / (1.0 - 0.35), 0.20, 1.0)
    projection_strength = diffusion * source_evidence
    projection_strength = ndimage.gaussian_filter(projection_strength, sigma=1.3)
    projection_strength[~write_mask] = 0.0
    # Bright source highlights on glass often still point partly in the screen
    # hue direction because they are refracting the saturated background. Do
    # not let screen-removal or local continuity average those ridges into flat
    # cyan patches; darker/mid-luma source-family pixels remain repairable.
    highlight_region = subject_region & ~shadow_layer & (alpha > 0.015)
    source_highlight_structure = write_mask & (alpha > 0.015) & (
        ((source_luma >= 185.0) & (source_chroma >= 36.0))
        | ((source_luma >= 224.0) & (source_chroma <= 96.0))
        | (alpha >= 0.92)
    )
    source_highlight_core = highlight_region & (
        ((source_luma >= 185.0) & (source_chroma >= 36.0))
        | ((source_luma >= 224.0) & (source_chroma <= 96.0))
    )
    source_highlight_distance = ndimage.distance_transform_edt(~source_highlight_core)
    # Specular bands are often a bright core plus a lower-luma chromatic skirt
    # after green-screen mixing. Preserve that local structure as source-owned
    # foreground; otherwise chroma-continuity repair can flatten the skirt into
    # a blocky cyan patch beside the highlight.
    source_highlight_foreground_keep = source_highlight_core | (
        (source_highlight_distance <= 12.0)
        & highlight_region
        & (source_luma >= 86.0)
        & (source_chroma >= 34.0)
    )
    projection_strength[source_highlight_structure] = 0.0
    projection_strength = np.clip(projection_strength, 0.0, 0.92).astype(np.float32)

    C_lin = io.srgb_to_linear(image_srgb).astype(np.float32)
    B_lin = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0].astype(np.float32)
    premultiplied = foreground_linear * alpha[..., None]
    source_premultiplied = C_lin - (1.0 - alpha[..., None]) * B_lin.reshape(1, 1, 3)
    source_premultiplied = np.clip(source_premultiplied, 0.0, np.maximum(alpha[..., None], 1e-6))
    premultiplied[write_mask] = source_premultiplied[write_mask]

    project_mask = projection_strength > 1e-4
    if project_mask.any():
        values = premultiplied[project_mask]
        other_max = np.maximum(values[:, other_channels[0]], values[:, other_channels[1]])
        excess = np.maximum(values[:, dominant] - other_max, 0.0)
        strength = projection_strength[project_mask].astype(np.float32)
        subtract_scale = (excess / max(float(B_lin[dominant]), 1e-6)) * strength
        premultiplied[project_mask] = np.clip(
            values - subtract_scale[:, None] * B_lin.reshape(1, 3),
            0.0,
            1.0,
        )

    # Lightly regularize only the source/background-family glass field. This
    # protects high highlights from being globally blurred while smoothing the
    # dither-like discontinuities produced by hard masks and straight-F solves.
    smooth_mask = write_mask & (source_bg_family | (projection_strength > 0.02))
    confidence = smooth_mask.astype(np.float32) * np.clip(alpha / 0.45, 0.20, 1.0)
    confidence = ndimage.gaussian_filter(confidence, sigma=0.8)
    if float(confidence.max(initial=0.0)) > 1e-5:
        denom = np.maximum(ndimage.gaussian_filter(confidence, sigma=1.0), 1e-6)
        smoothed = np.empty_like(premultiplied)
        for channel in range(3):
            smoothed[..., channel] = (
                ndimage.gaussian_filter(premultiplied[..., channel] * confidence, sigma=1.0) / denom
            )
        blend = 0.28 * np.clip(denom / 0.35, 0.0, 1.0)
        premultiplied[smooth_mask] = (
            premultiplied[smooth_mask] * (1.0 - blend[smooth_mask, None])
            + smoothed[smooth_mask] * blend[smooth_mask, None]
        )

    chroma_continuity_mask = np.zeros(alpha.shape, dtype=bool)
    nearest_continuity_mask = np.zeros(alpha.shape, dtype=bool)
    straight_after = np.zeros_like(premultiplied)
    visible = write_mask & (alpha > 1e-3)
    straight_after[visible] = np.clip(premultiplied[visible] / np.maximum(alpha[visible, None], 1e-3), 0.0, 1.0)
    straight_srgb = io.linear_to_srgb_u8(straight_after)
    straight_f32 = straight_srgb.astype(np.float32)
    straight_chroma = straight_f32.max(axis=-1) - straight_f32.min(axis=-1)
    straight_luma = straight_f32.mean(axis=-1)
    # Projection can correctly remove screen-green but still collapse a thin
    # refractive line to neutral grey. Restore color only from nearby
    # high-chroma glass pixels in the same proved basin; this is a local
    # continuity constraint on source-owned color, not a synthetic tint.
    chroma_seed = write_mask & (alpha >= 0.05) & (straight_chroma >= 66.0) & (straight_luma >= 85.0)
    seed_weight = chroma_seed.astype(np.float32) * np.clip(alpha / 0.40, 0.25, 1.0)
    seed_density = ndimage.gaussian_filter(seed_weight, sigma=2.0)
    if float(seed_density.max(initial=0.0)) > 1e-5:
        local_straight_color = np.empty_like(premultiplied)
        denom = np.maximum(seed_density, 1e-6)
        for channel in range(3):
            local_straight_color[..., channel] = (
                ndimage.gaussian_filter(straight_after[..., channel] * seed_weight, sigma=2.0) / denom
            )
        local_straight_color = np.clip(local_straight_color, 0.0, 1.0)
        local_srgb = io.linear_to_srgb_u8(local_straight_color).astype(np.float32)
        local_chroma = local_srgb.max(axis=-1) - local_srgb.min(axis=-1)
        chroma_continuity_mask = (
            write_mask
            & ~source_highlight_foreground_keep
            & (seed_density >= 0.045)
            & (straight_chroma < 72.0)
            & ((local_chroma - straight_chroma) >= 12.0)
            # Near-white specular highlights are allowed to be low-chroma; the
            # dirty failure is mid-luma grey on glass rims and inner edges.
            & (straight_luma < 220.0)
        )
        if chroma_continuity_mask.any():
            dark_neutral = (straight_luma < 80.0) & (straight_chroma < 52.0)
            chroma_blend = 0.58 * np.clip(seed_density / 0.22, 0.0, 1.0)
            chroma_blend[dark_neutral & chroma_continuity_mask] = np.maximum(
                chroma_blend[dark_neutral & chroma_continuity_mask],
                0.82 * np.clip(seed_density[dark_neutral & chroma_continuity_mask] / 0.16, 0.0, 1.0),
            )
            repaired_straight = (
                straight_after[chroma_continuity_mask] * (1.0 - chroma_blend[chroma_continuity_mask, None])
                + local_straight_color[chroma_continuity_mask] * chroma_blend[chroma_continuity_mask, None]
            )
            premultiplied[chroma_continuity_mask] = repaired_straight * alpha[chroma_continuity_mask, None]

    if chroma_seed.any():
        _, nearest = ndimage.distance_transform_edt(~chroma_seed, return_indices=True)
        dist_to_chroma_seed = ndimage.distance_transform_edt(~chroma_seed)
        nearest_color = straight_after[nearest[0], nearest[1]]
        nearest_srgb = io.linear_to_srgb_u8(nearest_color).astype(np.float32)
        nearest_chroma = nearest_srgb.max(axis=-1) - nearest_srgb.min(axis=-1)
        continuity_region = subject_region & ~shadow_layer & (alpha > 1e-3) & (
            glass_context | (dist_to_chroma_seed <= 18.0)
        )
        nearest_continuity_mask = (
            continuity_region
            & ~source_highlight_foreground_keep
            & (dist_to_chroma_seed <= 18.0)
            & (straight_chroma < 58.0)
            & ((nearest_chroma - straight_chroma) >= 12.0)
            & (straight_luma < 220.0)
        )
        if nearest_continuity_mask.any():
            dark_neutral = (straight_luma < 82.0) & (straight_chroma < 54.0)
            nearest_blend = np.full(alpha.shape, 0.42, dtype=np.float32)
            nearest_blend[dark_neutral] = 0.86
            repaired_straight = (
                straight_after[nearest_continuity_mask] * (1.0 - nearest_blend[nearest_continuity_mask, None])
                + nearest_color[nearest_continuity_mask] * nearest_blend[nearest_continuity_mask, None]
            )
            premultiplied[nearest_continuity_mask] = repaired_straight * alpha[nearest_continuity_mask, None]

    final_write_mask = (write_mask & ~source_highlight_foreground_keep) | chroma_continuity_mask | nearest_continuity_mask
    repaired = foreground_linear.copy()
    repaired[final_write_mask] = np.clip(
        premultiplied[final_write_mask] / np.maximum(alpha[final_write_mask, None], 1e-3),
        0.0,
        1.0,
    )
    return repaired, {
        "method": "source_preserving_saturated_glass_foreground",
        "applied": True,
        "pixels": int(write_mask.sum()),
        "projection_pixels": int(project_mask.sum()),
        "smooth_pixels": int(smooth_mask.sum()),
        "seed_pixels": int(seed.sum()),
        "source_highlight_protected_pixels": int(source_highlight_structure.sum()),
        "source_highlight_foreground_kept_pixels": int(source_highlight_foreground_keep.sum()),
        "chroma_continuity_pixels": int(chroma_continuity_mask.sum()),
        "nearest_chroma_continuity_pixels": int(nearest_continuity_mask.sum()),
        "mean_projection_strength": float(projection_strength[project_mask].mean()) if project_mask.any() else 0.0,
        "hole_fraction": hole_fraction,
        "soft_fraction": soft_fraction,
    }


def _foreground_from_known_bg(C_lin: np.ndarray, B_lin: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    fg = C_lin.copy()
    solve = a > 1e-3
    fg[solve] = (C_lin[solve] - (1.0 - a[solve, None]) * B_lin.reshape(1, 3)) / np.maximum(a[solve, None], 1e-3)
    fg[~solve] = 0.0
    return np.clip(fg, 0.0, 1.0).astype(np.float32)


def _empty_result(
    image_srgb: np.ndarray,
    bg: tuple[int, int, int],
    *,
    accepted: bool,
    reason: str,
) -> SolidGraphicResult:
    h, w = image_srgb.shape[:2]
    empty_mask = np.zeros((h, w), dtype=bool)
    masks = {
        "external_background": empty_mask.copy(),
        "opaque_subject": empty_mask.copy(),
        "subject_hole": empty_mask.copy(),
        "soft_subject_layer": empty_mask.copy(),
        "shadow_layer": empty_mask.copy(),
        "unknown_fallback": np.ones((h, w), dtype=bool),
    }
    zeros = np.zeros((h, w), dtype=np.float32)
    return SolidGraphicResult(
        accepted=accepted,
        reason=reason,
        confidence=0.0,
        background_color=bg,
        alpha=zeros,
        subject_alpha=zeros,
        foreground_linear=np.zeros((h, w, 3), dtype=np.float32),
        rgba_rgb_linear=np.zeros((h, w, 3), dtype=np.float32),
        ownership_masks=masks,
        debug={},
    )


__all__ = ["SolidGraphicResult", "analyze_solid_bg_graphic"]
