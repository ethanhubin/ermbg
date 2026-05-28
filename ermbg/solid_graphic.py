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
            "internal_hole_shadow_pixels": int(internal_hole_shadow.sum()),
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


def _known_background_projection_strength(
    bg: tuple[int, int, int],
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
    subject_mask: np.ndarray,
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
