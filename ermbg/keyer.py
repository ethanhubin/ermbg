"""Chromatic / luminance key matting on top of a known background color.

When B is fixed (the system's "specified background" contract), the cleanest
signal for "is this pixel background?" is the perceptual distance from each
pixel to B. Two flavors:

  ``chromatic_key_alpha``  — uses full OKLab distance. Best for saturated B
      (green/cyan/magenta), and still useful as auxiliary "not equal to B"
      evidence for pale subjects on white/black backgrounds.

  ``luminance_key_alpha``  — uses |ΔL| only. The right tool for white or
      black backgrounds as a primary key: dark subjects on a white screen
      separate cleanly by lightness alone. It is paired with full-color known-B
      repair for pale colored interiors whose lightness is close to B.

  ``key_alpha(..., mode=...)`` — dispatcher.

  ``merge_alpha_components`` — keep the matting net's α everywhere it's confident,
      and add back small connected components from the key α that the matting
      net missed (e.g. an isolated red dot when the model focused on a bigger
      star). Does *not* override the matting α on the main subject.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import io
from .colorspace import oklab_distance, srgb_to_oklab


@dataclass
class KeyerThresholds:
    """OKLab ΔE thresholds for the soft key.

    Pixels with d <= ``bg_max`` are full background (α=0).
    Pixels with d >= ``fg_min`` are full foreground (α=1).
    In between, α ramps linearly. Defaults are tuned for a saturated screen
    (green/cyan/magenta), but are also used as auxiliary full-color evidence
    for known white/black background repair.
    """

    bg_max: float = 6.0
    fg_min: float = 22.0


def chromatic_key_alpha(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Soft chromatic-key α in [0, 1] from full OKLab distance to B.

    Returns float32 H×W. Higher = more foreground.
    """
    t = thresholds or KeyerThresholds()
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    d = oklab_distance(lab, bg_lab).astype(np.float32)
    alpha = np.clip((d - t.bg_max) / max(t.fg_min - t.bg_max, 1e-6), 0.0, 1.0)
    return alpha.astype(np.float32)


def luminance_key_alpha(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Soft luminance-key α from OKLab L-channel distance to B.

    Designed for white / black backgrounds. Pixels whose lightness matches B's
    lightness are background (α=0); pixels far in lightness are foreground.
    Pure-color subjects on a white screen (red logo, dark cartoon) separate
    cleanly here even though chromatic distance also works — but a *bright*
    saturated subject on white is hard for both, by construction.
    """
    t = thresholds or KeyerThresholds()
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    # OKLab L is in [0, 1]; rescale to a ΔE-like 0..100 range for threshold parity.
    d = np.abs(lab[..., 0] - bg_lab[0]).astype(np.float32) * 100.0
    alpha = np.clip((d - t.bg_max) / max(t.fg_min - t.bg_max, 1e-6), 0.0, 1.0)
    return alpha.astype(np.float32)


def key_alpha(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    mode: str = "chromatic",
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Dispatch to chromatic or luminance keyer."""
    if mode == "chromatic":
        return chromatic_key_alpha(image_srgb, background_color, thresholds)
    if mode == "luminance":
        return luminance_key_alpha(image_srgb, background_color, thresholds)
    raise ValueError(f"Unknown keyer mode: {mode!r}")


def gate_alpha_by_keyer(
    matting_alpha: np.ndarray,
    key_alpha: np.ndarray,
    bg_confidence_threshold: float = 0.08,
    fg_protect_threshold: float = 0.85,
) -> tuple[np.ndarray, dict]:
    """Cap matting α by key α where the keyer is confident the pixel is bg.

    Motivation: BiRefNet-matting can over-feather hard-edged graphics on a
    pure-color background, producing a wide low-α halo (α∈(0, 0.3]) on what
    should be cleanly transparent pixels. When the keyer — which has direct
    access to the known B — strongly disagrees ("this pixel's color IS the
    background"), it's the keyer that's right.

    Rules:
      - For pixels where key α < ``bg_confidence_threshold``, cap matting α by
        key α (i.e. force them down toward 0).
      - For pixels where matting α >= ``fg_protect_threshold``, leave alone:
        these are confident foreground, e.g. hair against bg where matting's
        soft α is the *correct* signal even if the keyer would call those
        pixels "bg colored".
      - Pixels in between are untouched. The merge step (recall booster)
        runs separately and additively.

    The default ``bg_confidence_threshold = 0.08`` is intentionally tight —
    we only want to remove obvious halo, not erode legitimate hairy / fuzzy
    edges where the keyer's hard ramp is overconfident.

    Returns:
      gated_alpha: float32 H×W
      info: dict with `pixels_gated`, `mean_drop`
    """
    m = matting_alpha.astype(np.float32)
    k = key_alpha.astype(np.float32)

    bg_confident = k < bg_confidence_threshold
    fg_protected = m >= fg_protect_threshold
    gate_zone = bg_confident & ~fg_protected

    gated = m.copy()
    if gate_zone.any():
        gated[gate_zone] = np.minimum(m[gate_zone], k[gate_zone])

    return gated, {
        "pixels_gated": int(gate_zone.sum()),
        "mean_drop": float((m[gate_zone] - gated[gate_zone]).mean()) if gate_zone.any() else 0.0,
    }


def _flood_exterior(mask: np.ndarray) -> np.ndarray:
    """Return pixels connected to the image border inside ``mask``."""
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    work = mask.astype(np.uint8).copy()

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


def repair_alpha_with_subject_support(
    matting_alpha: np.ndarray,
    key_alpha: np.ndarray,
    subject_support: np.ndarray,
    *,
    key_fg_threshold: float = 0.75,
    matting_low_threshold: float = 0.65,
    support_threshold: float = 0.5,
    fg_anchor_threshold: float = 0.85,
    exterior_margin_px: int = 2,
    min_component_area_ratio: float = 0.00002,
    feather_radius: int = 1,
    target_alpha_floor: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Repair low-α regions only where an external subject mask owns them.

    This is intentionally *not* a generic hole fill. The supplied
    ``subject_support`` is a separate ownership signal from a stronger or more
    targeted segmenter. The keyer only supplies known-background evidence, and
    topology only keeps the repair away from the external contour.

    Pixels are eligible when:
      - the external subject mask includes them,
      - the keyer is confident foreground,
      - the matting α is too low,
      - they are not in the support's exterior fringe,
      - the connected component is anchored to confident matting foreground.

    Returns the repaired α and a small debug dict.
    """
    m = matting_alpha.astype(np.float32)
    k = key_alpha.astype(np.float32)
    support_f = subject_support.astype(np.float32)
    if m.shape != k.shape or m.shape != support_f.shape:
        raise ValueError("matting_alpha, key_alpha, and subject_support must have the same HxW shape")

    support = support_f >= support_threshold
    if not support.any():
        return m.copy(), {
            "used": True,
            "accepted_components": 0,
            "accepted_pixels": 0,
            "rejected_components": 0,
            "component_areas": [],
        }

    exterior = _flood_exterior(~support)
    if exterior_margin_px > 0:
        # Distance is measured in pixels from exterior background. Candidate
        # pixels close to exterior are likely outer antialiasing, not owned
        # interior content.
        dist_to_exterior = cv2.distanceTransform((~exterior).astype(np.uint8), cv2.DIST_L2, 3)
        away_from_exterior = dist_to_exterior >= float(exterior_margin_px)
    else:
        away_from_exterior = ~exterior

    candidate = support & away_from_exterior & (k >= key_fg_threshold) & (m <= matting_low_threshold)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)

    h, w = m.shape
    min_area = max(1.0, min_component_area_ratio * float(h * w))
    confident_fg = (m >= fg_anchor_threshold) & support
    accepted = np.zeros_like(candidate)
    accepted_areas: list[int] = []
    rejected = 0

    anchor_kernel = np.ones((3, 3), np.uint8)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            rejected += 1
            continue
        comp = labels == i
        comp_touch = cv2.dilate(comp.astype(np.uint8), anchor_kernel, iterations=2).astype(bool)
        if not (comp_touch & confident_fg).any():
            rejected += 1
            continue
        accepted |= comp
        accepted_areas.append(area)

    repaired = m.copy()
    if accepted.any():
        target = np.maximum(m, np.minimum(k, support_f))
        if target_alpha_floor is not None:
            target = np.maximum(target, float(target_alpha_floor))
        repaired[accepted] = target[accepted]

        if feather_radius > 0:
            ksize = 2 * feather_radius + 1
            soft = cv2.GaussianBlur(
                accepted.astype(np.float32),
                (ksize, ksize),
                sigmaX=float(feather_radius),
            )
            feather_zone = (soft > 0.0) & support & away_from_exterior
            blended = m + soft * np.maximum(target - m, 0.0)
            repaired[feather_zone] = np.maximum(repaired[feather_zone], blended[feather_zone])

    return np.clip(repaired, 0.0, 1.0).astype(np.float32), {
        "used": True,
        "accepted_components": len(accepted_areas),
        "accepted_pixels": int(accepted.sum()),
        "rejected_components": rejected,
        "component_areas": accepted_areas,
    }


def repair_alpha_with_known_bg_key(
    matting_alpha: np.ndarray,
    full_color_key_alpha: np.ndarray,
    *,
    key_fg_threshold: float = 0.45,
    matting_low_threshold: float = 0.65,
    support_threshold: float = 0.35,
    fg_anchor_threshold: float = 0.85,
    exterior_margin_px: int = 3,
    min_component_area_ratio: float = 0.00002,
    feather_radius: int = 1,
    target_alpha_floor: float = 0.90,
) -> tuple[np.ndarray, dict]:
    """Repair low-α holes using only known-background color evidence.

    This is the non-semantic sibling of ``repair_alpha_with_subject_support``.
    For white/black graphic assets, a luminance-only key can miss pale colored
    subject regions whose lightness is close to the background. The full OKLab
    distance to the known background still carries the missing evidence:
    "this pixel is not actually B".

    The key output is *not* used as final alpha. It is only used as a rough
    support map, then the same topology guards apply:
      - avoid the exterior fringe,
      - require keyer foreground confidence,
      - require the candidate to touch confident matting foreground,
      - only raise alpha, never lower it.
    """
    repaired, info = repair_alpha_with_subject_support(
        matting_alpha,
        full_color_key_alpha,
        full_color_key_alpha,
        key_fg_threshold=key_fg_threshold,
        matting_low_threshold=matting_low_threshold,
        support_threshold=support_threshold,
        fg_anchor_threshold=fg_anchor_threshold,
        exterior_margin_px=exterior_margin_px,
        min_component_area_ratio=min_component_area_ratio,
        feather_radius=feather_radius,
        target_alpha_floor=target_alpha_floor,
    )
    info["source"] = "known_bg_full_color_key"
    info["key_fg_threshold"] = key_fg_threshold
    info["support_threshold"] = support_threshold
    info["exterior_margin_px"] = exterior_margin_px
    info["target_alpha_floor"] = target_alpha_floor
    return repaired, info


def repair_opaque_interior_with_known_bg_key(
    matting_alpha: np.ndarray,
    full_color_key_alpha: np.ndarray,
    *,
    key_fg_threshold: float = 0.92,
    support_threshold: float = 0.50,
    min_alpha_threshold: float = 0.78,
    target_alpha_floor: float = 0.98,
    fg_anchor_threshold: float = 0.90,
    exterior_margin_px: int = 3,
    min_component_area_ratio: float = 0.0001,
    max_repair_area_ratio: float = 0.70,
    shadow_protect_mask: np.ndarray | None = None,
    material_protect_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Snap under-opaque known-B subject interiors toward full alpha.

    This is deliberately separate from low-alpha hole repair. On solid known
    backgrounds, a matting net can keep a hard UI/product surface at α≈0.75–0.9
    even when color evidence says those pixels are fully foreground. The
    general signal is not a filename or icon size; it is a key-supported
    component with an interior distance from exterior background and an anchor
    of high-confidence matte foreground.

    Guards:
      - only inside the keyer's foreground support;
      - only pixels away from the support exterior, preserving antialiasing;
      - require a nearby high-alpha anchor within the same support component;
      - never repair supplied shadow/material protection masks.
    """
    m = matting_alpha.astype(np.float32)
    k = full_color_key_alpha.astype(np.float32)
    if m.shape != k.shape:
        raise ValueError("matting_alpha and full_color_key_alpha must have the same HxW shape")
    if shadow_protect_mask is not None and shadow_protect_mask.shape != m.shape:
        raise ValueError("shadow_protect_mask must have the same HxW shape")
    if material_protect_mask is not None and material_protect_mask.shape != m.shape:
        raise ValueError("material_protect_mask must have the same HxW shape")

    support = k >= support_threshold
    if not support.any():
        return m.copy(), {
            "used": True,
            "accepted_components": 0,
            "accepted_pixels": 0,
            "rejected_components": 0,
            "component_areas": [],
        }

    exterior = _flood_exterior(~support)
    if exterior_margin_px > 0:
        dist_to_exterior = cv2.distanceTransform((~exterior).astype(np.uint8), cv2.DIST_L2, 3)
        interior = dist_to_exterior >= float(exterior_margin_px)
    else:
        interior = ~exterior

    protect = np.zeros_like(support, dtype=bool)
    if shadow_protect_mask is not None:
        protect |= np.asarray(shadow_protect_mask, dtype=bool)
    if material_protect_mask is not None:
        protect |= np.asarray(material_protect_mask, dtype=bool)

    candidate = (
        support
        & interior
        & ~protect
        & (k >= key_fg_threshold)
        & (m >= min_alpha_threshold)
        & (m < target_alpha_floor)
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    h, w = m.shape
    img_area = float(h * w)
    min_area = max(1.0, min_component_area_ratio * img_area)
    max_repair_area = max(min_area, max_repair_area_ratio * img_area)
    accepted = np.zeros_like(candidate)
    accepted_areas: list[int] = []
    rejected = 0

    for i in range(1, n_labels):
        support_comp = labels == i
        comp_candidate = candidate & support_comp
        area = int(comp_candidate.sum())
        if area <= 0:
            continue
        if area < min_area or area > max_repair_area:
            rejected += 1
            continue
        if not ((m >= fg_anchor_threshold) & support_comp & interior & ~protect).any():
            rejected += 1
            continue
        accepted |= comp_candidate
        accepted_areas.append(area)

    repaired = m.copy()
    if accepted.any():
        target = np.maximum(k, float(target_alpha_floor))
        repaired[accepted] = np.maximum(repaired[accepted], target[accepted])

    return np.clip(repaired, 0.0, 1.0).astype(np.float32), {
        "used": True,
        "source": "known_bg_opaque_interior",
        "accepted_components": len(accepted_areas),
        "accepted_pixels": int(accepted.sum()),
        "rejected_components": rejected,
        "component_areas": accepted_areas,
        "key_fg_threshold": key_fg_threshold,
        "support_threshold": support_threshold,
        "min_alpha_threshold": min_alpha_threshold,
        "target_alpha_floor": target_alpha_floor,
        "exterior_margin_px": exterior_margin_px,
    }


def resolve_hard_edge_alpha_with_known_bg_key(
    matting_alpha: np.ndarray,
    full_color_key_alpha: np.ndarray,
    *,
    image_srgb: np.ndarray | None = None,
    background_color: tuple[int, int, int] | np.ndarray | None = None,
    key_bg_threshold: float = 0.02,
    key_fg_threshold: float = 0.98,
    shadow_protect_mask: np.ndarray | None = None,
    material_protect_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Resolve clean hard-edge alpha from known-background key topology.

    For a solid-screen UI/product/logo asset, center pixels are not ambiguous:
    only the exterior contour should be soft. The generic preconditions are
    topological, not pixel-size tuned: the key must have exterior background,
    a non-empty opaque core, a transition band smaller than that core, and every
    transition pixel must be shallower than the core's median inward distance.
    Broad hair/fur/smoke transitions therefore stay with the matting net.
    """
    m = matting_alpha.astype(np.float32)
    k = full_color_key_alpha.astype(np.float32)
    if m.shape != k.shape:
        raise ValueError("matting_alpha and full_color_key_alpha must have the same HxW shape")
    if image_srgb is not None and image_srgb.shape[:2] != m.shape:
        raise ValueError("image_srgb must share image HxW")
    if shadow_protect_mask is not None and shadow_protect_mask.shape != m.shape:
        raise ValueError("shadow_protect_mask must have the same HxW shape")
    if material_protect_mask is not None and material_protect_mask.shape != m.shape:
        raise ValueError("material_protect_mask must have the same HxW shape")

    h, w = m.shape
    area = float(h * w)
    support = k > float(key_bg_threshold)
    support_pixels = int(support.sum())
    support_ratio = support_pixels / max(area, 1.0)
    if support_pixels == 0:
        return m.copy(), {
            "used": True,
            "applied": False,
            "raised_pixels": 0,
            "lowered_pixels": 0,
            "reason": "no key support",
            "support_ratio": support_ratio,
        }

    exterior_bg = _flood_exterior(~support)
    exterior_pixels = int(exterior_bg.sum())
    exterior_ratio = float(exterior_pixels) / max(area, 1.0)
    if exterior_pixels == 0:
        return m.copy(), {
            "used": True,
            "applied": False,
            "raised_pixels": 0,
            "lowered_pixels": 0,
            "reason": "no exterior background",
            "support_ratio": support_ratio,
            "exterior_bg_ratio": exterior_ratio,
        }

    dist_to_exterior = cv2.distanceTransform((~exterior_bg).astype(np.uint8), cv2.DIST_L2, 3)
    core = k >= float(key_fg_threshold)
    core_pixels = int(core.sum())
    if core_pixels == 0:
        return m.copy(), {
            "used": True,
            "applied": False,
            "raised_pixels": 0,
            "lowered_pixels": 0,
            "reason": "no opaque key core",
            "support_ratio": support_ratio,
            "exterior_bg_ratio": exterior_ratio,
        }

    transition = support & ~core
    transition_pixels = int(transition.sum())
    transition_fraction = float(transition.sum()) / max(float(support_pixels), 1.0)
    transition_width_px = float(dist_to_exterior[transition].max()) if transition.any() else 0.0
    core_median_depth_px = float(np.median(dist_to_exterior[core]))
    transition_is_outer_band = transition_pixels < core_pixels and transition_width_px < core_median_depth_px
    if transition_pixels and not transition_is_outer_band:
        return m.copy(), {
            "used": True,
            "applied": False,
            "raised_pixels": 0,
            "lowered_pixels": 0,
            "reason": "broad key transition",
            "support_ratio": support_ratio,
            "exterior_bg_ratio": exterior_ratio,
            "transition_fraction": transition_fraction,
            "transition_width_px": transition_width_px,
            "core_median_depth_px": core_median_depth_px,
            "core_pixels": core_pixels,
            "transition_pixels": transition_pixels,
        }

    protect = np.zeros_like(support, dtype=bool)
    if shadow_protect_mask is not None:
        protect |= np.asarray(shadow_protect_mask, dtype=bool)
    if material_protect_mask is not None:
        protect |= np.asarray(material_protect_mask, dtype=bool)

    target = np.where(transition, k, np.maximum(m, k))
    spill_limit = np.ones_like(m, dtype=np.float32)
    spill_limited = np.zeros_like(support, dtype=bool)
    if image_srgb is not None and background_color is not None:
        spill_limit = _dominant_screen_spill_alpha_limit(
            image_srgb,
            background_color,
            fallback_shape=m.shape,
        )
        exterior_band = support & (dist_to_exterior < core_median_depth_px)
        spill_limited = exterior_band & (spill_limit < target - 1.0 / 255.0)
        target[spill_limited] = spill_limit[spill_limited]

    change_mask = support & ~protect & (np.abs(target - m) > 1.0 / 255.0)
    raise_mask = change_mask & (target > m)
    lower_mask = change_mask & (target < m)
    out = m.copy()
    out[change_mask] = target[change_mask]

    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "used": True,
        "applied": bool(change_mask.any()),
        "raised_pixels": int(raise_mask.sum()),
        "lowered_pixels": int(lower_mask.sum()),
        "spill_limited_pixels": int(spill_limited.sum()),
        "transition_pixels": transition_pixels,
        "core_pixels": core_pixels,
        "support_ratio": support_ratio,
        "exterior_bg_ratio": exterior_ratio,
        "transition_fraction": transition_fraction,
        "transition_width_px": transition_width_px,
        "core_median_depth_px": core_median_depth_px,
        "key_bg_threshold": key_bg_threshold,
        "key_fg_threshold": key_fg_threshold,
    }


def _dominant_screen_spill_alpha_limit(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    fallback_shape: tuple[int, int],
) -> np.ndarray:
    """Alpha ceiling that makes recovered F no longer screen-channel dominated.

    For a saturated known screen, an antialiased foreground edge should satisfy
    ``C = alpha*F + (1-alpha)*B``. If the observed pixel's dominant screen
    channel is still too high, solve that equation for the largest alpha that
    would make the recovered foreground's screen channel no greater than at
    least one non-screen channel. Pixels without a saturated dominant screen
    receive a neutral limit of 1.
    """
    bg = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    B = io.srgb_to_linear(bg)[0, 0].astype(np.float32)
    order = np.argsort(B)[::-1]
    d = int(order[0])
    if B[d] <= B[int(order[1])] + 1e-6:
        return np.ones(fallback_shape, dtype=np.float32)

    C = io.srgb_to_linear(image_srgb).astype(np.float32)
    caps: list[np.ndarray] = []
    for c in range(3):
        if c == d:
            continue
        denom = float(B[d] - B[c])
        if denom <= 1e-6:
            continue
        cap = (C[..., c] - B[c] - C[..., d] + B[d]) / denom
        caps.append(cap.astype(np.float32))
    if not caps:
        return np.ones(fallback_shape, dtype=np.float32)
    limit = np.maximum.reduce(caps)
    return np.clip(limit, 0.0, 1.0).astype(np.float32)


def repair_hard_edge_alpha(
    image_srgb: np.ndarray,
    matting_alpha: np.ndarray,
    key_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    key_fg_threshold: float = 0.90,
    matting_low_threshold: float = 0.85,
    lightness_contrast_min: float = 55.0,
    fg_anchor_threshold: float = 0.85,
    anchor_dilate_px: int = 2,
    min_component_area_ratio: float = 0.000005,
    max_component_area_ratio: float = 0.02,
    target_alpha_floor: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Restore high-contrast hard-edge strokes that the matting net softened.

    This is the first local ``hard_edge`` policy. It is deliberately narrower
    than a generic ``alpha = max(alpha, key)`` rule: candidates must be strongly
    separated from the known background in lightness, be foreground according to
    the keyer, currently have under-estimated alpha, stay small enough to look
    like an edge/stroke component, and touch confident foreground nearby.
    """
    m = matting_alpha.astype(np.float32)
    k = key_alpha.astype(np.float32)
    if image_srgb.shape[:2] != m.shape or k.shape != m.shape:
        raise ValueError("image_srgb, matting_alpha, and key_alpha must share HxW")

    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    lightness_contrast = np.abs(lab[..., 0] - bg_lab[0]).astype(np.float32) * 100.0

    candidate = (
        (k >= key_fg_threshold)
        & (m < matting_low_threshold)
        & (lightness_contrast >= lightness_contrast_min)
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)

    h, w = m.shape
    img_area = float(h * w)
    min_area = max(1.0, min_component_area_ratio * img_area)
    max_area = max(min_area, max_component_area_ratio * img_area)
    confident_fg = m >= fg_anchor_threshold
    anchor_kernel = np.ones((3, 3), np.uint8)

    accepted = np.zeros_like(candidate)
    accepted_areas: list[int] = []
    rejected = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            rejected += 1
            continue
        comp = labels == i
        comp_touch = cv2.dilate(
            comp.astype(np.uint8),
            anchor_kernel,
            iterations=max(1, int(anchor_dilate_px)),
        ).astype(bool)
        if not (comp_touch & confident_fg).any():
            rejected += 1
            continue
        accepted |= comp
        accepted_areas.append(area)

    repaired = m.copy()
    if accepted.any():
        target = np.maximum(k, float(target_alpha_floor))
        repaired[accepted] = np.maximum(repaired[accepted], target[accepted])

    return np.clip(repaired, 0.0, 1.0).astype(np.float32), {
        "used": True,
        "accepted_components": len(accepted_areas),
        "accepted_pixels": int(accepted.sum()),
        "rejected_components": rejected,
        "component_areas": accepted_areas,
        "key_fg_threshold": key_fg_threshold,
        "matting_low_threshold": matting_low_threshold,
        "lightness_contrast_min": lightness_contrast_min,
        "target_alpha_floor": target_alpha_floor,
    }


def merge_alpha_components(
    matting_alpha: np.ndarray,
    chromatic_alpha: np.ndarray,
    min_component_area_ratio: float = 0.0005,
    max_component_area_ratio: float = 0.5,
    matting_present_coverage: float = 0.30,
    fg_threshold: float = 0.5,
    feather_radius: int = 1,
) -> tuple[np.ndarray, dict]:
    """Patch missing subjects from key α back into matting α.

    Workflow:
      1. Binarize chromatic_alpha at ``fg_threshold`` and find connected components.
      2. For each chromatic component, decide whether matting_alpha already
         represents it. We use *coverage*: what fraction of the component
         pixels does matting_alpha consider foreground (α ≥ ``fg_threshold``)?
         Below ``matting_present_coverage`` (default 30%) we treat matting as
         having missed it and patch it in.
      3. Patched components keep the chromatic α via ``maximum``, so we never
         decrease an existing α. The chromatic α is feathered by a small
         Gaussian (``feather_radius``) before merging — the chromatic key
         produces a hard 0→1 step (typical soft-edge fraction <1%), and
         dropping that into the matting α as-is creates visible aliasing on
         the patched component vs the AA edges everywhere else. A 1px
         Gaussian gives back roughly the AA the matting net would have
         produced if it had seen the subject.
      4. The matting α elsewhere is left unchanged.

    Returns:
      merged_alpha: float32 H×W in [0, 1]
      info: dict with keys ``patched_components``, ``component_areas``
    """
    h, w = matting_alpha.shape
    img_area = float(h * w)
    min_area = min_component_area_ratio * img_area
    max_area = max_component_area_ratio * img_area

    chrom_bin = (chromatic_alpha >= fg_threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(chrom_bin, connectivity=8)

    merged = matting_alpha.astype(np.float32).copy()
    patched: list[int] = []
    areas: list[int] = []

    # Pre-soften the chromatic α once. Cheap (single Gaussian) and only
    # consumed inside the per-component path.
    if feather_radius > 0:
        ksize = 2 * feather_radius + 1
        chrom_soft = cv2.GaussianBlur(
            chromatic_alpha.astype(np.float32),
            (ksize, ksize),
            sigmaX=float(feather_radius),
        )
    else:
        chrom_soft = chromatic_alpha.astype(np.float32)

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp_mask = labels == i
        # Fraction of the component that matting also considers foreground.
        coverage = float((matting_alpha[comp_mask] >= fg_threshold).mean())
        if coverage >= matting_present_coverage:
            continue
        # Patch the feathered chromatic α into the merged map. Use the
        # component's bounding box so we also pick up the feather "bleed"
        # one pixel outside the strict component mask.
        x, y, ww, hh = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        # Expand bbox by feather radius and clamp
        pad = feather_radius + 1
        y0, y1 = max(0, y - pad), min(h, y + hh + pad)
        x0, x1 = max(0, x - pad), min(w, x + ww + pad)
        # Restrict the feathered region to a dilation of the component
        # so we don't pull α from neighboring blobs.
        comp_dil = cv2.dilate(
            comp_mask[y0:y1, x0:x1].astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=feather_radius + 1,
        ).astype(bool)
        sub = chrom_soft[y0:y1, x0:x1].copy()
        sub[~comp_dil] = 0.0
        merged[y0:y1, x0:x1] = np.maximum(merged[y0:y1, x0:x1], sub)
        patched.append(i)
        areas.append(area)

    return merged, {"patched_components": len(patched), "component_areas": areas}


__all__ = [
    "KeyerThresholds",
    "chromatic_key_alpha",
    "luminance_key_alpha",
    "key_alpha",
    "gate_alpha_by_keyer",
    "repair_hard_edge_alpha",
    "repair_opaque_interior_with_known_bg_key",
    "resolve_hard_edge_alpha_with_known_bg_key",
    "repair_alpha_with_known_bg_key",
    "repair_alpha_with_subject_support",
    "merge_alpha_components",
]
