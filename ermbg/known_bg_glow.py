"""Known-background solver for simple additive/soft glow icons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class KnownBgGlowAnalysis:
    accepted: bool
    reason: str
    mode: str
    background_color: tuple[int, int, int]
    target_color: tuple[int, int, int]
    support_pixels: int
    support_fraction: float
    largest_component_fraction: float
    soft_fraction: float
    outer_fraction: float
    strong_fraction: float
    residual_median: float
    residual_p90: float
    target_distance: float
    alpha_mean: float
    outer_roughness_p90: float = 0.0
    falloff_correlation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "mode": self.mode,
            "background_color": list(self.background_color),
            "target_color": list(self.target_color),
            "support_pixels": self.support_pixels,
            "support_fraction": self.support_fraction,
            "largest_component_fraction": self.largest_component_fraction,
            "soft_fraction": self.soft_fraction,
            "outer_fraction": self.outer_fraction,
            "strong_fraction": self.strong_fraction,
            "residual_median": self.residual_median,
            "residual_p90": self.residual_p90,
            "target_distance": self.target_distance,
            "alpha_mean": self.alpha_mean,
            "outer_roughness_p90": self.outer_roughness_p90,
            "falloff_correlation": self.falloff_correlation,
        }


@dataclass(frozen=True)
class KnownBgGlowResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    debug: dict[str, Any]


def _estimate_target_color(image_srgb: np.ndarray, background_color: tuple[int, int, int]) -> np.ndarray:
    rgb = image_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32)
    delta = rgb - bg.reshape(1, 1, 3)
    dist = np.linalg.norm(delta, axis=2)
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    candidate = dist >= max(18.0, float(np.percentile(dist, 90.0)))
    if not candidate.any():
        candidate = dist >= float(np.percentile(dist, 98.0))
    if not candidate.any():
        return np.asarray([255.0, 255.0, 255.0], dtype=np.float32)
    threshold = float(np.percentile(luma[candidate], 90.0))
    bright = candidate & (luma >= threshold)
    pixels = rgb[bright if bright.any() else candidate]
    return np.clip(np.median(pixels, axis=0), 0.0, 255.0).astype(np.float32)


def _solve_alpha(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    target_color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = image_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32)
    direction = target_color.astype(np.float32) - bg
    denom = float(np.dot(direction, direction))
    if denom <= 1e-6:
        empty = np.zeros(rgb.shape[:2], dtype=np.float32)
        return empty, np.full(rgb.shape[:2], np.inf, dtype=np.float32)
    alpha = np.clip(np.sum((rgb - bg.reshape(1, 1, 3)) * direction.reshape(1, 1, 3), axis=2) / denom, 0.0, 1.0)
    recon = bg.reshape(1, 1, 3) + alpha[..., None] * direction.reshape(1, 1, 3)
    residual = np.linalg.norm(rgb - recon, axis=2).astype(np.float32)
    return alpha.astype(np.float32), residual


def _solve_adaptive_ray(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = image_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32)
    delta = rgb - bg.reshape(1, 1, 3)
    distance = np.linalg.norm(delta, axis=2).astype(np.float32)
    scale = np.full(distance.shape, np.inf, dtype=np.float32)
    for channel in range(3):
        d = delta[..., channel]
        candidate = np.full(distance.shape, np.inf, dtype=np.float32)
        increasing = d > 1e-3
        decreasing = d < -1e-3
        candidate[increasing] = (255.0 - bg[channel]) / d[increasing]
        candidate[decreasing] = (0.0 - bg[channel]) / d[decreasing]
        scale = np.minimum(scale, candidate)
    scale = np.where(np.isfinite(scale) & (scale >= 1.0), scale, 1.0).astype(np.float32)
    alpha = np.clip(1.0 / scale, 0.0, 1.0)
    foreground = np.clip(bg.reshape(1, 1, 3) + delta * scale[..., None], 0.0, 255.0).astype(np.uint8)
    endpoint_distance = np.linalg.norm(foreground.astype(np.float32) - bg.reshape(1, 1, 3), axis=2)
    # Tiny screen-color variations can be extended to arbitrary RGB-cube
    # endpoints. Require the inferred endpoint to be materially away from the
    # screen color so low-alpha background noise does not become colored glow.
    alpha = np.where((distance > 6.0) & (endpoint_distance >= 80.0), alpha, 0.0).astype(np.float32)
    return alpha, foreground, distance


def _adaptive_ray_metrics(alpha: np.ndarray) -> dict[str, Any]:
    support = alpha >= 0.02
    support_pixels = int(support.sum())
    if support_pixels == 0:
        return {
            "support_pixels": 0,
            "support_fraction": 0.0,
            "largest_component_fraction": 0.0,
            "soft_fraction": 0.0,
            "outer_fraction": 0.0,
            "strong_fraction": 0.0,
            "alpha_mean": 0.0,
            "outer_roughness_p90": 0.0,
            "falloff_correlation": 0.0,
            "component_count": 0,
            "main_component": np.zeros(alpha.shape, dtype=bool),
        }
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        main_component = np.zeros(alpha.shape, dtype=bool)
        largest_area = 0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_label = 1 + int(np.argmax(areas))
        main_component = labels == main_label
        largest_area = int(areas.max())
    main_alpha = np.where(main_component, alpha, 0.0).astype(np.float32)
    main_pixels = int(main_component.sum())
    mid = main_component & (main_alpha >= 0.03) & (main_alpha <= 0.75)
    outer = main_component & (main_alpha >= 0.02) & (main_alpha <= 0.35)
    strong = main_component & (main_alpha >= 0.35)
    blur = cv2.GaussianBlur(main_alpha, (0, 0), 3)
    roughness = np.abs(main_alpha - blur)
    if outer.any():
        outer_roughness_p90 = float(np.percentile(roughness[outer], 90.0))
    else:
        outer_roughness_p90 = 0.0
    if int(mid.sum()) >= 100:
        distance_to_exterior = cv2.distanceTransform(main_component.astype(np.uint8), cv2.DIST_L2, 3)
        distance_values = distance_to_exterior[mid].ravel()
        alpha_values = main_alpha[mid].ravel()
        if float(distance_values.std()) > 1e-6 and float(alpha_values.std()) > 1e-6:
            corr = np.corrcoef(distance_values, alpha_values)[0, 1]
            falloff_correlation = float(corr) if np.isfinite(corr) else 0.0
        else:
            falloff_correlation = 0.0
    else:
        falloff_correlation = 0.0
    return {
        "support_pixels": support_pixels,
        "support_fraction": float(support_pixels / max(1, alpha.size)),
        "largest_component_fraction": float(largest_area / max(1, support_pixels)),
        "soft_fraction": float(mid.sum() / max(1, main_pixels)),
        "outer_fraction": float(outer.sum() / max(1, main_pixels)),
        "strong_fraction": float(strong.sum() / max(1, main_pixels)),
        "alpha_mean": float(alpha.mean()),
        "outer_roughness_p90": outer_roughness_p90,
        "falloff_correlation": falloff_correlation,
        "component_count": int(num_labels - 1),
        "main_component": main_component,
    }


def _repair_additive_adaptive_foreground(
    image_srgb: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
    keep: np.ndarray,
    background_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    bg = np.asarray(background_color, dtype=np.float32)
    target = np.asarray(target_color, dtype=np.float32)
    target_delta = target - bg
    max_target_delta = float(np.max(np.abs(target_delta)))
    # This repair is only for additive glow fields: every target channel moves
    # away from the known screen in the same positive direction. Blue/purple
    # adaptive glows over green screens need their real negative green channel,
    # so they keep the unconstrained adaptive endpoint.
    if max_target_delta < 50.0 or bool(np.any(target_delta < -4.0)):
        return foreground, alpha, 0

    fg_float = foreground.astype(np.float32)
    fg_min = np.min(fg_float, axis=2)
    fg_max = np.max(fg_float, axis=2)
    fg_chroma = fg_max - fg_min
    bright_highlight = keep & (alpha >= 0.35) & (fg_max >= 225.0) & (fg_min >= 150.0) & (fg_chroma <= 96.0)
    endpoint = foreground.astype(np.int16)
    screen_hue_endpoint = (
        (endpoint[..., 1] > endpoint[..., 0] + 20)
        | (endpoint[..., 2] > endpoint[..., 0] + 20)
    )
    repair = keep & (alpha > 0.0) & ~bright_highlight & screen_hue_endpoint
    if not bool(repair.any()):
        return foreground, alpha, 0

    repaired_float = foreground.astype(np.float32).copy()
    # For additive glow, screen-hued adaptive endpoints are not real
    # foreground colors. Project them back onto the robust glow color line, then
    # re-fit alpha against the known background so replay over the original
    # screen stays close while complementary previews have no green speckles.
    repaired_float[repair] = target.reshape(1, 1, 3)
    rgb = image_srgb.astype(np.float32)
    direction = repaired_float - bg.reshape(1, 1, 3)
    denom = np.sum(direction * direction, axis=2)
    delta = rgb - bg.reshape(1, 1, 3)
    repaired_alpha = alpha.copy()
    solved = repair & (denom > 1e-6)
    alpha_fit = np.clip(np.sum(delta * direction, axis=2) / np.maximum(denom, 1e-6), 0.0, 1.0)
    repaired_alpha[solved] = alpha_fit[solved]
    repaired = np.clip(repaired_float + 0.5, 0, 255).astype(np.uint8)
    return repaired, repaired_alpha.astype(np.float32), int(repair.sum())


def _is_chromatic_swap_target(
    background_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
) -> bool:
    bg = np.asarray(background_color, dtype=np.float32)
    target = np.asarray(target_color, dtype=np.float32)
    delta = target - bg
    return bool(np.max(delta) >= 80.0 and np.min(delta) <= -80.0)


def _solve_chromatic_swap_ray(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target_color, dtype=np.float32)
    alpha, residual = _solve_alpha(image_srgb, background_color, target)
    rgb = image_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32)
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    source_distance = np.linalg.norm(rgb - bg.reshape(1, 1, 3), axis=2)
    bright_core = (luma >= 205.0) & (chroma <= 92.0) & (source_distance >= 160.0)
    # In chromatic-swap glows, the transition from saturated glow color to
    # white highlight is foreground material, not transparency. Only the
    # transition from that material field back to the known screen should be
    # solved as alpha.
    chromatic_material = (
        (rgb[..., 1] >= 150.0)
        & (rgb[..., 1] >= rgb[..., 2] + 35.0)
        & (source_distance >= 120.0)
    )
    foreground_material = bright_core | chromatic_material
    foreground = np.broadcast_to(np.clip(target + 0.5, 0, 255).astype(np.uint8).reshape(1, 1, 3), image_srgb.shape).copy()
    foreground[foreground_material] = image_srgb[foreground_material]
    alpha = np.where(foreground_material, 1.0, alpha).astype(np.float32)
    return alpha.astype(np.float32), foreground, residual


def _chromatic_swap_debug(
    image_srgb: np.ndarray,
    foreground: np.ndarray,
    alpha: np.ndarray,
    keep: np.ndarray,
    background_color: tuple[int, int, int],
) -> dict[str, float]:
    bg = np.asarray(background_color, dtype=np.float32)
    replay = foreground.astype(np.float32) * alpha[..., None] + bg.reshape(1, 1, 3) * (1.0 - alpha[..., None])
    error = np.linalg.norm(replay - image_srgb.astype(np.float32), axis=2)
    if keep.any():
        kept_residual = error[keep]
        reconstruction_error_p95 = float(np.percentile(kept_residual, 95.0))
        reconstruction_error_p99 = float(np.percentile(kept_residual, 99.0))
    else:
        reconstruction_error_p95 = 0.0
        reconstruction_error_p99 = 0.0
    return {
        "reconstruction_error_p95": reconstruction_error_p95,
        "reconstruction_error_p99": reconstruction_error_p99,
    }


def analyze_known_bg_glow(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
) -> KnownBgGlowAnalysis:
    """Detect a simple continuous glow explained by one known-B mixing line.

    This is intentionally narrow. The accepted class is a large, connected
    soft component whose pixels fit ``C ~= alpha * F + (1-alpha) * B`` with a
    bright foreground endpoint. The residual and connectivity gates reject
    ordinary screen spill, scalar shadows, hard same-hue UI material, and
    fragmented particle effects.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("analyze_known_bg_glow() expects HxWx3 sRGB uint8")

    target = _estimate_target_color(image_srgb, background_color)
    bg = np.asarray(background_color, dtype=np.float32)
    target_distance = float(np.linalg.norm(target - bg))
    alpha, residual = _solve_alpha(image_srgb, background_color, target)
    model_support = (alpha >= 0.025) & (residual <= 14.0)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        model_support.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros_like(model_support, dtype=bool)
    largest_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 128:
            continue
        component = labels == label
        has_strong_core = bool(np.any(component & (alpha >= 0.45) & (residual <= 10.0)))
        if has_strong_core:
            keep |= component
            largest_area = max(largest_area, area)

    support_pixels = int(keep.sum())
    support_fraction = float(support_pixels / max(1, alpha.size))
    largest_component_fraction = float(largest_area / max(1, support_pixels))
    if support_pixels:
        kept_alpha = alpha[keep]
        kept_residual = residual[keep]
        soft_fraction = float(np.count_nonzero((kept_alpha >= 0.05) & (kept_alpha <= 0.75)) / support_pixels)
        strong_fraction = float(np.count_nonzero(kept_alpha >= 0.45) / support_pixels)
        residual_median = float(np.median(kept_residual))
        residual_p90 = float(np.percentile(kept_residual, 90.0))
        alpha_mean = float(kept_alpha.mean())
    else:
        soft_fraction = 0.0
        strong_fraction = 0.0
        residual_median = 0.0
        residual_p90 = 0.0
        alpha_mean = 0.0

    accepted = True
    reason = "accepted"
    if target_distance < 80.0:
        accepted = False
        reason = "target too close to background"
    elif support_fraction < 0.08:
        accepted = False
        reason = "insufficient glow support"
    elif support_fraction > 0.72:
        accepted = False
        reason = "support covers too much of frame"
    elif largest_component_fraction < 0.92:
        accepted = False
        reason = "support is fragmented"
    elif soft_fraction < 0.45:
        accepted = False
        reason = "transition band is too narrow"
    elif strong_fraction < 0.08:
        accepted = False
        reason = "missing bright glow core"
    elif residual_p90 > 10.0:
        accepted = False
        reason = "known-background glow model residual too high"

    target_u8 = tuple(int(np.clip(round(c), 0, 255)) for c in target)
    if accepted:
        return KnownBgGlowAnalysis(
            accepted=True,
            reason=reason,
            mode="single_target_line",
            background_color=tuple(int(c) for c in background_color),
            target_color=target_u8,
            support_pixels=support_pixels,
            support_fraction=support_fraction,
            largest_component_fraction=largest_component_fraction,
            soft_fraction=soft_fraction,
            outer_fraction=0.0,
            strong_fraction=strong_fraction,
            residual_median=residual_median,
            residual_p90=residual_p90,
            target_distance=target_distance,
            alpha_mean=alpha_mean,
        )

    adaptive_alpha, adaptive_foreground, _distance = _solve_adaptive_ray(image_srgb, background_color)
    adaptive = _adaptive_ray_metrics(adaptive_alpha)
    adaptive_reason = "accepted"
    adaptive_accepted = True
    # Adaptive-ray glow allows the foreground hue to vary pixel-by-pixel. The
    # acceptance gates therefore use topology and field smoothness instead of a
    # fixed target color: a large main component, a real low-alpha exterior
    # falloff band, and low outer roughness. Speckled particle effects fail the
    # roughness/correlation gates even when they cover a similar area.
    if adaptive["support_fraction"] < 0.08:
        adaptive_accepted = False
        adaptive_reason = "insufficient adaptive glow support"
    elif adaptive["support_fraction"] > 0.75:
        adaptive_accepted = False
        adaptive_reason = "adaptive support covers too much of frame"
    elif adaptive["largest_component_fraction"] < 0.94:
        adaptive_accepted = False
        adaptive_reason = "adaptive support is fragmented"
    elif adaptive["soft_fraction"] < 0.40:
        adaptive_accepted = False
        adaptive_reason = "adaptive transition band is too narrow"
    elif adaptive["outer_fraction"] < 0.25:
        adaptive_accepted = False
        adaptive_reason = "missing continuous low-alpha exterior glow"
    elif adaptive["outer_roughness_p90"] > 0.06:
        long_side = max(image_srgb.shape[:2])
        adaptive_target_probe = tuple(int(c) for c in np.median(adaptive_foreground[adaptive["main_component"]], axis=0)) if adaptive["support_pixels"] else target_u8
        chromatic_swap_coherent = (
            _is_chromatic_swap_target(background_color, adaptive_target_probe)
            and adaptive["outer_roughness_p90"] <= 0.09
            and adaptive["falloff_correlation"] >= 0.80
            and adaptive["largest_component_fraction"] >= 0.985
            and adaptive["outer_fraction"] >= 0.45
            and adaptive["soft_fraction"] >= 0.55
        )
        textured_but_coherent = (
            long_side < 128
            and adaptive["outer_roughness_p90"] <= 0.10
            and adaptive["falloff_correlation"] >= 0.90
            and adaptive["largest_component_fraction"] >= 0.985
            and adaptive["outer_fraction"] >= 0.42
            and adaptive["soft_fraction"] >= 0.60
            and adaptive["component_count"] <= 2
        )
        # Low-resolution glows can quantize a coherent falloff into visible
        # steps. Keep the normal texture guard for particles/noise, but accept
        # small icons when continuity, a broad low-alpha exterior, and strong
        # distance/alpha correlation still prove one glow field.
        if not (textured_but_coherent or chromatic_swap_coherent):
            adaptive_accepted = False
            adaptive_reason = "outer glow falloff is too textured"
    elif adaptive["falloff_correlation"] < 0.78:
        adaptive_accepted = False
        adaptive_reason = "outer glow does not fade coherently to background"

    adaptive_target = tuple(int(c) for c in np.median(adaptive_foreground[adaptive["main_component"]], axis=0)) if adaptive["support_pixels"] else target_u8
    adaptive_mode = "chromatic_swap_ray" if _is_chromatic_swap_target(background_color, adaptive_target) else "adaptive_ray"
    return KnownBgGlowAnalysis(
        accepted=adaptive_accepted,
        reason=adaptive_reason if adaptive_accepted else adaptive_reason,
        mode=adaptive_mode if adaptive_accepted else "rejected",
        background_color=tuple(int(c) for c in background_color),
        target_color=adaptive_target,
        support_pixels=int(adaptive["support_pixels"]),
        support_fraction=float(adaptive["support_fraction"]),
        largest_component_fraction=float(adaptive["largest_component_fraction"]),
        soft_fraction=float(adaptive["soft_fraction"]),
        outer_fraction=float(adaptive["outer_fraction"]),
        strong_fraction=float(adaptive["strong_fraction"]),
        residual_median=residual_median,
        residual_p90=residual_p90,
        target_distance=target_distance,
        alpha_mean=float(adaptive["alpha_mean"]),
        outer_roughness_p90=float(adaptive["outer_roughness_p90"]),
        falloff_correlation=float(adaptive["falloff_correlation"]),
    )


def matte_known_bg_glow(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    target_color: tuple[int, int, int] | None = None,
    *,
    mode: str = "auto",
) -> KnownBgGlowResult:
    """Return straight RGBA for a simple known-background glow."""
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("matte_known_bg_glow() expects HxWx3 sRGB uint8")
    if mode == "auto":
        analysis = analyze_known_bg_glow(image_srgb, background_color)
        mode = analysis.mode if analysis.accepted else "single_target_line"
        if target_color is None and analysis.accepted:
            target_color = analysis.target_color
    if mode == "chromatic_swap_ray":
        if target_color is None:
            target_color = tuple(int(c) for c in _estimate_target_color(image_srgb, background_color))
        alpha, foreground, _residual = _solve_chromatic_swap_ray(image_srgb, background_color, tuple(int(c) for c in target_color))
        metrics = _adaptive_ray_metrics(alpha)
        keep = metrics["main_component"].astype(bool)
        alpha = np.where(keep, alpha, 0.0).astype(np.float32)
        foreground = foreground.copy()
        foreground[~keep] = 0
        alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rgba = np.dstack([foreground, alpha_u8]).astype(np.uint8)
        error_debug = _chromatic_swap_debug(image_srgb, foreground, alpha, keep, background_color)
        debug = {
            "source": "known_bg_glow_chromatic_swap_solver",
            "mode": "chromatic_swap_ray",
            "background_color": [int(c) for c in background_color],
            "target_color": [int(c) for c in target_color],
            "support_pixels": int((alpha > 0.0).sum()),
            "alpha_mean": float(alpha.mean()),
            "outer_roughness_p90": float(metrics["outer_roughness_p90"]),
            "falloff_correlation": float(metrics["falloff_correlation"]),
            "foreground_repaired_pixels": 0,
            **error_debug,
        }
        return KnownBgGlowResult(rgba=rgba, alpha=alpha, foreground_srgb=foreground, debug=debug)

    if mode == "adaptive_ray":
        alpha, foreground, _distance = _solve_adaptive_ray(image_srgb, background_color)
        metrics = _adaptive_ray_metrics(alpha)
        keep = metrics["main_component"].astype(bool)
        alpha = np.where(keep, alpha, 0.0).astype(np.float32)
        if target_color is not None:
            target = np.asarray(target_color, dtype=np.uint8)
        elif keep.any():
            core = keep & (alpha >= 0.35)
            target = np.median(foreground[core if core.any() else keep], axis=0).astype(np.uint8)
        else:
            target = np.zeros(3, dtype=np.uint8)
        foreground = foreground.copy()
        foreground, alpha, repaired_pixels = _repair_additive_adaptive_foreground(
            image_srgb,
            foreground,
            alpha,
            keep,
            background_color,
            tuple(int(c) for c in target),
        )
        foreground[~keep] = 0
        alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rgba = np.dstack([foreground, alpha_u8]).astype(np.uint8)
        debug = {
            "source": "known_bg_glow_adaptive_ray_solver",
            "mode": "adaptive_ray",
            "background_color": [int(c) for c in background_color],
            "target_color": [int(c) for c in target],
            "support_pixels": int((alpha > 0.0).sum()),
            "alpha_mean": float(alpha.mean()),
            "outer_roughness_p90": float(metrics["outer_roughness_p90"]),
            "falloff_correlation": float(metrics["falloff_correlation"]),
            "foreground_repaired_pixels": repaired_pixels,
        }
        return KnownBgGlowResult(rgba=rgba, alpha=alpha, foreground_srgb=foreground, debug=debug)

    target = (
        np.asarray(target_color, dtype=np.float32)
        if target_color is not None
        else _estimate_target_color(image_srgb, background_color)
    )
    alpha, residual = _solve_alpha(image_srgb, background_color, target)
    support = (alpha >= 0.02) & (residual <= 14.0)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros_like(support, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        if area >= 128 and np.any(component & (alpha >= 0.45) & (residual <= 10.0)):
            keep |= component
    alpha = np.where(keep, alpha, 0.0).astype(np.float32)
    target_u8 = np.asarray(np.clip(target + 0.5, 0, 255), dtype=np.uint8)
    foreground = np.broadcast_to(target_u8.reshape(1, 1, 3), image_srgb.shape).copy()
    alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.dstack([foreground, alpha_u8]).astype(np.uint8)
    debug = {
        "source": "known_bg_glow_line_solver",
        "mode": "single_target_line",
        "background_color": [int(c) for c in background_color],
        "target_color": [int(c) for c in target_u8],
        "support_pixels": int((alpha > 0.0).sum()),
        "alpha_mean": float(alpha.mean()),
        "residual_median": float(np.median(residual[keep])) if keep.any() else 0.0,
        "residual_p90": float(np.percentile(residual[keep], 90.0)) if keep.any() else 0.0,
    }
    return KnownBgGlowResult(rgba=rgba, alpha=alpha, foreground_srgb=foreground, debug=debug)
