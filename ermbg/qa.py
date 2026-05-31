"""Multi-background QA: composite the RGBA onto N test backgrounds and score
the result. Plan §26.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import io


# (name, sRGB color or special "checker")
QA_BACKGROUNDS: list[tuple[str, Any]] = [
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("grey", (128, 128, 128)),
    ("cyan", (0, 200, 220)),
    ("magenta", (220, 30, 180)),
    ("checker", "checker"),
]


def _checker(h: int, w: int, size: int = 32) -> np.ndarray:
    yy, xx = np.indices((h, w))
    cells = ((yy // size) + (xx // size)) % 2
    img = np.where(cells[..., None], 200, 80).astype(np.uint8)
    return np.broadcast_to(img, (h, w, 3)).copy()


def composite(rgba: np.ndarray, bg_color_or_image) -> np.ndarray:
    """Composite an RGBA result onto a background. Linear-RGB blend.

    rgba: H x W x 4 sRGB uint8 (alpha in last channel).
    bg_color_or_image: 3-tuple sRGB or HxWx3 sRGB uint8 image.
    """
    h, w = rgba.shape[:2]
    if isinstance(bg_color_or_image, np.ndarray):
        bg = bg_color_or_image.astype(np.uint8)
        if bg.shape != (h, w, 3):
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    else:
        bg = np.broadcast_to(np.asarray(bg_color_or_image, dtype=np.uint8), (h, w, 3)).copy()

    F_lin = io.srgb_to_linear(rgba[..., :3])
    B_lin = io.srgb_to_linear(bg)
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    out_lin = a * F_lin + (1.0 - a) * B_lin
    return io.linear_to_srgb_u8(out_lin)


def edge_halo_score(rgba: np.ndarray, bg_color: tuple[int, int, int]) -> float:
    """OKLab ΔE between (composite vs bg) on **near-transparent** edge pixels.

    A 'halo' is by definition: pixels where alpha is very low (≤ 0.15) yet the
    composite still differs from the target background. We exclude pixels with
    higher alpha because there a colorimetric difference is *expected* — that's
    just the subject's color showing through, not a halo.

    Lower = cleaner edges (no halo).
    """
    from .colorspace import oklab_distance, srgb_to_oklab

    alpha = rgba[..., 3].astype(np.float32) / 255.0
    halo_band = (alpha > 0.0) & (alpha <= 0.15)
    if not halo_band.any():
        return 0.0
    composite_img = composite(rgba, bg_color)
    a_lab = srgb_to_oklab(composite_img[halo_band])
    bg_lab = srgb_to_oklab(np.asarray(bg_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    return float(np.mean(oklab_distance(a_lab, bg_lab)))


def alpha_noise_score(alpha: np.ndarray) -> float:
    """P95 of |∇α| inside the soft band. Lower = smoother alpha matte."""
    a = alpha if alpha.dtype == np.float32 else (alpha.astype(np.float32) / 255.0)
    gx = np.abs(np.diff(a, axis=1))
    gy = np.abs(np.diff(a, axis=0))
    grad = np.maximum(gx[:-1], gy[:, :-1])
    band = ((a[:-1, :-1] > 0.05) & (a[:-1, :-1] < 0.95))
    if not band.any():
        return 0.0
    return float(np.percentile(grad[band], 95.0))


def thin_structure_preservation(soft_mask: np.ndarray, alpha: np.ndarray) -> float:
    """How much of the original mask's small connected components survived?

    Returns ratio in [0, 1]. 1.0 = all preserved.
    """
    bin_in = (soft_mask > 0.5).astype(np.uint8)
    bin_out = (alpha > 0.05).astype(np.uint8)

    # Find small components (area < 0.5% of image) in the input mask.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_in, connectivity=8)
    if n <= 1:
        return 1.0
    h, w = bin_in.shape
    small_threshold = 0.005 * h * w
    survived = 0
    total = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > small_threshold:
            continue
        total += 1
        ys, xs = np.where(labels == i)
        if bin_out[ys, xs].any():
            survived += 1
    if total == 0:
        return 1.0
    return survived / total


def recomposition_error(image_srgb: np.ndarray, rgba: np.ndarray, bg_color: tuple[int, int, int]) -> float:
    """L2 distance in linear-RGB between the original image and (alpha*F + (1-α)*B).

    Plan §26.1.
    """
    composite_img = composite(rgba, bg_color)
    a_lin = io.srgb_to_linear(image_srgb)
    b_lin = io.srgb_to_linear(composite_img)
    diff = a_lin - b_lin
    return float(np.sqrt(np.mean(diff * diff)))


def run_qa(
    image_srgb: np.ndarray,
    rgba: np.ndarray,
    soft_mask: np.ndarray,
    background_color: tuple[int, int, int],
    out_dir: Path,
    write_lightwrap: bool = True,
) -> dict[str, Any]:
    """Composite to all standard backgrounds, save them, return aggregate metrics.

    When ``write_lightwrap`` is true (default), also writes a `_lightwrap` screen
    next to each composite so the user can compare halo suppression visually.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = rgba.shape[:2]
    halo_scores: dict[str, float] = {}
    composites: dict[str, np.ndarray] = {}
    if write_lightwrap:
        from .lightwrap import light_wrap_composite

    for name, spec in QA_BACKGROUNDS:
        if spec == "checker":
            bg = _checker(h, w)
            comp = composite(rgba, bg)
            # Halo score not well-defined for checker; skip.
        else:
            comp = composite(rgba, spec)
            halo_scores[name] = edge_halo_score(rgba, spec)
        io.save_rgb(out_dir / f"on_{name}.png", comp)
        composites[name] = comp

        if write_lightwrap:
            if spec == "checker":
                lw = light_wrap_composite(
                    rgba[..., :3], rgba[..., 3].astype(np.float32) / 255.0, bg
                )
            else:
                lw = light_wrap_composite(
                    rgba[..., :3], rgba[..., 3].astype(np.float32) / 255.0, spec
                )
            io.save_rgb(out_dir / f"on_{name}_lightwrap.png", lw)

    metrics = {
        "recomposition_error_on_observed_bg": recomposition_error(image_srgb, rgba, background_color),
        "edge_halo_score_per_bg": halo_scores,
        "edge_halo_score_mean": float(np.mean(list(halo_scores.values()))) if halo_scores else 0.0,
        "alpha_noise_p95": alpha_noise_score(rgba[..., 3]),
        "thin_structure_preservation": thin_structure_preservation(soft_mask, rgba[..., 3]),
    }
    return metrics


__all__ = [
    "composite",
    "edge_halo_score",
    "alpha_noise_score",
    "thin_structure_preservation",
    "recomposition_error",
    "run_qa",
    "QA_BACKGROUNDS",
]
