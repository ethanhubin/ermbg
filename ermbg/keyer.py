"""Chromatic / luminance key matting on top of a known background color.

When B is fixed (the system's "specified background" contract), the cleanest
signal for "is this pixel background?" is the perceptual distance from each
pixel to B. Two flavors:

  ``chromatic_key_alpha``  — uses full OKLab distance. Good for saturated B
      (green/cyan/magenta). Useless when B has near-zero chroma (white/black/grey)
      because then the signal is dominated by lightness, not color.

  ``luminance_key_alpha``  — uses |ΔL| only. The right tool for white or
      black backgrounds: dark subjects on a white screen separate cleanly by
      lightness alone. Cannot tell "white subject on white" from "background";
      that case is information-theoretically lost regardless of method.

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

from .colorspace import oklab_distance, srgb_to_oklab


@dataclass
class KeyerThresholds:
    """OKLab ΔE thresholds for the soft key.

    Pixels with d <= ``bg_max`` are full background (α=0).
    Pixels with d >= ``fg_min`` are full foreground (α=1).
    In between, α ramps linearly. Defaults are tuned for a saturated screen
    (green/cyan/magenta) — for a low-saturation B (e.g. white on white)
    these will be too tight and the keyer should be skipped or replaced.
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


def merge_alpha_components(
    matting_alpha: np.ndarray,
    chromatic_alpha: np.ndarray,
    min_component_area_ratio: float = 0.0005,
    max_component_area_ratio: float = 0.5,
    matting_present_coverage: float = 0.30,
    fg_threshold: float = 0.5,
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
         decrease an existing α.
      4. The matting α elsewhere is left unchanged so we don't lose its better
         edge feathering.

    The 'coverage' rule (rather than 'any pixel above ε') matters on white
    backgrounds: there, a missed small subject can pick up tiny BiRefNet halo
    leak (α≈0.05), and an 'any' check would falsely conclude matting saw it.

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

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp_mask = labels == i
        # Fraction of the component that matting also considers foreground.
        coverage = float((matting_alpha[comp_mask] >= fg_threshold).mean())
        if coverage >= matting_present_coverage:
            continue
        # Patch in the chromatic α for this component.
        merged[comp_mask] = np.maximum(merged[comp_mask], chromatic_alpha[comp_mask])
        patched.append(i)
        areas.append(area)

    return merged, {"patched_components": len(patched), "component_areas": areas}


__all__ = [
    "KeyerThresholds",
    "chromatic_key_alpha",
    "merge_alpha_components",
]
