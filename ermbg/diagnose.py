"""Background diagnoser: assess whether an image is suitable for direct
analytic matting (i.e., it already has a clean solid-color background).

This replaces the AI-probe pipeline. We no longer compare 'original vs probe' —
we just measure properties of the single input image and decide:

  - is the background pure enough to model as a constant B?
  - is B perceptually distinct from edge subject colors?
  - which areas are at risk (subject color too close to B)?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from . import metrics
from .colorspace import oklab_distance, srgb_to_oklab
from .segmenter import _band_radius


@dataclass
class DiagnoserThresholds:
    purity_sigma_max: float = 5.0          # bg std-dev across uint8 RGB channels
    edge_contrast_q10_min: float = 8.0     # OKLab Q10 edge-vs-bg distance


@dataclass
class DiagnosisReport:
    background_color: tuple[int, int, int]
    purity_sigma: float
    purity_passed: bool
    edge_contrast_q10: float
    edge_contrast_passed: bool
    verdict: str  # "ready" | "risky" | "not-pure-bg"
    risk_map: np.ndarray | None = None
    extras: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "background_color": list(self.background_color),
            "purity": {"sigma": self.purity_sigma, "passed": self.purity_passed},
            "edge_contrast_q10": {
                "value": self.edge_contrast_q10,
                "passed": self.edge_contrast_passed,
            },
            "verdict": self.verdict,
            "extras": self.extras or {},
        }


def _build_risk_map(
    image: np.ndarray, mask: np.ndarray, B: np.ndarray, band_radius: int
) -> np.ndarray:
    """For each unknown-band pixel, compute 1 - normalized OKLab distance to B.

    High value (close to 1) = pixel color is dangerously close to background.
    """
    soft = mask.astype(np.float32)
    if soft.max() > 1.5:
        soft /= 255.0
    binary = (soft > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(binary, kernel, iterations=band_radius)
    ero = cv2.erode(binary, kernel, iterations=band_radius)
    band = (dil > 0) & (ero == 0)

    risk = np.zeros(soft.shape, dtype=np.float32)
    if band.any():
        bg_lab = srgb_to_oklab(B.reshape(1, 1, 3)).reshape(3)
        edge_lab = srgb_to_oklab(image[band])
        d = oklab_distance(edge_lab, bg_lab)
        # Normalize: ΔE 0–30 → risk 1.0–0.0
        norm = np.clip(1.0 - d / 30.0, 0.0, 1.0)
        risk[band] = norm
    return risk


class BackgroundDiagnoser:
    """Decide if an image is ready for direct matting on its observed background."""

    def __init__(self, thresholds: DiagnoserThresholds | None = None) -> None:
        self.t = thresholds or DiagnoserThresholds()

    def diagnose(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        band_radius: int | None = None,
    ) -> DiagnosisReport:
        if band_radius is None:
            band_radius = _band_radius(image.shape)

        B = metrics.measure_background_color(image, mask, dilate_radius=band_radius)
        sigma = metrics.background_purity_sigma(image, mask, dilate_radius=band_radius)
        contrast = metrics.edge_contrast_q10(image, mask, B, band_radius=band_radius)

        purity_passed = sigma <= self.t.purity_sigma_max
        contrast_passed = contrast >= self.t.edge_contrast_q10_min

        if not purity_passed:
            verdict = "not-pure-bg"
        elif not contrast_passed:
            verdict = "risky"
        else:
            verdict = "ready"

        risk = _build_risk_map(image, mask, B, band_radius)
        return DiagnosisReport(
            background_color=tuple(int(c) for c in B),
            purity_sigma=float(sigma),
            purity_passed=bool(purity_passed),
            edge_contrast_q10=float(contrast),
            edge_contrast_passed=bool(contrast_passed),
            verdict=verdict,
            risk_map=risk,
            extras={"band_radius": int(band_radius)},
        )


__all__ = ["BackgroundDiagnoser", "DiagnoserThresholds", "DiagnosisReport"]
