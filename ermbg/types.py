"""Core data types passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProbeImage:
    """A solid-color background probe image with measurements."""

    image: np.ndarray  # H x W x 3, sRGB uint8
    background_color: tuple[int, int, int]  # target sRGB
    measured_background: np.ndarray | None = None  # actual background color (3,) or plane params
    mask: np.ndarray | None = None  # soft mask resegmented on this probe (float32 0..1)
    purity_score: float = 0.0  # background purity, higher is purer
    consistency_score: float = 0.0  # subject consistency vs original (0..1)
    contrast_score: float = 0.0  # edge-vs-bg perceptual distance Q10
    valid: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trimap:
    """Trimap for matting: definite-fg / definite-bg / unknown."""

    sure_fg: np.ndarray  # bool H x W
    sure_bg: np.ndarray
    unknown: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.sure_fg.shape  # type: ignore[return-value]


@dataclass
class MattingResult:
    """Shared rich matting output used by maintained backend helpers."""

    rgba: np.ndarray              # H x W x 4 sRGB uint8
    alpha: np.ndarray             # H x W float32 0..1
    foreground_srgb: np.ndarray   # H x W x 3 sRGB uint8
    foreground_linear: np.ndarray | None = None
    trimap: Trimap | None = None
    background_color: tuple[int, int, int] | None = None
    diagnosis: Any = None
    debug: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Per-probe validator output."""

    purity_sigma: float
    purity_passed: bool
    internal_iou: float
    internal_iou_passed: bool
    edge_hausdorff_px: float
    edge_hausdorff_passed: bool
    internal_dE_p95: float
    internal_dE_passed: bool
    edge_contrast_q10: float
    edge_contrast_passed: bool
    verdict: str  # "valid" | "weighted" | "rejected"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "purity": {"sigma": self.purity_sigma, "passed": self.purity_passed},
            "internal_iou": {"value": self.internal_iou, "passed": self.internal_iou_passed},
            "edge_hausdorff_px": {
                "value": self.edge_hausdorff_px,
                "passed": self.edge_hausdorff_passed,
            },
            "internal_dE_p95": {
                "value": self.internal_dE_p95,
                "passed": self.internal_dE_passed,
            },
            "edge_contrast_q10": {
                "value": self.edge_contrast_q10,
                "passed": self.edge_contrast_passed,
            },
            "verdict": self.verdict,
            "extras": self.extras,
        }
