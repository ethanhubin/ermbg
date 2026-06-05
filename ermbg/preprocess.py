"""Shared preprocessing for ERMBG inputs.

Preprocess runs before route or semantic analysis. Stage 2 keeps behavior
compatible with the existing Web flow while moving the observable decisions
into shared contracts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pipeline_contracts import (
    BackgroundModel,
    PreprocessAnalysis,
    PreprocessDecision,
    PreprocessItem,
)
from .slicer import analyze_checkerboard_background, normalize_checkerboard_background_to_light_square

REMOVE_CHECKERBOARD = "remove_checkerboard"
NORMALIZE_KNOWN_BACKGROUND = "normalize_known_background"


@dataclass(frozen=True)
class PreprocessResult:
    image_srgb: np.ndarray
    analysis: PreprocessAnalysis
    decision: PreprocessDecision


def _image_preprocess_id(image_srgb: np.ndarray) -> str:
    digest = hashlib.sha256(image_srgb.tobytes()).hexdigest()
    return f"pre_{digest[:16]}"


def _background_model_from_checkerboard(info: dict[str, Any]) -> BackgroundModel | None:
    color = info.get("background_color")
    if not (isinstance(color, list) and len(color) == 3):
        return None
    return BackgroundModel(
        color=tuple(int(c) for c in color),  # type: ignore[arg-type]
        source=str(info.get("source") or "checkerboard_background"),
        confidence=1.0 if info.get("accepted") else None,
        metadata={"checkerboard": info},
    )


def analyze_input_preprocess(image_srgb: np.ndarray) -> PreprocessAnalysis:
    """Return lightweight preprocess recommendations for an RGB input."""

    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    checkerboard = analyze_checkerboard_background(image_srgb)
    items: list[PreprocessItem] = []
    if checkerboard.get("accepted", False):
        items.append(
            PreprocessItem(
                id=REMOVE_CHECKERBOARD,
                label="Remove checkerboard",
                recommended=True,
                enabled_by_default=True,
                reason="detected_checkerboard_background",
                metadata=checkerboard,
            )
        )
    return PreprocessAnalysis(
        preprocess_id=_image_preprocess_id(image_srgb),
        items=items,
        background_model=_background_model_from_checkerboard(checkerboard),
        debug={"checkerboard": checkerboard},
    )


def apply_input_preprocess(
    image_srgb: np.ndarray,
    *,
    selected: list[str] | tuple[str, ...] | set[str] | None = None,
) -> PreprocessResult:
    """Apply selected input preprocessing items and return the shared decision."""

    analysis = analyze_input_preprocess(image_srgb)
    selected_ids = [str(item) for item in (selected or [])]
    selected_set = set(selected_ids)
    output = image_srgb
    applied: list[str] = []
    metadata: dict[str, Any] = {"checkerboard": checkerboard_info_from_analysis(analysis, requested=REMOVE_CHECKERBOARD in selected_set)}

    if REMOVE_CHECKERBOARD in selected_set:
        normalized, info = normalize_checkerboard_background_to_light_square(output)
        metadata["checkerboard"] = {"requested": True, **info}
        if info.get("applied", False):
            output = normalized
            applied.append(REMOVE_CHECKERBOARD)
    elif not metadata["checkerboard"].get("requested"):
        metadata["checkerboard"] = {"enabled": True, "requested": False, "applied": False, **analysis.debug.get("checkerboard", {})}

    decision = PreprocessDecision(
        selected=selected_ids,
        applied=applied,
        metadata=metadata,
        background_model=analysis.background_model,
    )
    return PreprocessResult(image_srgb=output, analysis=analysis, decision=decision)


def normalize_known_background_preprocess(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    bg_threshold: float = 3.5,
    fg_threshold: float = 24.0,
    adaptive: bool = True,
) -> tuple[np.ndarray, PreprocessDecision]:
    """Run the Known-B background-field normalization as a Preprocess item.

    This helper centralizes the existing mechanism without changing executor
    behavior yet. Later stages can pass its output to Analyze and Execute.
    """

    from .pymatting_refine import normalize_known_background_field

    normalized, info = normalize_known_background_field(
        image_srgb,
        background_color,
        bg_threshold=bg_threshold,
        fg_threshold=fg_threshold,
        adaptive=adaptive,
    )
    bg = tuple(int(c) for c in np.asarray(background_color, dtype=np.uint8).reshape(3))
    decision = PreprocessDecision(
        selected=[NORMALIZE_KNOWN_BACKGROUND],
        applied=[NORMALIZE_KNOWN_BACKGROUND] if info.get("applied", False) else [],
        metadata={"known_background_normalization": info},
        background_model=BackgroundModel(
            color=bg,
            source=str(info.get("source") or "known_background"),
            confidence=1.0,
            metadata={"normalization": info},
        ),
    )
    return normalized, decision


def checkerboard_info_from_analysis(analysis: PreprocessAnalysis, *, requested: bool) -> dict[str, Any]:
    info = dict(analysis.debug.get("checkerboard", {}))
    if requested:
        return {"requested": True, **info}
    return {"enabled": True, "requested": False, "applied": False, **info}


def checkerboard_info_from_decision(decision: PreprocessDecision) -> dict[str, Any]:
    info = decision.metadata.get("checkerboard")
    return dict(info) if isinstance(info, dict) else {"enabled": True, "requested": False, "applied": False}


__all__ = [
    "NORMALIZE_KNOWN_BACKGROUND",
    "REMOVE_CHECKERBOARD",
    "PreprocessResult",
    "analyze_input_preprocess",
    "apply_input_preprocess",
    "checkerboard_info_from_analysis",
    "checkerboard_info_from_decision",
    "normalize_known_background_preprocess",
]
