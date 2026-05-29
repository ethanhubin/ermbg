"""Local analysis and parameter selection for the CorridorKey backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .keyer import KeyerThresholds, chromatic_key_alpha

CorridorKeyScreenMode = Literal["auto", "green", "blue"]
CorridorKeyPreset = Literal["auto", "detail_safe", "spill_safe", "manual"]

GREEN_SCREEN_RGB = (0, 200, 0)
BLUE_SCREEN_RGB = (0, 80, 255)


@dataclass(frozen=True)
class CorridorKeyRecommendedSettings:
    gamma_space: str = "sRGB"
    despill_strength: float = 1.0
    refiner_strength: float = 1.0
    auto_despeckle: str = "On"
    despeckle_size: int = 400
    color_protection: bool = True
    protection_bg_max: float = 8.0
    protection_fg_min: float = 16.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorridorKeyAssetAnalysis:
    screen_mode: str
    requested_screen_mode: str
    preset: str
    background_color: tuple[int, int, int]
    background_confidence: float
    purity_sigma: float
    border_coverage: dict[str, float]
    subject_key_color_risk: float
    small_component_risk: bool
    recommended_settings: CorridorKeyRecommendedSettings
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["background_color"] = list(self.background_color)
        return payload


def _border_pixels(image_srgb: np.ndarray) -> np.ndarray:
    h, w = image_srgb.shape[:2]
    band = max(2, min(h, w) // 32)
    parts = [
        image_srgb[:band, :, :].reshape(-1, 3),
        image_srgb[-band:, :, :].reshape(-1, 3),
        image_srgb[:, :band, :].reshape(-1, 3),
        image_srgb[:, -band:, :].reshape(-1, 3),
    ]
    return np.concatenate(parts, axis=0)


def _family_mask(pixels: np.ndarray, family: str) -> np.ndarray:
    p = pixels.astype(np.int16)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    if family == "green":
        return (g >= 72) & (g >= r + 35) & (g >= b + 25)
    if family == "blue":
        return (b >= 72) & (b >= r + 35) & (b >= g + 20)
    raise ValueError(f"Unknown screen family: {family!r}")


def _estimate_background(pixels: np.ndarray, family: str, fallback: tuple[int, int, int]) -> tuple[tuple[int, int, int], float, float]:
    family_pixels = pixels[_family_mask(pixels, family)]
    coverage = float(len(family_pixels) / max(1, len(pixels)))
    if len(family_pixels) == 0:
        return fallback, coverage, 100.0

    bg = np.median(family_pixels, axis=0)
    bg_u8 = tuple(int(np.clip(round(float(v)), 0, 255)) for v in bg)
    lab = srgb_to_oklab(family_pixels.astype(np.uint8).reshape(1, -1, 3)).reshape(-1, 3)
    bg_lab = srgb_to_oklab(np.asarray(bg_u8, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    # This sigma is intentionally measured only over border pixels in the
    # selected color family. It keys on screen stability, while ignoring subject
    # pixels that touch the frame and would make whole-edge variance useless.
    purity_sigma = float(np.std(oklab_distance(lab, bg_lab))) if len(family_pixels) > 1 else 0.0
    return bg_u8, coverage, purity_sigma


def _screen_score(coverage: float, purity_sigma: float) -> float:
    purity_factor = float(np.clip(1.0 - purity_sigma / 18.0, 0.0, 1.0))
    return coverage * (0.55 + 0.45 * purity_factor)


def _foreground_component_stats(key_alpha: np.ndarray) -> tuple[bool, int]:
    support = key_alpha >= 0.45
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    if not areas:
        return False, 0
    image_area = float(key_alpha.size)
    small_limit = max(8, int(image_area * 0.0008))
    small_components = sum(1 for area in areas if area <= small_limit)
    return small_components >= 6, small_components


def _subject_key_color_risk(image_srgb: np.ndarray, background_color: tuple[int, int, int]) -> float:
    key_alpha = chromatic_key_alpha(
        image_srgb,
        background_color,
        KeyerThresholds(bg_max=5.5, fg_min=18.0),
    )
    candidate_subject = key_alpha >= 0.16
    if not candidate_subject.any():
        return 0.0

    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    delta_ab = lab[..., 1:] - bg_lab[1:]
    ab_distance = np.sqrt(np.sum(delta_ab * delta_ab, axis=-1)).astype(np.float32) * 100.0
    near_key_family = ab_distance <= 12.0
    # The risk is measured over pixels with some non-background evidence. It
    # catches same-hue UI material that a strong keyer/despill pass may erase,
    # without counting the stable screen itself as subject.
    return float((near_key_family & candidate_subject).sum() / max(1, int(candidate_subject.sum())))


def _recommend_settings(
    *,
    preset: str,
    subject_key_color_risk: float,
    small_component_risk: bool,
    purity_sigma: float,
) -> tuple[CorridorKeyRecommendedSettings, list[str]]:
    notes: list[str] = []
    if preset == "manual":
        return CorridorKeyRecommendedSettings(), ["manual preset leaves caller-provided settings unchanged"]
    if preset == "spill_safe":
        return (
            CorridorKeyRecommendedSettings(
                despill_strength=1.0,
                refiner_strength=1.15,
                auto_despeckle="On",
                despeckle_size=400,
                protection_bg_max=8.0,
                protection_fg_min=18.0,
            ),
            ["spill_safe preset favors stronger green-spill cleanup"],
        )
    if preset == "detail_safe":
        return (
            CorridorKeyRecommendedSettings(
                despill_strength=0.65,
                refiner_strength=0.85,
                auto_despeckle="Off",
                despeckle_size=64,
                protection_bg_max=6.0,
                protection_fg_min=14.0,
            ),
            ["detail_safe preset protects small ornaments and key-color subject material"],
        )

    settings = CorridorKeyRecommendedSettings()
    if subject_key_color_risk >= 0.025:
        # Empirical gate: a few percent of "not screen" pixels still being in
        # the screen hue family means UI material overlaps the key color. Use a
        # gentler refiner/despill path and tighter color-protection endpoints.
        settings = CorridorKeyRecommendedSettings(
            despill_strength=0.65,
            refiner_strength=0.85,
            auto_despeckle=settings.auto_despeckle,
            despeckle_size=settings.despeckle_size,
            protection_bg_max=6.0,
            protection_fg_min=14.0,
        )
        notes.append("subject_key_color_risk reduced despill/refiner and tightened color protection")
    if small_component_risk:
        settings = CorridorKeyRecommendedSettings(
            gamma_space=settings.gamma_space,
            despill_strength=settings.despill_strength,
            refiner_strength=settings.refiner_strength,
            auto_despeckle="Off",
            despeckle_size=64,
            color_protection=settings.color_protection,
            protection_bg_max=settings.protection_bg_max,
            protection_fg_min=settings.protection_fg_min,
        )
        notes.append("small UI components disabled auto despeckle")
    if purity_sigma >= 8.0:
        settings = CorridorKeyRecommendedSettings(
            gamma_space=settings.gamma_space,
            despill_strength=min(settings.despill_strength, 0.8),
            refiner_strength=min(settings.refiner_strength, 0.9),
            auto_despeckle=settings.auto_despeckle,
            despeckle_size=settings.despeckle_size,
            color_protection=settings.color_protection,
            protection_bg_max=settings.protection_bg_max,
            protection_fg_min=settings.protection_fg_min,
        )
        notes.append("low screen purity reduced aggressive refinement")
    if not notes:
        notes.append("standard settings selected for clean screen and low subject key-color risk")
    return settings, notes


def corridorkey_analyze_asset(
    image_srgb: np.ndarray,
    *,
    screen_mode: CorridorKeyScreenMode = "auto",
    preset: CorridorKeyPreset = "auto",
    fallback_background_color: tuple[int, int, int] = GREEN_SCREEN_RGB,
) -> CorridorKeyAssetAnalysis:
    """Analyze a game UI screen asset before CorridorKey execution.

    This is intentionally local and deterministic. It decides which screen
    family the upload appears to use, estimates the background color, and picks
    conservative CorridorKey settings for UI assets without loading any heavy
    model on the Mac.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("corridorkey_analyze_asset() expects HxWx3 sRGB uint8")
    if screen_mode not in {"auto", "green", "blue"}:
        raise ValueError("screen_mode must be auto, green, or blue")
    if preset not in {"auto", "detail_safe", "spill_safe", "manual"}:
        raise ValueError("preset must be auto, detail_safe, spill_safe, or manual")

    border = _border_pixels(image_srgb)
    green_bg, green_coverage, green_sigma = _estimate_background(border, "green", GREEN_SCREEN_RGB)
    blue_bg, blue_coverage, blue_sigma = _estimate_background(border, "blue", BLUE_SCREEN_RGB)
    green_score = _screen_score(green_coverage, green_sigma)
    blue_score = _screen_score(blue_coverage, blue_sigma)

    notes: list[str] = []
    if screen_mode == "green":
        selected_mode = "green"
        background_color = green_bg if green_coverage > 0.0 else fallback_background_color
        confidence = green_score
        purity_sigma = green_sigma
        notes.append("screen mode forced to green")
    elif screen_mode == "blue":
        selected_mode = "blue"
        background_color = blue_bg if blue_coverage > 0.0 else BLUE_SCREEN_RGB
        confidence = blue_score
        purity_sigma = blue_sigma
        notes.append("screen mode forced to blue")
    elif max(green_score, blue_score) < 0.08:
        selected_mode = "unknown"
        background_color = fallback_background_color
        confidence = max(green_score, blue_score)
        purity_sigma = min(green_sigma, blue_sigma)
        notes.append("no confident green or blue screen found; falling back to caller background color")
    elif blue_score > green_score * 1.15:
        selected_mode = "blue"
        background_color = blue_bg
        confidence = blue_score
        purity_sigma = blue_sigma
        notes.append("auto detected blue screen")
    else:
        selected_mode = "green"
        background_color = green_bg
        confidence = green_score
        purity_sigma = green_sigma
        notes.append("auto detected green screen")

    key_alpha = chromatic_key_alpha(image_srgb, background_color, KeyerThresholds(bg_max=5.5, fg_min=18.0))
    small_component_risk, small_count = _foreground_component_stats(key_alpha)
    if small_component_risk:
        notes.append(f"detected {small_count} small foreground components")

    subject_risk = _subject_key_color_risk(image_srgb, background_color)
    settings, setting_notes = _recommend_settings(
        preset=preset,
        subject_key_color_risk=subject_risk,
        small_component_risk=small_component_risk,
        purity_sigma=purity_sigma,
    )
    notes.extend(setting_notes)
    if selected_mode == "blue":
        notes.append("blue screen route is analysis-ready but CorridorKey blue model support still requires validation")

    return CorridorKeyAssetAnalysis(
        screen_mode=selected_mode,
        requested_screen_mode=screen_mode,
        preset=preset,
        background_color=background_color,
        background_confidence=float(confidence),
        purity_sigma=float(purity_sigma),
        border_coverage={"green": float(green_coverage), "blue": float(blue_coverage)},
        subject_key_color_risk=subject_risk,
        small_component_risk=small_component_risk,
        recommended_settings=settings,
        notes=notes,
    )


__all__ = [
    "BLUE_SCREEN_RGB",
    "GREEN_SCREEN_RGB",
    "CorridorKeyAssetAnalysis",
    "CorridorKeyRecommendedSettings",
    "corridorkey_analyze_asset",
]
