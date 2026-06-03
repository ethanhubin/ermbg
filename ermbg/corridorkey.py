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
BLUE_SCREEN_RGB = (0, 0, 255)


@dataclass(frozen=True)
class CorridorKeyRecommendedSettings:
    gamma_space: str = "sRGB"
    despill_strength: float = 1.0
    refiner_strength: float = 1.0
    auto_despeckle: str = "On"
    despeckle_size: int = 400
    color_protection: bool = True
    protection_bg_max: float = 12.0
    protection_fg_min: float = 28.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorridorKeyDecisionCandidate:
    profile: str
    label: str
    confidence: float
    settings: CorridorKeyRecommendedSettings
    reason: str
    selected: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["settings"] = self.settings.to_dict()
        return payload


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
    key_color_solid_fraction: float
    key_color_hard_density: float
    key_color_compact_fill: float
    key_color_compact_fraction: float
    key_transition_fraction: float
    same_key_opaque_plateau_confidence: float
    foreground_bbox_xyxy: tuple[int, int, int, int] | None
    foreground_aspect_ratio: float | None
    foreground_long_side: int
    hard_screen_residue_risk: float
    small_component_risk: bool
    parameter_profile: str
    decision_candidates: list[CorridorKeyDecisionCandidate]
    recommended_settings: CorridorKeyRecommendedSettings
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["background_color"] = list(self.background_color)
        if self.foreground_bbox_xyxy is not None:
            payload["foreground_bbox_xyxy"] = list(self.foreground_bbox_xyxy)
        payload["decision_candidates"] = [candidate.to_dict() for candidate in self.decision_candidates]
        payload["recommended_settings"] = self.recommended_settings.to_dict()
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


def _foreground_geometry(key_alpha: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float | None, int]:
    """Return largest foreground-support geometry for semantic routing.

    Canvas size is a weak proxy for asset type: a wide button can sit on a
    square export canvas, and a character can leave wide empty margins. Use the
    observed key support bbox so character/button routing follows the subject
    geometry rather than the file dimensions.
    """
    support = key_alpha >= 0.16
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return None, None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    label_offset = int(np.argmax(areas)) + 1
    x = int(stats[label_offset, cv2.CC_STAT_LEFT])
    y = int(stats[label_offset, cv2.CC_STAT_TOP])
    w = int(stats[label_offset, cv2.CC_STAT_WIDTH])
    h = int(stats[label_offset, cv2.CC_STAT_HEIGHT])
    if w <= 0 or h <= 0:
        return None, None, 0
    return (x, y, x + w, y + h), float(w / max(1, h)), int(max(w, h))


def _key_color_material_stats(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    key_alpha: np.ndarray,
) -> tuple[float, float, float, float, float]:
    candidate_subject = key_alpha >= 0.16
    if not candidate_subject.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0

    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    delta_ab = lab[..., 1:] - bg_lab[1:]
    ab_distance = np.sqrt(np.sum(delta_ab * delta_ab, axis=-1)).astype(np.float32) * 100.0
    near_key_family = ab_distance <= 12.0
    near_subject = near_key_family & candidate_subject
    near_count = int(near_subject.sum())
    if near_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    hard_subject = key_alpha >= 0.75
    near_hard = near_subject & hard_subject
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(near_subject.astype(np.uint8), connectivity=8)
    compact_fill = 0.0
    compact_fraction = 0.0
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        fills = areas / np.maximum(widths * heights, 1.0)
        score = fills * np.sqrt(areas / max(1.0, float(near_count)))
        idx = int(np.argmax(score))
        compact_fill = float(fills[idx])
        compact_fraction = float(areas[idx] / max(1.0, float(near_count)))
    # The risk is measured over pixels with some non-background evidence. It
    # catches same-hue UI material that a strong keyer/despill pass may erase,
    # without counting the stable screen itself as subject.
    subject_key_color_risk = float(near_count / max(1, int(candidate_subject.sum())))
    # Color protection should only become strong when near-key evidence is
    # anchored in hard/solid subject support. Broad soft ownership is usually
    # screen tint passing through translucent material, and CorridorKey should
    # own that alpha instead of an alpha floor rescuing it as solid subject.
    solid_fraction = float(near_hard.sum() / max(1, near_count))
    hard_density = float(near_hard.sum() / max(1, int(hard_subject.sum())))
    return subject_key_color_risk, solid_fraction, hard_density, compact_fill, compact_fraction


def _key_transition_fraction(key_alpha: np.ndarray) -> float:
    candidate_subject = key_alpha >= 0.16
    if not candidate_subject.any():
        return 0.0
    transition = (key_alpha >= 0.16) & (key_alpha < 0.75)
    return float(transition.sum() / max(1, int(candidate_subject.sum())))


def _same_key_opaque_plateau_confidence(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    key_alpha: np.ndarray,
) -> float:
    """Score same-key-family pixels as an opaque material plateau.

    A broad near-background-color area is not enough evidence for
    translucency: a flat cyan/blue/green UI material can be opaque even when it
    sits close to the screen hue. The distinguishing signal is whether the
    largest near-key component has substantial hard support. True glass/screen
    tint tends to be a low-alpha ramp with little or no plateau; opaque UI
    material has a coherent component whose interior can be explained as
    alpha=1, leaving only the boundary/gradient band for matting.
    """
    candidate_subject = key_alpha >= 0.16
    if not bool(candidate_subject.any()):
        return 0.0

    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    delta_ab = lab[..., 1:] - bg_lab[1:]
    ab_distance = np.sqrt(np.sum(delta_ab * delta_ab, axis=-1)).astype(np.float32) * 100.0
    near_subject = (ab_distance <= 12.0) & candidate_subject
    near_count = int(near_subject.sum())
    if near_count < max(32, int(key_alpha.size * 0.01)):
        return 0.0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(near_subject.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    label = int(np.argmax(areas)) + 1
    area = float(stats[label, cv2.CC_STAT_AREA])
    width = float(stats[label, cv2.CC_STAT_WIDTH])
    height = float(stats[label, cv2.CC_STAT_HEIGHT])
    if width <= 0.0 or height <= 0.0:
        return 0.0

    component = labels == label
    component_fraction = area / max(1.0, float(near_count))
    compact_fill = area / max(1.0, width * height)
    hard_fraction = float((key_alpha[component] >= 0.75).mean())
    transition_fraction = float(((key_alpha[component] >= 0.16) & (key_alpha[component] < 0.75)).mean())

    # Threshold intent:
    # - component_fraction/compact_fill require one coherent material plateau,
    #   not scattered spill/glow particles.
    # - hard_fraction is the main anti-glass signal: real transparent ramps have
    #   broad near-key support but little alpha>=0.75 plateau.
    # - transition_fraction guard keeps almost-all-ramp components from being
    #   rescued merely because they are flat in color.
    coherence = min(component_fraction / 0.60, compact_fill / 0.60)
    hard_support = hard_fraction / 0.35
    ramp_guard = np.clip((0.95 - transition_fraction) / 0.35, 0.0, 1.0)
    return float(np.clip(min(coherence, hard_support) * ramp_guard, 0.0, 1.0))


def _hard_screen_residue_risk(subject_key_color_risk: float, key_transition_fraction: float) -> float:
    """Score hard same-screen-family residue that needs stronger refinement.

    The failure class is measurable: a moderate amount of candidate foreground
    remains in the screen hue family, but the keyer transition band is tiny.
    That pattern describes hard shadow/antialias residue on a known screen,
    where CorridorKey may leave low-alpha black fringe. It deliberately excludes
    dominant same-hue material and glass/glow cases, which have either much
    higher same-hue ownership or broad soft transitions and need protection
    rather than more aggressive refinement.
    """
    moderate_same_hue = 0.04 <= subject_key_color_risk <= 0.25
    hard_transition = key_transition_fraction <= 0.015
    if not (moderate_same_hue and hard_transition):
        return 0.0
    transition_factor = float(np.clip((0.015 - key_transition_fraction) / 0.015, 0.0, 1.0))
    return float(subject_key_color_risk * transition_factor)


def _color_protection_endpoints(subject_key_color_risk: float, screen_mode: str) -> tuple[float, float, str]:
    """Generate color-protection thresholds from same-hue subject evidence.

    The feature is the fraction of candidate foreground that remains in the
    key-color a/b family. Low values mean any near-key soft edge is more likely
    screen mixing or spill, so protection starts farther from the background.
    High values mean actual subject material overlaps the screen hue family,
    so protection must start earlier and the model path becomes detail-safe.
    The middle band is interpolated to avoid sample-specific cliff behavior.
    """
    if screen_mode == "blue":
        # Blue screens overlap common purple/cyan game materials more often
        # than green screens. Sparse blue-family evidence is treated as possible
        # screen mixing, dominant same-family evidence gets material protection,
        # and the broad middle band interpolates so purple soft layers are not
        # stripped just because they are near the blue key family.
        low_risk = 0.08
        high_risk = 0.45
    else:
        low_risk = 0.015
        high_risk = 0.06
    if subject_key_color_risk <= low_risk:
        return 12.0, 28.0, "edge_cleanup"
    if subject_key_color_risk >= high_risk:
        return 6.0, 14.0, "key_color_material"
    t = float(np.clip((subject_key_color_risk - low_risk) / (high_risk - low_risk), 0.0, 1.0))
    bg_max = 12.0 * (1.0 - t) + 6.0 * t
    fg_min = 28.0 * (1.0 - t) + 14.0 * t
    return round(bg_max * 2.0) / 2.0, round(fg_min * 2.0) / 2.0, "balanced"


def _has_explicit_solid_key_material(
    solid_fraction: float,
    hard_density: float,
    compact_fill: float,
    compact_fraction: float,
) -> bool:
    """Return whether same-key-family color is likely real solid subject.

    This is the semantic gate for color protection. Solid UI/character material
    has near-key color anchored in opaque support; translucent ribbons, hair,
    smoke, and glass often have lots of near-key pixels but mostly in soft
    transitions. Those should be left to CorridorKey and foreground recovery,
    not rescued by a color-protection alpha floor.
    """
    hard_supported = (solid_fraction >= 0.25 and hard_density >= 0.08) or hard_density >= 0.12
    compact_same_color_region = compact_fill >= 0.85 and compact_fraction >= 0.35
    return bool(hard_supported or compact_same_color_region)


def _with_common_adjustments(
    settings: CorridorKeyRecommendedSettings,
    *,
    small_component_risk: bool,
    purity_sigma: float,
) -> CorridorKeyRecommendedSettings:
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
    return settings


def _candidate_confidences(
    *,
    subject_key_color_risk: float,
    key_color_solid_fraction: float,
    key_color_hard_density: float,
    key_color_compact_fill: float,
    key_color_compact_fraction: float,
    key_transition_fraction: float,
) -> tuple[float, float, float]:
    dominant_compact_same_hue = (
        subject_key_color_risk >= 0.45 and key_color_compact_fill >= 0.85 and key_color_compact_fraction >= 0.35
    )
    moderate_compact_support = (
        0.80 * min(key_color_compact_fill / 0.85, key_color_compact_fraction / 0.35)
        if subject_key_color_risk >= 0.08
        else 0.0
    )
    hard_support = max(
        1.0 if dominant_compact_same_hue else 0.0,
        moderate_compact_support,
        key_color_hard_density / 0.12,
        min(key_color_solid_fraction / 0.25, key_color_hard_density / 0.08),
        # Compact near-key evidence by itself can be a 1px antialias/shadow
        # residue on hard UI. Require some hard-alpha support before treating a
        # compact same-family run as subject material.
        min(key_color_compact_fill / 0.85, key_color_compact_fraction / 0.35, key_color_hard_density / 0.04),
    )
    material_confidence = float(np.clip(0.20 + 0.80 * hard_support, 0.0, 1.0))
    translucent_confidence = float(
        np.clip(
            0.30
            + 0.35 * (1.0 - np.clip(key_color_solid_fraction / 0.25, 0.0, 1.0))
            + 0.20 * np.clip(key_transition_fraction / 0.16, 0.0, 1.0)
            + 0.15 * (1.0 - np.clip(key_color_compact_fraction / 0.35, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    cleanup_confidence = float(np.clip(0.90 - subject_key_color_risk * 2.0, 0.35, 0.92))
    return material_confidence, translucent_confidence, cleanup_confidence


def _opaque_hard_ui_profile_candidates(
    *,
    image_aspect_ratio: float,
    screen_mode: str,
    subject_key_color_risk: float,
    key_color_solid_fraction: float,
    key_color_hard_density: float,
    key_color_compact_fill: float,
    key_color_compact_fraction: float,
    key_transition_fraction: float,
) -> list[CorridorKeyDecisionCandidate]:
    """Return hard opaque UI shadow-profile candidates from screen evidence.

    The split is intentionally semantic: no-shadow, hard-shadow, and soft-shadow
    buttons need different color-protection behavior even though all are opaque
    UI. Current aggregate color/alpha features cannot yet separate every
    translucent C-button collision, so local-diffusion recognition must later
    outrank these candidates.
    """
    candidates: list[CorridorKeyDecisionCandidate] = []
    # Current opaque UI shadow profiles are calibrated for wide button-like
    # assets in the semantic sample matrix. Square icons/characters can share
    # the same color statistics, so they must not be captured by this button
    # route until we add stronger component-geometry evidence.
    if image_aspect_ratio < 1.45:
        return candidates

    hard_shape = key_transition_fraction <= 0.025

    # B001-style no-shadow hard UI: only thin near-key edge residue is present.
    # Keep this gate on green for now because blue no-shadow translucent buttons
    # overlap the same aggregate statistics.
    no_shadow = (
        screen_mode == "green"
        and subject_key_color_risk <= 0.035
        and hard_shape
        and key_color_hard_density <= 0.02
    )
    if no_shadow:
        transition_bonus = 0.04 * float(np.clip((0.025 - key_transition_fraction) / 0.025, 0.0, 1.0))
        hard_density_bonus = 0.02 * float(np.clip((0.02 - key_color_hard_density) / 0.02, 0.0, 1.0))
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="opaque_hard_ui_no_shadow",
                label="硬边不透明 UI · 无阴影",
                confidence=float(np.clip(0.94 + transition_bonus + hard_density_bonus, 0.0, 0.99)),
                settings=CorridorKeyRecommendedSettings(),
                reason="Hard UI has low near-key residue and no measurable hard shadow band.",
            )
        )

    hard_shadow = (
        hard_shape
        and 0.08 <= subject_key_color_risk <= 0.25
        and key_color_hard_density >= 0.08
        and key_color_solid_fraction >= 0.70
    )
    if hard_shadow:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="opaque_hard_ui_hard_shadow",
                label="硬边不透明 UI · 硬阴影",
                confidence=1.0,
                settings=CorridorKeyRecommendedSettings(),
                reason="Near-key evidence is a hard, low-transition shadow band outside an opaque UI subject.",
            )
        )

    soft_shadow = (
        key_transition_fraction >= 0.08
        and 0.08 <= subject_key_color_risk <= 0.40
        and key_color_compact_fraction >= 0.90
    )
    if soft_shadow:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="opaque_hard_ui_soft_shadow",
                label="硬边不透明 UI · 软阴影",
                confidence=1.0,
                settings=CorridorKeyRecommendedSettings(),
                reason="Near-key ownership is broad and soft around an otherwise opaque UI subject.",
            )
        )

    return candidates


def _decision_candidates(
    *,
    image_aspect_ratio: float,
    image_long_side: int,
    foreground_aspect_ratio: float | None,
    foreground_long_side: int,
    screen_mode: str,
    subject_key_color_risk: float,
    key_color_solid_fraction: float,
    key_color_hard_density: float,
    key_color_compact_fill: float,
    key_color_compact_fraction: float,
    key_transition_fraction: float,
    same_key_opaque_plateau_confidence: float,
    hard_screen_residue_risk: float,
    small_component_risk: bool,
    purity_sigma: float,
) -> list[CorridorKeyDecisionCandidate]:
    semantic_candidates = _semantic_decision_candidates(
        image_aspect_ratio=image_aspect_ratio,
        image_long_side=image_long_side,
        foreground_aspect_ratio=foreground_aspect_ratio,
        foreground_long_side=foreground_long_side,
        screen_mode=screen_mode,
        subject_key_color_risk=subject_key_color_risk,
        key_color_solid_fraction=key_color_solid_fraction,
        key_color_hard_density=key_color_hard_density,
        key_color_compact_fill=key_color_compact_fill,
        key_color_compact_fraction=key_color_compact_fraction,
        key_transition_fraction=key_transition_fraction,
        same_key_opaque_plateau_confidence=same_key_opaque_plateau_confidence,
    )
    selected_index = int(np.argmax([candidate.confidence for candidate in semantic_candidates]))
    return [
        CorridorKeyDecisionCandidate(
            profile=candidate.profile,
            label=candidate.label,
            confidence=candidate.confidence,
            settings=_settings_for_profile(
                profile=candidate.profile,
                confidence=candidate.confidence,
                screen_mode=screen_mode,
                subject_key_color_risk=subject_key_color_risk,
                key_color_solid_fraction=key_color_solid_fraction,
                key_color_hard_density=key_color_hard_density,
                key_color_compact_fill=key_color_compact_fill,
                key_color_compact_fraction=key_color_compact_fraction,
                key_transition_fraction=key_transition_fraction,
                hard_screen_residue_risk=hard_screen_residue_risk,
                small_component_risk=small_component_risk,
                purity_sigma=purity_sigma,
            ),
            reason=candidate.reason,
            selected=index == selected_index,
        )
        for index, candidate in enumerate(semantic_candidates)
    ]


def _semantic_decision_candidates(
    *,
    image_aspect_ratio: float,
    image_long_side: int,
    foreground_aspect_ratio: float | None,
    foreground_long_side: int,
    screen_mode: str,
    subject_key_color_risk: float,
    key_color_solid_fraction: float,
    key_color_hard_density: float,
    key_color_compact_fill: float,
    key_color_compact_fraction: float,
    key_transition_fraction: float,
    same_key_opaque_plateau_confidence: float,
) -> list[CorridorKeyDecisionCandidate]:
    """Stage 1: choose semantic CorridorKey route, independent of tuning.

    The candidates here are intentionally based only on observable ownership
    semantics: whether same-key-family pixels look like solid subject material,
    screen-tinted translucency, or ordinary edge cleanup. Concrete CorridorKey
    strengths and color-protection thresholds are assigned later by
    _settings_for_profile(), so path-recognition tests can fail independently
    from parameter-tuning tests.
    """
    bg_max, fg_min, profile_from_distance = _color_protection_endpoints(subject_key_color_risk, screen_mode)
    del bg_max, fg_min
    material_conf, translucent_conf, cleanup_conf = _candidate_confidences(
        subject_key_color_risk=subject_key_color_risk,
        key_color_solid_fraction=key_color_solid_fraction,
        key_color_hard_density=key_color_hard_density,
        key_color_compact_fill=key_color_compact_fill,
        key_color_compact_fraction=key_color_compact_fraction,
        key_transition_fraction=key_transition_fraction,
    )
    subject_aspect = foreground_aspect_ratio if foreground_aspect_ratio is not None else image_aspect_ratio
    subject_long_side = foreground_long_side if foreground_long_side > 0 else image_long_side
    opaque_hard_ui_candidates = _opaque_hard_ui_profile_candidates(
        image_aspect_ratio=subject_aspect,
        screen_mode=screen_mode,
        subject_key_color_risk=subject_key_color_risk,
        key_color_solid_fraction=key_color_solid_fraction,
        key_color_hard_density=key_color_hard_density,
        key_color_compact_fill=key_color_compact_fill,
        key_color_compact_fraction=key_color_compact_fraction,
        key_transition_fraction=key_transition_fraction,
    )
    candidates: list[CorridorKeyDecisionCandidate] = []
    candidates.extend(opaque_hard_ui_candidates)

    # Size is only a coarse eligibility gate. The semantic claim is "character",
    # so the observed foreground support must also be roughly character-shaped;
    # otherwise wide buttons exported on square canvases are misrouted.
    composite_character = (
        0.75 <= image_aspect_ratio <= 1.35
        and 0.55 <= subject_aspect <= 1.45
        and subject_long_side >= 512
    )
    if composite_character:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="composite_character_corridor_only",
                label="复合角色 · 交给 CorridorKey",
                confidence=1.0,
                settings=CorridorKeyRecommendedSettings(),
                reason=(
                    "Large square character asset can mix opaque body, hair, glow, and translucent material; "
                    "avoid coarse color-protection ownership and let CorridorKey solve the matte."
                ),
            )
        )

    # Require dominant same-key ownership before calling this "opaque material":
    # lower-risk hard plateaus are usually shadow/antialias residue, not the
    # subject's main fill. The 0.85 plateau score then requires a coherent
    # component with enough alpha>=0.75 support to reject all-ramp glass/tint.
    if subject_key_color_risk >= 0.45 and same_key_opaque_plateau_confidence >= 0.85:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="opaque_hard_ui_same_key_plateau",
                label="同幕色实体 UI",
                confidence=same_key_opaque_plateau_confidence,
                settings=CorridorKeyRecommendedSettings(),
                reason=(
                    "Near-screen color forms a coherent hard-supported plateau; "
                    "treat it as opaque UI material rather than screen-tinted translucency."
                ),
            )
        )

    translucent_button = (
        same_key_opaque_plateau_confidence < 0.85
        and subject_aspect >= 1.45
        and subject_key_color_risk >= 0.25
        and key_transition_fraction >= 0.30
    )
    if translucent_button:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="translucent_button",
                label="半透明按钮",
                confidence=1.0,
                settings=CorridorKeyRecommendedSettings(),
                reason=(
                    "Wide button has broad near-screen transition ownership; "
                    "color protection is unreliable for globally translucent material."
                ),
            )
        )

    candidates.append(
        CorridorKeyDecisionCandidate(
            profile="edge_cleanup",
            label="清理近幕色边缘",
            confidence=cleanup_conf,
            settings=CorridorKeyRecommendedSettings(),
            reason="Near-key evidence is weak or ambiguous; keep color protection narrow and let CorridorKey own soft edges.",
        )
    )

    if subject_key_color_risk >= 0.04:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile="screen_tinted_translucency",
                label="半透明层交给 CorridorKey",
                confidence=translucent_conf,
                settings=CorridorKeyRecommendedSettings(),
                reason="Near-key color is mostly in soft/transparent support; avoid alpha-floor protection.",
            )
        )

    if subject_key_color_risk >= 0.015:
        candidates.append(
            CorridorKeyDecisionCandidate(
                profile=profile_from_distance,
                label="保护主体实色",
                confidence=material_conf,
                settings=CorridorKeyRecommendedSettings(),
                reason="Near-key color has hard/compact subject-material evidence.",
            )
        )
    return candidates


def _settings_for_profile(
    *,
    profile: str,
    confidence: float,
    screen_mode: str,
    subject_key_color_risk: float,
    key_color_solid_fraction: float,
    key_color_hard_density: float,
    key_color_compact_fill: float,
    key_color_compact_fraction: float,
    key_transition_fraction: float,
    hard_screen_residue_risk: float,
    small_component_risk: bool,
    purity_sigma: float,
) -> CorridorKeyRecommendedSettings:
    """Stage 2: tune CorridorKey aggressively only after route semantics are clear.

    The split is deliberate: Stage 1 answers "which path is this?" while this
    stage answers "how hard should that path push?". Clear solid subject color
    gets stronger protection, clear translucent screen tint disables protection,
    and ambiguous balanced cases stay conservative.
    """
    del key_transition_fraction
    explicit_solid_material = _has_explicit_solid_key_material(
        key_color_solid_fraction,
        key_color_hard_density,
        key_color_compact_fill,
        key_color_compact_fraction,
    )

    if profile == "opaque_hard_ui_no_shadow":
        # Hard opaque UI should clean screen residue more assertively than the
        # negative edge-cleanup route, but this remains parameter-only: it does
        # not add an opaque-interior alpha snap or other execution repair.
        del confidence
        settings = CorridorKeyRecommendedSettings(
            despill_strength=1.0,
            refiner_strength=1.15,
            auto_despeckle="On",
            despeckle_size=400,
            color_protection=True,
            protection_bg_max=8.0,
            protection_fg_min=18.0,
        )
    elif profile == "opaque_hard_ui_same_key_plateau":
        del confidence
        # Same-screen-family material is the opposite of translucent tint: keep
        # color protection on and route-level logic will hand the case to
        # Known-B when a stable background exists.
        settings = CorridorKeyRecommendedSettings(
            despill_strength=1.0,
            refiner_strength=1.15,
            auto_despeckle="On",
            despeckle_size=400,
            color_protection=True,
            protection_bg_max=8.0,
            protection_fg_min=18.0,
        )
    elif profile == "opaque_hard_ui_hard_shadow":
        del confidence
        # Hard shadows are near-key evidence outside the opaque subject. Avoid
        # the hard-edge residue boost here: pushing refiner to the maximum can
        # fragment the already-detected cast shadow, causing the fallback
        # shadow patch to replace a mostly preserved shadow with an incomplete
        # measured component. Keep cleanup assertive but shadow-preserving.
        settings = CorridorKeyRecommendedSettings(
            despill_strength=1.0,
            refiner_strength=1.15,
            auto_despeckle="On",
            despeckle_size=400,
            color_protection=True,
            protection_bg_max=8.0,
            protection_fg_min=18.0,
        )
    elif profile == "opaque_hard_ui_soft_shadow":
        del confidence
        # Soft shadows need the subject protected, but broad near-key soft
        # ownership should remain available to the shadow path instead of being
        # rescued as solid material.
        settings = CorridorKeyRecommendedSettings(
            despill_strength=1.0,
            refiner_strength=1.15,
            auto_despeckle="On",
            despeckle_size=400,
            color_protection=True,
            protection_bg_max=8.0,
            protection_fg_min=18.0,
        )
    elif profile == "edge_cleanup":
        del confidence
        # Edge cleanup is a negative semantic claim: it says we did not find
        # key-family subject material. That is not the same as explicitly
        # proving what the subject is, so Stage 2 stays stable instead of
        # pushing refiner/protection hard and risking loss of ordinary details.
        settings = CorridorKeyRecommendedSettings(
            color_protection=True,
            protection_bg_max=12.0,
            protection_fg_min=28.0,
        )
    elif profile == "screen_tinted_translucency":
        # Translucent tint is not protected by alpha floor. A confident route
        # pushes CorridorKey/refinement harder because the near-key color is
        # judged to be screen contamination rather than solid subject paint.
        settings = CorridorKeyRecommendedSettings(
            refiner_strength=1.15 if confidence >= 0.75 else 1.0,
            auto_despeckle="Off",
            despeckle_size=64,
            color_protection=False,
            protection_bg_max=6.0,
            protection_fg_min=14.0,
        )
    elif profile == "translucent_button":
        del confidence
        # Whole-button translucency mixes subject and screen color throughout
        # the material. A color-protection alpha floor has no stable foreground
        # color to protect there, so it turns screen tint into dirty subject
        # residue. Let CorridorKey own the translucent subject and let the
        # source-reprojection shadow repair recover any cast shadow underneath.
        settings = CorridorKeyRecommendedSettings(
            despill_strength=1.0,
            refiner_strength=1.15,
            auto_despeckle="Off",
            despeckle_size=64,
            color_protection=False,
            protection_bg_max=6.0,
            protection_fg_min=14.0,
        )
    elif profile == "composite_character_corridor_only":
        del confidence
        # Composite characters can contain opaque armor/skin, hair strands,
        # glow, fabric, and transparent accessories in the same connected
        # subject. A color-protection floor is not smart enough to own those
        # materials safely; use this profile as a clean CorridorKey-capability
        # experiment with a full-frame hint and no color protection.
        settings = CorridorKeyRecommendedSettings(
            despill_strength=1.0,
            refiner_strength=1.0,
            auto_despeckle="Off",
            despeckle_size=64,
            color_protection=False,
            protection_bg_max=6.0,
            protection_fg_min=14.0,
        )
    elif profile == "key_color_material":
        # Once same-key hue is anchored in hard/compact subject support, protect
        # it aggressively; this is the opposite of the translucent path.
        aggressive = confidence >= 0.85 and explicit_solid_material
        settings = CorridorKeyRecommendedSettings(
            despill_strength=0.45 if aggressive else 0.65,
            refiner_strength=0.70 if aggressive else 0.85,
            color_protection=True,
            protection_bg_max=4.0 if aggressive else 6.0,
            protection_fg_min=10.0 if aggressive else 14.0,
        )
    elif profile == "balanced":
        bg_max, fg_min, _ = _color_protection_endpoints(subject_key_color_risk, screen_mode)
        settings = CorridorKeyRecommendedSettings(
            color_protection=True,
            protection_bg_max=bg_max,
            protection_fg_min=fg_min,
        )
    else:
        settings = CorridorKeyRecommendedSettings()

    if hard_screen_residue_risk >= 0.04 and profile not in {
        "screen_tinted_translucency",
        "translucent_button",
        "composite_character_corridor_only",
        "opaque_hard_ui_hard_shadow",
    }:
        settings = CorridorKeyRecommendedSettings(
            gamma_space=settings.gamma_space,
            despill_strength=settings.despill_strength,
            refiner_strength=max(settings.refiner_strength, 1.5),
            auto_despeckle=settings.auto_despeckle,
            despeckle_size=settings.despeckle_size,
            color_protection=settings.color_protection,
            protection_bg_max=settings.protection_bg_max,
            protection_fg_min=settings.protection_fg_min,
        )
    return _with_common_adjustments(
        settings,
        small_component_risk=small_component_risk,
        purity_sigma=purity_sigma,
    )


def _recommend_settings(
    *,
    preset: str,
    image_aspect_ratio: float,
    image_long_side: int,
    foreground_aspect_ratio: float | None,
    foreground_long_side: int,
    screen_mode: str,
    subject_key_color_risk: float,
    key_color_solid_fraction: float,
    key_color_hard_density: float,
    key_color_compact_fill: float,
    key_color_compact_fraction: float,
    key_transition_fraction: float,
    same_key_opaque_plateau_confidence: float,
    hard_screen_residue_risk: float,
    small_component_risk: bool,
    purity_sigma: float,
) -> tuple[CorridorKeyRecommendedSettings, str, list[CorridorKeyDecisionCandidate], list[str]]:
    notes: list[str] = []
    if preset == "manual":
        candidate = CorridorKeyDecisionCandidate(
            profile="manual",
            label="手动参数",
            confidence=1.0,
            settings=CorridorKeyRecommendedSettings(),
            reason="Manual preset leaves caller-provided settings unchanged.",
            selected=True,
        )
        return candidate.settings, candidate.profile, [candidate], ["manual preset leaves caller-provided settings unchanged"]
    if preset == "spill_safe":
        candidate = CorridorKeyDecisionCandidate(
            profile="spill_safe",
            label="强去溢色",
            confidence=1.0,
            settings=CorridorKeyRecommendedSettings(
                despill_strength=1.0,
                refiner_strength=1.15,
                auto_despeckle="On",
                despeckle_size=400,
                protection_bg_max=8.0,
                protection_fg_min=18.0,
            ),
            reason="Preset favors stronger screen-spill cleanup.",
            selected=True,
        )
        return candidate.settings, candidate.profile, [candidate], ["spill_safe preset favors stronger green-spill cleanup"]
    if preset == "detail_safe":
        candidate = CorridorKeyDecisionCandidate(
            profile="detail_safe",
            label="细节保护",
            confidence=1.0,
            settings=CorridorKeyRecommendedSettings(
                despill_strength=0.65,
                refiner_strength=0.85,
                auto_despeckle="Off",
                despeckle_size=64,
                protection_bg_max=6.0,
                protection_fg_min=14.0,
            ),
            reason="Preset protects small ornaments and key-color subject material.",
            selected=True,
        )
        return candidate.settings, candidate.profile, [candidate], ["detail_safe preset protects small ornaments and key-color subject material"]

    candidates = _decision_candidates(
        image_aspect_ratio=image_aspect_ratio,
        image_long_side=image_long_side,
        foreground_aspect_ratio=foreground_aspect_ratio,
        foreground_long_side=foreground_long_side,
        screen_mode=screen_mode,
        subject_key_color_risk=subject_key_color_risk,
        key_color_solid_fraction=key_color_solid_fraction,
        key_color_hard_density=key_color_hard_density,
        key_color_compact_fill=key_color_compact_fill,
        key_color_compact_fraction=key_color_compact_fraction,
        key_transition_fraction=key_transition_fraction,
        same_key_opaque_plateau_confidence=same_key_opaque_plateau_confidence,
        hard_screen_residue_risk=hard_screen_residue_risk,
        small_component_risk=small_component_risk,
        purity_sigma=purity_sigma,
    )
    selected = next(candidate for candidate in candidates if candidate.selected)
    settings = selected.settings
    parameter_profile = selected.profile
    notes.append(f"stage1 selected {selected.profile} semantic path at confidence {selected.confidence:.2f}")
    if parameter_profile == "key_color_material":
        notes.append("stage2 key-color material tuning reduced despill/refiner and tightened color protection")
    if hard_screen_residue_risk >= 0.04 and parameter_profile != "screen_tinted_translucency":
        notes.append("stage2 hard screen-family residue tuning increased refiner strength")
    if parameter_profile == "opaque_hard_ui_no_shadow":
        notes.append("stage2 opaque-hard-UI no-shadow tuning uses assertive cleanup parameters without alpha snapping")
    elif parameter_profile == "opaque_hard_ui_same_key_plateau":
        notes.append("stage2 same-key opaque plateau tuning keeps the deterministic hard-UI path")
    elif parameter_profile == "opaque_hard_ui_hard_shadow":
        notes.append("stage2 opaque-hard-UI hard-shadow tuning uses conservative color protection for the shadow band")
    elif parameter_profile == "opaque_hard_ui_soft_shadow":
        notes.append("stage2 opaque-hard-UI soft-shadow tuning keeps broad soft ownership out of material protection")
    elif parameter_profile == "edge_cleanup":
        notes.append("stage2 edge-cleanup tuning narrowed color protection away from near-screen fringe")
    elif parameter_profile == "balanced":
        notes.append("stage2 balanced tuning interpolated color protection from same-hue foreground risk")
    elif parameter_profile == "screen_tinted_translucency":
        notes.append("stage2 translucent tuning disabled color protection")
    elif parameter_profile == "translucent_button":
        notes.append("stage2 translucent-button tuning disabled color protection")
    elif parameter_profile == "composite_character_corridor_only":
        notes.append("stage2 composite-character corridor-only tuning disabled color protection and automatic hint constraints")
    if small_component_risk:
        notes.append("small UI components disabled auto despeckle")
    if purity_sigma >= 8.0:
        notes.append("low screen purity reduced aggressive refinement")
    if not notes:
        notes.append("standard settings selected for clean screen and low subject key-color risk")
    return settings, parameter_profile, candidates, notes


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
    foreground_bbox, foreground_aspect, foreground_long_side = _foreground_geometry(key_alpha)
    small_component_risk, small_count = _foreground_component_stats(key_alpha)
    if small_component_risk:
        notes.append(f"detected {small_count} small foreground components")

    subject_risk, solid_fraction, hard_density, compact_fill, compact_fraction = _key_color_material_stats(
        image_srgb,
        background_color,
        key_alpha,
    )
    transition_fraction = _key_transition_fraction(key_alpha)
    same_key_plateau_confidence = _same_key_opaque_plateau_confidence(
        image_srgb,
        background_color,
        key_alpha,
    )
    hard_residue_risk = _hard_screen_residue_risk(subject_risk, transition_fraction)
    settings, parameter_profile, decision_candidates, setting_notes = _recommend_settings(
        preset=preset,
        image_aspect_ratio=float(image_srgb.shape[1] / max(1, image_srgb.shape[0])),
        image_long_side=int(max(image_srgb.shape[:2])),
        foreground_aspect_ratio=foreground_aspect,
        foreground_long_side=foreground_long_side,
        screen_mode=selected_mode,
        subject_key_color_risk=subject_risk,
        key_color_solid_fraction=solid_fraction,
        key_color_hard_density=hard_density,
        key_color_compact_fill=compact_fill,
        key_color_compact_fraction=compact_fraction,
        key_transition_fraction=transition_fraction,
        same_key_opaque_plateau_confidence=same_key_plateau_confidence,
        hard_screen_residue_risk=hard_residue_risk,
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
        key_color_solid_fraction=solid_fraction,
        key_color_hard_density=hard_density,
        key_color_compact_fill=compact_fill,
        key_color_compact_fraction=compact_fraction,
        key_transition_fraction=transition_fraction,
        same_key_opaque_plateau_confidence=same_key_plateau_confidence,
        foreground_bbox_xyxy=foreground_bbox,
        foreground_aspect_ratio=foreground_aspect,
        foreground_long_side=foreground_long_side,
        hard_screen_residue_risk=hard_residue_risk,
        small_component_risk=small_component_risk,
        parameter_profile=parameter_profile,
        decision_candidates=decision_candidates,
        recommended_settings=settings,
        notes=notes,
    )


__all__ = [
    "BLUE_SCREEN_RGB",
    "GREEN_SCREEN_RGB",
    "CorridorKeyAssetAnalysis",
    "CorridorKeyDecisionCandidate",
    "CorridorKeyRecommendedSettings",
    "corridorkey_analyze_asset",
]
