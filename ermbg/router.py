"""Front-end router: classify the input and pick a matting strategy.

The matting pipeline can take many shapes — chromatic key on saturated B,
luminance key on white/black, no-op pass-through when the input is already
transparent, etc. Rather than scatter ``if`` checks through ``matte()``, we
do one classification pass up-front and produce a ``Strategy`` that the
pipeline executes mechanically.

Two attributes drive the routing:

  ``bg_type``     describes the observed background.
      "rgba_passthrough" — input already has a usable alpha channel, no work needed
      "saturated"        — high-chroma B (green/cyan/magenta) → chromatic key + unmix
      "white"            — bright low-chroma B → luminance key + unmix
      "black"            — dark low-chroma B → luminance key + unmix
      "grey"             — mid-lightness low-chroma B → luminance key (weak signal)
      "noisy"            — purity_sigma over threshold → fall back to matting net only
      "unknown"          — couldn't decide

  ``image_type``  describes subject content. Coarse: "graphic" (vector / cartoon /
      logo, hard edges, flat color) vs "photo" (photographic / natural).
      Graphics get tighter keyer thresholds and prefer α binarization.

The output ``Strategy`` is a frozen recipe with all the knobs the pipeline
needs. New strategies are added by extending the ``classify_strategy`` table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .keyer import KeyerThresholds, chromatic_key_alpha


# Thresholds for classification (all in OKLab ΔE-style units).
_BG_CHROMA_SATURATED = 8.0     # bg_chroma >= this → "saturated"
_BG_LIGHTNESS_WHITE = 85.0     # bg_L * 100 >= this → "white"
_BG_LIGHTNESS_BLACK = 15.0     # bg_L * 100 <= this → "black"
_PASSTHROUGH_ALPHA_FRAC = 0.05 # source α has at least this fraction of
                               # not-fully-opaque pixels → already keyed
_GRAPHIC_FLAT_FRAC = 0.6       # >= this fraction of subject pixels share
                               # a color exactly → likely vector/cartoon


@dataclass(frozen=True)
class Strategy:
    """Frozen recipe describing how to matte a single image."""

    name: str
    bg_type: str
    image_type: str
    keyer_mode: str | None       # "chromatic" | "luminance" | None
    keyer_thresholds: KeyerThresholds | None
    despill: str                 # "auto" | "unmix" | "chroma_cap" | "local_borrow" | "closed_form" | "none"
    use_keyer_merge: bool        # whether to patch missing components from key α
    use_keyer_gate: bool = False # whether to cap matting α by key α where keyer is bg-confident
    passthrough: bool = False    # if True, skip matting net entirely
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    """Production auto-route selected by ERMBG.

    ``Strategy`` above is kept as the legacy matting-pipeline recipe. This
    route decision is the new public auto contract: ERMBG decides which concrete
    backend owns the image and which parameter family that backend should use.
    """

    route: str
    asset_kind: str
    backend: str
    params: dict[str, Any]
    confidence: float
    reasons: list[str]
    analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        corridorkey_analysis = self.analysis.get("corridorkey_analysis")
        parameter_profile = (
            corridorkey_analysis.get("parameter_profile")
            if isinstance(corridorkey_analysis, dict)
            else None
        )
        payload = {
            "requested_backend": "auto",
            "requested_algorithm": "auto",
            "route": self.route,
            "algorithm": self.backend,
            "asset_kind": self.asset_kind,
            "parameter_profile": parameter_profile,
            "execution_profile": self.params.get("execution_profile"),
            "params": self.params,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "analysis": self.analysis,
            # Compatibility for older Web/batch consumers.
            "reason": self.reasons[0] if self.reasons else "",
        }
        if isinstance(corridorkey_analysis, dict):
            payload["corridorkey_analysis"] = corridorkey_analysis
        return payload


# Hygiene thresholds — when an input arrives with an alpha channel, we still
# verify the matte is clean before deciding to pass it through. Otherwise we
# silently propagate a bad asset (white edges, banded α, residual bg in RGB)
# downstream. All thresholds in OKLab ΔE units.
_HYGIENE_FRINGE_DE_MAX = 16.0       # edge-band ΔE vs interior subject
_HYGIENE_LOW_ALPHA_RES_MAX = 14.0   # mean OKLab |C - F_interior| at α≈0
_HYGIENE_BIMODAL_MAX = 0.985        # fraction of α exactly 0 or 1


@dataclass(frozen=True)
class AlphaHygiene:
    """Quality scores for an existing source α matte.

    All produced **without ground truth**: the only inputs are the RGBA itself.

    Each score is "lower is cleaner" (except ``bimodal_fraction`` which is
    direct: too high → matte is hard-binarized, no soft edges, RGBA likely
    came from a coarse segmenter).

    ``clean`` is the AND of three checks; if false, the router will re-matte
    instead of passing the source α through.
    """

    fringe_dE: float
    low_alpha_residual: float
    bimodal_fraction: float
    has_soft_edge: bool
    clean: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fringe_dE": self.fringe_dE,
            "low_alpha_residual": self.low_alpha_residual,
            "bimodal_fraction": self.bimodal_fraction,
            "has_soft_edge": self.has_soft_edge,
            "clean": self.clean,
            "reason": self.reason,
        }


def _interior_color(image_srgb: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
    """Median OKLab color over fully-opaque interior pixels, eroded so edge
    pollution doesn't bias the reference. Returns None when no usable
    interior exists.

    For premultiplied RGBA the opaque interior has α≈1 already, so RGB=F
    directly. For straight RGBA same. Either way we read the median sRGB
    color and convert to OKLab.
    """
    binary = (alpha >= 0.95).astype(np.uint8)
    interior = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=4)
    if interior.sum() < 16:
        interior = binary
        if interior.sum() < 8:
            return None
    pixels = image_srgb[interior > 0]
    median = np.median(pixels, axis=0).astype(np.uint8)
    return srgb_to_oklab(median.reshape(1, 1, 3)).reshape(3)


def assess_source_alpha(image_srgb: np.ndarray, alpha: np.ndarray) -> AlphaHygiene:
    """Decide whether a source RGBA's α is clean enough to pass through.

    Three unsupervised checks:

      1. Fringe ΔE — for pixels in α∈(0.05, 0.6] (the soft edge band), recover
         the implied foreground color using both straight (F=C) and premultiplied
         (F=C/α) interpretations and compare to the opaque interior color.
         A clean matte yields one of the two interpretations close to interior;
         a halo-y matte yields neither close, in OKLab.

      2. Low-α residual — for α≈0 pixels, classify each pixel as either
         "premultiplied (RGB≈0)" or "straight (RGB≈some constant bg)". A
         leak of *interior* color into transparent regions is suspicious;
         constant non-interior bg color also is (means the asset stored
         the original bg). We measure the second case via std-dev of low-α
         RGB minus distance-to-interior.

      3. Bimodal fraction — how much of α is at the 0/1 extremes? If almost
         everything is 0 or 1, the matte was hard-binarized — re-matting
         can recover anti-aliasing.

    Returns an ``AlphaHygiene`` with ``clean=True`` only if all checks pass.
    """
    from . import io as ermbg_io

    h, w = alpha.shape
    a = alpha.astype(np.float32)

    # --- (3) bimodal fraction
    extreme = ((a < 0.01) | (a > 0.99)).mean()
    has_soft = bool(((a > 0.05) & (a < 0.95)).any())

    interior_lab = _interior_color(image_srgb, a)
    if interior_lab is None:
        return AlphaHygiene(
            fringe_dE=float("inf"),
            low_alpha_residual=float("inf"),
            bimodal_fraction=float(extreme),
            has_soft_edge=has_soft,
            clean=False,
            reason="no usable opaque interior; cannot trust source α",
        )

    # --- (1) fringe ΔE on the soft edge band
    soft_band = (a > 0.05) & (a <= 0.6)
    if soft_band.sum() >= 32:
        # Recover an implied F under three interpretations and take the closest
        # to interior. This makes the check robust to sRGB-vs-linear premul
        # conventions and to straight-α encoding:
        #   straight:           F = C
        #   sRGB premul:        F = C / α        (in sRGB byte values)
        #   linear premul:      F = linearize(C)/α then back to sRGB
        a_band = a[soft_band, None]
        c_srgb = image_srgb[soft_band].astype(np.float32)
        c_lin = ermbg_io.srgb_to_linear(image_srgb)[soft_band]
        f_straight = c_srgb.astype(np.uint8)
        f_premul_srgb = np.clip(c_srgb / np.maximum(a_band, 1e-3), 0, 255).astype(np.uint8)
        f_premul_lin = ermbg_io.linear_to_srgb_u8(np.clip(c_lin / np.maximum(a_band, 1e-3), 0, 1))

        d_s = oklab_distance(srgb_to_oklab(f_straight), interior_lab)
        d_psrgb = oklab_distance(srgb_to_oklab(f_premul_srgb), interior_lab)
        d_plin = oklab_distance(srgb_to_oklab(f_premul_lin), interior_lab)
        # Per-pixel best of the three → mean.
        fringe = float(np.mean(np.minimum(np.minimum(d_s, d_psrgb), d_plin)))
    else:
        fringe = 0.0

    # --- (2) low-α residual
    very_low = a < 0.05
    if very_low.sum() >= 32:
        from .colorspace import linear_rgb_to_oklab

        low_rgb = ermbg_io.srgb_to_linear(image_srgb[very_low])
        low_std = float(np.mean(np.std(low_rgb, axis=0)))
        low_mean_lab = linear_rgb_to_oklab(np.mean(low_rgb, axis=0).reshape(1, 1, 3)).reshape(3)
        zero_lab = np.zeros(3, dtype=np.float32)
        d_to_zero = float(np.sqrt(np.sum((low_mean_lab - zero_lab) ** 2))) * 100.0
        d_to_interior = float(np.sqrt(np.sum((low_mean_lab - interior_lab) ** 2))) * 100.0
        # Three regimes:
        # (a) premultiplied (RGB=0): low_mean ≈ zero → d_to_zero small → low_res = 0
        # (b) interior-leaked (RGB ≈ F_interior): d_to_interior small but
        #     d_to_zero large → fail. Reported value = "how much interior leaked"
        #     scaled by inverse distance.
        # (c) original-bg-leaked (RGB ≈ original solid bg): d_to_zero large,
        #     d_to_interior also non-trivial → fail with magnitude d_to_zero
        #     (we don't know what bg was, but anything that isn't black is
        #     leaky for hygiene purposes).
        if d_to_zero <= 6.0:
            low_res = 0.0  # premul, fine
        elif d_to_interior < d_to_zero * 0.5:
            # Closer to interior than to zero → likely interior leak.
            # Penalty grows as d_to_interior shrinks (closer = worse).
            low_res = max(d_to_zero - d_to_interior, 20.0)
        else:
            low_res = d_to_zero  # generic non-zero transparent → also leak
        # Penalty for high spread (multiple bg colors in transparent regions).
        if low_std > 0.05:
            low_res = max(low_res, 100.0 * low_std)
    else:
        low_res = 0.0

    fringe_ok = fringe <= _HYGIENE_FRINGE_DE_MAX
    low_ok = low_res <= _HYGIENE_LOW_ALPHA_RES_MAX
    bimodal_ok = extreme <= _HYGIENE_BIMODAL_MAX or has_soft

    clean = bool(fringe_ok and low_ok and bimodal_ok)
    if clean:
        reason = ""
    else:
        bad = []
        if not fringe_ok:
            bad.append(f"fringe ΔE={fringe:.1f}>{_HYGIENE_FRINGE_DE_MAX}")
        if not low_ok:
            bad.append(f"low-α residual={low_res:.1f}>{_HYGIENE_LOW_ALPHA_RES_MAX}")
        if not bimodal_ok:
            bad.append(f"α binarized ({extreme*100:.1f}% at 0/1)")
        reason = "; ".join(bad)

    return AlphaHygiene(
        fringe_dE=fringe,
        low_alpha_residual=low_res,
        bimodal_fraction=float(extreme),
        has_soft_edge=has_soft,
        clean=clean,
        reason=reason,
    )


# --- classification helpers ------------------------------------------------


def _bg_lab_stats(image_srgb: np.ndarray, source_alpha: np.ndarray | None) -> tuple[float, float, float]:
    """Return (bg_lightness, bg_chroma, bg_purity_sigma_uint8) of the corner band.

    When ``source_alpha`` is provided, transparent regions are preferred for
    bg sampling — they aren't really "background" in the photographic sense.
    If the alpha is fully opaque (no transparent pixels in the corner band),
    we fall back to using the corner band directly.
    """
    h, w = image_srgb.shape[:2]
    band = max(4, int(0.04 * min(h, w)))
    edge = np.zeros((h, w), dtype=bool)
    edge[:band, :] = True
    edge[-band:, :] = True
    edge[:, :band] = True
    edge[:, -band:] = True
    if source_alpha is not None:
        masked = edge & (source_alpha < 0.05)
        if masked.sum() >= 16:
            edge = masked
    if edge.sum() < 16:
        return 50.0, 0.0, 0.0

    pixels = image_srgb[edge]
    edge_sigma = float(np.std(pixels.astype(np.float32), axis=0).mean())
    if source_alpha is None and edge_sigma > 18.0:
        corner_pixels = _stable_corner_bg_pixels(image_srgb)
        if corner_pixels is not None:
            pixels = corner_pixels
    median = np.median(pixels, axis=0).astype(np.uint8)
    sigma = float(np.std(pixels.astype(np.float32), axis=0).mean())
    lab = srgb_to_oklab(median.reshape(1, 1, 3)).reshape(3)
    L = float(lab[0]) * 100.0
    C = float(np.sqrt(lab[1] ** 2 + lab[2] ** 2)) * 100.0
    return L, C, sigma


def _stable_corner_bg_pixels(image_srgb: np.ndarray) -> np.ndarray | None:
    """Return corner samples when they are a cleaner bg estimate than the edge.

    Real game/UI icons often fill most of a tiny sprite and touch the image
    border, so a whole-edge sample mixes subject rim colors into the measured
    background. These empirical gates key on a different signal: all four
    corner patches agree with each other and are internally low-variance. That
    protects small corner-visible solid-background assets without treating a
    noisy photo edge as a keyable screen.
    """
    h, w = image_srgb.shape[:2]
    size = max(2, min(8, int(round(min(h, w) * 0.06))))
    if h < size * 2 or w < size * 2:
        return None
    patches = [
        image_srgb[:size, :size],
        image_srgb[:size, -size:],
        image_srgb[-size:, :size],
        image_srgb[-size:, -size:],
    ]
    medians = np.asarray([np.median(p.reshape(-1, 3), axis=0) for p in patches], dtype=np.float32)
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    corner_agreement = float(np.std(medians, axis=0).mean())
    corner_sigma = float(np.std(pixels.astype(np.float32), axis=0).mean())
    if corner_agreement <= 6.0 and corner_sigma <= 10.0:
        return pixels
    return None


def _detect_image_type(image_srgb: np.ndarray) -> str:
    """Coarse 'graphic' vs 'photo' detector.

    Heuristic: graphics tend to have a small color palette (lots of pixels
    sharing the exact same RGB triplet). Photos almost never do because of
    sensor noise. We look at the top-K most common colors covering > F of
    pixels.
    """
    small = cv2.resize(image_srgb, (128, 128), interpolation=cv2.INTER_AREA)
    flat = small.reshape(-1, 3)
    # quantize to 5-bit per channel so near-identical color shades count.
    q = (flat >> 3).astype(np.int32)
    keys = q[:, 0] * 32 * 32 + q[:, 1] * 32 + q[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    counts.sort()
    # fraction covered by the 8 most common quantized colors
    top = counts[-8:].sum() / counts.sum()
    return "graphic" if top >= _GRAPHIC_FLAT_FRAC else "photo"


def _is_passthrough(source_alpha: np.ndarray | None) -> bool:
    """True when source already carries a usable alpha matte."""
    if source_alpha is None:
        return False
    not_opaque = source_alpha < 0.99
    return bool(not_opaque.mean() >= _PASSTHROUGH_ALPHA_FRAC)


# --- main entrypoint -------------------------------------------------------


def classify_strategy(
    image_srgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
) -> Strategy:
    """Inspect the input and return the strategy that should be applied.

    Args:
        image_srgb: H×W×3 uint8 sRGB. For RGBA inputs, this is the un-composited
            color channel (caller obtains it from ``io.load_image_with_alpha``).
        source_alpha: H×W float32 [0,1] alpha from the source file, or None.

    Returns:
        A frozen ``Strategy`` describing keyer mode / despill / passthrough flags.

    RGBA inputs go through an unsupervised hygiene check before being passed
    through. Only clean source α is reused; halo-y / banded / leaky α gets
    re-matted (the strategy continues into normal saturated/white/black logic).
    """
    extras: dict[str, Any] = {}

    if _is_passthrough(source_alpha):
        hygiene = assess_source_alpha(image_srgb, source_alpha)
        extras["hygiene"] = hygiene.to_dict()
        if hygiene.clean:
            return Strategy(
                name="rgba_passthrough",
                bg_type="rgba_passthrough",
                image_type="any",
                keyer_mode=None,
                keyer_thresholds=None,
                despill="none",
                use_keyer_merge=False,
                passthrough=True,
                notes="Source α is clean; copy through unchanged.",
                extras=extras,
            )
        # Source has α but it's dirty — fall through and re-matte. The
        # source α is discarded; the caller is expected to re-composite
        # the RGB onto a known background before invoking the matting net.
        extras["passthrough_rejected"] = hygiene.reason

    L, C, sigma = _bg_lab_stats(image_srgb, source_alpha)
    image_type = _detect_image_type(image_srgb)
    extras.update({"bg_L": L, "bg_C": C, "bg_sigma": sigma, "image_type": image_type})

    # Tight thresholds for graphics (hard edges, expected to be clean), wider
    # for photos (anti-aliased / hairy edges, allow more soft-edge slack).
    if image_type == "graphic":
        thr = KeyerThresholds(bg_max=4.0, fg_min=14.0)
        # Hard-edged graphics get the gate: keyer veto on confident-bg pixels
        # to remove BiRefNet's wide soft halo on solid backgrounds. Photos do
        # not — there, soft edges (hair, fur) need to survive even though
        # the keyer would call those pixels "bg colored".
        gate_default = True
    else:
        thr = KeyerThresholds(bg_max=6.0, fg_min=22.0)
        gate_default = False

    if sigma > 18.0:
        return Strategy(
            name="noisy_bg",
            bg_type="noisy",
            image_type=image_type,
            keyer_mode=None,
            keyer_thresholds=None,
            despill="local_borrow",
            use_keyer_merge=False,
            notes=f"Background σ={sigma:.1f} too high for keying; fall back to net only.",
            extras=extras,
        )

    if C >= _BG_CHROMA_SATURATED:
        return Strategy(
            name="saturated_bg",
            bg_type="saturated",
            image_type=image_type,
            keyer_mode="chromatic",
            keyer_thresholds=thr,
            despill="auto",
            use_keyer_merge=True,
            use_keyer_gate=gate_default,
            notes=f"Saturated B (chroma={C:.1f}); chromatic key + unmix + chroma cap.",
            extras=extras,
        )

    if L >= _BG_LIGHTNESS_WHITE:
        return Strategy(
            name="white_bg",
            bg_type="white",
            image_type=image_type,
            keyer_mode="luminance",
            keyer_thresholds=thr,
            despill="unmix",
            use_keyer_merge=True,
            use_keyer_gate=gate_default,
            notes=f"White-ish B (L={L:.1f}); luminance key + unmix (no chroma cap).",
            extras=extras,
        )

    if L <= _BG_LIGHTNESS_BLACK:
        return Strategy(
            name="black_bg",
            bg_type="black",
            image_type=image_type,
            keyer_mode="luminance",
            keyer_thresholds=thr,
            despill="unmix",
            use_keyer_merge=True,
            use_keyer_gate=gate_default,
            notes=f"Black-ish B (L={L:.1f}); luminance key + unmix.",
            extras=extras,
        )

    return Strategy(
        name="grey_bg",
        bg_type="grey",
        image_type=image_type,
        keyer_mode="luminance",
        keyer_thresholds=KeyerThresholds(bg_max=10.0, fg_min=30.0),  # weak signal: widen
        despill="local_borrow",
        use_keyer_merge=False,
        notes=f"Mid-grey B (L={L:.1f}, C={C:.1f}); weak key signal, prefer net + local borrow.",
        extras=extras,
    )


def _route_asset_kind(
    image_srgb: np.ndarray,
    profile: str,
    screen_mode: str,
) -> str:
    del image_srgb
    if profile == "composite_character_corridor_only":
        return "character"
    if screen_mode not in {"green", "blue"}:
        return "unknown"
    if profile.startswith("opaque_hard_ui") or profile in {
        "translucent_button",
        "screen_tinted_translucency",
        "edge_cleanup",
        "balanced",
        "key_color_material",
    }:
        return "button"
    return "unknown"


def _pymatting_route_params(
    background_color: tuple[int, int, int],
    *,
    execution_profile: str = "pymatting-known-b",
    trimap_mode: str = "standard",
    unknown_grow_px: int = 0,
    auto_adapt: bool = True,
) -> dict[str, Any]:
    return {
        "execution_profile": execution_profile,
        "pymatting_method": "cf",
        "pymatting_image_space": "linear",
        "pymatting_bg_source": "custom",
        "pymatting_bg_color": tuple(int(c) for c in background_color),
        "pymatting_bg_threshold": 3.5,
        "pymatting_fg_threshold": 24.0,
        "pymatting_boundary_band_px": 2,
        "pymatting_auto_adapt": bool(auto_adapt),
        "pymatting_cg_maxiter": 1000,
        "pymatting_cg_rtol": 1e-6,
        "pymatting_trimap_mode": trimap_mode,
        "pymatting_unknown_grow_px": int(unknown_grow_px),
    }


def _corridorkey_route_params(analysis: Any, *, execution_profile: str) -> dict[str, Any]:
    settings = analysis.recommended_settings
    return {
        "execution_profile": execution_profile,
        "corridorkey_execution_profile": execution_profile,
        "corridorkey_screen_mode": analysis.screen_mode,
        "corridorkey_preset": "auto",
        "corridorkey_gamma_space": settings.gamma_space,
        "corridorkey_despill_strength": settings.despill_strength,
        "corridorkey_refiner_strength": settings.refiner_strength,
        "corridorkey_auto_despeckle": settings.auto_despeckle,
        "corridorkey_despeckle_size": settings.despeckle_size,
        "corridorkey_color_protection": settings.color_protection,
        "corridorkey_protection_bg_max": settings.protection_bg_max,
        "corridorkey_protection_fg_min": settings.protection_fg_min,
        # Auto routes should let CorridorKey choose useful hints for complex
        # edges, translucency, and characters. Deterministic hard buttons route
        # to PyMatting instead, so this no longer risks hard-UI hint fragility.
        "corridorkey_auto_mask": True,
    }


def _known_bg_glow_route_params(
    background_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "execution_profile": "known-bg-glow",
        "known_bg_glow_mode": mode,
        "known_bg_glow_bg_color": tuple(int(c) for c in background_color),
        "known_bg_glow_target_color": tuple(int(c) for c in target_color),
    }


def _complex_boundary_score(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
) -> tuple[bool, dict[str, Any]]:
    """Detect glass/translucent button edges missed by aggregate CK profiles.

    Failure mode: blue-screen translucent/real-glass buttons can look like
    ordinary ``edge_cleanup`` or ``balanced`` in same-key-color statistics
    because their material is far from the blue screen hue. The observable
    signal is either very high luminance-gradient percentiles from specular/glass
    texture or a meaningful mid-alpha band from screen-tinted translucency. Hard
    outlined UI can have strong contours, so the gradient threshold is
    intentionally high and the semi-alpha fraction provides the separate
    glass/translucency escape hatch.
    """
    key = chromatic_key_alpha(
        image_srgb,
        background_color,
        KeyerThresholds(bg_max=5.5, fg_min=18.0),
    )
    support = key >= 0.16
    support_pixels = int(support.sum())
    semi_alpha_fraction = float(np.count_nonzero((key > 0.16) & (key < 0.92)) / max(1, support_pixels))
    distance_to_exterior = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    interior_support = distance_to_exterior >= 3.0
    interior_semi_alpha = interior_support & (key > 0.16) & (key < 0.92)
    interior_semi_alpha_fraction = float(np.count_nonzero(interior_semi_alpha) / max(1, support_pixels))
    interior_semi_alpha_density = float(np.count_nonzero(interior_semi_alpha) / max(1, int(interior_support.sum())))
    if support_pixels < max(32, int(key.size * 0.04)):
        return False, {
            "enabled": True,
            "support_pixels": support_pixels,
            "semi_alpha_fraction": semi_alpha_fraction,
            "interior_semi_alpha_fraction": interior_semi_alpha_fraction,
            "interior_semi_alpha_density": interior_semi_alpha_density,
            "reason": "insufficient known-B support",
        }

    gray = cv2.cvtColor(image_srgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    edges = cv2.Canny(gray, 40, 120) > 0
    boundary = (
        cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        & ~cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    )
    edge_in_boundary = edges & boundary
    n_edge_labels, _edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(
        edge_in_boundary.astype(np.uint8),
        connectivity=8,
    )
    small_edge_components = int(
        sum(1 for i in range(1, n_edge_labels) if 2 <= int(edge_stats[i, cv2.CC_STAT_AREA]) <= 40)
    )
    boundary_edge_fraction = float(np.count_nonzero(edge_in_boundary) / max(1, int(boundary.sum())))
    support_gradient = gradient[support].astype(np.float32)
    p75 = float(np.percentile(support_gradient, 75.0))
    mean = float(support_gradient.mean())
    # Crisp outlines can have very high gradient energy at the edge. Require a
    # measurable non-edge semi-alpha population before treating strong gradients
    # as glass/complex-material evidence; hard outlined icons stay Known-B.
    gradient_gate = p75 >= 200.0 and semi_alpha_fraction >= 0.06
    # A broad mid-alpha band alone is not enough for "glass": soft/contact
    # shadows on hard UI also produce lots of mid-alpha key evidence, but with
    # low local structure. Require meaningful gradient energy for the pure
    # semi-alpha gate so heavy shadows stay on deterministic PyMatting.
    semi_alpha_gate = semi_alpha_fraction >= 0.18 and p75 >= 100.0
    # Broad glass/translucency evidence can split across two imperfect
    # observables: a material band that is not strong enough to pass the pure
    # semi-alpha gate, plus distributed specular/edge gradients that are below
    # the very-high hard contour guard. When both moderate signals appear on
    # known-screen UI, prefer CorridorKey so complex boundaries stay with the
    # learned keyer instead of the deterministic hard-edge solver.
    combined_glass_gate = p75 >= 160.0 and semi_alpha_fraction >= 0.14
    # This gate is specifically for "the middle is translucent", not soft
    # outline antialiasing. Distance-to-exterior excludes the edge band, and
    # the area/density thresholds keep contact shadows from becoming glass.
    interior_material_gate = interior_semi_alpha_fraction >= 0.30 and interior_semi_alpha_density >= 0.35
    # Character-like transparent fabric and hair can present as many small,
    # disconnected edge fragments plus a modest but real semi-alpha population.
    # This is topology + alpha evidence, not image size or aspect.
    fine_detail_translucent_gate = (
        small_edge_components >= 250
        and semi_alpha_fraction >= 0.05
        and boundary_edge_fraction >= 0.25
    )
    accepted = (
        gradient_gate
        or semi_alpha_gate
        or combined_glass_gate
        or interior_material_gate
        or fine_detail_translucent_gate
    )
    return accepted, {
        "enabled": True,
        "support_pixels": support_pixels,
        "support_fraction": float(support.mean()),
        "semi_alpha_fraction": semi_alpha_fraction,
        "interior_semi_alpha_fraction": interior_semi_alpha_fraction,
        "interior_semi_alpha_density": interior_semi_alpha_density,
        "boundary_edge_fraction": boundary_edge_fraction,
        "small_edge_components": small_edge_components,
        "support_gradient_p75": p75,
        "support_gradient_mean": mean,
        "gradient_gate": gradient_gate,
        "semi_alpha_gate": semi_alpha_gate,
        "combined_glass_gate": combined_glass_gate,
        "interior_material_gate": interior_material_gate,
        "fine_detail_translucent_gate": fine_detail_translucent_gate,
        # Experience-driven but feature-bound: hard outlined buttons can reach
        # the low hundreds from a single crisp contour; glass/specular buttons
        # that need CorridorKey show either much higher distributed gradients
        # or a broad semi-alpha band from known-background blending.
        "support_gradient_p75_min": 200.0,
        "semi_alpha_fraction_min": 0.18,
        "semi_alpha_gradient_p75_min": 100.0,
        "combined_support_gradient_p75_min": 160.0,
        "combined_semi_alpha_fraction_min": 0.14,
        "interior_semi_alpha_fraction_min": 0.30,
        "interior_semi_alpha_density_min": 0.35,
        "fine_detail_small_edge_components_min": 250,
        "fine_detail_semi_alpha_fraction_min": 0.05,
        "fine_detail_boundary_edge_fraction_min": 0.25,
        "reason": "" if accepted else "below complex-boundary and semi-alpha gates",
    }


def classify_route(
    image_srgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    *,
    screen_mode: str = "auto",
    preset: str = "auto",
    fallback_background_color: tuple[int, int, int] = (0, 200, 0),
) -> RouteDecision:
    """Select the concrete production route for ``backend='auto'``.

    Broad policy:
    - clean source alpha passes through;
    - CorridorKey is used only for green/blue known-screen assets where its
      learned translucent/complex-boundary behavior is valuable;
    - deterministic hard UI/buttons use the more stable known-B PyMatting path;
    - stable non-green/blue flat backgrounds also use known-B PyMatting;
    - unstable/unknown backgrounds fall back to PyMatting with the configured
      fallback background color rather than invoking RMBG.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("classify_route() expects HxWx3 sRGB uint8")

    legacy = classify_strategy(image_srgb, source_alpha=source_alpha)
    if legacy.passthrough:
        return RouteDecision(
            route="rgba_passthrough",
            asset_kind="rgba",
            backend="rgba_passthrough",
            params={},
            confidence=1.0,
            reasons=["clean_source_alpha"],
            analysis={"strategy": {
                "name": legacy.name,
                "bg_type": legacy.bg_type,
                "image_type": legacy.image_type,
                "extras": legacy.extras,
            }},
        )

    from .corridorkey import corridorkey_analyze_asset
    from .pymatting_refine import estimate_stable_background_color

    ck = corridorkey_analyze_asset(
        image_srgb,
        screen_mode=screen_mode,  # type: ignore[arg-type]
        preset=preset,  # type: ignore[arg-type]
        fallback_background_color=fallback_background_color,
    )
    profile = ck.parameter_profile
    # CorridorKey is only production-safe when the green/blue screen is a
    # confident observed screen (or explicitly forced by the caller). Random
    # colorful/noisy borders can otherwise look like sparse green/blue coverage
    # and incorrectly pull unknown photos into the learned keyer route.
    known_corridor_screen = ck.screen_mode in {"green", "blue"} and (
        screen_mode != "auto" or ck.background_confidence >= 0.45
    )
    reasons: list[str] = []
    analysis: dict[str, Any] = {
        "strategy": {
            "name": legacy.name,
            "bg_type": legacy.bg_type,
            "image_type": legacy.image_type,
            "extras": legacy.extras,
        },
        "corridorkey_analysis": ck.to_dict(),
    }

    if known_corridor_screen:
        from .known_bg_glow import analyze_known_bg_glow

        glow = analyze_known_bg_glow(image_srgb, ck.background_color)
        analysis["known_bg_glow"] = glow.to_dict()
        if glow.accepted:
            reasons.append(f"known_bg_{glow.mode}_glow_uses_known_bg_glow")
            return RouteDecision(
                route="known_bg_glow",
                asset_kind="glow",
                backend="known_bg_glow",
                params=_known_bg_glow_route_params(glow.background_color, glow.target_color, mode=glow.mode),
                confidence=float(max(0.70, ck.background_confidence)),
                reasons=reasons,
                analysis=analysis,
            )

    asset_kind = _route_asset_kind(
        image_srgb,
        profile,
        ck.screen_mode if known_corridor_screen else "unknown",
    )
    button_corridor_profiles = {"translucent_button"}
    complex_button_boundary, complex_button_info = _complex_boundary_score(
        image_srgb,
        ck.background_color,
    )
    analysis["complex_button_boundary"] = complex_button_info
    complex_button_can_use_corridorkey = (
        complex_button_boundary
        and not profile.startswith("opaque_hard_ui")
        and profile != "key_color_material"
    )
    if known_corridor_screen and (
        asset_kind in {"icon", "character"}
        or (asset_kind == "button" and (profile in button_corridor_profiles or complex_button_can_use_corridorkey))
    ):
        if asset_kind == "character":
            execution_profile = "corridorkey-character"
        elif asset_kind == "button":
            execution_profile = "corridorkey-transparent-button"
        elif profile == "screen_tinted_translucency":
            execution_profile = "corridorkey-effect-icon"
        else:
            execution_profile = "corridorkey-shaped-icon"

        params = _corridorkey_route_params(ck, execution_profile=execution_profile)
        if asset_kind == "button":
            # The router owns the transparent/glass-button decision and sends
            # the complete execution recipe. Downstream code should not infer
            # button behavior from CorridorKey's generic semantic profile.
            params["corridorkey_hard_ui_hint_mode"] = "translucent_button"
        if complex_button_can_use_corridorkey and asset_kind == "button" and profile not in button_corridor_profiles:
            reasons.append(f"button_{profile}_complex_boundary_uses_corridorkey")
        else:
            reasons.append(f"{asset_kind}_{profile}_uses_corridorkey")
        return RouteDecision(
            route="corridorkey",
            asset_kind=asset_kind,
            backend="corridorkey",
            params=params,
            confidence=float(max(0.50, ck.background_confidence)),
            reasons=reasons,
            analysis=analysis,
        )

    stable_bg, stable_info = estimate_stable_background_color(
        image_srgb,
        seed_bg=ck.background_color if known_corridor_screen else None,
        seed_source="route_screen_analysis",
        seed_info={
            "screen_mode": ck.screen_mode,
            "background_confidence": float(ck.background_confidence),
            "border_coverage": dict(ck.border_coverage),
        }
        if known_corridor_screen
        else None,
    )
    analysis["stable_background"] = stable_info
    if stable_info.get("accepted", False):
        trimap_mode = "standard"
        unknown_grow_px = 0
        # Threshold intent: the semantic profile can be won by key_color_material
        # even when a same-key opaque plateau exists. The outline trimap should
        # depend on measured plateau+outline evidence instead of that winner.
        # Keep the plateau gate below the semantic 0.85 cutoff so outlined
        # icon/button shapes with white glyphs still qualify, but require the
        # dominant same-key ownership gate and a successful outline extractor to
        # avoid using this mode on weak spill/glow residue.
        same_key_outline_candidate = (
            asset_kind == "button"
            and float(ck.subject_key_color_risk) >= 0.45
            and float(ck.same_key_opaque_plateau_confidence) >= 0.65
        )
        if same_key_outline_candidate:
            from .pymatting_refine import analyze_same_key_opaque_body_outline

            body_outline = analyze_same_key_opaque_body_outline(
                image_srgb,
                stable_bg,
                bg_threshold=3.5,
            )
            analysis["same_key_opaque_body_outline"] = body_outline
            if body_outline.get("accepted", False):
                trimap_mode = "same_key_opaque_body_outline"
                unknown_grow_px = 2
        if asset_kind == "button":
            reasons.append(f"button_{profile}_uses_known_b_pymatting")
        elif not known_corridor_screen:
            reasons.append("non_green_blue_stable_background_uses_known_b_pymatting")
        else:
            reasons.append(f"deterministic_{profile}_uses_known_b_pymatting")
        return RouteDecision(
            route="pymatting_known_b",
            asset_kind=asset_kind if asset_kind != "unknown" else "known_bg_graphic",
            backend="pymatting_known_b",
            params=_pymatting_route_params(
                stable_bg,
                execution_profile="pymatting-hard-button" if asset_kind == "button" else "pymatting-known-bg",
                trimap_mode=trimap_mode,
                unknown_grow_px=unknown_grow_px,
                auto_adapt=not known_corridor_screen,
            ),
            confidence=float(max(0.45, ck.background_confidence, 0.80)),
            reasons=reasons,
            analysis=analysis,
        )

    reasons.append("unknown_or_unstable_background_uses_pymatting_fallback")
    fallback_params = _pymatting_route_params(
        fallback_background_color,
        execution_profile="pymatting-fallback",
    )
    # Unknown fallback is not a true measured known-B screen. The normal
    # adaptive thresholding treats noisy/colorful borders as background noise
    # and can raise the foreground threshold until the trimap has no foreground
    # seeds. Use fixed thresholds so PyMatting receives a valid, bounded trimap.
    fallback_params["pymatting_auto_adapt"] = False
    return RouteDecision(
        route="pymatting_fallback",
        asset_kind=asset_kind if asset_kind != "unknown" else "unknown_fallback",
        backend="pymatting_fallback",
        # Fallback is intentionally deterministic and bounded: unknown inputs
        # may not have a true known-B screen, but the production auto path must
        # stay on the PyMatting/CorridorKey family and never trigger the slow
        # RMBG model path from the RouteMatte node.
        params=fallback_params,
        confidence=float(max(0.10, ck.background_confidence)),
        reasons=reasons,
        analysis=analysis,
    )


__all__ = [
    "Strategy",
    "RouteDecision",
    "AlphaHygiene",
    "classify_strategy",
    "classify_route",
    "assess_source_alpha",
]
