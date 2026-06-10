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
            self.params.get("parameter_profile")
            or (
                corridorkey_analysis.get("parameter_profile")
                if isinstance(corridorkey_analysis, dict)
                else None
            )
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


@dataclass(frozen=True)
class RouteCandidate:
    """Analyze-stage model candidate.

    A candidate owns one complete executable route decision plus the model
    evidence that made it plausible. ``classify_route()`` remains a compatibility
    wrapper over the selected default candidate.
    """

    id: str
    decision: RouteDecision
    evidence: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    default: bool = False

    def to_route_decision(self) -> RouteDecision:
        return self.decision

    def to_dict(self) -> dict[str, Any]:
        payload = self.decision.to_dict()
        payload.update(
            {
                "id": self.id,
                "default": self.default,
                "evidence": self.evidence,
                "risks": self.risks,
                "route_candidate_id": self.id,
            }
        )
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
    parameter_profile: str = "known_b_standard",
    trimap_mode: str = "standard",
    unknown_grow_px: int = 0,
) -> dict[str, Any]:
    return {
        "execution_profile": execution_profile,
        "parameter_profile": parameter_profile,
        "pymatting_method": "cf",
        "pymatting_image_space": "linear",
        "pymatting_bg_source": "custom",
        "pymatting_bg_color": tuple(int(c) for c in background_color),
        "pymatting_bg_threshold": 3.5,
        "pymatting_fg_threshold": 24.0,
        "pymatting_boundary_band_px": 2,
        "pymatting_adapt_bg_threshold": False,
        "pymatting_adapt_fg_threshold": True,
        "pymatting_adapt_boundary_band": True,
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
        # Every CorridorKey auto route defaults to a full-frame 0.32 soft prior.
        # auto_mask=False keeps the executor on the constant-prior path without
        # computing any feature hint.
        "corridorkey_auto_mask": False,
    }


def _known_bg_glow_route_params(
    background_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "execution_profile": "known-bg-glow",
        "parameter_profile": f"known_bg_glow_{mode}",
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


def _character_like_foreground(
    image_srgb: np.ndarray,
    ck_analysis: Any,
    complex_boundary_info: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Detect fine-detail complex foregrounds for CorridorKey parameters.

    This is intentionally topology/alpha evidence, not canvas geometry. Export
    padding, atlas cells, and crop choices can change aspect ratio without
    changing the matting model, so width/height ratio must not decide the route.
    The gate asks whether the known-screen foreground has many small boundary
    fragments plus a real soft/translucent population.
    """

    del image_srgb
    bbox = getattr(ck_analysis, "foreground_bbox_xyxy", None)
    small_component_risk = bool(getattr(ck_analysis, "small_component_risk", False))
    if not bbox:
        return False, {
            "enabled": True,
            "accepted": False,
            "reason": "missing foreground geometry",
        }

    x1, y1, x2, y2 = [int(v) for v in bbox]
    fine_detail_edges = int(complex_boundary_info.get("small_edge_components") or 0)
    semi_alpha_fraction = float(complex_boundary_info.get("semi_alpha_fraction") or 0.0)
    boundary_edge_fraction = float(complex_boundary_info.get("boundary_edge_fraction") or 0.0)
    fine_detail_gate = (
        bool(complex_boundary_info.get("fine_detail_translucent_gate"))
        or fine_detail_edges >= 240
    )
    accepted = (
        fine_detail_gate
        and semi_alpha_fraction >= 0.05
        and boundary_edge_fraction >= 0.20
    )
    return accepted, {
        "enabled": True,
        "accepted": bool(accepted),
        "bbox_xyxy": [x1, y1, x2, y2],
        "small_component_risk": small_component_risk,
        "fine_detail_small_edge_components": fine_detail_edges,
        "semi_alpha_fraction": semi_alpha_fraction,
        "boundary_edge_fraction": boundary_edge_fraction,
        "fine_detail_gate": bool(fine_detail_gate),
        "min_semi_alpha_fraction": 0.05,
        "min_boundary_edge_fraction": 0.20,
        "reason": "" if accepted else "below fine-detail topology/alpha gates",
    }


def _opaque_known_b_model_evidence(ck_analysis: Any) -> tuple[bool, dict[str, Any]]:
    """Return whether near-screen evidence is better modeled as hard opaque UI.

    This guard separates same-key opaque material and hard shadow bands from
    translucent ramps before CorridorKey can claim them. It uses measured
    support/plateau evidence instead of profile labels.
    """

    same_key_plateau_confidence = float(getattr(ck_analysis, "same_key_opaque_plateau_confidence", 0.0) or 0.0)
    key_color_solid_fraction = float(getattr(ck_analysis, "key_color_solid_fraction", 0.0) or 0.0)
    key_color_hard_density = float(getattr(ck_analysis, "key_color_hard_density", 0.0) or 0.0)
    key_color_compact_fraction = float(getattr(ck_analysis, "key_color_compact_fraction", 0.0) or 0.0)
    solid_hard_band = (
        key_color_solid_fraction >= 0.55
        and key_color_hard_density >= 0.08
        and key_color_compact_fraction >= 0.80
    )
    same_key_plateau = same_key_plateau_confidence >= 0.85
    accepted = bool(same_key_plateau or solid_hard_band)
    return accepted, {
        "enabled": True,
        "accepted": accepted,
        "same_key_plateau": bool(same_key_plateau),
        "solid_hard_band": bool(solid_hard_band),
        "same_key_opaque_plateau_confidence": same_key_plateau_confidence,
        "key_color_solid_fraction": key_color_solid_fraction,
        "key_color_hard_density": key_color_hard_density,
        "key_color_compact_fraction": key_color_compact_fraction,
        "same_key_plateau_confidence_min": 0.85,
        "solid_fraction_min": 0.55,
        "hard_density_min": 0.08,
        "compact_fraction_min": 0.80,
        "reason": "" if accepted else "no coherent opaque known-B material support",
    }


def _fine_detail_composite_evidence(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    ck_analysis: Any,
    complex_boundary_info: dict[str, Any],
    opaque_known_b_model: bool,
) -> tuple[bool, dict[str, Any]]:
    """Detect hard-body plus fine-detail/translucent composite foregrounds.

    This evidence is deliberately model-level. It does not look at aspect ratio,
    canvas padding, or foreground bbox fractions; size is used only to normalize
    support and component thresholds.
    """

    key = chromatic_key_alpha(
        image_srgb,
        background_color,
        KeyerThresholds(bg_max=5.5, fg_min=18.0),
    )
    support = key >= 0.16
    support_pixels = int(support.sum())
    support_fraction = float(support.mean())
    hard_support = key >= 0.92
    semi = (key > 0.16) & (key < 0.92)
    if support_pixels < max(32, int(key.size * 0.04)):
        return False, {
            "enabled": True,
            "accepted": False,
            "support_pixels": support_pixels,
            "support_fraction": support_fraction,
            "reason": "insufficient foreground support",
        }

    hard_pixels = image_srgb[hard_support]
    if hard_pixels.size:
        q = (hard_pixels >> 3).astype(np.int32)
        hard_color_bins = int(np.unique(q[:, 0] * 32 * 32 + q[:, 1] * 32 + q[:, 2]).size)
    else:
        hard_color_bins = 0

    semi_labels, _semi_map, semi_stats, _ = cv2.connectedComponentsWithStats(
        semi.astype(np.uint8),
        connectivity=8,
    )
    semi_alpha_islands = int(
        sum(1 for i in range(1, semi_labels) if 4 <= int(semi_stats[i, cv2.CC_STAT_AREA]) <= 5000)
    )
    interior_semi_largest = int(
        max((int(semi_stats[i, cv2.CC_STAT_AREA]) for i in range(1, semi_labels)), default=0)
    )
    distance_to_exterior = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    thin_structure_fraction = float(np.count_nonzero(support & (distance_to_exterior <= 2.0)) / max(1, support_pixels))

    fine_edge_components = int(complex_boundary_info.get("small_edge_components") or 0)
    boundary_edge_fraction = float(complex_boundary_info.get("boundary_edge_fraction") or 0.0)
    key_transition_fraction = float(getattr(ck_analysis, "key_transition_fraction", 0.0) or 0.0)
    small_component_risk = bool(getattr(ck_analysis, "small_component_risk", False))
    key_color_compact_fraction = float(getattr(ck_analysis, "key_color_compact_fraction", 0.0) or 0.0)
    semi_alpha_fraction = float(complex_boundary_info.get("semi_alpha_fraction") or 0.0)
    fine_boundary_gate = (
        support_fraction <= 0.75
        and fine_edge_components >= 240
        and boundary_edge_fraction >= 0.23
        and hard_color_bins >= 500
        and not opaque_known_b_model
    )
    soft_effect_gate = (
        support_fraction <= 0.45
        and key_transition_fraction >= 0.12
        and semi_alpha_fraction >= 0.14
        and semi_alpha_islands >= 24
        and hard_color_bins >= 350
        and small_component_risk
        and not opaque_known_b_model
    )
    soft_boundary_detail_gate = (
        support_fraction <= 0.35
        and boundary_edge_fraction >= 0.24
        and semi_alpha_fraction >= 0.035
        and hard_color_bins >= 700
        and semi_alpha_islands >= 24
        and interior_semi_largest <= 64
        and key_color_compact_fraction <= 0.08
        and not opaque_known_b_model
    )
    accessory_gate = semi_alpha_islands >= 8 or interior_semi_largest >= max(64, int(round(key.size * 0.001)))
    accepted = bool((fine_boundary_gate or soft_effect_gate or soft_boundary_detail_gate) and accessory_gate)
    return accepted, {
        "enabled": True,
        "accepted": accepted,
        "fine_boundary_gate": bool(fine_boundary_gate),
        "soft_effect_gate": bool(soft_effect_gate),
        "soft_boundary_detail_gate": bool(soft_boundary_detail_gate),
        "fine_edge_components": fine_edge_components,
        "boundary_edge_fraction": boundary_edge_fraction,
        "key_transition_fraction": key_transition_fraction,
        "semi_alpha_fraction": semi_alpha_fraction,
        "hard_color_bins": hard_color_bins,
        "semi_alpha_islands": semi_alpha_islands,
        "interior_semi_largest": interior_semi_largest,
        "thin_structure_fraction": thin_structure_fraction,
        "support_pixels": support_pixels,
        "support_fraction": support_fraction,
        "small_component_risk": small_component_risk,
        "key_color_compact_fraction": key_color_compact_fraction,
        "opaque_known_b_model_guard": bool(opaque_known_b_model),
        "support_fraction_max": 0.75,
        "fine_edge_components_min": 240,
        "boundary_edge_fraction_min": 0.23,
        "hard_color_bins_min": 500,
        "semi_alpha_islands_min": 8,
        "soft_effect_support_fraction_max": 0.45,
        "soft_effect_key_transition_fraction_min": 0.12,
        "soft_effect_semi_alpha_fraction_min": 0.14,
        "soft_effect_semi_alpha_islands_min": 24,
        "soft_effect_hard_color_bins_min": 350,
        "soft_boundary_detail_support_fraction_max": 0.35,
        "soft_boundary_detail_boundary_edge_fraction_min": 0.24,
        "soft_boundary_detail_semi_alpha_fraction_min": 0.035,
        "soft_boundary_detail_hard_color_bins_min": 700,
        "soft_boundary_detail_semi_alpha_islands_min": 24,
        "soft_boundary_detail_interior_semi_largest_max": 64,
        "soft_boundary_detail_key_color_compact_fraction_max": 0.08,
        "base_gate": bool(fine_boundary_gate),
        "accessory_gate": bool(accessory_gate),
        "reason": "" if accepted else "below fine-detail composite evidence gate",
    }


def build_route_candidates(
    image_srgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    *,
    screen_mode: str = "auto",
    preset: str = "auto",
    fallback_background_color: tuple[int, int, int] = (0, 200, 0),
) -> list[RouteCandidate]:
    """Build executable model candidates for ``backend='auto'``.

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
        raise ValueError("build_route_candidates() expects HxWx3 sRGB uint8")

    legacy = classify_strategy(image_srgb, source_alpha=source_alpha)
    if legacy.passthrough:
        return [
            RouteCandidate(
                id="route_rgba_passthrough",
                default=True,
                decision=RouteDecision(
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
                ),
                evidence={"alpha_hygiene": legacy.extras.get("hygiene", {})},
            )
        ]

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
    model_background = ck.background_color if known_corridor_screen else stable_bg

    candidates: list[RouteCandidate] = []
    if stable_info.get("accepted", False):
        from .known_bg_glow import analyze_known_bg_glow

        glow = analyze_known_bg_glow(image_srgb, model_background)
        analysis["known_bg_glow"] = glow.to_dict()
        if glow.accepted:
            candidates.append(
                RouteCandidate(
                    id="route_known_bg_glow",
                    default=True,
                    decision=RouteDecision(
                        route="known_bg_glow",
                        asset_kind="glow",
                        backend="known_bg_glow",
                        params=_known_bg_glow_route_params(glow.background_color, glow.target_color, mode=glow.mode),
                        confidence=float(max(0.70, ck.background_confidence)),
                        reasons=[f"known_bg_{glow.mode}_glow_uses_known_bg_glow"],
                        analysis=analysis,
                    ),
                    evidence={
                        "background_solvability": stable_info,
                        "glow_evidence": glow.to_dict(),
                    },
                )
            )

    complex_button_boundary, complex_button_info = _complex_boundary_score(
        image_srgb,
        model_background,
    )
    analysis["complex_button_boundary"] = complex_button_info
    opaque_known_b_model, opaque_known_b_info = _opaque_known_b_model_evidence(ck)
    analysis["opaque_known_b_model"] = opaque_known_b_info
    character_like, character_like_info = _character_like_foreground(
        image_srgb,
        ck,
        complex_button_info,
    )
    analysis["character_like_foreground"] = character_like_info
    fine_detail_composite, fine_detail_info = _fine_detail_composite_evidence(
        image_srgb,
        model_background,
        ck_analysis=ck,
        complex_boundary_info=complex_button_info,
        opaque_known_b_model=opaque_known_b_model,
    )
    analysis["fine_detail_composite_evidence"] = fine_detail_info
    complex_translucent_model = (
        (complex_button_boundary or fine_detail_composite)
        and not opaque_known_b_model
    )
    asset_kind = _route_asset_kind(
        image_srgb,
        profile,
        ck.screen_mode if known_corridor_screen else "unknown",
    )
    same_key_opaque_button_outline: dict[str, Any] = {
        "enabled": bool(known_corridor_screen and asset_kind == "button"),
        "accepted": False,
        "reason": "not a known green/blue same-key button",
    }
    same_key_opaque_button = False
    if known_corridor_screen and stable_info.get("accepted", False) and asset_kind == "button":
        try:
            from .pymatting_refine import analyze_same_key_opaque_body_outline

            same_key_opaque_button_outline = analyze_same_key_opaque_body_outline(
                image_srgb,
                model_background,
                bg_threshold=3.5,
            )
        except Exception as exc:
            same_key_opaque_button_outline = {
                "enabled": True,
                "accepted": False,
                "reason": f"same_key_outline_probe_failed: {exc}",
            }
        same_key_opaque_button = bool(
            opaque_known_b_info.get("same_key_plateau", False)
            and same_key_opaque_button_outline.get("accepted", False)
        )
    analysis["same_key_opaque_button_outline"] = same_key_opaque_button_outline
    analysis["same_key_button_model"] = {
        "enabled": bool(known_corridor_screen and asset_kind == "button"),
        "accepted": bool(same_key_opaque_button),
        "screen_mode": ck.screen_mode,
        "opaque_plateau": bool(opaque_known_b_info.get("same_key_plateau", False)),
        "outline_accepted": bool(same_key_opaque_button_outline.get("accepted", False)),
        "policy": "offer_corridorkey_translucent_and_known_b_opaque_outline"
        if same_key_opaque_button
        else "standard_route_candidates",
    }
    if same_key_opaque_button and not candidates:
        opaque_params = _pymatting_route_params(
            stable_bg,
            execution_profile="pymatting-hard-button",
            parameter_profile="known_b_same_key_opaque_outline",
            trimap_mode="same_key_opaque_body_outline",
            unknown_grow_px=0,
        )
        translucent_params = _corridorkey_route_params(
            ck,
            execution_profile="corridorkey-transparent-button",
        )
        translucent_params.update(
            {
                "parameter_profile": "corridorkey_same_key_translucent_button",
                "same_key_button_interpretation": "semi_transparent_corridorkey",
            }
        )
        return [
            RouteCandidate(
                id="route_pymatting_known_b_same_key_opaque",
                default=True,
                decision=RouteDecision(
                    route="pymatting_known_b",
                    asset_kind="button",
                    backend="pymatting_known_b",
                    params=opaque_params,
                    confidence=float(max(0.88, ck.same_key_opaque_plateau_confidence)),
                    reasons=["same_key_button_outline_uses_known_b_opaque_proxy"],
                    analysis=analysis,
                ),
                evidence={
                    "background_solvability": stable_info,
                    "same_key_opaque_button_outline": same_key_opaque_button_outline,
                },
            ),
            RouteCandidate(
                id="route_corridorkey_same_key_translucent",
                decision=RouteDecision(
                    route="corridorkey",
                    asset_kind="button",
                    backend="corridorkey",
                    params=translucent_params,
                    confidence=0.64,
                    reasons=["same_key_button_translucent_counter_candidate_uses_corridorkey"],
                    analysis=analysis,
                ),
                evidence={
                    "background_solvability": stable_info,
                    "same_key_opaque_button_outline": same_key_opaque_button_outline,
                    "translucent_interpretation": {
                        "assumption": "same-key button interior is semi-transparent screen material"
                    },
                },
                risks=["opaque_same_key_counter_candidate"],
            ),
        ]

    if stable_info.get("accepted", False):
        known_b_reasons: list[str] = []
        if asset_kind == "button":
            known_b_reasons.append(f"button_{profile}_uses_known_b_pymatting")
        elif not known_corridor_screen:
            known_b_reasons.append("non_green_blue_stable_background_uses_known_b_pymatting")
        else:
            known_b_reasons.append(f"deterministic_{profile}_uses_known_b_pymatting")
        known_b_confidence = 0.78 if known_corridor_screen and complex_translucent_model else float(max(0.45, ck.background_confidence, 0.80))
        candidates.append(
            RouteCandidate(
                id="route_pymatting_known_b",
                decision=RouteDecision(
                    route="pymatting_known_b",
                    asset_kind=asset_kind if asset_kind != "unknown" else "known_bg_graphic",
                    backend="pymatting_known_b",
                    params=_pymatting_route_params(
                        stable_bg,
                        execution_profile="pymatting-hard-button" if asset_kind == "button" else "pymatting-known-bg",
                        parameter_profile="known_b_hard_button_standard"
                        if asset_kind == "button"
                        else "known_b_background_standard",
                    ),
                    confidence=float(known_b_confidence),
                    reasons=known_b_reasons,
                    analysis=analysis,
                ),
                evidence={
                    "background_solvability": stable_info,
                    "opaque_known_b_evidence": {
                        "accepted": True,
                        "stable_known_background": True,
                        "route_profile_after_model": profile,
                    },
                },
                risks=["fine_detail_composite_model_ambiguity"] if known_corridor_screen and fine_detail_composite else [],
            )
        )

    if known_corridor_screen and stable_info.get("accepted", False) and complex_translucent_model:
        fine_detail_character_composite = bool(
            fine_detail_info.get("fine_boundary_gate") or fine_detail_info.get("soft_effect_gate")
        )
        soft_boundary_detail = bool(fine_detail_info.get("soft_boundary_detail_gate"))
        if character_like or fine_detail_character_composite:
            asset_kind = "character"
            execution_profile = "corridorkey-character"
        elif soft_boundary_detail:
            asset_kind = "button"
            execution_profile = "corridorkey-shaped-icon"
        else:
            asset_kind = "button"
            execution_profile = "corridorkey-transparent-button"

        params = _corridorkey_route_params(ck, execution_profile=execution_profile)
        if soft_boundary_detail and not fine_detail_character_composite:
            ck_reasons = ["soft_boundary_detail_foreground_uses_corridorkey"]
        elif fine_detail_composite:
            ck_reasons = ["fine_detail_composite_foreground_uses_corridorkey"]
        elif character_like:
            ck_reasons = ["fine_detail_complex_foreground_uses_corridorkey"]
        else:
            ck_reasons = ["complex_translucent_known_screen_uses_corridorkey"]
        ck_confidence = float(max(0.86, ck.background_confidence)) if (complex_button_boundary or fine_detail_composite) else 0.72
        candidates.append(
            RouteCandidate(
                id="route_corridorkey",
                decision=RouteDecision(
                    route="corridorkey",
                    asset_kind=asset_kind,
                    backend="corridorkey",
                    params=params,
                    confidence=ck_confidence,
                    reasons=ck_reasons,
                    analysis=analysis,
                ),
                evidence={
                    "background_solvability": stable_info,
                    "translucent_mix_evidence": complex_button_info,
                    "fine_detail_composite_evidence": fine_detail_info,
                    "opaque_known_b_model_guard": opaque_known_b_info,
                },
                risks=["known_b_hard_opaque_counter_candidate"] if candidates else [],
            )
        )

    if candidates:
        default = select_default_route_candidate(candidates)
        return [
            RouteCandidate(
                id=candidate.id,
                decision=candidate.decision,
                evidence=candidate.evidence,
                risks=candidate.risks,
                default=candidate.id == default.id,
            )
            for candidate in candidates
        ]

    fallback_params = _pymatting_route_params(
        fallback_background_color,
        execution_profile="pymatting-fallback",
        parameter_profile="pymatting_fallback",
    )
    # Unknown fallback is not a true measured known-B screen. Keep every adaptive
    # inference off so PyMatting receives a valid, bounded trimap.
    fallback_params["pymatting_adapt_bg_threshold"] = False
    fallback_params["pymatting_adapt_fg_threshold"] = False
    fallback_params["pymatting_adapt_boundary_band"] = False
    return [
        RouteCandidate(
            id="route_pymatting_fallback",
            default=True,
            decision=RouteDecision(
                route="pymatting_fallback",
                asset_kind=asset_kind if asset_kind != "unknown" else "unknown_fallback",
                backend="pymatting_fallback",
                # Fallback is intentionally deterministic and bounded: unknown inputs
                # may not have a true known-B screen, but the production auto path must
                # stay on the PyMatting/CorridorKey family and never trigger the slow
                # RMBG model path from the RouteMatte node.
                params=fallback_params,
                confidence=float(max(0.10, ck.background_confidence)),
                reasons=["unknown_or_unstable_background_uses_pymatting_fallback"],
                analysis=analysis,
            ),
            evidence={
                "background_solvability": stable_info,
                "fallback_risk": {"unstable_or_unknown_background": True},
            },
            risks=["background_not_stably_solved"],
        )
    ]


def select_default_route_candidate(candidates: list[RouteCandidate]) -> RouteCandidate:
    """Select the default executable candidate from Analyze model candidates."""

    if not candidates:
        raise ValueError("select_default_route_candidate() requires at least one candidate")
    explicit = [candidate for candidate in candidates if candidate.default]
    if explicit:
        return explicit[0]
    return max(candidates, key=lambda candidate: float(candidate.decision.confidence))


def classify_route(
    image_srgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    *,
    screen_mode: str = "auto",
    preset: str = "auto",
    fallback_background_color: tuple[int, int, int] = (0, 200, 0),
) -> RouteDecision:
    """Compatibility wrapper returning the selected default route candidate."""

    return select_default_route_candidate(
        build_route_candidates(
            image_srgb,
            source_alpha=source_alpha,
            screen_mode=screen_mode,
            preset=preset,
            fallback_background_color=fallback_background_color,
        )
    ).to_route_decision()


__all__ = [
    "Strategy",
    "RouteDecision",
    "RouteCandidate",
    "AlphaHygiene",
    "classify_strategy",
    "build_route_candidates",
    "select_default_route_candidate",
    "classify_route",
    "assess_source_alpha",
]
