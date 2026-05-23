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
from .keyer import KeyerThresholds


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
    median = np.median(pixels, axis=0).astype(np.uint8)
    sigma = float(np.std(pixels.astype(np.float32), axis=0).mean())
    lab = srgb_to_oklab(median.reshape(1, 1, 3)).reshape(3)
    L = float(lab[0]) * 100.0
    C = float(np.sqrt(lab[1] ** 2 + lab[2] ** 2)) * 100.0
    return L, C, sigma


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


__all__ = ["Strategy", "AlphaHygiene", "classify_strategy", "assess_source_alpha"]
