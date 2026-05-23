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

from .colorspace import srgb_to_oklab
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
    passthrough: bool = False    # if True, skip matting net entirely
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


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
    """
    if _is_passthrough(source_alpha):
        return Strategy(
            name="rgba_passthrough",
            bg_type="rgba_passthrough",
            image_type="any",
            keyer_mode=None,
            keyer_thresholds=None,
            despill="none",
            use_keyer_merge=False,
            passthrough=True,
            notes="Source already has alpha; copy through unchanged.",
        )

    L, C, sigma = _bg_lab_stats(image_srgb, source_alpha)
    image_type = _detect_image_type(image_srgb)

    # Tight thresholds for graphics (hard edges, expected to be clean), wider
    # for photos (anti-aliased / hairy edges, allow more soft-edge slack).
    if image_type == "graphic":
        thr = KeyerThresholds(bg_max=4.0, fg_min=14.0)
    else:
        thr = KeyerThresholds(bg_max=6.0, fg_min=22.0)

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
            extras={"bg_L": L, "bg_C": C, "bg_sigma": sigma},
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
            notes=f"Saturated B (chroma={C:.1f}); chromatic key + unmix + chroma cap.",
            extras={"bg_L": L, "bg_C": C, "bg_sigma": sigma},
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
            notes=f"White-ish B (L={L:.1f}); luminance key + unmix (no chroma cap).",
            extras={"bg_L": L, "bg_C": C, "bg_sigma": sigma},
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
            notes=f"Black-ish B (L={L:.1f}); luminance key + unmix.",
            extras={"bg_L": L, "bg_C": C, "bg_sigma": sigma},
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
        extras={"bg_L": L, "bg_C": C, "bg_sigma": sigma},
    )


__all__ = ["Strategy", "classify_strategy"]
