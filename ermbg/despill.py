"""Spill suppression algorithms for solid-color background matting.

Three independent strategies. None of them produce alpha — they take an existing
alpha matte and return cleaned-up linear-RGB foreground colors.

  chroma_cap     : classic green-screen "min(G, max(R, B))" generalised. Cheap,
                   only works when B has a dominant channel (saturated screens).
  local_borrow   : Photoshop-style decontaminate-colors. For each translucent
                   pixel, weighted-average the K nearest sure_fg pixels'
                   colors. Robust on any B but slower.
  closed_form    : pymatting's Levin closed-form alpha + ML-foreground. Joint
                   solution, slowest, theoretically optimal.

All operate in linear RGB and return linear RGB.
"""

from __future__ import annotations

import numpy as np

# B with any channel below this is considered "not a saturated screen".
_DOMINANT_CHANNEL_MIN = 0.3   # in linear RGB
_DOMINANT_DELTA_MIN = 0.15    # dominant - second-largest must exceed this


# ---------------------------------------------------------------------------
# (A) Chroma cap (classic green-screen despill, generalised to any saturated B)
# ---------------------------------------------------------------------------


def has_dominant_screen_channel(background_linear: np.ndarray) -> int | None:
    """Return the index (0/1/2) of B's dominant channel, or None if B is not
    a saturated single-channel screen (e.g. black, grey, white)."""
    B = np.asarray(background_linear, dtype=np.float32).reshape(3)
    order = np.argsort(B)[::-1]
    top, second = order[0], order[1]
    if B[top] < _DOMINANT_CHANNEL_MIN:
        return None
    if B[top] - B[second] < _DOMINANT_DELTA_MIN:
        return None
    return int(top)


def chroma_cap(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    """Despill by capping the dominant-screen channel at the max of the others.

    For green screen B = (0, ~1, 0):
        F'.G = min(F.G, max(F.R, F.B))  -- the classic Vlahos cap
    For any saturated B with dominant channel d:
        F'.d = min(F.d, max(F[other_two]))

    For non-saturated B (black/grey/white) this is a no-op (returns input).

    ``strength`` in [0,1] interpolates between input and capped output. ``alpha``
    is unused here but kept in the signature for API uniformity.
    """
    del alpha
    F = image_linear.astype(np.float32)
    d = has_dominant_screen_channel(background_linear)
    if d is None:
        return F.copy()

    other = [i for i in range(3) if i != d]
    cap = np.maximum(F[..., other[0]], F[..., other[1]])
    new_d = np.minimum(F[..., d], cap)
    out = F.copy()
    out[..., d] = (1.0 - strength) * F[..., d] + strength * new_d
    return out


# ---------------------------------------------------------------------------
# (B) Local foreground borrowing (Photoshop "Decontaminate Colors")
# ---------------------------------------------------------------------------


def local_foreground_borrow(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
    fg_alpha_threshold: float = 0.95,
    band_alpha_low: float = 0.05,
    band_alpha_high: float = 0.95,
    k: int = 16,
    spatial_sigma: float = 24.0,
    color_sigma: float = 0.15,
    max_band_pixels: int | None = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """For each translucent pixel, replace its color with a KDTree-weighted
    average of the K nearest fully-opaque pixels' colors. Direct port of
    Photoshop "Decontaminate Colors" idea.

    Pixels with α >= ``fg_alpha_threshold`` are passed through unchanged.
    Pixels with α <= ``band_alpha_low`` are set to background-color (will be
    matted out anyway).
    """
    from scipy.spatial import cKDTree

    del background_linear  # not used; included for API symmetry
    F = image_linear.astype(np.float32).copy()
    h, w = F.shape[:2]

    fg_mask = alpha >= fg_alpha_threshold
    band_mask = (alpha > band_alpha_low) & (alpha < band_alpha_high)

    fg_ys, fg_xs = np.where(fg_mask)
    if fg_ys.size == 0:
        return F  # nothing to borrow from
    fg_coords = np.stack([fg_ys, fg_xs], axis=1).astype(np.float32)
    fg_colors = F[fg_ys, fg_xs]
    tree = cKDTree(fg_coords)

    band_ys, band_xs = np.where(band_mask)
    n = band_ys.size
    if n == 0:
        return F

    if max_band_pixels is not None and n > max_band_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_band_pixels, replace=False)
    else:
        idx = np.arange(n)

    sub_ys = band_ys[idx]
    sub_xs = band_xs[idx]
    sub_coords = np.stack([sub_ys, sub_xs], axis=1).astype(np.float32)
    sub_colors = F[sub_ys, sub_xs]

    k_eff = min(k, fg_coords.shape[0])
    dists, neigh_idx = tree.query(sub_coords, k=k_eff)
    dists = np.atleast_2d(dists)
    neigh_idx = np.atleast_2d(neigh_idx)

    neighbor_colors = fg_colors[neigh_idx]                                 # (n_sub, k, 3)
    color_diffs = neighbor_colors - sub_colors[:, None, :]
    color_dist = np.sqrt(np.sum(color_diffs * color_diffs, axis=-1))       # (n_sub, k)

    w_space = np.exp(-(dists ** 2) / (2 * spatial_sigma ** 2))
    w_color = np.exp(-(color_dist ** 2) / (2 * color_sigma ** 2))
    w = w_space * w_color
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum < 1e-8, 1.0, w_sum)
    borrowed = (neighbor_colors * w[..., None]).sum(axis=1) / w_sum
    F[sub_ys, sub_xs] = borrowed.astype(np.float32)

    # Fill any band pixels we skipped via nearest of computed ones.
    if max_band_pixels is not None and n > max_band_pixels:
        skipped = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        if skipped.size:
            sub_tree = cKDTree(sub_coords)
            sk_coords = np.stack([band_ys[skipped], band_xs[skipped]], axis=1).astype(np.float32)
            _, ni = sub_tree.query(sk_coords, k=1)
            F[band_ys[skipped], band_xs[skipped]] = borrowed[ni].astype(np.float32)

    return F


# ---------------------------------------------------------------------------
# (C) Closed-form matting (pymatting). Computes alpha *and* F jointly.
# ---------------------------------------------------------------------------


def closed_form_matting(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
    fg_threshold: float = 0.95,
    bg_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Run pymatting's closed-form alpha + ML foreground.

    Builds a trimap from the input alpha (sure_fg / sure_bg / unknown), then
    lets pymatting solve jointly for refined alpha and foreground.

    Returns (alpha_refined, foreground_linear). Slow on large images.
    """
    try:
        from pymatting import estimate_alpha_cf, estimate_foreground_ml
    except ImportError as e:
        raise ImportError(
            "closed_form despill requires pymatting. "
            "Install with `pip install pymatting`."
        ) from e

    del background_linear  # pymatting derives B from the trimap, not from us
    image = np.clip(image_linear, 0.0, 1.0).astype(np.float64)

    trimap = np.full(alpha.shape, 0.5, dtype=np.float64)
    trimap[alpha >= fg_threshold] = 1.0
    trimap[alpha <= bg_threshold] = 0.0

    refined_alpha = estimate_alpha_cf(image, trimap)
    foreground = estimate_foreground_ml(image, refined_alpha)
    return refined_alpha.astype(np.float32), foreground.astype(np.float32)


# ---------------------------------------------------------------------------
# (D) Closed-form unmix given known B and α
# ---------------------------------------------------------------------------


def unmix_foreground(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
    alpha_floor: float = 0.05,
    fallback_method: str = "local_borrow",
) -> np.ndarray:
    """Closed-form F given known B and α: ``F = (C - (1-α)·B) / α``.

    This is the textbook inverse of the compositing equation. It is exact when
    α and B are exact, and degrades smoothly elsewhere. Two safety nets:

      1. Pixels with α < ``alpha_floor`` cannot give a stable F (division
         blows up); for those we fall back to ``fallback_method`` (default:
         local KDTree borrow from sure_fg neighbors).
      2. The result is clipped to [0, 1] in linear RGB before return so a
         tiny B-measurement error doesn't produce out-of-gamut F values.
    """
    C = image_linear.astype(np.float32)
    B = np.asarray(background_linear, dtype=np.float32).reshape(3)
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)[..., None]

    safe_a = np.maximum(a, alpha_floor)
    F = (C - (1.0 - a) * B) / safe_a

    if fallback_method:
        low_alpha = a[..., 0] < alpha_floor
        if low_alpha.any():
            if fallback_method == "local_borrow":
                F_fallback = local_foreground_borrow(C, B, alpha)
            elif fallback_method == "background":
                F_fallback = np.broadcast_to(B, C.shape).copy()
            else:
                raise ValueError(f"Unknown fallback_method: {fallback_method!r}")
            F[low_alpha] = F_fallback[low_alpha]

    return np.clip(F, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def apply_despill(
    method: str,
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the requested despill, return (alpha_out, foreground_out).

    Methods:
      auto         : pick unmix when B is in-gamut (always); chroma_cap as
                     additive cleanup if B has a dominant saturated channel.
                     This is the recommended default given the system's
                     "known B" contract.
      unmix        : closed-form F = (C-(1-α)B)/α, plus low-α fallback.
      chroma_cap   : Vlahos cap on B's dominant channel + local_borrow.
                     Returns input unchanged if B has no dominant channel.
      local_borrow : KDTree color borrow from sure_fg neighbors only.
      closed_form  : pymatting joint α + F.
      none         : no despill, baseline.
    """
    if method == "none":
        return alpha, image_linear.astype(np.float32).copy()
    if method == "auto":
        F = unmix_foreground(image_linear, background_linear, alpha)
        if has_dominant_screen_channel(background_linear) is not None:
            F = chroma_cap(F, background_linear, alpha=alpha)
        return alpha, F
    if method == "unmix":
        F = unmix_foreground(image_linear, background_linear, alpha)
        return alpha, F
    if method == "chroma_cap":
        F = chroma_cap(image_linear, background_linear, alpha=alpha)
        # Then locally borrow on remaining halo pixels for safety.
        F = local_foreground_borrow(F, background_linear, alpha)
        return alpha, F
    if method == "local_borrow":
        F = local_foreground_borrow(image_linear, background_linear, alpha)
        return alpha, F
    if method == "closed_form":
        a, F = closed_form_matting(image_linear, background_linear, alpha)
        return a, F
    raise ValueError(f"Unknown despill method: {method!r}")


__all__ = [
    "has_dominant_screen_channel",
    "chroma_cap",
    "local_foreground_borrow",
    "closed_form_matting",
    "unmix_foreground",
    "apply_despill",
]
