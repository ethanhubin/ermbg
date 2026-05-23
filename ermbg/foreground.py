"""Local foreground reference color estimation.

Plan section 17. For each unknown-band pixel, find K nearest sure_fg pixels
(spatial kNN) and weight them by spatial distance + color similarity to the
unknown pixel. Operates in linear RGB.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .types import Trimap


def estimate_foreground_reference(
    image_linear: np.ndarray,
    trimap: Trimap,
    k: int = 16,
    spatial_sigma: float = 24.0,
    max_unknown_pixels: int | None = 200_000,
    seed: int = 0,
) -> np.ndarray:
    """Return F_ref (H x W x 3, linear RGB) for every pixel.

    For sure_fg pixels, F_ref = the pixel itself.
    For sure_bg pixels, F_ref = 0 (unused downstream).
    For unknown pixels, F_ref is a kNN-weighted average of sure_fg neighbors.
    """
    h, w, _ = image_linear.shape
    F_ref = image_linear.copy()

    # In sure_bg, F_ref is irrelevant. Set to a neutral grey to avoid NaN when
    # something downstream divides by zero.
    F_ref[trimap.sure_bg] = 0.5

    fg_ys, fg_xs = np.where(trimap.sure_fg)
    if fg_ys.size == 0:
        # No definite foreground -> can't estimate. Return image as fallback.
        return F_ref

    fg_coords = np.stack([fg_ys, fg_xs], axis=1).astype(np.float32)
    fg_colors = image_linear[fg_ys, fg_xs]
    tree = cKDTree(fg_coords)

    un_ys, un_xs = np.where(trimap.unknown)
    n_unknown = un_ys.size
    if n_unknown == 0:
        return F_ref

    # Optional subsample to cap cost on huge images; we then fill the missing
    # unknown pixels by nearest neighbor of the computed ones.
    if max_unknown_pixels is not None and n_unknown > max_unknown_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_unknown, size=max_unknown_pixels, replace=False)
    else:
        idx = np.arange(n_unknown)

    sub_ys = un_ys[idx]
    sub_xs = un_xs[idx]
    sub_coords = np.stack([sub_ys, sub_xs], axis=1).astype(np.float32)
    sub_colors = image_linear[sub_ys, sub_xs]

    k_eff = min(k, fg_coords.shape[0])
    dists, neigh_idx = tree.query(sub_coords, k=k_eff)
    dists = np.atleast_2d(dists)
    neigh_idx = np.atleast_2d(neigh_idx)

    neighbor_colors = fg_colors[neigh_idx]  # (n_sub, k, 3)
    color_diffs = neighbor_colors - sub_colors[:, None, :]
    color_dist = np.sqrt(np.sum(color_diffs * color_diffs, axis=-1))  # (n_sub, k)

    # Combined weight: gaussian on space + gaussian on color.
    color_sigma = 0.15  # in linear-RGB units
    w_space = np.exp(-(dists ** 2) / (2 * spatial_sigma ** 2))
    w_color = np.exp(-(color_dist ** 2) / (2 * color_sigma ** 2))
    w = w_space * w_color
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum < 1e-8, 1.0, w_sum)
    f_ref_sub = (neighbor_colors * w[..., None]).sum(axis=1) / w_sum

    F_ref[sub_ys, sub_xs] = f_ref_sub.astype(np.float32)

    # Fill any unknown pixels we skipped via nearest of the *computed* unknown points.
    if max_unknown_pixels is not None and n_unknown > max_unknown_pixels:
        skipped = np.setdiff1d(np.arange(n_unknown), idx, assume_unique=False)
        if skipped.size:
            comp_tree = cKDTree(sub_coords)
            sk_coords = np.stack([un_ys[skipped], un_xs[skipped]], axis=1).astype(np.float32)
            _, ni = comp_tree.query(sk_coords, k=1)
            F_ref[un_ys[skipped], un_xs[skipped]] = f_ref_sub[ni].astype(np.float32)

    return F_ref


__all__ = ["estimate_foreground_reference"]
