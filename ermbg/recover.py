"""Foreground RGB recovery and background-color decontamination.

Plan §22-24. Operates in linear RGB.
"""

from __future__ import annotations

import numpy as np


def recover_foreground(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
    f_ref: np.ndarray,
    alpha_safe: float = 0.05,
) -> np.ndarray:
    """F = (C - (1 - α) B) / α  with safe blending for low alpha.

    Above α_safe, we use the recovered F. Below, F = F_ref (so we don't blow up
    color noise). In the transition zone we blend.
    """
    C = image_linear
    B = np.broadcast_to(background_linear, C.shape) if background_linear.ndim == 1 else background_linear
    a = alpha[..., None]

    # Recovered F via inversion.
    F_recovered = (C - (1.0 - a) * B) / np.maximum(a, alpha_safe)

    # Soft selector between F_recovered and F_ref by alpha.
    # alpha >= 0.8: trust recovered
    # 0.2 <= alpha < 0.8: blend
    # alpha < 0.2: trust F_ref
    w = np.clip((alpha - 0.2) / (0.8 - 0.2), 0.0, 1.0)[..., None]
    F = w * F_recovered + (1.0 - w) * f_ref
    return np.clip(F, 0.0, 1.0).astype(np.float32)


def decontaminate(
    foreground_linear: np.ndarray,
    f_ref: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Project away the background-color component from F (plan §23.3).

        spill_dir = normalize(B - F_ref)
        pollution = max(0, <F - F_ref, spill_dir>)
        F_clean   = F - k(α) * pollution * spill_dir

    k(α) is highest in the soft-edge band 0.2..0.6 and decays at the extremes,
    matching the plan.
    """
    F = foreground_linear
    B = np.broadcast_to(background_linear, F.shape) if background_linear.ndim == 1 else background_linear

    spill = B - f_ref
    spill_norm = np.linalg.norm(spill, axis=-1, keepdims=True)
    spill_dir = np.where(spill_norm > 1e-6, spill / np.maximum(spill_norm, 1e-6), 0.0)

    diff = F - f_ref
    pollution = np.sum(diff * spill_dir, axis=-1, keepdims=True)
    pollution = np.clip(pollution, 0.0, None)

    # k(α) is a soft bump centered on alpha=0.4.
    a = alpha[..., None]
    k = np.exp(-((a - 0.4) ** 2) / (2 * 0.25 ** 2))
    k = k * strength

    F_clean = F - k * pollution * spill_dir
    return np.clip(F_clean, 0.0, 1.0).astype(np.float32)


def fix_halo(
    image_linear: np.ndarray,
    background_linear: np.ndarray,
    alpha: np.ndarray,
    color_threshold: float = 6.0,   # OKLab ΔE
    sharpness: float = 1.5,
) -> np.ndarray:
    """Plan §24: where the observed pixel is colorimetrically very close to B,
    drive α towards 0 instead of trusting whatever the alpha estimator produced.

    Uses an OKLab-distance gate (perceptually uniform) and a smooth ramp so it
    won't introduce hard edges:

        d = OKLab_distance(C, B)
        gate = clamp((d / color_threshold) ** sharpness, 0, 1)
        alpha_out = min(alpha, gate)
    """
    from .colorspace import linear_rgb_to_oklab, oklab_distance

    C_lab = linear_rgb_to_oklab(image_linear)
    B_lab = linear_rgb_to_oklab(
        np.broadcast_to(background_linear, (1, 1, 3))
    ).reshape(3)
    d = oklab_distance(C_lab, B_lab)
    gate = np.clip((d / color_threshold) ** sharpness, 0.0, 1.0).astype(np.float32)
    return np.minimum(alpha, gate)


__all__ = ["recover_foreground", "decontaminate", "fix_halo"]
