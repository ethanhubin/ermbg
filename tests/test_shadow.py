"""Tests for known-background shadow extraction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ermbg import io
from ermbg.matting import matte
from ermbg.planner import RiskRegion
from ermbg.shadow import ShadowPrior, estimate_shadow_alpha, shadow_prior_from_regions

pytestmark = pytest.mark.core


class _StubSegmenter:
    def __init__(self, alpha: np.ndarray):
        self.alpha = alpha.astype(np.float32)

    def segment(self, image, object_prompt=None):
        del image, object_prompt
        return self.alpha


def _green_subject_with_shadow(h: int = 96, w: int = 128, shadow: bool = True):
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([220, 30, 30], dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.float32)
    subject[26:58, 38:88] = 1.0

    shadow_alpha = np.zeros((h, w), dtype=np.float32)
    if shadow:
        hard = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(hard, (68, 66), (38, 12), 0.0, 0.0, 360.0, 1.0, -1)
        shadow_alpha = cv2.GaussianBlur(hard, (13, 13), sigmaX=4.0) * 0.45
        shadow_alpha[subject > 0] = 0.0

    B_lin = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    F_lin = io.srgb_to_linear(fg.reshape(1, 1, 3))[0, 0]
    bg_shadow = (1.0 - shadow_alpha[..., None]) * B_lin
    C_lin = subject[..., None] * F_lin + (1.0 - subject[..., None]) * bg_shadow
    image = io.linear_to_srgb_u8(C_lin)
    return image, subject, shadow_alpha, tuple(int(c) for c in bg)


def _green_subject_with_hard_shadow(h: int = 96, w: int = 128):
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([220, 30, 30], dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.float32)
    subject[24:54, 38:86] = 1.0

    shadow_alpha = np.zeros((h, w), dtype=np.float32)
    shadow_alpha[60:78, 44:102] = 0.48
    shadow_alpha[subject > 0] = 0.0

    B_lin = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    F_lin = io.srgb_to_linear(fg.reshape(1, 1, 3))[0, 0]
    bg_shadow = (1.0 - shadow_alpha[..., None]) * B_lin
    C_lin = subject[..., None] * F_lin + (1.0 - subject[..., None]) * bg_shadow
    image = io.linear_to_srgb_u8(C_lin)
    return image, subject, shadow_alpha, tuple(int(c) for c in bg)


def _scaled_background_color(bg: np.ndarray, scale: float) -> np.ndarray:
    scaled = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0] * float(scale)
    return io.linear_to_srgb_u8(scaled.reshape(1, 1, 3))[0, 0]


def test_estimate_shadow_alpha_detects_scalar_darkening_shadow():
    image, subject, shadow_gt, bg = _green_subject_with_shadow(shadow=True)

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg)

    shadow_core = (shadow_gt > 0.25) & (subject < 0.1)
    shadow_edge = (shadow_gt > 0.02) & (shadow_gt < 0.07) & (subject < 0.1)
    assert info["detected"] is True
    assert info["pixels"] > 300
    assert shadow_alpha[shadow_core].mean() > 0.20
    assert (shadow_alpha[shadow_edge] > 0.0).mean() > 0.20
    assert shadow_alpha[shadow_edge].mean() < shadow_alpha[shadow_core].mean()
    assert ((shadow_alpha > 0.0) & (shadow_alpha < 0.08)).sum() > 0
    assert info["boundary"]["boundary_mode"] == "soft"


def test_estimate_shadow_alpha_preserves_hard_shadow_boundary_mode():
    image, subject, shadow_gt, bg = _green_subject_with_hard_shadow()

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg)

    hard_core = (shadow_gt > 0.4) & (subject < 0.1)
    assert info["detected"] is True
    assert info["boundary"]["boundary_mode"] == "hard"
    assert info["boundary"]["boundary_falloff_px"] <= 2.0
    assert shadow_alpha[hard_core].mean() > 0.35


def test_estimate_shadow_alpha_rejects_clean_background():
    image, subject, _, bg = _green_subject_with_shadow(shadow=False)

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg)

    assert info["detected"] is False
    assert info["pixels"] == 0
    assert float(shadow_alpha.max()) == 0.0


def test_subject_prior_prevents_dark_subject_interior_becoming_shadow():
    h, w = 96, 128
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    subject_prior = np.zeros((h, w), dtype=np.float32)
    subject_prior[24:66, 36:92] = 1.0
    image[24:66, 36:92] = (210, 35, 35)

    hole = np.zeros((h, w), dtype=bool)
    hole[38:55, 52:76] = True
    image[hole] = _scaled_background_color(bg, 0.58)

    undersegmented_alpha = subject_prior.copy()
    undersegmented_alpha[hole] = 0.0
    prior = ShadowPrior(subject_mask=subject_prior, source="vlm_subject")

    shadow_alpha, info = estimate_shadow_alpha(
        image,
        undersegmented_alpha,
        tuple(int(c) for c in bg),
        prior=prior,
    )

    assert info["detected"] is False
    assert info["prior"]["has_subject_mask"] is True
    assert float(shadow_alpha[hole].max()) == 0.0


def test_shadow_search_prior_constrains_scalar_darkening_evidence():
    image, subject, _, bg = _green_subject_with_shadow(shadow=False)
    bg_arr = np.array(bg, dtype=np.uint8)
    dark_patch = np.zeros(subject.shape, dtype=bool)
    dark_patch[70:88, 84:118] = True
    image[dark_patch] = _scaled_background_color(bg_arr, 0.55)

    search = np.zeros(subject.shape, dtype=np.float32)
    search[58:82, 34:78] = 1.0
    prior = ShadowPrior(shadow_search_mask=search, source="vlm_shadow_search")

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg, prior=prior)

    assert info["detected"] is False
    assert info["prior"]["has_shadow_search_mask"] is True
    assert float(shadow_alpha[dark_patch].max()) == 0.0


def test_shadow_prior_from_planner_regions_maps_semantic_masks():
    shape = (24, 32)
    subject_mask = np.zeros(shape, dtype=bool)
    subject_mask[5:18, 8:21] = True
    shadow_mask = np.zeros(shape, dtype=bool)
    shadow_mask[16:21, 7:24] = True

    prior = shadow_prior_from_regions(
        [
            RiskRegion(id="subject_0", kind="subject_owned_region", mask=subject_mask),
            RiskRegion(id="shadow_0", kind="owned_shadow_candidate", mask=shadow_mask),
        ],
        shape,
        source="vlm_fixture",
    )

    assert prior.source == "vlm_fixture"
    assert prior.subject_mask is not None
    assert prior.shadow_ownership_mask is not None
    assert int(prior.subject_mask.sum()) == int(subject_mask.sum())
    assert int(prior.shadow_ownership_mask.sum()) == int(shadow_mask.sum())


def test_matte_composites_detected_shadow_behind_subject():
    image, subject, shadow_gt, _ = _green_subject_with_shadow(shadow=True)

    result = matte(image, segmenter=_StubSegmenter(subject))
    shadow_core = (shadow_gt > 0.25) & (subject < 0.1)

    assert result.debug["shadow"]["detected"] is True
    assert result.debug["subject_alpha"][shadow_core].mean() < 0.05
    assert result.debug["shadow_alpha"][shadow_core].mean() > 0.20
    assert result.alpha[shadow_core].mean() > 0.20
