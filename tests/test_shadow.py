"""Tests for known-background shadow extraction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ermbg import io
from ermbg.planner import RiskRegion
from ermbg.shadow import (
    ShadowPrior,
    _remove_small_visible_shadow_components,
    estimate_shadow_alpha,
    exterior_scalar_darkening_mask,
    remove_small_display_shadow_components,
    shadow_alpha_to_display_alpha,
    shadow_prior_from_regions,
)

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


def _white_subject_with_hard_shadow_and_weak_tail(h: int = 128, w: int = 160):
    bg = np.array([255, 255, 255], dtype=np.uint8)
    fg = np.array([220, 30, 30], dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.float32)
    subject[30:68, 48:104] = 1.0

    shadow_alpha = np.zeros((h, w), dtype=np.float32)
    shadow_alpha[76:94, 42:118] = 0.50
    shadow_alpha[72:100, 30:132] = np.maximum(shadow_alpha[72:100, 30:132], 0.012)
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


def test_estimate_shadow_alpha_smooths_soft_shadow_measurement_noise():
    """Soft cast-shadow opacity should be a coherent field, not salt noise.

    The dark salt pixel is still scalar-darkened known background, but it is a
    measurement outlier inside an otherwise soft shadow. The recovered alpha
    should keep the soft shadow while suppressing the isolated spike.
    """
    h, w = 96, 128
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([220, 30, 30], dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.float32)
    subject[26:58, 38:88] = 1.0

    shadow_alpha = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(shadow_alpha, (68, 66), (38, 12), 0.0, 0.0, 360.0, 1.0, -1)
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (13, 13), sigmaX=4.0) * 0.18
    shadow_alpha[72, 78] = 0.75
    shadow_alpha[subject > 0] = 0.0

    B_lin = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    F_lin = io.srgb_to_linear(fg.reshape(1, 1, 3))[0, 0]
    bg_shadow = (1.0 - shadow_alpha[..., None]) * B_lin
    image = io.linear_to_srgb_u8(subject[..., None] * F_lin + (1.0 - subject[..., None]) * bg_shadow)

    recovered, info = estimate_shadow_alpha(image, subject, tuple(int(c) for c in bg))

    local = recovered[70:75, 76:81]
    assert info["detected"] is True
    assert info["boundary"]["boundary_mode"] == "soft"
    assert float(recovered[72, 78]) < 0.35
    assert float(local.max() - local.mean()) < 0.08


def test_estimate_shadow_alpha_removes_tiny_visible_shadow_islands():
    """Final visible shadow alpha should not contain tiny detached islands."""
    alpha = np.zeros((32, 48), dtype=np.float32)
    alpha[14:22, 16:34] = 0.25
    alpha[4:6, 4:6] = 0.25

    filtered, info = _remove_small_visible_shadow_components(
        alpha,
        min_area=8.0,
        core_alpha_threshold=0.08,
    )

    assert info["small_visible_components_removed"] == 1
    assert info["small_visible_pixels_removed"] == 4
    assert float(filtered[4:6, 4:6].max()) == 0.0
    assert float(filtered[14:22, 16:34].mean()) == 0.25


def test_display_shadow_filter_removes_tiny_visible_islands():
    alpha = np.zeros((32, 48), dtype=np.float32)
    alpha[14:22, 16:34] = 20.0 / 255.0
    alpha[4:6, 4:6] = 20.0 / 255.0
    alpha[5:10, 6] = 2.0 / 255.0  # sub-visible bridge should not connect the island.

    filtered, info = remove_small_display_shadow_components(alpha, min_area=8.0)

    assert info["display_small_components_removed"] == 1
    assert info["display_small_pixels_removed"] == 4
    assert float(filtered[4:6, 4:6].max()) == 0.0
    assert float(filtered[5:10, 6].max()) > 0.0
    assert float(filtered[14:22, 16:34].mean()) == pytest.approx(20.0 / 255.0)


def test_estimate_shadow_alpha_preserves_hard_shadow_boundary_mode():
    image, subject, shadow_gt, bg = _green_subject_with_hard_shadow()

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg)

    hard_core = (shadow_gt > 0.4) & (subject < 0.1)
    assert info["detected"] is True
    assert info["boundary"]["boundary_mode"] == "hard"
    assert info["boundary"]["boundary_falloff_px"] <= 2.0
    assert shadow_alpha[hard_core].mean() > 0.35


def test_estimate_shadow_alpha_keeps_hard_boundary_with_weak_tail_support():
    image, subject, shadow_gt, bg = _white_subject_with_hard_shadow_and_weak_tail()

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg)

    hard_core = (shadow_gt > 0.4) & (subject < 0.1)
    weak_tail = (shadow_gt > 0.0) & (shadow_gt < 0.05) & (subject < 0.1)
    assert info["detected"] is True
    assert info["boundary"]["boundary_mode"] == "hard"
    assert info["boundary"]["boundary_falloff_px"] <= 2.0
    assert info["boundary"]["support_expansion_ratio"] >= 2.5
    assert shadow_alpha[hard_core].mean() > 0.35
    assert shadow_alpha[weak_tail].mean() < shadow_alpha[hard_core].mean()


def test_shadow_alpha_to_display_alpha_matches_srgb_viewer_composite():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    physical = np.array([[0.0, 0.5, 0.8]], dtype=np.float32)

    display = shadow_alpha_to_display_alpha(physical, tuple(int(c) for c in bg))

    assert float(display[0, 0]) == 0.0
    assert float(display[0, 1]) < float(physical[0, 1])
    assert float(display[0, 2]) < float(physical[0, 2])

    bg_srgb = bg.astype(np.float32) / 255.0
    viewer_comp = (1.0 - display[..., None]) * bg_srgb.reshape(1, 1, 3)
    target = io.linear_to_srgb(
        (1.0 - physical[..., None])
        * io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0].reshape(1, 1, 3)
    )
    assert np.abs(viewer_comp[..., 1] - target[..., 1]).max() < 1e-5


def test_estimate_shadow_alpha_rejects_clean_background():
    image, subject, _, bg = _green_subject_with_shadow(shadow=False)

    shadow_alpha, info = estimate_shadow_alpha(image, subject, bg)

    assert info["detected"] is False
    assert info["pixels"] == 0
    assert float(shadow_alpha.max()) == 0.0


def test_estimate_shadow_alpha_rejects_outline_residue_as_shadow():
    """Dark semitransparent edge residue is not a cast shadow.

    The residue satisfies the same scalar-darkening equation as a physical
    shadow, but it is tightly glued to the subject boundary and overlaps the
    subject's own soft alpha. That geometry identifies antialiasing/outline
    contamination rather than exterior shadow ownership.
    """
    h, w = 96, 160
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    subject = np.zeros((h, w), dtype=np.float32)
    subject[28:64, 38:122] = 1.0

    residue = np.zeros((h, w), dtype=bool)
    residue[24:28, 40:120] = True
    residue[64:68, 40:120] = True
    residue[30:64, 34:38] = True
    residue[30:64, 122:126] = True
    subject[residue] = 0.12
    image[residue] = _scaled_background_color(bg, 0.55)
    image[subject >= 1.0] = (230, 170, 20)

    shadow_alpha, info = estimate_shadow_alpha(image, subject, tuple(int(c) for c in bg))

    assert info["detected"] is False
    assert info["boundary"]["boundary_residue_rejected"] is True
    assert float(shadow_alpha[residue].max()) == 0.0


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


def test_subject_material_prior_excludes_translucent_darkening_from_shadow():
    h, w = 96, 128
    bg = np.array([255, 255, 255], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    subject_alpha = np.zeros((h, w), dtype=np.float32)
    subject_alpha[26:62, 36:92] = 0.45
    image[26:62, 36:92] = _scaled_background_color(bg, 0.70)

    material = np.zeros((h, w), dtype=np.float32)
    material[24:64, 34:94] = 1.0
    prior = ShadowPrior(subject_material_mask=material, source="translucent_material")

    shadow_alpha, info = estimate_shadow_alpha(
        image,
        subject_alpha,
        tuple(int(c) for c in bg),
        prior=prior,
    )

    assert info["detected"] is False
    assert info["prior"]["has_subject_material_mask"] is True
    assert float(shadow_alpha[material > 0].max()) == 0.0


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


def test_exterior_scalar_darkening_mask_keeps_internal_subject_material():
    h, w = 64, 80
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    exterior_shadow = np.zeros((h, w), dtype=bool)
    exterior_shadow[50:56, 10:70] = True
    image[exterior_shadow] = _scaled_background_color(bg, 0.55)
    internal_material = np.zeros((h, w), dtype=bool)
    internal_material[24:36, 30:48] = True
    image[internal_material] = _scaled_background_color(bg, 0.55)

    known_bg = np.ones((h, w), dtype=bool)
    known_bg[18:44, 24:56] = False
    mask, info = exterior_scalar_darkening_mask(image, tuple(int(c) for c in bg), known_bg)

    assert info["pixels"] > 0
    assert mask[exterior_shadow].mean() > 0.95
    assert not mask[internal_material].any()


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

