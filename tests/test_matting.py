"""End-to-end matting test on a synthetic 'clean bg' image."""

from __future__ import annotations

import numpy as np

from ermbg import io
from ermbg.matting import _stabilize_foreground_for_export, matte
from ermbg.qa import recomposition_error


class _StubSegmenter:
    """Returns a precomputed soft mask, bypassing BiRefNet."""

    def __init__(self, soft_mask: np.ndarray):
        self.soft_mask = soft_mask

    def segment(self, image, object_prompt=None):
        return self.soft_mask


def _make_case(h=160, w=160, fg=(220, 60, 80), bg=(20, 20, 20)):
    fg_arr = np.array(fg, dtype=np.uint8)
    bg_arr = np.array(bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    alpha_gt = np.clip((50.0 - r) / 8.0, 0.0, 1.0).astype(np.float32)

    F_lin = io.srgb_to_linear(np.broadcast_to(fg_arr, (h, w, 3)))
    B_lin = io.srgb_to_linear(bg_arr.reshape(1, 1, 3))[0, 0]
    a = alpha_gt[..., None]
    C_lin = a * F_lin + (1.0 - a) * B_lin
    image = io.linear_to_srgb_u8(C_lin)
    return image, alpha_gt


def test_matte_end_to_end_synthetic():
    image, alpha_gt = _make_case()
    seg = _StubSegmenter(alpha_gt)
    result = matte(image, segmenter=seg)

    # diagnosis must say 'ready' (clean bg, well-separated colors).
    assert result.diagnosis.verdict == "ready"

    # alpha close to GT.
    err = np.abs(result.alpha - alpha_gt)
    assert err.mean() < 0.03

    # Recomposition error on the observed bg should be tiny.
    rec = recomposition_error(image, result.rgba, result.background_color)
    assert rec < 0.02


def test_matte_reuses_precomputed_soft_mask_without_segmenter_call():
    image, alpha_gt = _make_case()

    class _ExplodingSegmenter:
        def segment(self, image, object_prompt=None):
            raise AssertionError("segmenter should not be called when soft_mask is provided")

    result = matte(image, segmenter=_ExplodingSegmenter(), soft_mask=alpha_gt)

    assert result.alpha.shape == alpha_gt.shape
    assert np.abs(result.alpha - alpha_gt).mean() < 0.03


def test_matte_solid_graphic_prepass_skips_segmenter(monkeypatch):
    h, w = 96, 112
    image = np.full((h, w, 3), (0, 200, 0), dtype=np.uint8)
    image[20:70, 30:82] = (220, 40, 30)

    def explode_build_segmenter(*args, **kwargs):
        raise AssertionError("solid graphic prepass should avoid building segmenter")

    monkeypatch.setattr("ermbg.matting.build_segmenter", explode_build_segmenter)

    result = matte(image)

    assert result.debug["strategy"]["name"] == "solid_bg_graphic"
    assert result.debug["solid_graphic"]["accepted"] is True
    assert result.alpha[25:65, 35:77].mean() > 0.99
    assert float(result.alpha[:8, :8].max()) == 0.0


def test_matte_injected_segmenter_keeps_legacy_repair_path():
    h, w = 96, 96
    image = np.full((h, w, 3), (0, 200, 0), dtype=np.uint8)
    image[20:76, 24:72] = (25, 80, 220)

    soft = np.zeros((h, w), dtype=np.float32)
    soft[20:46, 24:72] = 1.0
    soft[46:76, 24:72] = 0.08

    result = matte(image, segmenter=_StubSegmenter(soft), shadow_mode="on")

    assert result.debug["strategy"]["name"] == "saturated_bg"
    assert "solid_graphic" not in result.debug


def test_matte_solid_graphic_owns_saturated_hard_edge_antialiasing(monkeypatch):
    """Default clean solid-screen graphics should resolve hard edges before the
    BiRefNet/keyer repair stack is involved."""
    h, w = 96, 96
    bg = np.array((0, 200, 0), dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    image[20:76, 20:76] = (40, 110, 230)
    image[20:22, 20:76] = (3, 170, 20)
    image[74:76, 20:76] = (3, 170, 20)
    image[20:76, 20:22] = (3, 170, 20)
    image[20:76, 74:76] = (3, 170, 20)

    def explode_build_segmenter(*args, **kwargs):
        raise AssertionError("solid graphic hard edge should not build segmenter")

    monkeypatch.setattr("ermbg.matting.build_segmenter", explode_build_segmenter)

    result = matte(image)

    assert result.debug["strategy"]["name"] == "solid_bg_graphic"
    assert result.alpha[28:68, 28:68].min() > 0.99
    assert result.debug["ownership_masks"]["soft_subject_layer"][20:22, 26:70].mean() > 0.95
    assert "saturated_hard_edge_key_resolve" not in result.debug["keyer"]


def test_matte_repairs_pale_panel_on_known_white_background():
    """A pale colored panel on known white should be repaired from full-color
    known-B evidence, not require an external subject mask."""
    h, w = 96, 96
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    image[20:76, 20:76] = (214, 236, 210)
    image[20:32, 20:76] = (50, 120, 60)  # dark attached foreground anchor

    soft = np.zeros((h, w), dtype=np.float32)
    soft[20:76, 20:76] = 1.0
    soft[42:58, 48:64] = 0.08  # model recall hole in the pale panel

    result = matte(image, segmenter=_StubSegmenter(soft))
    repair = result.debug["keyer"].get("known_bg_repair", {})

    assert result.debug["strategy"]["name"] == "white_bg"
    assert repair["accepted_components"] == 1
    assert repair["accepted_pixels"] > 0
    assert result.alpha[46:54, 52:60].mean() > 0.85


def test_matte_repairs_saturated_known_bg_low_alpha_interior():
    """A saturated-screen subject can be one connected key component where
    only part of the component has matting-net recall. The known-B repair
    should restore the internal low-alpha region without relying on a second
    subject mask."""
    h, w = 96, 96
    image = np.full((h, w, 3), (0, 200, 0), dtype=np.uint8)
    image[20:76, 24:72] = (25, 80, 220)
    image[20:34, 34:62] = (240, 230, 120)  # attached high-recall anchor

    soft = np.zeros((h, w), dtype=np.float32)
    soft[20:46, 24:72] = 1.0
    soft[46:76, 24:72] = 0.08

    result = matte(image, segmenter=_StubSegmenter(soft), shadow_mode="on")
    repair = result.debug["keyer"].get("saturated_known_bg_repair", {})

    assert result.debug["strategy"]["name"] == "saturated_bg"
    assert result.debug["keyer"]["patched_components"] == 0
    assert repair["accepted_components"] == 1
    assert repair["accepted_pixels"] > 0
    assert result.alpha[52:70, 30:66].mean() > 0.85


def test_matte_snaps_saturated_known_bg_opaque_interior():
    """A known-B hard asset interior should not remain semi-transparent when
    color evidence and topology both say it is opaque foreground."""
    h, w = 96, 96
    image = np.full((h, w, 3), (0, 200, 0), dtype=np.uint8)
    image[20:76, 20:76] = (40, 110, 230)
    image[34:62, 34:62] = (60, 130, 245)

    soft = np.zeros((h, w), dtype=np.float32)
    soft[20:76, 20:76] = 0.82
    soft[34:62, 34:62] = 1.0

    result = matte(image, segmenter=_StubSegmenter(soft), shadow_mode="on")
    repair = result.debug["keyer"].get("saturated_opaque_interior_repair", {})

    assert result.debug["strategy"]["name"] == "saturated_bg"
    assert repair["accepted_components"] == 1
    assert repair["accepted_pixels"] > 0
    assert result.alpha[26:70, 26:70].min() >= 0.97


def test_matte_resolves_saturated_hard_edge_from_known_bg_key():
    """For clean solid-screen graphics, key topology says only the exterior
    contour is soft; the center must not inherit under-opaque model alpha."""
    h, w = 96, 96
    bg = np.array((0, 200, 0), dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    image[20:76, 20:76] = (40, 110, 230)
    image[20:22, 20:76] = (3, 170, 20)  # exterior antialiasing band
    image[74:76, 20:76] = (3, 170, 20)
    image[20:76, 20:22] = (3, 170, 20)
    image[20:76, 74:76] = (3, 170, 20)

    soft = np.zeros((h, w), dtype=np.float32)
    soft[20:76, 20:76] = 0.82
    soft[20:22, 20:76] = 0.92  # model over-owns the ambiguous contour
    soft[74:76, 20:76] = 0.92
    soft[20:76, 20:22] = 0.92
    soft[20:76, 74:76] = 0.92

    result = matte(image, segmenter=_StubSegmenter(soft), shadow_mode="on")
    resolved = result.debug["keyer"].get("saturated_hard_edge_key_resolve", {})

    assert resolved["applied"] is True
    assert resolved["raised_pixels"] > 0
    assert resolved["lowered_pixels"] > 0
    assert result.alpha[28:68, 28:68].min() >= 0.98
    assert result.alpha[20:22, 26:70].mean() < 0.92


def test_matte_reclassifies_exterior_scalar_darkening_out_of_subject_alpha():
    """Darkened known background connected to the exterior is shadow/background
    evidence, not subject color to make opaque and chroma-cap to black."""
    h, w = 96, 96
    bg = np.array((0, 200, 0), dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    image[20:70, 22:74] = (40, 110, 230)
    image[74:82, 18:78] = io.linear_to_srgb_u8(
        io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0].reshape(1, 1, 3) * 0.45
    )[0, 0]

    soft = np.zeros((h, w), dtype=np.float32)
    soft[20:70, 22:74] = 1.0
    soft[74:82, 18:78] = 1.0  # model/keyer incorrectly owns exterior darkening

    result = matte(image, segmenter=_StubSegmenter(soft), shadow_mode="on")
    reclass = result.debug["keyer"].get("exterior_scalar_darkening_reclassified", {})

    assert reclass["pixels"] > 0
    assert result.debug["subject_alpha"][76:80, 24:72].mean() < 0.05


def test_foreground_export_extends_sure_material_into_background_dominated_rgb():
    """Straight foreground RGB is undefined where background dominates.

    Export should extend nearby sure material color there instead of exposing
    black/green inverse-composite artifacts as if they were valid foreground.
    """
    h, w = 24, 24
    foreground = np.zeros((h, w, 3), dtype=np.float32)
    foreground[8:16, 8:16] = io.srgb_to_linear(np.array([40, 110, 230], dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    subject_alpha = np.zeros((h, w), dtype=np.float32)
    subject_alpha[8:16, 8:16] = 1.0
    subject_alpha[6:18, 6:18] = np.maximum(subject_alpha[6:18, 6:18], 0.25)

    stabilized, info = _stabilize_foreground_for_export(foreground, subject_alpha)
    stabilized_srgb = io.linear_to_srgb_u8(stabilized)

    assert info["filled_pixels"] > 0
    assert stabilized_srgb[0, 0].tolist() == [40, 110, 230]
    assert stabilized_srgb[6, 12].tolist() == [40, 110, 230]


def test_foreground_export_cleans_measured_shadow_color_layer():
    """Standalone foreground RGB is a clean subject-color layer, not RGBA RGB."""
    h, w = 24, 24
    foreground = np.zeros((h, w, 3), dtype=np.float32)
    foreground[8:16, 8:16] = io.srgb_to_linear(np.array([40, 110, 230], dtype=np.uint8).reshape(1, 1, 3))[0, 0]
    subject_alpha = np.zeros((h, w), dtype=np.float32)
    subject_alpha[8:16, 8:16] = 1.0

    stabilized, info = _stabilize_foreground_for_export(foreground, subject_alpha)
    stabilized_srgb = io.linear_to_srgb_u8(stabilized)

    assert info["filled_pixels"] > 0
    assert stabilized_srgb[18, 10].tolist() == [40, 110, 230]
    assert stabilized_srgb[0, 0].tolist() == [40, 110, 230]


def test_matte_repairs_thin_hard_edge_outline_on_known_white_background():
    h, w = 96, 96
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    image[32:72, 24:72] = (230, 0, 0)
    image[31, 24:72] = (20, 20, 20)

    soft = np.zeros((h, w), dtype=np.float32)
    soft[32:72, 24:72] = 1.0
    soft[31, 24:72] = 0.25

    result = matte(image, segmenter=_StubSegmenter(soft))
    repair = result.debug["keyer"].get("hard_edge_repair", {})

    assert result.debug["strategy"]["name"] == "white_bg"
    assert repair["accepted_components"] == 1
    assert repair["accepted_pixels"] >= 40
    assert result.alpha[31, 26:70].min() > 0.94
