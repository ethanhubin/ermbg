"""End-to-end matting test on a synthetic 'clean bg' image."""

from __future__ import annotations

import numpy as np

from ermbg import io
from ermbg.matting import matte
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
