"""Tests for the isolated solid-background graphic engine."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ermbg import io, matte_image
from ermbg.colorspace import oklab_distance, srgb_to_oklab
from ermbg.qa import composite
from ermbg.solid_graphic import analyze_solid_bg_graphic, _local_background_family_continuity_mask

pytestmark = pytest.mark.core


def _composite(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    B = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    F = io.srgb_to_linear(fg.reshape(1, 1, 3))[0, 0]
    C = alpha[..., None] * F + (1.0 - alpha[..., None]) * B
    return io.linear_to_srgb_u8(C)


def _rgb_neighbor_jump_percentile(image: np.ndarray, mask: np.ndarray, crop: tuple[int, int, int, int], percentile: float) -> float:
    x0, y0, x1, y1 = crop
    pixels = image[y0:y1, x0:x1].astype(np.float32)
    local_mask = mask[y0:y1, x0:x1]
    dx = np.sqrt(np.mean((pixels[:, 1:] - pixels[:, :-1]) ** 2, axis=2))
    dy = np.sqrt(np.mean((pixels[1:, :] - pixels[:-1, :]) ** 2, axis=2))
    mx = local_mask[:, 1:] & local_mask[:, :-1]
    my = local_mask[1:, :] & local_mask[:-1, :]
    values = np.concatenate([dx[mx].ravel(), dy[my].ravel()])
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def test_crisp_graphic_on_green_uses_exterior_topology():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (96, 112, 3)).copy()
    image[20:70, 30:82] = np.array([220, 40, 30], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.background_color == (0, 200, 0)
    assert result.alpha[25:65, 35:77].mean() == pytest.approx(1.0)
    assert float(result.alpha[:8, :8].max()) == 0.0
    assert result.ownership_masks["opaque_subject"][20:70, 30:82].mean() > 0.99


def test_subject_touching_corner_uses_dominant_border_background():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (96, 112, 3)).copy()
    image[:58, :64] = np.array([220, 40, 30], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.background_color == (0, 200, 0)
    assert result.debug["background"]["source"] == "border_mode"
    assert result.alpha[8:48, 8:54].mean() == pytest.approx(1.0)
    assert float(result.alpha[-8:, -8:].max()) == 0.0


def test_detached_tiny_subject_colored_specks_do_not_become_subject():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (96, 112, 3)).copy()
    image[20:70, 30:82] = np.array([220, 40, 30], dtype=np.uint8)
    image[8:10, 96:98] = np.array([220, 40, 30], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.alpha[25:65, 35:77].mean() == pytest.approx(1.0)
    assert float(result.alpha[8:10, 96:98].max()) == 0.0
    assert result.ownership_masks["unknown_fallback"][8:10, 96:98].mean() == pytest.approx(1.0)


@pytest.mark.parametrize("bg", [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_low_chroma_solid_backgrounds_are_supported(bg):
    bg_arr = np.array(bg, dtype=np.uint8)
    image = np.broadcast_to(bg_arr, (80, 90, 3)).copy()
    image[18:58, 24:66] = np.array([190, 35, 150], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.background_color == bg
    assert result.alpha[24:52, 30:60].mean() > 0.99
    assert float(result.alpha[:6, :6].max()) == 0.0


def test_enclosed_background_basin_becomes_subject_hole():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (96, 96, 3)).copy()
    image[18:78, 18:78] = np.array([230, 55, 35], dtype=np.uint8)
    image[38:58, 38:58] = bg

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.ownership_masks["subject_hole"][40:56, 40:56].mean() > 0.95
    assert float(result.alpha[42:54, 42:54].max()) == 0.0
    assert result.alpha[24:34, 24:34].mean() == pytest.approx(1.0)


def test_scalar_darkening_inside_hole_is_neutral_shadow_not_subject_green():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (112, 112, 3)).copy()
    image[18:94, 18:94] = np.array([230, 55, 35], dtype=np.uint8)
    image[40:72, 40:72] = bg

    B = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    dark_bg = io.linear_to_srgb_u8((B * 0.38).reshape(1, 1, 3))[0, 0]
    image[40:72, 40:44] = dark_bg
    image[68:72, 40:72] = dark_bg

    result = analyze_solid_bg_graphic(image)
    shadow_layer = result.ownership_masks["shadow_layer"]
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)

    assert result.accepted is True
    assert result.ownership_masks["subject_hole"][48:64, 48:64].mean() > 0.95
    assert result.debug["internal_hole_shadow_pixels"] >= 200
    assert shadow_layer[42:70, 41:43].mean() > 0.95
    assert int(rgba_rgb[shadow_layer, 1].max()) == 0

    rgba = np.dstack([rgba_rgb, (result.alpha * 255.0 + 0.5).astype(np.uint8)])
    reconstructed = composite(rgba, tuple(int(x) for x in bg))
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rec_y = (io.srgb_to_linear(reconstructed[shadow_layer]) * weights.reshape(1, 3)).sum(axis=1)
    src_y = (io.srgb_to_linear(image[shadow_layer]) * weights.reshape(1, 3)).sum(axis=1)
    assert float(np.abs(rec_y - src_y).max()) < 0.003


def test_dark_background_family_inside_subject_can_prove_hole_shadow_without_exact_seed():
    bg = np.array([10, 190, 18], dtype=np.uint8)
    image = np.broadcast_to(bg, (112, 112, 3)).copy()
    image[18:94, 18:94] = np.array([230, 55, 35], dtype=np.uint8)

    B = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    dark_bg = io.linear_to_srgb_u8((B * 0.08).reshape(1, 1, 3))[0, 0]
    image[44:70, 48:66] = dark_bg

    result = analyze_solid_bg_graphic(image)
    shadow_layer = result.ownership_masks["shadow_layer"]
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)

    assert result.accepted is True
    assert result.debug["internal_hole_shadow_pixels"] >= 400
    assert shadow_layer[46:68, 50:64].mean() > 0.95
    assert int(rgba_rgb[shadow_layer, 1].max()) == 0


def test_same_color_thin_internal_decoration_does_not_become_hole():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (96, 96, 3)).copy()
    image[18:78, 18:78] = np.array([230, 55, 35], dtype=np.uint8)
    image[44:47, 28:68] = bg

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert int(result.ownership_masks["subject_hole"].sum()) == 0
    assert result.alpha[44:47, 28:68].mean() == pytest.approx(1.0)
    assert result.ownership_masks["opaque_subject"][44:47, 28:68].mean() > 0.99


def test_exterior_scalar_shadow_is_separate_from_subject_alpha():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([230, 55, 35], dtype=np.uint8)
    subject = np.zeros((104, 128), dtype=np.float32)
    subject[24:62, 42:86] = 1.0
    shadow = np.zeros_like(subject)
    shadow[70:86, 34:98] = 0.45

    B = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    F = io.srgb_to_linear(fg.reshape(1, 1, 3))[0, 0]
    C = subject[..., None] * F + (1.0 - subject[..., None]) * ((1.0 - shadow[..., None]) * B)
    image = io.linear_to_srgb_u8(C)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.ownership_masks["shadow_layer"][72:84, 40:92].mean() > 0.95
    assert result.ownership_masks["opaque_subject"][30:56, 48:80].mean() > 0.99
    assert result.debug["mask_pixels"]["shadow_layer"] > 500
    assert result.debug["exterior_shadow_feather"]["applied"] is False
    assert result.debug["shadow_luminance_reconstruction"]["max_luminance_abs_error"] < 0.002
    rgba = np.dstack(
        [
            io.linear_to_srgb_u8(result.rgba_rgb_linear),
            (np.clip(result.alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    shadow_layer = result.ownership_masks["shadow_layer"]
    reconstructed = composite(rgba, tuple(int(x) for x in bg))
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rec_y = (io.srgb_to_linear(reconstructed[shadow_layer]) * weights.reshape(1, 3)).sum(axis=1)
    src_y = (io.srgb_to_linear(image[shadow_layer]) * weights.reshape(1, 3)).sum(axis=1)
    # The reusable shadow layer should not carry green-screen chroma. It is a
    # neutral darkening layer whose known-B recomposite preserves brightness,
    # which is the perceptual constraint Ethan chose over exact RGB replay.
    assert int(io.linear_to_srgb_u8(result.rgba_rgb_linear)[shadow_layer, 1].max()) == 0
    assert float(np.abs(rec_y - src_y).max()) < 0.003


def test_subject_owned_glow_remains_soft_layer_not_background():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    fg = np.array([230, 55, 35], dtype=np.uint8)
    alpha = np.zeros((112, 112), dtype=np.float32)
    alpha[38:74, 38:74] = 1.0
    glow = np.zeros_like(alpha)
    cv2.rectangle(glow, (30, 30), (82, 82), 1.0, -1)
    glow = cv2.GaussianBlur(glow, (15, 15), sigmaX=4.0) * 0.35
    alpha = np.maximum(alpha, glow)
    image = _composite(bg, fg, alpha)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    soft = result.ownership_masks["soft_subject_layer"]
    assert int(soft.sum()) > 0
    assert result.alpha[34:38, 44:68].mean() > 0.05
    assert float(result.ownership_masks["shadow_layer"].sum()) == 0.0


def test_low_alpha_background_green_leak_in_soft_layer_is_removed():
    path = Path("samples/vlm_eval_game/ui_icon_glow_soft_hard/green.png")
    if not path.exists():
        pytest.skip("G06 game sample is not present")
    image = io.load_rgb(path)

    result = analyze_solid_bg_graphic(image)
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)
    alpha_u8 = (result.alpha * 255.0 + 0.5).astype(np.uint8)
    bg = np.asarray(result.background_color, dtype=np.uint8)
    dominant = int(np.argmax(bg))
    other_channels = [idx for idx in range(3) if idx != dominant]
    green_margin = rgba_rgb[..., dominant].astype(np.int16) - np.maximum(
        rgba_rgb[..., other_channels[0]].astype(np.int16),
        rgba_rgb[..., other_channels[1]].astype(np.int16),
    )
    low_alpha_green = (
        (alpha_u8 > 0)
        & (alpha_u8 <= 90)
        & (green_margin > 15)
        & ~result.ownership_masks["shadow_layer"]
    )

    assert result.accepted is True
    assert result.debug["soft_background_leak_pixels"] > 0
    assert int(low_alpha_green.sum()) < 120


def test_translucent_glass_interior_background_mix_is_removed():
    bg = np.array([6, 181, 18], dtype=np.uint8)
    image = np.broadcast_to(bg, (128, 144, 3)).copy()
    image[20:108, 18:126] = np.array([230, 246, 252], dtype=np.uint8)
    image[34:94, 34:110] = np.array([120, 220, 135], dtype=np.uint8)
    image[46:82, 48:96] = np.array([42, 192, 55], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)
    alpha_u8 = (result.alpha * 255.0 + 0.5).astype(np.uint8)
    bg_arr = np.asarray(result.background_color, dtype=np.uint8)
    dominant = int(np.argmax(bg_arr))
    other_channels = [idx for idx in range(3) if idx != dominant]
    green_margin = rgba_rgb[..., dominant].astype(np.int16) - np.maximum(
        rgba_rgb[..., other_channels[0]].astype(np.int16),
        rgba_rgb[..., other_channels[1]].astype(np.int16),
    )
    residual = (alpha_u8 > 0) & (green_margin > 15)

    assert result.accepted is True
    assert result.debug["soft_background_leak_pixels"] >= 1500
    assert float(result.alpha[50:78, 52:92].max()) == 0.0
    assert int(residual[46:82, 48:96].sum()) == 0


def test_opaque_background_family_flecks_inside_glass_do_not_turn_grey():
    bg = np.array([6, 181, 18], dtype=np.uint8)
    image = np.broadcast_to(bg, (128, 144, 3)).copy()
    image[20:108, 18:126] = np.array([230, 246, 252], dtype=np.uint8)
    image[34:94, 34:110] = np.array([120, 220, 135], dtype=np.uint8)
    # Small lifted screen-color flecks inside the glass basin are the G04
    # failure mode: if classified as opaque and then de-greened, they become
    # black/grey dirt. The surrounding glass/hole context should delete them.
    fleck_color = np.array([38, 191, 52], dtype=np.uint8)
    image[56:60, 58:66] = fleck_color
    image[68:72, 76:86] = fleck_color
    image[48:50, 92:110] = fleck_color

    result = analyze_solid_bg_graphic(image)
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)
    alpha_u8 = (result.alpha * 255.0 + 0.5).astype(np.uint8)
    flecks = np.zeros(alpha_u8.shape, dtype=bool)
    flecks[56:60, 58:66] = True
    flecks[68:72, 76:86] = True
    flecks[48:50, 92:110] = True
    grey_dirt = (
        flecks
        & (alpha_u8 > 0)
        & (rgba_rgb[..., 0] < 80)
        & (rgba_rgb[..., 1] < 90)
        & (rgba_rgb[..., 2] < 90)
    )

    assert result.accepted is True
    assert result.debug["opaque_glass_leak_pixels"] >= int(flecks.sum())
    assert int(grey_dirt.sum()) == 0
    assert float(result.alpha[flecks].max()) == 0.0


def test_background_family_threshold_diffuses_from_local_neighbors():
    bg = np.array([6, 181, 18], dtype=np.uint8)
    image = np.broadcast_to(bg, (64, 72, 3)).copy()
    context = np.zeros(image.shape[:2], dtype=bool)
    context[18:46, 18:54] = True
    image[context] = np.array([52, 198, 66], dtype=np.uint8)
    local_only = np.zeros(image.shape[:2], dtype=bool)
    local_only[29:35, 32:40] = True
    image[local_only] = np.array([70, 205, 85], dtype=np.uint8)
    isolated = np.zeros(image.shape[:2], dtype=bool)
    isolated[8:14, 8:16] = True
    image[isolated] = np.array([70, 205, 85], dtype=np.uint8)

    candidate = local_only | isolated
    local_bg_family = _local_background_family_continuity_mask(
        image,
        tuple(int(x) for x in bg),
        candidate,
        context,
    )
    bg_delta = oklab_distance(
        srgb_to_oklab(image),
        srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0],
    )

    assert float(bg_delta[local_only].min()) > 8.0
    assert local_bg_family[local_only].all()
    assert not local_bg_family[isolated].any()


def test_glass_solve_does_not_turn_near_background_pixels_purple():
    path = Path("samples/vlm_eval_game/ui_glass_button_soft_shadow/green.png")
    if not path.exists():
        pytest.skip("G04 game sample is not present")
    image = io.load_rgb(path)

    result = analyze_solid_bg_graphic(image)
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)
    alpha_u8 = (result.alpha * 255.0 + 0.5).astype(np.uint8)
    bg = np.asarray(result.background_color, dtype=np.uint8)
    bg_delta = oklab_distance(
        srgb_to_oklab(image),
        srgb_to_oklab(bg.reshape(1, 1, 3))[0, 0],
    )
    dominant = int(np.argmax(bg))
    other_channels = [idx for idx in range(3) if idx != dominant]
    other_max = np.maximum(
        rgba_rgb[..., other_channels[0]].astype(np.int16),
        rgba_rgb[..., other_channels[1]].astype(np.int16),
    )
    false_hue = (
        (alpha_u8 > 0)
        & (alpha_u8 <= 64)
        & (bg_delta <= 8.0)
        & ((other_max - rgba_rgb[..., dominant].astype(np.int16)) > 25)
        & ~result.ownership_masks["shadow_layer"]
    )

    assert result.accepted is True
    assert result.debug["glass_internal_shadow_reclassified_pixels"] > 5000
    assert result.debug["glass_soft_foreground_stabilization"]["applied"] is True
    thin_ridge_repair = result.debug["thin_glass_foreground_ridge_repair"]
    assert thin_ridge_repair["applied"] is True
    assert thin_ridge_repair["rgb_repaired"] is True
    assert thin_ridge_repair["high_alpha_black_pixels"] > 50
    assert 1000 < thin_ridge_repair["pixels"] < 8000
    assert result.ownership_masks["shadow_layer"][382:754, 1026:1081].mean() < 0.05
    assert np.percentile(result.alpha[430:790, 152:168], 90) <= 0.75
    corner = np.s_[820:940, 135:570]
    fg_luma = rgba_rgb.astype(np.float32).mean(axis=-1)
    remaining_corner_black_arc = (
        (result.alpha > 0.60)
        & (fg_luma < 45.0)
        & (result.ownership_masks["soft_subject_layer"] | result.ownership_masks["opaque_subject"])
        & ~result.ownership_masks["shadow_layer"]
    )
    assert int(remaining_corner_black_arc[corner].sum()) < 20
    assert int(false_hue.sum()) < 400
    source_preserving = result.debug["source_preserving_glass_foreground"]
    assert source_preserving["applied"] is True
    assert source_preserving["smooth_pixels"] > 100000
    assert source_preserving["source_highlight_protected_pixels"] > 10000
    assert source_preserving["source_highlight_foreground_kept_pixels"] > 10000
    assert source_preserving["source_bg_residue_repaired_pixels"] > 10000
    assert source_preserving["chroma_continuity_pixels"] > 1000
    assert source_preserving["nearest_chroma_continuity_pixels"] > 1000
    glass_field = result.debug["glass_continuous_field"]
    assert glass_field["applied"] is True
    assert glass_field["alpha_pixels"] > 50000
    assert glass_field["color_pixels"] > 10000
    assert glass_field["mean_alpha_delta"] < 0.04
    gap_restore = result.debug["glass_color_shifted_gap_restore"]
    assert gap_restore["applied"] is True
    assert gap_restore["pixels"] > 1000
    shadow_feather = result.debug["exterior_shadow_feather"]
    assert shadow_feather["applied"] is True
    assert shadow_feather["added_pixels"] > 1000

    rgba = np.dstack([rgba_rgb, alpha_u8])
    on_gray = composite(rgba, (128, 128, 128))
    subject = result.ownership_masks["soft_subject_layer"] | result.ownership_masks["opaque_subject"]
    visible_subject = subject & (result.alpha > 0.015)
    bottom = np.s_[int(image.shape[0] * 0.62) : int(image.shape[0] * 0.86), int(image.shape[1] * 0.08) : int(image.shape[1] * 0.94)]
    left_corner = np.s_[int(image.shape[0] * 0.45) : int(image.shape[0] * 0.78), 0 : int(image.shape[1] * 0.48)]
    # Broad green-screen glass should preserve the continuous source signal in
    # premultiplied color. The threshold guards against straight-F/mask
    # fragmentation turning the bottom rim and corner into dithered cyan/black
    # segments after export.
    assert _rgb_neighbor_jump_percentile(
        on_gray,
        visible_subject,
        (bottom[1].start, bottom[0].start, bottom[1].stop, bottom[0].stop),
        95,
    ) < 42.0
    assert _rgb_neighbor_jump_percentile(
        on_gray,
        visible_subject,
        (left_corner[1].start, left_corner[0].start, left_corner[1].stop, left_corner[0].stop),
        95,
    ) < 40.0
    inner_top = np.s_[300:420, 150:1120]
    blue_purple_rim_crack = (
        visible_subject
        & (result.alpha > 0.03)
        & ((rgba_rgb[..., 2].astype(np.int16) - rgba_rgb[..., 1].astype(np.int16)) > 25)
        & (fg_luma < 160.0)
    )
    # In broad glass, pixels with secondary-channel source evidence should be
    # exported from the continuous source field. If they are treated as pure
    # screen spill, straight-F solving produces blue/purple dashed cracks on the
    # inner rim even though the source rim is continuous.
    assert int(blue_purple_rim_crack[inner_top].sum()) < 1200
    left_lower_bend = np.s_[820:900, 130:230]
    fg_chroma = rgba_rgb.astype(np.int16).max(axis=-1) - rgba_rgb.astype(np.int16).min(axis=-1)
    dirty_dark_neutral = (
        visible_subject
        & ~result.ownership_masks["shadow_layer"]
        & (result.alpha > 0.025)
        & (fg_luma < 70.0)
        & (fg_chroma < 50)
    )
    # Local chroma continuity should stop source-premultiplied glass evidence
    # from reintroducing a black/grey notch where a colored refractive line is
    # adjacent but the density-based glass context has a small gap.
    assert int(dirty_dark_neutral[left_lower_bend].sum()) == 0
    # The same source-continuity machinery must not flatten bright chromatic
    # refraction bands. Those bands are valid source evidence even when their
    # hue is close to the green screen family.
    left_highlight = np.s_[835:900, 120:245]
    on_gray_luma = on_gray.astype(np.float32).mean(axis=-1)
    assert float(np.percentile(on_gray_luma[left_highlight], 99.0)) > 230.0
    # Color-shifted fragments that still carry red/blue source evidence should
    # not be punched out as background holes, while the broad center basin must
    # remain transparent. This guards the corner-gap class without encoding a
    # one-pixel coordinate workaround.
    assert float(result.alpha[791:815, 149:166].mean()) > 0.30
    assert float(result.alpha[790:840, 220:315].mean()) < 0.02
    shadow_alpha = np.clip(result.alpha - result.subject_alpha, 0.0, 1.0)
    shadow_tail = shadow_alpha[900:1080, 80:1180]
    # Broad generated cast shadows should export as a continuous soft field.
    # The p99 guard ignores deliberate contact edges and catches noisy tail
    # support/alpha discontinuities that are obvious on white backgrounds.
    assert float(np.percentile(np.abs(np.diff(shadow_tail, axis=1)), 99.0)) < 0.04
    assert float(np.percentile(np.abs(np.diff(shadow_tail, axis=0)), 99.0)) < 0.05


def test_same_hue_ui_panel_texture_is_opaque_material_not_transparency():
    path = Path("samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png")
    if not path.exists():
        pytest.skip("G02 game sample is not present")
    image = io.load_rgb(path)

    result = analyze_solid_bg_graphic(image)

    h, w = image.shape[:2]
    panel = np.s_[int(h * 0.30) : int(h * 0.58), int(w * 0.26) : int(w * 0.74)]
    assert result.accepted is True
    assert result.debug["promoted_internal_background_material_pixels"] > 4000
    assert int((result.alpha[panel] < 0.80).sum()) < 4200
    assert float(np.percentile(result.alpha[panel], 2.0)) > 0.95


def test_saturated_hard_edge_antialiasing_is_soft_subject_not_shadow():
    """A green-screen hard edge may be scalar-darkened B, but contour topology
    says it is antialiasing owned by the subject rather than a cast shadow."""
    h, w = 96, 96
    bg = np.array((0, 200, 0), dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    image[20:76, 20:76] = (40, 110, 230)
    image[20:22, 20:76] = (3, 170, 20)
    image[74:76, 20:76] = (3, 170, 20)
    image[20:76, 20:22] = (3, 170, 20)
    image[20:76, 74:76] = (3, 170, 20)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    soft = result.ownership_masks["soft_subject_layer"]
    shadow = result.ownership_masks["shadow_layer"]
    assert result.alpha[28:68, 28:68].min() > 0.99
    assert soft[20:22, 26:70].mean() > 0.95
    assert result.alpha[20:22, 26:70].mean() < 0.60
    assert int(shadow.sum()) < 20


def test_tiny_background_colored_edge_dust_is_removed_after_foreground_solve():
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (96, 112, 3)).copy()
    image[20:76, 28:84] = np.array([220, 40, 30], dtype=np.uint8)
    dust_pixels = [(20, 39), (20, 53), (33, 28), (45, 83), (75, 61), (63, 28)]
    for y, x in dust_pixels:
        image[y, x] = np.array([7, 215, 4], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is True
    assert result.debug["known_background_projection_pixels"] >= len(dust_pixels)
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)
    for y, x in dust_pixels:
        assert result.alpha[y, x] > 0.0
        assert int(rgba_rgb[y, x, 1]) < 100
    assert result.alpha[30:66, 38:74].mean() == pytest.approx(1.0)


def test_internal_background_colored_material_is_protected_as_component():
    bg = np.array([8, 207, 17], dtype=np.uint8)
    image = np.broadcast_to(bg, (112, 128, 3)).copy()
    image[18:94, 20:108] = np.array([220, 40, 30], dtype=np.uint8)
    image[42:70, 48:84] = np.array([4, 150, 70], dtype=np.uint8)
    image[50:62, 58:76] = np.array([4, 145, 60], dtype=np.uint8)
    image[18, 44] = np.array([12, 222, 18], dtype=np.uint8)
    image[93, 72] = np.array([12, 222, 18], dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)
    rgba_rgb = io.linear_to_srgb_u8(result.rgba_rgb_linear)

    assert result.accepted is True
    assert result.debug["internal_background_material_pixels"] >= 900
    assert result.debug["known_background_projection_pixels"] >= 2
    assert int(rgba_rgb[52:60, 60:74, 1].mean()) > 110
    assert int(rgba_rgb[18, 44, 1]) < 100
    assert int(rgba_rgb[93, 72, 1]) < 100


def test_noisy_photo_like_input_falls_back():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(80, 90, 3), dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is False
    assert "unstable" in result.reason


def test_solid_background_photo_like_subject_falls_back():
    rng = np.random.default_rng(4)
    image = np.full((96, 112, 3), [0, 200, 0], dtype=np.uint8)
    image[20:76, 24:88] = rng.integers(0, 256, size=(56, 64, 3), dtype=np.uint8)

    result = analyze_solid_bg_graphic(image)

    assert result.accepted is False
    assert result.reason == "subject palette is photo-like"


def test_real_small_ui_icon_can_use_solid_graphic_prepass():
    path = Path("samples/regression/small_ui_icon_green/input.png")
    if not path.exists():
        pytest.skip("real regression sample is not present")

    result = matte_image(str(path), backend="auto", qa=False)

    assert result.strategy_name == "solid_bg_graphic"
    assert result.report["strategy"]["extras"]["fallback_strategy"] == "saturated_bg"


def test_real_wide_star_button_can_use_solid_graphic_prepass():
    path = Path("samples/regression/star_badge_button_green/input.png")
    if not path.exists():
        pytest.skip("real regression sample is not present")

    result = matte_image(str(path), backend="auto", qa=False)

    assert result.strategy_name == "solid_bg_graphic"
    assert result.report["strategy"]["extras"]["fallback_strategy"] == "saturated_bg"
    assert result.report["strategy"]["extras"]["solid_graphic_confidence"] > 0.90
