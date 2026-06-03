from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ermbg.direct_worker import direct_matte_auto
from ermbg.known_bg_glow import _chromatic_material_weight, analyze_known_bg_glow, matte_known_bg_glow
from ermbg.router import classify_route


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rgb(path: str) -> np.ndarray:
    return np.array(Image.open(PROJECT_ROOT / path).convert("RGB"))


def _small_textured_coherent_glow() -> np.ndarray:
    h, w = 79, 72
    bg = np.array([3, 182, 5], dtype=np.float32)
    y, x = np.mgrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dx = x - cx
    dy = y - cy
    radius = np.sqrt((dx / 30.0) ** 2 + (dy / 32.0) ** 2)
    theta = np.arctan2(dy, dx)
    alpha = np.clip(1.0 - radius, 0.0, 1.0) ** 0.9
    alpha = np.where(radius < 0.38, 1.0, alpha)

    # Blocky outer steps mimic low-resolution glow quantization. The field is
    # still one connected halo whose alpha fades coherently from the core.
    block_pattern = (((x // 4) * 17 + ((y // 4) * 31)) % 7 - 3) / 3.0
    outer = (alpha >= 0.02) & (alpha <= 0.35)
    alpha = np.clip(alpha + 0.08 * block_pattern * outer, 0.0, 1.0)

    hue_mix = (np.sin(theta * 2.0) + 1.0) / 2.0
    foreground = np.empty((h, w, 3), dtype=np.float32)
    foreground[..., 0] = 255.0
    foreground[..., 1] = 210.0 + 45.0 * hue_mix
    foreground[..., 2] = 35.0 + 200.0 * (1.0 - hue_mix)
    rgb = bg.reshape(1, 1, 3) + alpha[..., None] * (foreground - bg.reshape(1, 1, 3))
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def _blue_screen_green_chromatic_glow() -> np.ndarray:
    h = w = 160
    bg = np.array([1, 4, 233], dtype=np.float32)
    green = np.array([1, 254, 11], dtype=np.float32)
    white = np.array([255, 255, 255], dtype=np.float32)
    y, x = np.mgrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    dx = x - cx
    dy = y - cy
    radius = np.sqrt(dx * dx + dy * dy)
    halo = np.clip(1.0 - radius / 60.0, 0.0, 1.0) ** 1.45
    rays = np.zeros((h, w), dtype=np.float32)
    for angle in np.linspace(0.0, np.pi, 4, endpoint=False):
        normal = np.abs(dx * np.sin(angle) - dy * np.cos(angle))
        along = np.abs(dx * np.cos(angle) + dy * np.sin(angle))
        rays = np.maximum(rays, np.exp(-(normal / 1.3) ** 2) * np.exp(-along / 58.0))
    alpha = np.clip(np.maximum(halo * 0.70, rays), 0.0, 1.0)
    core = np.clip(1.0 - radius / 15.0, 0.0, 1.0) ** 0.6
    foreground = green.reshape(1, 1, 3) * (1.0 - core[..., None]) + white.reshape(1, 1, 3) * core[..., None]
    rgb = bg.reshape(1, 1, 3) * (1.0 - alpha[..., None]) + foreground * alpha[..., None]

    # Blue-screen perturbations inside the low-alpha halo are the failure mode:
    # unconstrained adaptive rays can extend them back to blue foreground
    # endpoints that show up as speckles on complementary backgrounds.
    noise_band = (alpha > 0.04) & (alpha < 0.40) & (((x * 17 + y * 31) % 19) == 0)
    rgb[noise_band] = rgb[noise_band] * 0.65 + bg.reshape(1, 1, 3) * 0.35
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def _wide_full_frame_continuous_glow() -> np.ndarray:
    h, w = 41, 140
    bg = np.array([3, 184, 6], dtype=np.float32)
    y, x = np.mgrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    dy = np.minimum(y, h - 1 - y).astype(np.float32)
    dx = np.minimum(x, w - 1 - x).astype(np.float32)
    edge_distance = np.minimum(dy / 20.0, np.clip(dx / 12.0, 0.0, 1.0))
    alpha = np.clip(edge_distance, 0.0, 1.0) ** 1.2
    alpha *= np.clip(1.0 - (np.abs(x - cx) / (w * 0.44)) ** 6, 0.0, 1.0)
    core = np.exp(-((x - cx) / (w * 0.08)) ** 2 - ((y - cy) / (h * 0.08)) ** 2)
    alpha = np.maximum(alpha, core * 0.9)

    foreground = np.broadcast_to(np.array([236, 255, 55], dtype=np.float32), (h, w, 3)).copy()
    highlight = np.exp(-((x - cx) / (w * 0.08)) ** 2 - ((y - cy) / (h * 0.06)) ** 2)
    foreground = foreground * (1.0 - highlight[..., None]) + np.array([255, 255, 255], dtype=np.float32) * highlight[..., None]
    rgb = bg.reshape(1, 1, 3) * (1.0 - alpha[..., None]) + foreground * alpha[..., None]
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def test_known_bg_glow_detects_i019_smooth_white_glow():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d09_soft_alpha_smooth_white_glow_green/green.png")

    analysis = analyze_known_bg_glow(image, (3, 195, 9))

    assert analysis.accepted is True
    assert analysis.support_fraction > 0.60
    assert analysis.soft_fraction > 0.70
    assert analysis.residual_p90 < 5.0
    assert analysis.target_color[0] > 245
    assert analysis.target_color[1] > 245
    assert analysis.target_color[2] > 245


def test_known_bg_glow_rejects_hard_icon_material():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_a01_hard_boundary_strong_outline/green.png")

    analysis = analyze_known_bg_glow(image, (0, 200, 0))

    assert analysis.accepted is False
    assert analysis.support_fraction > 0.0


def test_known_bg_glow_rejects_complex_hard_core_effect_icon():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d01_soft_alpha_glow_hard_core/green.png")

    decision = classify_route(image)
    analysis = decision.analysis["known_bg_glow"]

    assert analysis["accepted"] is False
    assert analysis["reason"] == "strong glow core is too textured"
    assert analysis["strong_core_gradient_p90"] >= 160.0
    assert decision.route == "corridorkey"
    assert decision.asset_kind == "icon"


def test_known_bg_glow_accepts_small_textured_but_coherent_halo():
    image = _small_textured_coherent_glow()

    analysis = analyze_known_bg_glow(image, (3, 182, 5))
    decision = classify_route(image)

    assert analysis.accepted is True
    assert analysis.mode == "adaptive_ray"
    assert analysis.outer_roughness_p90 > 0.06
    assert analysis.outer_roughness_p90 <= 0.10
    assert analysis.falloff_correlation > 0.90
    assert decision.route == "known_bg_glow"
    assert decision.params["known_bg_glow_mode"] == "adaptive_ray"


def test_known_bg_glow_accepts_tightly_cropped_full_frame_halo():
    image = _wide_full_frame_continuous_glow()

    analysis = analyze_known_bg_glow(image, (3, 184, 6))

    assert analysis.accepted is True
    assert analysis.mode == "adaptive_ray"
    assert analysis.support_fraction > 0.75
    assert analysis.outer_roughness_p90 < 0.055
    assert analysis.falloff_correlation > 0.90


def test_auto_route_sends_wide_glow_to_known_bg_glow_before_button_classification():
    image = _wide_full_frame_continuous_glow()

    decision = classify_route(image)

    assert decision.route == "known_bg_glow"
    assert decision.asset_kind == "glow"
    assert decision.params["execution_profile"] == "known-bg-glow"
    assert decision.analysis["corridorkey_analysis"]["foreground_aspect_ratio"] >= 1.45


def test_adaptive_known_bg_glow_repairs_additive_screen_color_endpoints():
    image = _small_textured_coherent_glow()
    image[39, 36] = (255, 255, 255)
    for y, x in [(14, 33), (15, 33), (16, 33), (42, 41), (55, 37), (58, 39)]:
        image[y, x] = (12, 185, 7)

    result = matte_known_bg_glow(
        image,
        (3, 182, 5),
        target_color=(255, 219, 44),
        mode="adaptive_ray",
    )
    visible = result.alpha > 0.01
    foreground = result.foreground_srgb.astype(np.int16)
    screen_green = visible & (foreground[..., 1] > foreground[..., 0] + 20) & (foreground[..., 1] > foreground[..., 2] + 20)

    changed = result.foreground_srgb[[14, 15, 16, 42, 55, 58], [33, 33, 33, 41, 37, 39]]

    assert result.debug["foreground_repaired_pixels"] > 0
    assert np.all(changed[:, 0] >= changed[:, 1])
    assert int(screen_green.sum()) == 0
    assert result.foreground_srgb[39, 36].min() >= 220


def test_known_bg_glow_uses_chromatic_swap_ray_for_blue_screen_green_glow():
    image = _blue_screen_green_chromatic_glow()

    analysis = analyze_known_bg_glow(image, (1, 4, 233))
    result = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=2.0,
    )
    visible = result.alpha > 0.005
    foreground = result.foreground_srgb.astype(np.int16)
    fg_float = result.foreground_srgb.astype(np.float32)
    bright_core = visible & (fg_float.max(axis=2) >= 205.0) & ((fg_float.max(axis=2) - fg_float.min(axis=2)) <= 92.0)
    non_green = visible & ~bright_core & (
        (foreground[..., 1] < foreground[..., 0] + 20)
        | (foreground[..., 1] < foreground[..., 2] + 20)
    )
    blue_dirty = visible & ~bright_core & (foreground[..., 2] > foreground[..., 1] + 20) & (foreground[..., 2] > foreground[..., 0] + 20)
    source_distance = np.linalg.norm(image.astype(np.float32) - np.array([1, 4, 233], dtype=np.float32), axis=2)
    source_i16 = image.astype(np.int16)
    material = visible & (source_i16[..., 1] >= 150) & (source_i16[..., 1] >= source_i16[..., 2] + 35) & (source_distance >= 120)

    assert analysis.accepted is True
    assert analysis.mode == "chromatic_swap_ray"
    assert result.debug["mode"] == "chromatic_swap_ray"
    assert result.debug["background_removal_strength"] == 2.0
    assert int(blue_dirty.sum()) == 0


def test_chromatic_swap_ray_preserves_white_highlight_while_removing_blue_screen():
    image = _blue_screen_green_chromatic_glow()
    analysis = analyze_known_bg_glow(image, (1, 4, 233))

    result = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=2.0,
    )
    source = image.astype(np.float32)
    source_chroma = source.max(axis=2) - source.min(axis=2)
    source_distance = np.linalg.norm(source - np.array([1, 4, 233], dtype=np.float32), axis=2)
    white_highlight = (source.max(axis=2) >= 205.0) & (source_chroma <= 92.0) & (source_distance >= 160.0)
    foreground = result.foreground_srgb.astype(np.float32)
    foreground_chroma = foreground.max(axis=2) - foreground.min(axis=2)
    restored_highlight = white_highlight & (result.alpha >= 0.60) & (foreground.max(axis=2) >= 205.0) & (foreground_chroma <= 112.0)

    assert analysis.mode == "chromatic_swap_ray"
    assert float(np.median(result.alpha[white_highlight])) >= 0.70
    assert int(restored_highlight.sum()) >= int(white_highlight.sum() * 0.65)


def test_chromatic_swap_ray_rejects_dark_endpoint_speckles():
    image = _blue_screen_green_chromatic_glow()
    image[40, 80] = (90, 110, 150)

    analysis = analyze_known_bg_glow(image, (1, 4, 233))
    result = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=2.0,
    )
    foreground = result.foreground_srgb.astype(np.float32)
    dark_endpoint = (result.alpha > 0.005) & (foreground.max(axis=2) < 80.0)

    assert analysis.mode == "chromatic_swap_ray"
    assert int(dark_endpoint.sum()) == 0
    assert float(result.alpha[40, 80]) == 0.0
    assert int(result.foreground_srgb[40, 80].max()) == 0


def test_chromatic_swap_ray_smooths_material_to_alpha_transition():
    image = _blue_screen_green_chromatic_glow()

    analysis = analyze_known_bg_glow(image, (1, 4, 233))
    result = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
    )
    row = image.shape[0] // 2
    source_distance = np.linalg.norm(image.astype(np.float32) - np.array([1, 4, 233], dtype=np.float32), axis=2)
    supported_pair = (
        (result.alpha[row, :-1] > 0.005)
        & (result.alpha[row, 1:] > 0.005)
        & (source_distance[row, :-1] >= 95.0)
        & (source_distance[row, 1:] >= 95.0)
    )
    alpha_jump = np.abs(np.diff(result.alpha[row]))
    transition = (result.alpha[row] >= 0.45) & (result.alpha[row] <= 0.75)
    transition_fg = result.foreground_srgb[row][transition].astype(np.int16)
    row_rgb = image[row].astype(np.float32)
    row_luma = row_rgb[:, 0] * 0.2126 + row_rgb[:, 1] * 0.7152 + row_rgb[:, 2] * 0.0722
    bright_material_bridge = (
        (source_distance[row] >= 225.0)
        & (row_luma >= 175.0)
        & (result.alpha[row] < 0.99)
    )

    assert analysis.mode == "chromatic_swap_ray"
    assert float(alpha_jump[supported_pair].max()) < 0.40
    assert int(transition.sum()) >= 10
    transition_chroma = transition_fg.max(axis=1) - transition_fg.min(axis=1)
    transition_is_material_color = (transition_fg[:, 1] >= transition_fg[:, 2] + 20) | (transition_chroma <= 92)
    assert np.all(transition_is_material_color)
    assert 0.70 <= float(np.median(result.alpha[row][bright_material_bridge])) <= 0.90


def test_chromatic_swap_ray_material_strength_controls_bridge_alpha():
    image = _blue_screen_green_chromatic_glow()
    analysis = analyze_known_bg_glow(image, (1, 4, 233))

    low = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=0.0,
    )
    high = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=2.0,
    )
    row = image.shape[0] // 2
    source_distance = np.linalg.norm(image.astype(np.float32) - np.array([1, 4, 233], dtype=np.float32), axis=2)
    row_rgb = image[row].astype(np.float32)
    row_luma = row_rgb[:, 0] * 0.2126 + row_rgb[:, 1] * 0.7152 + row_rgb[:, 2] * 0.0722
    bridge = (source_distance[row] >= 225.0) & (row_luma >= 175.0) & (low.alpha[row] < 0.99)

    bg_near = (source_distance[row] >= 95.0) & (source_distance[row] <= 180.0)

    assert high.debug["background_removal_strength"] == 2.0
    assert float(np.median(low.alpha[row][bridge])) >= 0.45
    assert float(np.median(high.alpha[row][bridge])) >= 0.45
    assert float(np.median(low.alpha[row][bg_near]) - np.median(high.alpha[row][bg_near])) >= 0.10


def test_chromatic_swap_ray_low_strength_keeps_highlight_retreat_connected():
    image = _blue_screen_green_chromatic_glow()
    analysis = analyze_known_bg_glow(image, (1, 4, 233))

    result = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=0.0,
    )
    row = image.shape[0] // 2
    transition = result.alpha[row, 60:87]

    assert float(transition.min()) >= 0.45
    assert float(np.abs(np.diff(transition)).max()) < 0.16


def test_chromatic_swap_ray_softens_highlight_protection_boundary():
    image = _blue_screen_green_chromatic_glow()
    analysis = analyze_known_bg_glow(image, (1, 4, 233))

    result = matte_known_bg_glow(
        image,
        (1, 4, 233),
        target_color=analysis.target_color,
        mode=analysis.mode,
        material_strength=2.0,
    )
    row = image.shape[0] // 2
    alpha_jump = np.abs(np.diff(result.alpha[row]))

    assert float(alpha_jump.max()) < 0.18


def test_chromatic_swap_ray_material_strength_does_not_expand_foreground_protection():
    image = _blue_screen_green_chromatic_glow()
    analysis = analyze_known_bg_glow(image, (1, 4, 233))

    _low_weight, low_proven = _chromatic_material_weight(
        image,
        (1, 4, 233),
        analysis.target_color,
        material_strength=0.0,
    )
    _high_weight, high_proven = _chromatic_material_weight(
        image,
        (1, 4, 233),
        analysis.target_color,
        material_strength=2.0,
    )

    assert int(low_proven.sum()) == 0
    assert int(high_proven.sum()) == 0
    assert float(_low_weight.max()) == 0.0
    assert float(_high_weight.max()) == 0.0


def test_auto_route_sends_i019_to_direct_known_bg_glow():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d09_soft_alpha_smooth_white_glow_green/green.png")

    decision = classify_route(image)

    assert decision.route == "known_bg_glow"
    assert decision.backend == "known_bg_glow"
    assert decision.to_dict()["algorithm"] == "known_bg_glow"
    assert decision.params["execution_profile"] == "known-bg-glow"
    assert decision.reasons == ["known_bg_single_target_line_glow_uses_known_bg_glow"]


def test_direct_worker_known_bg_glow_outputs_soft_transparent_layer():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d09_soft_alpha_smooth_white_glow_green/green.png")

    result = direct_matte_auto(image, shadow_mode="off")

    assert result.metadata["algorithm"] == "known_bg_glow"
    assert result.metadata["execution_backend"] == "direct-known-bg-glow"
    assert result.metadata["route"] == "known_bg_glow"
    assert result.response.strategy_name == "direct_known_bg_glow"
    assert int((result.response.alpha > 0.02).sum()) > 42000
    assert 0.18 < float(result.response.alpha.mean()) < 0.24
    foreground = result.response.foreground_srgb[result.response.alpha > 0.45]
    assert foreground[:, 1].mean() > 245.0
