from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ermbg.direct_worker import direct_matte_auto
from ermbg.known_bg_glow import analyze_known_bg_glow
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


def test_auto_route_sends_i019_to_direct_known_bg_glow():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d09_soft_alpha_smooth_white_glow_green/green.png")

    decision = classify_route(image)

    assert decision.route == "known_bg_glow"
    assert decision.backend == "known_bg_glow"
    assert decision.to_dict()["algorithm"] == "known_bg_glow"
    assert decision.params["execution_profile"] == "known-bg-glow"
    assert decision.reasons == ["icon_key_color_material_uses_known_bg_glow"]


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
