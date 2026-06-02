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


def test_auto_route_sends_i019_to_direct_known_bg_glow():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d09_soft_alpha_smooth_white_glow_green/green.png")

    decision = classify_route(image)

    assert decision.route == "known_bg_glow"
    assert decision.backend == "direct-known-bg-glow"
    assert decision.params["execution_profile"] == "known-bg-glow"
    assert decision.reasons == ["icon_key_color_material_uses_known_bg_glow"]


def test_direct_worker_known_bg_glow_outputs_soft_transparent_layer():
    image = _rgb("samples/corridorkey_semantic/icon/icon_icon_d09_soft_alpha_smooth_white_glow_green/green.png")

    result = direct_matte_auto(image, shadow_mode="off")

    assert result.metadata["selected_backend"] == "direct-known-bg-glow"
    assert result.metadata["execution_backend"] == "direct-known-bg-glow"
    assert result.metadata["route"] == "known_bg_glow"
    assert result.response.strategy_name == "direct_known_bg_glow"
    assert int((result.response.alpha > 0.02).sum()) > 42000
    assert 0.18 < float(result.response.alpha.mean()) < 0.24
    foreground = result.response.foreground_srgb[result.response.alpha > 0.45]
    assert foreground[:, 1].mean() > 245.0
