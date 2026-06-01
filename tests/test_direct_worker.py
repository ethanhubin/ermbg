from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import ermbg.direct_worker as direct_worker
from ermbg.api import MatteResponse
from ermbg.router import RouteDecision


def _rgba(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    if alpha is None:
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    return np.dstack([rgb, (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)])


def test_direct_matte_auto_reuses_route_corridorkey_analysis(monkeypatch):
    rgb = np.full((3, 4, 3), (0, 200, 0), dtype=np.uint8)
    ck_analysis = {
        "screen_mode": "green",
        "background_color": [0, 200, 0],
        "parameter_profile": "translucent_button",
    }
    captured: dict[str, object] = {}

    def fake_classify_route(*args, **kwargs):
        return RouteDecision(
            route="corridorkey",
            asset_kind="icon",
            backend="comfy-corridorkey",
            params={"corridorkey_auto_mask": True},
            confidence=0.9,
            reasons=["test"],
            analysis={"corridorkey_analysis": ck_analysis},
        )

    def fake_matte_corridorkey_direct(image, *, corridorkey_analysis, params, **kwargs):
        captured["analysis"] = corridorkey_analysis
        captured["params"] = params
        return MatteResponse(
            rgba=_rgba(image),
            alpha=np.ones(image.shape[:2], dtype=np.float32),
            foreground_srgb=image.copy(),
            strategy_name="direct_corridorkey",
            background_color=(0, 200, 0),
            debug={},
        )

    monkeypatch.setattr(direct_worker, "classify_route", fake_classify_route)
    monkeypatch.setattr(direct_worker, "matte_corridorkey_direct", fake_matte_corridorkey_direct)

    result = direct_worker.direct_matte_auto(rgb, shadow_mode="off")

    assert captured["analysis"] is ck_analysis
    assert captured["params"] == {"corridorkey_auto_mask": True}
    assert result.metadata["selected_backend"] == "comfy-corridorkey"
    assert result.metadata["execution_backend"] == "direct-corridorkey"
    assert result.metadata["execution_profile"] is None
    assert result.response.strategy_name == "direct_corridorkey"
    assert result.timings["route_sec"] >= 0.0
    assert result.timings["backend_sec"] >= 0.0


def test_matte_corridorkey_direct_uses_fake_client_without_comfy():
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    calls: list[dict[str, object]] = []

    class FakeClient:
        def matte(self, image_srgb, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                rgba=_rgba(image_srgb, alpha),
                alpha=alpha,
                foreground_srgb=image_srgb.copy(),
                hint_alpha=kwargs["hint_alpha"],
                raw_alpha=alpha,
                color_protection_alpha=np.zeros(alpha.shape, dtype=np.float32),
                debug={"hint": {"source": kwargs["hint_source"]}, "timings": {"total_sec": 0.0}},
            )

    result = direct_worker.matte_corridorkey_direct(
        rgb,
        corridorkey_analysis={
            "screen_mode": "green",
            "background_color": [0, 200, 0],
            "parameter_profile": "translucent_button",
        },
        params={"corridorkey_auto_mask": True},
        bg_color=(0, 200, 0),
        shadow_mode="off",
        corridorkey_client=FakeClient(),
    )

    assert result.strategy_name == "direct_corridorkey"
    assert result.report["despill_method"] == "direct_corridorkey"
    assert calls[0]["screen_color"] == "green"
    assert calls[0]["execution_profile"] == "corridorkey-transparent-button"
    assert calls[0]["hint_source"] == "glass_all_white_corridorkey_hint"


def test_matte_corridorkey_direct_keeps_known_bg_hint_for_shaped_profiles():
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    calls: list[dict[str, object]] = []

    class FakeClient:
        def matte(self, image_srgb, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                rgba=_rgba(image_srgb, alpha),
                alpha=alpha,
                foreground_srgb=image_srgb.copy(),
                hint_alpha=np.zeros(alpha.shape, dtype=np.float32)
                if kwargs["hint_alpha"] is None
                else kwargs["hint_alpha"],
                raw_alpha=alpha,
                color_protection_alpha=np.zeros(alpha.shape, dtype=np.float32),
                debug={"hint": {"source": kwargs["hint_source"]}, "timings": {"total_sec": 0.0}},
            )

    direct_worker.matte_corridorkey_direct(
        rgb,
        corridorkey_analysis={
            "screen_mode": "green",
            "background_color": [0, 200, 0],
            "parameter_profile": "key_color_material",
        },
        params={"corridorkey_auto_mask": True},
        bg_color=(0, 200, 0),
        shadow_mode="off",
        corridorkey_client=FakeClient(),
    )

    assert calls[0]["hint_alpha"] is None
    assert calls[0]["hint_source"] is None
    assert calls[0]["execution_profile"] == "corridorkey-shaped-icon"
