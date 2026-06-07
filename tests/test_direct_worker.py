from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

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
            backend="corridorkey",
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
    assert result.metadata["algorithm"] == "corridorkey"
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


def test_matte_corridorkey_direct_applies_user_masks_after_model():
    rgb = np.full((8, 8, 3), (0, 200, 0), dtype=np.uint8)
    alpha = np.full(rgb.shape[:2], 0.25, dtype=np.float32)
    keep_mask = np.zeros(rgb.shape[:2], dtype=np.float32)
    remove_mask = np.zeros(rgb.shape[:2], dtype=np.float32)
    keep_mask[1:3, 1:3] = 1.0
    remove_mask[5:7, 5:7] = 1.0

    class FakeClient:
        def matte(self, image_srgb, **kwargs):
            return SimpleNamespace(
                rgba=_rgba(image_srgb, alpha),
                alpha=alpha.copy(),
                foreground_srgb=image_srgb.copy(),
                hint_alpha=np.ones(alpha.shape, dtype=np.float32),
                raw_alpha=alpha.copy(),
                color_protection_alpha=np.zeros(alpha.shape, dtype=np.float32),
                debug={"hint": {"source": kwargs["hint_source"]}, "timings": {"total_sec": 0.0}},
            )

    result = direct_worker.matte_corridorkey_direct(
        rgb,
        corridorkey_analysis={
            "screen_mode": "green",
            "background_color": [0, 200, 0],
            "parameter_profile": "key_color_material",
        },
        params={
            "corridorkey_auto_mask": False,
            "user_keep_mask": keep_mask,
            "user_remove_mask": remove_mask,
        },
        bg_color=(0, 200, 0),
        shadow_mode="off",
        corridorkey_client=FakeClient(),
    )

    assert np.all(result.alpha[1:3, 1:3] == 1.0)
    assert np.all(result.alpha[5:7, 5:7] == 0.0)
    info = result.debug["semantic_execution"]
    assert info["user_keep_pixels"] == 4
    assert info["user_remove_pixels"] == 4


def test_matte_corridorkey_direct_does_not_apply_screen_material_alpha_constraints():
    rgb = np.full((8, 8, 3), (0, 200, 0), dtype=np.uint8)
    rgb[2:6, 2:6] = (120, 180, 120)
    alpha = np.ones(rgb.shape[:2], dtype=np.float32)

    class FakeClient:
        def matte(self, image_srgb, **kwargs):
            return SimpleNamespace(
                rgba=_rgba(image_srgb, alpha),
                alpha=alpha.copy(),
                foreground_srgb=image_srgb.copy(),
                hint_alpha=np.ones(alpha.shape, dtype=np.float32),
                raw_alpha=alpha.copy(),
                color_protection_alpha=np.zeros(alpha.shape, dtype=np.float32),
                debug={"hint": {"source": kwargs["hint_source"]}, "timings": {"total_sec": 0.0}},
            )

    result = direct_worker.matte_corridorkey_direct(
        rgb,
        corridorkey_analysis={
            "screen_mode": "green",
            "background_color": [0, 200, 0],
            "parameter_profile": "screen_tinted_translucency",
        },
        params={
            "corridorkey_auto_mask": True,
            "semantic_decision": {"screen_material_policy": "background"},
        },
        bg_color=(0, 200, 0),
        shadow_mode="off",
        corridorkey_client=FakeClient(),
    )

    assert np.all(result.alpha == 1.0)
    info = result.debug["semantic_execution"]
    assert info["semantic_decision"] == {"screen_material_policy": "background"}
    assert info["semantic_decision_applied"] is False
    assert info["keep_floor_pixels"] == 0
    assert info["remove_pixels"] == 0


def test_matte_corridorkey_direct_does_not_apply_glass_core_and_gradient_alpha_constraints():
    image_path = (
        Path(__file__).resolve().parents[1]
        / "samples/corridorkey_semantic/icon/icon_icon_d11_glass_portal_blue/blue.png"
    )
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    alpha = np.full(rgb.shape[:2], 0.95, dtype=np.float32)

    class FakeClient:
        def matte(self, image_srgb, **kwargs):
            return SimpleNamespace(
                rgba=_rgba(image_srgb, alpha),
                alpha=alpha.copy(),
                foreground_srgb=image_srgb.copy(),
                hint_alpha=np.ones(alpha.shape, dtype=np.float32),
                raw_alpha=alpha.copy(),
                color_protection_alpha=np.zeros(alpha.shape, dtype=np.float32),
                debug={"hint": {"source": kwargs["hint_source"]}, "timings": {"total_sec": 0.0}},
            )

    result = direct_worker.matte_corridorkey_direct(
        rgb,
        corridorkey_analysis={
            "screen_mode": "blue",
            "background_color": [0, 37, 252],
            "parameter_profile": "balanced",
        },
        params={
            "corridorkey_auto_mask": True,
            "semantic_decision": {
                "glass_core_policy": "transparent",
                "soft_alpha_gradient_policy": "preserve",
            },
        },
        bg_color=(0, 37, 252),
        shadow_mode="off",
        corridorkey_client=FakeClient(),
    )

    assert np.allclose(result.alpha, alpha)
    info = result.debug["semantic_execution"]
    assert info["semantic_decision"] == {
        "glass_core_policy": "transparent",
        "soft_alpha_gradient_policy": "preserve",
    }
    assert info["semantic_decision_applied"] is False
    assert info["keep_floor_pixels"] == 0
    assert info["alpha_cap_pixels"] == 0
    assert info["remove_pixels"] == 0


def test_matte_corridorkey_direct_applies_semantic_hint_variant_before_model():
    image_path = (
        Path(__file__).resolve().parents[1]
        / "samples/corridorkey_semantic/icon/icon_icon_d11_glass_portal_blue/blue.png"
    )
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    alpha = np.full(rgb.shape[:2], 0.5, dtype=np.float32)
    calls: list[dict[str, object]] = []

    class FakeClient:
        def matte(self, image_srgb, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                rgba=_rgba(image_srgb, alpha),
                alpha=alpha.copy(),
                foreground_srgb=image_srgb.copy(),
                hint_alpha=kwargs["hint_alpha"],
                raw_alpha=alpha.copy(),
                color_protection_alpha=np.zeros(alpha.shape, dtype=np.float32),
                debug={"hint": {"source": kwargs["hint_source"]}, "timings": {"total_sec": 0.0}},
            )

    result = direct_worker.matte_corridorkey_direct(
        rgb,
        corridorkey_analysis={
            "screen_mode": "blue",
            "background_color": [0, 37, 252],
            "parameter_profile": "balanced",
        },
        params={
            "corridorkey_auto_mask": True,
            "semantic_decision": {
                "policy": "corridorkey_hint_variant",
                "corridorkey_hint_variant": "feature_internal_opaque",
            },
        },
        bg_color=(0, 37, 252),
        shadow_mode="off",
        corridorkey_client=FakeClient(),
    )

    assert calls[0]["hint_source"] == "semantic_corridorkey_hint_variant:feature_internal_opaque"
    hint = calls[0]["hint_alpha"]
    assert isinstance(hint, np.ndarray)
    assert hint.shape == rgb.shape[:2]
    assert 0.0 <= float(hint.min()) <= float(hint.max()) <= 1.0
    assert result.debug["corridorkey_hint_plan"]["variant"] == "feature_internal_opaque"
    info = result.debug["semantic_execution"]
    assert info["semantic_hint_variant"] == "feature_internal_opaque"
    assert info["semantic_decision_applied"] is False
    assert info["keep_floor_pixels"] == 0
    assert info["alpha_cap_pixels"] == 0
    assert info["remove_pixels"] == 0


def test_direct_matte_from_decision_passes_semantic_decision(monkeypatch):
    rgb = np.full((8, 8, 3), (255, 255, 255), dtype=np.uint8)
    captured: dict[str, object] = {}

    def fake_known_b(image, **kwargs):
        captured.update(kwargs)
        return MatteResponse(
            rgba=_rgba(image),
            alpha=np.ones(image.shape[:2], dtype=np.float32),
            foreground_srgb=image.copy(),
            strategy_name="pymatting_known_b",
            background_color=(255, 255, 255),
            debug={},
        )

    monkeypatch.setattr(direct_worker, "_matte_image_pymatting_known_b", fake_known_b)

    direct_worker.direct_matte_from_decision(
        rgb,
        decision=RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={
                "pymatting_bg_color": (255, 255, 255),
                "semantic_decision": {"enclosed_near_bg_policy": "subject"},
            },
            confidence=1.0,
            reasons=["test"],
        ),
    )

    assert captured["semantic_decision"] == {"enclosed_near_bg_policy": "subject"}


def test_direct_matte_from_decision_passes_user_masks(monkeypatch):
    rgb = np.full((8, 8, 3), (255, 255, 255), dtype=np.uint8)
    keep_mask = np.zeros((8, 8), dtype=np.float32)
    remove_mask = np.zeros((8, 8), dtype=np.float32)
    keep_mask[2:4, 2:4] = 1.0
    remove_mask[5:7, 5:7] = 1.0
    captured: dict[str, object] = {}

    def fake_known_b(image, **kwargs):
        captured.update(kwargs)
        return MatteResponse(
            rgba=_rgba(image),
            alpha=np.ones(image.shape[:2], dtype=np.float32),
            foreground_srgb=image.copy(),
            strategy_name="pymatting_known_b",
            background_color=(255, 255, 255),
            debug={},
        )

    monkeypatch.setattr(direct_worker, "_matte_image_pymatting_known_b", fake_known_b)

    direct_worker.direct_matte_from_decision(
        rgb,
        decision=RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={
                "pymatting_bg_color": (255, 255, 255),
                "user_keep_mask": keep_mask,
                "user_remove_mask": remove_mask,
            },
            confidence=1.0,
            reasons=["test"],
        ),
    )

    assert np.asarray(captured["user_keep_mask"]).sum() == 4.0
    assert np.asarray(captured["user_remove_mask"]).sum() == 4.0
