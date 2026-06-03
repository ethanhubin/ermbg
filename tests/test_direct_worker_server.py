from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import ermbg.direct_worker_server as server
from ermbg.api import MatteResponse
from ermbg.direct_worker import DirectWorkerResult
from ermbg.router import RouteDecision

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(rgb: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _result(rgb: np.ndarray, *, execution_profile: str = "pymatting-hard-button") -> DirectWorkerResult:
    alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    trimap = np.full(rgb.shape[:2], 128, dtype=np.uint8)
    response = MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=rgb,
        strategy_name="direct_pymatting_known_b",
        background_color=(0, 200, 0),
        report={"strategy": {"name": "direct_test"}},
        debug={
            "trimap_u8": trimap,
            "pymatting_known_b": {
                "background_normalization": {"applied": True},
                "trimap": {"unknown_pixels": int(trimap.size)},
                "parameters": {"fg_threshold": 24.0},
            },
            "shadow": {"method": "unknown_domain_same_background_reconstruction"},
        },
    )
    return DirectWorkerResult(
        response=response,
        timings={"route_sec": 0.01, "backend_sec": 0.02},
        metadata={
            "algorithm": "pymatting_known_b",
            "execution_backend": "direct-pymatting-known-b",
            "route": "pymatting_known_b",
            "asset_kind": "button",
            "parameter_profile": "opaque_hard_ui_no_shadow",
            "execution_profile": execution_profile,
        },
    )


def test_direct_worker_server_matte_endpoint(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)

    monkeypatch.setattr(
        server,
        "classify_route",
        lambda *args, **kwargs: RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={"execution_profile": "pymatting-hard-button"},
            confidence=1.0,
            reasons=["test"],
        ),
    )
    monkeypatch.setattr(server, "_run_prepared_main", lambda prepared, **kwargs: _result(prepared.rgb))

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={"include_image": "false"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["execution_backend"] == "direct-pymatting-known-b"
    assert payload["execution_profile"] == "pymatting-hard-button"
    assert payload["algorithm_debug"]["git_sha"]
    assert payload["algorithm_debug"]["pymatting_known_b"]["trimap"]["unknown_pixels"] == 6
    assert "rgba_png_base64" not in payload
    assert "trimap_png_base64" not in payload
    assert payload["server_elapsed_sec"] >= 0.0


def test_direct_worker_server_health_reports_capabilities():
    client = TestClient(server.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "direct-worker"
    assert payload["version"]
    assert "git_sha" in payload
    assert payload["capabilities"]["route_profile_contract"] is True
    assert payload["capabilities"]["direct_pymatting_known_b"] is True
    assert payload["capabilities"]["direct_corridorkey"] is True
    assert payload["capabilities"]["direct_known_bg_glow"] is True
    assert payload["capabilities"]["batch_matte"] is True


def test_direct_worker_server_matte_endpoint_can_force_corridorkey(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    captured = {}

    monkeypatch.setattr(
        server,
        "classify_route",
        lambda *args, **kwargs: RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={"execution_profile": "pymatting-hard-button"},
            confidence=1.0,
            reasons=["test"],
            analysis={"corridorkey_analysis": {"parameter_profile": "opaque_hard_ui_no_shadow"}},
        ),
    )

    def fake_run(prepared, **kwargs):
        captured["backend"] = prepared.decision.backend
        captured["route"] = prepared.decision.route
        captured["params"] = prepared.decision.params
        result = _result(prepared.rgb, execution_profile="corridorkey-shaped-icon")
        result.metadata["algorithm"] = prepared.decision.backend
        result.metadata["execution_backend"] = "direct-corridorkey"
        result.metadata["route"] = prepared.decision.route
        return result

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={
            "include_image": "false",
            "execution_backend": "direct-corridorkey",
            "corridorkey_gamma_space": "Linear",
            "corridorkey_despill_strength": "0.25",
            "corridorkey_refiner_strength": "1.5",
            "corridorkey_auto_despeckle": "Off",
            "corridorkey_despeckle_size": "64",
            "corridorkey_auto_mask": "true",
            "corridorkey_color_protection": "false",
            "corridorkey_protection_bg_max": "6",
            "corridorkey_protection_fg_min": "14",
            "corridorkey_preset": "detail_safe",
            "corridorkey_hard_ui_hint_mode": "translucent_button",
        },
    )

    assert response.status_code == 200
    assert captured["backend"] == "corridorkey"
    assert captured["route"] == "corridorkey"
    assert captured["params"]["corridorkey_gamma_space"] == "Linear"
    assert captured["params"]["corridorkey_despill_strength"] == 0.25
    assert captured["params"]["corridorkey_refiner_strength"] == 1.5
    assert captured["params"]["corridorkey_auto_despeckle"] == "Off"
    assert captured["params"]["corridorkey_despeckle_size"] == 64
    assert captured["params"]["corridorkey_auto_mask"] is True
    assert captured["params"]["corridorkey_color_protection"] is False
    assert captured["params"]["corridorkey_protection_bg_max"] == 6.0
    assert captured["params"]["corridorkey_protection_fg_min"] == 14.0
    assert captured["params"]["corridorkey_preset"] == "detail_safe"
    assert captured["params"]["corridorkey_hard_ui_hint_mode"] == "translucent_button"
    payload = response.json()
    assert payload["algorithm"] == "corridorkey"
    assert payload["execution_backend"] == "direct-corridorkey"
    assert payload["execution_backend"] == "direct-corridorkey"
    assert payload["route"] == "corridorkey"


def test_direct_worker_server_matte_endpoint_can_force_known_bg_glow(monkeypatch):
    rgb = np.full((16, 16, 3), (0, 200, 0), dtype=np.uint8)
    rgb[4:12, 4:12] = (80, 230, 255)
    captured = {}

    monkeypatch.setattr(
        server,
        "classify_route",
        lambda *args, **kwargs: RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={"execution_profile": "pymatting-hard-button"},
            confidence=1.0,
            reasons=["test"],
            analysis={"corridorkey_analysis": {"background_color": [0, 200, 0]}},
        ),
    )

    def fake_run(prepared, **kwargs):
        captured["backend"] = prepared.decision.backend
        captured["route"] = prepared.decision.route
        captured["params"] = prepared.decision.params
        captured["analysis"] = prepared.decision.analysis
        result = _result(prepared.rgb, execution_profile="known-bg-glow")
        result.response.strategy_name = "direct_known_bg_glow"
        result.response.debug["known_bg_glow"] = {"mode": prepared.decision.params["known_bg_glow_mode"]}
        result.metadata["algorithm"] = prepared.decision.backend
        result.metadata["execution_backend"] = "direct-known-bg-glow"
        result.metadata["route"] = prepared.decision.route
        result.metadata["execution_profile"] = prepared.decision.params["execution_profile"]
        return result

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={
            "include_image": "false",
            "execution_backend": "direct-known-bg-glow",
            "known_bg_glow_material_strength": "1.7",
        },
    )

    assert response.status_code == 200
    assert captured["backend"] == "known_bg_glow"
    assert captured["route"] == "known_bg_glow"
    assert captured["params"]["execution_profile"] == "known-bg-glow"
    assert captured["params"]["known_bg_glow_mode"] in {"single_target_line", "adaptive_ray", "chromatic_swap_ray"}
    assert captured["params"]["known_bg_glow_bg_color"] == (0, 200, 0)
    assert captured["params"]["known_bg_glow_material_strength"] == 1.7
    assert captured["analysis"]["known_bg_glow"]["background_color"] == [0, 200, 0]
    payload = response.json()
    assert payload["algorithm"] == "known_bg_glow"
    assert payload["execution_backend"] == "direct-known-bg-glow"
    assert payload["execution_backend"] == "direct-known-bg-glow"
    assert payload["route"] == "known_bg_glow"
    assert payload["algorithm_debug"]["known_bg_glow"]["mode"] == captured["params"]["known_bg_glow_mode"]


def test_direct_worker_manual_known_bg_glow_preserves_chromatic_swap_ray_mode():
    image = np.asarray(
        Image.open(
            PROJECT_ROOT / "samples/corridorkey_semantic/icon/icon_icon_d01_soft_alpha_glow_hard_core/green.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )
    decision = server.classify_route(image)

    assert decision.backend == "known_bg_glow"
    assert decision.params["known_bg_glow_mode"] == "chromatic_swap_ray"

    forced = server._apply_execution_backend_override(
        decision,
        "direct-known-bg-glow",
        rgb=image,
        fallback_bg_color=(0, 200, 0),
    )

    assert forced.backend == "known_bg_glow"
    assert forced.route == "known_bg_glow"
    assert forced.params["known_bg_glow_mode"] == "chromatic_swap_ray"
    assert forced.params["known_bg_glow_bg_color"] == decision.params["known_bg_glow_bg_color"]
    assert forced.params["known_bg_glow_target_color"] == decision.params["known_bg_glow_target_color"]


def test_direct_worker_server_omitted_corridorkey_forms_preserve_route_params(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    captured = {}

    monkeypatch.setattr(
        server,
        "classify_route",
        lambda *args, **kwargs: RouteDecision(
            route="corridorkey",
            asset_kind="icon",
            backend="corridorkey",
            params={
                "execution_profile": "corridorkey-shaped-icon",
                "corridorkey_auto_mask": True,
                "corridorkey_protection_bg_max": 4.0,
                "corridorkey_protection_fg_min": 10.0,
            },
            confidence=1.0,
            reasons=["test"],
            analysis={"corridorkey_analysis": {"parameter_profile": "key_color_material"}},
        ),
    )

    def fake_run(prepared, **kwargs):
        captured["params"] = prepared.decision.params
        result = _result(prepared.rgb, execution_profile="corridorkey-shaped-icon")
        result.metadata["algorithm"] = prepared.decision.backend
        result.metadata["execution_backend"] = "direct-corridorkey"
        result.metadata["route"] = prepared.decision.route
        return result

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={"include_image": "false"},
    )

    assert response.status_code == 200
    assert captured["params"]["corridorkey_auto_mask"] is True
    assert captured["params"]["corridorkey_protection_bg_max"] == 4.0
    assert captured["params"]["corridorkey_protection_fg_min"] == 10.0


def test_direct_worker_server_batch_endpoint_preserves_order(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    decisions = [
        RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={"execution_profile": "pymatting-hard-button"},
            confidence=1.0,
            reasons=["test"],
        ),
        RouteDecision(
            route="corridorkey",
            asset_kind="icon",
            backend="corridorkey",
            params={"execution_profile": "corridorkey-effect-icon"},
            confidence=1.0,
            reasons=["test"],
            analysis={"corridorkey_analysis": {"parameter_profile": "screen_tinted_translucency"}},
        ),
    ]
    state = SimpleNamespace(index=0)

    def fake_classify_route(*args, **kwargs):
        decision = decisions[state.index]
        state.index += 1
        return decision

    def fake_run(prepared, **kwargs):
        profile = prepared.decision.params["execution_profile"]
        backend = "direct-corridorkey" if prepared.decision.backend == "corridorkey" else "direct-pymatting-known-b"
        result = _result(prepared.rgb, execution_profile=profile)
        result.metadata["algorithm"] = prepared.decision.backend
        result.metadata["execution_backend"] = backend
        return result

    monkeypatch.setattr(server, "classify_route", fake_classify_route)
    monkeypatch.setattr(server, "_run_prepared_main", fake_run)
    monkeypatch.setattr(server, "_executor", lambda: None)

    client = TestClient(server.app)
    response = client.post(
        "/batch-matte",
        files=[
            ("files", ("first.png", _png_bytes(rgb), "image/png")),
            ("files", ("second.png", _png_bytes(rgb), "image/png")),
        ],
        data={"include_images": "false"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok_count"] == 2
    assert [row["filename"] for row in payload["runs"]] == ["first.png", "second.png"]
    assert payload["runs"][0]["execution_profile"] == "pymatting-hard-button"
    assert payload["runs"][1]["execution_profile"] == "corridorkey-effect-icon"
