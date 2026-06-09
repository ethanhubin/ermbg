from __future__ import annotations

import json
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


def _rgba_png_bytes(rgba: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _gray_png_bytes(gray: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(gray.astype(np.uint8), mode="L").save(buf, format="PNG")
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
                "trimap": {"unknown_pixels": int(trimap.size)},
                "parameters": {"fg_threshold": 24.0},
            },
            "shadow": {"method": "unknown_domain_same_background_reconstruction"},
            "hint": {"source": "provided_alpha_hint", "min": 0.0, "max": 1.0, "mean": 0.5},
            "corridorkey_mask": {"convention": "corridorkey_shaped_foreground_hint", "mean": 0.5},
            "semantic_execution": {
                "semantic_decision": {"policy": "review_only"},
                "semantic_decision_applied": False,
            },
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


def test_direct_worker_server_health_reports_corridorkey_unavailable(monkeypatch):
    monkeypatch.setattr(
        server,
        "_corridorkey_runtime_status",
        lambda: {
            "available": False,
            "torch_importable": True,
            "torch_cuda_available": False,
            "import_error": "No module named 'corridor_key'",
        },
    )

    client = TestClient(server.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["direct_corridorkey"] is False
    assert payload["corridorkey_runtime"]["available"] is False
    assert "corridor_key" in payload["corridorkey_runtime"]["import_error"]


def test_direct_worker_server_health_reports_corridorkey_available(monkeypatch):
    monkeypatch.setattr(
        server,
        "_corridorkey_runtime_status",
        lambda: {
            "available": True,
            "torch_importable": True,
            "torch_cuda_available": True,
            "runner": "direct_processor_fallback",
            "module": "corridor_key",
        },
    )

    client = TestClient(server.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["direct_corridorkey"] is True
    assert payload["corridorkey_runtime"]["runner"] == "direct_processor_fallback"


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
    assert payload["algorithm_debug"]["hint"]["source"] == "provided_alpha_hint"
    assert payload["algorithm_debug"]["corridorkey_mask"]["convention"] == "corridorkey_shaped_foreground_hint"
    assert payload["algorithm_debug"]["semantic_execution"] == {
        "semantic_decision": {"policy": "review_only"},
        "semantic_decision_applied": False,
    }
    assert "rgba_png_base64" not in payload
    assert "trimap_png_base64" not in payload
    assert payload["server_elapsed_sec"] >= 0.0


def test_direct_worker_server_routes_with_uploaded_source_alpha(monkeypatch):
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    rgba[..., :3] = (220, 30, 30)
    rgba[..., 3] = 255
    rgba[:2, :2, 3] = 0
    captured: dict[str, object] = {}

    def fake_classify_route(rgb, *, source_alpha, **kwargs):
        del kwargs
        captured["source_alpha"] = source_alpha
        return RouteDecision(
            route="rgba_passthrough",
            asset_kind="rgba",
            backend="rgba_passthrough",
            params={},
            confidence=1.0,
            reasons=["clean_source_alpha"],
        )

    def fake_run(prepared, **kwargs):
        del kwargs
        captured["prepared_alpha"] = prepared.source_alpha
        return DirectWorkerResult(
            response=MatteResponse(
                rgba=rgba,
                alpha=rgba[..., 3].astype(np.float32) / 255.0,
                foreground_srgb=rgba[..., :3],
                strategy_name="direct_passthrough",
                background_color=(0, 200, 0),
                debug={},
            ),
            timings={"route_sec": 0.01, "backend_sec": 0.02},
            metadata={
                "algorithm": "rgba_passthrough",
                "execution_backend": "direct-passthrough",
                "route": "rgba_passthrough",
                "asset_kind": "rgba",
                "parameter_profile": None,
                "execution_profile": None,
            },
        )

    monkeypatch.setattr(server, "classify_route", fake_classify_route)
    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _rgba_png_bytes(rgba), "image/png")},
        data={"include_image": "false"},
    )

    assert response.status_code == 200
    alpha = captured["source_alpha"]
    assert isinstance(alpha, np.ndarray)
    assert np.array_equal((alpha * 255.0 + 0.5).astype(np.uint8), rgba[..., 3])
    assert captured["prepared_alpha"] is alpha
    assert response.json()["execution_backend"] == "direct-passthrough"


def test_direct_worker_server_matte_endpoint_consumes_explicit_route_decision(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    captured: dict[str, object] = {}

    def fail_classify(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execute stage must not classify when route_decision is provided")

    def fake_run(prepared, **kwargs):
        del kwargs
        captured["decision"] = prepared.decision
        return _result(prepared.rgb)

    monkeypatch.setattr(server, "classify_route", fail_classify)
    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    route_decision = {
        "route": "pymatting_known_b",
        "algorithm": "pymatting_known_b",
        "backend": "pymatting_known_b",
        "asset_kind": "button",
        "execution_profile": "pymatting-hard-button",
        "confidence": 0.93,
        "reasons": ["analyze_contract"],
        "params": {"pymatting_bg_source": "custom", "pymatting_bg_color": [0, 200, 0]},
        "analysis": {"background": {"kind": "known_b"}},
    }

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={"include_image": "false", "route_decision": json.dumps(route_decision)},
    )

    assert response.status_code == 200
    decision = captured["decision"]
    assert isinstance(decision, RouteDecision)
    assert decision.route == "pymatting_known_b"
    assert decision.backend == "pymatting_known_b"
    assert decision.asset_kind == "button"
    assert decision.params["pymatting_bg_source"] == "custom"
    assert decision.params["pymatting_bg_color"] == [0, 200, 0]
    assert decision.analysis == {"background": {"kind": "known_b"}}
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["execution_backend"] == "direct-pymatting-known-b"


def test_direct_worker_server_explicit_route_decision_can_still_force_backend(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    captured: dict[str, object] = {}

    def fail_classify(*args, **kwargs):
        del args, kwargs
        raise AssertionError("execute stage must not classify when route_decision is provided")

    def fake_run(prepared, **kwargs):
        del kwargs
        captured["decision"] = prepared.decision
        result = _result(prepared.rgb, execution_profile="corridorkey-shaped-icon")
        result.metadata["algorithm"] = prepared.decision.backend
        result.metadata["execution_backend"] = "direct-corridorkey"
        result.metadata["route"] = prepared.decision.route
        return result

    monkeypatch.setattr(server, "classify_route", fail_classify)
    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    route_decision = {
        "route": "pymatting_known_b",
        "algorithm": "pymatting_known_b",
        "backend": "pymatting_known_b",
        "asset_kind": "button",
        "execution_profile": "pymatting-hard-button",
        "confidence": 0.93,
        "reasons": ["analyze_contract"],
        "params": {"pymatting_bg_source": "custom", "pymatting_bg_color": [0, 200, 0]},
        "analysis": {"corridorkey_analysis": {"parameter_profile": "opaque_hard_ui_no_shadow"}},
    }

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={
            "include_image": "false",
            "route_decision": json.dumps(route_decision),
            "execution_backend": "direct-corridorkey",
        },
    )

    assert response.status_code == 200
    decision = captured["decision"]
    assert isinstance(decision, RouteDecision)
    assert decision.route == "corridorkey"
    assert decision.backend == "corridorkey"
    assert "manual_direct_corridorkey_backend" in decision.reasons
    assert response.json()["execution_backend"] == "direct-corridorkey"


def test_direct_worker_server_accepts_known_b_preprocess_contract(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    captured: dict[str, object] = {}

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

    def fake_run(prepared, **kwargs):
        captured.update(prepared.decision.params)
        return _result(prepared.rgb)

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={
            "execution_backend": "direct-pymatting-known-b",
            "include_image": "false",
            "pymatting_bg_source": "custom",
            "pymatting_bg_color": "1,2,3",
            "pymatting_bg_threshold": "4.5",
            "pymatting_fg_threshold": "28",
        },
    )

    assert response.status_code == 200
    assert captured["pymatting_bg_source"] == "custom"
    assert captured["pymatting_bg_color"] == (1, 2, 3)
    assert captured["pymatting_bg_threshold"] == 4.5
    assert captured["pymatting_fg_threshold"] == 28.0
    assert not any(key.startswith("pymatting_") and "preprocess" in key for key in captured)
    assert not any(key.startswith("pymatting_") and "normalization" in key for key in captured)


def test_direct_worker_server_accepts_explicit_candidate_trimap(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    trimap = np.array([[0, 128, 255], [63, 191, 250]], dtype=np.uint8)
    captured: dict[str, object] = {}

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

    def fake_run(prepared, **kwargs):
        del kwargs
        captured.update(prepared.decision.params)
        return _result(prepared.rgb)

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={
            "image": ("case.png", _png_bytes(rgb), "image/png"),
            "pymatting_explicit_trimap": ("trimap.png", _gray_png_bytes(trimap), "image/png"),
        },
        data={"execution_backend": "direct-pymatting-known-b", "include_image": "false"},
    )

    assert response.status_code == 200
    explicit = captured["pymatting_explicit_trimap"]
    assert isinstance(explicit, np.ndarray)
    np.testing.assert_array_equal(explicit, np.array([[0, 128, 255], [0, 128, 255]], dtype=np.uint8))


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
    assert payload["capabilities"]["direct_corridorkey"] is payload["corridorkey_runtime"]["available"]
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


def test_direct_worker_server_matte_endpoint_passes_semantic_decision(monkeypatch):
    rgb = np.full((16, 16, 3), (255, 255, 255), dtype=np.uint8)
    captured = {}

    monkeypatch.setattr(
        server,
        "classify_route",
        lambda *args, **kwargs: RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={"execution_profile": "pymatting-known-bg"},
            confidence=1.0,
            reasons=["test"],
        ),
    )

    def fake_run(prepared, **kwargs):
        captured["params"] = prepared.decision.params
        return _result(prepared.rgb, execution_profile="pymatting-known-bg")

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={"image": ("case.png", _png_bytes(rgb), "image/png")},
        data={
            "include_image": "false",
            "semantic_decision": json.dumps({"enclosed_near_bg_policy": "subject"}),
        },
    )

    assert response.status_code == 200
    assert captured["params"]["semantic_decision"] == {"enclosed_near_bg_policy": "subject"}


def test_direct_worker_server_matte_endpoint_passes_user_masks(monkeypatch):
    rgb = np.full((8, 8, 3), (255, 255, 255), dtype=np.uint8)
    keep = np.zeros((8, 8), dtype=np.uint8)
    remove = np.zeros((8, 8), dtype=np.uint8)
    keep[2:4, 2:4] = 255
    remove[5:7, 5:7] = 255
    captured = {}

    monkeypatch.setattr(
        server,
        "classify_route",
        lambda *args, **kwargs: RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="pymatting_known_b",
            params={"execution_profile": "pymatting-known-bg"},
            confidence=1.0,
            reasons=["test"],
        ),
    )

    def fake_run(prepared, **kwargs):
        captured["params"] = prepared.decision.params
        return _result(prepared.rgb, execution_profile="pymatting-known-bg")

    monkeypatch.setattr(server, "_run_prepared_main", fake_run)

    client = TestClient(server.app)
    response = client.post(
        "/matte",
        files={
            "image": ("case.png", _png_bytes(rgb), "image/png"),
            "user_keep_mask": ("keep.png", _png_bytes(np.dstack([keep, keep, keep])), "image/png"),
            "user_remove_mask": ("remove.png", _png_bytes(np.dstack([remove, remove, remove])), "image/png"),
        },
        data={"include_image": "false"},
    )

    assert response.status_code == 200
    assert np.asarray(captured["params"]["user_keep_mask"]).sum() == 4.0
    assert np.asarray(captured["params"]["user_remove_mask"]).sum() == 4.0


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
                "corridorkey_hard_ui_hint_mode": "translucent_button",
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
    assert captured["params"]["corridorkey_hard_ui_hint_mode"] == "translucent_button"
    assert "corridorkey_screen_mode" not in captured["params"]
    assert "corridorkey_preset" not in captured["params"]
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
