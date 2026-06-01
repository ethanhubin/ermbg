from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import ermbg.direct_worker_server as server
from ermbg.api import MatteResponse
from ermbg.direct_worker import DirectWorkerResult
from ermbg.router import RouteDecision


def _png_bytes(rgb: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _result(rgb: np.ndarray, *, execution_profile: str = "pymatting-hard-button") -> DirectWorkerResult:
    alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    response = MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=rgb,
        strategy_name="direct_pymatting_known_b",
        background_color=(0, 200, 0),
        report={"strategy": {"name": "direct_test"}},
        debug={},
    )
    return DirectWorkerResult(
        response=response,
        timings={"route_sec": 0.01, "backend_sec": 0.02},
        metadata={
            "selected_backend": "comfy-pymatting-known-b",
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
            backend="comfy-pymatting-known-b",
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
    assert "rgba_png_base64" not in payload
    assert payload["server_elapsed_sec"] >= 0.0


def test_direct_worker_server_health_reports_capabilities():
    client = TestClient(server.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "direct-worker"
    assert payload["version"]
    assert payload["capabilities"]["route_profile_contract"] is True
    assert payload["capabilities"]["direct_pymatting_known_b"] is True
    assert payload["capabilities"]["direct_corridorkey"] is True
    assert payload["capabilities"]["batch_matte"] is True


def test_direct_worker_server_batch_endpoint_preserves_order(monkeypatch):
    rgb = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
    decisions = [
        RouteDecision(
            route="pymatting_known_b",
            asset_kind="button",
            backend="comfy-pymatting-known-b",
            params={"execution_profile": "pymatting-hard-button"},
            confidence=1.0,
            reasons=["test"],
        ),
        RouteDecision(
            route="corridorkey",
            asset_kind="icon",
            backend="comfy-corridorkey",
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
        backend = "direct-corridorkey" if prepared.decision.backend == "comfy-corridorkey" else "direct-pymatting-known-b"
        result = _result(prepared.rgb, execution_profile=profile)
        result.metadata["selected_backend"] = prepared.decision.backend
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
