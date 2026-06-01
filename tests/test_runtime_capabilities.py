from __future__ import annotations

from typing import Any

import pytest

from ermbg import runtime_capabilities as caps


class _FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def test_inspect_comfy_runtime_reports_ermbg_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, *, timeout: float) -> _FakeResponse:
        calls.append(url)
        assert timeout == 1.5
        if url.endswith("/system_stats"):
            return _FakeResponse({"system": {"os": "test"}})
        if url.endswith("/object_info"):
            return _FakeResponse(
                {
                    "ErmbgRouteMatte": {},
                    "ErmbgRouteStrategy": {},
                    "ErmbgPyMattingKnownB": {},
                }
            )
        raise AssertionError(url)

    monkeypatch.setattr(caps.requests, "get", fake_get)

    payload = caps.inspect_comfy_runtime(comfy_url="http://comfy.test/", timeout=1.5)

    assert payload["status"] == "ok"
    assert payload["url"] == "http://comfy.test"
    assert payload["capabilities"]["system_stats"] is True
    assert payload["capabilities"]["object_info"] is True
    assert payload["capabilities"]["ermbg_route_matte"] is True
    assert payload["capabilities"]["ermbg_route_strategy"] is True
    assert payload["capabilities"]["ermbg_pymatting_known_b"] is True
    assert calls == ["http://comfy.test/system_stats", "http://comfy.test/object_info"]


def test_inspect_comfy_runtime_can_skip_object_info(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, *, timeout: float) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse({"system": {"os": "test"}})

    monkeypatch.setattr(caps.requests, "get", fake_get)

    payload = caps.inspect_comfy_runtime(
        comfy_url="http://comfy.test",
        timeout=2.0,
        include_object_info=False,
    )

    assert payload["status"] == "ok"
    assert payload["capabilities"]["system_stats"] is True
    assert payload["capabilities"]["object_info"] is False
    assert payload["nodes"] == {}
    assert calls == ["http://comfy.test/system_stats"]


def test_inspect_direct_worker_runtime_passes_health_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, timeout: float) -> _FakeResponse:
        assert url == "http://worker.test/health"
        assert timeout == 2.5
        return _FakeResponse(
            {
                "status": "ok",
                "backend": "direct-worker",
                "version": "9.9.9",
                "capabilities": {"batch_matte": True},
            }
        )

    monkeypatch.setattr(caps.requests, "get", fake_get)

    payload = caps.inspect_direct_worker_runtime(direct_worker_url="http://worker.test/", timeout=2.5)

    assert payload["status"] == "ok"
    assert payload["url"] == "http://worker.test"
    assert payload["backend"] == "direct-worker"
    assert payload["version"] == "9.9.9"
    assert payload["capabilities"]["batch_matte"] is True


def test_collect_runtime_capabilities_combines_all_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caps, "inspect_comfy_runtime", lambda **kwargs: {"status": "ok", "kind": "comfy"})
    monkeypatch.setattr(caps, "inspect_direct_worker_runtime", lambda **kwargs: {"status": "ok", "kind": "worker"})

    payload = caps.collect_runtime_capabilities(timeout=1.0, include_object_info=False, include_comfy=True)

    assert payload["status"] == "ok"
    assert payload["local"]["capabilities"]["router"] is True
    assert payload["comfy"]["kind"] == "comfy"
    assert payload["direct_worker"]["kind"] == "worker"


def test_collect_runtime_capabilities_skips_comfy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_comfy(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Comfy should not be probed by default")

    monkeypatch.setattr(caps, "inspect_comfy_runtime", fail_comfy)
    monkeypatch.setattr(caps, "inspect_direct_worker_runtime", lambda **kwargs: {"status": "ok", "kind": "worker"})

    payload = caps.collect_runtime_capabilities(timeout=1.0, include_object_info=False)

    assert payload["status"] == "ok"
    assert payload["comfy"]["status"] == "disabled"
    assert payload["direct_worker"]["kind"] == "worker"
