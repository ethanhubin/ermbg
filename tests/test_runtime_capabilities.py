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


def test_inspect_direct_worker_runtime_reports_remote_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caps.requests, "get", lambda url, *, timeout: _FakeResponse({"status": "ok"}))

    payload = caps.inspect_direct_worker_runtime(direct_worker_url="http://192.168.0.8:7871", timeout=1.0)

    assert payload["location"] == "remote"


def test_inspect_direct_worker_runtime_reports_location_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, *, timeout: float) -> _FakeResponse:
        raise ConnectionError("refused")

    monkeypatch.setattr(caps.requests, "get", boom)

    # Even when the worker is down the UI must still be able to say *which*
    # worker (local vs remote) is unreachable.
    payload = caps.inspect_direct_worker_runtime(direct_worker_url="http://127.0.0.1:7871", timeout=1.0)

    assert payload["status"] == "error"
    assert payload["location"] == "local"


def test_collect_runtime_capabilities_skips_comfy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_comfy(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Comfy should not be probed by default")

    monkeypatch.setattr(caps, "inspect_comfy_runtime", fail_comfy)
    monkeypatch.setattr(caps, "inspect_direct_worker_runtime", lambda **kwargs: {"status": "ok", "kind": "worker"})

    payload = caps.collect_runtime_capabilities(timeout=1.0, include_object_info=False)

    assert payload["status"] == "ok"
    assert payload["comfy"]["status"] == "disabled"
    assert payload["direct_worker"]["kind"] == "worker"
