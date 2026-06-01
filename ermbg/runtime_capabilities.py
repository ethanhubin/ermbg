"""Runtime capability probes for local, ComfyUI, and Direct Worker backends."""

from __future__ import annotations

from importlib import metadata
from typing import Any

import requests

from .comfy import DEFAULT_COMFY_URL
from .settings import get_direct_worker_url

DEFAULT_DIRECT_WORKER_URL = get_direct_worker_url()

ERMBG_COMFY_NODE_KEYS = (
    "ErmbgRouteMatte",
    "ErmbgRouteStrategy",
    "ErmbgPyMattingKnownB",
    "ErmbgClassify",
    "ErmbgMasksToImages",
)


def get_ermbg_version() -> str:
    """Return the installed package version without importing the full package API."""
    try:
        return metadata.version("ermbg")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def local_capabilities() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": get_ermbg_version(),
        "capabilities": {
            "router": True,
            "pymatting_known_b": True,
            "opencv_numpy": True,
            "heavy_generation_local": False,
        },
    }


def _error_payload(url: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "url": url.rstrip("/"),
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def inspect_comfy_runtime(
    *,
    comfy_url: str = DEFAULT_COMFY_URL,
    timeout: float = 3.0,
    include_object_info: bool = True,
) -> dict[str, Any]:
    """Probe a ComfyUI server and report ERMBG node availability.

    ``/object_info`` can be large, so callers should use this only for explicit
    capability checks and not in latency-sensitive image processing paths.
    """
    base_url = comfy_url.rstrip("/")
    payload: dict[str, Any] = {
        "status": "unknown",
        "url": base_url,
        "capabilities": {
            "system_stats": False,
            "object_info": False,
            "ermbg_route_matte": False,
            "ermbg_route_strategy": False,
            "ermbg_pymatting_known_b": False,
        },
        "nodes": {},
    }
    try:
        stats_response = requests.get(f"{base_url}/system_stats", timeout=timeout)
        stats_response.raise_for_status()
        payload["status"] = "ok"
        payload["capabilities"]["system_stats"] = True
        stats = stats_response.json()
        if isinstance(stats, dict):
            payload["system_stats"] = stats
    except Exception as exc:
        return _error_payload(base_url, exc)

    if not include_object_info:
        return payload

    try:
        object_response = requests.get(f"{base_url}/object_info", timeout=timeout)
        object_response.raise_for_status()
        object_info = object_response.json()
        if not isinstance(object_info, dict):
            raise TypeError("Comfy /object_info did not return a JSON object")
    except Exception as exc:
        payload["status"] = "partial"
        payload["object_info_error"] = str(exc)
        payload["object_info_error_type"] = type(exc).__name__
        return payload

    payload["capabilities"]["object_info"] = True
    nodes = {key: key in object_info for key in ERMBG_COMFY_NODE_KEYS}
    payload["nodes"] = nodes
    payload["capabilities"]["ermbg_route_matte"] = nodes.get("ErmbgRouteMatte", False)
    payload["capabilities"]["ermbg_route_strategy"] = nodes.get("ErmbgRouteStrategy", False)
    payload["capabilities"]["ermbg_pymatting_known_b"] = nodes.get("ErmbgPyMattingKnownB", False)
    return payload


def inspect_direct_worker_runtime(
    *,
    direct_worker_url: str = DEFAULT_DIRECT_WORKER_URL,
    timeout: float = 3.0,
) -> dict[str, Any]:
    base_url = direct_worker_url.rstrip("/")
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("Direct Worker /health did not return a JSON object")
    except Exception as exc:
        return _error_payload(base_url, exc)

    payload = dict(data)
    payload.setdefault("status", "ok")
    payload["url"] = base_url
    payload.setdefault("capabilities", {})
    return payload


def disabled_comfy_runtime(*, comfy_url: str = DEFAULT_COMFY_URL) -> dict[str, Any]:
    return {
        "status": "disabled",
        "url": comfy_url.rstrip("/"),
        "capabilities": {
            "system_stats": False,
            "object_info": False,
            "ermbg_route_matte": False,
            "ermbg_route_strategy": False,
            "ermbg_pymatting_known_b": False,
        },
        "nodes": {},
    }


def collect_runtime_capabilities(
    *,
    comfy_url: str = DEFAULT_COMFY_URL,
    direct_worker_url: str = DEFAULT_DIRECT_WORKER_URL,
    timeout: float = 3.0,
    include_object_info: bool = True,
    include_comfy: bool = False,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "local": local_capabilities(),
        "comfy": inspect_comfy_runtime(
            comfy_url=comfy_url,
            timeout=timeout,
            include_object_info=include_object_info,
        ) if include_comfy else disabled_comfy_runtime(comfy_url=comfy_url),
        "direct_worker": inspect_direct_worker_runtime(
            direct_worker_url=direct_worker_url,
            timeout=timeout,
        ),
    }


__all__ = [
    "ERMBG_COMFY_NODE_KEYS",
    "collect_runtime_capabilities",
    "disabled_comfy_runtime",
    "get_ermbg_version",
    "inspect_comfy_runtime",
    "inspect_direct_worker_runtime",
    "local_capabilities",
]
