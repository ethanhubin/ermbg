"""Standard ERMBG run artifact manifests."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "ermbg.run.v1"


def _version() -> str:
    try:
        return metadata.version("ermbg")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def relpath(path: str | Path | None, root: str | Path) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(p)


def route_from_response(result: Any) -> dict[str, Any]:
    debug = getattr(result, "debug", {})
    auto_route = debug.get("auto_route") if isinstance(debug, dict) else None
    if not isinstance(auto_route, dict):
        auto_route = {}
    return {
        "algorithm": auto_route.get("algorithm") or auto_route.get("route"),
        "route": auto_route.get("route"),
        "asset_kind": auto_route.get("asset_kind"),
        "parameter_profile": auto_route.get("parameter_profile"),
        "execution_profile": auto_route.get("execution_profile"),
        "confidence": auto_route.get("confidence"),
        "reasons": auto_route.get("reasons"),
    }


def runtime_from_response(result: Any, *, requested_backend: str | None = None) -> dict[str, Any]:
    debug = getattr(result, "debug", {})
    runtime: dict[str, Any] = {
        "requested_backend": requested_backend,
        "backend": debug.get("backend") if isinstance(debug, dict) else None,
        "strategy": getattr(result, "strategy_name", None),
    }
    if isinstance(debug, dict):
        if "server_elapsed_sec" in debug:
            runtime["server_elapsed_sec"] = debug.get("server_elapsed_sec")
        direct_worker = debug.get("direct_worker")
        if isinstance(direct_worker, dict):
            runtime["kind"] = "direct-worker"
            runtime["execution_backend"] = direct_worker.get("execution_backend")
            runtime["algorithm"] = direct_worker.get("algorithm") or direct_worker.get("route")
            runtime["execution_server_url"] = debug.get("execution_server_url")
            runtime["execution_server"] = debug.get("execution_server")
            runtime["server_fallback_chain"] = debug.get("server_fallback_chain")
        elif str(runtime.get("backend") or "").startswith("comfy"):
            runtime["kind"] = "comfy"
        elif runtime.get("backend") == "direct-worker":
            runtime["kind"] = "direct-worker"
        else:
            runtime["kind"] = "local"
    return runtime


def build_run_manifest(
    *,
    run_dir: str | Path,
    input_path: str | Path | None = None,
    outputs: dict[str, str | Path | None] | None = None,
    request: dict[str, Any] | None = None,
    route: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    result: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    resolved_outputs = {
        key: relpath(value, root)
        for key, value in (outputs or {}).items()
        if value is not None
    }
    manifest = {
        "schema": SCHEMA,
        "input": relpath(input_path, root),
        "outputs": resolved_outputs,
        "request": request or {},
        "route": route if route is not None else (route_from_response(result) if result is not None else {}),
        "runtime": runtime if runtime is not None else (runtime_from_response(result) if result is not None else {}),
        "report": relpath(report_path, root),
        "versions": {"ermbg": _version()},
    }
    if result is not None:
        manifest["result"] = {
            "strategy": getattr(result, "strategy_name", None),
            "background": list(getattr(result, "background_color", []) or []),
        }
    if extra:
        manifest["extra"] = extra
    return json_safe(manifest)


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


__all__ = [
    "SCHEMA",
    "build_run_manifest",
    "json_safe",
    "relpath",
    "route_from_response",
    "runtime_from_response",
    "write_run_manifest",
]
