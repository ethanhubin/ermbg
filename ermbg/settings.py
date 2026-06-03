"""Project configuration helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "ermbg.config.json"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "ermbg.local.json"
DEFAULT_DIRECT_WORKER_URL = "http://127.0.0.1:7871"
DEFAULT_COMFY_URL = ""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _load_config() -> dict[str, Any]:
    # Machine-specific service endpoints and Web backend selection live in the
    # ignored local config so switching between Mac/remote/Windows environments
    # does not dirty the tracked default config.
    return _deep_merge(_load_json_config(CONFIG_PATH), _load_json_config(LOCAL_CONFIG_PATH))


def _configured_direct_worker_values() -> list[tuple[str, str, int]]:
    values: list[tuple[str, str, int]] = []
    for source, config in (("default", _load_json_config(CONFIG_PATH)), ("local", _load_json_config(LOCAL_CONFIG_PATH))):
        services = config.get("services") if isinstance(config, dict) else None
        if not isinstance(services, dict):
            continue
        direct_workers = services.get("direct_worker_urls")
        if isinstance(direct_workers, dict):
            for index, (name, url) in enumerate(direct_workers.items(), start=1):
                if isinstance(url, str) and url.strip():
                    values.append((str(name).strip() or source, url.strip(), index * 100))
        elif isinstance(direct_workers, list):
            for index, item in enumerate(direct_workers, start=1):
                if isinstance(item, str) and item.strip():
                    values.append((f"{source}-{index}", item.strip(), index * 100))
                elif isinstance(item, dict) and isinstance(item.get("url"), str) and item["url"].strip():
                    priority_raw = item.get("priority", index * 100)
                    try:
                        priority = int(priority_raw)
                    except (TypeError, ValueError):
                        priority = index * 100
                    values.append((str(item.get("name") or f"{source}-{index}").strip(), item["url"].strip(), priority))
        direct_worker_url = services.get("direct_worker_url")
        if isinstance(direct_worker_url, str) and direct_worker_url.strip():
            values.append((source, direct_worker_url.strip(), 0 if source == "local" else 50))
    return values


def _direct_worker_endpoint_name(url: str, fallback: str) -> str:
    lowered = url.lower()
    if "127.0.0.1" in lowered or "localhost" in lowered or "[::1]" in lowered:
        return "local"
    if fallback in {"default", "local"}:
        return "remote"
    return fallback


def direct_worker_location(url: str) -> str:
    """Classify a Direct Worker URL as ``local`` or ``remote`` for display.

    The Web UI needs an at-a-glance signal of which worker is in use; reading a
    raw IP (127.0.0.1 vs 192.168.x) and decoding it by hand is the source of the
    "am I on local or remote?" confusion. A loopback host is the only thing that
    is unambiguously this machine, so everything else is reported as remote.
    """
    lowered = url.lower()
    if "127.0.0.1" in lowered or "localhost" in lowered or "[::1]" in lowered:
        return "local"
    return "remote"


def get_direct_worker_servers() -> list[dict[str, Any]]:
    """Return configured Direct Worker servers ordered by priority.

    A local worker and a LAN/remote worker are the same kind of runtime here:
    just named URLs with priority and fallback order.
    """
    servers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, raw_url, priority in _configured_direct_worker_values():
        url = raw_url.rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        servers.append({"name": name or f"server-{len(servers) + 1}", "url": url, "priority": int(priority)})
    env_url = os.environ.get("ERMBG_DIRECT_URL") or read_dotenv_value("ERMBG_DIRECT_URL")
    if env_url and env_url.strip():
        url = env_url.strip().rstrip("/")
        servers = [server for server in servers if str(server.get("url")) != url]
        servers.insert(0, {"name": "env", "url": url, "priority": -100})
    primary = get_direct_worker_url()
    if primary and primary not in {str(server.get("url")) for server in servers}:
        servers.append({"name": "primary", "url": primary, "priority": 0})
    servers.sort(key=lambda server: (int(server.get("priority", 0)), str(server.get("name", ""))))
    return servers


def _dotenv_paths() -> tuple[Path, ...]:
    cwd_env = Path.cwd() / ".env"
    project_env = PROJECT_ROOT / ".env"
    if cwd_env == project_env:
        return (project_env,)
    return (cwd_env, project_env)


def read_dotenv_value(name: str) -> str | None:
    for env_path in _dotenv_paths():
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                value = value.strip().strip("\"'")
                return value or None
    return None


def _config_value(path: str) -> Any:
    value: Any = _load_config()
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def get_setting(path: str, *, env: str | None = None, default: str = "") -> str:
    if env:
        value = os.environ.get(env) or read_dotenv_value(env)
        if value:
            return value.strip()
    value = _config_value(path)
    if value is None:
        return default
    return str(value).strip()


def get_bool_setting(path: str, *, env: str | None = None, default: bool = False) -> bool:
    raw = get_setting(path, env=env, default="1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def get_direct_worker_url() -> str:
    return get_setting("services.direct_worker_url", env="ERMBG_DIRECT_URL", default=DEFAULT_DIRECT_WORKER_URL).rstrip("/")


def get_direct_worker_endpoints() -> dict[str, str]:
    """Return every configured Direct Worker endpoint, preserving local+remote.

    `get_direct_worker_url()` returns the primary runtime URL after env/local
    overrides. This companion intentionally reads the tracked and gitignored
    configs separately so a local remote override does not hide the tracked
    localhost default from Web diagnostics and backend selectors.
    """
    endpoints: dict[str, str] = {}
    for fallback, raw_url, _priority in _configured_direct_worker_values():
        url = raw_url.rstrip("/")
        if not url:
            continue
        base_name = _direct_worker_endpoint_name(url, fallback)
        name = base_name
        suffix = 2
        while name in endpoints and endpoints[name] != url:
            name = f"{base_name}-{suffix}"
            suffix += 1
        endpoints[name] = url
    env_url = os.environ.get("ERMBG_DIRECT_URL") or read_dotenv_value("ERMBG_DIRECT_URL")
    if env_url and env_url.strip():
        endpoints["env"] = env_url.strip().rstrip("/")
    primary = get_direct_worker_url()
    if primary not in endpoints.values():
        endpoints.setdefault(_direct_worker_endpoint_name(primary, "primary"), primary)
    return endpoints


def get_comfy_url() -> str:
    return get_setting("services.comfy_url", env="COMFY_URL", default=DEFAULT_COMFY_URL).rstrip("/")


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_COMFY_URL",
    "DEFAULT_DIRECT_WORKER_URL",
    "LOCAL_CONFIG_PATH",
    "PROJECT_ROOT",
    "direct_worker_location",
    "get_bool_setting",
    "get_comfy_url",
    "get_direct_worker_endpoints",
    "get_direct_worker_servers",
    "get_direct_worker_url",
    "get_setting",
    "read_dotenv_value",
]
