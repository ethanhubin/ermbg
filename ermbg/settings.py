"""Project configuration helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "ermbg.config.json"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "ermbg.local.json"
DEFAULT_DIRECT_WORKER_URL = "http://192.168.0.8:7871"
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


def get_comfy_url() -> str:
    return get_setting("services.comfy_url", env="COMFY_URL", default=DEFAULT_COMFY_URL).rstrip("/")


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_COMFY_URL",
    "DEFAULT_DIRECT_WORKER_URL",
    "LOCAL_CONFIG_PATH",
    "PROJECT_ROOT",
    "get_bool_setting",
    "get_comfy_url",
    "get_direct_worker_url",
    "get_setting",
    "read_dotenv_value",
]
