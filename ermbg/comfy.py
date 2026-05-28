"""Shared ComfyUI connection settings."""

from __future__ import annotations

import os
from pathlib import Path

FALLBACK_COMFY_URL = "http://192.168.0.8:8000"


def _dotenv_paths() -> tuple[Path, ...]:
    project_root = Path(__file__).resolve().parents[1]
    cwd_env = Path.cwd() / ".env"
    project_env = project_root / ".env"
    if cwd_env == project_env:
        return (project_env,)
    return (cwd_env, project_env)


def _read_dotenv_value(name: str) -> str | None:
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


def get_comfy_url() -> str:
    return (os.environ.get("COMFY_URL") or _read_dotenv_value("COMFY_URL") or FALLBACK_COMFY_URL).rstrip("/")


DEFAULT_COMFY_URL = get_comfy_url()

__all__ = ["DEFAULT_COMFY_URL", "FALLBACK_COMFY_URL", "get_comfy_url"]
