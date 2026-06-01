"""Shared ComfyUI connection settings."""

from __future__ import annotations

from .settings import get_comfy_url

FALLBACK_COMFY_URL = get_comfy_url()


DEFAULT_COMFY_URL = get_comfy_url()

__all__ = ["DEFAULT_COMFY_URL", "FALLBACK_COMFY_URL", "get_comfy_url"]
