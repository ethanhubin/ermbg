"""ERMBG ComfyUI custom nodes.

Drop this directory under ``ComfyUI/custom_nodes/ermbg`` so ComfyUI picks up
``NODE_CLASS_MAPPINGS`` from ``ermbg_nodes.py``.
"""

from __future__ import annotations

from .ermbg_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
