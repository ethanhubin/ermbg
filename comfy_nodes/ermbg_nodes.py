"""ComfyUI custom nodes for ERMBG route preview utilities."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch


def _bg_tuple(s: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"bg_color must be 'R,G,B', got {s!r}")
    return tuple(parts)


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE [B, H, W, C] float [0,1] → numpy [H, W, 3] uint8 (first batch)."""
    arr = image[0].detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _mask_to_numpy(mask: torch.Tensor | None) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        arr = mask.detach().cpu().numpy()
    elif mask.ndim == 3:
        arr = mask[0].detach().cpu().numpy()
    elif mask.ndim == 4:
        arr = mask[0, ..., 0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported MASK tensor shape: {tuple(mask.shape)}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _dev_reload_ermbg_modules() -> str:
    """Reload route/matting helper modules when ERMBG_DEV_RELOAD=1 is set.

    ComfyUI keeps custom nodes and imported Python modules alive between
    prompts. During ERMBG algorithm work, synced source files on disk are
    otherwise invisible until a full ComfyUI restart. This opt-in path reloads
    pure Python route/helper modules while leaving the API module alive, so
    iteration stays fast for ordinary threshold/topology changes.
    """
    if os.environ.get("ERMBG_DEV_RELOAD", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return ""

    module_names = [
        "ermbg.colorspace",
        "ermbg.io",
        "ermbg.diagnose",
        "ermbg.keyer",
        "ermbg.despill",
        "ermbg.shadow",
        "ermbg.pymatting_refine",
        "ermbg.solid_graphic",
        "ermbg.corridorkey",
        "ermbg.router",
    ]
    reloaded: list[str] = []
    for name in module_names:
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)
            reloaded.append(name.rsplit(".", 1)[-1])

    api = sys.modules.get("ermbg.api")
    router = sys.modules.get("ermbg.router")
    if api is not None and router is not None:
        # Keep ermbg.api imported once, but refresh the function references that
        # api.py bound at import time.
        api.classify_strategy = router.classify_strategy
        api.classify_route = router.classify_route

    return f"dev_reload={','.join(reloaded)}" if reloaded else "dev_reload=none"


class ErmbgClassify:
    """Fast preview: returns the strategy ERMBG would pick, no matting net.

    Useful for branching ComfyUI graphs ("if bg is white, do X else Y") and
    for inspecting how the router classifies new generated images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"source_mask": ("MASK",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("bg_type", "image_type", "json")
    FUNCTION = "run"
    CATEGORY = "ERMBG"

    def run(self, image: torch.Tensor, source_mask: torch.Tensor | None = None):
        from ermbg import classify_image

        rgb = _image_to_numpy(image)
        alpha = _mask_to_numpy(source_mask)
        if alpha is not None:
            rgba = np.dstack([rgb, (alpha * 255 + 0.5).astype(np.uint8)])
            s = classify_image(rgba)
        else:
            s = classify_image(rgb)
        import json

        payload = {
            "name": s.name,
            "bg_type": s.bg_type,
            "image_type": s.image_type,
            "keyer_mode": s.keyer_mode,
            "despill": s.despill,
            "passthrough": s.passthrough,
            "notes": s.notes,
            "extras": s.extras,
        }
        return (s.bg_type, s.image_type, json.dumps(payload, indent=2, ensure_ascii=False))


class ErmbgRouteStrategy:
    """Server-side ERMBG route decision for ComfyUI graphs.

    This node runs the same deterministic route strategy as Web/API auto mode
    inside the Comfy process. Graphs can use its JSON to branch to CorridorKey,
    PyMatting Known-B, or RMBG without duplicating Mac-side routing logic.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "screen_mode": (["auto", "green", "blue"], {"default": "auto"}),
                "preset": (["auto", "detail_safe", "spill_safe", "manual"], {"default": "auto"}),
                "fallback_bg_color": (
                    "STRING",
                    {"default": "0,200,0", "multiline": False, "tooltip": "R,G,B fallback used when no screen color is detected"},
                ),
            },
            "optional": {"source_mask": ("MASK",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("backend", "route", "asset_kind", "json")
    FUNCTION = "run"
    CATEGORY = "ERMBG"

    def run(
        self,
        image: torch.Tensor,
        screen_mode: str,
        preset: str,
        fallback_bg_color: str,
        source_mask: torch.Tensor | None = None,
    ):
        reload_note = _dev_reload_ermbg_modules()
        from ermbg.router import classify_route

        rgb = _image_to_numpy(image)
        alpha = _mask_to_numpy(source_mask)
        decision = classify_route(
            rgb,
            source_alpha=alpha,
            screen_mode=screen_mode,
            preset=preset,
            fallback_background_color=_bg_tuple(fallback_bg_color),
        )

        import json

        payload = decision.to_dict()
        if reload_note:
            payload["dev_reload"] = reload_note
        return (
            decision.backend,
            decision.route,
            decision.asset_kind,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )


class ConvertMasksToImages:
    """Small compatibility node for ERMBG workflows.

    Some ComfyUI installs do not include a generic MASK -> IMAGE converter.
    Keeping this node in ERMBG lets remote RouteMatte/debug workflows depend
    only on core Comfy nodes plus ERMBG's own custom nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "ERMBG"

    def run(self, masks: torch.Tensor):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim == 3:
            images = masks.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif masks.ndim == 4 and masks.shape[-1] == 1:
            images = masks.repeat(1, 1, 1, 3)
        elif masks.ndim == 4 and masks.shape[-1] == 3:
            images = masks
        else:
            raise ValueError(f"Unsupported MASK tensor shape: {tuple(masks.shape)}")
        return (torch.clamp(images.float(), 0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "ErmbgClassify": ErmbgClassify,
    "ErmbgRouteStrategy": ErmbgRouteStrategy,
    "Convert Masks to Images": ConvertMasksToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ErmbgClassify": "ERMBG Classify (preview)",
    "ErmbgRouteStrategy": "ERMBG Route Strategy",
    "Convert Masks to Images": "Convert Masks to Images",
}
