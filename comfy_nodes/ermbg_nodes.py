"""ComfyUI custom node: ERMBG AutoMatte.

Single-node end-to-end matting that uses ERMBG's front-end router to pick the
right strategy (saturated/white/black/grey/passthrough) automatically.

Inputs:
  IMAGE (required)        — the image to matte
  MASK  (optional)        — source α from a previous step. If provided and
                            clean enough, the router may pass it through
                            without re-running the matting net.

Outputs:
  IMAGE  — clean subject foreground RGB (despilled; not shadow-composited)
  MASK   — final α
  STRING — one-line summary "strategy_name | despill | notes"
  IMAGE  — RGB companion for final RGBA compositing, including shadow color
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import torch

from ermbg import matte_image


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


def _numpy_image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy [H, W, 3] uint8 → ComfyUI IMAGE [1, H, W, 3] float."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.unsqueeze(0)


def _numpy_mask_to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32))
    return t.unsqueeze(0)


def _dev_reload_ermbg_modules() -> str:
    """Reload algorithm modules when ERMBG_DEV_RELOAD=1 is set.

    ComfyUI keeps custom nodes and imported Python modules alive between
    prompts. During ERMBG algorithm work, synced source files on disk are
    otherwise invisible until a full ComfyUI restart. This opt-in path reloads
    pure Python decision/matting modules while leaving the API module and its
    segmenter cache alive, so iteration stays fast and model weights are not
    reloaded for ordinary threshold/topology changes.
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
        "ermbg.solid_graphic",
        "ermbg.router",
        "ermbg.matting",
    ]
    reloaded: list[str] = []
    for name in module_names:
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)
            reloaded.append(name.rsplit(".", 1)[-1])

    api = sys.modules.get("ermbg.api")
    router = sys.modules.get("ermbg.router")
    matting = sys.modules.get("ermbg.matting")
    if api is not None and router is not None and matting is not None:
        # Keep ermbg.api imported once so _SEGMENTER_CACHE survives dev reloads,
        # but refresh the function references that api.py bound at import time.
        api.classify_strategy = router.classify_strategy
        api._matte_internal = matting.matte

    return f"dev_reload={','.join(reloaded)}" if reloaded else "dev_reload=none"


class ErmbgAutoMatte:
    """Auto-routed matting. Strategy is decided per-image; user only chooses
    overrides if they really want to."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "despill": (
                    ["auto (router decides)", "unmix", "chroma_cap", "local_borrow", "closed_form", "none"],
                    {"default": "auto (router decides)"},
                ),
                "use_keyer": (["auto (router decides)", "force_on", "force_off"], {"default": "auto (router decides)"}),
                "bg_color": (
                    "STRING",
                    {"default": "0,200,0", "multiline": False, "tooltip": "R,G,B used when re-compositing dirty RGBA"},
                ),
                "matting_model": (
                    "STRING",
                    {"default": "ZhengPeng7/BiRefNet-matting", "multiline": False},
                ),
                "shadow_mode": (["on", "auto", "off"], {"default": "on"}),
            },
            "optional": {
                "source_mask": ("MASK", {"tooltip": "If you have an existing α (e.g. from a prior segment), pass it here. The router will reuse it when clean."}),
                "subject_mask": ("MASK", {"tooltip": "Independent subject ownership mask used only to repair subject-owned low-alpha holes."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "summary", "rgba_rgb")
    FUNCTION = "run"
    CATEGORY = "ERMBG"

    def run(
        self,
        image: torch.Tensor,
        despill: str,
        use_keyer: str,
        bg_color: str,
        matting_model: str,
        shadow_mode: str,
        source_mask: torch.Tensor | None = None,
        subject_mask: torch.Tensor | None = None,
    ):
        reload_note = _dev_reload_ermbg_modules()
        rgb = _image_to_numpy(image)
        alpha = _mask_to_numpy(source_mask)
        support = _mask_to_numpy(subject_mask)

        # If user passed a source mask, fold it into the rgba contract
        # `matte_image` expects (it accepts ndarray with HxWx4 or PIL with α).
        if alpha is not None:
            if alpha.shape != rgb.shape[:2]:
                # Try transpose (some ComfyUI nodes give MASK in [H, W])
                alpha = np.broadcast_to(alpha, rgb.shape[:2]).copy()
            rgba_in = np.dstack([rgb, (alpha * 255 + 0.5).astype(np.uint8)])
            input_arg = rgba_in
        else:
            input_arg = rgb

        despill_arg = None if despill.startswith("auto") else despill
        if use_keyer == "force_on":
            keyer_arg: bool | None = True
        elif use_keyer == "force_off":
            keyer_arg = False
        else:
            keyer_arg = None

        result = matte_image(
            input_arg,
            qa=False,
            matting_model=matting_model,
            bg_color=_bg_tuple(bg_color),
            despill=despill_arg,
            use_keyer=keyer_arg,
            subject_mask=support,
            shadow_mode=shadow_mode,
        )

        # Outputs keep two RGB semantics separate. ``foreground`` is clean
        # subject color for inspection/decontamination; ``rgba_rgb`` is the
        # color layer that must be paired with ``alpha`` to preserve shadows in
        # the final transparent result.
        fg_tensor = _numpy_image_to_tensor(result.foreground_srgb)
        rgba_rgb_tensor = _numpy_image_to_tensor(result.rgba[..., :3])
        alpha_tensor = _numpy_mask_to_tensor(result.alpha)

        notes = result.report.get("strategy", {}).get("notes", "")
        despill_used = result.report.get("despill_method", "?")
        summary = f"{result.strategy_name} | despill={despill_used} | {notes}"
        if reload_note:
            summary = f"{summary} | {reload_note}"

        return (fg_tensor, alpha_tensor, summary, rgba_rgb_tensor)


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


class ConvertMasksToImages:
    """Small compatibility node for ERMBG workflows.

    Some ComfyUI installs do not include a generic MASK -> IMAGE converter.
    Keeping this node in ERMBG makes the remote AutoMatte workflow depend only
    on core Comfy nodes plus ERMBG's own custom nodes.
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
    "ErmbgAutoMatte": ErmbgAutoMatte,
    "ErmbgClassify": ErmbgClassify,
    "Convert Masks to Images": ConvertMasksToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ErmbgAutoMatte": "ERMBG AutoMatte",
    "ErmbgClassify": "ERMBG Classify (preview)",
    "Convert Masks to Images": "Convert Masks to Images",
}
