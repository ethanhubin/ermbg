"""ComfyUI custom node: ERMBG AutoMatte.

Single-node end-to-end matting that uses ERMBG's front-end router to pick the
right strategy (saturated/white/black/grey/passthrough) automatically.

Inputs:
  IMAGE (required)        — the image to matte
  MASK  (optional)        — source α from a previous step. If provided and
                            clean enough, the router may pass it through
                            without re-running the matting net.

Outputs:
  IMAGE  — clean foreground RGB (despilled)
  MASK   — final α
  STRING — one-line summary "strategy_name | despill | notes"
"""

from __future__ import annotations

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
    arr = mask[0].detach().cpu().numpy()
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _numpy_image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy [H, W, 3] uint8 → ComfyUI IMAGE [1, H, W, 3] float."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.unsqueeze(0)


def _numpy_mask_to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32))
    return t.unsqueeze(0)


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
            },
            "optional": {
                "source_mask": ("MASK", {"tooltip": "If you have an existing α (e.g. from a prior segment), pass it here. The router will reuse it when clean."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("foreground", "alpha", "summary")
    FUNCTION = "run"
    CATEGORY = "ERMBG"

    def run(
        self,
        image: torch.Tensor,
        despill: str,
        use_keyer: str,
        bg_color: str,
        matting_model: str,
        source_mask: torch.Tensor | None = None,
    ):
        rgb = _image_to_numpy(image)
        alpha = _mask_to_numpy(source_mask)

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
        )

        # Outputs: foreground RGB (premultiplied is more compositing-friendly,
        # but ComfyUI IMAGE convention is straight, so we return the despilled
        # foreground straight and α as a separate MASK).
        fg_tensor = _numpy_image_to_tensor(result.foreground_srgb)
        alpha_tensor = _numpy_mask_to_tensor(result.alpha)

        notes = result.report.get("strategy", {}).get("notes", "")
        despill_used = result.report.get("despill_method", "?")
        summary = f"{result.strategy_name} | despill={despill_used} | {notes}"

        return (fg_tensor, alpha_tensor, summary)


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


NODE_CLASS_MAPPINGS = {
    "ErmbgAutoMatte": ErmbgAutoMatte,
    "ErmbgClassify": ErmbgClassify,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ErmbgAutoMatte": "ERMBG AutoMatte",
    "ErmbgClassify": "ERMBG Classify (preview)",
}
