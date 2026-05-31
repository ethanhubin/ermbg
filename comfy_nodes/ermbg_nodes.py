"""ComfyUI custom nodes for ERMBG route strategy and known-background matting."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from ermbg import matte_image
from ermbg.corridorkey_runner import LocalCorridorKeyClient


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()) if value.size else 0.0,
            "max": float(value.max()) if value.size else 0.0,
            "mean": float(value.mean()) if value.size else 0.0,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _optional_source_alpha(mask: torch.Tensor | None) -> np.ndarray | None:
    alpha = _mask_to_numpy(mask)
    if alpha is None:
        return None
    # Comfy LoadImage returns an all-zero mask for RGB files. For RGBA files its
    # MASK follows Comfy's transparency convention (0=opaque, 1=transparent), so
    # invert it back to ERMBG's alpha convention before routing.
    if float(alpha.max(initial=0.0)) <= 0.0:
        return None
    return np.clip(1.0 - alpha, 0.0, 1.0).astype(np.float32)


def _effective_backend(result: Any) -> str:
    auto_route = result.debug.get("auto_route")
    if isinstance(auto_route, dict):
        selected = auto_route.get("selected_backend")
        if isinstance(selected, str) and selected:
            return selected
    backend = result.debug.get("backend")
    if isinstance(backend, str) and backend:
        return backend
    return result.strategy_name.replace("_", "-")


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


class ErmbgPyMattingKnownB:
    """Known-background PyMatting solver for hard-edged generated assets.

    This node is intentionally narrow: it assumes the source image was rendered
    over a stable flat background and uses a conservative trimap around the
    known-background boundary. It is useful for A/B testing antialias recovery
    without running the full ERMBG or remote RMBG paths.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["cf", "knn", "lbdm", "lkm", "rw", "sm"], {"default": "cf"}),
                "image_space": (["linear", "sRGB"], {"default": "linear"}),
                "bg_source": (["auto", "green", "blue", "custom"], {"default": "auto"}),
                "bg_color": (
                    "STRING",
                    {"default": "0,200,0", "multiline": False, "tooltip": "R,G,B used when bg_source=custom"},
                ),
                "bg_threshold": (
                    "FLOAT",
                    {"default": 3.5, "min": 0.0, "max": 80.0, "step": 0.1},
                ),
                "fg_threshold": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.0, "max": 160.0, "step": 0.5},
                ),
                "boundary_band_px": (
                    "INT",
                    {"default": 2, "min": 0, "max": 64, "step": 1},
                ),
                "auto_adapt": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Auto-calibrate effective bg/fg thresholds and unknown band from the current image."},
                ),
                "cg_maxiter": (
                    "INT",
                    {"default": 1000, "min": 1, "max": 20000, "step": 50},
                ),
                "cg_rtol": (
                    "FLOAT",
                    {"default": 0.000001, "min": 0.000000001, "max": 0.01, "step": 0.000001},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "summary", "rgba_rgb", "trimap")
    FUNCTION = "run"
    CATEGORY = "ERMBG"

    def run(
        self,
        image: torch.Tensor,
        method: str,
        image_space: str,
        bg_source: str,
        bg_color: str,
        bg_threshold: float,
        fg_threshold: float,
        boundary_band_px: int,
        auto_adapt: bool,
        cg_maxiter: int,
        cg_rtol: float,
    ):
        reload_note = _dev_reload_ermbg_modules()
        rgb = _image_to_numpy(image)
        custom_bg = _bg_tuple(bg_color) if bg_source == "custom" else None

        result = matte_image(
            rgb,
            qa=False,
            backend="pymatting-known-b",
            shadow_mode="off",
            pymatting_method=method,
            pymatting_image_space=image_space,
            pymatting_bg_source=bg_source,
            pymatting_bg_color=custom_bg,
            pymatting_bg_threshold=bg_threshold,
            pymatting_fg_threshold=fg_threshold,
            pymatting_boundary_band_px=boundary_band_px,
            pymatting_auto_adapt=bool(auto_adapt),
            pymatting_cg_maxiter=cg_maxiter,
            pymatting_cg_rtol=cg_rtol,
        )

        fg_tensor = _numpy_image_to_tensor(result.foreground_srgb)
        rgba_rgb_tensor = _numpy_image_to_tensor(result.rgba[..., :3])
        alpha_tensor = _numpy_mask_to_tensor(result.alpha)
        trimap_tensor = _numpy_image_to_tensor(np.repeat(result.debug["trimap_u8"][..., None], 3, axis=2))

        pm = result.debug.get("pymatting_known_b", {})
        params = pm.get("parameters", {})
        trimap = pm.get("trimap", {})
        summary = (
            f"pymatting_known_b | method={params.get('method', method)} | "
            f"auto={params.get('auto_adapt', auto_adapt)} | "
            f"bg={tuple(result.background_color)} | unknown={trimap.get('unknown_pixels', '?')}"
        )
        if reload_note:
            summary = f"{summary} | {reload_note}"

        return (fg_tensor, alpha_tensor, summary, rgba_rgb_tensor, trimap_tensor)


class _LocalCorridorKeyClient(LocalCorridorKeyClient):
    def __init__(self) -> None:
        super().__init__(
            backend_label="comfy-corridorkey",
            prompt_id="local-comfy-node",
            prefer_loaded_node=True,
        )


class ErmbgRouteMatte:
    """Full ERMBG auto router + selected matting path inside ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_mode": (["on", "off", "auto"], {"default": "on"}),
                "corridorkey_screen_mode": (["auto", "green", "blue"], {"default": "auto"}),
                "corridorkey_preset": (["auto", "detail_safe", "spill_safe", "manual"], {"default": "auto"}),
                "corridorkey_hard_ui_hint_mode": (
                    [
                        "all_white",
                        "bbox_2px",
                        "boundary_2px",
                        "boundary_2px_shadow_safe",
                        "boundary_2px_shadow_safe_edge_floor",
                        "translucent_button",
                    ],
                    {"default": "bbox_2px"},
                ),
                "fallback_bg_color": ("STRING", {"default": "0,200,0", "multiline": False}),
                "pymatting_method": (["cf", "knn", "lbdm", "lkm", "rw", "sm"], {"default": "cf"}),
                "pymatting_image_space": (["linear", "sRGB"], {"default": "linear"}),
                "pymatting_bg_source": (["auto", "green", "blue", "custom"], {"default": "auto"}),
                "pymatting_bg_color": ("STRING", {"default": "0,200,0", "multiline": False}),
                "pymatting_bg_threshold": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 80.0, "step": 0.1}),
                "pymatting_fg_threshold": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 160.0, "step": 0.5}),
                "pymatting_boundary_band_px": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "pymatting_auto_adapt": ("BOOLEAN", {"default": True}),
                "pymatting_cg_maxiter": ("INT", {"default": 1000, "min": 1, "max": 20000, "step": 50}),
                "pymatting_cg_rtol": ("FLOAT", {"default": 0.000001, "min": 0.000000001, "max": 0.01, "step": 0.000001}),
            },
            "optional": {"source_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "summary", "rgba_rgb", "aux")
    FUNCTION = "run"
    CATEGORY = "ERMBG"
    OUTPUT_NODE = True

    def run(
        self,
        image: torch.Tensor,
        shadow_mode: str,
        corridorkey_screen_mode: str,
        corridorkey_preset: str,
        corridorkey_hard_ui_hint_mode: str,
        fallback_bg_color: str,
        pymatting_method: str,
        pymatting_image_space: str,
        pymatting_bg_source: str,
        pymatting_bg_color: str,
        pymatting_bg_threshold: float,
        pymatting_fg_threshold: float,
        pymatting_boundary_band_px: int,
        pymatting_auto_adapt: bool,
        pymatting_cg_maxiter: int,
        pymatting_cg_rtol: float,
        source_mask: torch.Tensor | None = None,
    ):
        reload_note = _dev_reload_ermbg_modules()
        from ermbg.api import (
            _auto_backend_for_image,
            _matte_image_comfy_corridorkey,
            _matte_image_passthrough,
            _matte_image_pymatting_known_b,
        )

        started = time.perf_counter()
        rgb = _image_to_numpy(image)
        source_alpha = _optional_source_alpha(source_mask)
        fallback_bg = _bg_tuple(fallback_bg_color)
        selected_backend, auto_route, decision = _auto_backend_for_image(
            rgb,
            source_alpha=source_alpha,
            screen_mode=corridorkey_screen_mode,
            preset=corridorkey_preset,
            fallback_background_color=fallback_bg,
        )
        params = dict(decision.params)
        if selected_backend == "passthrough":
            result = _matte_image_passthrough(
                rgb,
                source_alpha,
                src_path=None,
                output_dir=None,
                qa=False,
                auto_route=auto_route,
            )
        elif selected_backend == "comfy-pymatting-known-b":
            result = _matte_image_pymatting_known_b(
                rgb,
                src_path=None,
                output_dir=None,
                qa=False,
                shadow_mode=shadow_mode,
                method=params.get("pymatting_method", pymatting_method),
                image_space=params.get("pymatting_image_space", pymatting_image_space),
                bg_source=params.get("pymatting_bg_source", pymatting_bg_source),
                bg_color=params.get("pymatting_bg_color", _bg_tuple(pymatting_bg_color)),
                bg_threshold=params.get("pymatting_bg_threshold", pymatting_bg_threshold),
                fg_threshold=params.get("pymatting_fg_threshold", pymatting_fg_threshold),
                boundary_band_px=params.get("pymatting_boundary_band_px", pymatting_boundary_band_px),
                auto_adapt=params.get("pymatting_auto_adapt", pymatting_auto_adapt),
                cg_maxiter=params.get("pymatting_cg_maxiter", pymatting_cg_maxiter),
                cg_rtol=params.get("pymatting_cg_rtol", pymatting_cg_rtol),
                auto_route=auto_route,
            )
            result.strategy_name = "comfy_pymatting_known_b"
            result.report["strategy"]["name"] = "comfy_pymatting_known_b"
            result.report["strategy"]["notes"] = "PyMatting Known-B executed inside the ERMBG Comfy route node."
            result.debug["backend"] = "comfy-pymatting-known-b"
        elif selected_backend == "comfy-corridorkey":
            result = _matte_image_comfy_corridorkey(
                rgb,
                src_path=None,
                output_dir=None,
                qa=False,
                bg_color=fallback_bg,
                shadow_mode=shadow_mode,
                comfy_url="local-comfy-node",
                screen_mode=params.get("corridorkey_screen_mode", corridorkey_screen_mode),
                preset=params.get("corridorkey_preset", corridorkey_preset),
                hard_ui_hint_mode=params.get("corridorkey_hard_ui_hint_mode", corridorkey_hard_ui_hint_mode),
                corridorkey_client=_LocalCorridorKeyClient(),
                auto_mask=params.get("corridorkey_auto_mask", False),
                apply_color_protection=params.get("corridorkey_color_protection", None),
                color_protection_bg_max=params.get("corridorkey_protection_bg_max", 12.0),
                color_protection_fg_min=params.get("corridorkey_protection_fg_min", 28.0),
                gamma_space=params.get("corridorkey_gamma_space", "sRGB"),
                despill_strength=params.get("corridorkey_despill_strength", 1.0),
                refiner_strength=params.get("corridorkey_refiner_strength", 1.0),
                auto_despeckle=params.get("corridorkey_auto_despeckle", "On"),
                despeckle_size=params.get("corridorkey_despeckle_size", 400),
                execution_profile=params.get("corridorkey_execution_profile", "auto"),
                auto_route=auto_route,
            )
        elif selected_backend == "comfy-rmbg":
            raise RuntimeError("ErmbgRouteMatte auto no longer invokes RMBG fallback; route to PyMatting instead.")
        else:
            raise RuntimeError(f"Unsupported route backend in ErmbgRouteMatte: {selected_backend}")

        alpha = np.clip(result.alpha.astype(np.float32), 0.0, 1.0)
        foreground = result.foreground_srgb.astype(np.uint8)
        rgba_rgb = result.rgba[..., :3].astype(np.uint8)
        aux = np.repeat((alpha[..., None] * 255.0 + 0.5).astype(np.uint8), 3, axis=2)
        timings = dict(result.debug.get("timings", {})) if isinstance(result.debug.get("timings"), dict) else {}
        timings["node_total_sec"] = time.perf_counter() - started
        result.debug["timings"] = timings
        if reload_note:
            result.debug["dev_reload"] = reload_note
        payload = {
            "requested_backend": "auto",
            "selected_backend": _effective_backend(result),
            "route": auto_route.get("route"),
            "asset_kind": auto_route.get("asset_kind"),
            "parameter_profile": auto_route.get("parameter_profile"),
            "background_color": list(result.background_color),
            "strategy_name": result.strategy_name,
            "report": _json_safe(result.report),
            "debug": _json_safe(result.debug),
        }
        summary = json.dumps(payload, ensure_ascii=False)
        return {
            "ui": {"text": [summary]},
            "result": (
                _numpy_image_to_tensor(foreground),
                _numpy_mask_to_tensor(alpha),
                summary,
                _numpy_image_to_tensor(rgba_rgb),
                _numpy_image_to_tensor(aux),
            ),
        }


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
    "ErmbgRouteMatte": ErmbgRouteMatte,
    "ErmbgPyMattingKnownB": ErmbgPyMattingKnownB,
    "Convert Masks to Images": ConvertMasksToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ErmbgClassify": "ERMBG Classify (preview)",
    "ErmbgRouteStrategy": "ERMBG Route Strategy",
    "ErmbgRouteMatte": "ERMBG Route Matte",
    "ErmbgPyMattingKnownB": "ERMBG PyMatting Known-B",
    "Convert Masks to Images": "Convert Masks to Images",
}
