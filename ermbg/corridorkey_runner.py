"""Shared in-process CorridorKey runner used by Comfy and direct-worker paths."""

from __future__ import annotations

import sys
import time
import inspect
from pathlib import Path
from typing import Any

import numpy as np


def _image_to_numpy(image: Any) -> np.ndarray:
    arr = image[0].detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _mask_to_numpy(mask: Any | None) -> np.ndarray | None:
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


def _numpy_image_to_tensor(arr: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _numpy_mask_to_tensor(arr: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


class LocalCorridorKeyClient:
    """Run CorridorKey in the current Python process with one shared adapter.

    ERMBG has two server-side entry points: the Comfy custom node and the
    non-Comfy direct worker. Both must feed CorridorKey through this adapter so
    the same profile parameters and hint/mask conventions cannot drift.
    """

    _processor: Any | None = None
    _corridorkey_node: Any | None = None

    def __init__(
        self,
        *,
        backend_label: str,
        prompt_id: str,
        prefer_loaded_node: bool = True,
    ) -> None:
        self.backend_label = backend_label
        self.prompt_id = prompt_id
        self.prefer_loaded_node = bool(prefer_loaded_node)

    @staticmethod
    def _ensure_import_path() -> None:
        candidates = [
            Path.cwd() / "custom_nodes" / "ComfyUI-CorridorKey",
            Path(__file__).resolve().parents[1] / "ComfyUI-CorridorKey",
            Path("C:/ComfyUI/custom_nodes/ComfyUI-CorridorKey"),
            Path("E:/ComfyUI/custom_nodes/ComfyUI-CorridorKey"),
        ]
        for path in candidates:
            if path.exists():
                text = str(path)
                if text not in sys.path:
                    sys.path.insert(0, text)

    @staticmethod
    def _registry_node_class(name: str) -> Any | None:
        registry = sys.modules.get("nodes")
        if registry is None:
            try:
                import nodes as registry  # type: ignore[import-not-found]
            except Exception:
                registry = None
        mapping = getattr(registry, "NODE_CLASS_MAPPINGS", None)
        if isinstance(mapping, dict):
            node_cls = mapping.get(name)
            if node_cls is not None:
                return node_cls
        return None

    @classmethod
    def _loaded_corridorkey_node_class(cls) -> Any | None:
        registry_node = cls._registry_node_class("CorridorKey")
        if registry_node is not None:
            return registry_node

        for module in list(sys.modules.values()):
            mapping = getattr(module, "NODE_CLASS_MAPPINGS", None)
            if not isinstance(mapping, dict):
                continue
            node_cls = mapping.get("CorridorKey")
            if node_cls is not None:
                return node_cls

        cls._ensure_import_path()
        try:
            import nodes as corridor_nodes  # type: ignore[import-not-found]
        except Exception:
            return None
        node_cls = getattr(corridor_nodes, "CorridorKey", None)
        return node_cls

    @classmethod
    def _get_loaded_node(cls) -> Any | None:
        node_cls = cls._loaded_corridorkey_node_class()
        if node_cls is None:
            return None
        if cls._corridorkey_node is None or not isinstance(cls._corridorkey_node, node_cls):
            cls._corridorkey_node = node_cls()
        return cls._corridorkey_node

    @classmethod
    def _get_processor(cls) -> Any:
        cls._ensure_import_path()
        from corridor_key import CorridorKeyProcessor  # type: ignore[import-not-found]

        if cls._processor is None:
            cls._processor = CorridorKeyProcessor()
        return cls._processor

    @classmethod
    def _corridorkey_mask_tensor_from_hint(
        cls,
        hint: np.ndarray,
        *,
        screen_color: str,
        execution_profile: str,
        hint_source: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        image_to_mask_cls = cls._registry_node_class("ImageToMask")
        if image_to_mask_cls is not None:
            try:
                hint_u8 = np.clip(hint * 255.0 + 0.5, 0, 255).astype(np.uint8)
                hint_rgb = np.repeat(hint_u8[..., None], 3, axis=2)
                node = image_to_mask_cls()
                function_name = getattr(image_to_mask_cls, "FUNCTION", "image_to_mask")
                function = getattr(node, function_name)
                converted = function(_numpy_image_to_tensor(hint_rgb), "red")
                mask_tensor = converted[0] if isinstance(converted, (tuple, list)) else converted
                mask_arr = _mask_to_numpy(mask_tensor)
                if mask_arr is None:
                    raise RuntimeError("ImageToMask returned no mask")
                return mask_tensor, {
                    "convention": "comfy_image_to_mask_node",
                    "source_node": f"{image_to_mask_cls.__module__}.{image_to_mask_cls.__name__}",
                    "min": float(mask_arr.min()),
                    "max": float(mask_arr.max()),
                    "mean": float(mask_arr.mean()),
                }
            except Exception:
                pass

        corridorkey_mask = np.clip(hint, 0.0, 1.0).astype(np.float32)
        hint_min = float(corridorkey_mask.min()) if corridorkey_mask.size else 0.0
        hint_max = float(corridorkey_mask.max()) if corridorkey_mask.size else 0.0
        if hint_max <= 0.001:
            convention = "corridorkey_full_frame_zero_hint"
        elif hint_min >= 0.999:
            convention = "corridorkey_full_frame_foreground_hint"
        else:
            convention = "corridorkey_shaped_foreground_hint"
        return _numpy_mask_to_tensor(corridorkey_mask), {
            "convention": convention,
            "min": float(corridorkey_mask.min()),
            "max": float(corridorkey_mask.max()),
            "mean": float(corridorkey_mask.mean()),
        }

    def matte(
        self,
        image_srgb: np.ndarray,
        *,
        background_color: tuple[int, int, int] = (0, 200, 0),
        hint_alpha: np.ndarray | None = None,
        gamma_space: str = "sRGB",
        screen_color: str = "green",
        despill_strength: float = 1.0,
        refiner_strength: float = 1.0,
        auto_despeckle: str = "On",
        despeckle_size: int = 400,
        hint_source: str | None = None,
        apply_color_protection: bool = True,
        color_protection_bg_max: float = 12.0,
        color_protection_fg_min: float = 28.0,
        protect_hint_supported_material: bool = False,
        execution_profile: str = "auto",
    ) -> Any:
        if hint_alpha is None:
            from .probe.comfyui_corridorkey import build_corridorkey_hint

            hint_alpha = build_corridorkey_hint(image_srgb, background_color)
            hint_source = hint_source or "known_bg_chromatic_key_eroded_blur"
        else:
            hint_source = hint_source or "provided_alpha_hint"

        from .probe.comfyui_corridorkey import (
            ComfyCorridorKeyResult,
            KeyerThresholds,
            apply_key_color_protection,
        )

        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        image_tensor = _numpy_image_to_tensor(image_srgb)
        hint = np.clip(hint_alpha.astype(np.float32), 0.0, 1.0)
        hint_tensor, corridorkey_mask_debug = self._corridorkey_mask_tensor_from_hint(
            hint,
            screen_color=str(screen_color),
            execution_profile=str(execution_profile),
            hint_source=hint_source,
        )
        step_start = time.perf_counter()
        loaded_node = self._get_loaded_node() if self.prefer_loaded_node else None
        if loaded_node is not None:
            # The CorridorKey node wrapper is the source of truth for the Comfy
            # path. Direct worker uses the same wrapper when it can import it,
            # so profile parity is not split between two call conventions.
            runner = "loaded_comfy_node"
            runner_module = type(loaded_node).__module__
            node_kwargs = {
                "image": image_tensor,
                "mask": hint_tensor,
                "gamma_space": str(gamma_space),
                "despill_strength": float(despill_strength),
                "refiner_strength": float(refiner_strength),
                "auto_despeckle": str(auto_despeckle),
                "despeckle_size": int(despeckle_size),
                "unique_id": None,
            }
            if "screen_color" in inspect.signature(loaded_node.run).parameters:
                node_kwargs["screen_color"] = str(screen_color)
            foreground_tensor, alpha_tensor, _processed, _qc = loaded_node.run(**node_kwargs)
        else:
            self._ensure_import_path()
            from corridor_key import CorridorKeySettings  # type: ignore[import-not-found]

            settings_kwargs = {
                "gamma_space": str(gamma_space),
                "despill_strength": float(despill_strength),
                "refiner_strength": float(refiner_strength),
                "auto_despeckle": str(auto_despeckle),
                "despeckle_size": int(despeckle_size),
            }
            if "screen_color" in inspect.signature(CorridorKeySettings).parameters:
                settings_kwargs["screen_color"] = str(screen_color)
            settings = CorridorKeySettings(**settings_kwargs)
            runner = "direct_processor_fallback"
            runner_module = "corridor_key.CorridorKeyProcessor"
            foreground_tensor, alpha_tensor, _processed, _qc = self._get_processor().refine(
                image=image_tensor,
                mask=hint_tensor,
                settings=settings,
                progress_callback=lambda *_args: None,
            )
        timings["corridorkey_refine_sec"] = time.perf_counter() - step_start
        foreground = _image_to_numpy(foreground_tensor)
        raw_alpha = _mask_to_numpy(alpha_tensor)
        if raw_alpha is None:
            raise RuntimeError("CorridorKey returned no alpha")

        alpha = np.clip(raw_alpha.astype(np.float32), 0.0, 1.0)
        color_protection = np.zeros(alpha.shape, dtype=np.float32)
        protection_debug: dict[str, Any] = {"enabled": False}
        if apply_color_protection:
            step_start = time.perf_counter()
            foreground, alpha, color_protection, protection_stats = apply_key_color_protection(
                image_srgb=image_srgb,
                foreground_srgb=foreground,
                alpha=alpha,
                background_color=background_color,
                thresholds=KeyerThresholds(
                    bg_max=float(color_protection_bg_max),
                    fg_min=float(color_protection_fg_min),
                ),
                trusted_material_alpha=hint if protect_hint_supported_material else None,
            )
            timings["color_protection_sec"] = time.perf_counter() - step_start
            protection_debug = {"enabled": True, **protection_stats}
        alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rgba = np.dstack([foreground, alpha_u8]).astype(np.uint8)
        timings["total_sec"] = time.perf_counter() - total_start
        return ComfyCorridorKeyResult(
            rgba=rgba,
            alpha=alpha.astype(np.float32),
            foreground_srgb=foreground.astype(np.uint8),
            hint_alpha=hint,
            raw_alpha=np.clip(raw_alpha, 0.0, 1.0).astype(np.float32),
            color_protection_alpha=color_protection.astype(np.float32),
            debug={
                "backend": self.backend_label,
                "prompt_id": self.prompt_id,
                "server_image": "in_memory",
                "server_mask": "in_memory",
                "background_color": list(background_color),
                "settings": {
                    "gamma_space": gamma_space,
                    "screen_color": screen_color,
                    "despill_strength": float(despill_strength),
                    "refiner_strength": float(refiner_strength),
                    "auto_despeckle": auto_despeckle,
                    "despeckle_size": int(despeckle_size),
                    "apply_color_protection": bool(apply_color_protection),
                    "protect_hint_supported_material": bool(protect_hint_supported_material),
                    "execution_profile": execution_profile,
                    "runner": runner,
                    "runner_module": runner_module,
                },
                "hint": {
                    "source": hint_source,
                    "min": float(hint.min()),
                    "max": float(hint.max()),
                    "mean": float(hint.mean()),
                },
                "corridorkey_mask": corridorkey_mask_debug,
                "color_protection": protection_debug,
                "timings": timings,
            },
        )


__all__ = ["LocalCorridorKeyClient"]
