"""HTTP client for the ERMBG direct worker service."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from .api import ImageLike, MatteResponse
from .settings import get_direct_worker_url

DEFAULT_DIRECT_WORKER_URL = get_direct_worker_url()


def _to_png_bytes(image: ImageLike) -> bytes:
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            mask = arr.astype(np.float32)
            if mask.max(initial=0.0) <= 1.0:
                mask = mask * 255.0
            pil = Image.fromarray(np.clip(mask + 0.5, 0, 255).astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            pil = Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB")
        else:
            raise ValueError("direct worker image must be path, PIL image, or HxWx3/4 numpy array")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _rgba_from_base64(value: str) -> np.ndarray:
    data = base64.b64decode(value)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), dtype=np.uint8)


def matte_image_direct_worker(
    image: ImageLike,
    *,
    direct_worker_url: str = DEFAULT_DIRECT_WORKER_URL,
    execution_backend: str = "auto",
    shadow_mode: str = "auto",
    corridorkey_gamma_space: str | None = None,
    corridorkey_despill_strength: float | None = None,
    corridorkey_refiner_strength: float | None = None,
    corridorkey_auto_despeckle: str | None = None,
    corridorkey_despeckle_size: int | None = None,
    corridorkey_auto_mask: bool | None = None,
    corridorkey_color_protection: bool | None = None,
    corridorkey_protection_bg_max: float | None = None,
    corridorkey_protection_fg_min: float | None = None,
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    corridorkey_hint_mask: Any | None = None,
    corridorkey_hard_ui_hint_mode: str = "bbox_2px",
    known_bg_glow_material_strength: float | None = None,
    pymatting_method: str | None = None,
    pymatting_image_space: str | None = None,
    pymatting_bg_source: str | None = None,
    pymatting_bg_color: tuple[int, int, int] | None = None,
    pymatting_bg_threshold: float | None = None,
    pymatting_fg_threshold: float | None = None,
    pymatting_boundary_band_px: int | None = None,
    pymatting_auto_adapt: bool | None = None,
    pymatting_cg_maxiter: int | None = None,
    pymatting_cg_rtol: float | None = None,
    pymatting_trimap_mode: str | None = None,
    pymatting_unknown_grow_px: int | None = None,
    pymatting_input_preprocessed_known_b: bool | None = None,
    pymatting_background_normalization: dict[str, Any] | None = None,
    route_decision: dict[str, Any] | None = None,
    semantic_decision: dict[str, Any] | None = None,
    user_keep_mask: Any | None = None,
    user_remove_mask: Any | None = None,
    fallback_bg_color: tuple[int, int, int] = (0, 200, 0),
    timeout: float = 240.0,
) -> MatteResponse:
    """Run one image through the remote direct-worker HTTP backend."""
    files = {"image": ("input.png", _to_png_bytes(image), "image/png")}
    if corridorkey_hint_mask is not None:
        files["corridorkey_hint_mask"] = ("hint_mask.png", _to_png_bytes(corridorkey_hint_mask), "image/png")
    if user_keep_mask is not None:
        files["user_keep_mask"] = ("user_keep_mask.png", _to_png_bytes(user_keep_mask), "image/png")
    if user_remove_mask is not None:
        files["user_remove_mask"] = ("user_remove_mask.png", _to_png_bytes(user_remove_mask), "image/png")
    data = {
        "execution_backend": execution_backend,
        "shadow_mode": shadow_mode,
        "corridorkey_screen_mode": corridorkey_screen_mode,
        "corridorkey_preset": corridorkey_preset,
        "corridorkey_hard_ui_hint_mode": corridorkey_hard_ui_hint_mode,
        "fallback_bg_color": ",".join(str(int(c)) for c in fallback_bg_color),
        "include_image": "true",
    }
    if corridorkey_gamma_space is not None:
        data["corridorkey_gamma_space"] = corridorkey_gamma_space
    if corridorkey_despill_strength is not None:
        data["corridorkey_despill_strength"] = str(float(corridorkey_despill_strength))
    if corridorkey_refiner_strength is not None:
        data["corridorkey_refiner_strength"] = str(float(corridorkey_refiner_strength))
    if corridorkey_auto_despeckle is not None:
        data["corridorkey_auto_despeckle"] = corridorkey_auto_despeckle
    if corridorkey_despeckle_size is not None:
        data["corridorkey_despeckle_size"] = str(int(corridorkey_despeckle_size))
    if corridorkey_auto_mask is not None:
        data["corridorkey_auto_mask"] = "true" if corridorkey_auto_mask else "false"
    if corridorkey_color_protection is not None:
        data["corridorkey_color_protection"] = "true" if corridorkey_color_protection else "false"
    if corridorkey_protection_bg_max is not None:
        data["corridorkey_protection_bg_max"] = str(float(corridorkey_protection_bg_max))
    if corridorkey_protection_fg_min is not None:
        data["corridorkey_protection_fg_min"] = str(float(corridorkey_protection_fg_min))
    if known_bg_glow_material_strength is not None:
        data["known_bg_glow_material_strength"] = str(float(known_bg_glow_material_strength))
    if pymatting_method is not None:
        data["pymatting_method"] = str(pymatting_method)
    if pymatting_image_space is not None:
        data["pymatting_image_space"] = str(pymatting_image_space)
    if pymatting_bg_source is not None:
        data["pymatting_bg_source"] = str(pymatting_bg_source)
    if pymatting_bg_color is not None:
        data["pymatting_bg_color"] = ",".join(str(int(c)) for c in pymatting_bg_color)
    if pymatting_bg_threshold is not None:
        data["pymatting_bg_threshold"] = str(float(pymatting_bg_threshold))
    if pymatting_fg_threshold is not None:
        data["pymatting_fg_threshold"] = str(float(pymatting_fg_threshold))
    if pymatting_boundary_band_px is not None:
        data["pymatting_boundary_band_px"] = str(int(pymatting_boundary_band_px))
    if pymatting_auto_adapt is not None:
        data["pymatting_auto_adapt"] = "true" if pymatting_auto_adapt else "false"
    if pymatting_cg_maxiter is not None:
        data["pymatting_cg_maxiter"] = str(int(pymatting_cg_maxiter))
    if pymatting_cg_rtol is not None:
        data["pymatting_cg_rtol"] = str(float(pymatting_cg_rtol))
    if pymatting_trimap_mode is not None:
        data["pymatting_trimap_mode"] = str(pymatting_trimap_mode)
    if pymatting_unknown_grow_px is not None:
        data["pymatting_unknown_grow_px"] = str(int(pymatting_unknown_grow_px))
    if pymatting_input_preprocessed_known_b is not None:
        data["pymatting_input_preprocessed_known_b"] = "true" if pymatting_input_preprocessed_known_b else "false"
    if pymatting_background_normalization is not None:
        data["pymatting_background_normalization"] = json.dumps(pymatting_background_normalization)
    if route_decision is not None:
        data["route_decision"] = json.dumps(route_decision)
    if semantic_decision:
        data["semantic_decision"] = json.dumps(semantic_decision)
    response = requests.post(
        f"{direct_worker_url.rstrip('/')}/matte",
        files=files,
        data=data,
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    encoded = payload.get("rgba_png_base64")
    if not isinstance(encoded, str):
        raise RuntimeError("direct worker response missing rgba_png_base64")
    rgba = _rgba_from_base64(encoded)
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    background = payload.get("background")
    if isinstance(background, list) and len(background) == 3:
        background_color = tuple(int(c) for c in background)
    else:
        background_color = tuple(int(c) for c in fallback_bg_color)
    actual_execution_backend = str(payload.get("execution_backend") or "direct-worker")
    strategy_name = actual_execution_backend.replace("-", "_")
    requested_backend = "direct-worker" if execution_backend == "auto" else execution_backend
    auto_route = {
        "requested_backend": requested_backend,
        "requested_algorithm": execution_backend,
        "algorithm": payload.get("algorithm") or payload.get("route"),
        "execution_backend": payload.get("execution_backend"),
        "route": payload.get("route"),
        "asset_kind": payload.get("asset_kind"),
        "parameter_profile": payload.get("parameter_profile"),
        "execution_profile": payload.get("execution_profile"),
        "shadow_mode": payload.get("shadow_mode") or shadow_mode,
    }
    debug = {
        "backend": requested_backend,
        "execution_server_url": direct_worker_url.rstrip("/"),
        "direct_worker": payload,
        "auto_route": auto_route,
        "timings": payload.get("timings", {}),
        "server_elapsed_sec": payload.get("server_elapsed_sec"),
        "soft_mask": alpha,
    }
    return MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=rgba[..., :3].astype(np.uint8),
        strategy_name=strategy_name,
        background_color=background_color,  # type: ignore[arg-type]
        report={"direct_worker": payload},
        output_dir=None,
        debug=debug,
    )


__all__ = ["DEFAULT_DIRECT_WORKER_URL", "matte_image_direct_worker"]
