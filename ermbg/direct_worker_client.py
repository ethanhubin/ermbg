"""HTTP client for the ERMBG direct worker service."""

from __future__ import annotations

import base64
import io
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
        if arr.ndim == 3 and arr.shape[2] >= 3:
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
    shadow_mode: str = "on",
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    corridorkey_hard_ui_hint_mode: str = "bbox_2px",
    fallback_bg_color: tuple[int, int, int] = (0, 200, 0),
    timeout: float = 240.0,
) -> MatteResponse:
    """Run one image through the remote direct-worker HTTP backend."""
    files = {"image": ("input.png", _to_png_bytes(image), "image/png")}
    data = {
        "shadow_mode": shadow_mode,
        "corridorkey_screen_mode": corridorkey_screen_mode,
        "corridorkey_preset": corridorkey_preset,
        "corridorkey_hard_ui_hint_mode": corridorkey_hard_ui_hint_mode,
        "fallback_bg_color": ",".join(str(int(c)) for c in fallback_bg_color),
        "include_image": "true",
    }
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
    execution_backend = str(payload.get("execution_backend") or "direct-worker")
    strategy_name = execution_backend.replace("-", "_")
    auto_route = {
        "requested_backend": "direct-worker",
        "selected_backend": payload.get("selected_backend"),
        "execution_backend": payload.get("execution_backend"),
        "route": payload.get("route"),
        "asset_kind": payload.get("asset_kind"),
        "parameter_profile": payload.get("parameter_profile"),
        "execution_profile": payload.get("execution_profile"),
    }
    debug = {
        "backend": "direct-worker",
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
