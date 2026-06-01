"""OpenAI gpt-image-1 probe generator.

Uses POST /v1/images/edits — the image-edit endpoint accepts an image + a mask
that marks the region to repaint, plus an instruction. We mark everything
*outside* the dilated subject as paintable, so the model is constrained to
only change the background.

This backend has been the most reliable in our Phase 1 testing for the
'replace background with solid color' constraint, because gpt-image-1 is
instruction-tuned and respects mask boundaries strictly.

Auth: requires the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import cv2
import numpy as np
import requests
from loguru import logger
from PIL import Image

from .generator import ProbeGenerator


_API_URL = "https://api.openai.com/v1/images/edits"


_COLOR_BG_PHRASE: dict[tuple[int, int, int], str] = {
    (250, 250, 250): "pure white (#FAFAFA)",
    (8, 8, 8): "near-black (#080808)",
    (0, 200, 220): "saturated cyan (#00C8DC)",
    (220, 30, 180): "saturated magenta (#DC1EB4)",
    (0, 0, 255): "pure saturated blue (#0000FF)",
    (0, 80, 255): "saturated blue (#0050FF)",
    (0, 200, 60): "saturated green (#00C83C)",
}


def _instruction(color: tuple[int, int, int]) -> str:
    key = tuple(int(c) for c in color)
    bg = _COLOR_BG_PHRASE.get(key, f"solid color rgb{key}")
    return (
        f"Replace the masked region with a perfectly flat, uniform {bg} studio backdrop. "
        "The background must be a single solid color across the entire masked area, "
        "matching the requested RGB/hex value exactly wherever possible, "
        "with absolutely no shadow, no gradient, no texture, no objects, no environment. "
        "Do not modify or repaint anything outside the masked region."
    )


def _to_png_rgba_bytes(image_rgb: np.ndarray, alpha: np.ndarray | None = None) -> bytes:
    """Encode an image (with optional alpha) as PNG bytes for the API."""
    if alpha is None:
        rgba = np.dstack([image_rgb, np.full(image_rgb.shape[:2], 255, np.uint8)])
    else:
        rgba = np.dstack([image_rgb, alpha])
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _build_mask_alpha(subject_mask: np.ndarray, dilate_radius: int) -> np.ndarray:
    """Mask alpha for the API: opaque (255) = preserve, transparent (0) = repaint.

    Subject region is dilated and marked opaque so the model never touches it.
    """
    soft = subject_mask.astype(np.float32)
    if soft.max() > 1.5:
        soft /= 255.0
    binary = (soft > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    locked = cv2.dilate(binary, kernel, iterations=dilate_radius)
    # API convention: alpha=0 = "edit here". So preserved subject alpha = 255.
    alpha = locked.astype(np.uint8)
    return alpha


class OpenAIImageProbeGenerator(ProbeGenerator):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-image-1",
        size: str = "auto",  # 'auto' | '1024x1024' | '1024x1536' | '1536x1024'
        quality: str = "high",  # 'low' | 'medium' | 'high' | 'auto'
        api_key: str | None = None,
        timeout: float = 240.0,
        subject_lock_dilate: int = 6,
    ) -> None:
        self.model = model
        self.size = size
        self.quality = quality
        self.timeout = timeout
        self.subject_lock_dilate = subject_lock_dilate
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAI generator requires OPENAI_API_KEY in the environment."
            )

    def _pick_size(self, w: int, h: int) -> tuple[int, int, str]:
        """Pick the closest gpt-image-1 size and target dims."""
        if self.size != "auto":
            tw, th = (int(x) for x in self.size.split("x"))
            return tw, th, self.size
        # gpt-image-1 sizes: 1024x1024, 1024x1536 (portrait), 1536x1024 (landscape)
        ar = w / h
        if 0.85 <= ar <= 1.18:
            return 1024, 1024, "1024x1024"
        if ar < 0.85:
            return 1024, 1536, "1024x1536"
        return 1536, 1024, "1536x1024"

    def generate(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
        background_color: tuple[int, int, int],
        seed: int | None = None,
        object_prompt: str | None = None,
    ) -> np.ndarray:
        del seed  # gpt-image-1 doesn't expose seeds
        del object_prompt  # not needed; mask is the constraint
        if image.dtype != np.uint8:
            raise ValueError("OpenAI generator expects uint8 sRGB input.")

        h, w = image.shape[:2]
        tw, th, size_str = self._pick_size(w, h)

        # Resize input + mask to API canvas (preserve aspect ratio via center-fit on a tw x th canvas).
        scale = min(tw / w, th / h)
        sw, sh = int(round(w * scale)), int(round(h * scale))
        image_r = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(subject_mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_LINEAR)
        # Pad to canvas with the target background color (so the padded area is also "background").
        bg_color_arr = np.array(background_color, dtype=np.uint8)
        canvas = np.broadcast_to(bg_color_arr, (th, tw, 3)).copy()
        mask_canvas = np.zeros((th, tw), dtype=np.float32)
        x0 = (tw - sw) // 2
        y0 = (th - sh) // 2
        canvas[y0:y0 + sh, x0:x0 + sw] = image_r
        mask_canvas[y0:y0 + sh, x0:x0 + sw] = mask_r

        alpha = _build_mask_alpha(mask_canvas, self.subject_lock_dilate)
        png_image = _to_png_rgba_bytes(canvas, alpha=np.full((th, tw), 255, np.uint8))
        png_mask = _to_png_rgba_bytes(canvas, alpha=alpha)

        instruction = _instruction(background_color)
        logger.debug(f"OpenAI edit: size={size_str} prompt='{instruction[:80]}...'")

        files = {
            "image": ("input.png", png_image, "image/png"),
            "mask": ("mask.png", png_mask, "image/png"),
        }
        data = {
            "model": self.model,
            "prompt": instruction,
            "size": size_str,
            "quality": self.quality,
            "n": "1",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(_API_URL, headers=headers, files=files, data=data, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI /images/edits failed: {resp.status_code} {resp.text[:500]}")
        payload = resp.json()
        b64 = payload["data"][0]["b64_json"]
        out_bytes = base64.b64decode(b64)
        out = np.asarray(Image.open(BytesIO(out_bytes)).convert("RGB"), dtype=np.uint8)

        # Crop back the centered region and resize to original HxW.
        out_crop = out[y0:y0 + sh, x0:x0 + sw]
        if out_crop.shape[:2] != (h, w):
            out_crop = cv2.resize(out_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out_crop
