"""Image I/O and sRGB <-> linear-RGB conversions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_with_alpha(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image, returning (rgb_uint8, source_alpha_or_None).

    If the source already carries an alpha channel (RGBA / LA / paletted PNG
    with transparency), the alpha is returned separately as float32 [0, 1] so
    the caller can decide whether to pass-through, re-matte, or composite.
    The RGB part is the raw subject color (NOT pre-multiplied), suitable for
    feeding into a matting model when the caller wants to override.
    """
    img = Image.open(str(path))
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[..., :3].copy()
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        return rgb, alpha
    return np.asarray(img.convert("RGB"), dtype=np.uint8), None


def load_rgb(
    path: str | Path,
    background: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Load an image as H x W x 3 sRGB uint8.

    If the source has an alpha channel, transparent regions are composited onto
    ``background`` (default: green-screen RGB(0, 200, 0)). PIL's default is to
    drop alpha and leave transparent pixels as black, which silently violates
    the "known background" contract of the matting pipeline.
    """
    rgb, alpha = load_image_with_alpha(path)
    if alpha is None:
        return rgb
    bg = background if background is not None else (0, 200, 0)
    bg_arr = np.broadcast_to(np.asarray(bg, dtype=np.uint8), rgb.shape[:2] + (3,))
    a = alpha[..., None]
    out_lin = a * srgb_to_linear(rgb) + (1.0 - a) * srgb_to_linear(bg_arr)
    return linear_to_srgb_u8(out_lin)


def load_rgba(path: str | Path) -> np.ndarray:
    img = Image.open(str(path)).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_uint8(image), mode="RGB").save(str(p))


def save_rgba(path: str | Path, rgba: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_uint8(rgba), mode="RGBA").save(str(p))


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    """Save a soft mask (float 0..1 or uint8) as 8-bit grayscale PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        m = np.clip(mask, 0.0, 1.0)
        m = (m * 255.0 + 0.5).astype(np.uint8)
    else:
        m = mask
    Image.fromarray(m, mode="L").save(str(p))


def _as_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = np.clip(arr, 0.0, 1.0) if arr.dtype.kind == "f" else arr.astype(np.float32) / 255.0
    return (a * 255.0 + 0.5).astype(np.uint8)


# --- sRGB <-> linear RGB ----------------------------------------------------
# IEC 61966-2-1 transform.

def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """sRGB (uint8 or float 0..1) -> linear RGB float32 0..1."""
    if image.dtype == np.uint8:
        x = image.astype(np.float32) / 255.0
    else:
        x = image.astype(np.float32)
    a = 0.055
    out = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return out.astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """linear RGB float -> sRGB float 0..1."""
    x = np.clip(image.astype(np.float32), 0.0, 1.0)
    a = 0.055
    out = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)
    return out.astype(np.float32)


def linear_to_srgb_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(linear_to_srgb(image), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
