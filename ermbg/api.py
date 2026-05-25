"""High-level Python API for ERMBG.

Two entrypoints aimed at code integrators:

  ``matte_image(...)`` — one-shot: load image, route, matte, return RGBA + report.
  ``classify_image(...)`` — fast preview: return only the strategy that *would*
      be applied, without running the matting net.

Both accept a path (``str``/``Path``), a numpy array (RGB/RGBA uint8), or a
PIL ``Image``. ``matte_image`` optionally writes the standard output set
(rgba / alpha / foreground / trimap / report.json / qa/) to a directory.

Example::

    from ermbg import matte_image
    r = matte_image("input.png", output_dir="out/")
    r.rgba.shape          # (H, W, 4)
    r.strategy_name       # 'saturated_bg'
    r.report['qa']['edge_halo_score_mean']
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

from . import io as ermbg_io
from .matting import matte as _matte_internal
from .qa import run_qa
from .router import Strategy, classify_strategy
from .segmenter import build_segmenter

ImageLike = Union[str, Path, np.ndarray, Image.Image]
MaskLike = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class MatteResponse:
    """Result of ``matte_image``. ``rgba`` is always present; the rest is metadata."""

    rgba: np.ndarray                       # H×W×4 sRGB uint8
    alpha: np.ndarray                      # H×W float32 [0, 1]
    foreground_srgb: np.ndarray            # H×W×3 sRGB uint8
    strategy_name: str                     # e.g. 'saturated_bg' / 'rgba_passthrough'
    background_color: tuple[int, int, int] # measured B (sRGB)
    report: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None         # where files were written (if any)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _to_rgb_and_alpha(image: ImageLike) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """Normalize any supported input to (rgb_uint8, source_alpha_or_None, source_path_or_None)."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        rgb, alpha = ermbg_io.load_image_with_alpha(path)
        return rgb, alpha, str(path)

    if isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            return rgba[..., :3].copy(), rgba[..., 3].astype(np.float32) / 255.0, None
        return np.asarray(image.convert("RGB"), dtype=np.uint8), None, None

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError("ndarray input must be uint8 sRGB")
        if image.ndim == 3 and image.shape[2] == 4:
            return image[..., :3].copy(), image[..., 3].astype(np.float32) / 255.0, None
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy(), None, None
        raise ValueError(f"ndarray must be HxWx3 or HxWx4 uint8, got shape {image.shape}")

    raise TypeError(f"Unsupported input type: {type(image)}")


def _to_mask(mask: MaskLike | None, shape: tuple[int, int], name: str) -> np.ndarray | None:
    """Normalize a mask-like value to H×W float32 [0,1]."""
    if mask is None:
        return None

    if isinstance(mask, (str, Path)):
        arr = np.asarray(Image.open(mask).convert("L"), dtype=np.float32) / 255.0
    elif isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    elif isinstance(mask, np.ndarray):
        is_uint8 = mask.dtype == np.uint8
        arr = mask.astype(np.float32)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[..., 3]
            elif arr.shape[2] == 3:
                arr = arr.mean(axis=2)
            else:
                raise ValueError(f"{name} ndarray must be HxW, HxWx3, or HxWx4")
        if is_uint8 or arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0
    else:
        raise TypeError(f"Unsupported {name} type: {type(mask)}")

    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_image(image: ImageLike) -> Strategy:
    """Run only the front-end router. Cheap (no matting net), good for previews.

    Returns the ``Strategy`` that ``matte_image`` would use. Inspect
    ``.bg_type``, ``.image_type``, ``.notes``, ``.extras``.
    """
    rgb, alpha, _ = _to_rgb_and_alpha(image)
    return classify_strategy(rgb, source_alpha=alpha)


def matte_image(
    image: ImageLike,
    output_dir: str | Path | None = None,
    qa: bool = False,
    matting_model: str = "ZhengPeng7/BiRefNet-matting",
    backend: str = "auto",
    bg_color: tuple[int, int, int] = (0, 200, 0),
    despill: str | None = None,
    use_keyer: bool | None = None,
    subject_mask: MaskLike | None = None,
) -> MatteResponse:
    """Matte one image end-to-end.

    Args:
        image: path, numpy array (HxWx3 or HxWx4 uint8 sRGB), or PIL Image.
        output_dir: if set, write rgba/alpha/foreground/trimap PNGs and
            ``report.json`` here. If ``qa=True``, also writes ``qa/on_*.png``.
        qa: run multi-background composite QA. Adds ~6 image saves and the
            full halo/recomp/binarization metric block to the report.
        matting_model: HF id of BiRefNet variant.
        backend: ``auto`` | ``birefnet`` | ``grabcut``.
        bg_color: composite color used when an RGBA source is dirty enough
            that the router falls through to re-matte (since the matting net
            needs RGB on a known constant bg). Default is the green-screen
            target so the first stage's outputs route well.
        despill, use_keyer: optional manual overrides; default ``None`` lets
            the router decide.
        subject_mask: optional H×W ownership mask from an independent segmenter.
            When provided, ERMBG may repair keyer-supported low-alpha holes
            inside this mask without raising the subject's external soft edge.
    """
    rgb, alpha, src_path = _to_rgb_and_alpha(image)
    subject_support = _to_mask(subject_mask, rgb.shape[:2], "subject_mask")

    # If source has α but the router decides to re-matte, the matting net
    # needs RGB on a known bg, not the raw (possibly premul or leaky) RGB.
    strat_preview = classify_strategy(rgb, source_alpha=alpha)
    if alpha is not None and not strat_preview.passthrough:
        bg_arr = np.broadcast_to(np.asarray(bg_color, dtype=np.uint8), rgb.shape[:2] + (3,))
        a4 = alpha[..., None]
        rgb_lin = ermbg_io.srgb_to_linear(rgb)
        bg_lin = ermbg_io.srgb_to_linear(bg_arr)
        rgb = ermbg_io.linear_to_srgb_u8(a4 * rgb_lin + (1.0 - a4) * bg_lin)

    seg = build_segmenter(backend=backend, model_id=matting_model)
    result = _matte_internal(
        rgb,
        source_alpha=alpha,
        segmenter=seg,
        despill=despill,
        use_keyer=use_keyer,
        subject_support=subject_support,
    )

    # Build report.
    report: dict[str, Any] = {
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis is not None else None,
        "background_color": list(result.background_color),
        "despill_method": result.debug.get("despill_method"),
        "matting_model": matting_model,
        "keyer": result.debug.get("keyer", {}),
        "strategy": result.debug.get("strategy", {}),
    }

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", result.rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", result.alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", result.foreground_srgb)
        ermbg_io.save_mask(out_dir / f"{stem}_trimap.png", result.debug["trimap_u8"])

        if qa:
            qa_dir = out_dir / f"{stem}_qa"
            qa_metrics = run_qa(
                image_srgb=rgb,
                rgba=result.rgba,
                soft_mask=result.debug["soft_mask"],
                background_color=result.background_color,
                out_dir=qa_dir,
            )
            report["qa"] = qa_metrics
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2))

        (out_dir / f"{stem}.report.json").write_text(json.dumps(report, indent=2))

    elif qa:
        # qa requested without output dir: still compute metrics, just don't save composites
        qa_metrics = run_qa(
            image_srgb=rgb,
            rgba=result.rgba,
            soft_mask=result.debug["soft_mask"],
            background_color=result.background_color,
            out_dir=Path("/tmp/_ermbg_qa_discard"),  # writes happen here; user can ignore
        )
        report["qa"] = qa_metrics

    return MatteResponse(
        rgba=result.rgba,
        alpha=result.alpha,
        foreground_srgb=result.foreground_srgb,
        strategy_name=result.debug.get("strategy", {}).get("name", "unknown"),
        background_color=result.background_color,
        report=report,
        output_dir=out_dir,
    )


__all__ = ["matte_image", "classify_image", "MatteResponse", "ImageLike", "MaskLike"]
