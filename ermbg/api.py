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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image

from . import io as ermbg_io
from .artifacts import build_run_manifest, write_run_manifest
from .comfy import DEFAULT_COMFY_URL
from .corridorkey_hint import corridorkey_full_frame_prior_value
from .qa import run_qa
from .router import RouteDecision, Strategy, classify_route, classify_strategy
from .types import Trimap

ImageLike = Union[str, Path, np.ndarray, Image.Image]
MaskLike = Union[str, Path, np.ndarray, Image.Image]
_SEGMENTER_CACHE: dict[tuple[str, str, int, str], Any] = {}


def _mask_like_to_bool(mask: np.ndarray | None, shape: tuple[int, int], *, field: str) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise ValueError(f"{field} shape must match image shape")
    if arr.dtype == bool:
        return arr.astype(bool)
    values = arr.astype(np.float32)
    if float(values.max(initial=0.0)) > 1.5:
        values = values / 255.0
    return values >= 0.5


def _known_b_explicit_trimap(
    explicit_trimap: np.ndarray,
    *,
    shape: tuple[int, int],
    semantic_decision: dict[str, Any],
    user_keep_mask: np.ndarray | None,
    user_remove_mask: np.ndarray | None,
) -> tuple[Trimap, dict[str, Any]]:
    """Convert an Analyze candidate trimap preview into executor trimap state."""

    arr = np.asarray(explicit_trimap)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape:
        raise ValueError("pymatting_explicit_trimap shape must match image shape")
    values = arr.astype(np.float32)
    if float(values.max(initial=0.0)) <= 1.5:
        values = values * 255.0
    sure_bg = values < 64.0
    sure_fg = values > 191.0
    unknown = ~(sure_bg | sure_fg)
    keep_mask = _mask_like_to_bool(user_keep_mask, shape, field="user_keep_mask")
    remove_mask = _mask_like_to_bool(user_remove_mask, shape, field="user_remove_mask")
    keep_pixels = 0
    remove_pixels = 0
    if keep_mask is not None:
        keep_only = keep_mask.copy()
        if remove_mask is not None:
            keep_only &= ~remove_mask
        keep_pixels = int(keep_only.sum())
        sure_fg[keep_only] = True
        sure_bg[keep_only] = False
        unknown[keep_only] = False
    if remove_mask is not None:
        remove_pixels = int(remove_mask.sum())
        sure_bg[remove_mask] = True
        sure_fg[remove_mask] = False
        unknown[remove_mask] = False
    trimap = Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)
    info = {
        "method": "explicit_candidate_trimap",
        "source": "analyze_candidate_preview",
        "explicit": True,
        "semantic_decision": dict(semantic_decision),
        "user_mask_decision": {
            "keep_pixels": keep_pixels,
            "remove_pixels": remove_pixels,
            "remove_overrides_keep": True,
        },
        "sure_fg_pixels": int(trimap.sure_fg.sum()),
        "sure_bg_pixels": int(trimap.sure_bg.sum()),
        "unknown_pixels": int(trimap.unknown.sum()),
    }
    return trimap, info


def build_segmenter(*args: Any, **kwargs: Any) -> Any:
    """Compatibility stub for removed legacy full-matting callers."""
    del args, kwargs
    raise RuntimeError("legacy ERMBG full-matting segmenters were removed")


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
    debug: dict[str, Any] = field(default_factory=dict)


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


def _manifest_route_from_report(report: dict[str, Any]) -> dict[str, Any]:
    auto_route = report.get("auto_route")
    if isinstance(auto_route, dict):
        return {
            "algorithm": auto_route.get("algorithm") or auto_route.get("route") or auto_route.get("backend"),
            "route": auto_route.get("route"),
            "asset_kind": auto_route.get("asset_kind"),
            "parameter_profile": auto_route.get("parameter_profile"),
            "execution_profile": auto_route.get("execution_profile"),
            "confidence": auto_route.get("confidence"),
            "reasons": auto_route.get("reasons"),
        }
    strategy = report.get("strategy") if isinstance(report.get("strategy"), dict) else {}
    return {
        "algorithm": report.get("algorithm") or report.get("backend"),
        "route": strategy.get("bg_type"),
        "asset_kind": strategy.get("image_type"),
        "parameter_profile": None,
        "execution_profile": None,
    }


def _write_output_manifest(
    *,
    out_dir: Path,
    stem: str,
    src_path: str | None,
    report: dict[str, Any],
    outputs: dict[str, Path],
    report_path: Path,
    requested_backend: str | None = None,
) -> None:
    strategy = report.get("strategy") if isinstance(report.get("strategy"), dict) else {}
    debug = report.get("debug") if isinstance(report.get("debug"), dict) else {}
    runtime = {
        "requested_backend": requested_backend,
        "backend": report.get("backend") or debug.get("backend"),
        "strategy": strategy.get("name"),
        "server_elapsed_sec": debug.get("server_elapsed_sec") or report.get("server_elapsed_sec"),
    }
    manifest = build_run_manifest(
        run_dir=out_dir,
        input_path=src_path,
        outputs=outputs,
        request={"backend": requested_backend},
        route=_manifest_route_from_report(report),
        runtime=runtime,
        report_path=report_path,
        extra={"stem": stem},
    )
    write_run_manifest(out_dir / "manifest.json", manifest)


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


def classify_image_route(
    image: ImageLike,
    *,
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    bg_color: tuple[int, int, int] = (0, 200, 0),
) -> RouteDecision:
    """Run ERMBG's production auto-route strategy without executing matting."""
    rgb, alpha, _ = _to_rgb_and_alpha(image)
    return classify_route(
        rgb,
        source_alpha=alpha,
        screen_mode=corridorkey_screen_mode,
        preset=corridorkey_preset,
        fallback_background_color=bg_color,
    )


def _auto_backend_for_image(
    image_srgb: np.ndarray,
    *,
    source_alpha: np.ndarray | None = None,
    screen_mode: str,
    preset: str,
    fallback_background_color: tuple[int, int, int],
) -> tuple[str, dict[str, Any], RouteDecision]:
    """Route production auto mode through ERMBG's strategy layer."""
    decision = classify_route(
        image_srgb,
        source_alpha=source_alpha,
        screen_mode=screen_mode,  # type: ignore[arg-type]
        preset=preset,  # type: ignore[arg-type]
        fallback_background_color=fallback_background_color,
    )
    return decision.backend, decision.to_dict(), decision


def _pymatting_known_b_auto_background_fallback_info(
    *,
    selected_bg: tuple[int, int, int],
    auto_bg_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "accepted": True,
        "reason": "fallback_after_unstable_auto_background",
        "source": "auto_fallback_best_effort",
        "background_color": list(selected_bg),
        "auto_background": {
            **auto_bg_info,
            "accepted": False,
        },
    }


def prepare_known_b_preprocessed_input(
    image_srgb: np.ndarray,
    *,
    bg_source: str,
    bg_color: tuple[int, int, int] | None,
    bg_threshold: float,
    fg_threshold: float,
    adaptive: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int], dict[str, Any], dict[str, Any]]:
    """Resolve Known-B background and run the preprocess-owned normalization."""

    from .preprocess import repair_known_background_preprocess
    from .pymatting_refine import estimate_stable_background_color

    bg_source_key = str(bg_source or "auto").strip().lower()
    if bg_source_key == "auto":
        selected_bg, bg_info = estimate_stable_background_color(image_srgb)
        if not bg_info.get("accepted", False):
            bg_info = _pymatting_known_b_auto_background_fallback_info(
                selected_bg=selected_bg,
                auto_bg_info=bg_info,
            )
    elif bg_source_key == "green":
        selected_bg = (0, 200, 0)
        bg_info = {"accepted": True, "source": "preset_green", "reason": "preset"}
    elif bg_source_key == "blue":
        selected_bg = (0, 0, 200)
        bg_info = {"accepted": True, "source": "preset_blue", "reason": "preset"}
    elif bg_source_key == "custom":
        if bg_color is None:
            raise ValueError("pymatting_bg_color is required when pymatting_bg_source='custom'")
        selected_bg = tuple(int(np.clip(c, 0, 255)) for c in bg_color)
        bg_info = {"accepted": True, "source": "custom", "reason": "custom", "background_color": list(selected_bg)}
    else:
        raise ValueError("pymatting_bg_source must be auto, green, blue, or custom")

    normalized, decision = repair_known_background_preprocess(
        image_srgb,
        selected_bg,
        bg_threshold=float(bg_threshold),
        fg_threshold=float(fg_threshold),
        adaptive=bool(adaptive),
    )
    normalization = dict(decision.metadata.get("known_background_normalization") or {})
    preprocess_info = {
        "selected": decision.selected,
        "applied": decision.applied,
        "metadata": decision.metadata,
        "known_background_normalization": normalization,
    }
    return normalized, selected_bg, bg_info, preprocess_info


def matte_image(
    image: ImageLike,
    output_dir: str | Path | None = None,
    qa: bool = False,
    matting_model: str = "ZhengPeng7/BiRefNet-matting",
    backend: str = "auto",
    input_size: int = 1024,
    bg_color: tuple[int, int, int] = (0, 200, 0),
    despill: str | None = None,
    use_keyer: bool | None = None,
    subject_mask: MaskLike | None = None,
    shadow_mode: str = "auto",
    vlm_prior: bool = False,
    vlm_provider: str = "openai",
    vlm_model: str = "gpt-4o-mini",
    vlm_prior_mode: str = "shadow",
    comfy_url: str = DEFAULT_COMFY_URL,
    solid_graphic_prepass: bool = True,
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    solid_graphic_alpha_refiner: str = "heuristic",
    pymatting_method: str = "cf",
    pymatting_image_space: str = "linear",
    pymatting_bg_source: str = "auto",
    pymatting_bg_color: tuple[int, int, int] | None = None,
    pymatting_bg_threshold: float = 3.5,
    pymatting_fg_threshold: float = 24.0,
    pymatting_boundary_band_px: int = 2,
    pymatting_adapt_bg_threshold: bool = False,
    pymatting_adapt_fg_threshold: bool = True,
    pymatting_adapt_boundary_band: bool = True,
    pymatting_cg_maxiter: int = 1000,
    pymatting_cg_rtol: float = 1e-6,
    pymatting_trimap_mode: str = "standard",
    pymatting_unknown_grow_px: int = 0,
    semantic_decision: dict[str, Any] | None = None,
    user_keep_mask: MaskLike | None = None,
    user_remove_mask: MaskLike | None = None,
) -> MatteResponse:
    """Matte one image end-to-end.

    Args:
        image: path, numpy array (HxWx3 or HxWx4 uint8 sRGB), or PIL Image.
        output_dir: if set, write rgba/alpha/foreground/trimap PNGs and
            ``report.json`` here. If ``qa=True``, also writes ``qa/on_*.png``.
        qa: run multi-background composite QA. Adds ~6 image saves and the
            full halo/recomp/binarization metric block to the report.
        matting_model: deprecated compatibility argument for removed legacy
            full-matting backends. It is ignored by maintained route backends.
        backend: ``auto`` | ``auto-local`` | ``corridorkey`` |
            ``pymatting_known_b`` | ``known_bg_glow`` | ``passthrough``.
            ``auto`` routes through ERMBG's strategy layer, then dispatches to
            CorridorKey, PyMatting Known-B, Known-B Glow, or clean-RGBA passthrough.
        input_size: deprecated compatibility argument for removed legacy
            full-matting backends.
        bg_color: fallback composite color used when an RGBA source needs to be
            converted back to RGB before a selected backend runs.
        despill, use_keyer: deprecated compatibility arguments. Maintained
            backends derive despill/keyer behavior from the selected route.
        subject_mask: removed with the old ERMBG ownership-repair path.
        shadow_mode: ``auto`` uses backend defaults: PyMatting Known-B keeps
            shadow recovery on, while CorridorKey skips its shadow patch.
            ``on`` and ``off`` force the behavior.
        vlm_prior: removed with the old semantic ownership-repair path.
        vlm_provider, vlm_prior_mode: deprecated compatibility arguments.
        solid_graphic_prepass: deprecated compatibility argument; ERMBG auto
            routing now performs the maintained path selection.
        solid_graphic_alpha_refiner: deprecated compatibility argument.
        corridorkey_screen_mode: ``auto``, ``green``, or ``blue`` for the
            CorridorKey route estimation. ``auto`` estimates the key screen from
            border evidence during auto routing.
        corridorkey_preset: ``auto``, ``detail_safe``, ``spill_safe``, or
            ``manual``. Forwarded to auto routing for CorridorKey route/profile
            selection.
        pymatting_*: known-background PyMatting controls.
            The most important knobs are background source, trimap thresholds,
            and unknown-band width; solver knobs are exposed for A/B only.
    """
    rgb, alpha, src_path = _to_rgb_and_alpha(image)
    user_keep_alpha = _to_mask(user_keep_mask, rgb.shape[:2], "user_keep_mask")
    user_remove_alpha = _to_mask(user_remove_mask, rgb.shape[:2], "user_remove_mask")
    if subject_mask is not None:
        raise ValueError("subject_mask belonged to the removed legacy ERMBG matting path")
    if vlm_prior:
        raise ValueError("vlm_prior belonged to the removed legacy ERMBG matting path")
    if backend in {"birefnet", "grabcut", "comfy-ermbg"}:
        raise ValueError(f"backend={backend!r} was removed; use backend='auto' or an explicit routed backend")
    shadow_mode = str(shadow_mode or "auto").strip().lower()
    if shadow_mode not in {"auto", "on", "off"}:
        raise ValueError("shadow_mode must be 'auto', 'on', or 'off'")

    auto_route: dict[str, Any] | None = None
    auto_params: dict[str, Any] = {}
    if backend in {"auto", "auto-local"}:
        auto_decision: RouteDecision
        backend, auto_route, auto_decision = _auto_backend_for_image(
            rgb,
            source_alpha=alpha,
            screen_mode=corridorkey_screen_mode,
            preset=corridorkey_preset,
            fallback_background_color=bg_color,
        )
        auto_params = dict(auto_decision.params)
    if backend == "passthrough":
        return _matte_image_passthrough(rgb, alpha, src_path=src_path, output_dir=output_dir, qa=qa, auto_route=auto_route)

    remote_full_backends = {"corridorkey", "pymatting_known_b", "pymatting_fallback", "known_bg_glow"}
    local_known_bg_backends = {"pymatting-known-b", "direct-known-bg-glow"}
    if alpha is not None and (
        backend in remote_full_backends
        or backend in local_known_bg_backends
    ):
        bg_arr = np.broadcast_to(np.asarray(bg_color, dtype=np.uint8), rgb.shape[:2] + (3,))
        a4 = alpha[..., None]
        rgb_lin = ermbg_io.srgb_to_linear(rgb)
        bg_lin = ermbg_io.srgb_to_linear(bg_arr)
        rgb = ermbg_io.linear_to_srgb_u8(a4 * rgb_lin + (1.0 - a4) * bg_lin)

    if backend == "comfy-rmbg":
        return _matte_image_comfy_rmbg(
            rgb,
            src_path=src_path,
            output_dir=output_dir,
            qa=qa,
            bg_color=bg_color,
            comfy_url=comfy_url,
            auto_route=auto_route,
        )

    if backend in {"pymatting_known_b", "pymatting_fallback"}:
        if vlm_prior:
            raise ValueError("PyMatting Known-B does not support local vlm_prior")
        known_b_rgb, known_b_bg, known_b_bg_info, preprocess_info = prepare_known_b_preprocessed_input(
            rgb,
            bg_source=auto_params.get("pymatting_bg_source", pymatting_bg_source),
            bg_color=auto_params.get("pymatting_bg_color", pymatting_bg_color),
            bg_threshold=auto_params.get("pymatting_bg_threshold", pymatting_bg_threshold),
            fg_threshold=auto_params.get("pymatting_fg_threshold", pymatting_fg_threshold),
            adaptive=False,
        )
        return _matte_image_pymatting_known_b(
            known_b_rgb,
            src_path=src_path,
            output_dir=output_dir,
            qa=qa,
            shadow_mode=shadow_mode,
            method=auto_params.get("pymatting_method", pymatting_method),
            image_space=auto_params.get("pymatting_image_space", pymatting_image_space),
            bg_source="custom",
            bg_color=known_b_bg,
            bg_info_override=known_b_bg_info,
            requested_bg_source=auto_params.get("pymatting_bg_source", pymatting_bg_source),
            preprocess_info=preprocess_info,
            bg_threshold=auto_params.get("pymatting_bg_threshold", pymatting_bg_threshold),
            fg_threshold=auto_params.get("pymatting_fg_threshold", pymatting_fg_threshold),
            boundary_band_px=auto_params.get("pymatting_boundary_band_px", pymatting_boundary_band_px),
            adapt_bg_threshold=auto_params.get("pymatting_adapt_bg_threshold", pymatting_adapt_bg_threshold),
            adapt_fg_threshold=auto_params.get("pymatting_adapt_fg_threshold", pymatting_adapt_fg_threshold),
            adapt_boundary_band=auto_params.get("pymatting_adapt_boundary_band", pymatting_adapt_boundary_band),
            cg_maxiter=auto_params.get("pymatting_cg_maxiter", pymatting_cg_maxiter),
            cg_rtol=auto_params.get("pymatting_cg_rtol", pymatting_cg_rtol),
            trimap_mode=auto_params.get("pymatting_trimap_mode", pymatting_trimap_mode),
            unknown_grow_px=auto_params.get("pymatting_unknown_grow_px", pymatting_unknown_grow_px),
            semantic_decision=semantic_decision,
            user_keep_mask=user_keep_alpha,
            user_remove_mask=user_remove_alpha,
            auto_route=auto_route,
        )

    if backend in {"known_bg_glow", "direct-known-bg-glow"}:
        return _matte_image_known_bg_glow(
            rgb,
            src_path=src_path,
            output_dir=output_dir,
            qa=qa,
            bg_color=auto_params.get("known_bg_glow_bg_color", bg_color),
            target_color=auto_params.get("known_bg_glow_target_color", (255, 255, 255)),
            mode=auto_params.get("known_bg_glow_mode", "single_target_line"),
            auto_route=auto_route,
        )

    if backend == "pymatting-known-b":
        if vlm_prior:
            raise ValueError("backend='pymatting-known-b' does not support local vlm_prior")
        known_b_rgb, known_b_bg, known_b_bg_info, preprocess_info = prepare_known_b_preprocessed_input(
            rgb,
            bg_source=auto_params.get("pymatting_bg_source", pymatting_bg_source),
            bg_color=auto_params.get("pymatting_bg_color", pymatting_bg_color),
            bg_threshold=auto_params.get("pymatting_bg_threshold", pymatting_bg_threshold),
            fg_threshold=auto_params.get("pymatting_fg_threshold", pymatting_fg_threshold),
            adaptive=False,
        )
        return _matte_image_pymatting_known_b(
            known_b_rgb,
            src_path=src_path,
            output_dir=output_dir,
            qa=qa,
            shadow_mode=shadow_mode,
            method=auto_params.get("pymatting_method", pymatting_method),
            image_space=auto_params.get("pymatting_image_space", pymatting_image_space),
            bg_source="custom",
            bg_color=known_b_bg,
            bg_info_override=known_b_bg_info,
            requested_bg_source=auto_params.get("pymatting_bg_source", pymatting_bg_source),
            preprocess_info=preprocess_info,
            bg_threshold=auto_params.get("pymatting_bg_threshold", pymatting_bg_threshold),
            fg_threshold=auto_params.get("pymatting_fg_threshold", pymatting_fg_threshold),
            boundary_band_px=auto_params.get("pymatting_boundary_band_px", pymatting_boundary_band_px),
            adapt_bg_threshold=auto_params.get("pymatting_adapt_bg_threshold", pymatting_adapt_bg_threshold),
            adapt_fg_threshold=auto_params.get("pymatting_adapt_fg_threshold", pymatting_adapt_fg_threshold),
            adapt_boundary_band=auto_params.get("pymatting_adapt_boundary_band", pymatting_adapt_boundary_band),
            cg_maxiter=auto_params.get("pymatting_cg_maxiter", pymatting_cg_maxiter),
            cg_rtol=auto_params.get("pymatting_cg_rtol", pymatting_cg_rtol),
            trimap_mode=auto_params.get("pymatting_trimap_mode", pymatting_trimap_mode),
            unknown_grow_px=auto_params.get("pymatting_unknown_grow_px", pymatting_unknown_grow_px),
            semantic_decision=semantic_decision,
            user_keep_mask=user_keep_alpha,
            user_remove_mask=user_remove_alpha,
            auto_route=auto_route,
        )

    raise ValueError(f"Unsupported backend after ERMBG route cleanup: {backend!r}")


def _matte_image_known_bg_glow(
    rgb: np.ndarray,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    bg_color: tuple[int, int, int],
    target_color: tuple[int, int, int],
    mode: str = "single_target_line",
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    from .known_bg_glow import matte_known_bg_glow

    bg = tuple(int(np.clip(c, 0, 255)) for c in bg_color)
    target = tuple(int(np.clip(c, 0, 255)) for c in target_color)
    result = matte_known_bg_glow(rgb, bg, target, mode=mode)
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(bg),
        "despill_method": "known_bg_glow_line_solver",
        "matting_model": "none",
        "keyer": {},
        "shadow": {"mode": "off", "applied": False, "reason": "glow route has no shadow layer"},
        "semantic_prior": {},
        "strategy": {
            "name": "known_bg_glow",
            "bg_type": "known_background",
            "image_type": "glow_icon",
            "keyer_mode": "known_bg_glow",
            "despill": "line_unmix",
            "passthrough": False,
            "notes": "Simple glow solved directly from a known background mixing line.",
            "extras": result.debug,
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route
    if qa:
        report["qa"] = run_qa(
            image_srgb=rgb,
            rgba=result.rgba,
            soft_mask=result.alpha,
            background_color=bg,
            out_dir=Path("/tmp/_ermbg_api_glow_qa_discard"),
        )

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", result.rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", result.alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", result.foreground_srgb)
        report_path = out_dir / f"{stem}.report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_output_manifest(
            out_dir=out_dir,
            stem=stem,
            src_path=src_path,
            report=report,
            outputs={
                "rgba": out_dir / f"{stem}_rgba.png",
                "alpha": out_dir / f"{stem}_alpha.png",
                "foreground": out_dir / f"{stem}_foreground.png",
            },
            report_path=report_path,
            requested_backend="direct-known-bg-glow",
        )

    debug = {
        "backend": "direct-known-bg-glow",
        "known_bg_glow": result.debug,
        "strategy": report["strategy"],
        "soft_mask": result.alpha,
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=result.rgba,
        alpha=result.alpha,
        foreground_srgb=result.foreground_srgb,
        strategy_name="known_bg_glow",
        background_color=bg,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


def _matte_image_passthrough(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    if alpha is None:
        raise ValueError("passthrough route requires source alpha")
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    rgba = np.dstack([rgb, (a * 255.0 + 0.5).astype(np.uint8)])
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": [0, 0, 0],
        "despill_method": "passthrough",
        "matting_model": "none",
        "keyer": {},
        "shadow": {"mode": "passthrough", "applied": False},
        "semantic_prior": {},
        "strategy": {
            "name": "rgba_passthrough",
            "bg_type": "rgba_passthrough",
            "image_type": "rgba",
            "keyer_mode": None,
            "despill": "none",
            "passthrough": True,
            "notes": "Clean source alpha passed through by ERMBG route strategy.",
            "extras": {},
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route
    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", a)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", rgb)
        report_path = out_dir / f"{stem}.report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_output_manifest(
            out_dir=out_dir,
            stem=stem,
            src_path=src_path,
            report=report,
            outputs={
                "rgba": out_dir / f"{stem}_rgba.png",
                "alpha": out_dir / f"{stem}_alpha.png",
                "foreground": out_dir / f"{stem}_foreground.png",
            },
            report_path=report_path,
            requested_backend="passthrough",
        )
    elif qa:
        # Passthrough has no known observed background, so QA composites are not
        # meaningful. Keep the flag accepted for compatibility and report why.
        report["qa"] = {"skipped": True, "reason": "rgba_passthrough"}
    debug = {
        "strategy": report["strategy"],
        "soft_mask": a,
        "shadow_alpha": np.zeros(a.shape, dtype=np.float32),
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=rgba,
        alpha=a,
        foreground_srgb=rgb,
        strategy_name="rgba_passthrough",
        background_color=(0, 0, 0),
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


def _write_common_outputs(
    *,
    rgb: np.ndarray,
    rgba: np.ndarray,
    alpha: np.ndarray,
    foreground: np.ndarray,
    report: dict[str, Any],
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    background_color: tuple[int, int, int],
    requested_backend: str | None = None,
) -> Path | None:
    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", foreground)
        if qa:
            qa_dir = out_dir / f"{stem}_qa"
            qa_metrics = run_qa(
                image_srgb=rgb,
                rgba=rgba,
                soft_mask=alpha,
                background_color=background_color,
                out_dir=qa_dir,
            )
            report["qa"] = qa_metrics
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
        report_path = out_dir / f"{stem}.report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_output_manifest(
            out_dir=out_dir,
            stem=stem,
            src_path=src_path,
            report=report,
            outputs={
                "rgba": out_dir / f"{stem}_rgba.png",
                "alpha": out_dir / f"{stem}_alpha.png",
                "foreground": out_dir / f"{stem}_foreground.png",
            },
            report_path=report_path,
            requested_backend=requested_backend,
        )
    elif qa:
        qa_metrics = run_qa(
            image_srgb=rgb,
            rgba=rgba,
            soft_mask=alpha,
            background_color=background_color,
            out_dir=Path("/tmp/_ermbg_qa_discard"),
        )
        report["qa"] = qa_metrics
    return out_dir


def _matte_image_pymatting_known_b(
    rgb: np.ndarray,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    shadow_mode: str,
    method: str,
    image_space: str,
    bg_source: str,
    bg_color: tuple[int, int, int] | None,
    bg_threshold: float,
    fg_threshold: float,
    boundary_band_px: int,
    adapt_bg_threshold: bool,
    adapt_fg_threshold: bool,
    adapt_boundary_band: bool,
    cg_maxiter: int,
    cg_rtol: float,
    trimap_mode: str = "standard",
    unknown_grow_px: int = 0,
    bg_info_override: dict[str, Any] | None = None,
    requested_bg_source: str | None = None,
    preprocess_info: dict[str, Any] | None = None,
    semantic_decision: dict[str, Any] | None = None,
    user_keep_mask: np.ndarray | None = None,
    user_remove_mask: np.ndarray | None = None,
    explicit_trimap: np.ndarray | None = None,
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    from .pymatting_refine import (
        build_known_background_trimap,
        build_same_key_opaque_color_restore_mask,
        build_same_key_opaque_inner_opaque_mask,
        build_same_key_opaque_proxy_subject_mask,
        estimate_alpha_with_pymatting,
        estimate_stable_background_color,
    )
    from .trimap import trimap_to_uint8

    method = method.removeprefix("pymatting-").lower().replace("_", "-")
    if method not in {"cf", "knn", "lbdm", "lkm", "rw", "sm"}:
        raise ValueError(f"Unsupported pymatting method: {method!r}")
    if image_space not in {"linear", "sRGB"}:
        raise ValueError("pymatting_image_space must be 'linear' or 'sRGB'")
    bg_source = bg_source.strip().lower()
    if bg_source not in {"auto", "green", "blue", "custom"}:
        raise ValueError("pymatting_bg_source must be auto, green, blue, or custom")
    if not 0.0 <= float(bg_threshold) < float(fg_threshold):
        raise ValueError("pymatting_bg_threshold must be non-negative and less than pymatting_fg_threshold")
    if int(boundary_band_px) < 0:
        raise ValueError("pymatting_boundary_band_px must be >= 0")
    if int(cg_maxiter) <= 0:
        raise ValueError("pymatting_cg_maxiter must be > 0")
    if float(cg_rtol) <= 0.0:
        raise ValueError("pymatting_cg_rtol must be > 0")

    timings: dict[str, float] = {}
    total_started = time.perf_counter()
    step_started = time.perf_counter()
    bg_info: dict[str, Any]
    effective_bg_source = bg_source
    effective_adapt_bg_threshold = bool(adapt_bg_threshold)
    effective_adapt_fg_threshold = bool(adapt_fg_threshold)
    effective_adapt_boundary_band = bool(adapt_boundary_band)
    if bg_source == "auto":
        selected_bg, bg_info = estimate_stable_background_color(rgb)
        if not bg_info.get("accepted", False):
            bg_info = _pymatting_known_b_auto_background_fallback_info(
                selected_bg=selected_bg,
                auto_bg_info=bg_info,
            )
            effective_bg_source = "custom"
            effective_adapt_bg_threshold = False
    elif bg_source == "green":
        selected_bg = (0, 200, 0)
        bg_info = {"accepted": True, "source": "preset_green", "reason": "preset"}
    elif bg_source == "blue":
        selected_bg = (0, 0, 200)
        bg_info = {"accepted": True, "source": "preset_blue", "reason": "preset"}
    else:
        if bg_color is None:
            raise ValueError("pymatting_bg_color is required when pymatting_bg_source='custom'")
        selected_bg = tuple(int(np.clip(c, 0, 255)) for c in bg_color)
        bg_info = {"accepted": True, "source": "custom", "reason": "custom", "background_color": list(selected_bg)}
    if bg_info_override is not None:
        bg_info = dict(bg_info_override)
    timings["background_resolve_sec"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    requested_trimap_mode = str(trimap_mode or "standard")
    semantic_decision_payload = dict(semantic_decision or {})
    effective_trimap_mode = requested_trimap_mode
    solver_rgb = rgb
    proxy_input_rgb: np.ndarray | None = None
    proxy_subject_mask: np.ndarray | None = None
    proxy_subject_info: dict[str, Any] = {"enabled": False}
    same_key_inner_opaque_mask: np.ndarray | None = None
    same_key_inner_floor_mask: np.ndarray | None = None
    same_key_inner_floor_info: dict[str, Any] = {"enabled": False}
    same_key_color_restore_mask: np.ndarray | None = None
    same_key_color_restore_info: dict[str, Any] = {"enabled": False}
    same_key_opaque_edge_solver_info: dict[str, Any] = {"enabled": False}
    if requested_trimap_mode == "same_key_opaque_body_outline":
        proxy_subject_mask, proxy_subject_info = build_same_key_opaque_proxy_subject_mask(
            rgb,
            selected_bg,
            bg_threshold=float(bg_threshold),
            expand_px=0,
        )
        same_key_inner_opaque_mask, same_key_inner_floor_info = build_same_key_opaque_inner_opaque_mask(
            rgb,
            selected_bg,
            bg_threshold=float(bg_threshold),
            outer_guard_px=1.0,
        )
        same_key_color_restore_mask, same_key_color_restore_info = build_same_key_opaque_color_restore_mask(
            rgb,
            selected_bg,
            bg_threshold=float(bg_threshold),
        )
        # Same-key opaque UI needs a non-screen subject color so PyMatting sees
        # clear foreground evidence inside the measured body-outline trimap.
        # Use the known-B complement instead of a fixed color so green, blue,
        # and other saturated screen colors all get a high-contrast proxy.
        proxy_color = (255 - np.asarray(selected_bg, dtype=np.uint8)).astype(np.uint8)
        solver_rgb = rgb.copy()
        solver_rgb[proxy_subject_mask] = proxy_color
        proxy_input_rgb = solver_rgb
        proxy_subject_info = {
            **proxy_subject_info,
            "proxy_color": [int(c) for c in proxy_color],
            "proxy_color_source": "background_complement",
            "restore_uses_same_mask": True,
            "solver_trimap_mode": requested_trimap_mode,
        }
    timings["same_key_proxy_sec"] = time.perf_counter() - step_started

    processing_rgb = solver_rgb

    step_started = time.perf_counter()
    if explicit_trimap is not None:
        trimap, trimap_info = _known_b_explicit_trimap(
            explicit_trimap,
            shape=processing_rgb.shape[:2],
            semantic_decision=semantic_decision_payload,
            user_keep_mask=user_keep_mask,
            user_remove_mask=user_remove_mask,
        )
    else:
        # Body-outline tracing is geometry evidence from the original asset.
        # Proxy painting is color evidence for PyMatting; using it for tracing
        # can erase the measured ridge that accepted this route.
        trimap_source_rgb = (
            rgb
            if effective_trimap_mode == "same_key_opaque_body_outline"
            else processing_rgb
        )
        trimap, trimap_info = build_known_background_trimap(
            trimap_source_rgb,
            selected_bg,
            bg_threshold=float(bg_threshold),
            fg_threshold=float(fg_threshold),
            boundary_band_px=int(boundary_band_px),
            adapt_bg_threshold=effective_adapt_bg_threshold,
            adapt_fg_threshold=effective_adapt_fg_threshold,
            adapt_boundary_band=effective_adapt_boundary_band,
            trimap_mode=effective_trimap_mode,
            unknown_grow_px=int(unknown_grow_px),
            semantic_decision=semantic_decision_payload,
            user_keep_mask=user_keep_mask,
            user_remove_mask=user_remove_mask,
        )
    timings["trimap_sec"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    use_same_key_opaque_edge_solver = (
        requested_trimap_mode == "same_key_opaque_body_outline"
        and explicit_trimap is None
        and same_key_color_restore_mask is not None
    )
    if use_same_key_opaque_edge_solver:
        alpha, foreground_srgb, same_key_opaque_edge_solver_info = _solve_same_key_opaque_known_b_edge(
            rgb,
            background_color=selected_bg,
            support_mask=same_key_color_restore_mask,
            trimap_sure_bg=trimap.sure_bg,
            inner_opaque_mask=same_key_inner_opaque_mask,
        )
        pm_debug = {
            "used": False,
            "method": "known_b_opaque_ui_edge_solver",
            "applied": False,
            "reason": "same-key opaque UI uses known-B contour-coverage unmix solver",
            "unknown_pixels": int(trimap.unknown.sum()),
            "image_space": "sRGB",
        }
        alpha_pinhole_repair = {"used": False, "reason": "known-B opaque edge solver owns alpha"}
        consistency_repair = {"used": False, "reason": "known-B opaque edge solver produces physical foreground"}
        timings["pymatting_solve_sec"] = 0.0
        timings["foreground_unmix_repair_sec"] = time.perf_counter() - step_started
    else:
        pm = estimate_alpha_with_pymatting(
            processing_rgb,
            trimap,
            method=method,
            image_space=image_space,
            cg_maxiter=int(cg_maxiter),
            cg_rtol=float(cg_rtol),
        )
        timings["pymatting_solve_sec"] = time.perf_counter() - step_started

        step_started = time.perf_counter()
        alpha = np.clip(pm.alpha.astype(np.float32), 0.0, 1.0)
        alpha, alpha_pinhole_repair = _repair_known_b_alpha_pinholes(
            processing_rgb,
            alpha,
            background_color=selected_bg,
        )

        C_lin = ermbg_io.srgb_to_linear(processing_rgb).astype(np.float32)
        B_lin = ermbg_io.srgb_to_linear(np.asarray(selected_bg, dtype=np.uint8).reshape(1, 1, 3))[0, 0].astype(np.float32)
        foreground_linear = C_lin.copy()
        solve = alpha > 1e-3
        # Mechanism: once PyMatting has solved the narrow antialiasing band, recover
        # straight foreground against the measured known-B color so the exported RGB
        # does not retain green/blue-screen contribution at the edge.
        foreground_linear[solve] = (
            C_lin[solve] - (1.0 - alpha[solve, None]) * B_lin.reshape(1, 3)
        ) / np.maximum(alpha[solve, None], 1e-3)
        foreground_linear[~solve] = 0.0
        # Physical-consistency repair before clip. A single-known-background unmix is
        # underdetermined (3 observations, 4 unknowns), so PyMatting's alpha occasionally
        # admits an F outside [0,1] -- e.g. an opaque dark-brown outline solved as
        # alpha~0.78 forces F_g negative, which clip later turns into magenta fringe.
        # The dirty pixels are flagged by physics, not classification: any F channel
        # outside [0,1] means the (F, a) pair cannot recomposite onto any background
        # consistently. We resolve them by borrowing the nearest healthy neighbor's F
        # and re-projecting C onto the (F_neighbor, B) line to recover a self-consistent
        # alpha; healthy pixels are not touched.
        donor_foreground_linear: np.ndarray | None = None
        if proxy_subject_mask is not None:
            donor_foreground_linear = foreground_linear.copy()
            original_rgb_linear = ermbg_io.srgb_to_linear(rgb).astype(np.float32)
            donor_foreground_linear[proxy_subject_mask] = original_rgb_linear[proxy_subject_mask]
        alpha, foreground_linear, consistency_repair = _repair_known_b_unmix_consistency(
            alpha,
            foreground_linear,
            C_lin=C_lin,
            B_lin=B_lin,
            solve=solve,
            donor_foreground_linear=donor_foreground_linear,
        )
        foreground_linear = np.clip(foreground_linear, 0.0, 1.0).astype(np.float32)
        foreground_srgb = ermbg_io.linear_to_srgb_u8(foreground_linear)
        pm_debug = pm.debug
        timings["foreground_unmix_repair_sec"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    pymatting_alpha_before_inner_floor = alpha.copy()
    if same_key_inner_opaque_mask is not None:
        floor_eligible = same_key_inner_opaque_mask & ~trimap.sure_bg
        blocked_sure_bg = same_key_inner_opaque_mask & trimap.sure_bg
        lift = floor_eligible & (alpha < (254.0 / 255.0))
        alpha_lift = (1.0 - alpha[lift]).astype(np.float32) if bool(lift.any()) else np.asarray([], dtype=np.float32)
        if bool(floor_eligible.any()):
            alpha = alpha.copy()
            foreground_srgb = foreground_srgb.copy()
            # Same-key opaque UI has two separate edge classes: exterior AA and
            # internal material AA. PyMatting may solve body/stroke transitions
            # as semi-transparent because the source body is near the background.
            # The original outline support proves these guarded interior pixels
            # are material, so floor them before ShadowPatch decides what remains
            # available for exterior shadow repair.
            alpha[floor_eligible] = 1.0
            foreground_srgb[floor_eligible] = rgb[floor_eligible]
            same_key_inner_floor_mask = floor_eligible
        same_key_inner_floor_info = {
            **same_key_inner_floor_info,
            "eligible_pixels": int(floor_eligible.sum()),
            "blocked_sure_bg_pixels": int(blocked_sure_bg.sum()),
            "alpha_lift_pixels": int(lift.sum()),
            "alpha_lift_mean": float(alpha_lift.mean()) if alpha_lift.size else 0.0,
            "alpha_lift_max": float(alpha_lift.max()) if alpha_lift.size else 0.0,
            "applied_before_shadow_patch": bool(floor_eligible.any()),
        }
    subject_alpha = alpha
    subject_foreground_srgb = foreground_srgb
    shadow_patch_rgb = processing_rgb
    shadow_patch_foreground_srgb = subject_foreground_srgb
    shadow_patch_source_info = {
        "image_source": "solver_input",
        "foreground_source": "solver_foreground",
        "same_key_source_replay": False,
    }
    if proxy_subject_mask is not None:
        shadow_patch_rgb = rgb
        shadow_patch_source_info = {
            "image_source": "pre_proxy_known_b_input",
            "foreground_source": "solver_foreground",
            "same_key_source_replay": True,
        }
        if same_key_color_restore_mask is not None and not same_key_opaque_edge_solver_info.get("enabled", False):
            # Shadow evidence for same-key opaque assets must be measured against
            # the source/preprocessed pixels, not the proxy-painted solver colors.
            # Restore measured subject support before the ownership contest so
            # scalar known-B darkening can be compared to the real source image.
            shadow_patch_foreground_srgb = subject_foreground_srgb.copy()
            shadow_patch_foreground_srgb[same_key_color_restore_mask] = rgb[same_key_color_restore_mask]
            shadow_patch_source_info["foreground_source"] = "source_restored_same_key_support"
            shadow_patch_source_info["restored_foreground_pixels"] = int(same_key_color_restore_mask.sum())
    shadow_repair_domain = trimap.unknown
    if same_key_opaque_edge_solver_info.get("enabled", False) and same_key_color_restore_mask is not None:
        # In the opaque same-key solver path, measured support has already been
        # explained by subject alpha + straight foreground with low known-B
        # replay error. Let ShadowPatch consider only unknown pixels outside that
        # support so it cannot reclassify the solved outline as shadow.
        shadow_repair_domain = trimap.unknown & ~same_key_color_restore_mask
        shadow_patch_source_info["repair_domain_source"] = "unknown_outside_same_key_support"
    alpha, rgba_rgb_srgb, shadow_alpha, shadow_alpha_physical, shadow_info = _pymatting_known_b_shadow_patch(
        shadow_patch_rgb,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=shadow_patch_foreground_srgb,
        background_color=selected_bg,
        shadow_mode=shadow_mode,
        repair_domain=shadow_repair_domain,
    )
    shadow_info["input_source"] = shadow_patch_source_info
    timings["shadow_patch_sec"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    if same_key_inner_floor_mask is not None:
        alpha = alpha.copy()
        subject_foreground_srgb = subject_foreground_srgb.copy()
        rgba_rgb_srgb = rgba_rgb_srgb.copy()
        shadow_alpha = shadow_alpha.copy()
        shadow_alpha_physical = shadow_alpha_physical.copy()
        alpha[same_key_inner_floor_mask] = 1.0
        subject_foreground_srgb[same_key_inner_floor_mask] = rgb[same_key_inner_floor_mask]
        rgba_rgb_srgb[same_key_inner_floor_mask] = rgb[same_key_inner_floor_mask]
        shadow_alpha[same_key_inner_floor_mask] = 0.0
        shadow_alpha_physical[same_key_inner_floor_mask] = 0.0
    if same_key_color_restore_mask is not None:
        # Proxy colors are solver-only. Restore straight RGB over the measured
        # original support only where ShadowPatch did not assign a real shadow
        # layer. Shadow-owned pixels must keep the black shadow RGB produced by
        # the source replay; restoring blue/green source RGB there turns a
        # correct shadow alpha back into same-key colored fringe.
        if same_key_opaque_edge_solver_info.get("enabled", False):
            # The known-B opaque edge solver already unmixed straight foreground
            # for soft exterior AA. Restoring source RGB on those pixels would
            # reintroduce known-background contribution, so only opaque material
            # pixels are restored verbatim.
            restore_after_shadow = (
                same_key_color_restore_mask
                & (shadow_alpha <= (1.0 / 255.0))
                & (subject_alpha >= (254.0 / 255.0))
            )
        else:
            restore_after_shadow = same_key_color_restore_mask & (shadow_alpha <= (1.0 / 255.0))
        blocked_by_shadow = same_key_color_restore_mask & ~restore_after_shadow
        same_key_color_restore_info = {
            **same_key_color_restore_info,
            "applied_after_shadow_pixels": int(restore_after_shadow.sum()),
            "shadow_blocked_pixels": int(blocked_by_shadow.sum()),
        }
        subject_foreground_srgb = subject_foreground_srgb.copy()
        rgba_rgb_srgb = rgba_rgb_srgb.copy()
        subject_foreground_srgb[restore_after_shadow] = rgb[restore_after_shadow]
        rgba_rgb_srgb[restore_after_shadow] = rgb[restore_after_shadow]
    if proxy_subject_mask is not None:
        # Keep the exact proxy-painted domain restored as well; the broader
        # color-restore support above covers same-key edges, while this remains
        # the minimal invariant for proxy-painted subject pixels.
        proxy_restore_after_shadow = proxy_subject_mask & (shadow_alpha <= (1.0 / 255.0))
        subject_foreground_srgb = subject_foreground_srgb.copy()
        rgba_rgb_srgb = rgba_rgb_srgb.copy()
        subject_foreground_srgb[proxy_restore_after_shadow] = rgb[proxy_restore_after_shadow]
        rgba_rgb_srgb[proxy_restore_after_shadow] = rgb[proxy_restore_after_shadow]
    alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    rgba = np.dstack([rgba_rgb_srgb, alpha_u8])
    timings["compose_sec"] = time.perf_counter() - step_started
    timings["total_internal_sec"] = time.perf_counter() - total_started

    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(selected_bg),
        "despill_method": "known_background_unmix",
        "matting_model": "pymatting",
        "keyer": {},
        "shadow": shadow_info,
        "semantic_prior": {},
        "strategy": {
            "name": "pymatting_known_b",
            "bg_type": "known",
            "image_type": "graphic",
            "keyer_mode": None,
            "despill": "known_background_unmix",
            "passthrough": False,
            "notes": "PyMatting alpha on a measured or specified known background.",
            "extras": {
                "background": bg_info,
                "pymatting": pm_debug,
                "trimap": trimap_info,
                "alpha_pinhole_repair": alpha_pinhole_repair,
                "unmix_consistency_repair": consistency_repair,
                "parameters": {
                    "method": method,
                    "image_space": image_space,
                    "bg_source": effective_bg_source,
                    "requested_bg_source": str(requested_bg_source or bg_source),
                    "bg_threshold": float(bg_threshold),
                    "fg_threshold": float(fg_threshold),
                    "boundary_band_px": int(boundary_band_px),
                    "adapt_bg_threshold": effective_adapt_bg_threshold,
                    "adapt_fg_threshold": effective_adapt_fg_threshold,
                    "adapt_boundary_band": effective_adapt_boundary_band,
                    "cg_maxiter": int(cg_maxiter),
                    "cg_rtol": float(cg_rtol),
                    "trimap_mode": requested_trimap_mode,
                    "effective_trimap_mode": effective_trimap_mode,
                    "unknown_grow_px": int(unknown_grow_px),
                    "explicit_trimap": bool(explicit_trimap is not None),
                    "semantic_decision": semantic_decision_payload,
                    "user_mask_decision": trimap_info.get("user_mask_decision", {}),
                },
                "same_key_proxy_subject": proxy_subject_info,
                "same_key_inner_opaque_floor": same_key_inner_floor_info,
                "same_key_color_restore": same_key_color_restore_info,
                "same_key_opaque_edge_solver": same_key_opaque_edge_solver_info,
                "timings": timings,
            },
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route
    if preprocess_info is not None:
        report["preprocess"] = preprocess_info

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", subject_foreground_srgb)
        ermbg_io.save_rgb(out_dir / f"{stem}_rgba_rgb.png", rgba_rgb_srgb)
        ermbg_io.save_mask(out_dir / f"{stem}_pymatting_subject_alpha.png", subject_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_pymatting_subject_alpha_raw.png", pymatting_alpha_before_inner_floor)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow.png", shadow_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow_physical.png", shadow_alpha_physical)
        ermbg_io.save_mask(out_dir / f"{stem}_trimap.png", trimap_to_uint8(trimap))
        if proxy_input_rgb is not None and proxy_subject_mask is not None:
            ermbg_io.save_rgb(out_dir / f"{stem}_proxy_input.png", proxy_input_rgb)
            ermbg_io.save_mask(out_dir / f"{stem}_proxy_subject_mask.png", proxy_subject_mask.astype(np.float32))
        if same_key_inner_opaque_mask is not None:
            ermbg_io.save_mask(out_dir / f"{stem}_same_key_inner_opaque_mask.png", same_key_inner_opaque_mask.astype(np.float32))
        if same_key_inner_floor_mask is not None:
            ermbg_io.save_mask(out_dir / f"{stem}_same_key_inner_opaque_floor_mask.png", same_key_inner_floor_mask.astype(np.float32))
        if same_key_color_restore_mask is not None:
            ermbg_io.save_mask(out_dir / f"{stem}_same_key_color_restore_mask.png", same_key_color_restore_mask.astype(np.float32))
        if qa:
            qa_dir = out_dir / f"{stem}_qa"
            qa_metrics = run_qa(
                image_srgb=processing_rgb,
                rgba=rgba,
                soft_mask=alpha,
                background_color=selected_bg,
                out_dir=qa_dir,
            )
            report["qa"] = qa_metrics
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
        report_path = out_dir / f"{stem}.report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_output_manifest(
            out_dir=out_dir,
            stem=stem,
            src_path=src_path,
            report=report,
            outputs={
                "rgba": out_dir / f"{stem}_rgba.png",
                "alpha": out_dir / f"{stem}_alpha.png",
                "foreground": out_dir / f"{stem}_foreground.png",
                "rgba_rgb": out_dir / f"{stem}_rgba_rgb.png",
                "trimap": out_dir / f"{stem}_trimap.png",
                "shadow": out_dir / f"{stem}_shadow.png",
                "pymatting_subject_alpha_raw": out_dir / f"{stem}_pymatting_subject_alpha_raw.png",
                **(
                    {
                        "proxy_input": out_dir / f"{stem}_proxy_input.png",
                        "proxy_subject_mask": out_dir / f"{stem}_proxy_subject_mask.png",
                    }
                    if proxy_input_rgb is not None and proxy_subject_mask is not None
                    else {}
                ),
                **(
                    {
                        "same_key_inner_opaque_mask": out_dir / f"{stem}_same_key_inner_opaque_mask.png",
                    }
                    if same_key_inner_opaque_mask is not None
                    else {}
                ),
                **(
                    {
                        "same_key_inner_opaque_floor_mask": out_dir / f"{stem}_same_key_inner_opaque_floor_mask.png",
                    }
                    if same_key_inner_floor_mask is not None
                    else {}
                ),
                **(
                    {
                        "same_key_color_restore_mask": out_dir / f"{stem}_same_key_color_restore_mask.png",
                    }
                    if same_key_color_restore_mask is not None
                    else {}
                ),
            },
            report_path=report_path,
            requested_backend="pymatting-known-b",
        )
    elif qa:
        qa_metrics = run_qa(
            image_srgb=processing_rgb,
            rgba=rgba,
            soft_mask=alpha,
            background_color=selected_bg,
            out_dir=Path("/tmp/_ermbg_qa_discard"),
        )
        report["qa"] = qa_metrics

    debug = {
        "strategy": report["strategy"],
        "soft_mask": alpha,
        "subject_alpha": subject_alpha,
        "pymatting_subject_alpha": subject_alpha,
        "pymatting_subject_alpha_raw": pymatting_alpha_before_inner_floor,
        "pymatting_subject_foreground": subject_foreground_srgb,
        "rgba_rgb": rgba_rgb_srgb,
        "shadow_alpha": shadow_alpha,
        "shadow_alpha_physical": shadow_alpha_physical,
        "trimap_u8": trimap_to_uint8(trimap),
        "pymatting_known_b": {
            "background": bg_info,
            "trimap": trimap_info,
            "pymatting": pm_debug,
            "alpha_pinhole_repair": alpha_pinhole_repair,
            "parameters": report["strategy"]["extras"]["parameters"],
            "semantic_decision": semantic_decision_payload,
            "user_mask_decision": trimap_info.get("user_mask_decision", {}),
            "same_key_proxy_subject": proxy_subject_info,
            "same_key_inner_opaque_floor": same_key_inner_floor_info,
            "same_key_color_restore": same_key_color_restore_info,
            "same_key_opaque_edge_solver": same_key_opaque_edge_solver_info,
        },
        "shadow": shadow_info,
        "timings": timings,
    }
    if preprocess_info is not None:
        debug["input_preprocess"] = preprocess_info
    if proxy_subject_mask is not None:
        debug["proxy_subject_mask"] = proxy_subject_mask
        debug["proxy_input_srgb"] = proxy_input_rgb
    if same_key_inner_opaque_mask is not None:
        debug["same_key_inner_opaque_mask"] = same_key_inner_opaque_mask
    if same_key_inner_floor_mask is not None:
        debug["same_key_inner_opaque_floor_mask"] = same_key_inner_floor_mask
    if same_key_color_restore_mask is not None:
        debug["same_key_color_restore_mask"] = same_key_color_restore_mask
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=subject_foreground_srgb,
        strategy_name="pymatting_known_b",
        background_color=selected_bg,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


def _matte_image_comfy_rmbg(
    rgb: np.ndarray,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    bg_color: tuple[int, int, int],
    comfy_url: str,
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    from .probe.comfyui_rmbg import ComfyUIRembgBaseline

    client = ComfyUIRembgBaseline(url=comfy_url)
    rgba = client.matte(rgb)
    alpha = (rgba[..., 3].astype(np.float32) / 255.0).clip(0.0, 1.0)
    foreground = rgba[..., :3].astype(np.uint8, copy=True)
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(bg_color),
        "despill_method": "remote_rmbg",
        "matting_model": "comfy-rmbg",
        "keyer": {},
        "shadow": {"mode": "none", "source": "comfy-rmbg", "applied": False},
        "semantic_prior": {},
        "strategy": {
            "name": "comfy_rmbg",
            "bg_type": "remote",
            "image_type": "remote",
            "keyer_mode": None,
            "despill": "remote_rmbg",
            "passthrough": False,
            "notes": "Unknown/unstable background fallback through remote ComfyUI RMBG.",
            "extras": {},
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", foreground)
        if qa:
            qa_dir = out_dir / f"{stem}_qa"
            qa_metrics = run_qa(
                image_srgb=rgb,
                rgba=rgba,
                soft_mask=alpha,
                background_color=bg_color,
                out_dir=qa_dir,
            )
            report["qa"] = qa_metrics
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
        report_path = out_dir / f"{stem}.report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_output_manifest(
            out_dir=out_dir,
            stem=stem,
            src_path=src_path,
            report=report,
            outputs={
                "rgba": out_dir / f"{stem}_rgba.png",
                "alpha": out_dir / f"{stem}_alpha.png",
                "foreground": out_dir / f"{stem}_foreground.png",
            },
            report_path=report_path,
            requested_backend="comfy-rmbg",
        )
    elif qa:
        qa_metrics = run_qa(
            image_srgb=rgb,
            rgba=rgba,
            soft_mask=alpha,
            background_color=bg_color,
            out_dir=Path("/tmp/_ermbg_qa_discard"),
        )
        report["qa"] = qa_metrics

    debug = {
        "strategy": report["strategy"],
        "soft_mask": alpha,
        "shadow_alpha": np.zeros(alpha.shape, dtype=np.float32),
        "backend": "comfy-rmbg",
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=foreground,
        strategy_name="comfy_rmbg",
        background_color=bg_color,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


def _repair_known_b_unmix_consistency(
    alpha: np.ndarray,
    foreground_linear: np.ndarray,
    *,
    C_lin: np.ndarray,
    B_lin: np.ndarray,
    solve: np.ndarray,
    donor_foreground_linear: np.ndarray | None = None,
    donor_exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Repair edge pixels whose unmix produced an out-of-gamut foreground.

    Single known-B unmix (3 observations, 4 unknowns) is underdetermined; the
    matting solver's alpha occasionally yields F outside [0,1]. The dirty set is
    flagged by physics: if any F channel is < 0 or > 1, the (F, a) pair cannot
    recomposite onto an arbitrary background consistently, which is exactly the
    invariant that a transparent matte must hold. Healthy pixels satisfy the
    invariant by construction and are not modified.

    For each dirty pixel we borrow the nearest healthy neighbor's F and project
    the source color onto the line through (B, F_neighbor) in linear RGB. The
    projection coefficient is the alpha that makes the borrowed F consistent
    with C; F is then re-derived from that alpha so the recovered (F, a) lies
    inside the gamut. Pixels with no usable healthy neighbor fall back to the
    minimum per-pixel alpha required to keep F physically in gamut.
    """
    a = alpha.astype(np.float32)
    F = foreground_linear.astype(np.float32)
    h, w = a.shape
    tol = 1.0e-3
    out_low = (F < -tol).any(axis=-1)
    out_high = (F > 1.0 + tol).any(axis=-1)
    dirty = solve & (out_low | out_high)
    info: dict[str, Any] = {
        "subject_pixels": int(solve.sum()),
        "dirty_pixels": int(dirty.sum()),
        "negative_pixels": int((solve & out_low).sum()),
        "overshoot_pixels": int((solve & out_high).sum()),
        "repaired_pixels": 0,
        "donor_repaired_pixels": 0,
        "physical_repaired_pixels": 0,
        "donor_foreground_override_pixels": 0,
        "donor_exclusion_pixels": 0,
        "healthy_donor_pixels": 0,
        "alpha_lift_mean": 0.0,
        "alpha_lift_max": 0.0,
    }
    if not bool(dirty.any()):
        return a, F, info

    F_for_donor = F
    if donor_foreground_linear is not None:
        F_for_donor = np.asarray(donor_foreground_linear, dtype=np.float32)
        if F_for_donor.shape != F.shape:
            raise ValueError("donor_foreground_linear must match foreground shape")
        info["donor_foreground_override_pixels"] = int(np.any(np.abs(F_for_donor - F) > 1.0e-6, axis=-1).sum())

    # Healthy donor: a subject pixel whose donor F lies in gamut. Same-key proxy
    # painting may pass a donor map where proxy-painted pixels use the original
    # material color rather than the synthetic solver color.
    healthy_mask = solve & ~dirty & (F_for_donor >= 0.0).all(axis=-1) & (F_for_donor <= 1.0).all(axis=-1)
    excluded_donor_pixels = 0
    if donor_exclusion_mask is not None:
        excluded = np.asarray(donor_exclusion_mask, dtype=bool)
        if excluded.shape != a.shape:
            raise ValueError("donor_exclusion_mask must match alpha shape")
        excluded_donor_pixels = int((healthy_mask & excluded).sum())
        healthy_mask = healthy_mask & ~excluded
    info["donor_exclusion_pixels"] = int(excluded_donor_pixels)
    info["healthy_donor_pixels"] = int(healthy_mask.sum())
    a_new = a.copy()
    F_new = F.copy()
    accept = np.zeros((h, w), dtype=bool)

    if bool(healthy_mask.any()):
        from scipy import ndimage

        # Nearest healthy donor for each pixel; only used at dirty positions.
        _, indices = ndimage.distance_transform_edt(~healthy_mask, return_indices=True)
        F_donor = F_for_donor[indices[0], indices[1]]
        direction = F_donor - B_lin.reshape(1, 1, 3)
        denom = np.sum(direction * direction, axis=-1)
        # Donor must point away from B in linear RGB; a degenerate (F_donor ~ B)
        # is not informative.
        usable = dirty & (denom > 1.0e-5)
        if bool(usable.any()):
            projected = np.sum((C_lin - B_lin.reshape(1, 1, 3)) * direction, axis=-1) / np.maximum(denom, 1.0e-6)
            repaired_alpha = np.clip(projected, 0.0, 1.0).astype(np.float32)
            # Only accept repairs that actually raise alpha; lowering alpha here
            # would eat into known-good subject seed and is not what the dirty
            # signal proves.
            accept = usable & (repaired_alpha > a_new + 1.0e-3)
            if bool(accept.any()):
                a_new[accept] = repaired_alpha[accept]
                a_safe = np.maximum(a_new[accept, None], 1.0e-3)
                F_new[accept] = (C_lin[accept] - (1.0 - a_new[accept, None]) * B_lin.reshape(1, 3)) / a_safe

    remaining_dirty = dirty & ~accept
    physical_accept = np.zeros((h, w), dtype=bool)
    if bool(remaining_dirty.any()):
        b = B_lin.reshape(1, 1, 3)
        c = C_lin
        min_alpha = np.zeros((h, w), dtype=np.float32)
        bg_positive = b > 1.0e-6
        lower = np.where(bg_positive, 1.0 - (c / np.maximum(b, 1.0e-6)), 0.0)
        min_alpha = np.maximum(min_alpha, np.max(lower, axis=-1))
        bg_below_white = (1.0 - b) > 1.0e-6
        upper = np.where(bg_below_white, (c - b) / np.maximum(1.0 - b, 1.0e-6), 0.0)
        min_alpha = np.maximum(min_alpha, np.max(upper, axis=-1))
        min_alpha = np.clip(min_alpha, 0.0, 1.0).astype(np.float32)
        physical_alpha = np.maximum(min_alpha, a_new).astype(np.float32)
        physical_accept = remaining_dirty & (physical_alpha > a_new + 1.0e-3)
        if bool(physical_accept.any()):
            a_new[physical_accept] = physical_alpha[physical_accept]
            a_safe = np.maximum(a_new[physical_accept, None], 1.0e-3)
            F_new[physical_accept] = (
                C_lin[physical_accept] - (1.0 - a_new[physical_accept, None]) * B_lin.reshape(1, 3)
            ) / a_safe

    repaired = accept | physical_accept
    if not bool(repaired.any()):
        return a, F, info

    lift = a_new[repaired] - a[repaired]
    info.update(
        {
            "repaired_pixels": int(repaired.sum()),
            "donor_repaired_pixels": int(accept.sum()),
            "physical_repaired_pixels": int(physical_accept.sum()),
            "alpha_lift_mean": float(lift.mean()),
            "alpha_lift_max": float(lift.max()),
        }
    )
    return a_new, F_new, info



def _repair_known_b_alpha_pinholes(
    image_srgb: np.ndarray,
    alpha: np.ndarray,
    *,
    background_color: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill tiny alpha=0 pinholes inside source-visible hard subject material.

    This is not a hole closer. Real known-B cutouts are handled in the trimap
    by connected same-background components. This guard only catches isolated
    PyMatting solver pinpricks where the source pixel is far from the known
    background but the solved alpha collapses to zero, which produces B056-like
    black dots inside otherwise opaque hard UI.
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0).copy()
    if image_srgb.dtype != np.uint8:
        raise ValueError("image_srgb must be uint8")
    from .colorspace import oklab_distance, srgb_to_oklab

    bg = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    distance = oklab_distance(srgb_to_oklab(image_srgb), srgb_to_oklab(bg)[0, 0])
    candidate = (a <= (10.0 / 255.0)) & (distance >= 20.0)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    max_area = int(max(2.0, min(8.0, round(float(a.size) * 0.00002))))
    high_alpha = a >= 0.95
    kept = np.zeros(a.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        comp = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        dilated = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        ring = dilated & ~comp
        surrounded = bool(ring.any() and float(high_alpha[ring].mean()) >= 0.60)
        keep = bool(area <= max_area and surrounded)
        if keep:
            kept |= comp
        components.append(
            {
                "area": area,
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "surrounded_by_high_alpha": surrounded,
                "keep": keep,
            }
        )
    components.sort(key=lambda item: (item["keep"], item["area"]), reverse=True)
    if bool(kept.any()):
        a[kept] = 1.0
    return a, {
        "used": True,
        "candidate_pixels": int(candidate.sum()),
        "repaired_pixels": int(kept.sum()),
        "max_component_area": int(max_area),
        "components": components[:12],
        "omitted_components": max(0, len(components) - 12),
    }


def _pymatting_known_b_shadow_patch(
    image_srgb: np.ndarray,
    *,
    subject_alpha: np.ndarray,
    subject_foreground_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    shadow_mode: str,
    repair_domain: np.ndarray,
    force_shadow_layer: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover a black shadow layer only inside the trimap unknown domain."""
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    if shadow_mode == "off":
        empty = np.zeros(subject.shape, dtype=np.float32)
        info = {
            "mode": shadow_mode,
            "source": "pymatting_known_b_shadow_patch",
            "subject_source": "pymatting_known_b",
            "detected": False,
            "applied": False,
            "reason": "shadow_mode=off",
        }
        return subject, subject_foreground_srgb, empty, empty, info

    return _pymatting_known_b_unknown_domain_shadow_patch(
        image_srgb,
        subject_alpha=subject,
        subject_foreground_srgb=subject_foreground_srgb,
        background_color=background_color,
        repair_domain=repair_domain,
        force_shadow_layer=force_shadow_layer,
    )


def _solve_same_key_opaque_known_b_edge(
    image_srgb: np.ndarray,
    *,
    background_color: tuple[int, int, int],
    support_mask: np.ndarray,
    trimap_sure_bg: np.ndarray,
    inner_opaque_mask: np.ndarray | None,
    aa_width_px: float = 1.75,
    edge_bias_px: float = -0.25,
    contour_coverage_scale: int = 8,
    contour_smoothing_epsilon_px: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Known-B contour-coverage unmix for same-key opaque UI silhouettes.

    Same-key opaque buttons are not a natural-alpha matting problem: their body
    can be nearly identical to the known background, while the outline can be
    explained either as subject AA or as darkened background. This solver makes
    the route-level opaque-UI decision explicit, converts the measured subject
    support contour into a subpixel coverage alpha, then derives straight
    foreground from known-B so same-background replay remains physically checked.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")
    shape = image_srgb.shape[:2]
    support = np.asarray(support_mask, dtype=bool)
    sure_bg = np.asarray(trimap_sure_bg, dtype=bool)
    if support.shape != shape:
        raise ValueError("support_mask must match image shape")
    if sure_bg.shape != shape:
        raise ValueError("trimap_sure_bg must match image shape")
    support = support & ~sure_bg
    if inner_opaque_mask is not None:
        inner_opaque = np.asarray(inner_opaque_mask, dtype=bool)
        if inner_opaque.shape != shape:
            raise ValueError("inner_opaque_mask must match image shape")
        inner_opaque = inner_opaque & support
    else:
        inner_opaque = np.zeros(shape, dtype=bool)

    alpha = np.zeros(shape, dtype=np.float32)
    foreground = np.zeros_like(image_srgb)
    info: dict[str, Any] = {
        "enabled": True,
        "method": "known_b_opaque_ui_contour_coverage_unmix",
        "support_pixels": int(support.sum()),
        "inner_opaque_pixels": int(inner_opaque.sum()),
        "aa_width_px": float(aa_width_px),
        "edge_bias_px": float(edge_bias_px),
        "contour_coverage_scale": int(contour_coverage_scale),
        "contour_smoothing_epsilon_px": float(contour_smoothing_epsilon_px),
        "soft_pixels": 0,
        "physical_alpha_lift_pixels": 0,
        "known_b_replay_error_mean_u8": 0.0,
        "known_b_replay_error_p95_u8": 0.0,
        "known_b_replay_error_max_u8": 0.0,
        "reason": "",
    }
    if not bool(support.any()):
        info["reason"] = "empty same-key support"
        return alpha, foreground, info

    coverage_alpha, coverage_info = _same_key_support_contour_coverage_alpha(
        support,
        scale=int(contour_coverage_scale),
        epsilon_px=float(contour_smoothing_epsilon_px),
    )
    inside_distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    geometric_alpha = np.clip((inside_distance + float(edge_bias_px)) / max(float(aa_width_px), 1.0e-6), 0.0, 1.0)
    alpha_source = "contour_coverage" if coverage_info.get("enabled", False) else "geometric_distance"
    if coverage_info.get("enabled", False):
        # The support mask is measured on integer pixels and can carry stair
        # steps from the same-key trace. Treat the largest support contour as
        # the measured silhouette, rasterize it at subpixel resolution, and use
        # that coverage only inside the measured support. This smooths AA
        # without inventing exterior subject pixels from tiny background drift.
        alpha[support] = coverage_alpha[support]
    else:
        alpha[support] = geometric_alpha[support]
    alpha[inner_opaque] = 1.0

    image = image_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    min_physical_alpha = _known_b_min_alpha_for_physical_foreground(image, bg)
    before_lift = alpha.copy()
    alpha[support] = np.maximum(alpha[support], np.minimum(1.0, min_physical_alpha[support] + 1.0e-4))
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    solve = alpha > 1.0e-6
    foreground_f = np.zeros_like(image)
    foreground_f[solve] = (
        image[solve] - (1.0 - alpha[solve, None]) * bg.reshape(1, 3)
    ) / np.maximum(alpha[solve, None], 1.0e-6)
    foreground = np.clip(foreground_f, 0.0, 255.0).astype(np.uint8)

    replay = alpha[..., None] * foreground.astype(np.float32) + (1.0 - alpha[..., None]) * bg
    replay_error = np.mean(np.abs(replay - image), axis=2)
    replay_domain = cv2.dilate(support.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    replay_values = replay_error[replay_domain]
    soft = (alpha > (1.0 / 255.0)) & (alpha < (254.0 / 255.0))
    physical_lift = support & (alpha > before_lift + (1.0 / 255.0))
    info.update(
        {
            "soft_pixels": int(soft.sum()),
            "physical_alpha_lift_pixels": int(physical_lift.sum()),
            "alpha_mean_on_support": float(alpha[support].mean()),
            "alpha_min_on_support": float(alpha[support].min()),
            "alpha_source": alpha_source,
            "contour_coverage": coverage_info,
            "known_b_replay_error_mean_u8": float(replay_values.mean()) if replay_values.size else 0.0,
            "known_b_replay_error_p95_u8": float(np.percentile(replay_values, 95.0)) if replay_values.size else 0.0,
            "known_b_replay_error_max_u8": float(replay_values.max()) if replay_values.size else 0.0,
        }
    )
    return alpha, foreground, info


def _same_key_support_contour_coverage_alpha(
    support_mask: np.ndarray,
    *,
    scale: int,
    epsilon_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subpixel coverage alpha for a measured same-key UI support contour."""
    support = np.asarray(support_mask, dtype=bool)
    h, w = support.shape
    empty = np.zeros((h, w), dtype=np.float32)
    scale_i = int(np.clip(scale, 2, 16))
    epsilon = max(0.0, float(epsilon_px))
    contours, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return empty, {
            "enabled": False,
            "reason": "missing support contour",
            "scale": scale_i,
            "epsilon_px": epsilon,
        }
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return empty, {
            "enabled": False,
            "reason": "support contour has fewer than three points",
            "scale": scale_i,
            "epsilon_px": epsilon,
            "contour_points": int(len(contour)),
        }
    approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
    if len(approx) < 3:
        approx = contour

    high = np.zeros((h * scale_i, w * scale_i), dtype=np.uint8)
    high_contour = (approx.astype(np.float32) * float(scale_i) + (float(scale_i) * 0.5)).astype(np.int32)
    cv2.drawContours(high, [high_contour], -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
    coverage = cv2.resize(high, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    coverage = np.clip(coverage, 0.0, 1.0).astype(np.float32)
    soft = (coverage > (1.0 / 255.0)) & (coverage < (254.0 / 255.0))
    return coverage, {
        "enabled": True,
        "method": "supersampled_largest_contour_coverage",
        "scale": scale_i,
        "epsilon_px": epsilon,
        "contour_points": int(len(contour)),
        "approx_points": int(len(approx)),
        "soft_pixels": int(soft.sum()),
    }


def _known_b_min_alpha_for_physical_foreground(image_srgb: np.ndarray, background_srgb: np.ndarray) -> np.ndarray:
    """Minimum alpha that keeps known-B straight foreground inside sRGB gamut."""
    image = image_srgb.astype(np.float32)
    bg = background_srgb.astype(np.float32).reshape(1, 1, 3)
    min_alpha = np.zeros(image.shape[:2], dtype=np.float32)
    for channel in range(3):
        c = image[..., channel]
        b = bg[..., channel]
        lower = (b - c) / np.maximum(b, 1.0e-6)
        upper = (c - b) / np.maximum(255.0 - b, 1.0e-6)
        required = np.where(c < b, lower, np.where(c > b, upper, 0.0))
        min_alpha = np.maximum(min_alpha, required.astype(np.float32))
    return np.clip(min_alpha, 0.0, 1.0).astype(np.float32)


def _pymatting_known_b_unknown_domain_shadow_patch(
    image_srgb: np.ndarray,
    *,
    subject_alpha: np.ndarray,
    subject_foreground_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    repair_domain: np.ndarray,
    force_shadow_layer: bool = False,
    trusted_alpha_threshold: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct only the shadow layer in trimap unknown.

    Contract for the conservative PyMatting path:
    - PyMatting owns the subject matte and foreground.
    - High-alpha subject pixels are frozen.
    - ShadowPatch writes only inside trimap unknown.
    - The solved display-space shadow must improve same-background replay.
    """
    domain = np.asarray(repair_domain, dtype=bool)
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    if domain.shape != subject.shape:
        raise ValueError("repair_domain must share image HxW")

    foreground = subject_foreground_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    image = image_srgb.astype(np.float32)
    trusted = subject >= float(trusted_alpha_threshold)
    repair = domain & ~trusted

    empty = np.zeros(subject.shape, dtype=np.float32)
    before = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject,
        empty,
    )

    bg3 = np.asarray(background_color, dtype=np.float32).reshape(3)
    dominant = int(np.argmax(bg3))
    sorted_bg = np.sort(bg3)
    screen_dominant_bg = bool(float(bg3[dominant]) >= 64.0 and float(sorted_bg[-1] - sorted_bg[-2]) >= 48.0)
    source_shadow = _known_bg_display_shadow_alpha(image_srgb, background_color)
    source_shadow_replay = (1.0 - source_shadow[..., None]) * bg
    source_shadow_error = np.mean(np.abs(source_shadow_replay - image), axis=2)
    edge_subject_aa, edge_subject_alpha, edge_subject_foreground, edge_subject_info = (
        _known_bg_subject_edge_aa_compete(
            image_srgb,
            background_color,
            subject,
            subject_foreground_srgb,
            repair,
            source_shadow,
            source_shadow_error,
        )
    )
    foreground_other_max = np.max(np.delete(foreground, dominant, axis=2), axis=2)
    repair_for_shadow = domain if force_shadow_layer else repair
    foreground_screen_like = (
        screen_dominant_bg
        & (foreground[..., dominant] > foreground_other_max + max(8.0, float(bg3[dominant]) * 0.04))
        & (foreground[..., dominant] < float(bg3[dominant]) * 0.70)
    )
    # If the source pixel is a clean scalar darkening of the known background
    # and PyMatting solved it as screen-colored foreground, prefer a real black
    # shadow layer over "green/blue foreground + lower subject alpha". This is
    # the enclosed-hole counterpart of the B003 hard-shadow failure mode.
    source_shadow_candidate = (
        repair_for_shadow
        & ~edge_subject_aa
        & foreground_screen_like
        & (subject > 0.05)
        & (source_shadow > 0.035)
        & (source_shadow_error < 8.0)
    )
    relaxed_screen_residue = (
        screen_dominant_bg
        & (foreground[..., dominant] > foreground_other_max + max(4.0, float(bg3[dominant]) * 0.02))
        & (foreground[..., dominant] < float(bg3[dominant]) * 0.85)
    )
    semantic_shadow_candidate = (
        force_shadow_layer
        & repair_for_shadow
        & ~edge_subject_aa
        & screen_dominant_bg
        & (foreground[..., dominant] > foreground_other_max + max(3.0, float(bg3[dominant]) * 0.012))
        & (foreground[..., dominant] < float(bg3[dominant]) * 0.98)
        & (subject > 0.05)
        & (source_shadow > 0.015)
        & (source_shadow_error < 10.0)
    )
    source_shadow_candidate |= semantic_shadow_candidate

    direction = foreground - bg
    direction_denom = np.sum(direction * direction, axis=2)
    projected_subject = np.sum((image - bg) * direction, axis=2) / np.maximum(direction_denom, 1.0e-6)
    projected_subject = np.clip(projected_subject, 0.0, 1.0).astype(np.float32)
    usable_projection = repair & (direction_denom > 16.0)
    subject_reduce_candidate = subject.copy()
    subject_reduce_candidate[usable_projection] = np.minimum(
        subject[usable_projection],
        projected_subject[usable_projection],
    )
    after_reduce_candidate = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject_reduce_candidate,
        empty,
    )
    reduce_subject = (
        usable_projection
        & ~source_shadow_candidate
        & (projected_subject + (1.0 / 255.0) < subject)
        & (after_reduce_candidate + 0.02 < before)
    )
    corrected_subject = subject.copy()
    corrected_subject[reduce_subject] = projected_subject[reduce_subject]

    forced_shadow_subject = corrected_subject.copy()
    forced_shadow_subject[source_shadow_candidate] = 0.0
    forced_shadow_display = np.where(source_shadow_candidate, source_shadow, 0.0).astype(np.float32)
    after_forced_shadow = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        forced_shadow_subject,
        forced_shadow_display,
    )
    after_subject = np.where(reduce_subject, after_reduce_candidate, before)
    force_shadow_write = source_shadow_candidate & (after_forced_shadow + 0.02 < after_subject)
    # A user/Analyze semantic ``Solve shadow`` choice is an ownership decision,
    # not a pure reconstruction contest. Screen-colored opaque foreground and a
    # black shadow layer can replay the source equally well against the known B;
    # when the candidate region has already been put into explicit trimap
    # unknown and matches the source-shadow model, honor that decomposition.
    if force_shadow_layer:
        force_shadow_write |= semantic_shadow_candidate
    clean_source_shadow = (
        repair_for_shadow
        & ~edge_subject_aa
        & relaxed_screen_residue
        & (subject > 0.05)
        & (source_shadow > 0.035)
        & (source_shadow_error < 8.0)
    )
    if bool(clean_source_shadow.any()):
        from scipy import ndimage

        labels, _ = ndimage.label(clean_source_shadow, structure=np.ones((3, 3), dtype=bool))
        seeded_labels = np.unique(labels[force_shadow_write]) if bool(force_shadow_write.any()) else np.array([])
        seeded_labels = seeded_labels[seeded_labels > 0]
        seeded_component_write = (
            np.isin(labels, seeded_labels) & clean_source_shadow
            if seeded_labels.size
            else np.zeros_like(force_shadow_write, dtype=bool)
        )
        source_shadow_connected_write = seeded_component_write & ~force_shadow_write
    else:
        source_shadow_connected_write = np.zeros_like(force_shadow_write, dtype=bool)

    source_shadow_write = force_shadow_write | source_shadow_connected_write
    reduce_subject &= ~source_shadow_write
    corrected_subject = subject.copy()
    corrected_foreground_srgb = subject_foreground_srgb.copy()
    edge_subject_write = edge_subject_aa & ~source_shadow_write
    corrected_subject[reduce_subject] = projected_subject[reduce_subject]
    corrected_subject[edge_subject_write] = edge_subject_alpha[edge_subject_write]
    corrected_subject[source_shadow_write] = 0.0
    corrected_foreground_srgb[edge_subject_write] = edge_subject_foreground[edge_subject_write]
    foreground = corrected_foreground_srgb.astype(np.float32)
    after_subject = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        corrected_foreground_srgb,
        corrected_subject,
        empty,
    )

    base = (1.0 - corrected_subject[..., None]) * bg
    residual = image - corrected_subject[..., None] * foreground
    usable = (bg >= 8.0) & (base >= 1e-3)
    weights = np.where(usable, base * base, 0.0).astype(np.float32)
    weight_sum = np.maximum(weights.sum(axis=2), 1e-6)
    channel_shadow = 1.0 - residual / np.maximum(base, 1.0)
    solved_shadow = np.clip((channel_shadow * weights).sum(axis=2) / weight_sum, 0.0, 1.0).astype(np.float32)

    after_candidate = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        corrected_foreground_srgb,
        corrected_subject,
        solved_shadow,
    )
    write = repair & (solved_shadow > 0.0) & (after_candidate + 0.02 < after_subject)
    write |= source_shadow_write
    shadow_display = np.where(write, solved_shadow, 0.0).astype(np.float32)
    shadow_display[source_shadow_write] = source_shadow[source_shadow_write]
    alpha, rgba_rgb_srgb = _composite_subject_with_display_shadow_srgb(
        corrected_foreground_srgb,
        corrected_subject,
        shadow_display,
    )
    shadow_alpha_physical = _display_shadow_alpha_to_physical_alpha(shadow_display, background_color)

    after = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        corrected_foreground_srgb,
        corrected_subject,
        shadow_display,
    )
    shadow_written = bool(write.any())
    applied = bool(shadow_written or reduce_subject.any())
    before_values = before[domain]
    after_values = after[domain]
    alpha_reduce = subject[reduce_subject] - corrected_subject[reduce_subject]
    info = {
        "method": "unknown_domain_bidirectional_same_background_reconstruction",
        "mode": "on",
        "source": "pymatting_known_b_shadow_patch",
        "subject_source": "pymatting_known_b_raw",
        "detected": applied,
        "applied": applied,
        "reason": "" if applied else "no unknown-domain same-background improvement",
        "pixels": int((write | reduce_subject).sum()),
        "shadow_pixels": int(write.sum()),
        "subject_alpha_reduced_pixels": int(reduce_subject.sum()),
        "subject_alpha_reduce_mean": float(alpha_reduce.mean()) if alpha_reduce.size else 0.0,
        "subject_alpha_reduce_p95": float(np.percentile(alpha_reduce, 95.0)) if alpha_reduce.size else 0.0,
        "subject_alpha_reduce_max": float(alpha_reduce.max()) if alpha_reduce.size else 0.0,
        "repair_domain_pixels": int(domain.sum()),
            "trusted_alpha_threshold": float(trusted_alpha_threshold),
            "trusted_domain_pixels": int((domain & trusted).sum()),
            "force_shadow_layer": bool(force_shadow_layer),
            "mean_alpha": float(shadow_display[write].mean()) if shadow_written else 0.0,
        "p95_alpha": float(np.percentile(shadow_display[write], 95.0)) if shadow_written else 0.0,
        "max_alpha": float(shadow_display[write].max()) if shadow_written else 0.0,
        "objective_shadow": {
            "enabled": True,
            "mode": "trimap_unknown_bidirectional_exact_replay",
            "repair_domain_pixels": int(domain.sum()),
            "candidate_pixels": int((repair & (solved_shadow > 0.0)).sum()),
            "written_pixels": int(write.sum()),
            "source_shadow_candidate_pixels": int(source_shadow_candidate.sum()),
            "semantic_forced_shadow_candidate_pixels": int(semantic_shadow_candidate.sum()),
            "source_shadow_seed_written_pixels": int(force_shadow_write.sum()),
            "source_shadow_connected_written_pixels": int(source_shadow_connected_write.sum()),
            "source_shadow_written_pixels": int(source_shadow_write.sum()),
            "subject_edge_aa_candidate_pixels": int(edge_subject_info["candidate_pixels"]),
            "subject_edge_aa_written_pixels": int(edge_subject_write.sum()),
            "subject_alpha_reduced_pixels": int(reduce_subject.sum()),
            "mean_abs_error_before_u8": float(before_values.mean()) if before_values.size else 0.0,
            "mean_abs_error_after_u8": float(after_values.mean()) if after_values.size else 0.0,
            "p95_abs_error_before_u8": float(np.percentile(before_values, 95.0)) if before_values.size else 0.0,
            "p95_abs_error_after_u8": float(np.percentile(after_values, 95.0)) if after_values.size else 0.0,
            "max_abs_error_after_u8": float(after_values.max()) if after_values.size else 0.0,
            "subject_edge_aa": edge_subject_info,
        },
    }
    return (
        alpha.astype(np.float32),
        rgba_rgb_srgb,
        shadow_display.astype(np.float32),
        shadow_alpha_physical.astype(np.float32),
        info,
    )


def _composite_subject_with_display_shadow_srgb(
    foreground_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    shadow_display_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite the domain-mode shadow in the same space it was solved in.

    ``_known_bg_display_shadow_alpha_under_subject`` solves for a black PNG
    display alpha in sRGB space. For domain ShadowPatch, the validation target
    is exact same-background replay, so the exported straight RGBA needs to
    represent that same display-space layer stack instead of switching back to
    the older linear-shadow compositor.
    """
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    shadow = np.clip(shadow_display_alpha.astype(np.float32), 0.0, 1.0) * (1.0 - subject)
    alpha = np.clip(subject + shadow, 0.0, 1.0).astype(np.float32)
    premul = foreground_srgb.astype(np.float32) * subject[..., None]
    rgb = np.zeros(foreground_srgb.shape, dtype=np.float32)
    nonzero = alpha > 1e-6
    rgb[nonzero] = premul[nonzero] / alpha[nonzero, None]
    return alpha, np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _image_border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    band = max(2, min(10, int(round(min(h, w) * 0.06))))
    mask = np.zeros((h, w), dtype=bool)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def _median_abs_deviation(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    return float(np.median(np.abs(vals - med)))


def _otsu_unit_interval(values: np.ndarray, *, fallback: float) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size < 16:
        return float(fallback)
    hist, edges = np.histogram(np.clip(vals, 0.0, 1.0), bins=96, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0:
        return float(fallback)
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not bool(valid.any()):
        return float(fallback)
    sum_bg = np.cumsum(hist * centers)
    sum_total = float(sum_bg[-1])
    mean_bg = np.zeros_like(sum_bg, dtype=np.float64)
    mean_fg = np.zeros_like(sum_bg, dtype=np.float64)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = np.zeros_like(mean_bg)
    between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    threshold = float(centers[int(np.argmax(between))])
    return float(np.clip(threshold, 0.15, 0.85))


def _otsu_positive_threshold(values: np.ndarray) -> float | None:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals) & (vals >= 0.0)]
    if vals.size < 16:
        return None
    hi = float(np.percentile(vals, 99.0))
    if hi <= 0.0:
        return None
    hist, edges = np.histogram(np.clip(vals, 0.0, hi), bins=96, range=(0.0, hi))
    total = float(hist.sum())
    if total <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) * 0.5
    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not bool(valid.any()):
        return None
    sum_bg = np.cumsum(hist * centers)
    sum_total = float(sum_bg[-1])
    mean_bg = np.zeros_like(sum_bg, dtype=np.float64)
    mean_fg = np.zeros_like(sum_bg, dtype=np.float64)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    between = np.zeros_like(mean_bg)
    between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    return float(centers[int(np.argmax(between))])


def _corridorkey_shadow_patch(
    image_srgb: np.ndarray,
    *,
    subject_alpha: np.ndarray,
    subject_foreground_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    shadow_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Add a measured known-background shadow layer below CorridorKey output.

    CorridorKey is treated as the subject owner and this patch never edits its
    subject alpha. It only measures exterior scalar darkening against the known
    screen color, then flattens a black shadow layer behind the subject for
    single-PNG export. This protects against the blue-screen failure class where
    CorridorKey keeps the hard subject edge but drops the soft contact shadow.
    """
    from .shadow import (
        ShadowThresholds,
        composite_subject_with_shadow,
        estimate_shadow_alpha,
        remove_small_display_shadow_components,
        shadow_alpha_to_display_alpha,
    )

    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    if shadow_mode in {"off", "auto"}:
        empty = np.zeros(subject.shape, dtype=np.float32)
        info = {
            "mode": shadow_mode,
            "source": "corridorkey_shadow_patch",
            "detected": False,
            "applied": False,
            "reason": f"shadow_mode={shadow_mode}",
        }
        return subject, subject_foreground_srgb, empty, empty, info

    # CorridorKey already owns the hard subject. For this patch we deliberately
    # use a broader known-B support than the generic matte path: the trigger
    # gate below stays conservative, but once a real missing cast shadow is
    # confirmed we want to cover the whole soft tail. Any overlap with the
    # subject is removed during the layer composite.
    shadow_physical, info = estimate_shadow_alpha(
        image_srgb,
        subject,
        background_color,
        thresholds=ShadowThresholds(
            min_strength=0.06,
            loose_min_strength=0.001,
            max_reconstruction_error=0.085,
            loose_error_multiplier=1.8,
            max_distance_ratio=0.20,
            max_distance_px=240,
            loose_distance_multiplier=2.4,
            min_component_area_ratio=0.00025,
            min_total_area_ratio=0.0015,
            boundary_falloff_px=36.0,
            contact_distance_ratio=0.05,
            contact_distance_px=56,
            contact_outer_feather_px=14.0,
        ),
    )
    info = dict(info)
    info["mode"] = shadow_mode
    info["source"] = "corridorkey_shadow_patch"
    shadow_display = shadow_alpha_to_display_alpha(shadow_physical, background_color)
    if info.get("detected"):
        min_display_shadow_area = float(
            max(8.0, float(ShadowThresholds().min_total_area_ratio) * float(subject.size))
        )
        shadow_display, display_filter_info = remove_small_display_shadow_components(
            shadow_display,
            min_area=min_display_shadow_area,
        )
        physical_mask = shadow_physical > 0.0
        info["display_safe"] = {
            "enabled": True,
            "mean_alpha": float(shadow_display[physical_mask].mean()) if physical_mask.any() else 0.0,
            "p95_alpha": float(np.percentile(shadow_display[physical_mask], 95.0)) if physical_mask.any() else 0.0,
            "max_alpha": float(shadow_display[physical_mask].max()) if physical_mask.any() else 0.0,
            **display_filter_info,
        }
        if not bool((shadow_display > 0.0).any()):
            info["detected"] = False
            info["applied"] = False
            info["reason"] = "display shadow below minimum area"

    shadow_display, patch_gate = _corridorkey_shadow_patch_gate(subject, shadow_display, info)
    info["patch_gate"] = patch_gate
    if not patch_gate["apply"]:
        info["applied"] = False
        info["reason"] = patch_gate["reason"]
        shadow_display = np.zeros(subject.shape, dtype=np.float32)
        shadow_physical = np.zeros(subject.shape, dtype=np.float32)
    else:
        shadow_display, source_refine_info = _refine_shadow_display_from_source_pixels(
            image_srgb,
            background_color,
            shadow_display,
            subject,
        )
        info["patch_gate"]["source_reprojection"] = source_refine_info
        shadow_physical = _display_shadow_alpha_to_physical_alpha(shadow_display, background_color)

    foreground_linear = ermbg_io.srgb_to_linear(subject_foreground_srgb)
    if patch_gate["apply"]:
        info["applied"] = True
        # CorridorKey owns the subject; this patch owns only shadow repair.
        # Low CK alpha in accepted shadow support is often shadow residue, but
        # low alpha glued to a real subject core is still subject AA. Keep that
        # subject edge above the repaired shadow layer and clear only exterior
        # residue before flattening the two layers into the single PNG alpha.
        shadow_patch_alpha_max = 0.25
        shadow_display, near_subject_info = _refine_near_subject_shadow_from_source_pixels(
            shadow_display,
            subject,
            image_srgb,
            background_color,
            subject_foreground_srgb,
        )
        subject_edge_ownership = _corridorkey_subject_edge_ownership(subject, shadow_display > 0.0)
        shadow_layer_execution = (
            (shadow_display > 0.0)
            & ((subject <= shadow_patch_alpha_max) | subject_edge_ownership)
        )
        blocked_by_subject = (shadow_display > 0.0) & ~shadow_layer_execution
        shadow_display = np.where(shadow_layer_execution, shadow_display, 0.0).astype(np.float32)
        shadow_physical = np.where(
            shadow_layer_execution,
            _display_shadow_alpha_to_physical_alpha(shadow_display, background_color),
            0.0,
        ).astype(np.float32)
        residue_execution = shadow_layer_execution & (subject <= shadow_patch_alpha_max) & ~subject_edge_ownership
        subject_for_shadow = np.where(residue_execution, 0.0, subject).astype(np.float32)
        info["patch_gate"]["shadow_patch_subject_alpha_max"] = float(shadow_patch_alpha_max)
        info["patch_gate"]["near_subject_reprojection"] = near_subject_info
        info["patch_gate"]["shadow_patch_pixels_blocked_by_subject_alpha"] = int(blocked_by_subject.sum())
        info["patch_gate"]["corridorkey_subject_edge_pixels_preserved"] = int(subject_edge_ownership.sum())
        info["patch_gate"]["corridorkey_shadow_residue_pixels_removed"] = int(
            (residue_execution & (subject > 0.0)).sum()
        )
        alpha, rgba_rgb_linear = composite_subject_with_shadow(
            foreground_linear,
            subject_for_shadow,
            shadow_display,
            # This is a literal layer stack: shadow below, CorridorKey subject
            # above. Near-subject source reprojection closes contact gaps before
            # the stack, so extra occluder blur would only muddy hard UI edges.
            subject_occlusion_blur_sigma=0.0,
        )
        rgba_rgb_srgb = ermbg_io.linear_to_srgb_u8(rgba_rgb_linear)
    else:
        alpha = subject
        rgba_rgb_srgb = subject_foreground_srgb
        shadow_display = np.zeros(subject.shape, dtype=np.float32)
        shadow_physical = np.zeros(subject.shape, dtype=np.float32)

    return (
        alpha.astype(np.float32),
        rgba_rgb_srgb,
        shadow_display.astype(np.float32),
        shadow_physical.astype(np.float32),
        info,
    )


def _refine_near_subject_shadow_from_source_pixels(
    shadow_display: np.ndarray,
    subject_alpha: np.ndarray,
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    subject_foreground_srgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproject all near-subject shadow pixels from the source image.

    Near a hard UI subject, source pixels may be pure shadow, subject AA over
    shadow, or a one-pixel support gap between the two. Treat them uniformly by
    solving C ~= a*F + (1-a)*(1-shadow)*B wherever there is nearby subject and
    nearby shadow evidence. CorridorKey still owns the subject layer; this only
    repairs the shadow layer underneath it.
    """
    shadow = np.clip(shadow_display.astype(np.float32), 0.0, 1.0)
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    shadow_mask = shadow > 0.0
    subject_core = subject >= 0.50
    if not bool(shadow_mask.any()) or not bool(subject_core.any()):
        return shadow, {
            "enabled": True,
            "repair_pixels": 0,
            "source_reproject_pixels": 0,
            "source_added_pixels": 0,
            "existing_shadow_pixels": 0,
            "reason": "missing shadow or subject core",
        }

    dist_to_core = cv2.distanceTransform((~subject_core).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_shadow = cv2.distanceTransform((~shadow_mask).astype(np.uint8), cv2.DIST_L2, 3)
    near_core = dist_to_core <= 3.0
    near_shadow = dist_to_shadow <= 3.0
    source_shadow = _known_bg_display_shadow_alpha_under_subject(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject,
    )
    source_visible = source_shadow >= (1.5 / 255.0)
    repair_domain = near_core & near_shadow & (subject < 0.50) & (shadow_mask | source_visible)
    if not bool(repair_domain.any()):
        return shadow, {
            "enabled": True,
            "repair_pixels": 0,
            "source_reproject_pixels": 0,
            "source_added_pixels": 0,
            "existing_shadow_pixels": 0,
            "reason": "",
        }
    out = shadow.copy()
    before_error = _near_subject_shadow_reprojection_error(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject,
        out,
        repair_domain,
    )
    source_reproject = repair_domain & (source_shadow > 0.0)
    out[source_reproject] = source_shadow[source_reproject]
    contact_gap = near_core & near_shadow & (subject < 0.08) & ~shadow_mask & ~source_visible
    if bool(contact_gap.any()):
        # UI shadow renderers often leave a subpixel strip of unchanged screen
        # color between a hard antialiased subject and the first measurable
        # shadow texel. After background removal that strip becomes a white
        # preview gap. Bridge only pixels caught between an accepted shadow and
        # a subject core; this cannot create shadows in B036-style no-shadow
        # cases because it needs an existing shadow component nearby.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        borrowed_shadow = cv2.dilate(shadow, kernel)
        falloff = np.clip(1.0 - (dist_to_shadow / 3.0), 0.0, 1.0).astype(np.float32)
        bridged = np.minimum(borrowed_shadow * falloff * 0.90, 0.45).astype(np.float32)
        bridge_pixels = contact_gap & (bridged >= (1.0 / 255.0))
        bridge_fraction_of_repair = float(bridge_pixels.sum()) / max(float(repair_domain.sum()), 1.0)
        bridge_fraction_of_shadow = float(bridge_pixels.sum()) / max(float(shadow_mask.sum()), 1.0)
        # Contact-gap bridging is a seam repair. If it becomes a large fraction
        # of the near-subject repair domain or accepted shadow, it is no longer
        # filling isolated subpixel holes; it is expanding the cast shadow along
        # the whole UI outline, which creates unstable grey rims on heavy-shadow
        # buttons. These empirical gates key on the repair's own scale rather
        # than any sample id or color.
        bridge_rejected_as_expansion = bool(
            bridge_fraction_of_repair > 0.12 or bridge_fraction_of_shadow > 0.05
        )
        if bridge_rejected_as_expansion:
            rejected_bridge_pixels = bridge_pixels
            bridge_pixels = np.zeros(shadow.shape, dtype=bool)
        else:
            rejected_bridge_pixels = np.zeros(shadow.shape, dtype=bool)
        out[bridge_pixels] = np.maximum(out[bridge_pixels], bridged[bridge_pixels])
    else:
        bridge_pixels = np.zeros(shadow.shape, dtype=bool)
        rejected_bridge_pixels = np.zeros(shadow.shape, dtype=bool)
        bridge_fraction_of_repair = 0.0
        bridge_fraction_of_shadow = 0.0
        bridge_rejected_as_expansion = False
    after_error = _near_subject_shadow_reprojection_error(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject,
        out,
        repair_domain,
    )
    improvement = before_error - after_error
    if improvement < max(0.12, before_error * 0.20):
        # Source reprojection is a repair, not a stylistic preference. B037-like
        # contact gaps show a clear reconstruction win; B007-like yellow/white
        # UI edges can satisfy the same geometry while barely improving the
        # source fit, and changing them creates visible material damage.
        return shadow, {
            "enabled": True,
            "repair_pixels": int(repair_domain.sum()),
            "source_reproject_pixels": 0,
            "source_added_pixels": 0,
            "contact_gap_bridge_pixels": 0,
            "existing_shadow_pixels": int((repair_domain & shadow_mask).sum()),
            "mean_abs_error_before_u8": before_error,
            "mean_abs_error_after_u8": after_error,
            "rejected_source_reproject_pixels": int(source_reproject.sum()),
            "rejected_contact_gap_bridge_pixels": int((bridge_pixels | rejected_bridge_pixels).sum()),
            "contact_gap_bridge_fraction_of_repair": float(bridge_fraction_of_repair),
            "contact_gap_bridge_fraction_of_shadow": float(bridge_fraction_of_shadow),
            "contact_gap_bridge_rejected_as_expansion": bool(bridge_rejected_as_expansion),
            "reason": "source reprojection did not significantly improve fit",
        }
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "enabled": True,
        "repair_pixels": int(repair_domain.sum()),
        "source_reproject_pixels": int(source_reproject.sum()),
        "source_added_pixels": int((source_reproject & ~shadow_mask).sum()),
        "contact_gap_bridge_pixels": int(bridge_pixels.sum()),
        "rejected_contact_gap_bridge_pixels": int(rejected_bridge_pixels.sum()),
        "contact_gap_bridge_fraction_of_repair": float(bridge_fraction_of_repair),
        "contact_gap_bridge_fraction_of_shadow": float(bridge_fraction_of_shadow),
        "contact_gap_bridge_rejected_as_expansion": bool(bridge_rejected_as_expansion),
        "existing_shadow_pixels": int((repair_domain & shadow_mask).sum()),
        "mean_abs_error_before_u8": before_error,
        "mean_abs_error_after_u8": after_error,
        "reason": "",
    }


def _known_bg_display_shadow_alpha_under_subject(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    subject_foreground_srgb: np.ndarray,
    subject_alpha: np.ndarray,
) -> np.ndarray:
    """Solve display shadow alpha below an existing CorridorKey subject layer."""
    image = image_srgb.astype(np.float32)
    foreground = subject_foreground_srgb.astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    base = (1.0 - subject[..., None]) * bg
    residual = image - subject[..., None] * foreground
    usable = base >= 4.0
    weights = np.where(usable, base * base, 0.0).astype(np.float32)
    weight_sum = np.maximum(weights.sum(axis=-1), 1e-6)
    channel_alpha = 1.0 - residual / np.maximum(base, 1.0)
    alpha = (channel_alpha * weights).sum(axis=-1) / weight_sum
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _known_bg_subject_edge_aa_compete(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    subject_alpha: np.ndarray,
    subject_foreground_srgb: np.ndarray,
    repair_domain: np.ndarray,
    source_shadow_alpha: np.ndarray,
    source_shadow_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Prefer subject AA over shadow when the local replay evidence is tied.

    Near a hard UI edge, the same pixel can be read as either subject coverage
    or scalar known-background shadow. A wrong shadow decision exports black
    color through the PNG and shows up as bright/dark speckles on previews.
    This arbitration is deliberately local: it only runs next to an opaque
    subject core, uses the nearest core color as foreground evidence, and
    compares direct source reconstruction against a pure-shadow explanation.
    """
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    domain = np.asarray(repair_domain, dtype=bool)
    empty_mask = np.zeros(subject.shape, dtype=bool)
    empty_alpha = np.zeros(subject.shape, dtype=np.float32)
    foreground_out = subject_foreground_srgb.copy()
    subject_core = subject >= 0.985
    if not bool(subject_core.any()):
        return empty_mask, empty_alpha, foreground_out, {
            "enabled": True,
            "candidate_pixels": 0,
            "written_pixels": 0,
            "reason": "missing opaque subject core",
        }

    from scipy import ndimage

    dist_to_core = cv2.distanceTransform((~subject_core).astype(np.uint8), cv2.DIST_L2, 3)
    candidate = domain & (subject > 0.012) & (subject < 0.985) & (dist_to_core <= 3.0)
    if not bool(candidate.any()):
        return empty_mask, empty_alpha, foreground_out, {
            "enabled": True,
            "candidate_pixels": 0,
            "written_pixels": 0,
            "reason": "",
        }

    _, indices = ndimage.distance_transform_edt(~subject_core, return_indices=True)
    nearest_foreground = image_srgb[indices[0], indices[1]].astype(np.float32)
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    image = image_srgb.astype(np.float32)
    source_shadow = np.clip(source_shadow_alpha.astype(np.float32), 0.0, 1.0)

    plain_bg = np.broadcast_to(bg, image.shape).astype(np.float32)
    shadowed_bg = (1.0 - source_shadow[..., None]) * bg
    plain_alpha, plain_error = _solve_subject_coverage_error(image, nearest_foreground, plain_bg)
    shadowed_alpha, shadowed_error = _solve_subject_coverage_error(image, nearest_foreground, shadowed_bg)
    use_shadowed_bg = shadowed_error < plain_error
    solved_alpha = np.where(use_shadowed_bg, shadowed_alpha, plain_alpha).astype(np.float32)
    subject_error = np.where(use_shadowed_bg, shadowed_error, plain_error).astype(np.float32)
    shadow_error = np.asarray(source_shadow_error, dtype=np.float32)

    current_error = _subject_shadow_reprojection_error_map(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject,
        np.zeros(subject.shape, dtype=np.float32),
    )
    # One 8-bit code value is the replay tolerance for "no clear winner". In
    # that tie zone, subject AA is the safer owner because a false shadow writes
    # black foreground into the exported RGBA, while a false subject-AA pixel
    # remains locally color-consistent with the nearby opaque edge.
    tie_tolerance_u8 = 1.0
    prefer = (
        candidate
        & (solved_alpha >= 0.02)
        & (solved_alpha <= 0.985)
        & (subject_error <= shadow_error + tie_tolerance_u8)
        & (subject_error <= current_error + tie_tolerance_u8)
    )
    alpha_out = np.zeros(subject.shape, dtype=np.float32)
    alpha_out[prefer] = solved_alpha[prefer]
    foreground_out[prefer] = np.clip(nearest_foreground[prefer], 0.0, 255.0).astype(np.uint8)
    return prefer, alpha_out, foreground_out, {
        "enabled": True,
        "candidate_pixels": int(candidate.sum()),
        "written_pixels": int(prefer.sum()),
        "used_shadowed_background_pixels": int((prefer & use_shadowed_bg).sum()),
        "mean_subject_error_u8": float(subject_error[prefer].mean()) if bool(prefer.any()) else 0.0,
        "mean_shadow_error_u8": float(shadow_error[prefer].mean()) if bool(prefer.any()) else 0.0,
        "mean_alpha": float(solved_alpha[prefer].mean()) if bool(prefer.any()) else 0.0,
        "tie_tolerance_u8": float(tie_tolerance_u8),
        "reason": "",
    }


def _solve_subject_coverage_error(
    image_srgb: np.ndarray,
    foreground_srgb: np.ndarray,
    background_srgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    direction = foreground_srgb.astype(np.float32) - background_srgb.astype(np.float32)
    denom = np.sum(direction * direction, axis=2)
    alpha = np.sum((image_srgb.astype(np.float32) - background_srgb.astype(np.float32)) * direction, axis=2)
    alpha = np.clip(alpha / np.maximum(denom, 1.0e-6), 0.0, 1.0).astype(np.float32)
    predicted = alpha[..., None] * foreground_srgb.astype(np.float32) + (1.0 - alpha[..., None]) * background_srgb.astype(np.float32)
    error = np.mean(np.abs(predicted - image_srgb.astype(np.float32)), axis=2).astype(np.float32)
    return alpha, error


def _subject_shadow_reprojection_error_map(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    subject_foreground_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    shadow_display: np.ndarray,
) -> np.ndarray:
    """Per-pixel sRGB replay error for subject + display shadow over known B."""
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    shadow = np.clip(shadow_display.astype(np.float32), 0.0, 1.0)
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    foreground = subject_foreground_srgb.astype(np.float32)
    predicted = subject[..., None] * foreground + (1.0 - subject[..., None]) * (1.0 - shadow[..., None]) * bg
    return np.abs(predicted - image_srgb.astype(np.float32)).mean(axis=2).astype(np.float32)


def _near_subject_shadow_reprojection_error(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    subject_foreground_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    shadow_display: np.ndarray,
    support: np.ndarray,
) -> float:
    pixels = int(support.sum())
    if pixels <= 0:
        return 0.0
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    shadow = np.clip(shadow_display.astype(np.float32), 0.0, 1.0)
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    foreground = subject_foreground_srgb.astype(np.float32)
    predicted = subject[..., None] * foreground + (1.0 - subject[..., None]) * (1.0 - shadow[..., None]) * bg
    base = (1.0 - subject[..., None]) * bg
    weights = np.where(base >= 4.0, base * base, 0.0).astype(np.float32)
    weight_sum = np.maximum(weights.sum(axis=-1), 1e-6)
    err = np.abs(predicted - image_srgb.astype(np.float32))
    weighted = (err * weights).sum(axis=-1) / weight_sum
    return float(weighted[support].mean())


def _corridorkey_subject_edge_ownership(subject_alpha: np.ndarray, shadow_execution: np.ndarray) -> np.ndarray:
    """Keep CorridorKey-owned subject AA where the shadow patch touches it.

    CorridorKey is still the subject owner. ShadowPatch may erase low CK alpha
    in exterior shadow support because that residue is often a weak, banded
    shadow estimate. The exception is low alpha glued to a real subject core:
    that is the subject antialiasing the user already trusts from CorridorKey,
    so the shadow layer should sit underneath instead of replacing it.
    """
    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    subject_core = subject >= 0.50
    if not bool(subject_core.any()):
        return np.zeros(subject.shape, dtype=bool)
    dist_to_core = cv2.distanceTransform((~subject_core).astype(np.uint8), cv2.DIST_L2, 3)
    return (
        np.asarray(shadow_execution, dtype=bool)
        & (subject >= 0.015)
        & (subject < 0.50)
        & (dist_to_core <= 2.5)
    )


def _known_bg_display_shadow_alpha(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
) -> np.ndarray:
    """Invert the PNG display composite that users visually inspect.

    Shadow extraction measures the physical known-background darkening in
    linear RGB. The exported artifact, however, is a black RGBA layer that will
    normally be composited by viewers in sRGB/display space. For hard shadows on
    a known screen, the most direct target is therefore:

        original_srgb ~= (1 - display_alpha) * background_srgb

    This inverse is only used inside already accepted shadow support, so subject
    colors with a coincidental background-channel ratio cannot create shadows on
    their own.
    """
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    image = image_srgb.astype(np.float32)
    usable = bg >= 8.0
    if not bool(usable.any()):
        return np.zeros(image_srgb.shape[:2], dtype=np.float32)

    weights = np.where(usable, bg * bg, 0.0).astype(np.float32)
    weight_sum = np.maximum(float(weights.sum()), 1e-6)
    channel_alpha = 1.0 - image / np.maximum(bg, 1.0)
    alpha = (channel_alpha * weights).sum(axis=-1) / weight_sum
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _display_shadow_alpha_to_physical_alpha(
    display_alpha: np.ndarray,
    background_color: tuple[int, int, int],
) -> np.ndarray:
    """Map display-space black alpha back to linear scalar-darkening strength."""
    alpha = np.clip(display_alpha.astype(np.float32), 0.0, 1.0)
    bg_u8 = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg_linear = ermbg_io.srgb_to_linear(bg_u8)[0, 0].astype(np.float32)
    denom = max(float(np.dot(bg_linear, bg_linear)), 1e-6)
    bg_srgb = bg_u8.astype(np.float32) / 255.0
    shadowed_srgb = (1.0 - alpha[..., None]) * bg_srgb
    shadowed_linear = ermbg_io.srgb_to_linear(np.clip(shadowed_srgb, 0.0, 1.0)).astype(np.float32)
    scale = np.tensordot(shadowed_linear, bg_linear, axes=([-1], [0])) / denom
    return np.clip(1.0 - scale, 0.0, 1.0).astype(np.float32)


def _shadow_reprojection_error(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    display_alpha: np.ndarray,
    support: np.ndarray,
) -> float:
    pixels = int(support.sum())
    if pixels <= 0:
        return 0.0
    bg = np.asarray(background_color, dtype=np.float32).reshape(1, 1, 3)
    predicted = (1.0 - np.clip(display_alpha.astype(np.float32), 0.0, 1.0)[..., None]) * bg
    usable = bg >= 8.0
    weights = np.where(usable, bg * bg, 0.0).astype(np.float32)
    weight_sum = np.maximum(float(weights.sum()), 1e-6)
    err = np.abs(predicted - image_srgb.astype(np.float32))
    weighted = (err * weights).sum(axis=-1) / weight_sum
    return float(weighted[support].mean())


def _refine_shadow_display_from_source_pixels(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int],
    shadow_display: np.ndarray,
    subject_alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use source pixels as the shadow reconstruction target.

    The accepted support says "this is exterior shadow"; the source image then
    tells us what alpha the exported black layer needs so that placing the PNG
    back over the same screen color reproduces the original pixels. A small
    normalized blur suppresses 8-bit/compression noise inside the support while
    preserving the measured support boundary by never expanding outside the
    accepted mask.
    """
    initial_support = (shadow_display > 0.0) & (subject_alpha <= 0.25)
    if not bool(initial_support.any()):
        return np.zeros(shadow_display.shape, dtype=np.float32), {
            "enabled": True,
            "pixels": 0,
            "reason": "empty accepted shadow support",
        }

    source_alpha = _known_bg_display_shadow_alpha(image_srgb, background_color)
    support_values = source_alpha[initial_support]
    grow_floor = max(6.0 / 255.0, float(np.percentile(support_values, 50.0)) * 0.30)
    rim_floor = 1.5 / 255.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    low_subject_bg = subject_alpha <= 0.25
    # The detector/gate can lose the antialiasing rim of a hard shadow. The
    # source pixels are the authority here: allow a two-pixel expansion for
    # clearly darkened source pixels, then recover the connected low-alpha rim
    # from the source itself so the exported shadow edge does not become a
    # binary stair-step.
    strong_grown_support = (
        cv2.dilate(initial_support.astype(np.uint8), kernel, iterations=2).astype(bool)
        & low_subject_bg
        & (source_alpha >= grow_floor)
    )
    rim_candidates = low_subject_bg & (source_alpha >= rim_floor)
    rim_seed = initial_support | strong_grown_support
    rim_labels_count, rim_labels, _, _ = cv2.connectedComponentsWithStats(
        rim_candidates.astype(np.uint8),
        connectivity=8,
    )
    aa_rim_support = np.zeros(initial_support.shape, dtype=bool)
    for rim_label in range(1, rim_labels_count):
        comp = rim_labels == rim_label
        if bool((comp & rim_seed).any()):
            aa_rim_support |= comp
    support = initial_support | strong_grown_support | aa_rim_support
    out = np.zeros(shadow_display.shape, dtype=np.float32)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8),
        connectivity=8,
    )
    components: list[dict[str, Any]] = []
    for label in range(1, labels_count):
        comp = labels == label
        if not bool(comp.any()):
            continue
        comp_source = np.where(comp, source_alpha, 0.0).astype(np.float32)
        comp_weight = comp.astype(np.float32)
        # Generated/PNG source shadows are usually smooth but can contain
        # one-code-value channel noise. Normalized convolution inside the
        # accepted component makes the patch follow the source gradient/plateau
        # without growing into subject edges or background.
        blurred_num = cv2.GaussianBlur(comp_source * comp_weight, (0, 0), sigmaX=0.65, sigmaY=0.65)
        blurred_den = cv2.GaussianBlur(comp_weight, (0, 0), sigmaX=0.65, sigmaY=0.65)
        smoothed = np.where(blurred_den > 1e-5, blurred_num / np.maximum(blurred_den, 1e-5), source_alpha)
        refined = np.clip(0.78 * source_alpha + 0.22 * smoothed, 0.0, 1.0)
        out[comp] = refined[comp]
        values = source_alpha[comp]
        components.append(
            {
                "area": int(stats[label, cv2.CC_STAT_AREA]),
                "bbox_xyxy": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "source_display_mean": float(values.mean()) if values.size else 0.0,
                "source_display_p50": float(np.percentile(values, 50.0)) if values.size else 0.0,
                "source_display_p90": float(np.percentile(values, 90.0)) if values.size else 0.0,
            }
        )

    before_error = _shadow_reprojection_error(image_srgb, background_color, shadow_display, support)
    after_error = _shadow_reprojection_error(image_srgb, background_color, out, support)
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "enabled": True,
        "pixels": int(support.sum()),
        "initial_pixels": int(initial_support.sum()),
        "source_grown_pixels": int((support & ~initial_support).sum()),
        "source_grow_floor": float(grow_floor),
        "source_rim_floor": float(rim_floor),
        "source_aa_rim_pixels": int((aa_rim_support & ~initial_support & ~strong_grown_support).sum()),
        "mean_abs_error_before_u8": before_error,
        "mean_abs_error_after_u8": after_error,
        "components": components[:8],
        "omitted_components": max(0, len(components) - 8),
        "reason": "",
    }


def _corridorkey_shadow_patch_gate(
    subject_alpha: np.ndarray,
    shadow_display: np.ndarray,
    shadow_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decide whether a CorridorKey shadow candidate is missing enough to patch.

    The detector may find a real shadow even when CorridorKey already preserved
    it as partial alpha. In that case adding another layer would double-darken
    the result. This gate is intentionally stricter than the candidate extractor:
    strong measured shadow evidence plus low existing CorridorKey alpha over
    the visible shadow support is required before the patch is applied.
    """
    shadow_patch_subject_alpha_max = 0.25
    empty = np.zeros(subject_alpha.shape, dtype=np.float32)
    if not shadow_info.get("detected"):
        return empty, {"apply": False, "reason": shadow_info.get("reason") or "no high-confidence shadow candidate"}

    subject = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    display = np.clip(shadow_display.astype(np.float32), 0.0, 1.0)
    visible = (display >= (6.0 / 255.0)) & (subject <= 0.75)
    pixels = int(visible.sum())
    if pixels <= 0:
        return empty, {"apply": False, "reason": "no visible exterior shadow support"}

    img_area = float(subject_alpha.size)
    area_ratio = float(pixels / max(1.0, img_area))
    display_values = display[visible].astype(np.float32)
    subject_values = subject[visible].astype(np.float32)
    display_mean = float(display_values.mean())
    display_p75 = float(np.percentile(display_values, 75.0))
    display_p95 = float(np.percentile(display_values, 95.0))
    subject_mean = float(subject_values.mean())
    subject_p75 = float(np.percentile(subject_values, 75.0))
    ratio_mean = subject_mean / max(display_mean, 1e-6)
    ratio_p75 = subject_p75 / max(display_p75, 1e-6)

    high_confidence = (
        area_ratio >= 0.0015
        and display_p95 >= 0.055
        and int(shadow_info.get("accepted_components", 0)) >= 1
    )
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        visible.astype(np.uint8),
        connectivity=8,
    )
    kept = np.zeros(subject.shape, dtype=bool)
    adjusted_display = display.copy()
    component_details: list[dict[str, Any]] = []
    min_component_pixels = max(8, int(round(img_area * 0.0015)))
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp_area_ratio = float(area / max(1.0, img_area))
        bbox_width_ratio = float(width / max(1, subject.shape[1]))
        bbox_height_ratio = float(height / max(1, subject.shape[0]))
        comp = labels == label
        comp_display = display[comp]
        comp_subject = subject[comp]
        comp_display_mean = float(comp_display.mean())
        comp_display_p75 = float(np.percentile(comp_display, 75.0))
        comp_display_p95 = float(np.percentile(comp_display, 95.0))
        comp_subject_mean = float(comp_subject.mean())
        comp_subject_p75 = float(np.percentile(comp_subject, 75.0))
        comp_shadow_domain = comp & (subject <= shadow_patch_subject_alpha_max)
        comp_shadow_domain_area = int(comp_shadow_domain.sum())
        if comp_shadow_domain_area > 0:
            domain_display = display[comp_shadow_domain]
            domain_subject = subject[comp_shadow_domain]
            domain_display_mean = float(domain_display.mean())
            domain_display_p75 = float(np.percentile(domain_display, 75.0))
            domain_display_p95 = float(np.percentile(domain_display, 95.0))
            domain_subject_mean = float(domain_subject.mean())
            domain_subject_p75 = float(np.percentile(domain_subject, 75.0))
        else:
            domain_display_mean = 0.0
            domain_display_p75 = 0.0
            domain_display_p95 = 0.0
            domain_subject_mean = 0.0
            domain_subject_p75 = 0.0
        # ShadowPatch is a contact/cast-shadow repair under the CorridorKey
        # subject layer. Broad vertical washes usually come from AI-generated
        # non-flat screen backgrounds or subject-material residue, not an
        # exterior ground shadow; applying a black layer there dirties the whole
        # asset. Keep components with a shadow-like horizontal footprint.
        shadow_like_shape = width >= max(6, int(round(height * 1.15)))
        broad_low_contrast_wash = (
            comp_area_ratio >= 0.18
            and bbox_width_ratio >= 0.65
            and bbox_height_ratio >= 0.50
            and comp_display_p95 <= 0.12
        )
        comp_high_confidence = (
            area >= min_component_pixels
            and comp_display_p95 >= 0.055
            and shadow_like_shape
            and not broad_low_contrast_wash
        )
        # Mixed candidates can contain both a true cast shadow and subject
        # outline antialiasing. Gate the missing-shadow claim on the low-alpha
        # shadow domain only; medium/high CK alpha blocks patch execution later
        # and should not suppress a genuinely under-reconstructed shadow tail.
        clearly_missing = (
            comp_shadow_domain_area >= min_component_pixels
            and domain_subject_mean <= max(0.035, domain_display_mean * 0.45)
            and domain_subject_p75 <= max(0.055, domain_display_p75 * 0.55)
        )
        # A hard UI shadow can be physically strong and smooth on the known
        # screen, while CorridorKey keeps only a weak, visibly banded alpha
        # residue. Treating any alpha as "preserved" leaves the poor CK estimate
        # in the export. This broader branch still requires a high measured
        # display shadow and less than roughly half of that opacity in CK alpha,
        # so already-preserved shadows are not double-darkened.
        under_reconstructed_shadow = (
            comp_shadow_domain_area >= min_component_pixels
            and domain_display_p75 >= 0.14
            and domain_subject_mean <= max(0.055, domain_display_mean * 0.55)
            and domain_subject_p75 <= max(0.085, domain_display_p75 * 0.55)
        )
        comp_missing = bool(clearly_missing or under_reconstructed_shadow)
        comp_apply = bool(comp_high_confidence and comp_missing)
        if comp_apply:
            if under_reconstructed_shadow:
                # Hard-shadow components have a measurable opacity plateau on
                # the known background. CK often turns that plateau into a
                # banded low-alpha estimate. A soft-shadow support mask is too
                # broad for this case, so use a strong core as the uniform
                # platform and keep only a one-pixel measured rim for AA. This
                # prevents the patch from exporting a fuzzy tail when the source
                # shadow is perceptually hard.
                core_floor = max(0.09, domain_display_p75 * 0.58)
                rim_floor = max(0.025, domain_display_p75 * 0.14)
                hard_shadow_core = comp_shadow_domain & (display >= core_floor)
                if hard_shadow_core.any():
                    core_u8 = hard_shadow_core.astype(np.uint8)
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    hard_shadow_core = cv2.morphologyEx(core_u8, cv2.MORPH_CLOSE, kernel).astype(bool) & comp_shadow_domain
                    rim = (
                        (cv2.dilate(hard_shadow_core.astype(np.uint8), kernel, iterations=1).astype(bool))
                        & comp_shadow_domain
                        & ~hard_shadow_core
                        & (display >= rim_floor)
                    )
                    kept |= hard_shadow_core | rim
                else:
                    rim = np.zeros(subject.shape, dtype=bool)
                    kept |= comp_shadow_domain
                adjusted_display[hard_shadow_core] = np.maximum(
                    adjusted_display[hard_shadow_core],
                    domain_display_p75,
                )
            else:
                hard_shadow_core = np.zeros(subject.shape, dtype=bool)
                rim = np.zeros(subject.shape, dtype=bool)
                kept |= comp
        component_details.append(
            {
                "area": area,
                "bbox_xyxy": [
                    left,
                    top,
                    left + width,
                    top + height,
                ],
                "area_ratio": comp_area_ratio,
                "bbox_width_ratio": bbox_width_ratio,
                "bbox_height_ratio": bbox_height_ratio,
                "shadow_like_shape": bool(shadow_like_shape),
                "broad_low_contrast_wash": bool(broad_low_contrast_wash),
                "display_mean": comp_display_mean,
                "display_p75": comp_display_p75,
                "display_p95": comp_display_p95,
                "corridorkey_alpha_mean": comp_subject_mean,
                "corridorkey_alpha_p75": comp_subject_p75,
                "shadow_domain_area": int(comp_shadow_domain_area),
                "shadow_domain_display_mean": domain_display_mean,
                "shadow_domain_display_p75": domain_display_p75,
                "shadow_domain_corridorkey_alpha_mean": domain_subject_mean,
                "shadow_domain_corridorkey_alpha_p75": domain_subject_p75,
                "high_confidence": bool(comp_high_confidence),
                "clearly_missing": bool(clearly_missing),
                "under_reconstructed_shadow": bool(under_reconstructed_shadow),
                "levelled_display_p75": float(domain_display_p75) if comp_apply and under_reconstructed_shadow else 0.0,
                "hard_shadow_core_pixels": int(hard_shadow_core.sum()) if comp_apply and under_reconstructed_shadow else 0,
                "hard_shadow_rim_pixels": int(rim.sum()) if comp_apply and under_reconstructed_shadow else 0,
                "missing_in_corridorkey": bool(comp_missing),
                "apply": comp_apply,
            }
        )
    missing_in_corridorkey = bool(kept.any())
    component_details.sort(key=lambda item: (item["apply"], item["high_confidence"], item["area"]), reverse=True)
    omitted_component_details = max(0, len(component_details) - 16)
    reported_component_details = component_details[:16]
    rejected_missing_shape_components = sum(
        1
        for item in component_details
        if item["missing_in_corridorkey"] and (not item["shadow_like_shape"] or item["broad_low_contrast_wash"])
    )
    apply = bool(high_confidence and missing_in_corridorkey)
    if not high_confidence:
        reason = "shadow candidate below high-confidence gate"
    elif not missing_in_corridorkey:
        if rejected_missing_shape_components:
            reason = "shadow candidates rejected as broad background wash or vertical subject residue"
        else:
            reason = "CorridorKey already preserved shadow alpha"
    else:
        reason = ""
    filtered = np.where(kept, adjusted_display, 0.0).astype(np.float32) if apply else empty
    return filtered, {
        "apply": apply,
        "reason": reason,
        "visible_pixels": pixels,
        "visible_area_ratio": area_ratio,
        "kept_visible_pixels": int(kept.sum()),
        "kept_components": int(sum(1 for item in component_details if item["apply"])),
        "candidate_components": int(len(component_details)),
        "rejected_missing_shape_components": int(rejected_missing_shape_components),
        "omitted_component_details": int(omitted_component_details),
        "component_min_pixels": int(min_component_pixels),
        "component_details": reported_component_details,
        "display_mean": display_mean,
        "display_p75": display_p75,
        "display_p95": display_p95,
        "corridorkey_alpha_mean": subject_mean,
        "corridorkey_alpha_p75": subject_p75,
        "corridorkey_to_shadow_mean_ratio": float(ratio_mean),
        "corridorkey_to_shadow_p75_ratio": float(ratio_p75),
        "high_confidence": bool(high_confidence),
        "missing_in_corridorkey": bool(missing_in_corridorkey),
    }


def _matte_image_known_bg_hard_ui_shadow(
    rgb: np.ndarray,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    analysis: Any,
    solver_result: Any,
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    selected_bg_color = solver_result.background_color
    alpha = solver_result.alpha
    rgba_rgb_srgb = solver_result.rgba_rgb_srgb
    rgba = np.dstack(
        [
            rgba_rgb_srgb,
            (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    shadow_rgba = np.dstack(
        [
            np.zeros(rgb.shape, dtype=np.uint8),
            (np.clip(solver_result.shadow_alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    subject_rgba = np.dstack(
        [
            solver_result.foreground_srgb,
            (np.clip(solver_result.subject_alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    hint_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    shadow_info = {
        "mode": "on",
        "source": "known_bg_hard_ui_shadow",
        "detected": bool((solver_result.shadow_alpha > 0.0).any()),
        "applied": True,
        "reason": "",
        **solver_result.debug.get("shadow_model", {}),
    }
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(selected_bg_color),
        "despill_method": "known_bg_hard_ui_shadow",
        "matting_model": "KnownBgHardUiShadow",
        "corridorkey_analysis": analysis.to_dict(),
        "known_bg_hard_ui_shadow": solver_result.debug,
        "keyer": {
            "used": True,
            "source": "known_bg_hard_ui_shadow",
            "hint": {"source": "analytic_known_background", "mean": 1.0},
        },
        "shadow": shadow_info,
        "semantic_prior": {},
        "strategy": {
            "name": "known_bg_hard_ui_shadow",
            "bg_type": f"saturated_{analysis.screen_mode}" if analysis.screen_mode in {"green", "blue"} else "known_screen",
            "image_type": "hard_ui_shadow",
            "keyer_mode": "known_bg_analytic",
            "despill": "known_bg_inverse_solve",
            "passthrough": False,
            "notes": "Analytic known-background hard UI solver separated subject ownership from scalar cast shadow.",
            "extras": solver_result.debug,
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", rgba_rgb_srgb)
        ermbg_io.save_rgba(out_dir / f"{stem}_corridorkey_subject_rgba.png", subject_rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_corridorkey_subject_alpha.png", solver_result.subject_alpha)
        ermbg_io.save_rgba(out_dir / f"{stem}_shadow_layer.png", shadow_rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow.png", solver_result.shadow_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow_physical.png", solver_result.shadow_alpha_physical)
        ermbg_io.save_mask(out_dir / f"{stem}_corridorkey_hint.png", hint_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_corridorkey_raw_alpha.png", solver_result.subject_alpha)
        if qa:
            qa_dir = out_dir / f"{stem}_qa"
            qa_metrics = run_qa(
                image_srgb=rgb,
                rgba=rgba,
                soft_mask=alpha,
                background_color=selected_bg_color,
                out_dir=qa_dir,
            )
            report["qa"] = qa_metrics
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
        report_path = out_dir / f"{stem}.report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_output_manifest(
            out_dir=out_dir,
            stem=stem,
            src_path=src_path,
            report=report,
            outputs={
                "rgba": out_dir / f"{stem}_rgba.png",
                "alpha": out_dir / f"{stem}_alpha.png",
                "foreground": out_dir / f"{stem}_foreground.png",
                "corridorkey_subject_rgba": out_dir / f"{stem}_corridorkey_subject_rgba.png",
                "corridorkey_hint": out_dir / f"{stem}_corridorkey_hint.png",
                "corridorkey_raw_alpha": out_dir / f"{stem}_corridorkey_raw_alpha.png",
                "shadow": out_dir / f"{stem}_shadow.png",
            },
            report_path=report_path,
            requested_backend="corridorkey",
        )
    elif qa:
        qa_metrics = run_qa(
            image_srgb=rgb,
            rgba=rgba,
            soft_mask=alpha,
            background_color=selected_bg_color,
            out_dir=Path("/tmp/_ermbg_qa_discard"),
        )
        report["qa"] = qa_metrics

    debug = {
        **solver_result.debug,
        "strategy": report["strategy"],
        "corridorkey_analysis": analysis.to_dict(),
        "known_bg_hard_ui_shadow": solver_result.debug,
        "soft_mask": alpha,
        "subject_alpha": solver_result.subject_alpha,
        "corridorkey_subject_rgba": subject_rgba,
        "corridorkey_hint": hint_alpha,
        "corridorkey_raw_alpha": solver_result.subject_alpha,
        "shadow_alpha": solver_result.shadow_alpha,
        "shadow_alpha_physical": solver_result.shadow_alpha_physical,
        "shadow_layer_rgba": shadow_rgba,
        "shadow": shadow_info,
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=rgba_rgb_srgb,
        strategy_name="known_bg_hard_ui_shadow",
        background_color=selected_bg_color,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


__all__ = ["matte_image", "classify_image", "MatteResponse", "ImageLike", "MaskLike"]
