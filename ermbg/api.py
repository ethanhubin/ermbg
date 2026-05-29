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

import cv2
import numpy as np
from PIL import Image

from . import io as ermbg_io
from .comfy import DEFAULT_COMFY_URL
from .matting import matte as _matte_internal
from .matting import solid_graphic_to_matting_result
from .qa import run_qa
from .router import Strategy, classify_strategy
from .segmenter import build_segmenter
from .solid_graphic import analyze_solid_bg_graphic

ImageLike = Union[str, Path, np.ndarray, Image.Image]
MaskLike = Union[str, Path, np.ndarray, Image.Image]

_SEGMENTER_CACHE: dict[tuple[str, str, int, str], Any] = {}


def _get_segmenter(backend: str, model_id: str, input_size: int, comfy_url: str):
    """Return a process-local segmenter for repeated API/Web calls.

    BiRefNet model construction dominates first-call latency and is wasteful in
    long-lived server processes. Cache only by explicit public knobs so tests
    and callers can still request independent backends/models/sizes.
    """
    key = (backend, model_id, int(input_size), comfy_url)
    seg = _SEGMENTER_CACHE.get(key)
    if seg is None:
        kwargs: dict[str, Any] = {
            "model_id": model_id,
            "input_size": input_size,
        }
        if backend == "comfy-rmbg":
            kwargs["url"] = comfy_url
        seg = build_segmenter(backend=backend, **kwargs)
        _SEGMENTER_CACHE[key] = seg
    return seg


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


def _auto_backend_for_image(
    image_srgb: np.ndarray,
    *,
    screen_mode: str,
    preset: str,
    fallback_background_color: tuple[int, int, int],
) -> tuple[str, dict[str, Any]]:
    """Route production auto mode without entering the ERMBG full pipeline.

    Current policy is intentionally simple: green/blue known-screen assets use
    CorridorKey plus local ShadowPatch; everything else falls directly to RMBG.
    ERMBG AutoMatte remains available as an explicit diagnostic backend, but is
    not part of automatic routing while CorridorKey/ShadowPatch is the game UI
    mainline.
    """
    from .corridorkey import corridorkey_analyze_asset

    analysis = corridorkey_analyze_asset(
        image_srgb,
        screen_mode=screen_mode,  # type: ignore[arg-type]
        preset=preset,  # type: ignore[arg-type]
        fallback_background_color=fallback_background_color,
    )
    if analysis.screen_mode in {"green", "blue"}:
        return "comfy-corridorkey", {
            "selected_backend": "comfy-corridorkey",
            "reason": f"{analysis.screen_mode}_screen",
            "corridorkey_analysis": analysis.to_dict(),
        }
    return "comfy-rmbg", {
        "selected_backend": "comfy-rmbg",
        "reason": "unknown_background",
        "corridorkey_analysis": analysis.to_dict(),
    }


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
    shadow_mode: str = "on",
    vlm_prior: bool = False,
    vlm_provider: str = "openai",
    vlm_model: str = "gpt-4o-mini",
    vlm_prior_mode: str = "shadow",
    comfy_url: str = DEFAULT_COMFY_URL,
    solid_graphic_prepass: bool = True,
    corridorkey_gamma_space: str = "sRGB",
    corridorkey_despill_strength: float = 1.0,
    corridorkey_refiner_strength: float = 1.0,
    corridorkey_auto_despeckle: str = "On",
    corridorkey_despeckle_size: int = 400,
    corridorkey_auto_mask: bool = False,
    corridorkey_color_protection: bool = True,
    corridorkey_protection_bg_max: float = 8.0,
    corridorkey_protection_fg_min: float = 16.0,
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    corridorkey_hint_mask: MaskLike | None = None,
) -> MatteResponse:
    """Matte one image end-to-end.

    Args:
        image: path, numpy array (HxWx3 or HxWx4 uint8 sRGB), or PIL Image.
        output_dir: if set, write rgba/alpha/foreground/trimap PNGs and
            ``report.json`` here. If ``qa=True``, also writes ``qa/on_*.png``.
        qa: run multi-background composite QA. Adds ~6 image saves and the
            full halo/recomp/binarization metric block to the report.
        matting_model: HF id of BiRefNet variant.
        backend: ``auto`` | ``birefnet`` | ``grabcut`` | ``comfy-rmbg`` |
            ``comfy-ermbg`` | ``comfy-corridorkey``. ``auto`` routes green/blue
            known-screen assets to CorridorKey+ShadowPatch and unknown
            backgrounds to RMBG fallback; it does not enter ERMBG AutoMatte.
            ``comfy-ermbg`` remains an explicit diagnostic backend.
            ``comfy-corridorkey`` runs CorridorKey remotely with a known-screen
            chroma-key alpha hint.
        input_size: square matting-net input size for BiRefNet backends.
        bg_color: composite color used when an RGBA source is dirty enough
            that the router falls through to re-matte (since the matting net
            needs RGB on a known constant bg). Default is the green-screen
            target so the first stage's outputs route well.
        despill, use_keyer: optional manual overrides; default ``None`` lets
            the router decide.
        subject_mask: optional H×W ownership mask from an independent segmenter.
            When provided, ERMBG may repair keyer-supported low-alpha holes
            inside this mask without raising the subject's external soft edge.
        shadow_mode: ``on`` preserves full shadow recovery, ``off`` skips it
            for faster previews, ``auto`` currently preserves ``on`` behavior.
        vlm_prior: call the optional VLM semantic-prior pass before
            despill. The model only classifies CV candidate regions; local code
            still computes alpha, foreground colors, and shadow strength.
        vlm_provider: ``openai`` or ``comfy-qwen``.
        vlm_prior_mode: ``shadow`` (default), ``material``, or ``all``.
        solid_graphic_prepass: when true, high-confidence solid-background
            graphics use the analytic ownership-first path before constructing
            a local matting segmenter.
        corridorkey_screen_mode: ``auto``, ``green``, or ``blue`` for the
            CorridorKey path. ``auto`` estimates the key screen from border
            evidence before submitting the remote workflow.
        corridorkey_preset: ``auto``, ``detail_safe``, ``spill_safe``, or
            ``manual``. Non-manual presets may override the individual
            CorridorKey knobs with analysis-driven recommendations.
        corridorkey_hint_mask: optional H×W coarse foreground hint for
            CorridorKey, for example from SAM3 or manual Web edits. It is a
            hint only; CorridorKey still computes the detail alpha.
    """
    rgb, alpha, src_path = _to_rgb_and_alpha(image)
    subject_support = _to_mask(subject_mask, rgb.shape[:2], "subject_mask")
    corridorkey_hint_alpha = _to_mask(corridorkey_hint_mask, rgb.shape[:2], "corridorkey_hint_mask")

    # If source has α but the router decides to re-matte, the matting net
    # needs RGB on a known bg, not the raw (possibly premul or leaky) RGB.
    strat_preview = classify_strategy(rgb, source_alpha=alpha)
    auto_route: dict[str, Any] | None = None
    if backend == "auto" and not strat_preview.passthrough:
        backend, auto_route = _auto_backend_for_image(
            rgb,
            screen_mode=corridorkey_screen_mode,
            preset=corridorkey_preset,
            fallback_background_color=bg_color,
        )

    remote_full_backends = {"comfy-ermbg", "comfy-corridorkey"}
    if alpha is not None and (backend in remote_full_backends or not strat_preview.passthrough):
        bg_arr = np.broadcast_to(np.asarray(bg_color, dtype=np.uint8), rgb.shape[:2] + (3,))
        a4 = alpha[..., None]
        rgb_lin = ermbg_io.srgb_to_linear(rgb)
        bg_lin = ermbg_io.srgb_to_linear(bg_arr)
        rgb = ermbg_io.linear_to_srgb_u8(a4 * rgb_lin + (1.0 - a4) * bg_lin)

    if backend in remote_full_backends:
        if vlm_prior:
            raise ValueError(f"backend={backend!r} does not support local vlm_prior")
        if subject_support is not None:
            raise ValueError(f"backend={backend!r} does not support local subject_mask")
    if backend == "comfy-corridorkey":
        return _matte_image_comfy_corridorkey(
            rgb,
            src_path=src_path,
            output_dir=output_dir,
            qa=qa,
            bg_color=bg_color,
            shadow_mode=shadow_mode,
            comfy_url=comfy_url,
            gamma_space=corridorkey_gamma_space,
            despill_strength=corridorkey_despill_strength,
            refiner_strength=corridorkey_refiner_strength,
            auto_despeckle=corridorkey_auto_despeckle,
            despeckle_size=corridorkey_despeckle_size,
            auto_mask=corridorkey_auto_mask,
            apply_color_protection=corridorkey_color_protection,
            color_protection_bg_max=corridorkey_protection_bg_max,
            color_protection_fg_min=corridorkey_protection_fg_min,
            screen_mode=corridorkey_screen_mode,
            preset=corridorkey_preset,
            hint_alpha=corridorkey_hint_alpha,
            auto_route=auto_route,
        )

    if backend == "comfy-ermbg":
        return _matte_image_comfy_ermbg(
            rgb,
            src_path=src_path,
            output_dir=output_dir,
            qa=qa,
            matting_model=matting_model,
            bg_color=bg_color,
            despill=despill,
            use_keyer=use_keyer,
            shadow_mode=shadow_mode,
            comfy_url=comfy_url,
        )

    semantic_prior = None
    soft_preview = None
    result = None
    can_try_solid_graphic = (
        solid_graphic_prepass
        and backend != "comfy-rmbg"
        and subject_support is None
        and not vlm_prior
        and despill is None
        and use_keyer is None
        and shadow_mode != "off"
        and strat_preview.bg_type in {"saturated", "white", "black", "grey"}
    )
    if can_try_solid_graphic:
        solid = analyze_solid_bg_graphic(rgb)
        if solid.accepted:
            result = solid_graphic_to_matting_result(solid, strat_preview, shadow_mode=shadow_mode)

    if result is None:
        seg = _get_segmenter(
            backend=backend,
            model_id=matting_model,
            input_size=input_size,
            comfy_url=comfy_url,
        )

    if result is None and vlm_prior:
        from .diagnose import BackgroundDiagnoser
        from .shadow import estimate_shadow_alpha
        from .vlm_semantic import (
            ComfyQwenVLMSemanticPriorClient,
            OpenAIVLMSemanticPriorClient,
            build_vlm_semantic_request,
            extract_shadow_candidate_regions,
            extract_subject_material_candidate_regions,
        )

        soft_preview = seg.segment(rgb)
        diag = BackgroundDiagnoser().diagnose(rgb, soft_preview)
        B = np.asarray(diag.background_color, dtype=np.uint8)
        shadow_alpha, _ = estimate_shadow_alpha(rgb, soft_preview, B)
        mode = vlm_prior_mode.strip().lower()
        if mode not in {"shadow", "material", "all"}:
            raise ValueError(f"Unknown vlm_prior_mode: {vlm_prior_mode!r}")
        regions = []
        if mode in {"shadow", "all"}:
            regions.extend(
                extract_shadow_candidate_regions(
                    rgb,
                    soft_preview,
                    B,
                    shadow_alpha=shadow_alpha,
                )
            )
        if mode in {"material", "all"}:
            regions.extend(
                extract_subject_material_candidate_regions(
                    rgb,
                    soft_preview,
                    B,
                    shadow_alpha=shadow_alpha,
                )
            )
        if regions:
            request = build_vlm_semantic_request(
                image_srgb=rgb,
                subject_alpha=soft_preview,
                background_color=tuple(int(c) for c in B),
                regions=regions,
                shadow_alpha=shadow_alpha,
            )
            if vlm_provider == "openai":
                client = OpenAIVLMSemanticPriorClient(
                    model=vlm_model,
                    env_path=Path(".env"),
                )
            elif vlm_provider == "comfy-qwen":
                client = ComfyQwenVLMSemanticPriorClient(
                    url=comfy_url,
                    model=vlm_model if vlm_model != "gpt-4o-mini" else "Qwen3-VL-4B-Instruct-FP8",
                )
            else:
                raise ValueError(f"Unknown vlm_provider: {vlm_provider!r}")
            semantic_prior = client.classify_request(request, regions, rgb.shape[:2])

    if result is None:
        result = _matte_internal(
            rgb,
            source_alpha=alpha,
            segmenter=seg,
            despill=despill,
            use_keyer=use_keyer,
            subject_support=subject_support,
            semantic_prior=semantic_prior,
            soft_mask=soft_preview,
            shadow_mode=shadow_mode,
            solid_graphic_prepass=False,
        )

    # Build report.
    report: dict[str, Any] = {
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis is not None else None,
        "background_color": list(result.background_color),
        "despill_method": result.debug.get("despill_method"),
        "matting_model": matting_model,
        "keyer": result.debug.get("keyer", {}),
        "shadow": result.debug.get("shadow", {}),
        "semantic_prior": result.debug.get("semantic_prior", {}),
        "strategy": result.debug.get("strategy", {}),
    }
    if auto_route is not None:
        report["auto_route"] = auto_route

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", result.rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", result.alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow.png", result.debug["shadow_alpha"])
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

    debug = dict(result.debug)
    if auto_route is not None:
        debug["auto_route"] = auto_route

    return MatteResponse(
        rgba=result.rgba,
        alpha=result.alpha,
        foreground_srgb=result.foreground_srgb,
        strategy_name=result.debug.get("strategy", {}).get("name", "unknown"),
        background_color=result.background_color,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


def _matte_image_comfy_ermbg(
    rgb: np.ndarray,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    matting_model: str,
    bg_color: tuple[int, int, int],
    despill: str | None,
    use_keyer: bool | None,
    shadow_mode: str,
    comfy_url: str,
) -> MatteResponse:
    from .probe.comfyui_ermbg_matte import ComfyUIErmbgMatteClient

    client = ComfyUIErmbgMatteClient(url=comfy_url)
    remote = client.matte(
        rgb,
        matting_model=matting_model,
        bg_color=bg_color,
        despill=despill,
        use_keyer=use_keyer,
        shadow_mode=shadow_mode,
    )
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(bg_color),
        "despill_method": "remote",
        "matting_model": matting_model,
        "keyer": {},
        "shadow": {"mode": shadow_mode, "source": "remote_comfy_ermbg"},
        "semantic_prior": {},
        "strategy": {
            "name": "comfy_ermbg",
            "bg_type": "remote",
            "image_type": "remote",
            "keyer_mode": None,
            "despill": "remote",
            "passthrough": False,
            "notes": "Full ERMBG pipeline executed by remote ComfyUI.",
            "extras": remote.debug,
        },
    }

    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_path).stem if src_path else "matte"
        ermbg_io.save_rgba(out_dir / f"{stem}_rgba.png", remote.rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_alpha.png", remote.alpha)
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", remote.foreground_srgb)
        if qa:
            qa_dir = out_dir / f"{stem}_qa"
            qa_metrics = run_qa(
                image_srgb=rgb,
                rgba=remote.rgba,
                soft_mask=remote.alpha,
                background_color=bg_color,
                out_dir=qa_dir,
            )
            report["qa"] = qa_metrics
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2))
        (out_dir / f"{stem}.report.json").write_text(json.dumps(report, indent=2))
    elif qa:
        qa_metrics = run_qa(
            image_srgb=rgb,
            rgba=remote.rgba,
            soft_mask=remote.alpha,
            background_color=bg_color,
            out_dir=Path("/tmp/_ermbg_qa_discard"),
        )
        report["qa"] = qa_metrics

    debug = {
        **remote.debug,
        "strategy": report["strategy"],
        "soft_mask": remote.alpha,
        "shadow_alpha": np.zeros(remote.alpha.shape, dtype=np.float32),
    }
    return MatteResponse(
        rgba=remote.rgba,
        alpha=remote.alpha,
        foreground_srgb=remote.foreground_srgb,
        strategy_name="comfy_ermbg",
        background_color=bg_color,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


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
    if shadow_mode == "off":
        empty = np.zeros(subject.shape, dtype=np.float32)
        info = {
            "mode": shadow_mode,
            "source": "corridorkey_shadow_patch",
            "detected": False,
            "applied": False,
            "reason": "shadow_mode=off",
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
        shadow_physical = np.where(shadow_display > 0.0, shadow_physical, 0.0)

    foreground_linear = ermbg_io.srgb_to_linear(subject_foreground_srgb)
    if patch_gate["apply"]:
        info["applied"] = True
        alpha, rgba_rgb_linear = composite_subject_with_shadow(
            foreground_linear,
            subject,
            shadow_display,
            # This is a literal layer stack: shadow below, CorridorKey subject
            # above. Extra occluder blur would create the contact gap we are
            # specifically trying to avoid for hard-edge/soft-shadow samples.
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
        # Mixed candidates can contain both a true cast shadow and dark subject
        # material/hole residue. Gating component-by-component keeps the real
        # missing shadow while preventing preserved subject-owned soft alpha from
        # raising the global statistics and suppressing the patch.
        comp_missing = (
            comp_subject_mean <= max(0.035, comp_display_mean * 0.45)
            and comp_subject_p75 <= max(0.055, comp_display_p75 * 0.55)
        )
        comp_apply = bool(comp_high_confidence and comp_missing)
        if comp_apply:
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
                "high_confidence": bool(comp_high_confidence),
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
    filtered = np.where(kept, display, 0.0).astype(np.float32) if apply else empty
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


def _matte_image_comfy_corridorkey(
    rgb: np.ndarray,
    *,
    src_path: str | None,
    output_dir: str | Path | None,
    qa: bool,
    bg_color: tuple[int, int, int],
    shadow_mode: str,
    comfy_url: str,
    gamma_space: str = "sRGB",
    despill_strength: float = 1.0,
    refiner_strength: float = 1.0,
    auto_despeckle: str = "On",
    despeckle_size: int = 400,
    auto_mask: bool = False,
    apply_color_protection: bool = True,
    color_protection_bg_max: float = 8.0,
    color_protection_fg_min: float = 16.0,
    screen_mode: str = "auto",
    preset: str = "auto",
    hint_alpha: np.ndarray | None = None,
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    from .corridorkey import corridorkey_analyze_asset
    from .probe.comfyui_corridorkey import ComfyUICorridorKeyClient

    analysis = corridorkey_analyze_asset(
        rgb,
        screen_mode=screen_mode,  # type: ignore[arg-type]
        preset=preset,  # type: ignore[arg-type]
        fallback_background_color=bg_color,
    )
    selected_bg_color = analysis.background_color
    if preset != "manual":
        settings = analysis.recommended_settings
        gamma_space = settings.gamma_space
        despill_strength = settings.despill_strength
        refiner_strength = settings.refiner_strength
        auto_despeckle = settings.auto_despeckle
        despeckle_size = settings.despeckle_size
        apply_color_protection = settings.color_protection
        color_protection_bg_max = settings.protection_bg_max
        color_protection_fg_min = settings.protection_fg_min

    client = ComfyUICorridorKeyClient(url=comfy_url)
    hint_source = None
    if hint_alpha is not None:
        hint_source = "provided_corridorkey_hint_mask"
    elif not auto_mask:
        # All-white hint is an intentional diagnostic/control mode for
        # CorridorKey: it removes our known-B mask generation from the equation
        # while keeping the remote model and post-processing path identical.
        hint_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        hint_source = "all_white_alpha_hint"
    remote = client.matte(
        rgb,
        background_color=selected_bg_color,
        hint_alpha=hint_alpha,
        hint_source=hint_source,
        gamma_space=gamma_space,
        screen_color=analysis.screen_mode if analysis.screen_mode in {"green", "blue"} else "auto",
        despill_strength=despill_strength,
        refiner_strength=refiner_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        apply_color_protection=apply_color_protection,
        color_protection_bg_max=color_protection_bg_max,
        color_protection_fg_min=color_protection_fg_min,
    )
    alpha, rgba_rgb_srgb, shadow_alpha, shadow_alpha_physical, shadow_info = _corridorkey_shadow_patch(
        rgb,
        subject_alpha=remote.alpha,
        subject_foreground_srgb=remote.foreground_srgb,
        background_color=selected_bg_color,
        shadow_mode=shadow_mode,
    )
    rgba = np.dstack(
        [
            rgba_rgb_srgb,
            (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    shadow_rgba = np.dstack(
        [
            np.zeros(rgb.shape, dtype=np.uint8),
            (np.clip(shadow_alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    bg_type = f"saturated_{analysis.screen_mode}" if analysis.screen_mode in {"green", "blue"} else "unknown_screen"
    image_type = f"ai_{analysis.screen_mode}_asset" if analysis.screen_mode in {"green", "blue"} else "ai_screen_asset"
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(selected_bg_color),
        "despill_method": "remote_corridorkey",
        "matting_model": "CorridorKey",
        "corridorkey_analysis": analysis.to_dict(),
        "keyer": {
            "used": True,
            "source": "known_bg_chromatic_key_alpha_hint" if auto_mask else "all_white_alpha_hint",
            "hint": remote.debug.get("hint", {}),
        },
        "shadow": shadow_info,
        "semantic_prior": {},
        "strategy": {
            "name": "comfy_corridorkey",
            "bg_type": bg_type,
            "image_type": image_type,
            "keyer_mode": "corridorkey",
            "despill": "remote_corridorkey",
            "passthrough": False,
            "notes": "CorridorKey executed by remote ComfyUI using ERMBG screen/color analysis.",
            "extras": remote.debug,
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
        ermbg_io.save_rgb(out_dir / f"{stem}_foreground.png", remote.foreground_srgb)
        ermbg_io.save_rgba(out_dir / f"{stem}_corridorkey_subject_rgba.png", remote.rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_corridorkey_subject_alpha.png", remote.alpha)
        ermbg_io.save_rgba(out_dir / f"{stem}_shadow_layer.png", shadow_rgba)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow.png", shadow_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_shadow_physical.png", shadow_alpha_physical)
        ermbg_io.save_mask(out_dir / f"{stem}_corridorkey_hint.png", remote.hint_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_corridorkey_raw_alpha.png", remote.raw_alpha)
        ermbg_io.save_mask(out_dir / f"{stem}_key_color_protection.png", remote.color_protection_alpha)
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
            (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2))
        (out_dir / f"{stem}.report.json").write_text(json.dumps(report, indent=2))
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
        **remote.debug,
        "strategy": report["strategy"],
        "corridorkey_analysis": analysis.to_dict(),
        "soft_mask": alpha,
        "subject_alpha": remote.alpha,
        "corridorkey_subject_rgba": remote.rgba,
        "corridorkey_hint": remote.hint_alpha,
        "corridorkey_raw_alpha": remote.raw_alpha,
        "key_color_protection": remote.color_protection_alpha,
        "shadow_alpha": shadow_alpha,
        "shadow_alpha_physical": shadow_alpha_physical,
        "shadow_layer_rgba": shadow_rgba,
        "shadow": shadow_info,
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=remote.foreground_srgb,
        strategy_name="comfy_corridorkey",
        background_color=selected_bg_color,
        report=report,
        output_dir=out_dir,
        debug=debug,
    )


__all__ = ["matte_image", "classify_image", "MatteResponse", "ImageLike", "MaskLike"]
