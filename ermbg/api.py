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
    corridorkey_protection_bg_max: float = 12.0,
    corridorkey_protection_fg_min: float = 28.0,
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
    after_error = _near_subject_shadow_reprojection_error(
        image_srgb,
        background_color,
        subject_foreground_srgb,
        subject,
        out,
        repair_domain,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "enabled": True,
        "repair_pixels": int(repair_domain.sum()),
        "source_reproject_pixels": int(source_reproject.sum()),
        "source_added_pixels": int((source_reproject & ~shadow_mask).sum()),
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
    key_color_protection = np.zeros(rgb.shape[:2], dtype=np.float32)
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
        ermbg_io.save_mask(out_dir / f"{stem}_key_color_protection.png", key_color_protection)
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
        **solver_result.debug,
        "strategy": report["strategy"],
        "corridorkey_analysis": analysis.to_dict(),
        "known_bg_hard_ui_shadow": solver_result.debug,
        "soft_mask": alpha,
        "subject_alpha": solver_result.subject_alpha,
        "corridorkey_subject_rgba": subject_rgba,
        "corridorkey_hint": hint_alpha,
        "corridorkey_raw_alpha": solver_result.subject_alpha,
        "key_color_protection": key_color_protection,
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
    color_protection_bg_max: float = 12.0,
    color_protection_fg_min: float = 28.0,
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
