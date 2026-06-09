"""HTTP server for the ERMBG direct worker validation path.

This is intentionally separate from the production Web/API/Comfy route. It is
for validating a remote service that bypasses ComfyUI's prompt queue while
reusing ERMBG's current route decisions and maintained matting executors.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PIL import Image

from .direct_worker import (
    DirectCorridorKeyClientFactory,
    DirectWorkerResult,
    direct_matte_from_decision,
)
from .corridorkey_runner import LocalCorridorKeyClient
from .router import RouteDecision, classify_route
from .runtime_capabilities import get_ermbg_version

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except Exception as exc:  # pragma: no cover - import error message path
    raise RuntimeError("ermbg.direct_worker_server requires the 'web' dependencies") from exc


@dataclass(frozen=True)
class PreparedRequest:
    index: int
    filename: str
    rgb: np.ndarray
    source_alpha: np.ndarray | None
    decision: RouteDecision
    route_sec: float
    decode_sec: float


_GPU_LOCK = threading.Lock()
_CK_FACTORY = DirectCorridorKeyClientFactory()
_CPU_EXECUTOR: ProcessPoolExecutor | None = None
_CPU_WORKERS = max(1, int(os.environ.get("ERMBG_DIRECT_CPU_WORKERS", min(4, os.cpu_count() or 1))))


def _executor() -> ProcessPoolExecutor | None:
    global _CPU_EXECUTOR
    if _CPU_WORKERS <= 1:
        return None
    if _CPU_EXECUTOR is None:
        _CPU_EXECUTOR = ProcessPoolExecutor(max_workers=_CPU_WORKERS)
    return _CPU_EXECUTOR


def _decode_image_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        image = Image.open(io.BytesIO(data))
        has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
        if has_alpha:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            return rgba[..., :3].copy(), rgba[..., 3].astype(np.float32) / 255.0
        return np.asarray(image.convert("RGB"), dtype=np.uint8), None
    except Exception as exc:
        raise ValueError(f"invalid image upload: {exc}") from exc


def _encode_rgba_png_base64(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_gray_png_base64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _git_sha() -> str:
    env_sha = os.environ.get("ERMBG_BUILD_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        root = os.path.dirname(os.path.dirname(__file__))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        ).strip()
    except Exception:
        return "unknown"


def _algorithm_debug(result: DirectWorkerResult) -> dict[str, Any]:
    debug = result.response.debug if isinstance(result.response.debug, dict) else {}
    known_b = debug.get("pymatting_known_b") if isinstance(debug.get("pymatting_known_b"), dict) else None
    known_bg_glow = debug.get("known_bg_glow") if isinstance(debug.get("known_bg_glow"), dict) else None
    hint = debug.get("hint") if isinstance(debug.get("hint"), dict) else None
    corridorkey_mask = debug.get("corridorkey_mask") if isinstance(debug.get("corridorkey_mask"), dict) else None
    corridorkey_hint_plan = (
        debug.get("corridorkey_hint_plan") if isinstance(debug.get("corridorkey_hint_plan"), dict) else None
    )
    settings = debug.get("settings") if isinstance(debug.get("settings"), dict) else None
    hard_ui_hint = debug.get("hard_ui_hint") if isinstance(debug.get("hard_ui_hint"), dict) else None
    shadow = debug.get("shadow") if isinstance(debug.get("shadow"), dict) else None
    semantic_execution = (
        debug.get("semantic_execution") if isinstance(debug.get("semantic_execution"), dict) else None
    )
    algorithm: dict[str, Any] = {
        "strategy": result.response.strategy_name,
        "execution_backend": result.metadata.get("execution_backend"),
        "git_sha": _git_sha(),
    }
    if known_b is not None:
        algorithm["pymatting_known_b"] = {
            "trimap": known_b.get("trimap"),
            "parameters": known_b.get("parameters"),
            "alpha_pinhole_repair": known_b.get("alpha_pinhole_repair"),
        }
    if known_bg_glow is not None:
        algorithm["known_bg_glow"] = known_bg_glow
    if hint is not None:
        algorithm["hint"] = hint
    if corridorkey_mask is not None:
        algorithm["corridorkey_mask"] = corridorkey_mask
    if corridorkey_hint_plan is not None:
        algorithm["corridorkey_hint_plan"] = corridorkey_hint_plan
    if settings is not None:
        algorithm["settings"] = settings
    if hard_ui_hint is not None:
        algorithm["hard_ui_hint"] = hard_ui_hint
    if shadow is not None:
        algorithm["shadow"] = shadow
    if semantic_execution is not None:
        algorithm["semantic_execution"] = semantic_execution
    return algorithm


def _metadata_payload(result: DirectWorkerResult) -> dict[str, Any]:
    payload = dict(result.metadata)
    payload["timings"] = result.timings
    payload["response_strategy"] = result.response.strategy_name
    payload["background"] = list(result.response.background_color)
    payload["algorithm_debug"] = _algorithm_debug(result)
    if isinstance(result.response.report, dict):
        payload["report"] = result.response.report
    return payload


def _case_payload(
    *,
    filename: str,
    result: DirectWorkerResult,
    decode_sec: float,
    include_image: bool,
) -> dict[str, Any]:
    timings = dict(result.timings)
    timings["decode_sec"] = float(decode_sec)
    payload = {
        "status": "ok",
        "filename": filename,
        "width": int(result.response.rgba.shape[1]),
        "height": int(result.response.rgba.shape[0]),
        "strategy": result.response.strategy_name,
        "background": list(result.response.background_color),
        "algorithm": result.metadata.get("algorithm"),
        "execution_backend": result.metadata.get("execution_backend"),
        "route": result.metadata.get("route"),
        "asset_kind": result.metadata.get("asset_kind"),
        "parameter_profile": result.metadata.get("parameter_profile"),
        "execution_profile": result.metadata.get("execution_profile"),
        "shadow_mode": result.metadata.get("shadow_mode"),
        "timings": timings,
        "algorithm_debug": _algorithm_debug(result),
    }
    if include_image:
        payload["rgba_png_base64"] = _encode_rgba_png_base64(result.response.rgba)
        debug = result.response.debug if isinstance(result.response.debug, dict) else {}
        trimap = debug.get("trimap_u8")
        if isinstance(trimap, np.ndarray) and trimap.ndim == 2:
            payload["trimap_png_base64"] = _encode_gray_png_base64(trimap)
    return payload


def _prepare_request(
    index: int,
    filename: str,
    data: bytes,
    *,
    corridorkey_screen_mode: str,
    corridorkey_preset: str,
    fallback_bg_color: tuple[int, int, int],
) -> PreparedRequest:
    t = time.perf_counter()
    rgb, source_alpha = _decode_image_bytes(data)
    decode_sec = time.perf_counter() - t
    t = time.perf_counter()
    decision = classify_route(
        rgb,
        source_alpha=source_alpha,
        screen_mode=corridorkey_screen_mode,  # type: ignore[arg-type]
        preset=corridorkey_preset,  # type: ignore[arg-type]
        fallback_background_color=fallback_bg_color,
    )
    route_sec = time.perf_counter() - t
    return PreparedRequest(
        index=index,
        filename=filename,
        rgb=rgb,
        source_alpha=source_alpha,
        decision=decision,
        route_sec=route_sec,
        decode_sec=decode_sec,
    )


def _route_decision_from_payload(payload: dict[str, Any]) -> RouteDecision:
    route = str(payload.get("route") or "")
    backend = str(payload.get("backend") or payload.get("algorithm") or route)
    params = payload.get("params")
    analysis = payload.get("analysis")
    if not isinstance(params, dict):
        params = {}
    if not isinstance(analysis, dict):
        analysis = {}
    corridorkey_analysis = payload.get("corridorkey_analysis")
    if isinstance(corridorkey_analysis, dict) and "corridorkey_analysis" not in analysis:
        analysis = {**analysis, "corridorkey_analysis": corridorkey_analysis}
    reasons = payload.get("reasons")
    if not isinstance(reasons, list):
        reason = payload.get("reason")
        reasons = [str(reason)] if reason else ["explicit_route_decision"]
    confidence = payload.get("confidence")
    return RouteDecision(
        route=route or backend,
        asset_kind=str(payload.get("asset_kind") or "unknown"),
        backend=backend,
        params=dict(params),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 1.0,
        reasons=[str(item) for item in reasons],
        analysis=dict(analysis),
    )


def _prepare_request_from_route_decision(
    index: int,
    filename: str,
    data: bytes,
    route_decision_payload: dict[str, Any],
) -> PreparedRequest:
    t = time.perf_counter()
    rgb, source_alpha = _decode_image_bytes(data)
    decode_sec = time.perf_counter() - t
    return PreparedRequest(
        index=index,
        filename=filename,
        rgb=rgb,
        source_alpha=source_alpha,
        decision=_route_decision_from_payload(route_decision_payload),
        route_sec=0.0,
        decode_sec=decode_sec,
    )


def _decision_background_color(decision: RouteDecision, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    ck_analysis = decision.analysis.get("corridorkey_analysis")
    if isinstance(ck_analysis, dict):
        raw = ck_analysis.get("background_color")
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            return tuple(int(np.clip(c, 0, 255)) for c in raw)
    return fallback


def _apply_execution_backend_override(
    decision: RouteDecision,
    execution_backend: str,
    *,
    rgb: np.ndarray,
    fallback_bg_color: tuple[int, int, int],
) -> RouteDecision:
    requested = execution_backend.strip().lower()
    if requested in {"", "auto", "direct-worker"}:
        return decision
    if requested in {"direct-pymatting-known-b", "pymatting-known-b", "pymatting_known_b"}:
        from .pymatting_refine import estimate_stable_background_color

        stable_bg, stable_info = estimate_stable_background_color(rgb)
        bg_color = stable_bg if stable_info.get("accepted", False) else fallback_bg_color
        params = {
            "execution_profile": "pymatting-known-bg",
            "parameter_profile": "known_b_manual_background_standard",
            "pymatting_method": "cf",
            "pymatting_image_space": "linear",
            "pymatting_bg_source": "custom",
            "pymatting_bg_color": tuple(int(c) for c in bg_color),
            "pymatting_bg_threshold": 3.5,
            "pymatting_fg_threshold": 24.0,
            "pymatting_boundary_band_px": 2,
            "pymatting_adapt_bg_threshold": False,
            "pymatting_adapt_fg_threshold": True,
            "pymatting_adapt_boundary_band": True,
            "pymatting_cg_maxiter": 1000,
            "pymatting_cg_rtol": 1e-6,
        }
        return replace(
            decision,
            route="pymatting_known_b",
            asset_kind=decision.asset_kind if decision.asset_kind != "unknown" else "known_bg_graphic",
            backend="pymatting_known_b",
            params=params,
            confidence=max(0.50, float(decision.confidence)),
            reasons=[*decision.reasons, "manual_pymatting_known_b_algorithm"],
            analysis={**decision.analysis, "manual_stable_background": stable_info},
        )
    if requested == "direct-known-bg-glow":
        from .known_bg_glow import analyze_known_bg_glow

        bg_color = _decision_background_color(decision, fallback_bg_color)
        glow = analyze_known_bg_glow(rgb, bg_color)
        mode = (
            glow.mode
            if glow.accepted and glow.mode in {"single_target_line", "adaptive_ray", "chromatic_swap_ray"}
            else "adaptive_ray"
        )
        params = {
            "execution_profile": "known-bg-glow",
            "parameter_profile": f"known_bg_glow_{mode}",
            "known_bg_glow_mode": mode,
            "known_bg_glow_bg_color": glow.background_color,
            "known_bg_glow_target_color": glow.target_color,
        }
        return replace(
            decision,
            route="known_bg_glow",
            asset_kind="icon",
            backend="known_bg_glow",
            params=params,
            confidence=max(0.50, float(decision.confidence)),
            reasons=[*decision.reasons, "manual_direct_known_bg_glow_backend"],
            analysis={**decision.analysis, "known_bg_glow": glow.to_dict()},
        )
    if requested != "direct-corridorkey":
        raise HTTPException(
            status_code=400,
            detail="execution_backend must be auto, direct-worker, direct-corridorkey, direct-pymatting-known-b, or direct-known-bg-glow",
        )
    if not isinstance(decision.analysis.get("corridorkey_analysis"), dict):
        raise HTTPException(status_code=400, detail="direct-corridorkey requires corridorkey analysis metadata")
    params = {
        key: value
        for key, value in decision.params.items()
        if not key.startswith("pymatting_") and key != "parameter_profile"
    }
    if decision.backend == "corridorkey":
        params["execution_profile"] = params.get("execution_profile") or "auto"
        params["corridorkey_execution_profile"] = params.get("corridorkey_execution_profile") or params["execution_profile"]
    else:
        params["execution_profile"] = "auto"
        params["corridorkey_execution_profile"] = "auto"
    return replace(
        decision,
        route="corridorkey",
        backend="corridorkey",
        params=params,
        reasons=[*decision.reasons, "manual_direct_corridorkey_backend"],
    )


def _with_corridorkey_params(decision: RouteDecision, params: dict[str, Any]) -> RouteDecision:
    if not params:
        return decision
    return replace(decision, params={**decision.params, **params})


def _with_known_bg_glow_params(
    decision: RouteDecision,
    *,
    known_bg_glow_material_strength: float | None,
) -> RouteDecision:
    if known_bg_glow_material_strength is None or decision.backend != "known_bg_glow":
        return decision
    strength = float(np.clip(float(known_bg_glow_material_strength), 0.0, 2.0))
    return replace(decision, params={**decision.params, "known_bg_glow_material_strength": strength})


def _parse_optional_rgb_triplet(text: str | None) -> tuple[int, int, int] | None:
    if text is None or not str(text).strip():
        return None
    return _parse_bg_color(str(text))


def _parse_json_object(text: str | None, field: str) -> dict[str, Any] | None:
    if text is None or not str(text).strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field} must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"{field} must be a JSON object")
    return payload


def _known_b_form_params(
    *,
    pymatting_method: str | None,
    pymatting_image_space: str | None,
    pymatting_bg_source: str | None,
    pymatting_bg_color: str | None,
    pymatting_bg_threshold: float | None,
    pymatting_fg_threshold: float | None,
    pymatting_boundary_band_px: int | None,
    pymatting_adapt_bg_threshold: bool | None,
    pymatting_adapt_fg_threshold: bool | None,
    pymatting_adapt_boundary_band: bool | None,
    pymatting_cg_maxiter: int | None,
    pymatting_cg_rtol: float | None,
    pymatting_trimap_mode: str | None,
    pymatting_unknown_grow_px: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if pymatting_method is not None:
        params["pymatting_method"] = str(pymatting_method).strip().lower()
    if pymatting_image_space is not None:
        params["pymatting_image_space"] = str(pymatting_image_space)
    if pymatting_bg_source is not None:
        params["pymatting_bg_source"] = str(pymatting_bg_source).strip().lower()
    color = _parse_optional_rgb_triplet(pymatting_bg_color)
    if color is not None:
        params["pymatting_bg_color"] = color
    if pymatting_bg_threshold is not None:
        params["pymatting_bg_threshold"] = float(pymatting_bg_threshold)
    if pymatting_fg_threshold is not None:
        params["pymatting_fg_threshold"] = float(pymatting_fg_threshold)
    if pymatting_boundary_band_px is not None:
        params["pymatting_boundary_band_px"] = int(pymatting_boundary_band_px)
    if pymatting_adapt_bg_threshold is not None:
        params["pymatting_adapt_bg_threshold"] = bool(pymatting_adapt_bg_threshold)
    if pymatting_adapt_fg_threshold is not None:
        params["pymatting_adapt_fg_threshold"] = bool(pymatting_adapt_fg_threshold)
    if pymatting_adapt_boundary_band is not None:
        params["pymatting_adapt_boundary_band"] = bool(pymatting_adapt_boundary_band)
    if pymatting_cg_maxiter is not None:
        params["pymatting_cg_maxiter"] = int(pymatting_cg_maxiter)
    if pymatting_cg_rtol is not None:
        params["pymatting_cg_rtol"] = float(pymatting_cg_rtol)
    if pymatting_trimap_mode is not None:
        params["pymatting_trimap_mode"] = str(pymatting_trimap_mode)
    if pymatting_unknown_grow_px is not None:
        params["pymatting_unknown_grow_px"] = int(pymatting_unknown_grow_px)
    return params


def _with_known_b_params(decision: RouteDecision, params: dict[str, Any]) -> RouteDecision:
    if not params or decision.backend not in {"pymatting_known_b", "pymatting_fallback"}:
        return decision
    return replace(decision, params={**decision.params, **params})


def _parse_semantic_decision(text: str | None) -> dict[str, Any]:
    if text is None or not str(text).strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="semantic_decision must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="semantic_decision must be a JSON object")
    return payload


def _with_semantic_decision_params(decision: RouteDecision, semantic_decision: dict[str, Any]) -> RouteDecision:
    if not semantic_decision:
        return decision
    return replace(decision, params={**decision.params, "semantic_decision": semantic_decision})


def _with_user_mask_params(
    decision: RouteDecision,
    *,
    user_keep_mask: np.ndarray | None,
    user_remove_mask: np.ndarray | None,
) -> RouteDecision:
    params = dict(decision.params)
    changed = False
    if user_keep_mask is not None:
        params["user_keep_mask"] = user_keep_mask
        changed = True
    if user_remove_mask is not None:
        params["user_remove_mask"] = user_remove_mask
        changed = True
    return replace(decision, params=params) if changed else decision


def _with_explicit_trimap_param(decision: RouteDecision, trimap: np.ndarray | None) -> RouteDecision:
    if trimap is None or decision.backend not in {"pymatting_known_b", "pymatting_fallback"}:
        return decision
    return replace(decision, params={**decision.params, "pymatting_explicit_trimap": trimap})


async def _read_mask_upload(upload: UploadFile | None) -> np.ndarray | None:
    if upload is None:
        return None
    data = await upload.read()
    mask = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


async def _read_trimap_upload(upload: UploadFile | None) -> np.ndarray | None:
    if upload is None:
        return None
    data = await upload.read()
    trimap = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.uint8)
    # Preserve the candidate's 3-state contract while tolerating PNG roundoff
    # from UI previews or tests: dark=sure-bg, mid=unknown, light=sure-fg.
    out = np.full(trimap.shape, 128, dtype=np.uint8)
    out[trimap < 64] = 0
    out[trimap > 191] = 255
    return out


def _corridorkey_form_params(
    *,
    corridorkey_gamma_space: str | None,
    corridorkey_despill_strength: float | None,
    corridorkey_refiner_strength: float | None,
    corridorkey_auto_despeckle: str | None,
    corridorkey_despeckle_size: int | None,
    corridorkey_auto_mask: bool | None,
    corridorkey_color_protection: bool | None,
    corridorkey_protection_bg_max: float | None,
    corridorkey_protection_fg_min: float | None,
    corridorkey_screen_mode: str | None,
    corridorkey_preset: str | None,
    corridorkey_hard_ui_hint_mode: str | None,
) -> dict[str, Any]:
    def text_or_none(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    params = {
        "corridorkey_gamma_space": corridorkey_gamma_space,
        "corridorkey_despill_strength": None
        if corridorkey_despill_strength is None
        else float(corridorkey_despill_strength),
        "corridorkey_refiner_strength": None
        if corridorkey_refiner_strength is None
        else float(corridorkey_refiner_strength),
        "corridorkey_auto_despeckle": corridorkey_auto_despeckle,
        "corridorkey_despeckle_size": None if corridorkey_despeckle_size is None else int(corridorkey_despeckle_size),
        "corridorkey_auto_mask": None if corridorkey_auto_mask is None else bool(corridorkey_auto_mask),
        "corridorkey_color_protection": None
        if corridorkey_color_protection is None
        else bool(corridorkey_color_protection),
        "corridorkey_protection_bg_max": None
        if corridorkey_protection_bg_max is None
        else float(corridorkey_protection_bg_max),
        "corridorkey_protection_fg_min": None
        if corridorkey_protection_fg_min is None
        else float(corridorkey_protection_fg_min),
        "corridorkey_screen_mode": text_or_none(corridorkey_screen_mode),
        "corridorkey_preset": text_or_none(corridorkey_preset),
        "corridorkey_hard_ui_hint_mode": text_or_none(corridorkey_hard_ui_hint_mode),
    }
    return {key: value for key, value in params.items() if value is not None}


def _is_cpu_parallel_route(decision: RouteDecision) -> bool:
    return decision.backend in {"pymatting_known_b", "pymatting_fallback", "known_bg_glow"}


def _run_prepared_main(
    prepared: PreparedRequest,
    *,
    shadow_mode: str,
    corridorkey_hard_ui_hint_mode: str,
    corridorkey_hint_mask: np.ndarray | None,
    fallback_bg_color: tuple[int, int, int],
) -> DirectWorkerResult:
    if prepared.decision.backend == "corridorkey":
        with _GPU_LOCK:
            return direct_matte_from_decision(
                prepared.rgb,
                decision=prepared.decision,
                source_alpha=prepared.source_alpha,
                shadow_mode=shadow_mode,
                corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
                corridorkey_hint_mask=corridorkey_hint_mask,
                fallback_bg_color=fallback_bg_color,
                ck_factory=_CK_FACTORY,
                route_sec=prepared.route_sec,
            )
    return direct_matte_from_decision(
        prepared.rgb,
        decision=prepared.decision,
        source_alpha=prepared.source_alpha,
        shadow_mode=shadow_mode,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
        corridorkey_hint_mask=corridorkey_hint_mask,
        fallback_bg_color=fallback_bg_color,
        ck_factory=_CK_FACTORY,
        route_sec=prepared.route_sec,
    )


def _cpu_worker(
    rgb: np.ndarray,
    source_alpha: np.ndarray | None,
    decision: RouteDecision,
    *,
    shadow_mode: str,
    corridorkey_hard_ui_hint_mode: str,
    fallback_bg_color: tuple[int, int, int],
    route_sec: float,
    include_image: bool,
) -> dict[str, Any]:
    t = time.perf_counter()
    result = direct_matte_from_decision(
        rgb,
        decision=decision,
        source_alpha=source_alpha,
        shadow_mode=shadow_mode,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
        fallback_bg_color=fallback_bg_color,
        ck_factory=None,
        route_sec=route_sec,
    )
    payload = _metadata_payload(result)
    payload["worker_elapsed_sec"] = time.perf_counter() - t
    if include_image:
        payload["rgba_png_base64"] = _encode_rgba_png_base64(result.response.rgba)
    payload["width"] = int(result.response.rgba.shape[1])
    payload["height"] = int(result.response.rgba.shape[0])
    return payload


def _parse_bg_color(text: str) -> tuple[int, int, int]:
    try:
        parts = [int(part.strip()) for part in text.split(",")]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="fallback_bg_color must be 'R,G,B'") from exc
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail="fallback_bg_color must contain 3 comma-separated integers")
    return tuple(int(np.clip(c, 0, 255)) for c in parts)


app = FastAPI(title="ERMBG Direct Worker", version="0.1.0")


def _corridorkey_runtime_status() -> dict[str, Any]:
    """Report whether the local process can actually execute CorridorKey.

    The health endpoint is used by Web/API routing before dispatching expensive
    jobs. Keep this probe import-only: it must catch missing local custom-node
    installs, but it must not instantiate the model or allocate GPU memory.
    """

    status: dict[str, Any] = {"available": False}
    try:
        import torch

        status["torch_importable"] = True
        status["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        status["torch_importable"] = False
        status["torch_error"] = str(exc)
        return status

    try:
        node_cls = LocalCorridorKeyClient._loaded_corridorkey_node_class()
    except Exception as exc:
        status["loaded_node_error"] = str(exc)
        node_cls = None
    if node_cls is not None:
        status.update(
            {
                "available": True,
                "runner": "loaded_comfy_node",
                "node_module": str(getattr(node_cls, "__module__", "")),
                "node_class": str(getattr(node_cls, "__name__", type(node_cls).__name__)),
            }
        )
        return status

    try:
        LocalCorridorKeyClient._ensure_import_path()
        module = importlib.import_module("corridor_key")
        getattr(module, "CorridorKeyProcessor")
        getattr(module, "CorridorKeySettings")
    except Exception as exc:
        status["import_error"] = str(exc)
        return status

    status.update(
        {
            "available": True,
            "runner": "direct_processor_fallback",
            "module": "corridor_key",
        }
    )
    return status


@app.get("/health")
def health() -> dict[str, Any]:
    corridorkey_runtime = _corridorkey_runtime_status()
    info: dict[str, Any] = {
        "status": "ok",
        "backend": "direct-worker",
        "version": get_ermbg_version(),
        "git_sha": _git_sha(),
        "cpu_workers": _CPU_WORKERS,
        "gpu_concurrency": 1,
        "capabilities": {
            "route_profile_contract": True,
            "direct_pymatting_known_b": True,
            "direct_pymatting_explicit_trimap": True,
            "direct_corridorkey": bool(corridorkey_runtime.get("available")),
            "direct_known_bg_glow": True,
            "batch_matte": True,
        },
        "corridorkey_runtime": corridorkey_runtime,
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


@app.post("/matte")
async def matte_endpoint(
    image: UploadFile = File(...),
    corridorkey_hint_mask: UploadFile | None = File(None),
    user_keep_mask: UploadFile | None = File(None),
    user_remove_mask: UploadFile | None = File(None),
    pymatting_explicit_trimap: UploadFile | None = File(None),
    execution_backend: str = Form("auto"),
    shadow_mode: str = Form("auto"),
    corridorkey_gamma_space: str | None = Form(None),
    corridorkey_despill_strength: float | None = Form(None),
    corridorkey_refiner_strength: float | None = Form(None),
    corridorkey_auto_despeckle: str | None = Form(None),
    corridorkey_despeckle_size: int | None = Form(None),
    corridorkey_auto_mask: bool | None = Form(None),
    corridorkey_color_protection: bool | None = Form(None),
    corridorkey_protection_bg_max: float | None = Form(None),
    corridorkey_protection_fg_min: float | None = Form(None),
    corridorkey_screen_mode: str | None = Form(None),
    corridorkey_preset: str | None = Form(None),
    corridorkey_hard_ui_hint_mode: str | None = Form(None),
    known_bg_glow_material_strength: float | None = Form(None),
    pymatting_method: str | None = Form(None),
    pymatting_image_space: str | None = Form(None),
    pymatting_bg_source: str | None = Form(None),
    pymatting_bg_color: str | None = Form(None),
    pymatting_bg_threshold: float | None = Form(None),
    pymatting_fg_threshold: float | None = Form(None),
    pymatting_boundary_band_px: int | None = Form(None),
    pymatting_adapt_bg_threshold: bool | None = Form(None),
    pymatting_adapt_fg_threshold: bool | None = Form(None),
    pymatting_adapt_boundary_band: bool | None = Form(None),
    pymatting_cg_maxiter: int | None = Form(None),
    pymatting_cg_rtol: float | None = Form(None),
    pymatting_trimap_mode: str | None = Form(None),
    pymatting_unknown_grow_px: int | None = Form(None),
    route_decision: str | None = Form(None),
    semantic_decision: str | None = Form(None),
    fallback_bg_color: str = Form("0,200,0"),
    include_image: bool = Form(True),
) -> dict[str, Any]:
    started = time.perf_counter()
    bg = _parse_bg_color(fallback_bg_color)
    effective_corridorkey_screen_mode = str(corridorkey_screen_mode or "auto")
    effective_corridorkey_preset = str(corridorkey_preset or "auto")
    effective_corridorkey_hard_ui_hint_mode = str(corridorkey_hard_ui_hint_mode or "bbox_2px")
    route_decision_payload = _parse_json_object(route_decision, "route_decision")
    semantic_decision_payload = _parse_semantic_decision(semantic_decision)
    ck_params = _corridorkey_form_params(
        corridorkey_gamma_space=corridorkey_gamma_space,
        corridorkey_despill_strength=corridorkey_despill_strength,
        corridorkey_refiner_strength=corridorkey_refiner_strength,
        corridorkey_auto_despeckle=corridorkey_auto_despeckle,
        corridorkey_despeckle_size=corridorkey_despeckle_size,
        corridorkey_auto_mask=corridorkey_auto_mask,
        corridorkey_color_protection=corridorkey_color_protection,
        corridorkey_protection_bg_max=corridorkey_protection_bg_max,
        corridorkey_protection_fg_min=corridorkey_protection_fg_min,
        corridorkey_screen_mode=corridorkey_screen_mode,
        corridorkey_preset=corridorkey_preset,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
    )
    known_b_params = _known_b_form_params(
        pymatting_method=pymatting_method,
        pymatting_image_space=pymatting_image_space,
        pymatting_bg_source=pymatting_bg_source,
        pymatting_bg_color=pymatting_bg_color,
        pymatting_bg_threshold=pymatting_bg_threshold,
        pymatting_fg_threshold=pymatting_fg_threshold,
        pymatting_boundary_band_px=pymatting_boundary_band_px,
        pymatting_adapt_bg_threshold=pymatting_adapt_bg_threshold,
        pymatting_adapt_fg_threshold=pymatting_adapt_fg_threshold,
        pymatting_adapt_boundary_band=pymatting_adapt_boundary_band,
        pymatting_cg_maxiter=pymatting_cg_maxiter,
        pymatting_cg_rtol=pymatting_cg_rtol,
        pymatting_trimap_mode=pymatting_trimap_mode,
        pymatting_unknown_grow_px=pymatting_unknown_grow_px,
    )
    data = await image.read()
    hint_mask = None
    if corridorkey_hint_mask is not None and corridorkey_auto_mask is not True:
        hint_data = await corridorkey_hint_mask.read()
        hint = np.asarray(Image.open(io.BytesIO(hint_data)).convert("L"), dtype=np.float32) / 255.0
        hint_mask = np.clip(hint, 0.0, 1.0).astype(np.float32)
    keep_mask = await _read_mask_upload(user_keep_mask)
    remove_mask = await _read_mask_upload(user_remove_mask)
    explicit_trimap = await _read_trimap_upload(pymatting_explicit_trimap)
    try:
        if route_decision_payload is not None:
            prepared = _prepare_request_from_route_decision(
                0,
                image.filename or "image",
                data,
                route_decision_payload,
            )
            decision = prepared.decision
        else:
            prepared = _prepare_request(
                0,
                image.filename or "image",
                data,
                corridorkey_screen_mode=effective_corridorkey_screen_mode,
                corridorkey_preset=effective_corridorkey_preset,
                fallback_bg_color=bg,
            )
            decision = prepared.decision
        decision = _apply_execution_backend_override(
            decision,
            execution_backend,
            rgb=prepared.rgb,
            fallback_bg_color=bg,
        )
        decision = _with_known_bg_glow_params(
            decision,
            known_bg_glow_material_strength=known_bg_glow_material_strength,
        )
        decision = _with_known_b_params(decision, known_b_params)
        decision = _with_semantic_decision_params(decision, semantic_decision_payload)
        decision = _with_user_mask_params(
            decision,
            user_keep_mask=keep_mask,
            user_remove_mask=remove_mask,
        )
        decision = _with_explicit_trimap_param(decision, explicit_trimap)
        prepared = replace(prepared, decision=_with_corridorkey_params(decision, ck_params))
        result = _run_prepared_main(
            prepared,
            shadow_mode=shadow_mode,
            corridorkey_hard_ui_hint_mode=effective_corridorkey_hard_ui_hint_mode,
            corridorkey_hint_mask=hint_mask,
            fallback_bg_color=bg,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload = _case_payload(
        filename=prepared.filename,
        result=result,
        decode_sec=prepared.decode_sec,
        include_image=include_image,
    )
    payload["server_elapsed_sec"] = time.perf_counter() - started
    return payload


@app.post("/batch-matte")
async def batch_matte_endpoint(
    files: list[UploadFile] = File(...),
    execution_backend: str = Form("auto"),
    shadow_mode: str = Form("auto"),
    corridorkey_gamma_space: str | None = Form(None),
    corridorkey_despill_strength: float | None = Form(None),
    corridorkey_refiner_strength: float | None = Form(None),
    corridorkey_auto_despeckle: str | None = Form(None),
    corridorkey_despeckle_size: int | None = Form(None),
    corridorkey_auto_mask: bool | None = Form(None),
    corridorkey_color_protection: bool | None = Form(None),
    corridorkey_protection_bg_max: float | None = Form(None),
    corridorkey_protection_fg_min: float | None = Form(None),
    corridorkey_screen_mode: str | None = Form(None),
    corridorkey_preset: str | None = Form(None),
    corridorkey_hard_ui_hint_mode: str | None = Form(None),
    known_bg_glow_material_strength: float | None = Form(None),
    fallback_bg_color: str = Form("0,200,0"),
    include_images: bool = Form(True),
) -> dict[str, Any]:
    started = time.perf_counter()
    bg = _parse_bg_color(fallback_bg_color)
    effective_corridorkey_screen_mode = str(corridorkey_screen_mode or "auto")
    effective_corridorkey_preset = str(corridorkey_preset or "auto")
    effective_corridorkey_hard_ui_hint_mode = str(corridorkey_hard_ui_hint_mode or "bbox_2px")
    ck_params = _corridorkey_form_params(
        corridorkey_gamma_space=corridorkey_gamma_space,
        corridorkey_despill_strength=corridorkey_despill_strength,
        corridorkey_refiner_strength=corridorkey_refiner_strength,
        corridorkey_auto_despeckle=corridorkey_auto_despeckle,
        corridorkey_despeckle_size=corridorkey_despeckle_size,
        corridorkey_auto_mask=corridorkey_auto_mask,
        corridorkey_color_protection=corridorkey_color_protection,
        corridorkey_protection_bg_max=corridorkey_protection_bg_max,
        corridorkey_protection_fg_min=corridorkey_protection_fg_min,
        corridorkey_screen_mode=corridorkey_screen_mode,
        corridorkey_preset=corridorkey_preset,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
    )
    prepared: list[PreparedRequest] = []
    rows: list[dict[str, Any] | None] = [None] * len(files)
    for index, upload in enumerate(files):
        data = await upload.read()
        try:
            item = _prepare_request(
                index,
                upload.filename or f"image_{index}",
                data,
                corridorkey_screen_mode=effective_corridorkey_screen_mode,
                corridorkey_preset=effective_corridorkey_preset,
                fallback_bg_color=bg,
            )
            decision = _apply_execution_backend_override(
                item.decision,
                execution_backend,
                rgb=item.rgb,
                fallback_bg_color=bg,
            )
            decision = _with_known_bg_glow_params(
                decision,
                known_bg_glow_material_strength=known_bg_glow_material_strength,
            )
            prepared.append(replace(item, decision=_with_corridorkey_params(decision, ck_params)))
        except Exception as exc:
            rows[index] = {
                "status": "error",
                "filename": upload.filename or f"image_{index}",
                "error": str(exc),
            }

    executor = _executor()
    futures: dict[Any, PreparedRequest] = {}
    for item in prepared:
        if executor is not None and _is_cpu_parallel_route(item.decision):
            futures[
                executor.submit(
                    _cpu_worker,
                    item.rgb,
                    item.source_alpha,
                    item.decision,
                    shadow_mode=shadow_mode,
                    corridorkey_hard_ui_hint_mode=effective_corridorkey_hard_ui_hint_mode,
                    fallback_bg_color=bg,
                    route_sec=item.route_sec,
                    include_image=include_images,
                )
            ] = item
            continue
        try:
            result = _run_prepared_main(
                item,
                shadow_mode=shadow_mode,
                corridorkey_hard_ui_hint_mode=effective_corridorkey_hard_ui_hint_mode,
                corridorkey_hint_mask=None,
                fallback_bg_color=bg,
            )
            rows[item.index] = _case_payload(
                filename=item.filename,
                result=result,
                decode_sec=item.decode_sec,
                include_image=include_images,
            )
        except Exception as exc:
            rows[item.index] = {
                "status": "error",
                "filename": item.filename,
                "error": str(exc),
            }

    for future in as_completed(futures):
        item = futures[future]
        try:
            payload = future.result()
            timings = dict(payload.get("timings", {}))
            timings["decode_sec"] = item.decode_sec
            timings["worker_elapsed_sec"] = float(payload.get("worker_elapsed_sec", 0.0))
            row = {
                "status": "ok",
                "filename": item.filename,
                "width": int(payload.get("width", item.rgb.shape[1])),
                "height": int(payload.get("height", item.rgb.shape[0])),
                "strategy": payload.get("response_strategy"),
                "background": payload.get("background"),
                "algorithm": payload.get("algorithm"),
                "execution_backend": payload.get("execution_backend"),
                "route": payload.get("route"),
                "asset_kind": payload.get("asset_kind"),
                "parameter_profile": payload.get("parameter_profile"),
                "execution_profile": payload.get("execution_profile"),
                "shadow_mode": payload.get("shadow_mode"),
                "timings": timings,
            }
            if include_images and isinstance(payload.get("rgba_png_base64"), str):
                row["rgba_png_base64"] = payload["rgba_png_base64"]
            rows[item.index] = row
        except Exception as exc:
            rows[item.index] = {
                "status": "error",
                "filename": item.filename,
                "error": str(exc),
            }

    final_rows = [row if row is not None else {"status": "error", "error": "internal missing row"} for row in rows]
    ok_count = sum(1 for row in final_rows if row.get("status") == "ok")
    return {
        "status": "ok" if ok_count == len(final_rows) else "partial",
        "case_count": len(final_rows),
        "ok_count": ok_count,
        "cpu_workers": _CPU_WORKERS,
        "gpu_concurrency": 1,
        "server_elapsed_sec": time.perf_counter() - started,
        "runs": final_rows,
    }


def main() -> None:
    global _CPU_WORKERS
    parser = argparse.ArgumentParser(description="Run the ERMBG direct worker HTTP server.")
    parser.add_argument("--host", default=os.environ.get("ERMBG_DIRECT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ERMBG_DIRECT_PORT", "7871")))
    parser.add_argument("--cpu-workers", type=int, default=_CPU_WORKERS)
    args = parser.parse_args()

    _CPU_WORKERS = max(1, int(args.cpu_workers))

    import uvicorn

    uvicorn.run("ermbg.direct_worker_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
