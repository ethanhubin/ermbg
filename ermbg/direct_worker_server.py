"""HTTP server for the ERMBG direct worker validation path.

This is intentionally separate from the production Web/API/Comfy route. It is
for validating a remote service that bypasses ComfyUI's prompt queue while
reusing ERMBG's current route decisions and maintained matting executors.
"""

from __future__ import annotations

import argparse
import base64
import io
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
    direct_matte_auto,
    direct_matte_from_decision,
)
from .runtime_capabilities import get_ermbg_version
from .router import RouteDecision, classify_route

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except Exception as exc:  # pragma: no cover - import error message path
    raise RuntimeError("ermbg.direct_worker_server requires the 'web' dependencies") from exc


@dataclass(frozen=True)
class PreparedRequest:
    index: int
    filename: str
    rgb: np.ndarray
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


def _decode_image_bytes(data: bytes) -> np.ndarray:
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
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
    shadow = debug.get("shadow") if isinstance(debug.get("shadow"), dict) else None
    algorithm: dict[str, Any] = {
        "strategy": result.response.strategy_name,
        "execution_backend": result.metadata.get("execution_backend"),
        "git_sha": _git_sha(),
    }
    if known_b is not None:
        algorithm["pymatting_known_b"] = {
            "background_normalization": known_b.get("background_normalization"),
            "trimap": known_b.get("trimap"),
            "parameters": known_b.get("parameters"),
            "alpha_pinhole_repair": known_b.get("alpha_pinhole_repair"),
        }
    if known_bg_glow is not None:
        algorithm["known_bg_glow"] = known_bg_glow
    if shadow is not None:
        algorithm["shadow"] = shadow
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
        "selected_backend": result.metadata.get("selected_backend"),
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
    rgb = _decode_image_bytes(data)
    decode_sec = time.perf_counter() - t
    t = time.perf_counter()
    decision = classify_route(
        rgb,
        source_alpha=None,
        screen_mode=corridorkey_screen_mode,  # type: ignore[arg-type]
        preset=corridorkey_preset,  # type: ignore[arg-type]
        fallback_background_color=fallback_bg_color,
    )
    route_sec = time.perf_counter() - t
    return PreparedRequest(
        index=index,
        filename=filename,
        rgb=rgb,
        decision=decision,
        route_sec=route_sec,
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
    if requested == "direct-known-bg-glow":
        from .known_bg_glow import analyze_known_bg_glow

        bg_color = _decision_background_color(decision, fallback_bg_color)
        glow = analyze_known_bg_glow(rgb, bg_color)
        mode = glow.mode if glow.accepted and glow.mode in {"single_target_line", "adaptive_ray"} else "adaptive_ray"
        params = {
            "execution_profile": "known-bg-glow",
            "known_bg_glow_mode": mode,
            "known_bg_glow_bg_color": glow.background_color,
            "known_bg_glow_target_color": glow.target_color,
        }
        return replace(
            decision,
            route="known_bg_glow",
            asset_kind="icon",
            backend="direct-known-bg-glow",
            params=params,
            confidence=max(0.50, float(decision.confidence)),
            reasons=[*decision.reasons, "manual_direct_known_bg_glow_backend"],
            analysis={**decision.analysis, "known_bg_glow": glow.to_dict()},
        )
    if requested != "direct-corridorkey":
        raise HTTPException(
            status_code=400,
            detail="execution_backend must be auto, direct-worker, direct-corridorkey, or direct-known-bg-glow",
        )
    if not isinstance(decision.analysis.get("corridorkey_analysis"), dict):
        raise HTTPException(status_code=400, detail="direct-corridorkey requires corridorkey analysis metadata")
    params = {key: value for key, value in decision.params.items() if not key.startswith("pymatting_")}
    params["execution_profile"] = "auto"
    params["corridorkey_execution_profile"] = "auto"
    return replace(
        decision,
        route="corridorkey",
        backend="comfy-corridorkey",
        params=params,
        reasons=[*decision.reasons, "manual_direct_corridorkey_backend"],
    )


def _with_corridorkey_params(decision: RouteDecision, params: dict[str, Any]) -> RouteDecision:
    if not params:
        return decision
    return replace(decision, params={**decision.params, **params})


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
    corridorkey_screen_mode: str,
    corridorkey_preset: str,
    corridorkey_hard_ui_hint_mode: str,
) -> dict[str, Any]:
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
        "corridorkey_screen_mode": corridorkey_screen_mode,
        "corridorkey_preset": corridorkey_preset,
        "corridorkey_hard_ui_hint_mode": corridorkey_hard_ui_hint_mode,
    }
    return {key: value for key, value in params.items() if value is not None}


def _is_cpu_parallel_route(decision: RouteDecision) -> bool:
    return decision.backend in {"comfy-pymatting-known-b", "direct-known-bg-glow"}


def _run_prepared_main(
    prepared: PreparedRequest,
    *,
    shadow_mode: str,
    corridorkey_hard_ui_hint_mode: str,
    fallback_bg_color: tuple[int, int, int],
) -> DirectWorkerResult:
    if prepared.decision.backend == "comfy-corridorkey":
        with _GPU_LOCK:
            return direct_matte_from_decision(
                prepared.rgb,
                decision=prepared.decision,
                shadow_mode=shadow_mode,
                corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
                fallback_bg_color=fallback_bg_color,
                ck_factory=_CK_FACTORY,
                route_sec=prepared.route_sec,
            )
    return direct_matte_from_decision(
        prepared.rgb,
        decision=prepared.decision,
        shadow_mode=shadow_mode,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
        fallback_bg_color=fallback_bg_color,
        ck_factory=_CK_FACTORY,
        route_sec=prepared.route_sec,
    )


def _cpu_worker(
    rgb: np.ndarray,
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


@app.get("/health")
def health() -> dict[str, Any]:
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
            "direct_corridorkey": True,
            "direct_known_bg_glow": True,
            "batch_matte": True,
        },
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
    corridorkey_screen_mode: str = Form("auto"),
    corridorkey_preset: str = Form("auto"),
    corridorkey_hard_ui_hint_mode: str = Form("bbox_2px"),
    fallback_bg_color: str = Form("0,200,0"),
    include_image: bool = Form(True),
) -> dict[str, Any]:
    started = time.perf_counter()
    bg = _parse_bg_color(fallback_bg_color)
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
    data = await image.read()
    try:
        prepared = _prepare_request(
            0,
            image.filename or "image",
            data,
            corridorkey_screen_mode=corridorkey_screen_mode,
            corridorkey_preset=corridorkey_preset,
            fallback_bg_color=bg,
        )
        decision = _apply_execution_backend_override(
            prepared.decision,
            execution_backend,
            rgb=prepared.rgb,
            fallback_bg_color=bg,
        )
        prepared = replace(prepared, decision=_with_corridorkey_params(decision, ck_params))
        result = _run_prepared_main(
            prepared,
            shadow_mode=shadow_mode,
            corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
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
    corridorkey_screen_mode: str = Form("auto"),
    corridorkey_preset: str = Form("auto"),
    corridorkey_hard_ui_hint_mode: str = Form("bbox_2px"),
    fallback_bg_color: str = Form("0,200,0"),
    include_images: bool = Form(True),
) -> dict[str, Any]:
    started = time.perf_counter()
    bg = _parse_bg_color(fallback_bg_color)
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
                corridorkey_screen_mode=corridorkey_screen_mode,
                corridorkey_preset=corridorkey_preset,
                fallback_bg_color=bg,
            )
            decision = _apply_execution_backend_override(
                item.decision,
                execution_backend,
                rgb=item.rgb,
                fallback_bg_color=bg,
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
                    item.decision,
                    shadow_mode=shadow_mode,
                    corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
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
                corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
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
                "selected_backend": payload.get("selected_backend"),
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
