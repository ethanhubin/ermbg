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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .direct_worker import (
    DirectCorridorKeyClientFactory,
    DirectWorkerResult,
    direct_matte_auto,
    direct_matte_from_decision,
)
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


def _metadata_payload(result: DirectWorkerResult) -> dict[str, Any]:
    payload = dict(result.metadata)
    payload["timings"] = result.timings
    payload["response_strategy"] = result.response.strategy_name
    payload["background"] = list(result.response.background_color)
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
        "timings": timings,
    }
    if include_image:
        payload["rgba_png_base64"] = _encode_rgba_png_base64(result.response.rgba)
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


def _is_cpu_parallel_route(decision: RouteDecision) -> bool:
    return decision.backend == "comfy-pymatting-known-b"


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
        "cpu_workers": _CPU_WORKERS,
        "gpu_concurrency": 1,
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
    shadow_mode: str = Form("on"),
    corridorkey_screen_mode: str = Form("auto"),
    corridorkey_preset: str = Form("auto"),
    corridorkey_hard_ui_hint_mode: str = Form("bbox_2px"),
    fallback_bg_color: str = Form("0,200,0"),
    include_image: bool = Form(True),
) -> dict[str, Any]:
    started = time.perf_counter()
    bg = _parse_bg_color(fallback_bg_color)
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
        result = _run_prepared_main(
            prepared,
            shadow_mode=shadow_mode,
            corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
            fallback_bg_color=bg,
        )
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
    shadow_mode: str = Form("on"),
    corridorkey_screen_mode: str = Form("auto"),
    corridorkey_preset: str = Form("auto"),
    corridorkey_hard_ui_hint_mode: str = Form("bbox_2px"),
    fallback_bg_color: str = Form("0,200,0"),
    include_images: bool = Form(True),
) -> dict[str, Any]:
    started = time.perf_counter()
    bg = _parse_bg_color(fallback_bg_color)
    prepared: list[PreparedRequest] = []
    rows: list[dict[str, Any] | None] = [None] * len(files)
    for index, upload in enumerate(files):
        data = await upload.read()
        try:
            prepared.append(
                _prepare_request(
                    index,
                    upload.filename or f"image_{index}",
                    data,
                    corridorkey_screen_mode=corridorkey_screen_mode,
                    corridorkey_preset=corridorkey_preset,
                    fallback_bg_color=bg,
                )
            )
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
