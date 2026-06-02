#!/usr/bin/env python3
"""Benchmark ERMBG's direct worker path without submitting ComfyUI prompts.

This script is intentionally a side-channel validation path. It does not change
the Web/API/Comfy production flow; it imports the same ERMBG route helpers and,
for CorridorKey, calls the local CorridorKey processor directly inside the
current Python process.

Typical remote run on the Windows/4090 host:

    cd C:/Users/darkv/ermbg_src
    E:/ComfyUI/.venv/Scripts/python.exe scripts/benchmark_direct_worker_path.py \
      --sample-id B001,B002,I001,I011,C001,B055 \
      --fixed-execution-backend direct-pymatting-known-b \
      --compare-summary out/auto_routematte_routefix_20260531/summary.json

Outputs are written under out/direct_worker_benchmark_<YYYYMMDD>_vNNN by
default. Each case gets a summary.json, a standard ERMBG manifest.json, and,
unless disabled, an rgba.png.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ermbg.artifacts import build_run_manifest, write_run_manifest
from ermbg.direct_worker import DirectCorridorKeyClientFactory, direct_matte_auto, direct_matte_from_decision
from ermbg.router import classify_route

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json"
DEFAULT_SAMPLE_IDS = "B001,B002,I001,I011,C001,B055,B046,B049"


@dataclass
class CaseBenchmark:
    status: str
    case: str
    sample_id: str
    input: str
    category: str
    image_size: list[int]
    requested_backend: str
    selected_backend: str | None
    execution_backend: str | None
    route: str | None
    asset_kind: str | None
    parameter_profile: str | None
    execution_profile: str | None
    timings: dict[str, float]
    output: dict[str, str]
    compare: dict[str, Any]
    error: str | None = None
    debug: dict[str, Any] | None = None


@dataclass
class PreparedCase:
    index: int
    case: dict[str, Any]
    case_key: str
    input_path: Path
    image_size: list[int]
    route_timings: dict[str, float]
    decision: Any


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            summary.update(
                {
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.mean()),
                }
            )
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _next_out_dir(prefix: str) -> Path:
    out_root = PROJECT_ROOT / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    version = 1
    while True:
        candidate = out_root / f"{prefix}_{date}_v{version:03d}"
        if not candidate.exists():
            return candidate
        version += 1


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("manifest must be a JSON object with a cases list")
    return payload


def _selected_cases(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [case for case in manifest["cases"] if isinstance(case, dict) and isinstance(case.get("input"), str)]
    if not args.all:
        sample_ids = {item.strip() for item in args.sample_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("sample_id", "")) in sample_ids]
    if args.category:
        categories = {item.strip() for item in args.category.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("category", "")) in categories]
    return cases


def _case_screen(case: dict[str, Any]) -> str:
    input_value = str(case.get("input", ""))
    for screen in ("green", "blue", "white"):
        if isinstance(case.get(screen), str) and str(case[screen]) == input_value:
            return screen
    screen = case.get("screen")
    return str(screen) if isinstance(screen, str) and screen else Path(input_value).stem


def _case_key(case: dict[str, Any]) -> str:
    return f"{case.get('sample_id')}_{case.get('id')}_{_case_screen(case)}"


def _load_comfy_compare(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("runs")
    if not isinstance(rows, list):
        raise ValueError("--compare-summary must point to a batch summary with a runs list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("case"), str):
            out[str(row["case"])] = row
    return out


def _compare_payload(case_key: str, compare_rows: dict[str, dict[str, Any]], direct_total: float) -> dict[str, Any]:
    row = compare_rows.get(case_key)
    if not row:
        return {}
    timings = row.get("timings") if isinstance(row.get("timings"), dict) else {}
    elapsed = row.get("elapsed_sec_client")
    payload: dict[str, Any] = {
        "comfy_backend": row.get("backend"),
        "comfy_elapsed_sec_client": elapsed,
        "comfy_timings": timings,
    }
    if isinstance(elapsed, (int, float)) and direct_total > 0.0:
        payload["speedup_vs_comfy_client"] = float(elapsed) / direct_total
        payload["saved_sec_vs_comfy_client"] = float(elapsed) - direct_total
    return payload


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_cuda_device_count"] = int(torch.cuda.device_count())
            info["torch_cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _write_rgba(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(path)


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    arr = np.asarray(mask)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0 + 0.5
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _response_image_arrays(response: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    rgba = getattr(response, "rgba", None)
    if isinstance(rgba, np.ndarray):
        arrays["rgba"] = rgba
    alpha = getattr(response, "alpha", None)
    if isinstance(alpha, np.ndarray):
        arrays["alpha"] = alpha
    foreground = getattr(response, "foreground_srgb", None)
    if isinstance(foreground, np.ndarray):
        arrays["foreground"] = foreground
    debug = getattr(response, "debug", None)
    if isinstance(debug, dict):
        trimap = debug.get("trimap_u8")
        if isinstance(trimap, np.ndarray) and trimap.ndim == 2:
            arrays["trimap"] = trimap
        shadow = debug.get("shadow_alpha")
        if isinstance(shadow, np.ndarray):
            arrays["shadow"] = shadow
        shadow_physical = debug.get("shadow_alpha_physical")
        if isinstance(shadow_physical, np.ndarray):
            arrays["shadow_physical"] = shadow_physical
    return arrays


def _write_case_images(case_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for key, array in arrays.items():
        path = case_dir / f"{key}.png"
        if key == "rgba":
            _write_rgba(path, array)
        elif key == "foreground":
            _write_rgb(path, array)
        else:
            _write_mask(path, array)
        outputs[key] = _rel(path)
    return outputs


def _classify_prepared_case(
    index: int,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[PreparedCase, np.ndarray]:
    case_key = _case_key(case)
    input_path = PROJECT_ROOT / str(case["input"])
    timings: dict[str, float] = {}
    t = time.perf_counter()
    rgb = _load_rgb(input_path)
    timings["load_decode_sec"] = time.perf_counter() - t
    t = time.perf_counter()
    decision = classify_route(
        rgb,
        source_alpha=None,
        screen_mode=args.corridorkey_screen_mode,  # type: ignore[arg-type]
        preset=args.corridorkey_preset,  # type: ignore[arg-type]
        fallback_background_color=tuple(args.fallback_bg_color),
    )
    timings["route_sec"] = time.perf_counter() - t
    return (
        PreparedCase(
            index=index,
            case=case,
            case_key=case_key,
            input_path=input_path,
            image_size=[int(rgb.shape[1]), int(rgb.shape[0])],
            route_timings=timings,
            decision=decision,
        ),
        rgb,
    )


def _cpu_parallel_backend(decision: Any) -> bool:
    return _execution_backend_from_decision(decision) == "direct-pymatting-known-b"


def _execution_backend_from_decision(decision: Any) -> str:
    selected_backend = str(getattr(decision, "backend", "") or "")
    if selected_backend == "comfy-pymatting-known-b":
        return "direct-pymatting-known-b"
    if selected_backend == "comfy-corridorkey":
        return "direct-corridorkey"
    if selected_backend == "passthrough":
        return "direct-passthrough"
    return selected_backend


def _requested_backend(args: argparse.Namespace) -> str:
    return str(getattr(args, "fixed_execution_backend", "") or "direct-auto")


def _fixed_backend_mismatch(prepared: PreparedCase, args: argparse.Namespace) -> str | None:
    fixed = str(getattr(args, "fixed_execution_backend", "") or "")
    if not fixed:
        return None
    actual = _execution_backend_from_decision(prepared.decision)
    if actual == fixed:
        return None
    return f"fixed execution backend {fixed!r} rejected routed backend {actual!r}"


def _run_prepared_case_in_main(
    prepared: PreparedCase,
    rgb: np.ndarray,
    *,
    args: argparse.Namespace,
    ck_factory: DirectCorridorKeyClientFactory,
    compare_rows: dict[str, dict[str, Any]],
    out_root: Path,
) -> CaseBenchmark:
    case_dir = out_root / prepared.case_key
    case_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    timings = dict(prepared.route_timings)
    output: dict[str, str] = {}
    result = direct_matte_from_decision(
        rgb,
        decision=prepared.decision,
        shadow_mode=args.shadow_mode,
        corridorkey_hard_ui_hint_mode=args.corridorkey_hard_ui_hint_mode,
        fallback_bg_color=tuple(args.fallback_bg_color),
        ck_factory=ck_factory,
        route_sec=timings.get("route_sec"),
    )
    timings.update(result.timings)
    if args.write_images:
        t = time.perf_counter()
        output.update(_write_case_images(case_dir, _response_image_arrays(result.response)))
        timings["encode_images_sec"] = time.perf_counter() - t
    timings["total_sec"] = (
        time.perf_counter() - total_start
        + timings.get("load_decode_sec", 0.0)
        + timings.get("route_sec", 0.0)
    )
    return CaseBenchmark(
        status="ok",
        case=prepared.case_key,
        sample_id=str(prepared.case.get("sample_id", "")),
        input=_rel(prepared.input_path),
        category=str(prepared.case.get("category", "")),
        image_size=prepared.image_size,
        requested_backend=_requested_backend(args),
        selected_backend=str(result.metadata.get("selected_backend")),
        execution_backend=str(result.metadata.get("execution_backend")),
        route=str(result.metadata.get("route")),
        asset_kind=str(result.metadata.get("asset_kind")),
        parameter_profile=str(result.metadata.get("parameter_profile")),
        execution_profile=str(result.metadata.get("execution_profile")),
        timings=timings,
        output=output,
        compare=_compare_payload(prepared.case_key, compare_rows, timings["total_sec"]),
        debug=_json_safe(result.response.debug) if args.include_debug else None,
    )


def _cpu_case_worker(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(payload["input_path"])
    total_start = time.perf_counter()
    timings = dict(payload["route_timings"])
    output: dict[str, Any] = {}
    try:
        t = time.perf_counter()
        rgb = _load_rgb(input_path)
        timings["worker_load_decode_sec"] = time.perf_counter() - t
        result = direct_matte_from_decision(
            rgb,
            decision=payload["decision"],
            shadow_mode=payload["shadow_mode"],
            corridorkey_hard_ui_hint_mode=payload["corridorkey_hard_ui_hint_mode"],
            fallback_bg_color=tuple(payload["fallback_bg_color"]),
            ck_factory=None,
            route_sec=timings.get("route_sec"),
        )
        timings.update(result.timings)
        if payload["write_images"]:
            output["arrays"] = _response_image_arrays(result.response)
        timings["worker_elapsed_sec"] = time.perf_counter() - total_start
        # In parallel mode, ``total_sec`` remains per-case compute accounting;
        # batch wall-clock speedup is reported separately at the aggregate level.
        timings["total_sec"] = (
            timings.get("load_decode_sec", 0.0)
            + timings.get("route_sec", 0.0)
            + timings.get("worker_load_decode_sec", 0.0)
            + timings.get("backend_sec", 0.0)
        )
        return {
            "status": "ok",
            "timings": timings,
            "metadata": result.metadata,
            "debug": _json_safe(result.response.debug) if payload["include_debug"] else None,
            "output": output,
            "error": None,
        }
    except Exception as exc:
        timings["worker_elapsed_sec"] = time.perf_counter() - total_start
        timings["total_sec"] = timings.get("load_decode_sec", 0.0) + timings["worker_elapsed_sec"]
        return {
            "status": "error",
            "timings": timings,
            "metadata": {},
            "debug": None,
            "output": output,
            "error": str(exc),
        }


def _row_from_worker_payload(
    prepared: PreparedCase,
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    compare_rows: dict[str, dict[str, Any]],
    out_root: Path,
) -> CaseBenchmark:
    case_dir = out_root / prepared.case_key
    case_dir.mkdir(parents=True, exist_ok=True)
    timings = dict(payload["timings"])
    output: dict[str, str] = {}
    worker_output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    arrays = worker_output.get("arrays") if isinstance(worker_output, dict) else None
    if args.write_images and isinstance(arrays, dict):
        t = time.perf_counter()
        image_arrays = {str(key): value for key, value in arrays.items() if isinstance(value, np.ndarray)}
        output.update(_write_case_images(case_dir, image_arrays))
        timings["encode_images_sec"] = time.perf_counter() - t
        timings["total_sec"] += timings["encode_images_sec"]
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return CaseBenchmark(
        status=str(payload["status"]),
        case=prepared.case_key,
        sample_id=str(prepared.case.get("sample_id", "")),
        input=_rel(prepared.input_path),
        category=str(prepared.case.get("category", "")),
        image_size=prepared.image_size,
        requested_backend=_requested_backend(args),
        selected_backend=str(metadata.get("selected_backend")) if metadata else None,
        execution_backend=str(metadata.get("execution_backend")) if metadata else None,
        route=str(metadata.get("route")) if metadata else None,
        asset_kind=str(metadata.get("asset_kind")) if metadata else None,
        parameter_profile=str(metadata.get("parameter_profile")) if metadata else None,
        execution_profile=str(metadata.get("execution_profile")) if metadata else None,
        timings=timings,
        output=output,
        compare=_compare_payload(prepared.case_key, compare_rows, timings["total_sec"]),
        error=payload.get("error"),
        debug=payload.get("debug") if args.include_debug else None,
    )


def _direct_matte(
    rgb: np.ndarray,
    *,
    shadow_mode: str,
    corridorkey_screen_mode: str,
    corridorkey_preset: str,
    corridorkey_hard_ui_hint_mode: str,
    fallback_bg_color: tuple[int, int, int],
    ck_factory: DirectCorridorKeyClientFactory,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    result = direct_matte_auto(
        rgb,
        shadow_mode=shadow_mode,
        corridorkey_screen_mode=corridorkey_screen_mode,
        corridorkey_preset=corridorkey_preset,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
        fallback_bg_color=fallback_bg_color,
        ck_factory=ck_factory,
    )
    return result.response, result.timings, result.metadata


def _summarize(rows: list[CaseBenchmark]) -> dict[str, Any]:
    ok = [row for row in rows if row.status == "ok"]
    by_backend: dict[str, list[CaseBenchmark]] = {}
    for row in ok:
        by_backend.setdefault(str(row.selected_backend), []).append(row)

    def metrics(items: list[CaseBenchmark]) -> dict[str, Any]:
        keys = sorted({key for item in items for key in item.timings})
        out: dict[str, Any] = {"count": len(items)}
        for key in keys:
            vals = [item.timings[key] for item in items if key in item.timings]
            if vals:
                out[key] = {
                    "avg": float(sum(vals) / len(vals)),
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                }
        speedups = [
            float(item.compare["speedup_vs_comfy_client"])
            for item in items
            if isinstance(item.compare.get("speedup_vs_comfy_client"), (int, float))
        ]
        saved = [
            float(item.compare["saved_sec_vs_comfy_client"])
            for item in items
            if isinstance(item.compare.get("saved_sec_vs_comfy_client"), (int, float))
        ]
        if speedups:
            out["compare_speedup_vs_comfy_client"] = {
                "avg": float(sum(speedups) / len(speedups)),
                "min": float(min(speedups)),
                "max": float(max(speedups)),
            }
        if saved:
            out["compare_saved_sec_vs_comfy_client"] = {
                "avg": float(sum(saved) / len(saved)),
                "min": float(min(saved)),
                "max": float(max(saved)),
            }
        return out

    return {
        "overall": metrics(ok),
        "by_backend": {backend: metrics(items) for backend, items in sorted(by_backend.items())},
    }


def _benchmark_backend(row: CaseBenchmark) -> str:
    return row.execution_backend or row.selected_backend or row.requested_backend


def _row_payload(row: CaseBenchmark, *, artifact_manifest: Path | None = None) -> dict[str, Any]:
    payload = asdict(row)
    payload["backend"] = _benchmark_backend(row)
    payload["outputs"] = dict(row.output)
    if artifact_manifest is not None:
        payload["artifact_manifest"] = _rel(artifact_manifest)
    return payload


def _write_case_manifest(*, row: CaseBenchmark, case_dir: Path, summary_path: Path) -> Path:
    output_paths = {
        key: path
        for key, value in row.output.items()
        if (path := _project_path(value)) is not None and path.exists()
    }
    manifest = build_run_manifest(
        run_dir=case_dir,
        input_path=_project_path(row.input),
        outputs=output_paths,
        request={
            "backend": row.requested_backend,
            "effective_backend": _benchmark_backend(row),
            "source_input": row.input,
        },
        route={
            "selected_backend": row.selected_backend,
            "route": row.route,
            "asset_kind": row.asset_kind,
            "parameter_profile": row.parameter_profile,
            "execution_profile": row.execution_profile,
        },
        runtime={
            "kind": "game-eval",
            "backend": _benchmark_backend(row),
            "selected_backend": row.selected_backend,
            "execution_backend": row.execution_backend,
            "elapsed_sec_client": row.timings.get("total_sec"),
        },
        report_path=summary_path,
        extra={
            "case_metadata": {
                "sample_id": row.sample_id,
                "category": row.category,
                "case": row.case,
            }
        },
    )
    return write_run_manifest(case_dir / "manifest.json", manifest)


def _write_batch_manifest(
    *,
    out_root: Path,
    aggregate: dict[str, Any],
    summary_path: Path,
    manifest_path: Path,
) -> Path:
    case_manifests = [
        row.get("artifact_manifest")
        for row in aggregate.get("runs", [])
        if isinstance(row, dict) and isinstance(row.get("artifact_manifest"), str)
    ]
    backends = sorted(
        {
            str(row.get("backend"))
            for row in aggregate.get("runs", [])
            if isinstance(row, dict) and row.get("backend")
        }
    )
    manifest = build_run_manifest(
        run_dir=out_root,
        outputs={"summary": summary_path},
        request={
            "backend": aggregate.get("backend"),
            "case_count": aggregate.get("case_count"),
            "manifest": _rel(manifest_path),
            "sample_filter": aggregate.get("sample_filter"),
            "category_filter": aggregate.get("category_filter"),
        },
        route={},
        runtime={
            "kind": "game-eval",
            "backend": aggregate.get("backend"),
            "backends": backends,
            "ok_count": aggregate.get("ok_count"),
        },
        report_path=summary_path,
        extra={"case_manifests": case_manifests},
    )
    return write_run_manifest(out_root / "manifest.json", manifest)


def _write_case_and_aggregate(
    *,
    row: CaseBenchmark,
    rows_by_case: dict[str, CaseBenchmark],
    cases: list[dict[str, Any]],
    out_root: Path,
    args: argparse.Namespace,
    batch_start: float,
) -> dict[str, Any]:
    case_dir = out_root / row.case
    case_dir.mkdir(parents=True, exist_ok=True)
    case_summary_path = case_dir / "summary.json"
    case_manifest = _write_case_manifest(row=row, case_dir=case_dir, summary_path=case_summary_path)
    case_payload = _row_payload(row, artifact_manifest=case_manifest)
    case_summary_path.write_text(json.dumps(case_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows_by_case[row.case] = row
    ordered_rows = [rows_by_case[_case_key(case)] for case in cases if _case_key(case) in rows_by_case]
    fixed_backend = str(getattr(args, "fixed_execution_backend", "") or "")
    aggregate = {
        "batch": _rel(out_root),
        "backend": fixed_backend or "direct-worker-benchmark",
        "manifest": _rel(args.manifest),
        "case_count": len(cases),
        "run_count": len(cases),
        "ok_count": sum(1 for item in ordered_rows if item.status == "ok"),
        "write_images": bool(args.write_images),
        "fixed_execution_backend": fixed_backend or None,
        "sample_filter": args.sample_id,
        "category_filter": args.category,
        "warmup_sample_id": args.warmup_sample_id or None,
        "compare_summary": _rel(args.compare_summary) if args.compare_summary else None,
        "cpu_workers": int(getattr(args, "cpu_workers", 1)),
        "cpu_parallel_backends": ["comfy-pymatting-known-b"],
        "batch_elapsed_sec": time.perf_counter() - batch_start,
        "runtime": _runtime_info(),
        "timing_summary": _summarize(ordered_rows),
        "runs": [_row_payload(item, artifact_manifest=out_root / item.case / "manifest.json") for item in ordered_rows],
    }
    batch_summary_path = out_root / "summary.json"
    batch_manifest = _write_batch_manifest(
        out_root=out_root,
        aggregate=aggregate,
        summary_path=batch_summary_path,
        manifest_path=args.manifest,
    )
    aggregate["artifact_manifest"] = _rel(batch_manifest)
    batch_summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def _error_row(
    *,
    case: dict[str, Any],
    case_key: str,
    input_path: Path,
    timings: dict[str, float],
    compare_rows: dict[str, dict[str, Any]],
    error: str,
    image_size: list[int] | None = None,
) -> CaseBenchmark:
    total = timings.get("total_sec", sum(float(v) for v in timings.values() if isinstance(v, (int, float))))
    timings["total_sec"] = float(total)
    return CaseBenchmark(
        status="error",
        case=case_key,
        sample_id=str(case.get("sample_id", "")),
        input=_rel(input_path),
        category=str(case.get("category", "")),
        image_size=image_size or list(case.get("image_size", [])),
        requested_backend="direct-auto",
        selected_backend=None,
        execution_backend=None,
        route=None,
        asset_kind=None,
        parameter_profile=None,
        execution_profile=None,
        timings=timings,
        output={},
        compare=_compare_payload(case_key, compare_rows, timings["total_sec"]),
        error=error,
        debug=None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest)
    cases = _selected_cases(args, manifest)
    compare_rows = _load_comfy_compare(args.compare_summary)
    out_root = args.out_dir or _next_out_dir("direct_worker_benchmark")
    out_root.mkdir(parents=True, exist_ok=True)
    ck_factory = DirectCorridorKeyClientFactory()
    rows_by_case: dict[str, CaseBenchmark] = {}
    aggregate: dict[str, Any] = {}
    batch_start = time.perf_counter()
    cpu_workers = max(1, int(getattr(args, "cpu_workers", 1)))

    if args.warmup_sample_id:
        warmup = next(
            (case for case in manifest["cases"] if isinstance(case, dict) and str(case.get("sample_id")) == args.warmup_sample_id),
            None,
        )
        if warmup is None:
            raise ValueError(f"--warmup-sample-id {args.warmup_sample_id!r} not found in manifest")
        warmup_path = PROJECT_ROOT / str(warmup["input"])
        print(f"[warmup] {args.warmup_sample_id} {warmup_path}", flush=True)
        warmup_rgb = _load_rgb(warmup_path)
        warmup_start = time.perf_counter()
        _direct_matte(
            warmup_rgb,
            shadow_mode=args.shadow_mode,
            corridorkey_screen_mode=args.corridorkey_screen_mode,
            corridorkey_preset=args.corridorkey_preset,
            corridorkey_hard_ui_hint_mode=args.corridorkey_hard_ui_hint_mode,
            fallback_bg_color=tuple(args.fallback_bg_color),
            ck_factory=ck_factory,
        )
        print(f"  warmup_total={time.perf_counter() - warmup_start:.3f}s", flush=True)

    executor: ProcessPoolExecutor | None = None
    futures: dict[Any, PreparedCase] = {}
    if cpu_workers > 1:
        executor = ProcessPoolExecutor(max_workers=cpu_workers)
        print(f"[batch] CPU parallel workers enabled: {cpu_workers} for comfy-pymatting-known-b", flush=True)

    try:
        for index, case in enumerate(cases, start=1):
            case_key = _case_key(case)
            input_path = PROJECT_ROOT / str(case["input"])
            print(f"[{index}/{len(cases)}] {case_key}", flush=True)
            try:
                prepared, rgb = _classify_prepared_case(index, case, args)
            except Exception as exc:
                row = _error_row(
                    case=case,
                    case_key=case_key,
                    input_path=input_path,
                    timings={},
                    compare_rows=compare_rows,
                    error=str(exc),
                )
                print(f"  ERROR {exc}", flush=True)
                aggregate = _write_case_and_aggregate(
                    row=row,
                    rows_by_case=rows_by_case,
                    cases=cases,
                    out_root=out_root,
                    args=args,
                    batch_start=batch_start,
                )
                continue
            mismatch = _fixed_backend_mismatch(prepared, args)
            if mismatch is not None:
                row = _error_row(
                    case=case,
                    case_key=case_key,
                    input_path=input_path,
                    timings=dict(prepared.route_timings),
                    compare_rows=compare_rows,
                    error=mismatch,
                    image_size=prepared.image_size,
                )
                row.requested_backend = _requested_backend(args)
                row.selected_backend = str(getattr(prepared.decision, "backend", "") or "")
                row.execution_backend = _execution_backend_from_decision(prepared.decision)
                row.route = str(getattr(prepared.decision, "route", "") or "")
                row.asset_kind = str(getattr(prepared.decision, "asset_kind", "") or "")
                row.execution_profile = str(getattr(prepared.decision, "params", {}).get("execution_profile", "") or "")
                print(f"  ERROR {mismatch}", flush=True)
                aggregate = _write_case_and_aggregate(
                    row=row,
                    rows_by_case=rows_by_case,
                    cases=cases,
                    out_root=out_root,
                    args=args,
                    batch_start=batch_start,
                )
                continue

            if executor is not None and _cpu_parallel_backend(prepared.decision):
                payload = {
                    "input_path": str(prepared.input_path),
                    "decision": prepared.decision,
                    "route_timings": prepared.route_timings,
                    "shadow_mode": args.shadow_mode,
                    "corridorkey_hard_ui_hint_mode": args.corridorkey_hard_ui_hint_mode,
                    "fallback_bg_color": tuple(args.fallback_bg_color),
                    "write_images": bool(args.write_images),
                    "include_debug": bool(args.include_debug),
                }
                futures[executor.submit(_cpu_case_worker, payload)] = prepared
                print(
                    "  queued direct-pymatting-known-b "
                    f"route={prepared.route_timings.get('route_sec', 0.0):.3f}s",
                    flush=True,
                )
                continue

            try:
                row = _run_prepared_case_in_main(
                    prepared,
                    rgb,
                    args=args,
                    ck_factory=ck_factory,
                    compare_rows=compare_rows,
                    out_root=out_root,
                )
                print(
                    "  "
                    f"{row.execution_backend} route_backend={row.selected_backend} total={row.timings['total_sec']:.3f}s "
                    f"route={row.timings.get('route_sec', 0.0):.3f}s "
                    f"backend={row.timings.get('backend_sec', 0.0):.3f}s",
                    flush=True,
                )
            except Exception as exc:
                row = _error_row(
                    case=case,
                    case_key=case_key,
                    input_path=input_path,
                    timings=dict(prepared.route_timings),
                    compare_rows=compare_rows,
                    error=str(exc),
                    image_size=prepared.image_size,
                )
                print(f"  ERROR {exc}", flush=True)
            aggregate = _write_case_and_aggregate(
                row=row,
                rows_by_case=rows_by_case,
                cases=cases,
                out_root=out_root,
                args=args,
                batch_start=batch_start,
            )

        for future in as_completed(futures):
            prepared = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = {
                    "status": "error",
                    "timings": dict(prepared.route_timings),
                    "metadata": {},
                    "debug": None,
                    "output": {},
                    "error": str(exc),
                }
            row = _row_from_worker_payload(
                prepared,
                payload,
                args=args,
                compare_rows=compare_rows,
                out_root=out_root,
            )
            if row.status == "ok":
                print(
                    "  "
                    f"{row.execution_backend} route_backend={row.selected_backend} total={row.timings['total_sec']:.3f}s "
                    f"route={row.timings.get('route_sec', 0.0):.3f}s "
                    f"backend={row.timings.get('backend_sec', 0.0):.3f}s "
                    f"worker={row.timings.get('worker_elapsed_sec', 0.0):.3f}s",
                    flush=True,
                )
            else:
                print(f"  ERROR {row.case} {row.error}", flush=True)
            aggregate = _write_case_and_aggregate(
                row=row,
                rows_by_case=rows_by_case,
                cases=cases,
                out_root=out_root,
                args=args,
                batch_start=batch_start,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not aggregate:
        fixed_backend = str(getattr(args, "fixed_execution_backend", "") or "")
        aggregate = {
            "batch": _rel(out_root),
            "backend": fixed_backend or "direct-worker-benchmark",
            "manifest": _rel(args.manifest),
            "case_count": 0,
            "run_count": 0,
            "ok_count": 0,
            "write_images": bool(args.write_images),
            "fixed_execution_backend": fixed_backend or None,
            "warmup_sample_id": args.warmup_sample_id or None,
            "compare_summary": _rel(args.compare_summary) if args.compare_summary else None,
            "cpu_workers": cpu_workers,
            "cpu_parallel_backends": ["comfy-pymatting-known-b"],
            "batch_elapsed_sec": time.perf_counter() - batch_start,
            "runtime": _runtime_info(),
            "timing_summary": _summarize([]),
            "runs": [],
        }
        batch_summary_path = out_root / "summary.json"
        batch_manifest = _write_batch_manifest(
            out_root=out_root,
            aggregate=aggregate,
            summary_path=batch_summary_path,
            manifest_path=args.manifest,
        )
        aggregate["artifact_manifest"] = _rel(batch_manifest)
        batch_summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--warmup-sample-id", default="", help="Optional sample id to execute once before measured cases.")
    parser.add_argument("--all", action="store_true", help="Run every manifest case instead of the representative sample-id list.")
    parser.add_argument("--category", default="", help="Optional comma-separated category filter.")
    parser.add_argument("--compare-summary", type=Path, default=None)
    parser.add_argument(
        "--fixed-execution-backend",
        choices=("direct-pymatting-known-b", "direct-corridorkey", "direct-passthrough"),
        default="",
        help="Require every selected case to execute on this direct backend; mismatches are recorded as errors.",
    )
    parser.add_argument("--shadow-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--corridorkey-screen-mode", choices=("auto", "green", "blue"), default="auto")
    parser.add_argument("--corridorkey-preset", choices=("auto", "detail_safe", "spill_safe", "manual"), default="auto")
    parser.add_argument(
        "--corridorkey-hard-ui-hint-mode",
        choices=(
            "all_white",
            "bbox_2px",
            "boundary_2px",
            "boundary_2px_shadow_safe",
            "boundary_2px_shadow_safe_edge_floor",
            "translucent_button",
        ),
        default="bbox_2px",
    )
    parser.add_argument("--fallback-bg-color", type=int, nargs=3, default=(0, 200, 0), metavar=("R", "G", "B"))
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Process workers for CPU-owned direct routes; use 1 to disable CPU batch parallelism.",
    )
    parser.add_argument("--no-debug", action="store_false", dest="include_debug", help="Do not include per-case debug summaries in JSON.")
    parser.add_argument("--no-write-images", action="store_false", dest="write_images")
    parser.set_defaults(include_debug=True, write_images=True)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "batch": summary["batch"],
                "case_count": summary["case_count"],
                "ok_count": summary["ok_count"],
                "summary": str((PROJECT_ROOT / summary["batch"] / "summary.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
