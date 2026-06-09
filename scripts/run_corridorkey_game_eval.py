#!/usr/bin/env python3
"""Run a remote matting backend over manifest-selected game-eval inputs."""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ermbg import matte_image
from ermbg.analyze import analyze_candidates
from ermbg.artifacts import build_run_manifest, route_from_response, runtime_from_response, write_run_manifest
from ermbg.comfy import DEFAULT_COMFY_URL
from ermbg.direct_worker_client import DEFAULT_DIRECT_WORKER_URL, matte_image_direct_worker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json"
EVAL_BACKENDS = ("auto", "direct-worker")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()) if value.size else 0.0,
            "max": float(value.max()) if value.size else 0.0,
            "mean": float(value.mean()) if value.size else 0.0,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _decode_png_data_url_gray(data_url: str) -> np.ndarray:
    marker = "base64,"
    if not data_url.startswith("data:image/png;") or marker not in data_url:
        raise ValueError("Analyze preview asset must be a PNG data URL")
    data = base64.b64decode(data_url.split(marker, 1)[1])
    return np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.uint8)


def _semantic_candidate_record(analysis_payload: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    candidates = analysis_payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("id") or "") == candidate_id:
            return candidate
    return None


def _selected_route_payload(analysis_payload: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    route_candidate_id = None
    if isinstance(candidate, dict):
        route_candidate_id = candidate.get("route_candidate_id")
        decision = candidate.get("decision")
        if route_candidate_id is None and isinstance(decision, dict):
            route_candidate_id = decision.get("route_candidate_id")
    if route_candidate_id is None:
        route_candidate_id = analysis_payload.get("default_route_candidate_id")
    route_candidates = analysis_payload.get("route_candidates")
    if isinstance(route_candidates, list) and route_candidate_id is not None:
        for route_candidate in route_candidates:
            if isinstance(route_candidate, dict) and route_candidate.get("id") == route_candidate_id:
                return dict(route_candidate)
    route = analysis_payload.get("route")
    return dict(route) if isinstance(route, dict) else {}


def _candidate_explicit_trimap(
    analysis_payload: dict[str, Any],
    candidate: dict[str, Any] | None,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    route = _selected_route_payload(analysis_payload, candidate)
    if str(route.get("algorithm") or route.get("backend") or "") != "pymatting_known_b":
        return None
    if not isinstance(candidate, dict):
        return None
    preview = candidate.get("preview")
    assets = preview.get("assets") if isinstance(preview, dict) else None
    asset_ref = str(assets.get("trimap") or "") if isinstance(assets, dict) else ""
    if not asset_ref:
        asset_ref = f"candidate:{candidate.get('id')}:trimap"
    preview_assets = analysis_payload.get("preview_assets")
    asset = preview_assets.get(asset_ref) if isinstance(preview_assets, dict) else None
    if not isinstance(asset, dict):
        return None
    if asset.get("execution_role") not in {None, "pymatting_explicit_trimap"}:
        return None
    data_url = asset.get("data_url")
    if not isinstance(data_url, str):
        return None
    trimap = _decode_png_data_url_gray(data_url)
    if trimap.shape != image_shape:
        raise ValueError("Analyze candidate trimap shape does not match input image")
    out = np.full(trimap.shape, 128, dtype=np.uint8)
    out[trimap < 64] = 0
    out[trimap > 191] = 255
    return out


def _analyze_default_execution_contract(
    input_path: Path,
    *,
    screen_mode: str,
    preset: str,
    fallback_background_color: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray | None]:
    image_srgb = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    analysis_payload = analyze_candidates(
        image_srgb,
        screen_mode=screen_mode,
        preset=preset,
        fallback_background_color=fallback_background_color,
    ).to_dict()
    selected_candidate_id = str(analysis_payload.get("default_candidate_id") or "auto_default")
    candidate = _semantic_candidate_record(analysis_payload, selected_candidate_id)
    route_decision = _selected_route_payload(analysis_payload, candidate)
    semantic_decision = dict(candidate.get("decision") or {}) if isinstance(candidate, dict) else {"policy": "auto_default"}
    explicit_trimap = _candidate_explicit_trimap(analysis_payload, candidate, image_srgb.shape[:2])
    contract = _analysis_contract_summary(
        analysis_payload,
        selected_candidate_id=selected_candidate_id,
        selected_candidate=candidate,
        route_decision=route_decision,
        explicit_trimap=explicit_trimap,
    )
    return route_decision, semantic_decision, contract, explicit_trimap


def _analysis_contract_summary(
    analysis_payload: dict[str, Any],
    *,
    selected_candidate_id: str,
    selected_candidate: dict[str, Any] | None,
    route_decision: dict[str, Any],
    explicit_trimap: np.ndarray | None,
) -> dict[str, Any]:
    route_candidates = [
        {
            "id": candidate.get("id"),
            "algorithm": candidate.get("algorithm"),
            "default": candidate.get("default"),
            "confidence": candidate.get("confidence"),
            "parameter_profile": candidate.get("parameter_profile"),
        }
        for candidate in analysis_payload.get("route_candidates", [])
        if isinstance(candidate, dict)
    ]
    semantic_candidates = [
        {
            "id": candidate.get("id"),
            "label": candidate.get("label"),
            "route_candidate_id": candidate.get("route_candidate_id"),
            "default": candidate.get("default"),
            "confidence": candidate.get("confidence"),
            "intent": candidate.get("intent"),
            "risk_level": candidate.get("risk_level"),
        }
        for candidate in analysis_payload.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    trimap_stats = None
    if explicit_trimap is not None:
        values, counts = np.unique(explicit_trimap, return_counts=True)
        trimap_stats = {str(int(value)): int(count) for value, count in zip(values, counts, strict=False)}
    return {
        "enabled": True,
        "analysis_id": analysis_payload.get("analysis_id"),
        "status": analysis_payload.get("status"),
        "default_route_candidate_id": analysis_payload.get("default_route_candidate_id"),
        "default_candidate_id": analysis_payload.get("default_candidate_id"),
        "selected_candidate_id": selected_candidate_id,
        "selected_route_candidate_id": route_decision.get("id") or route_decision.get("route_candidate_id"),
        "selected_algorithm": route_decision.get("algorithm") or route_decision.get("backend"),
        "selected_candidate_label": selected_candidate.get("label") if isinstance(selected_candidate, dict) else None,
        "selected_semantic_decision": selected_candidate.get("decision") if isinstance(selected_candidate, dict) else None,
        "explicit_trimap_states": trimap_stats,
        "route_candidates": route_candidates,
        "semantic_candidates": semantic_candidates,
    }


def _debug_timings(debug: dict[str, Any]) -> dict[str, float]:
    timings = debug.get("timings") if isinstance(debug, dict) else None
    if not isinstance(timings, dict):
        return {}
    keys = (
        "node_total_sec",
        "remote_total_sec",
        "remote_wait_sec",
        "remote_upload_sec",
        "remote_queue_sec",
        "corridorkey_refine_sec",
        "total_sec",
        "server_elapsed_sec",
        "route_sec",
        "backend_sec",
        "decode_sec",
        "worker_elapsed_sec",
    )
    out: dict[str, float] = {}
    for key in keys:
        value = timings.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault(str(row.get("backend", "unknown")), []).append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        elapsed = [float(item.get("elapsed_sec_client", 0.0)) for item in items]
        timing_rows = [item.get("timings", {}) for item in items if isinstance(item.get("timings"), dict)]
        metric_names = sorted({name for timing in timing_rows for name in timing})
        metrics = {}
        for name in metric_names:
            values = [float(timing[name]) for timing in timing_rows if isinstance(timing.get(name), (int, float))]
            if values:
                metrics[name] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return {
            "count": len(items),
            "elapsed_sec_client": {
                "avg": sum(elapsed) / len(elapsed) if elapsed else 0.0,
                "min": min(elapsed) if elapsed else 0.0,
                "max": max(elapsed) if elapsed else 0.0,
            },
            "timings": metrics,
        }

    return {
        "overall": summarize(ok_rows) if ok_rows else {"count": 0},
        "by_backend": {backend: summarize(items) for backend, items in sorted(groups.items())},
    }


def _rel(path: Path) -> str:
    # Emit POSIX separators so manifest path fields stay portable across OSes
    # (Windows str(Path) would otherwise leak backslashes into summary.json).
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return payload


def _load_cases(manifest_path: Path) -> list[dict[str, Any]]:
    payload = _load_manifest(manifest_path)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a cases list")
    return [case for case in cases if isinstance(case, dict)]


def _next_versioned_out_dir(prefix: str) -> Path:
    out_root = PROJECT_ROOT / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d")
    base = f"{prefix}_{date}"
    version = 1
    while (out_root / f"{base}_v{version:03d}").exists():
        version += 1
    return out_root / f"{base}_v{version:03d}"


def _backend_slug(backend: str) -> str:
    return backend.removeprefix("comfy-").replace("-", "_")


def _case_input_screen(case: dict[str, Any]) -> str:
    input_value = str(case.get("input", ""))
    for screen in ("green", "blue", "white"):
        if isinstance(case.get(screen), str) and str(case[screen]) == input_value:
            return screen
    screen = case.get("screen")
    if isinstance(screen, str) and screen:
        return screen
    path_screen = Path(input_value).stem
    return path_screen if path_screen else "input"


def _case_background(
    manifest: dict[str, Any],
    case: dict[str, Any],
    screen: str,
) -> tuple[int, int, int]:
    backgrounds = case.get("backgrounds")
    if isinstance(backgrounds, dict):
        value = backgrounds.get(screen)
        if isinstance(value, list) and len(value) == 3:
            return tuple(int(c) for c in value)
    manifest_backgrounds = manifest.get("backgrounds")
    if isinstance(manifest_backgrounds, dict):
        value = manifest_backgrounds.get(screen)
        if isinstance(value, list) and len(value) == 3:
            return tuple(int(c) for c in value)
    if screen == "blue":
        return (0, 0, 200)
    if screen == "white":
        return (255, 255, 255)
    return (0, 200, 0)


def _effective_backend(requested_backend: str, result: Any) -> str:
    if requested_backend == "auto":
        auto_route = getattr(result, "debug", {}).get("auto_route")
        if isinstance(auto_route, dict):
            selected = auto_route.get("selected_backend")
            if isinstance(selected, str) and selected:
                return selected
    return requested_backend


def _copy_backend_outputs(case_dir: Path, stem: str) -> None:
    for src_name, dst_name in [
        (f"{stem}_rgba.png", "rgba.png"),
        (f"{stem}_alpha.png", "alpha.png"),
        (f"{stem}_foreground.png", "foreground.png"),
        (f"{stem}_shadow.png", "shadow.png"),
        (f"{stem}_shadow_layer.png", "shadow_layer.png"),
        (f"{stem}_shadow_physical.png", "shadow_physical.png"),
        (f"{stem}_corridorkey_subject_rgba.png", "corridorkey_subject_rgba.png"),
        (f"{stem}_corridorkey_subject_alpha.png", "corridorkey_subject_alpha.png"),
        (f"{stem}_corridorkey_hint.png", "corridorkey_hint.png"),
        (f"{stem}_corridorkey_raw_alpha.png", "corridorkey_raw_alpha.png"),
        (f"{stem}_trimap.png", "trimap.png"),
    ]:
        src = case_dir / src_name
        if src.exists():
            shutil.copy2(src, case_dir / dst_name)


def _write_direct_worker_outputs(case_dir: Path, result: Any) -> None:
    rgba = np.asarray(result.rgba, dtype=np.uint8)
    alpha = (np.clip(np.asarray(result.alpha, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    foreground = np.asarray(result.foreground_srgb, dtype=np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(case_dir / "rgba.png")
    Image.fromarray(alpha, mode="L").save(case_dir / "alpha.png")
    Image.fromarray(foreground, mode="RGB").save(case_dir / "foreground.png")
    debug = getattr(result, "debug", {})
    direct_worker = debug.get("direct_worker") if isinstance(debug, dict) else None
    encoded_trimap = direct_worker.get("trimap_png_base64") if isinstance(direct_worker, dict) else None
    if isinstance(encoded_trimap, str):
        data = base64.b64decode(encoded_trimap)
        Image.open(io.BytesIO(data)).convert("L").save(case_dir / "trimap.png")
    encoded_hint = direct_worker.get("corridorkey_hint_png_base64") if isinstance(direct_worker, dict) else None
    if isinstance(encoded_hint, str):
        data = base64.b64decode(encoded_hint)
        Image.open(io.BytesIO(data)).convert("L").save(case_dir / "corridorkey_hint.png")


def _case_outputs(case_dir: Path) -> dict[str, str]:
    names = {
        "input": "input.png",
        "rgba": "rgba.png",
        "alpha": "alpha.png",
        "foreground": "foreground.png",
        "shadow": "shadow.png",
        "shadow_layer": "shadow_layer.png",
        "shadow_physical": "shadow_physical.png",
        "corridorkey_subject_rgba": "corridorkey_subject_rgba.png",
        "corridorkey_subject_alpha": "corridorkey_subject_alpha.png",
        "trimap": "trimap.png",
        "analyze_candidate_trimap": "analyze_candidate_trimap.png",
        "hint": "corridorkey_hint.png",
        "raw_alpha": "corridorkey_raw_alpha.png",
        "contact_sheet": "contact_sheet.png",
    }
    outputs = {}
    for key, name in names.items():
        path = case_dir / name
        if path.exists():
            outputs[key] = _rel(path)
    return outputs


def _write_case_manifest(
    *,
    case_dir: Path,
    input_path: Path,
    result: Any,
    requested_backend: str,
    effective_backend: str,
    summary_path: Path,
    case_metadata: dict[str, Any],
    elapsed_sec_client: float,
) -> Path:
    outputs = {
        "rgba": case_dir / "rgba.png",
        "alpha": case_dir / "alpha.png",
        "foreground": case_dir / "foreground.png",
        "trimap": case_dir / "trimap.png",
        "analyze_candidate_trimap": case_dir / "analyze_candidate_trimap.png",
        "hint": case_dir / "corridorkey_hint.png",
        "raw_alpha": case_dir / "corridorkey_raw_alpha.png",
        "shadow": case_dir / "shadow.png",
        "shadow_layer": case_dir / "shadow_layer.png",
        "shadow_physical": case_dir / "shadow_physical.png",
        "contact_sheet": case_dir / "contact_sheet.png",
    }
    manifest = build_run_manifest(
        run_dir=case_dir,
        input_path=case_dir / "input.png",
        outputs={key: value for key, value in outputs.items() if value.exists()},
        request={
            "backend": requested_backend,
            "effective_backend": effective_backend,
            "source_input": _rel(input_path),
        },
        route=route_from_response(result),
        runtime={
            **runtime_from_response(result, requested_backend=requested_backend),
            "backend": effective_backend,
            "elapsed_sec_client": elapsed_sec_client,
        },
        report_path=summary_path,
        result=result,
        extra={"case_metadata": case_metadata},
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
    manifest = build_run_manifest(
        run_dir=out_root,
        outputs={
            "summary": summary_path,
        },
        request={
            "backend": aggregate.get("backend"),
            "case_count": aggregate.get("case_count"),
            "run_count": aggregate.get("run_count"),
            "manifest": _rel(manifest_path),
            "category_filter": aggregate.get("category_filter"),
            "eval_overrides": aggregate.get("eval_overrides", {}),
        },
        route={},
        runtime={
            "kind": "game-eval",
            "backend": aggregate.get("backend"),
            "ok_count": aggregate.get("ok_count"),
        },
        report_path=summary_path,
        extra={
            "case_manifests": case_manifests,
            "screens": aggregate.get("screens", []),
        },
    )
    return write_run_manifest(out_root / "manifest.json", manifest)


def _checker_bg(size: tuple[int, int], cell: int = 16) -> Image.Image:
    w, h = size
    yy, xx = np.indices((h, w))
    light = ((xx // cell + yy // cell) % 2) == 0
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[light] = (230, 230, 230)
    arr[~light] = (150, 150, 150)
    return Image.fromarray(arr, "RGB")


def _composite(rgba: Image.Image, bg: Image.Image) -> Image.Image:
    base = bg.convert("RGBA")
    base.alpha_composite(rgba)
    return base.convert("RGB")


def _write_contact_sheet(case_dir: Path) -> None:
    rgba = Image.open(case_dir / "rgba.png").convert("RGBA")
    alpha = Image.open(case_dir / "alpha.png").convert("L")
    foreground = Image.open(case_dir / "foreground.png").convert("RGB")
    original = Image.open(case_dir / "input.png").convert("RGB")
    views = [
        ("input", original),
        ("foreground", foreground),
        ("alpha", alpha.convert("RGB")),
        ("checker", _composite(rgba, _checker_bg(rgba.size))),
        ("white", _composite(rgba, Image.new("RGB", rgba.size, "white"))),
        ("black", _composite(rgba, Image.new("RGB", rgba.size, "black"))),
    ]

    max_thumb = 320
    pad = 10
    label_h = 20
    thumbs: list[Image.Image] = []
    for label, image in views:
        scaled = image.copy()
        scaled.thumbnail((max_thumb, max_thumb), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max_thumb, max_thumb + label_h), (245, 245, 245))
        canvas.paste(scaled, ((max_thumb - scaled.width) // 2, label_h + (max_thumb - scaled.height) // 2))
        ImageDraw.Draw(canvas).text((4, 3), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 3
    rows = 2
    sheet = Image.new(
        "RGB",
        (cols * max_thumb + (cols - 1) * pad, rows * (max_thumb + label_h) + (rows - 1) * pad),
        (255, 255, 255),
    )
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * (max_thumb + pad), (i // cols) * (max_thumb + label_h + pad)))
    sheet.save(case_dir / "contact_sheet.png")


def _coverage_metrics(input_path: Path, alpha_path: Path, background: tuple[int, int, int], threshold: float) -> dict[str, Any]:
    alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.uint8)
    image = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    bg = np.asarray(background, dtype=np.int16)
    expected_subject = np.linalg.norm(image.astype(np.int16) - bg, axis=2) > float(threshold)
    yy = np.indices(alpha.shape)[0]
    lower = expected_subject & (yy > alpha.shape[0] * 0.5)
    upper = expected_subject & ~lower

    def coverage(mask: np.ndarray) -> float:
        denom = int(mask.sum())
        return float(((alpha > 128) & mask).sum() / max(1, denom))

    return {
        "background": list(background),
        "expected_subject_distance_threshold": float(threshold),
        "alpha_nonzero_pixels": int((alpha > 8).sum()),
        "alpha_mean": float(alpha.mean() / 255.0),
        "expected_subject_pixels": int(expected_subject.sum()),
        "expected_subject_alpha_coverage_gt_128": coverage(expected_subject),
        "lower_expected_subject_alpha_coverage_gt_128": coverage(lower),
        "upper_expected_subject_alpha_coverage_gt_128": coverage(upper),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest)
    use_analyze_candidates = bool(getattr(args, "use_analyze_candidates", True))
    shadow_mode = str(getattr(args, "shadow_mode", "on"))
    cases = [case for case in manifest.get("cases", []) if isinstance(case, dict)]
    if args.sample_id:
        sample_ids = {item.strip() for item in args.sample_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("sample_id", "")) in sample_ids]
    if args.category:
        categories = {item.strip() for item in args.category.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("category", "")) in categories]
    cases = [case for case in cases if isinstance(case.get("input"), str)]

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    total = len(cases)
    for case_index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        sample_id = str(case.get("sample_id") or f"G{case_index:02d}")
        input_path = PROJECT_ROOT / str(case["input"])
        screen = _case_input_screen(case)
        background = _case_background(manifest, case, screen)
        case_dir = out_root / f"{sample_id}_{case_id}_{screen}"
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, case_dir / "input.png")
        print(f"[{case_index}/{total}] {sample_id}-{screen[:1].upper()} {case_id}", flush=True)
        start = time.perf_counter()
        result: Any | None = None
        analysis_contract: dict[str, Any] | None = None
        try:
            if args.backend in {"direct-worker", "auto"}:
                route_decision = None
                semantic_decision = None
                explicit_trimap = None
                if use_analyze_candidates:
                    route_decision, semantic_decision, analysis_contract, explicit_trimap = _analyze_default_execution_contract(
                        input_path,
                        screen_mode="auto",
                        preset=args.corridorkey_preset,
                        fallback_background_color=background,
                    )
                result = matte_image_direct_worker(
                    input_path,
                    direct_worker_url=args.direct_worker_url,
                    execution_backend="auto" if args.backend == "auto" else "direct-worker",
                    shadow_mode=shadow_mode,
                    corridorkey_preset=args.corridorkey_preset,
                    route_decision=route_decision,
                    semantic_decision=semantic_decision,
                    pymatting_explicit_trimap=explicit_trimap,
                    fallback_bg_color=background,
                )
                _write_direct_worker_outputs(case_dir, result)
                if explicit_trimap is not None:
                    Image.fromarray(explicit_trimap, mode="L").save(case_dir / "analyze_candidate_trimap.png")
            else:
                result = matte_image(
                    input_path,
                    backend=args.backend,
                    output_dir=case_dir,
                    qa=False,
                    comfy_url=args.comfy_url,
                    corridorkey_preset=args.corridorkey_preset,
                    corridorkey_auto_mask=False,
                )
            elapsed = time.perf_counter() - start
            effective_backend = _effective_backend(args.backend, result)
            stem = input_path.stem
            if args.backend not in {"direct-worker", "auto"}:
                _copy_backend_outputs(case_dir, stem)
            _write_contact_sheet(case_dir)
            metrics = _coverage_metrics(case_dir / "input.png", case_dir / "alpha.png", background, args.subject_threshold)
            summary = {
                "status": "ok",
                "case": f"{sample_id}_{case_id}_{screen}",
                "backend": effective_backend,
                "requested_backend": args.backend,
                "input": _rel(input_path),
                "sample_screen": screen,
                "elapsed_sec_client": elapsed,
                "timings": _debug_timings(result.debug),
                "outputs": _case_outputs(case_dir),
                "remote_debug": _json_safe(result.debug),
                "analysis_contract": _json_safe(analysis_contract) if analysis_contract is not None else {"enabled": False},
                "quality_metrics": metrics,
                "case_metadata": case,
            }
            print(
                f"  alpha_mean={metrics['alpha_mean']:.3f} "
                f"coverage={metrics['expected_subject_alpha_coverage_gt_128']:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            summary = {
                "status": "error",
                "case": f"{sample_id}_{case_id}_{screen}",
                "backend": args.backend,
                "requested_backend": args.backend,
                "input": _rel(input_path),
                "sample_screen": screen,
                "error": str(exc),
                "case_metadata": case,
            }
            print(f"  ERROR {exc}", flush=True)
        summary_path = case_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if summary.get("status") == "ok" and result is not None:
            manifest_path = _write_case_manifest(
                case_dir=case_dir,
                input_path=input_path,
                result=result,
                requested_backend=args.backend,
                effective_backend=str(summary.get("backend", args.backend)),
                summary_path=summary_path,
                case_metadata=case,
                elapsed_sec_client=float(summary.get("elapsed_sec_client", 0.0)),
            )
            summary["artifact_manifest"] = _rel(manifest_path)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        runs.append(summary)
        aggregate = {
            "backend": args.backend,
            "batch": _rel(out_root),
            "case_count": len(cases),
            "run_count": total,
            "ok_count": sum(1 for row in runs if row.get("status") == "ok"),
            "screens": sorted({str(row.get("sample_screen", "")) for row in runs if row.get("sample_screen")}),
            "category_filter": args.category,
            "eval_overrides": {
                "corridorkey_preset": args.corridorkey_preset,
                "corridorkey_auto_mask": False,
                "shadow_mode": shadow_mode,
                "use_analyze_candidates": use_analyze_candidates,
            },
            "timing_summary": _timing_summary(runs),
            "runs": runs,
        }
        batch_summary_path = out_root / "summary.json"
        batch_summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        batch_manifest_path = _write_batch_manifest(
            out_root=out_root,
            aggregate=aggregate,
            summary_path=batch_summary_path,
            manifest_path=args.manifest,
        )
        aggregate["artifact_manifest"] = _rel(batch_manifest_path)
        batch_summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output batch directory. Default: out/<backend>_game_input_<YYYYMMDD>_vNNN.",
    )
    parser.add_argument("--backend", default="auto", choices=EVAL_BACKENDS)
    parser.add_argument("--sample-id", default="", help="Comma-separated sample ids, e.g. B001,I011,C004")
    parser.add_argument("--category", default="", help="Comma-separated manifest categories, e.g. button,icon")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--direct-worker-url", default=DEFAULT_DIRECT_WORKER_URL)
    parser.add_argument("--subject-threshold", type=float, default=35.0)
    parser.add_argument("--corridorkey-preset", default="auto", choices=("auto", "detail_safe", "spill_safe", "manual"))
    parser.add_argument("--shadow-mode", default="on", choices=("auto", "on", "off"))
    parser.add_argument(
        "--use-analyze-candidates",
        dest="use_analyze_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For direct-worker/auto runs, execute the Analyze default route/semantic candidate "
            "and send its explicit trimap to the worker when available."
        ),
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = _next_versioned_out_dir(f"{_backend_slug(args.backend)}_game_input")
    summary = run(args)
    print(
        json.dumps(
            {
                "batch": summary["batch"],
                "backend": summary["backend"],
                "run_count": summary["run_count"],
                "ok_count": summary["ok_count"],
                "summary": str((PROJECT_ROOT / summary["batch"] / "summary.json") if not str(summary["batch"]).startswith("/") else Path(summary["batch"]) / "summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
