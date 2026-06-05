#!/usr/bin/env python3
"""Capture the Stage 0 Web/API/Direct Worker baseline.

This script intentionally does not change ERMBG behavior. It runs a small,
mixed smoke set through the current Web HTTP API and writes standard
``ermbg.run.v1`` manifests under ``out/``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw

from ermbg.artifacts import build_run_manifest, json_safe, write_run_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_MANIFEST = PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json"


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


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _write_png_from_data_url(data_url: str, path: Path) -> None:
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    path.write_bytes(base64.b64decode(encoded))


def _strip_image_payloads(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_png_base64") and isinstance(item, str):
                out[key] = {"omitted_base64_chars": len(item)}
            elif key in {"rgba", "alpha", "foreground", "mask"} and isinstance(item, str) and item.startswith("data:image/"):
                out[key] = {"omitted_data_url_chars": len(item)}
            else:
                out[key] = _strip_image_payloads(item)
        return out
    if isinstance(value, list):
        return [_strip_image_payloads(item) for item in value]
    return value


def _route_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    auto_route = debug.get("auto_route") if isinstance(debug.get("auto_route"), dict) else {}
    direct = debug.get("direct_worker") if isinstance(debug.get("direct_worker"), dict) else {}
    return {
        "algorithm": payload.get("algorithm") or direct.get("algorithm") or auto_route.get("algorithm"),
        "route": payload.get("route") or direct.get("route") or auto_route.get("route"),
        "asset_kind": payload.get("asset_kind") or direct.get("asset_kind") or auto_route.get("asset_kind"),
        "parameter_profile": payload.get("parameter_profile")
        or direct.get("parameter_profile")
        or auto_route.get("parameter_profile"),
        "execution_profile": payload.get("execution_profile")
        or direct.get("execution_profile")
        or auto_route.get("execution_profile"),
    }


def _runtime_from_payload(payload: dict[str, Any], *, requested_backend: str) -> dict[str, Any]:
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    direct = debug.get("direct_worker") if isinstance(debug.get("direct_worker"), dict) else {}
    return {
        "kind": "direct-worker" if direct else "web",
        "requested_backend": requested_backend,
        "backend": payload.get("backend"),
        "strategy": payload.get("strategy"),
        "execution_backend": payload.get("execution_backend") or direct.get("execution_backend"),
        "execution_server_url": payload.get("execution_server_url")
        or debug.get("execution_server_url")
        or debug.get("web_direct_worker_url"),
        "server_elapsed_sec": payload.get("server_elapsed_sec") or direct.get("server_elapsed_sec"),
        "server_fallback_chain": debug.get("server_fallback_chain"),
    }


def _selected_candidate(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("selected") is True:
            return candidate
    return candidates[0] if candidates and isinstance(candidates[0], dict) else None


def _checker_image(size: int = 96, tile: int = 12) -> Image.Image:
    yy, xx = np.indices((size, size))
    parity = ((xx // tile + yy // tile) & 1).astype(bool)
    arr = np.where(
        parity[..., None],
        np.array([254, 254, 254], dtype=np.uint8),
        np.array([243, 243, 243], dtype=np.uint8),
    )
    arr[34:62, 28:68] = [120, 60, 210]
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _clean_rgba_image() -> Image.Image:
    image = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((14, 14, 66, 66), fill=(230, 40, 40, 255))
    draw.rectangle((38, 4, 48, 40), fill=(255, 220, 40, 210))
    return image


def _enclosed_near_b_ring() -> Image.Image:
    h, w = 96, 96
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    img[(r <= 30) & (r >= 12)] = (230, 0, 0)
    return Image.fromarray(img, mode="RGB")


def _unknown_fallback_image() -> Image.Image:
    rng = np.random.default_rng(42)
    bg = rng.integers(35, 220, size=(96, 96, 3), dtype=np.uint8)
    image = bg.astype(np.int16)
    image[28:70, 26:74] = np.array([35, 40, 220], dtype=np.int16)
    image[40:56, 38:62] = np.array([250, 180, 40], dtype=np.int16)
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def _synthetic_cases(out_root: Path) -> list[dict[str, Any]]:
    inputs_dir = out_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("clean_rgba_passthrough", "clean RGBA passthrough", _clean_rgba_image(), {}),
        (
            "checkerboard_background",
            "fake transparent checkerboard background",
            _checker_image(),
            {"remove_checkerboard": "true"},
        ),
        ("enclosed_near_b_ring", "enclosed near-B subject/hole dispute", _enclosed_near_b_ring(), {}),
        ("unknown_fallback_noise", "unknown unstable background fallback", _unknown_fallback_image(), {}),
    ]
    cases = []
    for case_id, label, image, form in specs:
        path = inputs_dir / f"{case_id}.png"
        image.save(path)
        cases.append(
            {
                "case_id": case_id,
                "label": label,
                "input": path,
                "source": "synthetic",
                "form": form,
            }
        )
    return cases


def _manifest_cases(sample_manifest: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    payload = json.loads(sample_manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else []
    out = []
    for sample_id in sample_ids:
        match = next((case for case in cases if isinstance(case, dict) and case.get("sample_id") == sample_id), None)
        if not match:
            raise ValueError(f"sample_id {sample_id!r} not found in {sample_manifest}")
        out.append(
            {
                "case_id": str(match["sample_id"]),
                "label": str(match.get("primary_ambiguity") or match.get("id") or match["sample_id"]),
                "input": PROJECT_ROOT / str(match["input"]),
                "source": "samples/corridorkey_semantic/manifest.json",
                "sample_metadata": match,
                "form": {},
            }
        )
    return out


def _post_json(url: str, *, files: dict[str, Any], data: dict[str, str], timeout: float) -> tuple[int, dict[str, Any] | None, str]:
    response = requests.post(url, files=files, data=data, timeout=timeout)
    text = response.text
    try:
        payload = response.json()
    except Exception:
        payload = None
    return response.status_code, payload, text[:1000]


def _run_case(
    *,
    case: dict[str, Any],
    out_root: Path,
    web_url: str,
    requested_backend: str,
    timeout: float,
) -> dict[str, Any]:
    input_path = Path(case["input"])
    case_dir = out_root / "cases" / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    copied_input = case_dir / "input.png"
    copied_input.write_bytes(input_path.read_bytes())
    png_data = copied_input.read_bytes()

    checker_status, checker_payload, checker_text = _post_json(
        f"{web_url.rstrip('/')}/api/checkerboard-background",
        files={"file": (copied_input.name, png_data, "image/png")},
        data={},
        timeout=timeout,
    )

    form = {"backend": requested_backend, "parameter_source": "auto", **dict(case.get("form") or {})}
    start = time.perf_counter()
    status, payload, text = _post_json(
        f"{web_url.rstrip('/')}/api/matte-candidates",
        files={"file": (copied_input.name, png_data, "image/png")},
        data=form,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start

    outputs: dict[str, str] = {"input": _rel(copied_input)}
    selected = _selected_candidate(payload or {})
    if selected and isinstance(selected.get("rgba"), str):
        rgba_path = case_dir / "rgba.png"
        _write_png_from_data_url(str(selected["rgba"]), rgba_path)
        outputs["rgba"] = _rel(rgba_path)
    if selected and isinstance(selected.get("alpha"), str):
        alpha_path = case_dir / "alpha.png"
        _write_png_from_data_url(str(selected["alpha"]), alpha_path)
        outputs["alpha"] = _rel(alpha_path)

    slim_payload = _strip_image_payloads(payload) if payload is not None else None
    summary = {
        "status": "ok" if status == 200 else "error",
        "http_status": status,
        "case_id": case["case_id"],
        "label": case["label"],
        "source": case["source"],
        "input": _rel(copied_input),
        "requested_backend": requested_backend,
        "fixed_execution_backend": None,
        "form": form,
        "elapsed_sec_client": elapsed,
        "checkerboard_probe": {
            "http_status": checker_status,
            "payload": checker_payload,
            "text": checker_text if checker_payload is None else None,
        },
        "route": _route_from_payload(payload or {}),
        "runtime": _runtime_from_payload(payload or {}, requested_backend=requested_backend),
        "candidate_count": len(payload.get("candidates", [])) if isinstance(payload, dict) and isinstance(payload.get("candidates"), list) else 0,
        "selected_candidate_id": selected.get("id") if selected else None,
        "outputs": outputs,
        "response": slim_payload,
        "response_text": text if payload is None else None,
        "sample_metadata": case.get("sample_metadata"),
    }
    summary_path = case_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = build_run_manifest(
        run_dir=case_dir,
        input_path=copied_input,
        outputs={key: PROJECT_ROOT / value for key, value in outputs.items() if key != "input"},
        request={
            "backend": requested_backend,
            "form": form,
            "source_input": _rel(input_path),
        },
        route=summary["route"],
        runtime=summary["runtime"],
        report_path=summary_path,
        extra={
            "case_id": case["case_id"],
            "label": case["label"],
            "checkerboard_probe": summary["checkerboard_probe"],
        },
    )
    manifest_path = write_run_manifest(case_dir / "manifest.json", manifest)
    summary["case_manifest"] = _rel(manifest_path)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.out_dir or _next_out_dir("stage0_baseline")
    out_root.mkdir(parents=True, exist_ok=True)

    web_health = requests.get(f"{args.web_url.rstrip('/')}/health", timeout=args.timeout).json()
    runtime_capabilities = requests.get(
        f"{args.web_url.rstrip('/')}/api/runtime-capabilities",
        params={"include_comfy": "false", "include_object_info": "false", "timeout": min(args.timeout, 5.0)},
        timeout=args.timeout,
    ).json()

    cases = _synthetic_cases(out_root)
    cases.extend(_manifest_cases(args.sample_manifest, [item.strip() for item in args.sample_ids.split(",") if item.strip()]))

    runs = [
        _run_case(
            case=case,
            out_root=out_root,
            web_url=args.web_url,
            requested_backend=args.backend,
            timeout=args.timeout,
        )
        for case in cases
    ]
    known_failures = []
    for run in runs:
        if run["case_id"] == "enclosed_near_b_ring":
            known_failures.append(
                {
                    "case_id": run["case_id"],
                    "class": "enclosed_near_b_subject_hole_dispute",
                    "current_status": "executes immediately through legacy matte-candidates; no Analyze/Decide semantic candidate gate",
                    "candidate_count": run.get("candidate_count"),
                    "selected_candidate_id": run.get("selected_candidate_id"),
                    "route": run.get("route"),
                }
            )

    aggregate = {
        "schema": "ermbg.stage0_baseline.v1",
        "batch": _rel(out_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "web_url": args.web_url.rstrip("/"),
        "backend": args.backend,
        "sample_manifest": _rel(args.sample_manifest),
        "web_health": web_health,
        "runtime_capabilities": runtime_capabilities,
        "case_count": len(runs),
        "ok_count": sum(1 for run in runs if run.get("status") == "ok"),
        "error_count": sum(1 for run in runs if run.get("status") != "ok"),
        "known_failures": known_failures,
        "runs": runs,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(json_safe(aggregate), indent=2, ensure_ascii=False), encoding="utf-8")
    batch_manifest = build_run_manifest(
        run_dir=out_root,
        outputs={"summary": summary_path},
        request={
            "backend": args.backend,
            "web_url": args.web_url.rstrip("/"),
            "sample_ids": args.sample_ids,
        },
        route={},
        runtime={
            "kind": "stage0-baseline",
            "backend": args.backend,
            "ok_count": aggregate["ok_count"],
        },
        report_path=summary_path,
        extra={"case_manifests": [run.get("case_manifest") for run in runs]},
    )
    manifest_path = write_run_manifest(out_root / "manifest.json", batch_manifest)
    aggregate["artifact_manifest"] = _rel(manifest_path)
    summary_path.write_text(json.dumps(json_safe(aggregate), indent=2, ensure_ascii=False), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-url", default="http://127.0.0.1:7860")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--sample-ids", default="B001,B003,I011,I013,C001")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "batch": summary["batch"],
                "case_count": summary["case_count"],
                "ok_count": summary["ok_count"],
                "error_count": summary["error_count"],
                "summary": str((PROJECT_ROOT / summary["batch"] / "summary.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
