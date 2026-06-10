#!/usr/bin/env python3
"""Smoke test the ERMBG direct worker HTTP service with real sample images."""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from ermbg.artifacts import build_run_manifest, write_run_manifest
from ermbg.direct_worker_client import DEFAULT_DIRECT_WORKER_URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json"
DEFAULT_SAMPLE_IDS = "B001,B002,I011,C001"


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


def _case_screen(case: dict[str, Any]) -> str:
    input_value = str(case.get("input", ""))
    for screen in ("green", "blue", "white"):
        if isinstance(case.get(screen), str) and str(case[screen]) == input_value:
            return screen
    screen = case.get("screen")
    return str(screen) if isinstance(screen, str) and screen else Path(input_value).stem


def _case_key(case: dict[str, Any]) -> str:
    return f"{case.get('sample_id')}_{case.get('id')}_{_case_screen(case)}"


def _selected_cases(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = [case for case in manifest["cases"] if isinstance(case, dict) and isinstance(case.get("input"), str)]
    if not args.all:
        sample_ids = {item.strip() for item in args.sample_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("sample_id", "")) in sample_ids]
    if args.category:
        categories = {item.strip() for item in args.category.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("category", "")) in categories]
    return cases


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args.manifest)
    cases = _selected_cases(args, manifest)
    cases_by_filename = {f"{_case_key(case)}.png": case for case in cases}
    out_root = args.out_dir or _next_out_dir("direct_worker_http_smoke")
    out_root.mkdir(parents=True, exist_ok=True)

    health = requests.get(f"{args.base_url.rstrip('/')}/health", timeout=args.timeout)
    health.raise_for_status()
    files = []
    opened = []
    try:
        for case in cases:
            path = PROJECT_ROOT / str(case["input"])
            fh = path.open("rb")
            opened.append(fh)
            files.append(("files", (f"{_case_key(case)}.png", fh, "image/png")))
        data = {
            "shadow_mode": args.shadow_mode,
            "corridorkey_screen_mode": args.corridorkey_screen_mode,
            "corridorkey_preset": args.corridorkey_preset,
            "fallback_bg_color": ",".join(str(c) for c in args.fallback_bg_color),
            "include_images": "true" if args.write_images else "false",
        }
        response = requests.post(
            f"{args.base_url.rstrip('/')}/batch-matte",
            files=files,
            data=data,
            timeout=args.timeout,
        )
        response.raise_for_status()
        payload = response.json()
    finally:
        for fh in opened:
            fh.close()

    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    for row in runs:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename", "case"))
        case_dir = out_root / filename.removesuffix(".png")
        case_dir.mkdir(parents=True, exist_ok=True)
        source_case = cases_by_filename.get(filename, {})
        input_path = PROJECT_ROOT / str(source_case["input"]) if isinstance(source_case.get("input"), str) else None
        encoded = row.pop("rgba_png_base64", None)
        if isinstance(encoded, str) and args.write_images:
            rgba_path = case_dir / "rgba.png"
            rgba_path.write_bytes(base64.b64decode(encoded))
            row["output"] = {"rgba": _rel(rgba_path)}
        case_summary = {
            **row,
            "fixed_backend": "auto",
            "actual_execution_backend": row.get("execution_backend"),
            "case_metadata": source_case,
        }
        case_summary_path = case_dir / "summary.json"
        case_summary_path.write_text(json.dumps(case_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        output_paths: dict[str, Path] = {}
        rgba_path = case_dir / "rgba.png"
        if rgba_path.exists():
            output_paths["rgba"] = rgba_path
        case_manifest = build_run_manifest(
            run_dir=case_dir,
            input_path=input_path,
            outputs=output_paths,
            request={
                "backend": "direct-worker",
                "fixed_backend": "auto",
                "source_input": _rel(input_path) if input_path is not None else None,
            },
            route={
                "algorithm": row.get("algorithm"),
                "route": row.get("route"),
                "asset_kind": row.get("asset_kind"),
                "parameter_profile": row.get("parameter_profile"),
                "execution_profile": row.get("execution_profile"),
            },
            runtime={
                "kind": "direct-worker",
                "backend": row.get("execution_backend"),
                "execution_backend": row.get("execution_backend"),
                "execution_server_url": args.base_url,
                "server_elapsed_sec": row.get("server_elapsed_sec"),
            },
            report_path=case_summary_path,
            extra={"case_metadata": source_case},
        )
        case_manifest_path = write_run_manifest(case_dir / "manifest.json", case_manifest)
        row["summary"] = _rel(case_summary_path)
        row["case_manifest"] = _rel(case_manifest_path)

    summary = {
        "batch": _rel(out_root),
        "base_url": args.base_url,
        "health": health.json(),
        "manifest": _rel(args.manifest),
        "sample_id": args.sample_id,
        "case_count": len(cases),
        "response": payload,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    batch_manifest = build_run_manifest(
        run_dir=out_root,
        outputs={"summary": summary_path},
        request={
            "backend": "direct-worker",
            "fixed_backend": "auto",
            "manifest": _rel(args.manifest),
            "sample_id": args.sample_id,
            "all": bool(args.all),
            "category": args.category,
        },
        runtime={
            "kind": "direct-worker-http-smoke",
            "backend": "direct-worker",
            "execution_server_url": args.base_url,
        },
        report_path=summary_path,
        extra={
            "case_manifests": [
                row.get("case_manifest")
                for row in runs
                if isinstance(row, dict) and isinstance(row.get("case_manifest"), str)
            ],
        },
    )
    manifest_path = write_run_manifest(out_root / "manifest.json", batch_manifest)
    summary["artifact_manifest"] = _rel(manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_DIRECT_WORKER_URL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--category", default="")
    parser.add_argument("--shadow-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--corridorkey-screen-mode", choices=("auto", "green", "blue"), default="auto")
    parser.add_argument("--corridorkey-preset", choices=("auto", "detail_safe", "spill_safe", "manual"), default="auto")
    parser.add_argument("--fallback-bg-color", type=int, nargs=3, default=(0, 200, 0), metavar=("R", "G", "B"))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-write-images", action="store_false", dest="write_images")
    parser.set_defaults(write_images=True)
    args = parser.parse_args()
    summary = run(args)
    response = summary["response"]
    print(
        json.dumps(
            {
                "batch": summary["batch"],
                "status": response.get("status"),
                "case_count": response.get("case_count"),
                "ok_count": response.get("ok_count"),
                "server_elapsed_sec": response.get("server_elapsed_sec"),
                "summary": str((PROJECT_ROOT / summary["batch"] / "summary.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
