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
            "corridorkey_hard_ui_hint_mode": args.corridorkey_hard_ui_hint_mode,
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
        encoded = row.pop("rgba_png_base64", None)
        if isinstance(encoded, str) and args.write_images:
            case_dir = out_root / str(row.get("filename", "case")).removesuffix(".png")
            case_dir.mkdir(parents=True, exist_ok=True)
            rgba_path = case_dir / "rgba.png"
            rgba_path.write_bytes(base64.b64decode(encoded))
            row["output"] = {"rgba": _rel(rgba_path)}

    summary = {
        "batch": _rel(out_root),
        "base_url": args.base_url,
        "health": health.json(),
        "manifest": _rel(args.manifest),
        "sample_id": args.sample_id,
        "case_count": len(cases),
        "response": payload,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7871")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--category", default="")
    parser.add_argument("--shadow-mode", choices=("on", "off", "auto"), default="on")
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
