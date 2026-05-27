#!/usr/bin/env python3
"""Run a production-path comfy-ermbg regression into a self-contained batch."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw

from ermbg import classify_image, matte_image


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


def _checker_bg(size: tuple[int, int], cell: int = 8) -> Image.Image:
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

    scale = 3
    pad = 10
    label_h = 18
    thumbs: list[Image.Image] = []
    for label, image in views:
        scaled = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (scaled.width, scaled.height + label_h), (245, 245, 245))
        canvas.paste(scaled, (0, label_h))
        ImageDraw.Draw(canvas).text((3, 2), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 3
    rows = 2
    sheet = Image.new(
        "RGB",
        (cols * thumbs[0].width + (cols - 1) * pad, rows * thumbs[0].height + (rows - 1) * pad),
        (255, 255, 255),
    )
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * (thumb.width + pad), (i // cols) * (thumb.height + pad)))
    sheet.save(case_dir / "contact_sheet.png")


def _load_case_metadata(input_path: Path) -> dict[str, Any]:
    case_json = input_path.with_name("case.json")
    if case_json.exists():
        return json.loads(case_json.read_text())
    return {}


def _coverage_metrics(input_path: Path, alpha_path: Path, background: tuple[int, int, int], threshold: float) -> dict[str, Any]:
    alpha_image = Image.open(alpha_path)
    if alpha_image.mode == "RGBA":
        alpha = np.asarray(alpha_image.getchannel("A"), dtype=np.uint8)
    else:
        alpha = np.asarray(alpha_image.convert("L"), dtype=np.uint8)
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


def _extract_candidate_png(response: dict[str, Any], out_path: Path) -> bool:
    candidates = response.get("candidates") or []
    if not candidates:
        return False
    candidate = candidates[0]
    for key in ("url", "image", "rgba", "preview"):
        value = candidate.get(key)
        if isinstance(value, str) and value.startswith("data:image"):
            out_path.write_bytes(base64.b64decode(value.split(",", 1)[1]))
            return True
    return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve()
    batch = Path(args.batch)
    metadata = _load_case_metadata(input_path)
    case_id = args.case_id or metadata.get("id") or input_path.parent.name or input_path.stem
    case_dir = batch / f"{case_id}_{args.phase}"
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, case_dir / "input.png")

    strategy = classify_image(input_path)
    start = time.perf_counter()
    result = matte_image(
        input_path,
        backend="comfy-ermbg",
        output_dir=case_dir,
        qa=args.qa,
        shadow_mode=args.shadow_mode,
        comfy_url=args.comfy_url,
    )
    elapsed = time.perf_counter() - start

    for src_name, dst_name in [
        ("input_rgba.png", "rgba.png"),
        ("input_alpha.png", "alpha.png"),
        ("input_foreground.png", "foreground.png"),
    ]:
        src = case_dir / src_name
        if src.exists():
            shutil.copy2(src, case_dir / dst_name)

    _write_contact_sheet(case_dir)
    background = tuple(int(c) for c in metadata.get("background", args.background))
    metrics = _coverage_metrics(case_dir / "input.png", case_dir / "alpha.png", background, args.subject_threshold)

    summary: dict[str, Any] = {
        "case": case_id,
        "phase": args.phase,
        "backend": "comfy-ermbg",
        "input": str(input_path),
        "shadow_mode": args.shadow_mode,
        "elapsed_sec_client": elapsed,
        "outputs": {
            "input": str(case_dir / "input.png"),
            "rgba": str(case_dir / "rgba.png"),
            "alpha": str(case_dir / "alpha.png"),
            "foreground": str(case_dir / "foreground.png"),
            "contact_sheet": str(case_dir / "contact_sheet.png"),
        },
        "local_router_preview": {
            "name": strategy.name,
            "bg_type": strategy.bg_type,
            "image_type": strategy.image_type,
            "keyer_mode": strategy.keyer_mode,
            "extras": strategy.extras,
        },
        "remote_debug": _json_safe(result.debug),
        "quality_metrics": metrics,
        "case_metadata": metadata,
    }

    if args.web_smoke:
        web_dir = case_dir / "web_smoke"
        web_dir.mkdir(exist_ok=True)
        with input_path.open("rb") as f:
            response = requests.post(
                f"{args.web_url.rstrip('/')}/api/matte-candidates",
                files={"file": (input_path.name, f, "image/png")},
                data={"backend": "comfy-ermbg"},
                timeout=args.web_timeout,
            )
        response.raise_for_status()
        payload = response.json()
        (web_dir / "response.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        candidate_path = web_dir / "candidate_auto.png"
        saved_candidate = _extract_candidate_png(payload, candidate_path)
        web_summary: dict[str, Any] = {
            "backend": payload.get("backend"),
            "strategy": payload.get("strategy"),
            "server_elapsed_sec": payload.get("server_elapsed_sec"),
            "has_server_elapsed_sec": isinstance(payload.get("server_elapsed_sec"), (int, float)),
            "candidate": str(candidate_path) if saved_candidate else None,
        }
        if saved_candidate:
            web_summary["quality_metrics"] = _coverage_metrics(case_dir / "input.png", candidate_path, background, args.subject_threshold)
        (web_dir / "summary.json").write_text(json.dumps(web_summary, indent=2, ensure_ascii=False))
        summary["web_smoke"] = web_summary

    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    aggregate_path = batch / "summary.json"
    if aggregate_path.exists():
        try:
            aggregate = json.loads(aggregate_path.read_text())
            runs = aggregate.get("runs", [])
        except Exception:
            runs = []
    else:
        runs = []
    runs = [r for r in runs if not (r.get("case") == case_id and r.get("phase") == args.phase)]
    runs.append(summary)
    aggregate_path.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Regression input image, usually samples/regression/<case>/input.png")
    parser.add_argument("--batch", default="out/comfy_ermbg_regression", help="Batch root under out/")
    parser.add_argument("--case-id", default="", help="Override case id; defaults to case.json id or parent folder")
    parser.add_argument("--phase", default="remote", help="Run label appended to the case directory")
    parser.add_argument("--comfy-url", default="http://192.168.0.8:8000")
    parser.add_argument("--shadow-mode", default="on", choices=["on", "auto", "off"])
    parser.add_argument("--background", type=int, nargs=3, default=(0, 200, 0), metavar=("R", "G", "B"))
    parser.add_argument("--subject-threshold", type=float, default=35.0)
    parser.add_argument("--qa", action="store_true", help="Also write ERMBG QA composites")
    parser.add_argument("--web-smoke", action="store_true", help="POST the same image through the local Web API")
    parser.add_argument("--web-url", default="http://127.0.0.1:7860")
    parser.add_argument("--web-timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "case": summary["case"],
                "phase": summary["phase"],
                "outputs": summary["outputs"],
                "quality_metrics": summary["quality_metrics"],
                "web_smoke": summary.get("web_smoke"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
