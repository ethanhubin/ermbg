#!/usr/bin/env python3
"""Run remote CorridorKey over the game-eval green variants."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ermbg import matte_image
from ermbg.comfy import DEFAULT_COMFY_URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "vlm_eval_game" / "manifest.json"


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


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_cases(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a cases list")
    return [case for case in cases if isinstance(case, dict)]


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
    cases = _load_cases(args.manifest)
    if args.sample_id:
        sample_ids = {item.strip() for item in args.sample_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("sample_id", "")) in sample_ids]

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        sample_id = str(case.get("sample_id") or f"G{index:02d}")
        input_path = PROJECT_ROOT / str(case["green"])
        case_dir = out_root / f"{sample_id}_{case_id}_green"
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, case_dir / "input.png")
        print(f"[{index}/{total}] {sample_id}-G {case_id}", flush=True)
        start = time.perf_counter()
        try:
            result = matte_image(
                input_path,
                backend="comfy-corridorkey",
                output_dir=case_dir,
                qa=False,
                comfy_url=args.comfy_url,
            )
            elapsed = time.perf_counter() - start
            stem = input_path.stem
            for src_name, dst_name in [
                (f"{stem}_rgba.png", "rgba.png"),
                (f"{stem}_alpha.png", "alpha.png"),
                (f"{stem}_foreground.png", "foreground.png"),
                (f"{stem}_corridorkey_hint.png", "corridorkey_hint.png"),
                (f"{stem}_corridorkey_raw_alpha.png", "corridorkey_raw_alpha.png"),
                (f"{stem}_key_color_protection.png", "key_color_protection.png"),
            ]:
                src = case_dir / src_name
                if src.exists():
                    shutil.copy2(src, case_dir / dst_name)
            _write_contact_sheet(case_dir)
            background = tuple(int(c) for c in case.get("backgrounds", {}).get("green", [0, 200, 0]))
            metrics = _coverage_metrics(case_dir / "input.png", case_dir / "alpha.png", background, args.subject_threshold)
            summary = {
                "status": "ok",
                "case": f"{sample_id}_{case_id}_green",
                "backend": "comfy-corridorkey",
                "input": _rel(input_path),
                "sample_variant": "green",
                "elapsed_sec_client": elapsed,
                "outputs": {
                    "input": _rel(case_dir / "input.png"),
                    "rgba": _rel(case_dir / "rgba.png"),
                    "alpha": _rel(case_dir / "alpha.png"),
                    "foreground": _rel(case_dir / "foreground.png"),
                    "hint": _rel(case_dir / "corridorkey_hint.png"),
                    "raw_alpha": _rel(case_dir / "corridorkey_raw_alpha.png"),
                    "key_color_protection": _rel(case_dir / "key_color_protection.png"),
                    "contact_sheet": _rel(case_dir / "contact_sheet.png"),
                },
                "remote_debug": _json_safe(result.debug),
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
                "case": f"{sample_id}_{case_id}_green",
                "backend": "comfy-corridorkey",
                "input": _rel(input_path),
                "sample_variant": "green",
                "error": str(exc),
                "case_metadata": case,
            }
            print(f"  ERROR {exc}", flush=True)
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        runs.append(summary)
        aggregate = {
            "backend": "comfy-corridorkey",
            "batch": _rel(out_root),
            "case_count": len(cases),
            "ok_count": sum(1 for row in runs if row.get("status") == "ok"),
            "variant": "green",
            "runs": runs,
        }
        (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "out" / "comfy_corridorkey_game_green_20260529")
    parser.add_argument("--sample-id", default="", help="Comma-separated sample ids, e.g. G02,G04,G06")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--subject-threshold", type=float, default=35.0)
    summary = run(parser.parse_args())
    print(
        json.dumps(
            {
                "batch": summary["batch"],
                "backend": summary["backend"],
                "case_count": summary["case_count"],
                "ok_count": summary["ok_count"],
                "summary": str((PROJECT_ROOT / summary["batch"] / "summary.json") if not str(summary["batch"]).startswith("/") else Path(summary["batch"]) / "summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
