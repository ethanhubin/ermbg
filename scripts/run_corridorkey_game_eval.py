#!/usr/bin/env python3
"""Run a remote matting backend over manifest-selected game-eval inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ermbg import matte_image
from ermbg.comfy import DEFAULT_COMFY_URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json"
EVAL_BACKENDS = ("auto", "comfy-corridorkey", "comfy-pymatting-known-b", "comfy-ermbg", "comfy-rmbg")
COLOR_PROTECTION_MODES = ("auto", "on", "off")
HARD_UI_HINT_MODES = (
    "all_white",
    "bbox_2px",
    "boundary_2px",
    "boundary_2px_shadow_safe",
    "boundary_2px_shadow_safe_edge_floor",
    "translucent_button",
)


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


def _color_protection_arg(mode: str) -> bool | None:
    if mode == "auto":
        return None
    return mode == "on"


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
        (f"{stem}_key_color_protection.png", "key_color_protection.png"),
    ]:
        src = case_dir / src_name
        if src.exists():
            shutil.copy2(src, case_dir / dst_name)


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
        "hint": "corridorkey_hint.png",
        "raw_alpha": "corridorkey_raw_alpha.png",
        "key_color_protection": "key_color_protection.png",
        "contact_sheet": "contact_sheet.png",
    }
    outputs = {}
    for key, name in names.items():
        path = case_dir / name
        if path.exists():
            outputs[key] = _rel(path)
    return outputs


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
        case_dir = out_root / f"{sample_id}_{case_id}_{screen}"
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, case_dir / "input.png")
        print(f"[{case_index}/{total}] {sample_id}-{screen[:1].upper()} {case_id}", flush=True)
        start = time.perf_counter()
        try:
            result = matte_image(
                input_path,
                backend=args.backend,
                output_dir=case_dir,
                qa=False,
                comfy_url=args.comfy_url,
                corridorkey_preset=args.corridorkey_preset,
                corridorkey_color_protection=_color_protection_arg(args.corridorkey_color_protection),
                corridorkey_auto_mask=args.corridorkey_hard_ui_hint_mode != "all_white",
                corridorkey_hard_ui_hint_mode=args.corridorkey_hard_ui_hint_mode,
            )
            elapsed = time.perf_counter() - start
            effective_backend = _effective_backend(args.backend, result)
            stem = input_path.stem
            _copy_backend_outputs(case_dir, stem)
            _write_contact_sheet(case_dir)
            background = _case_background(manifest, case, screen)
            metrics = _coverage_metrics(case_dir / "input.png", case_dir / "alpha.png", background, args.subject_threshold)
            summary = {
                "status": "ok",
                "case": f"{sample_id}_{case_id}_{screen}",
                "backend": effective_backend,
                "requested_backend": args.backend,
                "input": _rel(input_path),
                "sample_screen": screen,
                "elapsed_sec_client": elapsed,
                "outputs": _case_outputs(case_dir),
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
                "case": f"{sample_id}_{case_id}_{screen}",
                "backend": args.backend,
                "requested_backend": args.backend,
                "input": _rel(input_path),
                "sample_screen": screen,
                "error": str(exc),
                "case_metadata": case,
            }
            print(f"  ERROR {exc}", flush=True)
        (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
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
                "corridorkey_color_protection": args.corridorkey_color_protection,
                "corridorkey_auto_mask": args.corridorkey_hard_ui_hint_mode != "all_white",
                "corridorkey_hard_ui_hint_mode": args.corridorkey_hard_ui_hint_mode,
            },
            "runs": runs,
        }
        (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

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
    parser.add_argument("--subject-threshold", type=float, default=35.0)
    parser.add_argument("--corridorkey-preset", default="auto", choices=("auto", "detail_safe", "spill_safe", "manual"))
    parser.add_argument("--corridorkey-color-protection", default="auto", choices=COLOR_PROTECTION_MODES)
    parser.add_argument("--corridorkey-hard-ui-hint-mode", default="bbox_2px", choices=HARD_UI_HINT_MODES)
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
