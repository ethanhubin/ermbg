#!/usr/bin/env python3
"""Run remote ComfyUI ColorToMask chroma key over game-eval variants."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ermbg.comfy import DEFAULT_COMFY_URL
from ermbg.probe.comfyui_chroma_key import ComfyUIChromaKeyClient

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


def _save_result(case_dir: Path, result) -> None:
    Image.fromarray(result.rgba, mode="RGBA").save(case_dir / "rgba.png")
    Image.fromarray(np.clip(result.alpha * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L").save(case_dir / "alpha.png")
    Image.fromarray(result.foreground_srgb, mode="RGB").save(case_dir / "foreground.png")


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
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("--variants must include at least one variant")

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    client = ComfyUIChromaKeyClient(url=args.comfy_url)
    runs: list[dict[str, Any]] = []
    total = len(cases) * len(variants)
    run_index = 0
    for case in cases:
        case_id = str(case["id"])
        sample_id = str(case.get("sample_id") or f"G{len(runs) + 1:02d}")
        for variant in variants:
            run_index += 1
            if variant not in case:
                print(f"[{run_index}/{total}] {sample_id}-{variant[:1].upper()} {case_id}: SKIP missing variant", flush=True)
                continue
            input_path = PROJECT_ROOT / str(case[variant])
            background = tuple(int(c) for c in case.get("backgrounds", {}).get(variant, []))
            if len(background) != 3:
                manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
                background = tuple(int(c) for c in manifest.get("backgrounds", {}).get(variant, [0, 200, 0]))
            case_dir = out_root / f"{sample_id}_{case_id}_{variant}"
            case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, case_dir / "input.png")
            print(
                f"[{run_index}/{total}] {sample_id}-{variant[:1].upper()} {case_id} "
                f"key={background} threshold={args.threshold}",
                flush=True,
            )
            start = time.perf_counter()
            try:
                image = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
                result = client.matte(image, key_color=background, threshold=args.threshold)
                elapsed = time.perf_counter() - start
                _save_result(case_dir, result)
                metrics = _coverage_metrics(case_dir / "input.png", case_dir / "alpha.png", background, args.subject_threshold)
                summary = {
                    "status": "ok",
                    "case": f"{sample_id}_{case_id}_{variant}",
                    "backend": "comfy-chromakey",
                    "input": _rel(input_path),
                    "sample_variant": variant,
                    "elapsed_sec_client": elapsed,
                    "outputs": {
                        "input": _rel(case_dir / "input.png"),
                        "rgba": _rel(case_dir / "rgba.png"),
                        "alpha": _rel(case_dir / "alpha.png"),
                        "foreground": _rel(case_dir / "foreground.png"),
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
                    "case": f"{sample_id}_{case_id}_{variant}",
                    "backend": "comfy-chromakey",
                    "input": _rel(input_path),
                    "sample_variant": variant,
                    "error": str(exc),
                    "case_metadata": case,
                }
                print(f"  ERROR {exc}", flush=True)
            (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            runs.append(summary)
            aggregate = {
                "backend": "comfy-chromakey",
                "batch": _rel(out_root),
                "case_count": len(cases),
                "run_count": len(cases) * len(variants),
                "ok_count": sum(1 for row in runs if row.get("status") == "ok"),
                "variants": variants,
                "keyer": {
                    "node": "ColorToMask",
                    "threshold": int(args.threshold),
                    "note": "ColorToMask threshold is the ordinary chroma-key color range/tolerance.",
                },
                "runs": runs,
            }
            (out_root / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "out" / "comfy_chromakey_game_green_white_20260529")
    parser.add_argument("--sample-id", default="", help="Comma-separated sample ids, e.g. G02,G04,G06")
    parser.add_argument("--variants", default="green,white", help="Comma-separated variants from manifest, e.g. green,white")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--threshold", type=int, default=35, help="ColorToMask threshold/range")
    parser.add_argument("--subject-threshold", type=float, default=35.0)
    summary = run(parser.parse_args())
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
