#!/usr/bin/env python
"""Probe CorridorKey full-frame constant hint strengths."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from ermbg.corridorkey_hint import corridorkey_hint_strengths
from ermbg.direct_worker_client import matte_image_direct_worker
from ermbg.router import classify_route

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _save_gray(path: Path, values: np.ndarray) -> None:
    arr = np.clip(values.astype(np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba.astype(np.uint8), mode="RGBA").save(path)


def _alpha_stats(alpha: np.ndarray) -> dict[str, Any]:
    return {
        "min": float(alpha.min()) if alpha.size else 0.0,
        "max": float(alpha.max()) if alpha.size else 0.0,
        "mean": float(alpha.mean()) if alpha.size else 0.0,
        "nonzero_pixels": int((alpha > 0.001).sum()),
        "semi_alpha_pixels": int(((alpha > 0.02) & (alpha < 0.98)).sum()),
    }


def _diff_stats(alpha: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    diff = alpha.astype(np.float32) - baseline.astype(np.float32)
    return {
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff), initial=0.0)),
        "pixels_abs_ge_0_02": int((np.abs(diff) >= 0.02).sum()),
        "pixels_abs_ge_0_05": int((np.abs(diff) >= 0.05).sum()),
        "pixels_abs_ge_0_10": int((np.abs(diff) >= 0.10).sum()),
    }


def _worker_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _parse_strengths(text: str | None) -> tuple[float, ...]:
    if not text:
        return corridorkey_hint_strengths()
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(np.clip(float(item), 0.0, 1.0)))
    return tuple(values)


def run_probe(
    image_path: Path,
    *,
    out_dir: Path,
    direct_worker_url: str,
    run_remote: bool,
    timeout: float,
    strengths: tuple[float, ...],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(out_dir / "source.png")
    route = classify_route(image)
    route_payload = route.to_dict()
    ck = route.analysis.get("corridorkey_analysis") if isinstance(route.analysis, dict) else {}
    screen_mode = str(ck.get("screen_mode") if isinstance(ck, dict) else "auto")

    rows: list[dict[str, Any]] = []
    alphas: dict[float, np.ndarray] = {}
    worker_available = bool(run_remote and _worker_ok(direct_worker_url))
    for strength in strengths:
        value = float(np.clip(strength, 0.0, 1.0))
        case_id = f"hint_{int(round(value * 100)):03d}"
        hint = np.full(image.shape[:2], value, dtype=np.float32)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _save_gray(case_dir / "hint.png", hint)
        row: dict[str, Any] = {
            "id": case_id,
            "hint_value": value,
            "hint": _rel(case_dir / "hint.png"),
            "status": "hint_generated",
        }
        if worker_available:
            result = matte_image_direct_worker(
                image,
                direct_worker_url=direct_worker_url,
                execution_backend="direct-corridorkey",
                shadow_mode="off",
                corridorkey_hint_mask=hint,
                corridorkey_auto_mask=False,
                corridorkey_screen_mode=screen_mode,
                corridorkey_preset="auto",
                timeout=timeout,
            )
            alphas[value] = result.alpha
            _save_gray(case_dir / "alpha.png", result.alpha)
            _save_rgba(case_dir / "rgba.png", result.rgba)
            row.update(
                {
                    "status": "ok",
                    "alpha": _rel(case_dir / "alpha.png"),
                    "rgba": _rel(case_dir / "rgba.png"),
                    "alpha_stats": _alpha_stats(result.alpha),
                    "corridorkey_debug": {
                        "hint": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("hint"),
                        "settings": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("settings"),
                        "shadow": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("shadow"),
                        "semantic_execution": result.debug.get("direct_worker", {})
                        .get("algorithm_debug", {})
                        .get("semantic_execution"),
                        "server_elapsed_sec": result.debug.get("server_elapsed_sec"),
                    },
                }
            )
        rows.append(row)

    baseline = alphas.get(0.32)
    if baseline is not None:
        for row in rows:
            value = float(row["hint_value"])
            alpha = alphas.get(value)
            if alpha is None or np.isclose(value, 0.32):
                continue
            diff = alpha.astype(np.float32) - baseline.astype(np.float32)
            diff_vis = np.clip((diff * 0.5) + 0.5, 0.0, 1.0)
            case_dir = out_dir / str(row["id"])
            _save_gray(case_dir / "alpha_diff_vs_032_midgray.png", diff_vis)
            row["alpha_diff_vs_032"] = _diff_stats(alpha, baseline)
            row["alpha_diff_image"] = _rel(case_dir / "alpha_diff_vs_032_midgray.png")

    summary = {
        "schema": "ermbg.corridorkey_hint_strength_probe.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": _rel(image_path),
        "out_dir": _rel(out_dir),
        "direct_worker_url": direct_worker_url,
        "worker_available": worker_available,
        "route": route_payload,
        "hint_strengths": [float(value) for value in strengths],
        "results": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT
        / "samples/corridorkey_semantic/icon/icon_icon_d11_glass_portal_blue/blue.png",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--direct-worker-url", default="http://127.0.0.1:7871")
    parser.add_argument("--run-remote", action="store_true", help="Call Direct Worker for each hint strength.")
    parser.add_argument("--strengths", default=None, help="Comma-separated full-frame hint strengths.")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "out" / f"corridorkey_hint_strength_probe_{stamp}"
    summary = run_probe(
        args.image,
        out_dir=out_dir,
        direct_worker_url=str(args.direct_worker_url),
        run_remote=bool(args.run_remote),
        timeout=float(args.timeout),
        strengths=_parse_strengths(args.strengths),
    )
    print(json.dumps({"summary": summary["out_dir"], "worker_available": summary["worker_available"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
