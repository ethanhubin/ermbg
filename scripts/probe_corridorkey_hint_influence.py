#!/usr/bin/env python
"""Probe how CorridorKey responds to feature-driven hint variants."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from ermbg.corridorkey_hint import (
    build_corridorkey_hint_plan,
    corridorkey_hint_diagnostic_variants,
    corridorkey_hint_variants,
)
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


def _alpha_stats(alpha: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "min": float(alpha.min()) if alpha.size else 0.0,
        "max": float(alpha.max()) if alpha.size else 0.0,
        "mean": float(alpha.mean()) if alpha.size else 0.0,
        "nonzero_pixels": int((alpha > 0.001).sum()),
        "semi_alpha_pixels": int(((alpha > 0.02) & (alpha < 0.98)).sum()),
    }
    region_stats: dict[str, Any] = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            region_stats[name] = {"pixels": 0}
            continue
        values = alpha[mask]
        region_stats[name] = {
            "pixels": int(mask.sum()),
            "mean": float(values.mean()),
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        }
    stats["regions"] = region_stats
    return stats


def _diff_stats(alpha: np.ndarray, baseline: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    diff = alpha.astype(np.float32) - baseline.astype(np.float32)
    stats: dict[str, Any] = {
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff), initial=0.0)),
        "pixels_abs_ge_0_02": int((np.abs(diff) >= 0.02).sum()),
        "pixels_abs_ge_0_05": int((np.abs(diff) >= 0.05).sum()),
        "pixels_abs_ge_0_10": int((np.abs(diff) >= 0.10).sum()),
    }
    region_stats: dict[str, Any] = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            region_stats[name] = {"pixels": 0}
            continue
        values = diff[mask]
        region_stats[name] = {
            "pixels": int(mask.sum()),
            "mean": float(values.mean()),
            "mean_abs": float(np.mean(np.abs(values))),
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
        }
    stats["regions"] = region_stats
    return stats


def _worker_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def run_probe(
    image_path: Path,
    *,
    out_dir: Path,
    direct_worker_url: str,
    run_remote: bool,
    timeout: float,
    constant_hint_strengths: tuple[float, ...] = (),
    include_full_white_diagnostic: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(out_dir / "source.png")
    route = classify_route(image)
    route_payload = route.to_dict()
    ck = route.analysis.get("corridorkey_analysis") if isinstance(route.analysis, dict) else {}
    bg = ck.get("background_color") if isinstance(ck, dict) else None
    if not (isinstance(bg, list) and len(bg) == 3):
        bg_color = tuple(int(c) for c in route.params.get("pymatting_bg_color", (0, 200, 0)))
    else:
        bg_color = tuple(int(np.clip(c, 0, 255)) for c in bg)

    rows: list[dict[str, Any]] = []
    alphas: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] | None = None
    worker_available = bool(run_remote and _worker_ok(direct_worker_url))
    variants = list(corridorkey_hint_variants())
    if include_full_white_diagnostic:
        variants.extend(corridorkey_hint_diagnostic_variants())
    for variant in variants:
        plan = build_corridorkey_hint_plan(image, bg_color, variant=variant)
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        _save_gray(variant_dir / "hint.png", plan.hint)
        _save_gray(variant_dir / "feature_key_alpha.png", plan.features.key_alpha)
        if masks is None:
            masks = {
                "outline_mask": plan.features.outline_mask,
                "outline_inner_mask": plan.features.outline_inner_mask,
                "control_outline_mask": plan.features.control_outline_mask,
                "control_outline_inner_mask": plan.features.control_outline_inner_mask,
                "subject_support": plan.features.subject_support,
                "hard_subject": plan.features.hard_subject,
                "translucent_candidate": plan.features.translucent_candidate,
                "internal_transparency_candidate": plan.features.internal_transparency_candidate,
                "soft_boundary_candidate": plan.features.soft_boundary_candidate,
            }
            for name, mask in masks.items():
                _save_gray(out_dir / f"feature_{name}.png", mask.astype(np.float32))
        row: dict[str, Any] = {
            "variant": variant,
            "hint": _rel(variant_dir / "hint.png"),
            "metadata": plan.metadata,
            "status": "hint_generated",
        }
        if worker_available:
            result = matte_image_direct_worker(
                image,
                direct_worker_url=direct_worker_url,
                execution_backend="direct-corridorkey",
                shadow_mode="off",
                corridorkey_hint_mask=plan.hint,
                corridorkey_auto_mask=False,
                corridorkey_screen_mode=str(ck.get("screen_mode") if isinstance(ck, dict) else "auto"),
                corridorkey_preset="auto",
                corridorkey_hard_ui_hint_mode=None,
                timeout=timeout,
            )
            alphas[variant] = result.alpha
            _save_gray(variant_dir / "alpha.png", result.alpha)
            _save_rgba(variant_dir / "rgba.png", result.rgba)
            row["status"] = "ok"
            row["alpha"] = _rel(variant_dir / "alpha.png")
            row["rgba"] = _rel(variant_dir / "rgba.png")
            row["alpha_stats"] = _alpha_stats(result.alpha, masks or {})
            row["corridorkey_debug"] = {
                "hint": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("hint"),
                "corridorkey_mask": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("corridorkey_mask"),
                "settings": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("settings"),
                "hard_ui_hint": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("hard_ui_hint"),
                "server_elapsed_sec": result.debug.get("server_elapsed_sec"),
            }
        rows.append(row)

    if constant_hint_strengths:
        feature_plan = build_corridorkey_hint_plan(image, bg_color, variant="feature_balanced")
        if masks is None:
            masks = {
                "outline_mask": feature_plan.features.outline_mask,
                "outline_inner_mask": feature_plan.features.outline_inner_mask,
                "control_outline_mask": feature_plan.features.control_outline_mask,
                "control_outline_inner_mask": feature_plan.features.control_outline_inner_mask,
                "subject_support": feature_plan.features.subject_support,
                "hard_subject": feature_plan.features.hard_subject,
                "translucent_candidate": feature_plan.features.translucent_candidate,
                "internal_transparency_candidate": feature_plan.features.internal_transparency_candidate,
                "soft_boundary_candidate": feature_plan.features.soft_boundary_candidate,
            }
            for name, mask in masks.items():
                _save_gray(out_dir / f"feature_{name}.png", mask.astype(np.float32))
        for value in constant_hint_strengths:
            strength = float(np.clip(value, 0.0, 1.0))
            variant = f"constant_{strength:.2f}".replace(".", "p")
            hint = np.full(image.shape[:2], strength, dtype=np.float32)
            variant_dir = out_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            _save_gray(variant_dir / "hint.png", hint)
            _save_gray(variant_dir / "feature_key_alpha.png", feature_plan.features.key_alpha)
            row = {
                "variant": variant,
                "hint": _rel(variant_dir / "hint.png"),
                "metadata": {
                    "schema": "ermbg.corridorkey_hint_plan.v1",
                    "variant": variant,
                    "diagnostic": True,
                    "policy": {
                        "type": "full_frame_constant_strength",
                        "value": strength,
                    },
                    "hint": {
                        "min": strength,
                        "max": strength,
                        "mean": strength,
                        "nonzero_pixels": int((hint > 0.001).sum()),
                    },
                    "features": feature_plan.features.metadata,
                },
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
                    corridorkey_screen_mode=str(ck.get("screen_mode") if isinstance(ck, dict) else "auto"),
                    corridorkey_preset="auto",
                    corridorkey_hard_ui_hint_mode=None,
                    timeout=timeout,
                )
                alphas[variant] = result.alpha
                _save_gray(variant_dir / "alpha.png", result.alpha)
                _save_rgba(variant_dir / "rgba.png", result.rgba)
                row["status"] = "ok"
                row["alpha"] = _rel(variant_dir / "alpha.png")
                row["rgba"] = _rel(variant_dir / "rgba.png")
                row["alpha_stats"] = _alpha_stats(result.alpha, masks or {})
                row["corridorkey_debug"] = {
                    "hint": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("hint"),
                    "corridorkey_mask": result.debug.get("direct_worker", {})
                    .get("algorithm_debug", {})
                    .get("corridorkey_mask"),
                    "settings": result.debug.get("direct_worker", {}).get("algorithm_debug", {}).get("settings"),
                    "hard_ui_hint": result.debug.get("direct_worker", {})
                    .get("algorithm_debug", {})
                    .get("hard_ui_hint"),
                    "server_elapsed_sec": result.debug.get("server_elapsed_sec"),
                }
            rows.append(row)

    baseline = alphas.get("current_default_prior")
    if baseline is not None and masks is not None:
        for row in rows:
            variant = str(row["variant"])
            alpha = alphas.get(variant)
            if alpha is None or variant == "current_default_prior":
                continue
            diff = alpha.astype(np.float32) - baseline.astype(np.float32)
            diff_vis = np.clip((diff * 0.5) + 0.5, 0.0, 1.0)
            variant_dir = out_dir / variant
            _save_gray(variant_dir / "alpha_diff_vs_baseline_midgray.png", diff_vis)
            row["alpha_diff_vs_baseline"] = _diff_stats(alpha, baseline, masks)
            row["alpha_diff_image"] = _rel(variant_dir / "alpha_diff_vs_baseline_midgray.png")

    summary = {
        "schema": "ermbg.corridorkey_hint_probe.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image": _rel(image_path),
        "out_dir": _rel(out_dir),
        "direct_worker_url": direct_worker_url,
        "worker_available": worker_available,
        "route": route_payload,
        "background_color": [int(c) for c in bg_color],
        "variants": rows,
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
    parser.add_argument("--run-remote", action="store_true", help="Call /matte for each hint variant.")
    parser.add_argument(
        "--include-strength-diagnostics",
        action="store_true",
        help="Also probe full-frame constant hint strengths 0, 0.1, 0.32, 0.6, and 1.",
    )
    parser.add_argument(
        "--include-full-white-diagnostic",
        action="store_true",
        help="Also probe the full-frame white diagnostic upper bound.",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "out" / f"corridorkey_hint_probe_{stamp}"
    summary = run_probe(
        args.image,
        out_dir=out_dir,
        direct_worker_url=str(args.direct_worker_url),
        run_remote=bool(args.run_remote),
        timeout=float(args.timeout),
        constant_hint_strengths=(0.0, 0.1, 0.32, 0.6, 1.0) if args.include_strength_diagnostics else (),
        include_full_white_diagnostic=bool(args.include_full_white_diagnostic),
    )
    print(json.dumps({"summary": summary["out_dir"], "worker_available": summary["worker_available"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
