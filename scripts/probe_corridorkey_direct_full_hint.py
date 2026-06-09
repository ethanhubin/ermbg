#!/usr/bin/env python
"""Direct CorridorKey smoke with a full-frame hint.

This intentionally bypasses ERMBG route/analyze/execute logic. It calls the
shared CorridorKey runner directly and writes only the raw runner artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ermbg.corridorkey_runner import LocalCorridorKeyClient


def _parse_rgb(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB must be R,G,B")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _checker_composite(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    h, w = alpha.shape[:2]
    yy, xx = np.indices((h, w))
    checker_value = np.where(((xx // 16 + yy // 16) % 2) == 0, 196, 238).astype(np.float32)
    checker = np.repeat(checker_value[..., None], 3, axis=2)
    return np.clip(rgb * alpha + checker * (1.0 - alpha) + 0.5, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call CorridorKey directly with a full-frame hint.")
    parser.add_argument(
        "input",
        nargs="?",
        default="samples/corridorkey_semantic/character/character_char_a01_hair_hard_edge_glass_pendant_green/green.png",
    )
    parser.add_argument("--out-dir", default="out/probe_corridorkey_direct_full_hint_c001")
    parser.add_argument("--background-color", type=_parse_rgb, default=(0, 200, 0))
    parser.add_argument("--screen-color", default="green", choices=("green", "blue", "auto"))
    parser.add_argument("--hint-value", type=float, default=1.0)
    parser.add_argument("--gamma-space", default="sRGB")
    parser.add_argument("--despill-strength", type=float, default=1.0)
    parser.add_argument("--refiner-strength", type=float, default=1.0)
    parser.add_argument("--auto-despeckle", default="Off")
    parser.add_argument("--despeckle-size", type=int, default=64)
    parser.add_argument("--color-protection", action="store_true")
    parser.add_argument("--prefer-processor", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    hint = np.full(image.shape[:2], float(args.hint_value), dtype=np.float32)

    client = LocalCorridorKeyClient(
        backend_label="probe-direct-corridorkey",
        prompt_id="probe_full_frame_hint",
        prefer_loaded_node=not bool(args.prefer_processor),
    )
    result = client.matte(
        image,
        background_color=args.background_color,
        hint_alpha=hint,
        hint_source=f"probe_full_frame_hint_{float(args.hint_value):.3f}",
        gamma_space=args.gamma_space,
        screen_color=args.screen_color,
        despill_strength=args.despill_strength,
        refiner_strength=args.refiner_strength,
        auto_despeckle=args.auto_despeckle,
        despeckle_size=args.despeckle_size,
        apply_color_protection=bool(args.color_protection),
        execution_profile="corridorkey-character",
    )

    Image.fromarray(image, mode="RGB").save(out_dir / "input.png")
    Image.fromarray(result.rgba, mode="RGBA").save(out_dir / "rgba.png")
    Image.fromarray(result.foreground_srgb, mode="RGB").save(out_dir / "foreground.png")
    Image.fromarray(_checker_composite(result.rgba), mode="RGB").save(out_dir / "checker.png")
    _save_mask(out_dir / "hint.png", result.hint_alpha)
    _save_mask(out_dir / "alpha.png", result.alpha)
    _save_mask(out_dir / "raw_alpha.png", result.raw_alpha)
    _save_mask(out_dir / "color_protection_alpha.png", result.color_protection_alpha)

    alpha_u8 = result.rgba[..., 3]
    summary: dict[str, Any] = {
        "input": str(input_path),
        "background_color": list(args.background_color),
        "hint_value": float(args.hint_value),
        "outputs": {
            "rgba": "rgba.png",
            "alpha": "alpha.png",
            "raw_alpha": "raw_alpha.png",
            "foreground": "foreground.png",
            "hint": "hint.png",
            "checker": "checker.png",
            "color_protection_alpha": "color_protection_alpha.png",
        },
        "alpha": {
            "mean": float(result.alpha.mean()),
            "gt_32": int((alpha_u8 > 32).sum()),
            "gt_128": int((alpha_u8 > 128).sum()),
            "gt_220": int((alpha_u8 > 220).sum()),
        },
        "debug": result.debug,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
