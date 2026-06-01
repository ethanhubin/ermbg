"""Render RiskRegion overlays and planner context for one matte result.

Usage:
    .venv/bin/python scripts/06_risk_overlay.py \
        --input samples/legacy/inputs/10.png \
        --rgba samples/legacy/outputs/matte/10_rgba.png \
        --background 255,255,255 \
        --out out/risk_overlays/sample_10_risk_overlay.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg.keyer import key_alpha
from ermbg.planner import RiskRegion, build_planner_prompt_bundle
from ermbg.risk import (
    coalesce_risk_regions,
    extract_alpha_keyer_disagreement_regions,
    extract_hard_edge_candidate_regions,
    extract_same_bg_enclosed_regions,
)


COLORS = {
    "same_bg_enclosed_region": (255, 0, 255),
    "alpha_keyer_disagreement": (0, 220, 255),
    "hard_edge_candidate": (255, 170, 0),
}

DRAW_ORDER = {
    "alpha_keyer_disagreement": 0,
    "hard_edge_candidate": 1,
    "same_bg_enclosed_region": 2,
}


def _parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("background must be R,G,B")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("background values must be integers") from e
    if any(c < 0 or c > 255 for c in rgb):
        raise argparse.ArgumentTypeError("background values must be in [0,255]")
    return rgb  # type: ignore[return-value]


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _checker_composite(rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // cell + yy // cell) % 2) * 70 + 185).astype(np.uint8)
    checker_rgb = np.dstack([checker, checker, checker])
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3].astype(np.float32) * alpha + checker_rgb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def _region_summary(region: RiskRegion) -> dict[str, Any]:
    d = region.to_prompt_dict()
    # Larger regions first within each kind; this makes downstream inspection
    # stable without hiding any raw regions.
    d["priority"] = float(d["area"])
    return d


def extract_risk_regions(
    image: np.ndarray,
    rgba: np.ndarray,
    background: tuple[int, int, int],
) -> tuple[list[RiskRegion], dict[str, Any]]:
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    same_regions, same_info = extract_same_bg_enclosed_regions(image, rgba, background)
    full_key = key_alpha(image, background, mode="chromatic")
    alpha_keyer_regions, alpha_keyer_info = extract_alpha_keyer_disagreement_regions(alpha, full_key)
    lum_key = key_alpha(image, background, mode="luminance")
    hard_edge_regions, hard_edge_info = extract_hard_edge_candidate_regions(image, alpha, lum_key, background)
    regions = same_regions + alpha_keyer_regions + hard_edge_regions
    info = {
        "same_bg_enclosed_region": same_info,
        "alpha_keyer_disagreement": alpha_keyer_info,
        "hard_edge_candidate": hard_edge_info,
    }
    return regions, info


def _counts(regions: list[RiskRegion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        counts[region.kind] = counts.get(region.kind, 0) + 1
    return {kind: counts.get(kind, 0) for kind in COLORS}


def render_overlay(
    image: np.ndarray,
    rgba: np.ndarray,
    regions: list[RiskRegion],
) -> Image.Image:
    h, w = image.shape[:2]
    base_rgb = _checker_composite(rgba)
    overlay = image.copy()

    for region in sorted(regions, key=lambda r: DRAW_ORDER.get(r.kind, 99)):
        color = COLORS.get(region.kind, (255, 255, 0))
        mask = region.mask.astype(bool)
        color_arr = np.asarray(color, dtype=np.float32)
        overlay[mask] = (0.55 * overlay[mask].astype(np.float32) + 0.45 * color_arr).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    legend_h = 104
    canvas = Image.new("RGB", (w * 3, h + legend_h), (245, 245, 245))
    canvas.paste(Image.fromarray(image), (0, 0))
    canvas.paste(Image.fromarray(base_rgb), (w, 0))
    canvas.paste(Image.fromarray(overlay), (2 * w, 0))
    draw = ImageDraw.Draw(canvas)

    for i, label in enumerate(("input", "base rgba on checker", "risk overlay")):
        draw.rectangle([i * w, 0, i * w + 180, 26], fill=(0, 0, 0))
        draw.text((i * w + 8, 6), label, fill=(255, 255, 255))

    counts = _counts(regions)

    legend_y = h + 14
    for row, (kind, color) in enumerate(COLORS.items()):
        y = legend_y + row * 24
        draw.rectangle([16, y, 34, y + 18], fill=color)
        draw.text((42, y + 2), f"{kind}: {counts.get(kind, 0)}", fill=(20, 20, 20))
    draw.text((360, h + 14), json.dumps(counts, ensure_ascii=False), fill=(20, 20, 20))
    return canvas


def run(
    input_path: Path,
    rgba_path: Path,
    background: tuple[int, int, int],
    out_path: Path,
    *,
    coalesce: bool = True,
    merge_distance_px: int = 3,
) -> dict[str, Any]:
    image = _load_rgb(input_path)
    rgba = _load_rgba(rgba_path)
    if image.shape[:2] != rgba.shape[:2]:
        raise ValueError("input and rgba must share HxW")

    raw_regions, extraction_info = extract_risk_regions(image, rgba, background)
    regions = (
        coalesce_risk_regions(raw_regions, merge_distance_px=merge_distance_px)
        if coalesce
        else raw_regions
    )
    kind_order = {
        "same_bg_enclosed_region": 0,
        "hard_edge_candidate": 1,
        "alpha_keyer_disagreement": 2,
    }
    region_summary = sorted(
        (_region_summary(r) for r in regions),
        key=lambda r: (kind_order.get(str(r["kind"]), 99), -r["area"]),
    )
    bundle = build_planner_prompt_bundle(
        image_shape=image.shape,
        regions=regions,
        background_color=background,
        instructions=[
            "Return CandidatePlan JSON only.",
            "Use only registered tools and existing region_id values.",
            "Do not output alpha, RGBA, masks, or image-processing code.",
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(image, rgba, regions).save(out_path)
    json_path = out_path.with_suffix(".regions.json")
    bundle_path = out_path.with_suffix(".planner_bundle.json")
    payload = {
        "input": str(input_path),
        "rgba": str(rgba_path),
        "background_color": list(background),
        "raw_counts": _counts(raw_regions),
        "counts": _counts(regions),
        "coalesced": coalesce,
        "merge_distance_px": merge_distance_px if coalesce else 0,
        "extraction_info": extraction_info,
        "regions": region_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        **payload,
        "overlay": str(out_path),
        "regions_json": str(json_path),
        "planner_bundle_json": str(bundle_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Original RGB image")
    p.add_argument("--rgba", type=Path, required=True, help="Base matte RGBA PNG")
    p.add_argument("--background", type=_parse_rgb, required=True, help="Known background as R,G,B")
    p.add_argument("--out", type=Path, required=True, help="Output overlay PNG path")
    p.add_argument(
        "--coalesce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge nearby same-kind risk fragments for planner/VLM readability.",
    )
    p.add_argument("--merge-distance-px", type=int, default=3, help="Coalesce dilation distance in pixels")
    args = p.parse_args()

    result = run(
        args.input,
        args.rgba,
        args.background,
        args.out,
        coalesce=args.coalesce,
        merge_distance_px=args.merge_distance_px,
    )
    print(result["overlay"])
    print(result["regions_json"])
    print(result["planner_bundle_json"])
    print(json.dumps({"raw": result["raw_counts"], "shown": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
