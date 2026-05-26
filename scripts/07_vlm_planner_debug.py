"""Export a VLM planner request and optionally execute fixture candidates.

Usage:
    .venv/bin/python scripts/07_vlm_planner_debug.py \
        --input samples/legacy/inputs/10.png \
        --rgba samples/legacy/outputs/matte/10_rgba.png \
        --background 255,255,255 \
        --out-dir out/vlm_debug/sample_10 \
        --fixture out/vlm_debug/sample_10/fixture_response.json

The script never calls a real VLM. It writes the local visual/request payload so
we can inspect whether the future model will see enough context. If a fixture
JSON response is provided, it is parsed through the same CandidatePlan parser
and local executor used by future live clients.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg.executor import execute_plans
from ermbg.keyer import key_alpha
from ermbg.planner import CandidatePlan, RiskRegion, build_planner_prompt_bundle
from ermbg.risk import (
    coalesce_risk_regions,
    extract_alpha_keyer_disagreement_regions,
    extract_hard_edge_candidate_regions,
    extract_same_bg_enclosed_regions,
)
from ermbg.vlm_payload import VLMPlannerRequest, build_vlm_planner_request
from ermbg.vlm_planner import FixturePlannerClient
from ermbg.vlm_openai import OpenAIVLMPlannerClient


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


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, mode="RGBA").save(path)


def extract_debug_regions(
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    coalesce: bool = True,
    merge_distance_px: int = 3,
) -> tuple[list[RiskRegion], dict[str, Any]]:
    """Extract the same evidence families used by the risk overlay workflow."""
    alpha = base_rgba[..., 3].astype(np.float32) / 255.0
    same_regions, same_info = extract_same_bg_enclosed_regions(image_srgb, base_rgba, background_color)
    chroma_key = key_alpha(image_srgb, background_color, mode="chromatic")
    alpha_keyer_regions, alpha_keyer_info = extract_alpha_keyer_disagreement_regions(alpha, chroma_key)
    lum_key = key_alpha(image_srgb, background_color, mode="luminance")
    hard_edge_regions, hard_edge_info = extract_hard_edge_candidate_regions(
        image_srgb,
        alpha,
        lum_key,
        background_color,
    )
    raw_regions = same_regions + alpha_keyer_regions + hard_edge_regions
    regions = (
        coalesce_risk_regions(raw_regions, merge_distance_px=merge_distance_px)
        if coalesce
        else raw_regions
    )
    return regions, {
        "raw_counts": _counts(raw_regions),
        "counts": _counts(regions),
        "coalesced": coalesce,
        "merge_distance_px": merge_distance_px if coalesce else 0,
        "extraction_info": {
            "same_bg_enclosed_region": same_info,
            "alpha_keyer_disagreement": alpha_keyer_info,
            "hard_edge_candidate": hard_edge_info,
        },
    }


def _counts(regions: list[RiskRegion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        counts[region.kind] = counts.get(region.kind, 0) + 1
    return {
        "same_bg_enclosed_region": counts.get("same_bg_enclosed_region", 0),
        "alpha_keyer_disagreement": counts.get("alpha_keyer_disagreement", 0),
        "hard_edge_candidate": counts.get("hard_edge_candidate", 0),
    }


def _write_request(request: VLMPlannerRequest, out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = request.to_dict()
    attachment_dir = out_dir / "attachments"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for attachment in payload["attachments"]:
        filename = f"{attachment['id']}.png"
        png_path = attachment_dir / filename
        png_path.write_bytes(base64.b64decode(str(attachment["data_base64"])))
        manifest.append(
            {
                "id": attachment["id"],
                "purpose": attachment["purpose"],
                "region_id": attachment.get("region_id"),
                "width": attachment["width"],
                "height": attachment["height"],
                "path": str(png_path),
                "metadata": attachment.get("metadata", {}),
            }
        )

    (out_dir / "vlm_request.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (out_dir / "attachments_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _execute_fixture(
    fixture_path: Path,
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    out_dir: Path,
) -> list[dict[str, Any]]:
    client = FixturePlannerClient(fixture_path)
    bundle = build_planner_prompt_bundle(
        image_shape=image_srgb.shape,
        regions=regions,
        background_color=background_color,
    )
    plans: list[CandidatePlan] = client.plan(bundle)
    results = execute_plans(plans, regions, image_srgb, base_rgba, background_color=background_color)
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_payloads: list[dict[str, Any]] = []

    shutil.copyfile(fixture_path, out_dir / "vlm_response.json")
    for result in results:
        filename = f"{result.plan.id}.png"
        path = candidate_dir / filename
        _save_rgba(path, result.rgba)
        candidate_payloads.append(
            {
                **result.debug_dict(),
                "id": result.plan.id,
                "label": result.plan.label,
                "selected": result.plan.selected,
                "path": str(path),
            }
        )

    (out_dir / "candidate_plans.json").write_text(
        json.dumps([result.plan.to_dict() for result in results], indent=2, ensure_ascii=False)
    )
    (out_dir / "candidate_results.json").write_text(
        json.dumps(candidate_payloads, indent=2, ensure_ascii=False)
    )
    return candidate_payloads


def _execute_openai(
    request: VLMPlannerRequest,
    regions: list[RiskRegion],
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    background_color: tuple[int, int, int],
    out_dir: Path,
    *,
    model: str,
    timeout: float,
    env_path: Path | None,
) -> list[dict[str, Any]]:
    client = OpenAIVLMPlannerClient(
        model=model,
        timeout=timeout,
        env_path=env_path,
    )
    plans = client.plan_request(request)
    if client.last_request_payload is not None:
        (out_dir / "openai_request.json").write_text(
            json.dumps(client.last_request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if client.last_raw_response is not None:
        (out_dir / "vlm_raw_response.json").write_text(
            json.dumps(client.last_raw_response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    results = execute_plans(plans, regions, image_srgb, base_rgba, background_color=background_color)
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_payloads: list[dict[str, Any]] = []

    for result in results:
        filename = f"{result.plan.id}.png"
        path = candidate_dir / filename
        _save_rgba(path, result.rgba)
        candidate_payloads.append(
            {
                **result.debug_dict(),
                "id": result.plan.id,
                "label": result.plan.label,
                "selected": result.plan.selected,
                "path": str(path),
            }
        )

    (out_dir / "candidate_plans.json").write_text(
        json.dumps([result.plan.to_dict() for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "candidate_results.json").write_text(
        json.dumps(candidate_payloads, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return candidate_payloads


def run(
    input_path: Path,
    rgba_path: Path,
    background_color: tuple[int, int, int],
    out_dir: Path,
    *,
    fixture_path: Path | None = None,
    provider: str = "none",
    openai_model: str = "gpt-4o-mini",
    openai_timeout: float = 120.0,
    env_path: Path | None = None,
    coalesce: bool = True,
    merge_distance_px: int = 3,
    max_region_crops: int = 8,
) -> dict[str, Any]:
    image_srgb = _load_rgb(input_path)
    base_rgba = _load_rgba(rgba_path)
    if image_srgb.shape[:2] != base_rgba.shape[:2]:
        raise ValueError("input and rgba must share HxW")

    regions, evidence_info = extract_debug_regions(
        image_srgb,
        base_rgba,
        background_color,
        coalesce=coalesce,
        merge_distance_px=merge_distance_px,
    )
    request = build_vlm_planner_request(
        image_srgb=image_srgb,
        base_rgba=base_rgba,
        regions=regions,
        background_color=background_color,
        max_region_crops=max_region_crops,
    )
    manifest = _write_request(request, out_dir)
    summary: dict[str, Any] = {
        "input": str(input_path),
        "rgba": str(rgba_path),
        "background_color": list(background_color),
        "out_dir": str(out_dir),
        "vlm_request_json": str(out_dir / "vlm_request.json"),
        "attachments_manifest_json": str(out_dir / "attachments_manifest.json"),
        "region_count": len(regions),
        "attachment_count": len(manifest),
        "provider": provider,
        **evidence_info,
    }

    if fixture_path is not None and provider != "none":
        raise ValueError("--fixture and --provider are mutually exclusive")

    if fixture_path is not None:
        candidate_payloads = _execute_fixture(
            fixture_path,
            regions,
            image_srgb,
            base_rgba,
            background_color,
            out_dir,
        )
        summary.update(
            {
                "vlm_response_json": str(out_dir / "vlm_response.json"),
                "candidate_plans_json": str(out_dir / "candidate_plans.json"),
                "candidate_results_json": str(out_dir / "candidate_results.json"),
                "candidate_count": len(candidate_payloads),
                "candidate_paths": [payload["path"] for payload in candidate_payloads],
            }
        )
    elif provider == "openai":
        candidate_payloads = _execute_openai(
            request,
            regions,
            image_srgb,
            base_rgba,
            background_color,
            out_dir,
            model=openai_model,
            timeout=openai_timeout,
            env_path=env_path,
        )
        summary.update(
            {
                "openai_request_json": str(out_dir / "openai_request.json"),
                "vlm_raw_response_json": str(out_dir / "vlm_raw_response.json"),
                "candidate_plans_json": str(out_dir / "candidate_plans.json"),
                "candidate_results_json": str(out_dir / "candidate_results.json"),
                "candidate_count": len(candidate_payloads),
                "candidate_paths": [payload["path"] for payload in candidate_payloads],
                "openai_model": openai_model,
            }
        )
    elif provider != "none":
        raise ValueError(f"unknown provider: {provider}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Original RGB/RGBA image")
    p.add_argument("--rgba", type=Path, required=True, help="Base matte RGBA PNG")
    p.add_argument("--background", type=_parse_rgb, required=True, help="Known background as R,G,B")
    p.add_argument("--out-dir", type=Path, required=True, help="Output debug directory")
    p.add_argument("--fixture", type=Path, help="Optional fixture VLM response JSON")
    p.add_argument(
        "--provider",
        choices=["none", "openai"],
        default="none",
        help="Optional live VLM provider. Defaults to no network call.",
    )
    p.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI vision model for --provider openai")
    p.add_argument("--openai-timeout", type=float, default=120.0, help="OpenAI request timeout in seconds")
    p.add_argument("--env", type=Path, default=Path(".env"), help="Optional .env path for OPENAI_API_KEY")
    p.add_argument(
        "--coalesce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge nearby same-kind evidence fragments for planner readability.",
    )
    p.add_argument("--merge-distance-px", type=int, default=3)
    p.add_argument("--max-region-crops", type=int, default=8)
    args = p.parse_args()

    summary = run(
        args.input,
        args.rgba,
        args.background,
        args.out_dir,
        fixture_path=args.fixture,
        provider=args.provider,
        openai_model=args.openai_model,
        openai_timeout=args.openai_timeout,
        env_path=args.env,
        coalesce=args.coalesce,
        merge_distance_px=args.merge_distance_px,
        max_region_crops=args.max_region_crops,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
