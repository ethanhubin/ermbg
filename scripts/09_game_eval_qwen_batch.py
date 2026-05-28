"""Run Game Eval for G01-G09, green and white variants, using Comfy Qwen.

The batch writes a web-readable run under ``out/``. Batch names intentionally
include a small version number because a single day often has multiple runs:

    out/vlm_eval_game_qwen_gw_v001_YYYYMMDD/
      matte/<case_id>/<green|white>/
      vlm_qwen/<case_id>/<green|white>/
      vlm_qwen/eval_report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg import io
from ermbg.comfy import DEFAULT_COMFY_URL
from ermbg.executor import execute_plans
from ermbg.matting import matte as run_matte
from ermbg.segmenter import build_segmenter
from ermbg.vlm_payload import VLMPlannerRequest, build_vlm_planner_request
from ermbg.vlm_planner import parse_candidate_plans
from ermbg.vlm_semantic import (
    ComfyQwenVLMSemanticPriorClient,
    _extract_comfy_preview_text,
    build_vlm_semantic_request,
    extract_shadow_candidate_regions,
    parse_qwen_json_text,
)

import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "samples" / "vlm_eval_game" / "manifest.json"


def _load_vlm_debug_module():
    path = PROJECT_ROOT / "scripts" / "07_vlm_planner_debug.py"
    spec = importlib.util.spec_from_file_location("ermbg_vlm_planner_debug", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VLM_DEBUG = _load_vlm_debug_module()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")


def _load_cases(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a cases list")
    return [case for case in cases if isinstance(case, dict)]


def _save_matte_outputs(
    *,
    input_path: Path,
    image_srgb: np.ndarray,
    segmenter: Any,
    out_dir: Path,
) -> dict[str, Any]:
    result = run_matte(image_srgb, segmenter=segmenter)
    out_dir.mkdir(parents=True, exist_ok=True)
    io.save_rgb(out_dir / "input.png", image_srgb)
    io.save_rgba(out_dir / "rgba.png", result.rgba)
    io.save_mask(out_dir / "alpha.png", result.alpha)
    io.save_mask(out_dir / "shadow.png", result.debug["shadow_alpha"])
    # shadow.png is display-safe for visual review; shadow_physical.png keeps
    # the measured linear darkening strength for debugging thresholds.
    io.save_mask(out_dir / "shadow_physical.png", result.debug["shadow_alpha_physical"])
    io.save_rgb(out_dir / "foreground.png", result.foreground_srgb)
    io.save_mask(out_dir / "trimap.png", result.debug["trimap_u8"])
    report = {
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis is not None else None,
        "background_color": list(result.background_color),
        "despill_method": result.debug.get("despill_method"),
        "matting_model": "ZhengPeng7/BiRefNet-matting",
        "keyer": result.debug.get("keyer", {}),
        "shadow": result.debug.get("shadow", {}),
        "semantic_prior": result.debug.get("semantic_prior", {}),
        "strategy": result.debug.get("strategy", {}),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    shadow_alpha_physical = np.asarray(result.debug["shadow_alpha_physical"], dtype=np.float32)
    return {
        "input": _rel(input_path),
        "out_dir": _rel(out_dir),
        "rgba": _rel(out_dir / "rgba.png"),
        "alpha": _rel(out_dir / "alpha.png"),
        "shadow": _rel(out_dir / "shadow.png"),
        "shadow_physical": _rel(out_dir / "shadow_physical.png"),
        "foreground": _rel(out_dir / "foreground.png"),
        "trimap": _rel(out_dir / "trimap.png"),
        "report": _rel(out_dir / "report.json"),
        "background_color": list(result.background_color),
        "diagnosis": report["diagnosis"],
        "despill_method": report["despill_method"],
        "keyer": report["keyer"],
        "strategy": report["strategy"],
    }, result.rgba, shadow_alpha_physical > 0.0


def _contact_sheet(request: VLMPlannerRequest) -> np.ndarray:
    tiles: list[tuple[str, Image.Image]] = []
    for attachment in request.attachments:
        raw = base64.b64decode(attachment.data_base64)
        image = Image.open(BytesIO(raw)).convert("RGB")
        label = attachment.id if attachment.region_id is None else f"{attachment.id} {attachment.region_id}"
        tiles.append((label, image))

    tile_w, tile_h = 420, 340
    label_h = 28
    cols = 2 if len(tiles) <= 4 else 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        draw.text((x + 8, y + 8), label[:62], fill=(0, 0, 0))
        im = image.copy()
        im.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(im, (x + (tile_w - im.width) // 2, y + label_h + (tile_h - im.height) // 2))
    return np.asarray(sheet, dtype=np.uint8)


def _qwen_prompt(request: VLMPlannerRequest) -> str:
    manifest = [
        {
            "id": attachment.id,
            "purpose": attachment.purpose,
            "region_id": attachment.region_id,
            "width": attachment.width,
            "height": attachment.height,
            "metadata": attachment.metadata,
        }
        for attachment in request.attachments
    ]
    user_payload = {
        "planner_bundle": request.planner_bundle,
        "attachment_manifest": manifest,
        "required_json_shape": {
            "candidates": [
                {
                    "id": "short unique id",
                    "label": "short label",
                    "confidence": "0..1",
                    "selected": "boolean; exactly one true if candidates are returned",
                    "operations": [
                        {
                            "tool": "one supplied tool name",
                            "region_id": "one supplied region id allowed for that tool",
                            "parameters": {},
                        }
                    ],
                    "reason": "short reason",
                }
            ],
        },
    }
    return (
        "You are ERMBG's visual planning layer. The attached image is a labeled "
        "contact sheet with the original image, base matte previews, evidence "
        "overlay, and region crops. Interpret ONLY the supplied evidence regions.\n\n"
        "Return raw JSON only. No markdown. Do not invent tools or region ids. "
        "Do not output masks, alpha, RGBA, code, or prose outside JSON. Use only "
        "tools whose allowed_region_kinds match the chosen region kind. Each "
        "candidate must be a whole-image matte interpretation, not one candidate "
        "per region. Prefer one selected candidate unless the evidence is truly "
        "ambiguous.\n\n"
        "Policy for same_bg_enclosed_region: prefer preserve_hole when the region "
        "is an opening, slot, window, counter-shape, or cutout whose pixels match "
        "the known background. Use fill_same_color_region only when the crop clearly "
        "shows foreground artwork/material that intentionally has the same color as "
        "the background. For UI frames and buttons, interior apertures are normally "
        "transparent holes, not foreground fill.\n\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )


def _run_qwen_planner(
    *,
    request: VLMPlannerRequest,
    out_dir: Path,
    comfy_url: str,
    qwen_model: str,
    timeout: float,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = _contact_sheet(request)
    io.save_rgb(out_dir / "qwen_contact_sheet.png", sheet)
    client = ComfyQwenVLMSemanticPriorClient(
        url=comfy_url,
        model=qwen_model,
        timeout=timeout,
        max_new_tokens=2200,
        temperature=0.05,
    )
    server_name = client._upload_image(sheet, f"ermbg_game_eval_qwen_{uuid.uuid4().hex[:8]}.png")
    prompt = _qwen_prompt(request)
    workflow = client._build_workflow(server_name, prompt)
    started = time.monotonic()
    prompt_id = client._queue(workflow)
    history = client._wait(prompt_id)
    elapsed = time.monotonic() - started
    raw_text = _extract_comfy_preview_text(history)
    payload = parse_qwen_json_text(raw_text)
    plans = parse_candidate_plans(payload)
    (out_dir / "qwen_prompt.txt").write_text(prompt, encoding="utf-8")
    (out_dir / "qwen_workflow.json").write_text(json.dumps(workflow, indent=2, ensure_ascii=False))
    (out_dir / "qwen_raw_response.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "qwen_response.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return plans, {
        "prompt_id": prompt_id,
        "server_image": server_name,
        "elapsed_sec": elapsed,
        "model": qwen_model,
    }, payload


def _drop_unknown_region_operations(plans: list[Any], regions: list[Any]) -> list[dict[str, Any]]:
    valid_region_ids = {str(region.id) for region in regions}
    dropped: list[dict[str, Any]] = []
    for plan in plans:
        kept_ops = []
        for op in plan.operations:
            if op.region_id in valid_region_ids:
                kept_ops.append(op)
                continue
            # Qwen can occasionally enumerate a visual label outside the budgeted
            # planner region set. The batch should keep valid operations and
            # record the invalid reference instead of failing the whole run.
            dropped.append(
                {
                    "plan_id": plan.id,
                    "tool": op.tool,
                    "region_id": op.region_id,
                }
            )
        plan.operations = kept_ops
    return dropped


def _budget_regions(regions: list[Any], max_regions: int) -> list[Any]:
    if max_regions <= 0 or len(regions) <= max_regions:
        return list(regions)
    priority = {
        "owned_shadow_candidate": 0,
        "same_bg_enclosed_region": 1,
        "alpha_keyer_disagreement": 2,
        "hard_edge_candidate": 3,
    }
    ranked = sorted(
        regions,
        key=lambda region: (
            priority.get(getattr(region, "kind", "unknown"), 9),
            -int(getattr(region, "mask").sum()),
            str(getattr(region, "id", "")),
        ),
    )
    return ranked[:max_regions]


def _budget_tool_regions(regions: list[Any], max_regions: int) -> list[Any]:
    tool_regions = [region for region in regions if getattr(region, "kind", "") != "owned_shadow_candidate"]
    if max_regions <= 0 or len(tool_regions) <= max_regions:
        return list(tool_regions)
    priority = {
        "same_bg_enclosed_region": 0,
        "alpha_keyer_disagreement": 1,
        "hard_edge_candidate": 2,
    }
    ranked = sorted(
        tool_regions,
        key=lambda region: (
            priority.get(getattr(region, "kind", "unknown"), 9),
            -int(getattr(region, "mask").sum()),
            str(getattr(region, "id", "")),
        ),
    )
    return ranked[:max_regions]


def _region_counts(regions: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        kind = str(getattr(region, "kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "same_bg_enclosed_region": counts.get("same_bg_enclosed_region", 0),
        "alpha_keyer_disagreement": counts.get("alpha_keyer_disagreement", 0),
        "hard_edge_candidate": counts.get("hard_edge_candidate", 0),
        "owned_shadow_candidate": counts.get("owned_shadow_candidate", 0),
    }


def _semantic_regions(payload: dict[str, Any], shadow_region_ids: set[str]) -> list[dict[str, Any]]:
    raw = payload.get("semantic_regions")
    if not isinstance(raw, list):
        raw = payload.get("regions")
    if not isinstance(raw, list):
        return []

    semantic: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        region_id = item.get("region_id")
        role = item.get("role")
        if not isinstance(region_id, str) or region_id not in shadow_region_ids:
            continue
        if not isinstance(role, str):
            continue
        try:
            confidence = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        semantic.append(
            {
                "region_id": region_id,
                "role": role,
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "reason": str(item.get("reason", "")),
            }
        )
    return semantic


def _write_semantic_request(request: Any, out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    attachment_dir = out_dir / "attachments"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    attachments: list[dict[str, Any]] = []
    for attachment in request.attachments:
        raw = base64.b64decode(attachment.data_base64)
        path = attachment_dir / f"{attachment.id}.png"
        path.write_bytes(raw)
        item = {
            "id": attachment.id,
            "purpose": attachment.purpose,
            "mime_type": attachment.mime_type,
            "width": attachment.width,
            "height": attachment.height,
            "path": _rel(path),
            "metadata": dict(attachment.metadata),
        }
        if attachment.region_id is not None:
            item["region_id"] = attachment.region_id
        attachments.append(item)
    payload = {
        "system_prompt": request.system_prompt,
        "image": request.image,
        "regions": request.regions,
        "response_schema": request.response_schema,
        "attachments": attachments,
    }
    (out_dir / "vlm_semantic_request.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "attachments_manifest.json").write_text(
        json.dumps(attachments, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return attachments


def _run_qwen_shadow_semantic(
    *,
    image_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    background_color: tuple[int, int, int],
    shadow_regions: list[Any],
    out_dir: Path,
    comfy_url: str,
    qwen_model: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not shadow_regions:
        return [], {"skipped": True, "reason": "no shadow candidates"}

    request = build_vlm_semantic_request(
        image_srgb=image_srgb,
        subject_alpha=subject_alpha,
        background_color=background_color,
        regions=shadow_regions,
        thumbnail_max_side=640,
        crop_max_side=320,
    )
    manifest = _write_semantic_request(request, out_dir)
    client = ComfyQwenVLMSemanticPriorClient(
        url=comfy_url,
        model=qwen_model,
        timeout=timeout,
        max_new_tokens=1200,
        temperature=0.05,
    )
    started = time.monotonic()
    prior = client.classify_request(request, shadow_regions, subject_alpha.shape)
    elapsed = time.monotonic() - started
    raw_text = client.last_raw_text or ""
    payload = parse_qwen_json_text(raw_text) if raw_text else {"regions": [], "shadow_allowed": True}
    semantic_regions = _semantic_regions(payload, {str(region.id) for region in shadow_regions})
    (out_dir / "qwen_workflow.json").write_text(
        json.dumps(client.last_workflow or {}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "qwen_raw_response.txt").write_text(raw_text, encoding="utf-8")
    (out_dir / "qwen_response.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "semantic_prior.json").write_text(
        json.dumps(prior.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return semantic_regions, {
        "elapsed_sec": elapsed,
        "model": qwen_model,
        "attachment_count": len(manifest),
        "semantic_prior": prior.to_dict(),
        "qwen_response_json": _rel(out_dir / "qwen_response.json"),
    }


def _shadow_policy_eval(
    *,
    target_policy: list[Any],
    semantic_regions: list[dict[str, Any]],
    shadow_candidate_count: int,
) -> dict[str, Any]:
    required = "shadow_or_contact" in {str(item) for item in target_policy}
    if not required:
        return {
            "shadow_policy_required": False,
            "shadow_policy_hit": None,
            "shadow_policy_reason": "not required",
        }
    if shadow_candidate_count <= 0:
        return {
            "shadow_policy_required": True,
            "shadow_policy_hit": False,
            "shadow_policy_reason": "no shadow candidate was supplied",
        }
    accepted = [
        item
        for item in semantic_regions
        if item.get("role") == "shadow" and float(item.get("confidence", 0.0) or 0.0) >= 0.55
    ]
    return {
        "shadow_policy_required": True,
        "shadow_policy_hit": bool(accepted),
        "shadow_policy_reason": "owned shadow accepted" if accepted else "no owned shadow accepted by VLM",
    }


def _accepted_shadow_mask(
    shadow_regions: list[Any],
    semantic_regions: list[dict[str, Any]],
    shape: tuple[int, int],
) -> np.ndarray:
    accepted_ids = {
        str(item.get("region_id"))
        for item in semantic_regions
        if str(item.get("role", "")).lower() == "shadow"
    }
    mask = np.zeros(shape, dtype=bool)
    for region in shadow_regions:
        if str(region.id) in accepted_ids:
            mask |= np.asarray(region.mask, dtype=bool)
    return mask


def _run_variant_vlm(
    *,
    input_path: Path,
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    base_shadow_mask: np.ndarray,
    background_color: tuple[int, int, int],
    target_policy: list[Any],
    out_dir: Path,
    comfy_url: str,
    qwen_model: str,
    timeout: float,
    max_regions: int,
    max_region_crops: int,
) -> dict[str, Any]:
    debug_regions, evidence_info = VLM_DEBUG.extract_debug_regions(
        image_srgb,
        base_rgba,
        background_color,
        coalesce=True,
        merge_distance_px=3,
    )
    subject_alpha = base_rgba[..., 3].astype(np.float32) / 255.0
    shadow_regions: list[Any] = []
    if "shadow_or_contact" in {str(item) for item in target_policy}:
        shadow_regions = extract_shadow_candidate_regions(
            image_srgb,
            subject_alpha,
            background_color,
            max_regions=4,
        )

    tool_regions = _budget_tool_regions(debug_regions, max_regions)
    shadow_regions = _budget_regions(shadow_regions, 4)
    all_regions = list(tool_regions) + list(shadow_regions)
    shadow_region_ids = {str(region.id) for region in shadow_regions}
    evidence_info = dict(evidence_info)
    evidence_info["counts"] = _region_counts(all_regions)
    evidence_info["raw_counts"] = {
        **evidence_info.get("raw_counts", {}),
        "owned_shadow_candidate": len(shadow_regions),
    }
    extraction_info = dict(evidence_info.get("extraction_info", {}))
    extraction_info["owned_shadow_candidate"] = {
        "accepted_components": len(shadow_regions),
        "included_components": len(shadow_regions),
        "areas": [int(region.mask.sum()) for region in shadow_regions],
    }
    evidence_info["extraction_info"] = extraction_info
    request = build_vlm_planner_request(
        image_srgb=image_srgb,
        base_rgba=base_rgba,
        regions=tool_regions,
        background_color=background_color,
        max_region_crops=max_region_crops,
        thumbnail_max_side=640,
        crop_max_side=320,
    )
    manifest = VLM_DEBUG._write_request(request, out_dir)
    plans, qwen_info, qwen_payload = _run_qwen_planner(
        request=request,
        out_dir=out_dir,
        comfy_url=comfy_url,
        qwen_model=qwen_model,
        timeout=timeout,
    )
    del qwen_payload
    dropped_invalid_operations = _drop_unknown_region_operations(plans, tool_regions)
    if dropped_invalid_operations:
        qwen_info = {
            **qwen_info,
            "dropped_invalid_operations": dropped_invalid_operations,
        }
    semantic_regions, shadow_qwen_info = _run_qwen_shadow_semantic(
        image_srgb=image_srgb,
        subject_alpha=subject_alpha,
        background_color=background_color,
        shadow_regions=shadow_regions,
        out_dir=out_dir / "shadow_semantic",
        comfy_url=comfy_url,
        qwen_model=qwen_model,
        timeout=timeout,
    )
    shadow_policy = _shadow_policy_eval(
        target_policy=target_policy,
        semantic_regions=semantic_regions,
        shadow_candidate_count=len(shadow_region_ids),
    )
    # Accepted owned shadows are review/display pixels, not opaque foreground.
    # Protect them from VLM-selected repair tools that intentionally raise alpha.
    protected_shadow_mask = _accepted_shadow_mask(
        shadow_regions,
        semantic_regions,
        base_rgba.shape[:2],
    )
    if shadow_policy.get("shadow_policy_hit"):
        protected_shadow_mask |= np.asarray(base_shadow_mask, dtype=bool)
    (out_dir / "candidate_plans.json").write_text(
        json.dumps([plan.to_dict() for plan in plans], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    results = execute_plans(
        plans,
        tool_regions,
        image_srgb,
        base_rgba,
        background_color=background_color,
        protected_mask=protected_shadow_mask,
    )
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_payloads: list[dict[str, Any]] = []
    for result in results:
        path = candidate_dir / f"{result.plan.id}.png"
        io.save_rgba(path, result.rgba)
        candidate_payloads.append(
            {
                **result.debug_dict(),
                "id": result.plan.id,
                "label": result.plan.label,
                "selected": result.plan.selected,
                "path": _rel(path),
            }
        )
    (out_dir / "candidate_results.json").write_text(
        json.dumps(candidate_payloads, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "input": _rel(input_path),
        "rgba": _rel(out_dir.parents[2] / "matte_placeholder_should_be_overwritten.png"),
        "background_color": list(background_color),
        "out_dir": _rel(out_dir),
        "vlm_request_json": _rel(out_dir / "vlm_request.json"),
        "attachments_manifest_json": _rel(out_dir / "attachments_manifest.json"),
        "region_count": len(all_regions),
        "tool_region_count": len(tool_regions),
        "shadow_region_count": len(shadow_regions),
        "all_region_count": len(all_regions),
        "shadow_candidate_count": len(shadow_region_ids),
        "shadow_candidate_ids": sorted(shadow_region_ids),
        "protected_shadow_pixels": int(protected_shadow_mask.sum()),
        "semantic_regions": semantic_regions,
        **shadow_policy,
        "attachment_count": len(manifest),
        "provider": "comfy-qwen",
        "qwen": qwen_info,
        "shadow_qwen": shadow_qwen_info,
        **evidence_info,
        "qwen_response_json": _rel(out_dir / "qwen_response.json"),
        "candidate_plans_json": _rel(out_dir / "candidate_plans.json"),
        "candidate_results_json": _rel(out_dir / "candidate_results.json"),
        "candidate_count": len(candidate_payloads),
        "candidate_paths": [payload["path"] for payload in candidate_payloads],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _selected_tools(candidate_results_path: Path) -> tuple[list[str], list[str], str]:
    payload = json.loads(candidate_results_path.read_text(encoding="utf-8"))
    selected = [item for item in payload if isinstance(item, dict) and item.get("selected")]
    if not selected and isinstance(payload, list):
        selected = [item for item in payload if isinstance(item, dict)]
    plan_ids: list[str] = []
    tools: list[str] = []
    reason = ""
    for item in selected:
        plan = item.get("plan") if isinstance(item.get("plan"), dict) else {}
        plan_id = item.get("id") or plan.get("id")
        if isinstance(plan_id, str):
            plan_ids.append(plan_id)
        if not reason and isinstance(plan.get("reason"), str):
            reason = plan["reason"]
        operations = plan.get("operations")
        if isinstance(operations, list):
            for op in operations:
                if isinstance(op, dict) and isinstance(op.get("tool"), str) and op["tool"] not in tools:
                    tools.append(op["tool"])
    return plan_ids, tools, reason


def _expected_hit(selected_tools: list[str], expected_options: list[Any]) -> bool:
    selected = set(selected_tools)
    for option in expected_options:
        if isinstance(option, list) and all(isinstance(tool, str) and tool in selected for tool in option):
            return True
    return False


def _expected_tool_set(expected_options: list[Any]) -> set[str]:
    expected: set[str] = set()
    for option in expected_options:
        if isinstance(option, list):
            expected.update(str(tool) for tool in option if isinstance(tool, str))
    return expected


def _harmful_tools(selected_tools: list[str], expected_options: list[Any]) -> list[str]:
    selected = set(selected_tools)
    expected = _expected_tool_set(expected_options)
    harmful: list[str] = []
    if "preserve_hole" in expected and "fill_same_color_region" not in expected:
        if "fill_same_color_region" in selected:
            harmful.append("fill_same_color_region")
    if "fill_same_color_region" in expected and "preserve_hole" not in expected:
        if "preserve_hole" in selected:
            harmful.append("preserve_hole")
    return harmful


def _checker_composite(rgba: np.ndarray, cell: int = 28) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // cell + yy // cell) % 2) * 70 + 185).astype(np.uint8)
    bg = np.dstack([checker, checker, checker])
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgba[..., :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def _write_selected_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("selected_candidate_path")]
    if not ok_rows:
        return
    tile_w, tile_h, label_h, cols = 260, 260, 42, 6
    sheet = Image.new("RGB", (cols * tile_w, ((len(ok_rows) + cols - 1) // cols) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(ok_rows):
        rgba = np.asarray(Image.open(PROJECT_ROOT / row["selected_candidate_path"]).convert("RGBA"), dtype=np.uint8)
        tile = Image.fromarray(_checker_composite(rgba), mode="RGB")
        tile.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + label_h)
        draw.text((x + 6, y + 6), str(row["sample_code"]), fill=(0, 0, 0))
        draw.text((x + 6, y + 23), ",".join(row.get("selected_tools", []))[:36], fill=(0, 0, 0))
        sheet.paste(tile, (x + (tile_w - tile.width) // 2, y + label_h + (tile_h - tile.height) // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run(args: argparse.Namespace) -> None:
    cases = _load_cases(args.manifest)
    if args.sample_id:
        sample_ids = {item.strip() for item in args.sample_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("sample_id", "")) in sample_ids]
    if args.case_id:
        case_ids = {item.strip() for item in args.case_id.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("id", "")) in case_ids]
    if args.limit:
        cases = cases[: args.limit]
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    invalid_variants = sorted(set(variants) - {"green", "white"})
    if invalid_variants:
        raise ValueError(f"--variants only accepts green,white; got {','.join(invalid_variants)}")
    out_root = args.out_dir
    matte_root = out_root / "matte"
    vlm_root = out_root / "vlm_qwen"
    matte_root.mkdir(parents=True, exist_ok=True)
    vlm_root.mkdir(parents=True, exist_ok=True)

    existing_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume_existing:
        report_path = vlm_root / "eval_report.json"
        if report_path.exists():
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            for row in existing_report.get("rows", []):
                if isinstance(row, dict):
                    key = (str(row.get("sample_id", "")), str(row.get("sample_variant", "")))
                    existing_rows_by_key[key] = row

    segmenter = build_segmenter(backend=args.backend, model_id=args.matting_model)
    matte_summary_path = matte_root / "summary.json"
    matte_rows: list[dict[str, Any]] = []
    if args.resume_existing and matte_summary_path.exists():
        matte_rows = json.loads(matte_summary_path.read_text(encoding="utf-8"))
    report_rows_by_key = dict(existing_rows_by_key)
    total = len(cases) * len(variants)
    index = 0
    for case in cases:
        case_id = str(case["id"])
        sample_id = str(case.get("sample_id") or f"G{index + 1:02d}")
        for variant in variants:
            index += 1
            row_key = (sample_id, variant)
            sample_code = f"{sample_id}-{variant[:1].upper()}"
            input_path = PROJECT_ROOT / str(case[variant])
            matte_dir = matte_root / case_id / variant
            vlm_dir = vlm_root / case_id / variant
            print(f"[{index}/{total}] {sample_code} {case_id}/{variant}", flush=True)
            existing_row = existing_rows_by_key.get(row_key)
            summary_path = vlm_dir / "summary.json"
            candidate_results_path = vlm_dir / "candidate_results.json"
            if (
                args.resume_existing
                and existing_row
                and existing_row.get("status") == "ok"
                and summary_path.exists()
                and candidate_results_path.exists()
            ):
                print("  SKIP: existing ok", flush=True)
                continue
            row_base: dict[str, Any] = {
                "case_id": case_id,
                "sample_id": sample_id,
                "sample_code": sample_code,
                "sample_variant": variant,
                "category": case.get("category", ""),
                "primary_ambiguity": case.get("primary_ambiguity", ""),
                "shadow": case.get("shadow", {}),
                "expected_tool_options": case.get("expected", {}).get("target_tools", []),
                "target_policy": case.get("expected", {}).get("target_policy", []),
                "out_dir": _rel(vlm_dir),
                "status": "started",
            }
            try:
                image_srgb = io.load_rgb(input_path)
                matte_summary, rgba, base_shadow_mask = _save_matte_outputs(
                    input_path=input_path,
                    image_srgb=image_srgb,
                    segmenter=segmenter,
                    out_dir=matte_dir,
                )
                matte_summary.update(
                    {
                        "case_id": case_id,
                        "id": case_id,
                        "sample_id": sample_id,
                        "sample_code": sample_code,
                        "sample_variant": variant,
                    }
                )
                matte_rows.append(matte_summary)
                bg = tuple(int(c) for c in matte_summary["background_color"])
                vlm_summary = _run_variant_vlm(
                    input_path=input_path,
                    image_srgb=image_srgb,
                    base_rgba=rgba,
                    base_shadow_mask=base_shadow_mask,
                    background_color=bg,
                    target_policy=row_base["target_policy"],
                    out_dir=vlm_dir,
                    comfy_url=args.comfy_url,
                    qwen_model=args.qwen_model,
                    timeout=args.timeout,
                    max_regions=args.max_regions,
                    max_region_crops=args.max_region_crops,
                )
                vlm_summary["rgba"] = matte_summary["rgba"]
                (vlm_dir / "summary.json").write_text(json.dumps(vlm_summary, indent=2, ensure_ascii=False))
                selected_ids, selected_tools, selected_reason = _selected_tools(vlm_dir / "candidate_results.json")
                candidate_paths = vlm_summary.get("candidate_paths", [])
                selected_path = candidate_paths[0] if candidate_paths else None
                any_hit = _expected_hit(selected_tools, row_base["expected_tool_options"])
                harmful = _harmful_tools(selected_tools, row_base["expected_tool_options"])
                hit = any_hit and not harmful
                report_row = {
                    **row_base,
                    "status": "ok",
                    "background_color": list(bg),
                    "diagnosis_verdict": (matte_summary.get("diagnosis") or {}).get("verdict"),
                    "region_count": vlm_summary.get("region_count", 0),
                    "all_region_count": vlm_summary.get("all_region_count", 0),
                    "shadow_candidate_count": vlm_summary.get("shadow_candidate_count", 0),
                    "shadow_candidate_ids": vlm_summary.get("shadow_candidate_ids", []),
                    "semantic_regions": vlm_summary.get("semantic_regions", []),
                    "shadow_policy_required": vlm_summary.get("shadow_policy_required", False),
                    "shadow_policy_hit": vlm_summary.get("shadow_policy_hit"),
                    "shadow_policy_reason": vlm_summary.get("shadow_policy_reason", ""),
                    "counts": vlm_summary.get("counts", {}),
                    "raw_counts": vlm_summary.get("raw_counts", {}),
                    "candidate_count": vlm_summary.get("candidate_count", 0),
                    "selected_plan_ids": selected_ids,
                    "selected_tools": selected_tools,
                    "selected_reason": selected_reason,
                    "selected_candidate_path": selected_path,
                    "expected_any_hit": any_hit,
                    "harmful_tools": harmful,
                    "harmful_tool_selected": bool(harmful),
                    "expected_hit": hit,
                }
                print(
                    f"  regions={report_row['region_count']} candidates={report_row['candidate_count']} "
                    f"tools={','.join(selected_tools) or '-'} any_hit={any_hit} harmful={','.join(harmful) or '-'} hit={hit}",
                    flush=True,
                )
            except Exception as exc:
                report_row = {
                    **row_base,
                    "status": "error",
                    "error": str(exc),
                    "expected_any_hit": False,
                    "harmful_tools": [],
                    "harmful_tool_selected": False,
                    "expected_hit": False,
                }
                print(f"  ERROR: {exc}", flush=True)
            report_rows_by_key[row_key] = report_row
            report_rows = _ordered_report_rows(cases, variants, report_rows_by_key)
            (matte_root / "summary.json").write_text(json.dumps(matte_rows, indent=2, ensure_ascii=False))
            _write_eval_report(args, report_rows, out_root, matte_root, vlm_root)

    report_rows = _ordered_report_rows(cases, variants, report_rows_by_key)
    _write_selected_sheet(report_rows, vlm_root / "selected_candidates_checker_sheet.png")
    _write_eval_report(args, report_rows, out_root, matte_root, vlm_root)
    ok = sum(1 for row in report_rows if row.get("status") == "ok")
    hits = sum(1 for row in report_rows if row.get("expected_hit"))
    any_hits = sum(1 for row in report_rows if row.get("expected_any_hit"))
    harmful = sum(1 for row in report_rows if row.get("harmful_tool_selected"))
    print(
        f"Done: ok={ok}/{len(report_rows)} expected_hit={hits}/{len(report_rows)} "
        f"expected_any_hit={any_hits}/{len(report_rows)} harmful={harmful}/{len(report_rows)}"
    )
    print(f"Report: {vlm_root / 'eval_report.json'}")


def _ordered_report_rows(
    cases: list[dict[str, Any]],
    variants: tuple[str, ...],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        sample_id = str(case.get("sample_id") or f"G{idx + 1:02d}")
        for variant in variants:
            row = rows_by_key.get((sample_id, variant))
            if row is not None:
                rows.append(row)
    return rows


def _write_eval_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    out_root: Path,
    matte_root: Path,
    vlm_root: Path,
) -> None:
    ok = sum(1 for row in rows if row.get("status") == "ok")
    hits = sum(1 for row in rows if row.get("expected_hit"))
    any_hits = sum(1 for row in rows if row.get("expected_any_hit"))
    harmful = sum(1 for row in rows if row.get("harmful_tool_selected"))
    shadow_required = [row for row in rows if row.get("shadow_policy_required")]
    shadow_hits = sum(1 for row in shadow_required if row.get("shadow_policy_hit"))
    report = {
        "run_id": out_root.name,
        "model": f"comfy-qwen:{args.qwen_model}",
        "provider": "comfy-qwen",
        "sample_root": _rel(args.manifest.parent),
        "matte_root": _rel(matte_root),
        "vlm_root": _rel(vlm_root),
        "case_count": len(rows),
        "ok_count": ok,
        "expected_tool_hit_count": hits,
        "expected_any_tool_hit_count": any_hits,
        "harmful_tool_selected_count": harmful,
        "shadow_policy_hit_count": shadow_hits,
        "shadow_policy_required_count": len(shadow_required),
        "rows": rows,
    }
    vlm_root.mkdir(parents=True, exist_ok=True)
    (vlm_root / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = [
        "# Game Eval Qwen G/W",
        "",
        f"- run_id: `{out_root.name}`",
        f"- model: `comfy-qwen:{args.qwen_model}`",
        f"- ok: {ok}/{len(rows)}",
        f"- expected_tool_hit: {hits}/{len(rows)}",
        f"- expected_any_tool_hit: {any_hits}/{len(rows)}",
        f"- harmful_tool_selected: {harmful}/{len(rows)}",
        f"- shadow_policy_hit: {shadow_hits}/{len(shadow_required)}",
        "",
        "| sample | case | variant | regions | shadow | candidates | tools | any_hit | harmful | expected_hit | status |",
        "|---|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {case} | {variant} | {regions} | {shadow} | {candidates} | {tools} | {any_hit} | {harmful} | {hit} | {status} |".format(
                sample=row.get("sample_code", ""),
                case=row.get("case_id", ""),
                variant=row.get("sample_variant", ""),
                regions=row.get("region_count", 0),
                shadow=row.get("shadow_candidate_count", 0),
                candidates=row.get("candidate_count", 0),
                tools=", ".join(row.get("selected_tools", [])),
                any_hit=row.get("expected_any_hit", False),
                harmful=", ".join(row.get("harmful_tools", [])),
                hit=row.get("expected_hit", False),
                status=row.get("status", ""),
            )
        )
    (vlm_root / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "out" / "vlm_eval_game_qwen_gw_v001_20260526")
    p.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    p.add_argument("--qwen-model", default="Qwen3-VL-4B-Instruct-FP8")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--backend", default="auto")
    p.add_argument("--matting-model", default="ZhengPeng7/BiRefNet-matting")
    p.add_argument("--max-regions", type=int, default=24)
    p.add_argument("--max-region-crops", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample-id", default="", help="Comma-separated sample ids, e.g. G03")
    p.add_argument("--case-id", default="", help="Comma-separated case ids")
    p.add_argument("--variants", default="green,white", help="Comma-separated variants: green,white")
    p.add_argument("--resume-existing", action="store_true", help="Skip completed ok rows in an existing out-dir")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
