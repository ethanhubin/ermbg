"""Batch Qwen subject recognition -> CLIPSeg subject mask generation.

Runs over the current VLM eval sample manifests, asks remote Comfy Qwen3_VQA
for a concise foreground-subject prompt, then feeds that prompt into the
existing CLIPSeg -> ERMBG Comfy workflow to produce subject masks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg import io
from ermbg.comfy import DEFAULT_COMFY_URL
from ermbg.probe.comfyui_subject_mask import ComfyUISubjectMaskWorkflow
from ermbg.vlm_semantic import (
    ComfyQwenVLMSemanticPriorClient,
    _extract_comfy_preview_text,
    parse_qwen_json_text,
)


DEFAULT_SAMPLE_ROOTS = (Path("samples/vlm_eval"), Path("samples/vlm_eval_game"))


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_manifest_samples(sample_roots: list[Path]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for root in sample_roots:
        manifest = root / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        for case in data.get("cases", []):
            case_id = str(case["id"])
            if "green" in case:
                image_path = Path(case["green"])
                background = case.get("backgrounds", {}).get("green", [0, 200, 0])
            else:
                image_path = Path(case.get("green_input") or case["input"])
                background = case.get("green_background") or case.get("background", [0, 200, 0])
            samples.append(
                {
                    "group": root.name,
                    "id": case_id,
                    "sample_id": case.get("sample_id"),
                    "image": str(image_path),
                    "background": [int(c) for c in background],
                    "human_label": case.get("human_label"),
                    "primary_ambiguity": case.get("primary_ambiguity"),
                }
            )
    return samples


def _qwen_subject_prompt(user_hint: str | None = None) -> str:
    hint = f"\nKnown case hint: {user_hint}" if user_hint else ""
    return (
        "Look at this green-screen evaluation image and identify the complete "
        "foreground subject to keep for segmentation. Exclude the flat green "
        "background. Exclude cast shadows unless they are physically part of the "
        "object design; include outlines, fur, transparent material, holes, "
        "attached ribbons, small pieces, and all foreground objects that belong "
        "to the asset.\n"
        "The subject_prompt must describe only the foreground object(s). Do not "
        "write phrases such as 'on green background', 'against green background', "
        "'green screen', 'background', or 'backdrop'. Mention green only if green "
        "is visibly a material/color of the foreground subject itself.\n"
        "Return raw JSON only with exactly these keys:\n"
        '{"subject_prompt":"a concise CLIPSeg segmentation prompt, 6 to 24 words",'
        '"notes":"one short note about included/excluded parts"}'
        f"{hint}"
    )


def _extract_subject_prompt(raw_text: str) -> dict[str, str]:
    payload = parse_qwen_json_text(raw_text)
    prompt = str(payload.get("subject_prompt", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    if not prompt:
        raise RuntimeError(f"Qwen response missing subject_prompt: {raw_text[:400]}")
    prompt = re.sub(
        r"\s*(?:,?\s*(?:centered\s+)?(?:on|against|with)\s+(?:a\s+)?green\s+background|"
        r",?\s*(?:on|against|with)\s+(?:a\s+)?green\s+screen|"
        r",?\s*(?:on|against|with)\s+(?:a\s+)?flat\s+green\s+backdrop)\s*",
        " ",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\s{2,}", " ", prompt).strip(" ,.;")
    return {"subject_prompt": prompt, "notes": notes}


def _run_qwen_subject_recognition(
    image_srgb: np.ndarray,
    *,
    case_id: str,
    user_hint: str | None,
    comfy_url: str,
    qwen_model: str,
    timeout: float,
) -> dict[str, Any]:
    client = ComfyQwenVLMSemanticPriorClient(
        url=comfy_url,
        model=qwen_model,
        timeout=timeout,
        max_new_tokens=500,
        temperature=0.05,
    )
    server_name = client._upload_image(image_srgb, f"ermbg_qwen_subject_{_safe_id(case_id)}_{uuid.uuid4().hex[:8]}.png")
    workflow = client._build_workflow(server_name, _qwen_subject_prompt(user_hint))
    started = time.monotonic()
    prompt_id = client._queue(workflow)
    history = client._wait(prompt_id)
    elapsed = time.monotonic() - started
    raw_text = _extract_comfy_preview_text(history)
    parsed = _extract_subject_prompt(raw_text)
    return {
        "prompt_id": prompt_id,
        "elapsed_sec": elapsed,
        "server_image": server_name,
        "workflow": workflow,
        "raw_text": raw_text,
        **parsed,
    }


def _download_subject_mask(
    image_srgb: np.ndarray,
    *,
    upload_name: str,
    subject_prompt: str,
    background: list[int],
    out_dir: Path,
    filename_prefix: str,
    comfy_url: str,
    timeout: float,
) -> dict[str, Any]:
    runner = ComfyUISubjectMaskWorkflow(url=comfy_url, timeout=timeout)
    result = runner.run(
        image_srgb,
        subject_prompt=subject_prompt,
        filename_prefix=filename_prefix,
        bg_color=",".join(str(int(c)) for c in background),
        upload_name=upload_name,
        download_dir=out_dir,
    )
    return {
        "server_image": result["server_image"],
        "prompt_id": result["prompt_id"],
        "workflow": result["workflow"],
        "downloads": result["downloads"],
    }


def _mask_stats(mask_path: Path) -> dict[str, Any]:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    fg = mask >= 128
    if fg.any():
        ys, xs = np.where(fg)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    else:
        bbox = None
    return {
        "mask_pixels_ge_128": int(fg.sum()),
        "mask_area_ratio_ge_128": float(fg.mean()),
        "mask_mean": float(mask.mean() / 255.0),
        "mask_bbox_xyxy": bbox,
    }


def _find_download(downloads: list[dict[str, Any]], role: str) -> Path | None:
    for item in downloads:
        if item.get("role") == role:
            return Path(str(item["local_path"]))
    return None


def _overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB").resize(mask.size, Image.Resampling.LANCZOS)
    mask_l = mask.convert("L")
    arr = np.asarray(rgb, dtype=np.float32)
    m = np.asarray(mask_l, dtype=np.float32) / 255.0
    color = np.array([255, 50, 210], dtype=np.float32)
    arr = arr * (1.0 - 0.45 * m[..., None]) + color * (0.45 * m[..., None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _make_contact_sheet(rows: list[dict[str, Any]], out_path: Path, *, kind: str) -> None:
    tile_w, tile_h = 260, 260
    label_h = 42
    cols = 5
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return
    sheet = Image.new("RGB", (cols * tile_w, ((len(valid) + cols - 1) // cols) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(valid):
        case_dir = Path(row["out_dir"])
        image = Image.open(row["image"]).convert("RGB")
        mask_path = case_dir / "subject_mask.png"
        if kind == "mask":
            tile = Image.open(mask_path).convert("RGB")
        elif kind == "overlay":
            tile = _overlay_mask(image, Image.open(mask_path))
        else:
            tile = image
        tile.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * tile_w
        y = (idx // cols) * (tile_h + label_h)
        label = f"{row['group']}/{row['id']}"
        draw.text((x + 6, y + 6), label[:40], fill=(0, 0, 0))
        ratio = row.get("mask_area_ratio_ge_128", 0.0)
        draw.text((x + 6, y + 23), f"area>128 {ratio:.3f}", fill=(0, 0, 0))
        sheet.paste(tile, (x + (tile_w - tile.width) // 2, y + label_h + (tile_h - tile.height) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def run(args: argparse.Namespace) -> None:
    samples = _load_manifest_samples(args.sample_root)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No samples found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        case_key = f"{sample['group']}_{sample['id']}"
        case_dir = args.out_dir / _safe_id(case_key)
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(sample["image"])
        print(f"[{idx}/{len(samples)}] {case_key} -> {case_dir}", flush=True)
        row: dict[str, Any] = {**sample, "out_dir": str(case_dir), "status": "started"}
        try:
            image_srgb = io.load_rgb(image_path, background=tuple(sample["background"]))
            io.save_rgb(case_dir / "input.png", image_srgb)
            qwen = _run_qwen_subject_recognition(
                image_srgb,
                case_id=case_key,
                user_hint=sample.get("human_label"),
                comfy_url=args.comfy_url,
                qwen_model=args.qwen_model,
                timeout=args.timeout,
            )
            (case_dir / "qwen_workflow.json").write_text(json.dumps(qwen["workflow"], indent=2, ensure_ascii=False))
            (case_dir / "qwen_response.json").write_text(
                json.dumps({k: v for k, v in qwen.items() if k != "workflow"}, indent=2, ensure_ascii=False)
            )
            print(f"  Qwen subject_prompt: {qwen['subject_prompt']}", flush=True)
            comfy = _download_subject_mask(
                image_srgb,
                upload_name=f"ermbg_mask_{_safe_id(case_key)}_{uuid.uuid4().hex[:8]}.png",
                subject_prompt=qwen["subject_prompt"],
                background=sample["background"],
                out_dir=case_dir,
                filename_prefix=f"qwen_subject_{_safe_id(case_key)}",
                comfy_url=args.comfy_url,
                timeout=args.timeout,
            )
            (case_dir / "clipseg_ermbg_workflow.json").write_text(
                json.dumps(comfy["workflow"], indent=2, ensure_ascii=False)
            )
            (case_dir / "comfy_downloads.json").write_text(json.dumps(comfy["downloads"], indent=2, ensure_ascii=False))
            mask_path = _find_download(comfy["downloads"], "subject_mask")
            if mask_path is None:
                raise RuntimeError("subject_mask output was not downloaded")
            stats = _mask_stats(mask_path)
            row.update(
                {
                    "status": "ok",
                    "qwen_prompt_id": qwen["prompt_id"],
                    "mask_prompt_id": comfy["prompt_id"],
                    "subject_prompt": qwen["subject_prompt"],
                    "qwen_notes": qwen["notes"],
                    "subject_mask": str(mask_path),
                    **stats,
                }
            )
            print(f"  mask area ratio: {stats['mask_area_ratio_ge_128']:.3f}", flush=True)
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)
        rows.append(row)
        (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    _make_contact_sheet(rows, args.out_dir / "mask_contact_sheet.png", kind="mask")
    _make_contact_sheet(rows, args.out_dir / "overlay_contact_sheet.png", kind="overlay")
    ok = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Done: {ok}/{len(rows)} succeeded. Summary: {args.out_dir / 'summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-root", type=Path, action="append", default=list(DEFAULT_SAMPLE_ROOTS))
    p.add_argument("--out-dir", type=Path, default=Path("out/qwen_subject_mask_all_20260526"))
    p.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    p.add_argument("--qwen-model", default="Qwen3-VL-4B-Instruct-FP8")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
