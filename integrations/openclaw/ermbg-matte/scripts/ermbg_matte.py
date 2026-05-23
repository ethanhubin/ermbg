#!/usr/bin/env python3
"""ermbg-matte: route an image through ERMBG AutoMatte on the LAN ComfyUI.

Calls ERMBG AutoMatte (custom node) over ComfyUI's HTTP API. Outputs a clean
RGBA PNG plus a manifest with the strategy that the router picked. Modeled
after ../comfyui-rmbg/scripts/comfyui_rmbg.py for consistency.

Typical invocation::

    python3 skills/ermbg-matte/scripts/ermbg_matte.py --image /path/to/in.png

Outputs land in:
  ~/.openclaw/media/openclaw-production/images/ermbg/<uuid>/
    output.png       — clean RGBA
    workflow.json    — the API workflow that produced it
    manifest.json    — input/output paths, strategy summary, prompt_id, B
    history.json     — full ComfyUI history entry for the prompt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import os
import pathlib
import re
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request


COMFY = os.environ.get("COMFY_URL", "http://192.168.0.8:8000").rstrip("/")
ARCHIVE = pathlib.Path(
    os.environ.get(
        "ERMBG_ARCHIVE",
        os.path.expanduser("~/.openclaw/media/openclaw-production/images/ermbg"),
    )
)
SKILL_DIR = pathlib.Path(__file__).resolve().parent
WORKFLOW_TEMPLATE = SKILL_DIR / "workflow.template.json"


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "image"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def url_json(path: str, data: dict | None = None, timeout: int = 30) -> dict:
    url = COMFY + path
    if data is None:
        req = urllib.request.Request(url)
    else:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_image(image_path: pathlib.Path) -> str:
    """Multipart upload to ComfyUI /upload/image. Returns the stored filename."""
    boundary = f"----ermbg{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {mime}\r\n\r\n".encode())
    body.extend(image_path.read_bytes())
    body.extend(f"\r\n--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="overwrite"\r\n\r\n')
    body.extend(b"true")
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        COMFY + "/upload/image",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    return info["name"]


def download_view(item: dict, dest: pathlib.Path) -> None:
    qs = urllib.parse.urlencode(
        {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
    )
    with urllib.request.urlopen(COMFY + "/view?" + qs, timeout=120) as resp:
        dest.write_bytes(resp.read())


def wait_for_output(prompt_id: str, timeout_seconds: int) -> tuple[dict, dict]:
    """Poll /history/<id> until done. Returns (first_image, all_outputs)."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            hist = url_json(f"/history/{prompt_id}", timeout=20)
        except urllib.error.URLError:
            time.sleep(2)
            continue
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise SystemExit(
                    f"ComfyUI prompt {prompt_id} failed: {json.dumps(msgs, ensure_ascii=False)}"
                )
            outputs = hist[prompt_id].get("outputs", {})
            if outputs:
                # Find the SaveImageWithAlpha (or any node that produced "images")
                for node_id, payload in outputs.items():
                    if "images" in payload and payload["images"]:
                        return payload["images"][0], outputs
        time.sleep(2)
    raise SystemExit(f"timeout waiting for ComfyUI prompt: {prompt_id}")


# ---------------------------------------------------------------------------
# Workflow assembly
# ---------------------------------------------------------------------------


def build_workflow(
    uploaded_name: str,
    prefix: str,
    despill: str,
    use_keyer: str,
    bg_color: str,
    matting_model: str,
) -> dict:
    template = json.loads(WORKFLOW_TEMPLATE.read_text(encoding="utf-8"))
    # template uses ${input_image} / ${prefix} placeholders; substitute here so
    # the saved workflow.json reflects the exact run.
    s = json.dumps(template, ensure_ascii=False)
    s = s.replace("${input_image}", uploaded_name).replace("${prefix}", prefix)
    wf = json.loads(s)
    wf["2"]["inputs"]["despill"] = despill
    wf["2"]["inputs"]["use_keyer"] = use_keyer
    wf["2"]["inputs"]["bg_color"] = bg_color
    wf["2"]["inputs"]["matting_model"] = matting_model
    return wf


def output_dir_for(image_path: pathlib.Path) -> pathlib.Path:
    out = ARCHIVE / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}-{safe_name(image_path.stem)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Smart matting via ERMBG AutoMatte custom node on LAN ComfyUI."
    )
    ap.add_argument("--image", required=True, help="Input image path (RGB or RGBA)")
    ap.add_argument("--output-dir", help="Override default archive dir")
    ap.add_argument(
        "--despill",
        default="auto",
        choices=["auto", "unmix", "chroma_cap", "local_borrow", "closed_form", "none"],
    )
    ap.add_argument(
        "--use-keyer",
        default="auto",
        choices=["auto", "on", "off"],
        help="auto = router decides; on/off forces keyer state",
    )
    ap.add_argument("--bg-color", default="0,200,0", help="R,G,B for re-compositing dirty RGBA")
    ap.add_argument("--matting-model", default="ZhengPeng7/BiRefNet-matting")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    image_path = pathlib.Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    out_dir = pathlib.Path(args.output_dir).expanduser().resolve() if args.output_dir else output_dir_for(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "output.png"

    despill_arg = "auto (router decides)" if args.despill == "auto" else args.despill
    use_keyer_arg = {"auto": "auto (router decides)", "on": "force_on", "off": "force_off"}[args.use_keyer]

    # 1. Upload
    uploaded = upload_image(image_path)

    # 2. Build + save workflow
    prefix = f"ermbg/{uuid.uuid4().hex}_{safe_name(image_path.stem)}"
    wf = build_workflow(uploaded, prefix, despill_arg, use_keyer_arg, args.bg_color, args.matting_model)
    (out_dir / "workflow.json").write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. Submit prompt
    resp = url_json(
        "/prompt",
        {"prompt": wf, "client_id": "openclaw-ermbg-" + uuid.uuid4().hex},
        timeout=30,
    )
    prompt_id = resp["prompt_id"]
    print("PROMPT_ID", prompt_id, flush=True)

    # 4. Wait + download
    item, outputs = wait_for_output(prompt_id, args.timeout)
    download_view(item, output_path)

    # 5. Summary from custom-node STRING output (if surfaced in outputs payload)
    summary = ""
    for payload in outputs.values():
        for k in ("text", "string", "summary"):
            v = payload.get(k)
            if isinstance(v, list) and v:
                summary = str(v[0])
                break
        if summary:
            break

    manifest = {
        "input": str(image_path),
        "output": str(output_path),
        "comfy_url": COMFY,
        "prompt_id": prompt_id,
        "summary": summary,
        "options": {
            "despill": args.despill,
            "use_keyer": args.use_keyer,
            "bg_color": args.bg_color,
            "matting_model": args.matting_model,
        },
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("OUTPUT", output_path.resolve(), flush=True)
    if summary:
        print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("interrupted")
