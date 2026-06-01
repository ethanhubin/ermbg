#!/usr/bin/env python3
"""Run ERMBG smart matting through the remote ComfyUI RouteMatte node."""
import argparse
import datetime as dt
import json
import mimetypes
import os
import pathlib
import re
import struct
import sys
import time
import uuid
import urllib.parse
import urllib.request
import zlib

import numpy as np

DEFAULT_COMFY_URL = "http://192.168.0.8:8000"
ARCHIVE = pathlib.Path(
    os.environ.get(
        "ERMBG_ARCHIVE",
        os.path.expanduser("~/.openclaw/media/openclaw-production/images/ermbg"),
    )
)


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "image"


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgba(path):
    data = pathlib.Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SystemExit(f"expected PNG image: {path}")

    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, flt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if compression != 0 or flt != 0 or interlace != 0:
                raise SystemExit("only non-interlaced PNG output is supported")
            if bit_depth != 8:
                raise SystemExit("only 8-bit PNG output is supported")
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break

    channels_by_type = {0: 1, 2: 3, 6: 4}
    if color_type not in channels_by_type:
        raise SystemExit(f"unsupported PNG color type {color_type}")
    channels = channels_by_type[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    rows = np.zeros((height, stride), dtype=np.uint8)
    i = 0
    for y in range(height):
        filter_type = raw[i]
        i += 1
        scan = np.frombuffer(raw[i : i + stride], dtype=np.uint8).copy()
        i += stride
        prior = rows[y - 1] if y else np.zeros(stride, dtype=np.uint8)
        recon = np.zeros(stride, dtype=np.uint8)
        for x in range(stride):
            left = int(recon[x - channels]) if x >= channels else 0
            up = int(prior[x])
            upper_left = int(prior[x - channels]) if x >= channels else 0
            value = int(scan[x])
            if filter_type == 0:
                recon[x] = value
            elif filter_type == 1:
                recon[x] = (value + left) & 255
            elif filter_type == 2:
                recon[x] = (value + up) & 255
            elif filter_type == 3:
                recon[x] = (value + ((left + up) // 2)) & 255
            elif filter_type == 4:
                recon[x] = (value + paeth(left, up, upper_left)) & 255
            else:
                raise SystemExit(f"unsupported PNG filter type {filter_type}")
        rows[y] = recon

    arr = rows.reshape(height, width, channels)
    if color_type == 6:
        return arr.copy()
    if color_type == 2:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        return np.concatenate([arr, alpha], axis=2)
    gray = arr[:, :, 0:1]
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.concatenate([gray, gray, gray, alpha], axis=2)


def write_png_rgba(path, rgba):
    h, w, c = rgba.shape
    if c != 4:
        raise ValueError("expected RGBA image")
    rows = [b"\x00" + rgba[y].astype(np.uint8).tobytes() for y in range(h)]
    raw = b"".join(rows)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )
    pathlib.Path(path).write_bytes(data)


def url_json(comfy_url, path, data=None, timeout=30):
    url = comfy_url + path
    if data is None:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def upload_image(comfy_url, image_path):
    boundary = "----openclaw-ermbg-" + uuid.uuid4().hex
    filename = safe_name(image_path.name)
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode(),
        image_path.read_bytes(),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        comfy_url + "/upload/image",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read().decode())
    return resp.get("name") or filename


def route_matte_prompt(uploaded_name, prefix, args):
    return {
        "10": {"class_type": "LoadImage", "inputs": {"image": uploaded_name}},
        "20": {
            "class_type": "ErmbgRouteMatte",
            "inputs": {
                "image": ["10", 0],
                "source_mask": ["10", 1],
                "shadow_mode": args.shadow_mode,
                "corridorkey_screen_mode": args.corridorkey_screen_mode,
                "corridorkey_preset": args.corridorkey_preset,
                "corridorkey_hard_ui_hint_mode": args.corridorkey_hard_ui_hint_mode,
                "fallback_bg_color": args.bg_color,
                "pymatting_method": args.pymatting_method,
                "pymatting_image_space": args.pymatting_image_space,
                "pymatting_bg_source": args.pymatting_bg_source,
                "pymatting_bg_color": args.bg_color,
                "pymatting_bg_threshold": args.pymatting_bg_threshold,
                "pymatting_fg_threshold": args.pymatting_fg_threshold,
                "pymatting_boundary_band_px": args.pymatting_boundary_band_px,
                "pymatting_auto_adapt": args.pymatting_auto_adapt,
                "pymatting_cg_maxiter": args.pymatting_cg_maxiter,
                "pymatting_cg_rtol": args.pymatting_cg_rtol,
            },
        },
        "30": {
            "class_type": "SaveImage",
            "inputs": {"images": ["20", 0], "filename_prefix": prefix + "_foreground"},
        },
        "40": {"class_type": "Convert Masks to Images", "inputs": {"masks": ["20", 1]}},
        "50": {
            "class_type": "SaveImage",
            "inputs": {"images": ["40", 0], "filename_prefix": prefix + "_alpha"},
        },
        "60": {
            "class_type": "SaveImage",
            "inputs": {"images": ["20", 3], "filename_prefix": prefix + "_rgba_rgb"},
        },
        "70": {
            "class_type": "SaveImage",
            "inputs": {"images": ["20", 4], "filename_prefix": prefix + "_aux"},
        },
    }


def output_dir_for(image_path, requested):
    if requested:
        out = pathlib.Path(requested).expanduser().resolve()
    else:
        now = dt.datetime.now()
        out = ARCHIVE / now.strftime("%Y-%m-%d") / (
            now.strftime("%H%M%S") + "_" + safe_name(image_path.stem)
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_view(comfy_url, item, destination):
    params = {
        "filename": item.get("filename", ""),
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    }
    url = comfy_url + "/view?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        destination.write_bytes(r.read())


def wait_for_output(comfy_url, prompt_id, timeout_seconds):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        hist = url_json(comfy_url, f"/history/{prompt_id}", timeout=20)
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            if status.get("status_str") == "error":
                for kind, payload in status.get("messages", []):
                    if kind == "execution_error":
                        node = payload.get("node_type") or payload.get("node_id")
                        exc = payload.get("exception_type", "ExecutionError")
                        msg = payload.get("exception_message", "")
                        raise SystemExit(f"ComfyUI execution error in {node}: {exc}: {msg}")
                raise SystemExit(f"ComfyUI execution error: {status}")
            outputs = hist[prompt_id].get("outputs", {})
            if outputs:
                return outputs
            raise SystemExit("ComfyUI finished but returned no outputs")
        time.sleep(1)
    raise SystemExit(f"timeout waiting for ComfyUI prompt: {prompt_id}")


def image_item_for_node(outputs, node_id):
    images = outputs.get(str(node_id), {}).get("images", [])
    if not images:
        raise SystemExit(f"ComfyUI finished but node {node_id} returned no image")
    return images[0]


def route_summary_for_node(outputs, node_id):
    out = outputs.get(str(node_id), {})
    texts = out.get("text") or out.get("string") or []
    if not texts:
        return {}
    try:
        return json.loads(str(texts[0]))
    except json.JSONDecodeError:
        return {"raw": str(texts[0])}


def compose_route_rgba(rgba_rgb_path, alpha_path, output_path):
    rgba_rgb = read_png_rgba(rgba_rgb_path)
    alpha = read_png_rgba(alpha_path)[:, :, 0]
    h, w = rgba_rgb.shape[:2]
    if alpha.shape != (h, w):
        raise SystemExit(
            f"RouteMatte alpha size {alpha.shape[::-1]} does not match RGB size {(w, h)}"
        )
    rgba = np.dstack([rgba_rgb[:, :, :3], alpha]).astype(np.uint8)
    write_png_rgba(output_path, rgba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--output-dir", help="Directory for output artifacts")
    ap.add_argument("--comfy-url", default=os.environ.get("COMFY_URL", DEFAULT_COMFY_URL))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--shadow-mode", choices=["on", "off", "auto"], default="on")
    ap.add_argument("--bg-color", default="0,200,0")
    ap.add_argument("--corridorkey-screen-mode", choices=["auto", "green", "blue"], default="auto")
    ap.add_argument("--corridorkey-preset", choices=["auto", "detail_safe", "spill_safe", "manual"], default="auto")
    ap.add_argument(
        "--corridorkey-hard-ui-hint-mode",
        choices=[
            "all_white",
            "bbox_2px",
            "boundary_2px",
            "boundary_2px_shadow_safe",
            "boundary_2px_shadow_safe_edge_floor",
            "translucent_button",
        ],
        default="bbox_2px",
    )
    ap.add_argument("--pymatting-method", choices=["cf", "knn", "lbdm", "lkm", "rw", "sm"], default="cf")
    ap.add_argument("--pymatting-image-space", choices=["linear", "sRGB"], default="linear")
    ap.add_argument("--pymatting-bg-source", choices=["auto", "green", "blue", "custom"], default="auto")
    ap.add_argument("--pymatting-bg-threshold", type=float, default=3.5)
    ap.add_argument("--pymatting-fg-threshold", type=float, default=30.0)
    ap.add_argument("--pymatting-boundary-band-px", type=int, default=2)
    ap.add_argument("--pymatting-auto-adapt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pymatting-cg-maxiter", type=int, default=1000)
    ap.add_argument("--pymatting-cg-rtol", type=float, default=0.000001)
    args = ap.parse_args()

    image_path = pathlib.Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")
    comfy_url = args.comfy_url.rstrip("/")
    info = url_json(comfy_url, "/object_info", timeout=20)
    for node in ("ErmbgRouteMatte", "Convert Masks to Images"):
        if node not in info:
            raise SystemExit(f"{node} node not found on ComfyUI server")

    out_dir = output_dir_for(image_path, args.output_dir)
    output_path = out_dir / "output.png"
    foreground_path = out_dir / "foreground.png"
    alpha_path = out_dir / "alpha.png"
    rgba_rgb_path = out_dir / "rgba_rgb.png"
    aux_path = out_dir / "aux.png"
    metadata_path = out_dir / "metadata.json"
    workflow_path = out_dir / "workflow.json"
    history_path = out_dir / "history_outputs.json"
    manifest_path = out_dir / "manifest.json"

    prefix = f"ermbg/{uuid.uuid4().hex}_{safe_name(image_path.stem)}"
    uploaded = upload_image(comfy_url, image_path)
    prompt = route_matte_prompt(uploaded, prefix, args)
    workflow_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf-8")

    resp = url_json(
        comfy_url,
        "/prompt",
        {"prompt": prompt, "client_id": "openclaw-ermbg-" + uuid.uuid4().hex},
        timeout=30,
    )
    prompt_id = resp["prompt_id"]
    print("PROMPT_ID", prompt_id, flush=True)
    print("NODE", "ErmbgRouteMatte", flush=True)

    outputs = wait_for_output(comfy_url, prompt_id, args.timeout)
    download_view(comfy_url, image_item_for_node(outputs, "30"), foreground_path)
    download_view(comfy_url, image_item_for_node(outputs, "50"), alpha_path)
    download_view(comfy_url, image_item_for_node(outputs, "60"), rgba_rgb_path)
    download_view(comfy_url, image_item_for_node(outputs, "70"), aux_path)
    compose_route_rgba(rgba_rgb_path, alpha_path, output_path)

    route_summary = route_summary_for_node(outputs, "20")
    metadata_path.write_text(json.dumps(route_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "tool": "ermbg-matte",
        "input": str(image_path),
        "output": str(output_path),
        "foreground": str(foreground_path),
        "alpha": str(alpha_path),
        "rgba_rgb": str(rgba_rgb_path),
        "aux": str(aux_path),
        "metadata": str(metadata_path),
        "workflow": str(workflow_path),
        "history_outputs": str(history_path),
        "comfy_url": comfy_url,
        "node": "ErmbgRouteMatte",
        "prompt_id": prompt_id,
        "route": route_summary.get("route") or route_summary.get("strategy_name"),
        "selected_backend": route_summary.get("selected_backend"),
        "asset_kind": route_summary.get("asset_kind"),
        "parameter_profile": route_summary.get("parameter_profile"),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUTPUT", output_path.resolve(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("interrupted")
