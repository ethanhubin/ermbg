"""Small web UI for ERMBG.

The service keeps the browser flow intentionally narrow: upload one image,
run ``matte_image``, preview the returned RGBA PNG, and download it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from urllib.parse import quote

import numpy as np
import cv2
from PIL import Image, ImageDraw

try:
    from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, Response
except ImportError as e:  # pragma: no cover - exercised only without web extra
    raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

from .api import matte_image
from .candidates import MatteCandidate, generate_matte_candidates
from .local_ownership import generate_local_ownership_candidate
from .slicer import SliceBox, SliceResult, classify_ui_slice, crop_slice, slice_image

ALLOWED_BACKENDS = {
    "auto",
    "comfy-rmbg",
    "comfy-corridorkey",
    "pymatting-known-b",
    "comfy-pymatting-known-b",
}
REMOTE_DIRECT_BACKENDS = {
    "passthrough",
    "comfy-rmbg",
    "comfy-corridorkey",
    "pymatting-known-b",
    "comfy-pymatting-known-b",
}
WEB_SHADOW_MODE = "on"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME_EVAL_ROOT = PROJECT_ROOT / "out" / "local_ownership_full_20260527"
GAME_SAMPLE_REL = Path("samples") / "corridorkey_semantic"
LOCAL_OWNERSHIP_EVAL_PREFIX = "local_ownership_"
SOLID_GRAPHIC_EVAL_PREFIX = "solid_graphic_"
AUTO_EVAL_PREFIX = "auto_"
CORRIDORKEY_EVAL_PREFIX = "corridorkey_"
RMBG_EVAL_PREFIX = "rmbg_"
GAME_EVAL_RUN_PREFIXES = (
    LOCAL_OWNERSHIP_EVAL_PREFIX,
    SOLID_GRAPHIC_EVAL_PREFIX,
    AUTO_EVAL_PREFIX,
    CORRIDORKEY_EVAL_PREFIX,
    RMBG_EVAL_PREFIX,
)
FAST_GAME_EVAL_SAMPLE_IDS = ("B001", "B016", "B031", "B046", "I011", "I019", "C004", "C009")
GAME_EVAL_SCREENS = ("green", "blue")
# Fallback only applies in tests or broken installs where the manifest is not
# available. It mirrors the current B/I/C semantic manifest so progress does not
# silently drift back to the retired 78-sample set.
FALLBACK_GAME_EVAL_EXPECTED_TOTAL = 83
DEFAULT_GAME_EVAL_TEST_PATH = "auto"
GAME_EVAL_TEST_PATHS = {
    "auto": {
        "label": "Auto RouteMatte",
        "backend": "auto",
        "prefix": AUTO_EVAL_PREFIX,
    },
    "corridorkey": {
        "label": "CorridorKey",
        "backend": "comfy-corridorkey",
        "prefix": CORRIDORKEY_EVAL_PREFIX,
    },
    "rmbg": {
        "label": "RMBG",
        "backend": "comfy-rmbg",
        "prefix": RMBG_EVAL_PREFIX,
    },
}
SERVABLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REGION_BOX_COLORS = {
    "same_bg_enclosed_region": (0, 153, 255, 235),
    "alpha_keyer_disagreement": (179, 92, 255, 235),
    "hard_edge_candidate": (255, 160, 0, 235),
}
REGION_FILL_COLORS = {
    "same_bg_enclosed_region": (0, 153, 255, 28),
    "alpha_keyer_disagreement": (179, 92, 255, 24),
    "hard_edge_candidate": (255, 160, 0, 24),
}

app = FastAPI(title="ERMBG Web", version="0.1.0")
_GAME_EVAL_JOBS: dict[str, dict[str, object]] = {}
_GAME_EVAL_JOBS_LOCK = Lock()
_SLICE_CACHE_MAX = 4
_SLICE_WEB_MAX_PIXELS = 4_000_000
_SLICE_CACHE: OrderedDict[tuple[str, int, int], SliceResult] = OrderedDict()
_SLICE_CACHE_LOCK = Lock()


def _game_sample_root() -> Path:
    return PROJECT_ROOT / GAME_SAMPLE_REL


def _game_sample_manifest() -> Path:
    return _game_sample_root() / "manifest.json"


def _encode_png(rgba: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _image_from_upload_bytes(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from e

    return image.convert("RGBA" if image.mode == "RGBA" else "RGB")


def _load_upload_image(upload: UploadFile) -> Image.Image:
    return _image_from_upload_bytes(upload.file.read())


def _load_upload_image_with_digest(upload: UploadFile) -> tuple[Image.Image, str]:
    data = upload.file.read()
    return _image_from_upload_bytes(data), hashlib.sha256(data).hexdigest()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _matte_page_html()


def _matte_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1c2320; background: #f5f7f4; }
    * { box-sizing: border-box; }
    body { margin: 0; height: 100vh; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }
    .header-actions { min-width: 0; display: flex; align-items: center; gap: 14px; }
    .nav-link { color: #196f5a; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    main { width: min(1120px, 100%); height: calc(100vh - 56px); min-height: 0; margin: 0 auto; padding: 16px 24px; display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 24px; align-items: stretch; overflow: hidden; }
    form, .preview { background: #ffffff; border: 1px solid #d9dfd7; border-radius: 8px; }
    form { min-width: 0; min-height: 0; max-height: 100%; padding: 16px; display: grid; gap: 12px; align-content: start; overflow-y: auto; }
    label { display: grid; gap: 8px; font-size: 13px; font-weight: 600; color: #47524c; }
    .inline-label { display: grid; grid-template-columns: 76px minmax(0, 1fr); align-items: center; gap: 10px; }
    input, select, button { width: 100%; min-height: 40px; border-radius: 6px; border: 1px solid #b8c1b7; background: #ffffff; color: #1c2320; font: inherit; }
    input[type="file"] { padding: 8px; }
    button, a.download { display: inline-flex; align-items: center; justify-content: center; min-height: 42px; border: 0; border-radius: 6px; background: #196f5a; color: #ffffff; text-decoration: none; font-weight: 700; cursor: pointer; }
    button:disabled, a.download[aria-disabled="true"] { opacity: 0.55; cursor: not-allowed; pointer-events: none; }
    .settings { display: none; border: 1px solid #d9dfd7; border-radius: 6px; background: #fbfcfa; }
    .settings.is-visible { display: block; }
    .settings summary { min-height: 38px; display: flex; align-items: center; padding: 0 10px; color: #196f5a; font-size: 13px; font-weight: 800; cursor: pointer; user-select: none; }
    .settings-grid { display: grid; gap: 12px; padding: 0 10px 10px; }
    .settings-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .settings label { font-size: 12px; gap: 6px; }
    .check-label { display: flex; align-items: center; justify-content: space-between; gap: 10px; min-height: 38px; }
    .check-label input { width: 18px; min-height: 18px; }
    .color-range { display: grid; gap: 8px; padding: 8px 0 2px; }
    .range-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 12px; font-weight: 700; color: #47524c; }
    .range-value { min-width: 58px; text-align: right; color: #196f5a; }
    .dual-range { --range-low: 16%; --range-high: 33%; position: relative; height: 34px; }
    .range-rail, .range-fill { position: absolute; left: 0; right: 0; top: 15px; height: 6px; border-radius: 999px; pointer-events: none; }
    .range-rail { background: linear-gradient(90deg, #00c853 0%, #c8d28c 45%, #f2bc24 72%, #d84646 100%); opacity: 0.78; }
    .range-fill { left: var(--range-low); right: calc(100% - var(--range-high)); background: rgba(25, 111, 90, 0.58); box-shadow: 0 0 0 1px rgba(25, 111, 90, 0.18); }
    .dual-range input[type="range"] { position: absolute; inset: 0; width: 100%; min-height: 34px; margin: 0; padding: 0; appearance: none; -webkit-appearance: none; background: transparent; border: 0; pointer-events: none; }
    .dual-range input[type="range"]::-webkit-slider-runnable-track { height: 6px; background: transparent; border: 0; }
    .dual-range input[type="range"]::-webkit-slider-thumb { appearance: none; -webkit-appearance: none; width: 18px; height: 18px; margin-top: -6px; border: 2px solid #ffffff; border-radius: 50%; background: #196f5a; box-shadow: 0 1px 4px rgba(12, 17, 15, 0.28); pointer-events: auto; cursor: ew-resize; }
    .dual-range input[type="range"]::-moz-range-track { height: 6px; background: transparent; border: 0; }
    .dual-range input[type="range"]::-moz-range-thumb { width: 18px; height: 18px; border: 2px solid #ffffff; border-radius: 50%; background: #196f5a; box-shadow: 0 1px 4px rgba(12, 17, 15, 0.28); pointer-events: auto; cursor: ew-resize; }
    .range-labels { display: flex; justify-content: space-between; color: #6a746f; font-size: 11px; font-weight: 700; }
    .source-preview { display: none; gap: 8px; }
    .source-preview.is-visible { display: grid; }
    .mask-stage { width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100%; max-height: 100%; grid-template-rows: minmax(0, 1fr); overflow: hidden; }
    .canvas:not(.is-mask-mode) .mask-stage { display: none; }
    .canvas.is-mask-mode .mask-stage { display: grid; }
    .preview-stage { width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100%; max-height: 100%; display: grid; place-items: center; overflow: hidden; }
    .canvas.is-mask-mode .preview-stage { display: none; }
    .source-frame { position: relative; width: 100%; aspect-ratio: 4 / 3; max-height: 360px; min-height: 148px; display: grid; place-items: center; border: 1px solid #d9dfd7; border-radius: 6px; background-color: #eef2ec; background-image: linear-gradient(45deg, #d7dfd4 25%, transparent 25%), linear-gradient(-45deg, #d7dfd4 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d7dfd4 75%), linear-gradient(-45deg, transparent 75%, #d7dfd4 75%); background-position: 0 0, 0 10px, 10px -10px, -10px 0; background-size: 20px 20px; overflow: hidden; }
    .source-frame img { position: absolute; z-index: 1; left: 50%; top: 50%; display: block; width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: contain; object-position: center; transform: translate(-50%, -50%) scale(1); transform-origin: center center; will-change: transform; }
    .mask-stage .source-frame { height: 100%; min-height: 0; max-height: none; aspect-ratio: auto; }
    .mask-overlay { position: absolute; z-index: 2; display: none; touch-action: none; cursor: crosshair; opacity: 0.62; image-rendering: pixelated; transform-origin: center center; will-change: transform; }
    .source-frame.has-mask .mask-overlay { display: block; }
    .preview-statuses { min-width: 0; flex: 1 1 auto; overflow: hidden; }
    .preview-statuses .status { display: block; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .mask-toolbar { display: none; align-items: center; justify-content: flex-start; gap: 8px; flex-wrap: nowrap; min-height: 42px; padding: 5px 16px; border-bottom: 1px solid #d9dfd7; background: #fbfcfa; overflow: hidden; }
    .preview.is-mask-mode .mask-toolbar { display: flex; }
    .mask-toolbar label { display: flex; align-items: center; font-size: 12px; gap: 6px; white-space: nowrap; }
    .mask-toolbar button { width: auto; min-height: 32px; padding: 0 10px; white-space: nowrap; }
    .mask-tools { min-width: 0; display: flex; align-items: center; justify-content: flex-start; gap: 8px; flex-wrap: nowrap; }
    .mask-tools > label { width: auto; flex: 0 0 auto; }
    .mask-mode-toggle { display: inline-flex; align-items: center; gap: 4px; padding: 3px; border: 1px solid #cfd7cc; border-radius: 6px; background: #f7f9f6; }
    .mask-mode-button { min-height: 28px; border: 0; border-radius: 4px; background: transparent; color: #47524c; font-size: 12px; font-weight: 800; }
    .mask-mode-button[aria-pressed="true"] { background: #196f5a; color: #ffffff; }
    #mask-brush-size { width: 104px; min-height: 32px; }
    .mask-actions { display: flex; gap: 8px; flex: 0 0 auto; }
    .preview { min-height: 0; height: 100%; display: grid; grid-template-rows: 48px auto minmax(0, 1fr) 104px 56px; overflow: hidden; }
    .preview-bar, .preview-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 16px; border-bottom: 1px solid #d9dfd7; }
    .preview-actions { height: 56px; min-height: 56px; max-height: 56px; border-top: 1px solid #d9dfd7; border-bottom: 0; overflow: hidden; }
    .preview-actions a.download { flex: 0 0 auto; min-width: 128px; padding: 0 18px; white-space: nowrap; }
    .tabs { display: inline-flex; align-items: center; gap: 4px; padding: 3px; border: 1px solid #cfd7cc; border-radius: 6px; background: #f7f9f6; flex-shrink: 0; }
    .tab { width: auto; min-height: 30px; padding: 0 10px; border: 0; border-radius: 4px; background: transparent; color: #47524c; font-size: 12px; font-weight: 700; }
    .tab[aria-selected="true"] { background: #196f5a; color: #ffffff; }
    .status { font-size: 13px; color: #5d6862; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .canvas, .candidate-thumb { background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); }
    .canvas { width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100%; max-height: 100%; align-self: stretch; justify-self: stretch; display: grid; place-items: stretch; padding: 16px; overflow: hidden; contain: layout paint; touch-action: none; background-position: 0 0, 0 12px, 12px -12px, -12px 0; background-size: 24px 24px; }
    .canvas.is-mask-mode { padding: 0; }
    .canvas.is-mask-mode .source-frame { border: 0; border-radius: 0; }
    .canvas.has-image { cursor: grab; }
    .canvas.is-dragging { cursor: grabbing; }
    .canvas.bg-white { background: #ffffff; }
    .canvas.bg-black { background: #111514; }
    .canvas.bg-gray { background: #aeb7b1; }
    .canvas.bg-green { background: #00c853; }
    .canvas.bg-blue { background: #4aa3ff; }
    img { max-width: 100%; max-height: 68vh; object-fit: contain; image-rendering: auto; }
    .canvas img { max-width: 100%; max-height: 100%; }
    .result-image { width: 100%; height: 100%; object-fit: contain; transform-origin: center center; user-select: none; pointer-events: none; will-change: transform; align-self: center; justify-self: center; }
    .empty { color: #6a746f; font-size: 14px; }
    .candidate-panel { height: 104px; min-height: 104px; max-height: 104px; display: grid; grid-template-columns: auto 1fr; align-items: center; gap: 12px; padding: 12px 16px; border-top: 1px solid #d9dfd7; background: #fbfcfa; overflow: hidden; }
    .preview.is-mask-mode { grid-template-rows: 48px auto minmax(0, 1fr) 0 56px; }
    .preview.is-mask-mode .candidate-panel { display: none; min-height: 0; padding: 0; border: 0; }
    .candidate-title { font-size: 12px; font-weight: 800; color: #47524c; white-space: nowrap; }
    .candidate-list { min-width: 0; display: flex; gap: 8px; overflow-x: auto; padding: 2px; }
    .candidate-tab { width: 92px; min-width: 92px; min-height: 76px; display: grid; grid-template-rows: 48px auto; gap: 5px; padding: 5px; border: 1px solid #cfd7cc; border-radius: 6px; background: #ffffff; color: #47524c; cursor: pointer; }
    .candidate-tab[aria-selected="true"] { border-color: #196f5a; box-shadow: 0 0 0 2px rgba(25, 111, 90, 0.18); color: #1c2320; }
    .candidate-thumb { width: 100%; height: 48px; display: grid; place-items: center; overflow: hidden; border-radius: 4px; background-position: 0 0, 0 6px, 6px -6px, -6px 0; background-size: 12px 12px; }
    .candidate-thumb img { width: 100%; height: 100%; object-fit: contain; }
    .candidate-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; font-weight: 800; line-height: 1.1; }
    @media (max-width: 760px) { body { height: auto; min-height: 100vh; overflow: auto; } header { padding: 0 16px; } main { height: auto; min-height: 0; grid-template-columns: 1fr; padding: 16px; overflow: visible; } form { max-height: none; } .preview { min-height: 620px; height: min(720px, calc(100vh - 32px)); grid-template-rows: auto auto minmax(0, 1fr) 104px 56px; } .preview-bar { min-height: 84px; align-items: stretch; flex-direction: column; justify-content: center; padding: 10px 16px; } .tabs { width: 100%; overflow-x: auto; } .canvas { min-height: 0; height: 100%; } .candidate-panel { grid-template-columns: 1fr; align-items: stretch; gap: 8px; } .source-frame { aspect-ratio: 16 / 10; max-height: 340px; } .mask-stage .source-frame { height: 100%; min-height: 0; max-height: none; aspect-ratio: auto; } }
  </style>
</head>
<body>
  <header>
    <h1>ERMBG</h1>
    <div class="header-actions">
      <a class="nav-link" href="/slice">切图</a>
      <a class="nav-link" href="/eval/game">Game Eval</a>
      <span class="status" id="strategy">就绪</span>
    </div>
  </header>
  <main>
    <form id="matte-form">
      <label>图片<input id="file" name="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required></label>
      <label class="inline-label">后端<select id="backend" name="backend"><option value="auto" selected>Auto RouteMatte</option><option value="comfy-pymatting-known-b">comfy-pymatting-known-b</option><option value="pymatting-known-b">pymatting-known-b</option><option value="comfy-corridorkey">comfy-corridorkey</option><option value="comfy-rmbg">comfy-rmbg</option></select></label>
      <button id="submit" type="submit">抠图</button>
      <details class="settings" id="corridorkey-settings" open>
        <summary>[设置]</summary>
        <div class="settings-grid">
          <input id="ck-screen-mode" name="corridorkey_screen_mode" type="hidden" value="auto">
          <label class="inline-label">预设<select id="ck-preset" name="corridorkey_preset"><option value="auto" selected>自动</option><option value="detail_safe">细节保护</option><option value="spill_safe">强去溢色</option><option value="manual">手动参数</option></select></label>
          <label class="inline-label">硬 UI Hint<select id="ck-hard-ui-hint-mode" name="corridorkey_hard_ui_hint_mode" disabled><option value="all_white">全白</option><option value="bbox_2px" selected>bbox 2px</option><option value="boundary_2px">边界 2px</option><option value="boundary_2px_shadow_safe">边界 2px shadow-safe</option><option value="boundary_2px_shadow_safe_edge_floor">边界 2px edge floor</option><option value="translucent_button">玻璃/半透明</option></select></label>
          <label class="inline-label">色彩空间<select id="ck-gamma-space" name="corridorkey_gamma_space"><option value="sRGB" selected>sRGB</option><option value="Linear">Linear</option></select></label>
          <div class="settings-row">
            <label>去溢色<input id="ck-despill" name="corridorkey_despill_strength" type="number" min="0" max="1" step="0.01" value="1"></label>
            <label>精修强度<input id="ck-refiner" name="corridorkey_refiner_strength" type="number" min="0" max="4" step="0.1" value="1"></label>
          </div>
          <div class="settings-row">
            <label>去斑点<select id="ck-auto-despeckle" name="corridorkey_auto_despeckle"><option value="On" selected>开启</option><option value="Off">关闭</option></select></label>
            <label>斑点尺寸<input id="ck-despeckle-size" name="corridorkey_despeckle_size" type="number" min="0" max="4096" step="1" value="400"></label>
          </div>
          <label class="check-label"><span>自动 Mask</span><input id="ck-auto-mask" name="corridorkey_auto_mask" type="checkbox"></label>
          <label class="check-label"><span>颜色保护</span><input id="ck-color-protection" name="corridorkey_color_protection" type="checkbox" checked></label>
          <div class="color-range">
            <div class="range-head"><span>色彩范围</span><output class="range-value" id="ck-protect-range-value">12 / 28</output></div>
            <div class="dual-range" id="ck-protect-range">
              <span class="range-rail"></span>
              <span class="range-fill"></span>
              <input id="ck-protect-bg" name="corridorkey_protection_bg_max" type="range" min="0" max="64" step="0.5" value="12" aria-label="背景端点">
              <input id="ck-protect-fg" name="corridorkey_protection_fg_min" type="range" min="0.5" max="64" step="0.5" value="28" aria-label="保护端点">
            </div>
            <div class="range-labels"><span>背景</span><span>过渡</span><span>保护</span></div>
          </div>
        </div>
      </details>
      <details class="settings" id="pymatting-settings" open>
        <summary>[PyMatting]</summary>
        <div class="settings-grid">
          <label class="check-label"><span>自动适配</span><input id="pm-auto-adapt" name="pymatting_auto_adapt" type="checkbox" checked></label>
          <div class="settings-row">
            <label>算法<select id="pm-method" name="pymatting_method"><option value="cf" selected>closed form</option><option value="knn">KNN</option><option value="lbdm">learning based</option><option value="lkm">large kernel</option><option value="rw">random walk</option><option value="sm">shared matting</option></select></label>
            <label>色彩空间<select id="pm-image-space" name="pymatting_image_space"><option value="linear" selected>linear</option><option value="sRGB">sRGB</option></select></label>
          </div>
          <div class="settings-row">
            <label>背景<select id="pm-bg-source" name="pymatting_bg_source"><option value="auto" selected>auto</option><option value="green">green 0,200,0</option><option value="blue">blue 0,0,200</option><option value="custom">custom</option></select></label>
            <label>自定义 RGB<input id="pm-bg-color" name="pymatting_bg_color" type="text" value="0,200,0" inputmode="numeric"></label>
          </div>
          <details class="settings" id="pymatting-advanced">
            <summary>[高级]</summary>
            <div class="settings-grid">
              <div class="settings-row">
                <label>unknown 宽度<input id="pm-boundary-band" name="pymatting_boundary_band_px" type="number" min="0" max="16" step="1" value="2"></label>
                <label>CG maxiter<input id="pm-cg-maxiter" name="pymatting_cg_maxiter" type="number" min="100" max="10000" step="100" value="1000"></label>
              </div>
              <div class="settings-row">
                <label>背景阈值<input id="pm-bg-threshold" name="pymatting_bg_threshold" type="number" min="0" max="32" step="0.1" value="3.5"></label>
                <label>前景阈值<input id="pm-fg-threshold" name="pymatting_fg_threshold" type="number" min="0" max="96" step="0.5" value="30"></label>
              </div>
              <label>CG rtol<input id="pm-cg-rtol" name="pymatting_cg_rtol" type="number" min="0.00000001" max="0.01" step="any" value="0.000001"></label>
            </div>
          </details>
          <label class="check-label"><span>保留阴影</span><input id="shadow-enabled" name="shadow_enabled" type="checkbox" checked></label>
        </div>
      </details>
    </form>
    <section class="preview" id="preview-panel" aria-label="result preview">
      <div class="preview-bar">
        <strong>PNG 预览</strong>
        <div class="tabs" role="tablist" aria-label="预览背景">
          <button class="tab" type="button" role="tab" aria-selected="true" data-view="mask">遮罩</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="checker">棋盘</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="white">白底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="black">黑底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="gray">灰底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="green">绿幕</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="blue">蓝底</button>
        </div>
      </div>
      <div class="mask-toolbar" id="mask-toolbar" aria-label="遮罩工具栏">
        <div class="mask-tools" id="mask-tools">
          <button id="sam-mask-button" type="button">Sam3</button>
          <span class="mask-mode-toggle" role="group" aria-label="画笔模式">
            <button class="mask-mode-button" type="button" aria-pressed="true" data-mask-mode="keep">保留</button>
            <button class="mask-mode-button" type="button" aria-pressed="false" data-mask-mode="erase">擦除</button>
          </span>
          <label>尺寸<input id="mask-brush-size" type="range" min="4" max="96" step="1" value="28"></label>
          <div class="mask-actions">
            <button id="mask-clear-button" type="button">清空</button>
          </div>
        </div>
      </div>
      <div class="canvas" id="canvas">
        <div class="preview-stage" id="preview-stage"><span class="empty">结果会显示在这里</span></div>
        <div class="source-preview mask-stage" id="source-preview" aria-live="polite">
          <div class="source-frame" id="source-frame"><span class="empty">选择图片后显示预览</span></div>
        </div>
      </div>
      <div class="candidate-panel" aria-label="候选结果">
        <span class="candidate-title">候选</span>
        <div class="candidate-list" id="candidate-list" role="tablist" aria-label="候选缩略图"><span class="empty">候选会显示在这里</span></div>
      </div>
      <div class="preview-actions">
        <span class="preview-statuses">
          <span class="status" id="status">等待上传</span>
        </span>
        <a class="download" id="download" aria-disabled="true" download="ermbg_rgba.png">下载 PNG</a>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("matte-form");
    const file = document.getElementById("file");
    const backend = document.getElementById("backend");
    const submit = document.getElementById("submit");
    const statusEl = document.getElementById("status");
    const strategyEl = document.getElementById("strategy");
    const previewPanel = document.getElementById("preview-panel");
    const canvas = document.getElementById("canvas");
    const previewStage = document.getElementById("preview-stage");
    const download = document.getElementById("download");
    const candidateList = document.getElementById("candidate-list");
    const sourcePreview = document.getElementById("source-preview");
    const sourceFrame = document.getElementById("source-frame");
    const corridorSettings = document.getElementById("corridorkey-settings");
    const pymattingSettings = document.getElementById("pymatting-settings");
    const corridorSettingControls = Array.from(document.querySelectorAll("[name^='corridorkey_']"));
    const pymattingSettingControls = Array.from(document.querySelectorAll("[name^='pymatting_']"));
    const shadowEnabled = document.getElementById("shadow-enabled");
    const autoMask = document.getElementById("ck-auto-mask");
    const hardUiHintMode = document.getElementById("ck-hard-ui-hint-mode");
    const samMaskButton = document.getElementById("sam-mask-button");
    const metaEl = statusEl;
    const sourceMeta = statusEl;
    const maskStatus = statusEl;
    const maskBrushModeButtons = Array.from(document.querySelectorAll("[data-mask-mode]"));
    const maskBrushSize = document.getElementById("mask-brush-size");
    const maskClearButton = document.getElementById("mask-clear-button");
    const protectRange = document.getElementById("ck-protect-range");
    const protectBg = document.getElementById("ck-protect-bg");
    const protectFg = document.getElementById("ck-protect-fg");
    const protectRangeValue = document.getElementById("ck-protect-range-value");
    const backgroundTabs = Array.from(document.querySelectorAll("[data-bg]"));
    const viewTabs = Array.from(document.querySelectorAll("[data-view]"));
    const maskToolbarControls = Array.from(document.querySelectorAll("#mask-toolbar input, #mask-toolbar select, #mask-toolbar button"));
    let sourceUrl = null;
    let candidates = [];
    let activeCandidateIndex = -1;
    let activeView = "mask";
    let activeBackground = "checker";
    let resultImage = null;
    let previewScale = 1;
    let previewPanX = 0;
    let previewPanY = 0;
    let dragStart = null;
    let corridorkeyHintMaskFile = null;
    let sourceImage = null;
    let maskCanvas = null;
    let maskCtx = null;
    let maskDirty = false;
    let maskPainting = false;
    let samMaskRequestId = 0;
    let maskBrushMode = "keep";
    let maskScale = 1;
    let maskPanX = 0;
    let maskPanY = 0;

    function humanSize(bytes) { if (bytes < 1024) return `${bytes} B`; if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`; return `${(bytes / 1024 / 1024).toFixed(2)} MB`; }
    function formatElapsed(ms) { return `${(ms / 1000).toFixed(2)}s`; }
    function setBusy(isBusy) { submit.disabled = isBusy; file.disabled = isBusy; backend.disabled = isBusy; corridorSettingControls.forEach((control) => { control.disabled = isBusy; }); pymattingSettingControls.forEach((control) => { control.disabled = isBusy; }); shadowEnabled.disabled = isBusy; maskToolbarControls.forEach((control) => { control.disabled = isBusy; }); if (!isBusy) syncAutoMaskControls(); submit.textContent = isBusy ? "处理中" : "抠图"; }
    function syncBackendSettings() { corridorSettings.classList.toggle("is-visible", backend.value === "comfy-corridorkey"); pymattingSettings.classList.toggle("is-visible", backend.value === "pymatting-known-b" || backend.value === "comfy-pymatting-known-b"); }
    function syncAutoMaskControls() { hardUiHintMode.disabled = !autoMask.checked; }
    function rangeText(value) { return Number.isInteger(value) ? String(value) : value.toFixed(1); }
    function syncColorProtectionRange(changed) { const minGap = 0.5; const min = Number(protectBg.min); const max = Number(protectBg.max); let low = Number(protectBg.value); let high = Number(protectFg.value); if (low + minGap > high) { if (changed === protectBg) low = high - minGap; else high = low + minGap; } low = Math.max(min, Math.min(max - minGap, low)); high = Math.max(low + minGap, Math.min(max, high)); protectBg.value = String(low); protectFg.value = String(high); const lowPct = ((low - min) / (max - min)) * 100; const highPct = ((high - min) / (max - min)) * 100; protectRange.style.setProperty("--range-low", `${lowPct}%`); protectRange.style.setProperty("--range-high", `${highPct}%`); protectRangeValue.textContent = `${rangeText(low)} / ${rangeText(high)}`; }
    function syncPreviewMode() { const maskMode = activeView === "mask"; previewPanel.classList.toggle("is-mask-mode", maskMode); canvas.classList.toggle("is-mask-mode", maskMode); viewTabs.forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.view === activeView))); backgroundTabs.forEach((tab) => tab.setAttribute("aria-selected", String(activeView === "preview" && tab.dataset.bg === activeBackground))); if (maskMode) layoutMaskCanvas(); }
    function setPreviewView(view) { activeView = view; syncPreviewMode(); }
    function setPreviewBackground(mode) { activeBackground = mode; activeView = "preview"; canvas.classList.remove("bg-white", "bg-black", "bg-gray", "bg-green", "bg-blue"); if (mode !== "checker") canvas.classList.add(`bg-${mode}`); syncPreviewMode(); }
    function resetPreviewTransform() { previewScale = 1; previewPanX = 0; previewPanY = 0; dragStart = null; applyPreviewTransform(); }
    function applyPreviewTransform() { if (resultImage) resultImage.style.transform = `translate(${previewPanX}px, ${previewPanY}px) scale(${previewScale})`; }
    function resetMaskTransform() { maskScale = 1; maskPanX = 0; maskPanY = 0; applyMaskTransform(); }
    function maskTransformCss() { return `translate(-50%, -50%) translate(${maskPanX}px, ${maskPanY}px) scale(${maskScale})`; }
    function applyMaskTransform() { const transform = maskTransformCss(); if (sourceImage) sourceImage.style.transform = transform; if (maskCanvas) maskCanvas.style.transform = transform; }
    function resetResult() { candidates.forEach((candidate) => { if (candidate.revoke) URL.revokeObjectURL(candidate.url); }); candidates = []; activeCandidateIndex = -1; resultImage = null; resetPreviewTransform(); previewStage.innerHTML = '<span class="empty">结果会显示在这里</span>'; canvas.classList.remove("has-image", "is-dragging"); candidateList.innerHTML = '<span class="empty">候选会显示在这里</span>'; metaEl.textContent = "RGBA PNG"; download.removeAttribute("href"); download.setAttribute("aria-disabled", "true"); }
    function clearMaskState() { corridorkeyHintMaskFile = null; maskDirty = false; maskPainting = false; if (maskCanvas) { maskCanvas.remove(); maskCanvas = null; maskCtx = null; } sourceFrame.classList.remove("has-mask"); resetMaskTransform(); }
    function layoutMaskCanvas() { if (!sourceImage || sourceImage.naturalWidth <= 0 || sourceImage.naturalHeight <= 0) return; const frameRect = sourceFrame.getBoundingClientRect(); if (frameRect.width <= 0 || frameRect.height <= 0) { requestAnimationFrame(layoutMaskCanvas); return; } const fit = Math.min(frameRect.width / sourceImage.naturalWidth, frameRect.height / sourceImage.naturalHeight); if (!Number.isFinite(fit) || fit <= 0) return; const displayWidth = Math.max(1, sourceImage.naturalWidth * fit); const displayHeight = Math.max(1, sourceImage.naturalHeight * fit); const transform = maskTransformCss(); sourceImage.style.width = `${displayWidth}px`; sourceImage.style.height = `${displayHeight}px`; sourceImage.style.maxWidth = "none"; sourceImage.style.maxHeight = "none"; sourceImage.style.left = "50%"; sourceImage.style.top = "50%"; sourceImage.style.transform = transform; if (maskCanvas) { maskCanvas.style.left = "50%"; maskCanvas.style.top = "50%"; maskCanvas.style.width = `${displayWidth}px`; maskCanvas.style.height = `${displayHeight}px`; maskCanvas.style.transform = transform; } }
    function ensureMaskCanvas(width, height) { if (!maskCanvas) { maskCanvas = document.createElement("canvas"); maskCanvas.className = "mask-overlay"; sourceFrame.appendChild(maskCanvas); maskCanvas.addEventListener("pointerdown", beginMaskPaint); maskCanvas.addEventListener("pointermove", paintMask); maskCanvas.addEventListener("pointerup", endMaskPaint); maskCanvas.addEventListener("pointercancel", endMaskPaint); } maskCanvas.width = width; maskCanvas.height = height; maskCanvas.style.display = "block"; sourceFrame.classList.add("has-mask"); maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true }); layoutMaskCanvas(); }
    function exportHintMaskFile() { return new Promise((resolve) => { if (!maskCanvas || !maskCtx) { corridorkeyHintMaskFile = null; resolve(null); return; } const width = maskCanvas.width; const height = maskCanvas.height; const pixels = maskCtx.getImageData(0, 0, width, height); const exportCanvas = document.createElement("canvas"); exportCanvas.width = width; exportCanvas.height = height; const exportCtx = exportCanvas.getContext("2d"); const out = exportCtx.createImageData(width, height); for (let i = 0; i < pixels.data.length; i += 4) { const value = pixels.data[i + 3] > 8 ? 255 : 0; out.data[i] = value; out.data[i + 1] = value; out.data[i + 2] = value; out.data[i + 3] = 255; } exportCtx.putImageData(out, 0, 0); exportCanvas.toBlob((blob) => { if (!blob) { resolve(null); return; } corridorkeyHintMaskFile = new File([blob], "edited_hint_mask.png", { type: "image/png" }); resolve(corridorkeyHintMaskFile); }, "image/png"); }); }
    function updateHintMaskFile() { exportHintMaskFile(); }
    function waitForSourceImage() { return new Promise((resolve, reject) => { if (!sourceImage) { reject(new Error("请先选择图片")); return; } if (sourceImage.complete && sourceImage.naturalWidth > 0 && sourceImage.naturalHeight > 0) { resolve(sourceImage); return; } const done = () => { cleanup(); if (sourceImage.naturalWidth > 0 && sourceImage.naturalHeight > 0) resolve(sourceImage); else reject(new Error("图片预览尚未载入")); }; const fail = () => { cleanup(); reject(new Error("图片预览载入失败")); }; const cleanup = () => { sourceImage.removeEventListener("load", done); sourceImage.removeEventListener("error", fail); }; sourceImage.addEventListener("load", done, { once: true }); sourceImage.addEventListener("error", fail, { once: true }); }); }
    function loadMaskOverlay(dataUrl) { return new Promise((resolve, reject) => { const img = new Image(); img.onload = async () => { try { setPreviewView("mask"); sourcePreview.classList.add("is-visible"); await waitForSourceImage(); const width = sourceImage && sourceImage.naturalWidth > 0 ? sourceImage.naturalWidth : img.naturalWidth; const height = sourceImage && sourceImage.naturalHeight > 0 ? sourceImage.naturalHeight : img.naturalHeight; if (width <= 0 || height <= 0) throw new Error("mask 尺寸无效"); ensureMaskCanvas(width, height); maskCtx.clearRect(0, 0, width, height); maskCtx.globalCompositeOperation = "source-over"; maskCtx.drawImage(img, 0, 0, width, height); const pixels = maskCtx.getImageData(0, 0, width, height); const data = pixels.data; for (let i = 0; i < data.length; i += 4) { const value = data[i]; data[i] = 0; data[i + 1] = 190; data[i + 2] = 255; data[i + 3] = value > 8 ? 190 : 0; } maskCtx.putImageData(pixels, 0, 0); maskDirty = false; corridorkeyHintMaskFile = null; requestAnimationFrame(() => { layoutMaskCanvas(); maskCanvas.style.display = "block"; sourceFrame.classList.add("has-mask"); }); maskStatus.textContent = "已生成 Sam3 mask，可编辑后作为自定义 mask"; resolve(); } catch (error) { reject(error); } }; img.onerror = () => reject(new Error("Sam3 mask 载入失败")); img.src = dataUrl; }); }
    function canvasPoint(event) { const rect = maskCanvas.getBoundingClientRect(); return { x: ((event.clientX - rect.left) / rect.width) * maskCanvas.width, y: ((event.clientY - rect.top) / rect.height) * maskCanvas.height }; }
    function setMaskBrushMode(mode) { maskBrushMode = mode === "erase" ? "erase" : "keep"; maskBrushModeButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.maskMode === maskBrushMode))); }
    function drawMaskBrush(event) { if (!maskCanvas || !maskCtx) return; const p = canvasPoint(event); const radius = Number(maskBrushSize.value || 28); maskCtx.save(); maskCtx.globalCompositeOperation = maskBrushMode === "erase" ? "destination-out" : "source-over"; maskCtx.fillStyle = "rgba(0,190,255,0.62)"; maskCtx.beginPath(); maskCtx.arc(p.x, p.y, radius, 0, Math.PI * 2); maskCtx.fill(); maskCtx.restore(); maskDirty = true; updateHintMaskFile(); maskStatus.textContent = "edited mask"; }
    function beginMaskPaint(event) { if (!maskCanvas) return; event.preventDefault(); event.stopPropagation(); maskPainting = true; maskCanvas.setPointerCapture(event.pointerId); drawMaskBrush(event); }
    function paintMask(event) { if (!maskPainting) return; event.preventDefault(); event.stopPropagation(); drawMaskBrush(event); }
    function endMaskPaint(event) { if (!maskPainting) return; event.preventDefault(); event.stopPropagation(); maskPainting = false; try { maskCanvas.releasePointerCapture(event.pointerId); } catch (_) {} }
    function renderCandidateTabs() { candidateList.innerHTML = ""; if (!candidates.length) { candidateList.innerHTML = '<span class="empty">候选会显示在这里</span>'; return; } candidates.forEach((candidate, index) => { const button = document.createElement("button"); button.className = "candidate-tab"; button.type = "button"; button.role = "tab"; button.setAttribute("aria-selected", String(index === activeCandidateIndex)); button.dataset.index = String(index); button.title = candidate.label; const thumb = document.createElement("span"); thumb.className = "candidate-thumb"; const img = document.createElement("img"); img.src = candidate.url; img.alt = `${candidate.label} 缩略图`; thumb.appendChild(img); const label = document.createElement("span"); label.className = "candidate-name"; label.textContent = candidate.label; button.appendChild(thumb); button.appendChild(label); button.addEventListener("click", () => setActiveCandidate(index)); candidateList.appendChild(button); }); }
    function setActiveCandidate(index) { if (index < 0 || index >= candidates.length) return; const candidate = candidates[index]; activeCandidateIndex = index; resetPreviewTransform(); previewStage.innerHTML = ""; const img = document.createElement("img"); img.src = candidate.url; img.alt = candidate.label; img.draggable = false; img.className = "result-image"; resultImage = img; canvas.classList.add("has-image"); previewStage.appendChild(img); applyPreviewTransform(); download.href = candidate.url; download.download = candidate.downloadName; download.setAttribute("aria-disabled", "false"); metaEl.textContent = candidate.meta; renderCandidateTabs(); setPreviewView("preview"); }
    function setCandidatePayloads(payload, name) { resetResult(); const stem = name.replace(/\\.[^.]+$/, ""); candidates = (payload.candidates || []).map((candidate, index) => ({ url: candidate.rgba, revoke: false, label: candidate.label || `候选 ${index + 1}`, selected: candidate.selected === true, meta: `候选 ${index + 1} / ${payload.candidates.length} · ${candidate.kind || "RGBA PNG"}`, downloadName: candidate.filename || `${stem}_${candidate.id || `candidate_${index + 1}`}.png` })); if (!candidates.length) throw new Error("没有可显示的候选结果"); const selectedIndex = candidates.findIndex((candidate) => candidate.selected); setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0); }
    function dataUrlToFile(dataUrl, filename) { const [header, base64] = dataUrl.split(","); const mime = (header.match(/data:(.*);base64/) || [])[1] || "image/png"; const binary = atob(base64); const bytes = new Uint8Array(binary.length); for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i); return new File([bytes], filename, { type: mime }); }
    async function generateSamMask() { if (!file.files.length) { setPreviewView("mask"); maskStatus.textContent = "请先选择图片"; return; } const requestId = samMaskRequestId + 1; samMaskRequestId = requestId; setPreviewView("mask"); const formData = new FormData(); formData.append("file", file.files[0]); setBusy(true); maskStatus.textContent = "Sam3 生成中"; try { await waitForSourceImage(); const response = await fetch("/api/sam-mask", { method: "POST", body: formData }); if (!response.ok) { let message = "Sam3 mask 失败"; try { const payload = await response.json(); message = payload.detail || message; } catch (_) {} throw new Error(message); } const payload = await response.json(); if (requestId !== samMaskRequestId) return; await loadMaskOverlay(payload.mask); const elapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : ""; maskStatus.textContent = elapsed ? `已生成 Sam3 mask · ${elapsed}` : "已生成 Sam3 mask"; } catch (error) { if (requestId === samMaskRequestId) { clearMaskState(); maskStatus.textContent = error.message; } } finally { if (requestId === samMaskRequestId) setBusy(false); } }
    function loadPendingSlice() { const raw = sessionStorage.getItem("ermbgPendingSlice"); if (!raw) return; sessionStorage.removeItem("ermbgPendingSlice"); try { const pending = JSON.parse(raw); const sliceFile = dataUrlToFile(pending.rgb, pending.filename || "slice.png"); const transfer = new DataTransfer(); transfer.items.add(sliceFile); file.files = transfer.files; backend.value = "auto"; setPreviewView("mask"); sourcePreview.classList.add("is-visible"); sourceFrame.innerHTML = ""; const img = document.createElement("img"); sourceImage = img; resetMaskTransform(); img.alt = "切图预览"; img.onload = () => { resetMaskTransform(); layoutMaskCanvas(); }; img.onerror = () => { sourceMeta.textContent = "切图预览载入失败"; }; sourceFrame.appendChild(img); img.src = pending.rgb; sourceMeta.textContent = `${sliceFile.name} · ${pending.meta || "来自切图"}`; statusEl.textContent = "已载入切图，可直接抠图"; strategyEl.textContent = backend.value; } catch (error) { statusEl.textContent = "切图载入失败"; } }

    file.addEventListener("change", () => { if (!file.files.length) return; resetResult(); clearMaskState(); setPreviewView("mask"); statusEl.textContent = "等待抠图"; strategyEl.textContent = backend.value; if (sourceUrl) URL.revokeObjectURL(sourceUrl); const selected = file.files[0]; sourceUrl = URL.createObjectURL(selected); sourcePreview.classList.add("is-visible"); sourceFrame.innerHTML = ""; const img = document.createElement("img"); sourceImage = img; resetMaskTransform(); img.alt = "上传图片预览"; img.onload = () => { sourceMeta.textContent = `${img.naturalWidth}x${img.naturalHeight} · ${humanSize(selected.size)}`; resetMaskTransform(); layoutMaskCanvas(); }; img.onerror = () => { sourceMeta.textContent = `无法预览 · ${humanSize(selected.size)}`; }; sourceFrame.appendChild(img); img.src = sourceUrl; });
    backend.addEventListener("change", () => { strategyEl.textContent = backend.value; syncBackendSettings(); });
    protectBg.addEventListener("input", () => syncColorProtectionRange(protectBg));
    protectFg.addEventListener("input", () => syncColorProtectionRange(protectFg));
    autoMask.addEventListener("change", () => syncAutoMaskControls());
    samMaskButton.addEventListener("click", () => generateSamMask());
    maskBrushModeButtons.forEach((button) => button.addEventListener("click", () => setMaskBrushMode(button.dataset.maskMode)));
    maskClearButton.addEventListener("click", () => { if (!maskCanvas || !maskCtx) return; maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); maskDirty = true; updateHintMaskFile(); maskStatus.textContent = "edited mask"; });
    window.addEventListener("resize", () => layoutMaskCanvas());
    viewTabs.forEach((tab) => tab.addEventListener("click", () => setPreviewView(tab.dataset.view)));
    backgroundTabs.forEach((tab) => tab.addEventListener("click", () => setPreviewBackground(tab.dataset.bg)));
    canvas.addEventListener("wheel", (event) => { if (activeView === "mask") { if (!sourceImage) return; event.preventDefault(); const rect = sourceFrame.getBoundingClientRect(); const centerX = rect.left + rect.width / 2; const centerY = rect.top + rect.height / 2; const pointerX = event.clientX - centerX; const pointerY = event.clientY - centerY; const previousScale = maskScale; const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12; maskScale = Math.min(8, Math.max(1, maskScale * factor)); maskPanX = pointerX - ((pointerX - maskPanX) * maskScale) / previousScale; maskPanY = pointerY - ((pointerY - maskPanY) * maskScale) / previousScale; if (maskScale === 1) { maskPanX = 0; maskPanY = 0; } applyMaskTransform(); return; } if (!resultImage) return; event.preventDefault(); const rect = canvas.getBoundingClientRect(); const centerX = rect.left + rect.width / 2; const centerY = rect.top + rect.height / 2; const pointerX = event.clientX - centerX; const pointerY = event.clientY - centerY; const previousScale = previewScale; const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12; previewScale = Math.min(8, Math.max(0.2, previewScale * factor)); previewPanX = pointerX - ((pointerX - previewPanX) * previewScale) / previousScale; previewPanY = pointerY - ((pointerY - previewPanY) * previewScale) / previousScale; applyPreviewTransform(); }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => { if (activeView === "mask" || !resultImage) return; dragStart = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: previewPanX, panY: previewPanY }; canvas.setPointerCapture(event.pointerId); canvas.classList.add("is-dragging"); });
    canvas.addEventListener("pointermove", (event) => { if (!dragStart || dragStart.pointerId !== event.pointerId) return; previewPanX = dragStart.panX + event.clientX - dragStart.x; previewPanY = dragStart.panY + event.clientY - dragStart.y; applyPreviewTransform(); });
    function endDrag(event) { if (!dragStart || dragStart.pointerId !== event.pointerId) return; dragStart = null; canvas.classList.remove("is-dragging"); }
    canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag); canvas.addEventListener("dblclick", () => { if (activeView !== "mask") resetPreviewTransform(); });
    form.addEventListener("submit", async (event) => { event.preventDefault(); if (!file.files.length) return; const formData = new FormData(); formData.append("file", file.files[0]); formData.append("backend", backend.value); formData.append("shadow_enabled", shadowEnabled.checked ? "true" : "false"); corridorSettingControls.forEach((control) => { if (control.type === "checkbox") formData.append(control.name, control.checked ? "true" : "false"); else formData.append(control.name, control.value); }); pymattingSettingControls.forEach((control) => { if (control.type === "checkbox") formData.append(control.name, control.checked ? "true" : "false"); else formData.append(control.name, control.value); }); const shouldUseCustomMask = backend.value === "comfy-corridorkey" && !autoMask.checked && maskDirty; const hintMaskFile = shouldUseCustomMask ? await exportHintMaskFile() : null; if (hintMaskFile) formData.append("corridorkey_hint_mask", hintMaskFile); setBusy(true); statusEl.textContent = "正在抠图"; strategyEl.textContent = backend.value; const startedAt = performance.now(); try { const response = await fetch("/api/matte-candidates", { method: "POST", body: formData }); if (!response.ok) { let message = "处理失败"; try { const payload = await response.json(); message = payload.detail || message; } catch (_) {} throw new Error(message); } const payload = await response.json(); const elapsed = formatElapsed(performance.now() - startedAt); const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null; setCandidatePayloads(payload, file.files[0].name); const strategy = payload.strategy || "done"; const bg = Array.isArray(payload.background) ? payload.background.join(",") : ""; statusEl.textContent = serverElapsed ? `完成 · client ${elapsed} · server ${serverElapsed} · ${payload.backend || backend.value}` : `完成 · ${elapsed}`; strategyEl.textContent = bg ? `${strategy} · ${bg}` : strategy; } catch (error) { statusEl.textContent = error.message; } finally { setBusy(false); } });
    syncColorProtectionRange();
    syncBackendSettings();
    syncAutoMaskControls();
    syncPreviewMode();
    loadPendingSlice();
  </script>
</body>
</html>"""

    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1c2320;
      background: #f5f7f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid #d9dfd7;
      background: #ffffff;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .header-actions {
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .nav-link {
      color: #196f5a;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
      white-space: nowrap;
    }
    main {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      align-items: start;
    }
    form, .preview {
      background: #ffffff;
      border: 1px solid #d9dfd7;
      border-radius: 8px;
    }
    form {
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    label, .field {
      display: grid;
      gap: 8px;
      font-size: 13px;
      font-weight: 600;
      color: #47524c;
    }
    input, select, button {
      width: 100%;
      min-height: 40px;
      border-radius: 6px;
      border: 1px solid #b8c1b7;
      background: #ffffff;
      color: #1c2320;
      font: inherit;
    }
    input[type="file"] { padding: 8px; }
    button, a.download {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: #196f5a;
      color: #ffffff;
      text-decoration: none;
      font-weight: 700;
      cursor: pointer;
    }
    a.mode-button {
      text-decoration: none;
    }
    button:disabled, a.download[aria-disabled="true"] {
      opacity: 0.55;
      cursor: not-allowed;
      pointer-events: none;
    }
    .mode-switch {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      padding: 4px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #f7f9f6;
    }
    .mode-button {
      width: 100%;
      min-height: 34px;
      border: 0;
      background: transparent;
      color: #47524c;
      font-size: 13px;
    }
    .mode-button[aria-pressed="true"] {
      background: #196f5a;
      color: #ffffff;
    }
    .slice-settings {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .slice-settings.is-visible {
      display: grid;
    }
    .slice-settings input {
      min-height: 38px;
      padding: 0 10px;
    }
    .source-preview {
      display: none;
      gap: 10px;
    }
    .source-preview.is-visible {
      display: grid;
    }
    .source-frame {
      width: 100%;
      aspect-ratio: 4 / 3;
      min-height: 148px;
      display: grid;
      place-items: center;
      border: 1px solid #d9dfd7;
      border-radius: 6px;
      background-color: #eef2ec;
      background-image:
        linear-gradient(45deg, #d7dfd4 25%, transparent 25%),
        linear-gradient(-45deg, #d7dfd4 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d7dfd4 75%),
        linear-gradient(-45deg, transparent 75%, #d7dfd4 75%);
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }
    .source-frame img {
      display: block;
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      object-position: center;
    }
    .source-meta {
      min-height: auto;
      font-size: 12px;
      line-height: 1.4;
      color: #5d6862;
      overflow-wrap: anywhere;
    }
    .preview {
      min-height: 520px;
      display: grid;
      grid-template-rows: 48px 1fr 104px 56px;
      overflow: hidden;
    }
    .preview-bar, .preview-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 16px;
      border-bottom: 1px solid #d9dfd7;
    }
    .preview-actions {
      border-top: 1px solid #d9dfd7;
      border-bottom: 0;
    }
    .preview-actions button {
      width: auto;
      min-width: 108px;
      padding: 0 14px;
    }
    .tabs {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #f7f9f6;
      flex-shrink: 0;
    }
    .tab {
      width: auto;
      min-height: 30px;
      padding: 0 10px;
      border: 0;
      border-radius: 4px;
      background: transparent;
      color: #47524c;
      font-size: 12px;
      font-weight: 700;
    }
    .tab[aria-selected="true"] {
      background: #196f5a;
      color: #ffffff;
    }
    .status {
      font-size: 13px;
      color: #5d6862;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .canvas {
      min-height: 416px;
      display: grid;
      place-items: center;
      padding: 16px;
      overflow: hidden;
      touch-action: none;
      background-color: #e9eee6;
      background-image:
        linear-gradient(45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(-45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d3dbd0 75%),
        linear-gradient(-45deg, transparent 75%, #d3dbd0 75%);
      background-position: 0 0, 0 12px, 12px -12px, -12px 0;
      background-size: 24px 24px;
    }
    .canvas.has-image {
      cursor: grab;
    }
    .canvas.is-dragging {
      cursor: grabbing;
    }
    .canvas.bg-white { background: #ffffff; }
    .canvas.bg-black { background: #111514; }
    .canvas.bg-gray { background: #aeb7b1; }
    .canvas.bg-green { background: #00c853; }
    .canvas.bg-blue { background: #4aa3ff; }
    img {
      max-width: 100%;
      max-height: 68vh;
      object-fit: contain;
      image-rendering: auto;
    }
    .result-image {
      transform-origin: center center;
      user-select: none;
      pointer-events: none;
      will-change: transform;
    }
    .empty {
      color: #6a746f;
      font-size: 14px;
    }
    .candidate-panel {
      min-height: 104px;
      display: grid;
      grid-template-columns: auto 1fr;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-top: 1px solid #d9dfd7;
      background: #fbfcfa;
    }
    .candidate-title {
      font-size: 12px;
      font-weight: 800;
      color: #47524c;
      white-space: nowrap;
    }
    .candidate-list {
      min-width: 0;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 2px;
    }
    .candidate-tab {
      width: 92px;
      min-width: 92px;
      min-height: 76px;
      display: grid;
      grid-template-rows: 48px auto;
      gap: 5px;
      padding: 5px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #ffffff;
      color: #47524c;
      cursor: pointer;
    }
    .candidate-tab[aria-selected="true"] {
      border-color: #196f5a;
      box-shadow: 0 0 0 2px rgba(25, 111, 90, 0.18);
      color: #1c2320;
    }
    .candidate-thumb {
      width: 100%;
      height: 48px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 4px;
      background-color: #e9eee6;
      background-image:
        linear-gradient(45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(-45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d3dbd0 75%),
        linear-gradient(-45deg, transparent 75%, #d3dbd0 75%);
      background-position: 0 0, 0 6px, 6px -6px, -6px 0;
      background-size: 12px 12px;
    }
    .candidate-thumb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .candidate-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      font-weight: 800;
      line-height: 1.1;
    }
    .slice-list {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      max-height: 232px;
      overflow-y: auto;
      overflow-x: hidden;
    }
    .slice-row {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 8px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #ffffff;
    }
    .slice-thumb {
      width: 72px;
      height: 56px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 4px;
      background-color: #e9eee6;
      background-image:
        linear-gradient(45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(-45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d3dbd0 75%),
        linear-gradient(-45deg, transparent 75%, #d3dbd0 75%);
      background-position: 0 0, 0 6px, 6px -6px, -6px 0;
      background-size: 12px 12px;
    }
    .slice-thumb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .slice-info {
      min-width: 0;
      display: grid;
      gap: 3px;
    }
    .slice-label {
      font-size: 13px;
      font-weight: 800;
      color: #1c2320;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .slice-meta {
      font-size: 12px;
      color: #5d6862;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .slice-row button {
      width: auto;
      min-height: 34px;
      padding: 0 12px;
      white-space: nowrap;
    }
    @media (max-width: 760px) {
      header { padding: 0 16px; }
      main {
        grid-template-columns: 1fr;
        padding: 16px;
      }
      .preview {
        min-height: 420px;
        grid-template-rows: auto 1fr 104px 56px;
      }
      .preview-bar {
        min-height: 84px;
        align-items: stretch;
        flex-direction: column;
        justify-content: center;
        padding: 10px 16px;
      }
      .tabs {
        width: 100%;
        overflow-x: auto;
      }
      .canvas { min-height: 312px; }
      .candidate-panel {
        grid-template-columns: 1fr;
        align-items: stretch;
        gap: 8px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>ERMBG</h1>
    <div class="header-actions">
      <a class="nav-link" href="/eval/game">Game Eval</a>
      <span class="status" id="strategy">就绪</span>
    </div>
  </header>
  <main>
    <form id="matte-form">
      <label>
        图片
        <input id="file" name="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required>
      </label>
      <div class="source-preview" id="source-preview" aria-live="polite">
        <div class="source-frame" id="source-frame">
          <span class="empty">选择图片后显示预览</span>
        </div>
        <div class="source-meta" id="source-meta">未选择图片</div>
      </div>
      <div class="field">
        任务
        <span class="mode-switch" role="group" aria-label="任务">
          <a class="mode-button" href="/" data-task="matte" role="button" aria-pressed="true">抠图</a>
          <a class="mode-button" href="/slice" data-task="slice" role="button" aria-pressed="false">切图</a>
        </span>
        <select id="task" name="task" hidden>
          <option value="matte" selected>抠图</option>
          <option value="slice">切图</option>
        </select>
      </div>
      <label>
        后端
        <select id="backend" name="backend">
          <option value="auto" selected>Auto RouteMatte</option>
          <option value="comfy-pymatting-known-b">comfy-pymatting-known-b</option>
          <option value="pymatting-known-b">pymatting-known-b</option>
          <option value="comfy-corridorkey">comfy-corridorkey</option>
          <option value="comfy-rmbg">comfy-rmbg</option>
          
          
          
        </select>
      </label>
      <div class="slice-settings" id="slice-settings">
        <label>
          最小面积
          <input id="slice-min-area" name="min_area" type="number" min="1" step="1" value="64">
        </label>
        <label>
          边距
          <input id="slice-padding" name="padding" type="number" min="0" step="1" value="2">
        </label>
      </div>
      <button id="submit" type="submit">抠图</button>
    </form>
    <section class="preview" aria-label="result preview">
      <div class="preview-bar">
        <strong>PNG 预览</strong>
        <div class="tabs" role="tablist" aria-label="预览背景">
          <button class="tab" type="button" role="tab" aria-selected="true" data-bg="checker">棋盘</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="white">白底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="black">黑底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="gray">灰底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="green">绿幕</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="blue">蓝底</button>
        </div>
        <span class="status" id="status">等待上传</span>
      </div>
      <div class="canvas" id="canvas">
        <span class="empty">结果会显示在这里</span>
      </div>
      <div class="candidate-panel" aria-label="候选结果">
        <span class="candidate-title">候选</span>
        <div class="candidate-list" id="candidate-list" role="tablist" aria-label="候选缩略图">
          <span class="empty">候选会显示在这里</span>
        </div>
      </div>
      <div class="preview-actions">
        <span class="status" id="meta">RGBA PNG</span>
        <button id="confirm-slices" type="button" disabled hidden>生成切图</button>
        <a class="download" id="download" aria-disabled="true" download="ermbg_rgba.png">下载 PNG</a>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("matte-form");
    const file = document.getElementById("file");
    const backend = document.getElementById("backend");
    const task = document.getElementById("task");
    const modeButtons = Array.from(document.querySelectorAll(".mode-button"));
    const sliceSettings = document.getElementById("slice-settings");
    const sliceMinArea = document.getElementById("slice-min-area");
    const slicePadding = document.getElementById("slice-padding");
    const submit = document.getElementById("submit");
    const confirmSlices = document.getElementById("confirm-slices");
    const statusEl = document.getElementById("status");
    const strategyEl = document.getElementById("strategy");
    const canvas = document.getElementById("canvas");
    const download = document.getElementById("download");
    const candidateList = document.getElementById("candidate-list");
    const metaEl = document.getElementById("meta");
    const sourcePreview = document.getElementById("source-preview");
    const sourceFrame = document.getElementById("source-frame");
    const sourceMeta = document.getElementById("source-meta");
    const tabs = Array.from(document.querySelectorAll(".tab"));
    let sourceUrl = null;
    let candidates = [];
    let activeCandidateIndex = -1;
    let resultImage = null;
    let previewScale = 1;
    let previewPanX = 0;
    let previewPanY = 0;
    let dragStart = null;
    let slicePreviewPayload = null;

    function humanSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }

    function formatElapsed(ms) {
      return `${(ms / 1000).toFixed(2)}s`;
    }

    function setBusy(isBusy) {
      submit.disabled = isBusy;
      file.disabled = isBusy;
      backend.disabled = isBusy || task.value === "slice";
      task.disabled = isBusy;
      modeButtons.forEach((button) => {
        button.setAttribute("aria-disabled", String(isBusy));
      });
      sliceMinArea.disabled = isBusy;
      slicePadding.disabled = isBusy;
      confirmSlices.disabled = isBusy || !slicePreviewPayload;
      submit.textContent = isBusy ? "处理中" : (task.value === "slice" ? "自动标注" : "抠图");
    }

    function setTaskMode(mode) {
      task.value = mode;
      modeButtons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.task === mode));
      });
      sliceSettings.classList.toggle("is-visible", mode === "slice");
      backend.disabled = mode === "slice";
      submit.textContent = mode === "slice" ? "自动标注" : "抠图";
      statusEl.textContent = mode === "slice" ? "等待切图标注" : "等待抠图";
      confirmSlices.hidden = mode !== "slice";
      confirmSlices.disabled = !slicePreviewPayload;
      metaEl.textContent = mode === "slice" ? "切图预览" : "RGBA PNG";
      if (mode !== "slice") {
        slicePreviewPayload = null;
      }
    }

    function setPreviewBackground(mode) {
      canvas.classList.remove("bg-white", "bg-black", "bg-gray", "bg-green", "bg-blue");
      if (mode !== "checker") canvas.classList.add(`bg-${mode}`);
      tabs.forEach((tab) => {
        tab.setAttribute("aria-selected", String(tab.dataset.bg === mode));
      });
    }

    function resetResult() {
      candidates.forEach((candidate) => {
        if (candidate.revoke) URL.revokeObjectURL(candidate.url);
      });
      candidates = [];
      activeCandidateIndex = -1;
      resultImage = null;
      slicePreviewPayload = null;
      resetPreviewTransform();
      canvas.innerHTML = '<span class="empty">结果会显示在这里</span>';
      canvas.classList.remove("has-image", "is-dragging");
      candidateList.className = "candidate-list";
      candidateList.innerHTML = '<span class="empty">候选会显示在这里</span>';
      metaEl.textContent = "RGBA PNG";
      confirmSlices.disabled = true;
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
    }

    function clampScale(value) {
      return Math.min(8, Math.max(0.2, value));
    }

    function applyPreviewTransform() {
      if (!resultImage) return;
      resultImage.style.transform = `translate(${previewPanX}px, ${previewPanY}px) scale(${previewScale})`;
    }

    function resetPreviewTransform() {
      previewScale = 1;
      previewPanX = 0;
      previewPanY = 0;
      dragStart = null;
      applyPreviewTransform();
    }

    function renderCandidateTabs() {
      candidateList.innerHTML = "";
      if (!candidates.length) {
        candidateList.innerHTML = '<span class="empty">候选会显示在这里</span>';
        return;
      }
      candidates.forEach((candidate, index) => {
        const button = document.createElement("button");
        button.className = "candidate-tab";
        button.type = "button";
        button.role = "tab";
        button.setAttribute("aria-selected", String(index === activeCandidateIndex));
        button.dataset.index = String(index);
        button.title = candidate.label;

        const thumb = document.createElement("span");
        thumb.className = "candidate-thumb";
        const img = document.createElement("img");
        img.src = candidate.url;
        img.alt = `${candidate.label} 缩略图`;
        thumb.appendChild(img);

        const label = document.createElement("span");
        label.className = "candidate-name";
        label.textContent = candidate.label;

        button.appendChild(thumb);
        button.appendChild(label);
        button.addEventListener("click", () => setActiveCandidate(index));
        candidateList.appendChild(button);
      });
    }

    function setActiveCandidate(index) {
      if (index < 0 || index >= candidates.length) return;
      const candidate = candidates[index];
      activeCandidateIndex = index;
      resetPreviewTransform();
      canvas.innerHTML = "";
      const img = document.createElement("img");
      img.src = candidate.url;
      img.alt = candidate.label;
      img.draggable = false;
      img.className = "result-image";
      resultImage = img;
      canvas.classList.add("has-image");
      canvas.appendChild(img);
      applyPreviewTransform();
      download.href = candidate.url;
      download.download = candidate.downloadName;
      download.setAttribute("aria-disabled", "false");
      metaEl.textContent = candidate.meta;
      renderCandidateTabs();
    }

    function setDownload(blob, name) {
      resetResult();
      const url = URL.createObjectURL(blob);
      const stem = name.replace(/\\.[^.]+$/, "");
      candidates = [{
        url,
        revoke: true,
        label: "自动结果",
        meta: "候选 1 / 1 · RGBA PNG",
        downloadName: `${stem}_rgba.png`,
      }];
      setActiveCandidate(0);
    }

    function setCandidatePayloads(payload, name) {
      resetResult();
      const stem = name.replace(/\\.[^.]+$/, "");
      candidates = (payload.candidates || []).map((candidate, index) => ({
        url: candidate.rgba,
        revoke: false,
        label: candidate.label || `候选 ${index + 1}`,
        selected: candidate.selected === true,
        meta: `候选 ${index + 1} / ${payload.candidates.length} · ${candidate.kind || "RGBA PNG"}`,
        downloadName: candidate.filename || `${stem}_${candidate.id || `candidate_${index + 1}`}.png`,
      }));
      if (!candidates.length) {
        throw new Error("没有可显示的候选结果");
      }
      const selectedIndex = candidates.findIndex((candidate) => candidate.selected);
      setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0);
    }

    function setSlicePreviewPayload(payload) {
      resetResult();
      slicePreviewPayload = payload;
      canvas.innerHTML = "";
      const img = document.createElement("img");
      img.src = payload.annotated;
      img.alt = "自动切图标注预览";
      img.draggable = false;
      img.className = "result-image";
      resultImage = img;
      canvas.classList.add("has-image");
      canvas.appendChild(img);
      applyPreviewTransform();
      candidateList.innerHTML = '<span class="empty">确认标注后生成切图列表</span>';
      confirmSlices.hidden = false;
      confirmSlices.disabled = !payload.count;
      metaEl.textContent = `检测到 ${payload.count || 0} 个矩形`;
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
    }

    function dataUrlToBlob(dataUrl) {
      const [header, base64] = dataUrl.split(",");
      const mime = (header.match(/data:(.*);base64/) || [])[1] || "image/png";
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return new Blob([bytes], { type: mime });
    }

    async function matteSlice(crop) {
      const formData = new FormData();
      formData.append("file", dataUrlToBlob(crop.rgb), crop.filename);
      formData.append("backend", backend.value);
      setBusy(true);
      statusEl.textContent = `正在抠图 · ${crop.label}`;
      strategyEl.textContent = crop.label;
      const startedAt = performance.now();
      try {
        const response = await fetch("/api/matte-candidates", { method: "POST", body: formData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        const payload = await response.json();
        const elapsed = formatElapsed(performance.now() - startedAt);
        const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null;
        setCandidatePayloads(payload, crop.filename);
        const strategy = payload.strategy || "done";
        const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
        statusEl.textContent = serverElapsed ? `完成 · client ${elapsed} · server ${serverElapsed} · ${payload.backend || backend.value}` : `完成 · ${elapsed}`;
        strategyEl.textContent = bg ? `${crop.label} · ${strategy} · ${bg}` : `${crop.label} · ${strategy}`;
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    }

    function renderSliceCrops(payload) {
      resetResult();
      candidateList.className = "candidate-list slice-list";
      if (!payload.crops || !payload.crops.length) {
        candidateList.innerHTML = '<span class="empty">没有检测到可切割主体</span>';
        return;
      }
      canvas.innerHTML = '<span class="empty">选择下方切图进入抠图流程</span>';
      payload.crops.forEach((crop) => {
        const row = document.createElement("div");
        row.className = "slice-row";

        const thumb = document.createElement("span");
        thumb.className = "slice-thumb";
        const img = document.createElement("img");
        img.src = crop.rgb;
        img.alt = `${crop.label} 预览`;
        thumb.appendChild(img);

        const info = document.createElement("span");
        info.className = "slice-info";
        const label = document.createElement("span");
        label.className = "slice-label";
        label.textContent = crop.label;
        const meta = document.createElement("span");
        meta.className = "slice-meta";
        meta.textContent = crop.meta || "";
        info.appendChild(label);
        info.appendChild(meta);

        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "抠图";
        button.addEventListener("click", () => matteSlice(crop));

        row.appendChild(thumb);
        row.appendChild(info);
        row.appendChild(button);
        candidateList.appendChild(row);
      });
      confirmSlices.disabled = true;
      metaEl.textContent = `已生成 ${payload.count || payload.crops.length} 张切图`;
      statusEl.textContent = "切图完成";
    }

    file.addEventListener("change", () => {
      if (!file.files.length) return;
      resetResult();
      statusEl.textContent = task.value === "slice" ? "等待切图标注" : "等待抠图";
      strategyEl.textContent = backend.value;
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);

      const selected = file.files[0];
      sourceUrl = URL.createObjectURL(selected);
      sourcePreview.classList.add("is-visible");
      sourceFrame.innerHTML = "";
      const img = document.createElement("img");
      img.src = sourceUrl;
      img.alt = "上传图片预览";
      img.onload = () => {
        sourceMeta.textContent = `${selected.name} · ${img.naturalWidth}x${img.naturalHeight} · ${humanSize(selected.size)}`;
      };
      img.onerror = () => {
        sourceMeta.textContent = `${selected.name} · 无法预览 · ${humanSize(selected.size)}`;
      };
      sourceFrame.appendChild(img);
    });

    modeButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        resetResult();
        setTaskMode(button.dataset.task);
        const nextPath = button.dataset.task === "slice" ? "/slice" : "/";
        if (window.location.pathname !== nextPath) {
          window.history.replaceState(null, "", nextPath);
        }
      });
    });
    setTaskMode(window.location.pathname === "/slice" ? "slice" : task.value);

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => setPreviewBackground(tab.dataset.bg));
    });

    canvas.addEventListener("wheel", (event) => {
      if (!resultImage) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const pointerX = event.clientX - centerX;
      const pointerY = event.clientY - centerY;
      const previousScale = previewScale;
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      previewScale = clampScale(previewScale * factor);
      previewPanX = pointerX - ((pointerX - previewPanX) * previewScale) / previousScale;
      previewPanY = pointerY - ((pointerY - previewPanY) * previewScale) / previousScale;
      applyPreviewTransform();
    }, { passive: false });

    canvas.addEventListener("pointerdown", (event) => {
      if (!resultImage) return;
      dragStart = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        panX: previewPanX,
        panY: previewPanY,
      };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("is-dragging");
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      previewPanX = dragStart.panX + event.clientX - dragStart.x;
      previewPanY = dragStart.panY + event.clientY - dragStart.y;
      applyPreviewTransform();
    });

    function endDrag(event) {
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      dragStart = null;
      canvas.classList.remove("is-dragging");
    }

    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("dblclick", () => resetPreviewTransform());

    confirmSlices.addEventListener("click", async () => {
      if (!file.files.length || !slicePreviewPayload) return;
      const formData = new FormData();
      formData.append("file", file.files[0]);
      formData.append("min_area", sliceMinArea.value || "64");
      formData.append("padding", slicePadding.value || "2");
      setBusy(true);
      statusEl.textContent = "正在生成切图";
      try {
        const response = await fetch("/api/slice-crops", { method: "POST", body: formData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        renderSliceCrops(await response.json());
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!file.files.length) return;
      const formData = new FormData();
      formData.append("file", file.files[0]);
      if (task.value === "slice") {
        formData.append("min_area", sliceMinArea.value || "64");
        formData.append("padding", slicePadding.value || "2");
      } else {
        formData.append("backend", backend.value);
      }
      setBusy(true);
      statusEl.textContent = task.value === "slice" ? "正在自动标注" : "正在抠图";
      strategyEl.textContent = backend.value;
      const startedAt = performance.now();
      try {
        const endpoint = task.value === "slice" ? "/api/slice-preview" : "/api/matte-candidates";
        const response = await fetch(endpoint, { method: "POST", body: formData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        if (task.value === "slice") {
          const payload = await response.json();
          setSlicePreviewPayload(payload);
          const bg = Array.isArray(payload.background_color) ? payload.background_color.join(",") : "";
          statusEl.textContent = "标注完成";
          strategyEl.textContent = bg ? `slice · ${bg}` : "slice";
        } else {
          const payload = await response.json();
          const elapsed = formatElapsed(performance.now() - startedAt);
          const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null;
          setCandidatePayloads(payload, file.files[0].name);
          const strategy = payload.strategy || "done";
          const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
          statusEl.textContent = serverElapsed ? `完成 · client ${elapsed} · server ${serverElapsed} · ${payload.backend || backend.value}` : `完成 · ${elapsed}`;
          strategyEl.textContent = bg ? `${strategy} · ${bg}` : strategy;
        }
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    });
  </script>
</body>
</html>"""


@app.get("/slice", response_class=HTMLResponse)
def slice_page() -> str:
    return _slice_page_html()


def _slice_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG Slice</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1c2320; background: #f5f7f4; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }
    a { color: #196f5a; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    main { width: min(1120px, 100%); margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 320px 1fr; gap: 24px; align-items: start; }
    form, .workspace { background: #ffffff; border: 1px solid #d9dfd7; border-radius: 8px; }
    form { min-width: 0; min-height: 640px; max-height: 640px; padding: 16px; display: grid; grid-template-rows: auto auto auto minmax(0, 1fr) auto; gap: 12px; overflow: hidden; }
    label { display: grid; gap: 8px; font-size: 13px; font-weight: 700; color: #47524c; }
    input, button { width: 100%; min-height: 40px; border-radius: 6px; border: 1px solid #b8c1b7; background: #ffffff; color: #1c2320; font: inherit; }
    input[type="file"] { padding: 8px; }
    button { border: 0; background: #196f5a; color: #ffffff; font-weight: 800; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .settings { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .preview, .thumb { background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); background-size: 24px 24px; background-position: 0 0, 0 12px, 12px -12px, -12px 0; }
    .workspace { min-height: 640px; display: grid; grid-template-rows: 48px 1fr; overflow: hidden; }
    .bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 16px; border-bottom: 1px solid #d9dfd7; }
    .status { color: #5d6862; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .preview { min-height: 420px; display: grid; place-items: center; padding: 16px; overflow: hidden; }
    .preview img { max-width: 100%; max-height: 72vh; object-fit: contain; }
    .empty { color: #6a746f; font-size: 14px; }
    .left-list { min-width: 0; min-height: 0; height: 100%; max-height: 100%; display: block; overflow-y: auto; overflow-x: hidden; border: 1px solid #cfd7cc; border-radius: 6px; background: #ffffff; scrollbar-gutter: stable; }
    .row { width: 100%; min-width: 0; height: 72px; display: grid; grid-template-columns: 64px minmax(0, 1fr) 52px; gap: 8px; align-items: center; padding: 4px 6px; border: 0; border-bottom: 1px solid #d9dfd7; border-radius: 0; background: #ffffff; text-align: left; cursor: pointer; }
    .row:last-child { border-bottom: 0; }
    .row:hover { background: #f3f7f1; }
    .row[aria-selected="true"] { background: #d7eadf; }
    .thumb { width: 64px; height: 64px; display: grid; place-items: center; overflow: hidden; border-radius: 4px; background-size: 12px 12px; background-position: 0 0, 0 6px, 6px -6px, -6px 0; }
    .thumb img { display: block; width: 100%; height: 100%; max-width: 100%; max-height: 100%; object-fit: contain; object-position: center; }
    .info { min-width: 0; display: grid; gap: 2px; align-content: center; overflow: hidden; }
    .name { min-width: 0; font-size: 12px; line-height: 1.25; font-weight: 800; color: #1c2320; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .meta { min-width: 0; font-size: 11px; line-height: 1.25; font-weight: 600; color: #5d6862; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .row-action { width: 52px; min-width: 52px; height: 30px; min-height: 30px; padding: 0; border-radius: 6px; font-size: 12px; line-height: 1; visibility: hidden; }
    .row[aria-selected="true"] .row-action { visibility: visible; }
    .selected-actions { display: none; gap: 8px; }
    .selected-actions.is-visible { display: grid; }
    @media (max-width: 760px) { header { padding: 0 16px; } main { grid-template-columns: 1fr; padding: 16px; } form { min-height: 520px; } .workspace { min-height: 520px; } .preview { min-height: 320px; } }
  </style>
</head>
<body>
  <header>
    <h1>ERMBG 切图</h1>
    <a href="/">返回抠图</a>
  </header>
  <main>
    <form id="slice-form">
      <label>图片<input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required></label>
      <div class="settings">
        <label>最小面积<input id="min-area" type="number" min="1" step="1" value="64"></label>
        <label>边距<input id="padding" type="number" min="0" step="1" value="2"></label>
      </div>
      <button id="confirm" type="button" disabled>切图</button>
      <div class="left-list" id="list"><span class="empty">切图列表会显示在这里</span></div>
      <div class="selected-actions" id="selected-actions">
        <button id="matte-selected" type="button">抠图</button>
      </div>
    </form>
    <section class="workspace" aria-label="slice workspace">
      <div class="bar"><strong>切图预览</strong><span class="status" id="status">等待上传</span></div>
      <div class="preview" id="preview"><span class="empty">自动标注会显示在这里</span></div>
    </section>
  </main>
  <script>
    const form = document.getElementById("slice-form");
    const file = document.getElementById("file");
    const minArea = document.getElementById("min-area");
    const padding = document.getElementById("padding");
    const confirmButton = document.getElementById("confirm");
    const matteSelected = document.getElementById("matte-selected");
    const selectedActions = document.getElementById("selected-actions");
    const statusEl = document.getElementById("status");
    const preview = document.getElementById("preview");
    const list = document.getElementById("list");
    let hasPreview = false;
    let currentCrops = [];
    let selectedCrop = null;
    const SLICE_STATE_KEY = "ermbgSliceWorkspace";

    function setBusy(isBusy) {
      confirmButton.disabled = isBusy || !hasPreview;
      matteSelected.disabled = isBusy || !selectedCrop;
      file.disabled = isBusy;
      minArea.disabled = isBusy;
      padding.disabled = isBusy;
    }

    function formData() {
      const data = new FormData();
      data.append("file", file.files[0]);
      data.append("min_area", minArea.value || "64");
      data.append("padding", padding.value || "2");
      return data;
    }

    function currentSettings() {
      return {
        minArea: minArea.value || "64",
        padding: padding.value || "2",
      };
    }

    function saveSliceState(patch) {
      let current = {};
      try {
        current = JSON.parse(sessionStorage.getItem(SLICE_STATE_KEY) || "{}");
      } catch (_) {}
      sessionStorage.setItem(
        SLICE_STATE_KEY,
        JSON.stringify({ ...current, ...patch, settings: currentSettings() }),
      );
    }

    function clearSliceState() {
      sessionStorage.removeItem(SLICE_STATE_KEY);
    }

    function showPreview(payload) {
      hasPreview = Boolean(payload.count);
      preview.innerHTML = "";
      const img = document.createElement("img");
      img.src = payload.annotated;
      img.alt = "自动标注预览";
      preview.appendChild(img);
      list.innerHTML = '<span class="empty">确认标注后生成切图列表</span>';
      statusEl.textContent = `标注完成 · ${payload.count || 0} 个矩形`;
      confirmButton.disabled = !hasPreview;
      saveSliceState({ preview: payload, crops: null });
    }

    function sendToMatte(crop) {
      saveSliceState({ selectedCropId: crop.id || crop.filename });
      sessionStorage.setItem("ermbgPendingSlice", JSON.stringify(crop));
      window.location.href = "/";
    }

    function selectCrop(crop) {
      selectedCrop = crop;
      Array.from(list.querySelectorAll(".row")).forEach((row) => {
        row.setAttribute("aria-selected", String(row.dataset.cropId === crop.id));
      });
      preview.innerHTML = "";
      const img = document.createElement("img");
      img.src = crop.rgb;
      img.alt = `${crop.label} 预览`;
      preview.appendChild(img);
      statusEl.textContent = `已选择 ${crop.label}`;
      saveSliceState({ selectedCropId: crop.id || crop.filename });
    }

    function renderCrops(payload) {
      list.innerHTML = "";
      currentCrops = payload.crops || [];
      selectedCrop = null;
      selectedActions.classList.remove("is-visible");
      if (!payload.crops || !payload.crops.length) {
        list.innerHTML = '<span class="empty">没有检测到可切割主体</span>';
        return;
      }
      payload.crops.forEach((crop) => {
        const row = document.createElement("div");
        row.className = "row";
        row.dataset.cropId = crop.id;
        row.setAttribute("aria-selected", "false");
        const thumb = document.createElement("span");
        thumb.className = "thumb";
        const img = document.createElement("img");
        img.src = crop.rgb;
        img.alt = `${crop.label} 预览`;
        thumb.appendChild(img);
        const info = document.createElement("span");
        info.className = "info";
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = crop.label;
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = crop.meta || "";
        info.appendChild(name);
        info.appendChild(meta);
        const action = document.createElement("button");
        action.className = "row-action";
        action.type = "button";
        action.textContent = "抠图";
        action.addEventListener("click", (event) => {
          event.stopPropagation();
          sendToMatte(crop);
        });
        row.appendChild(thumb);
        row.appendChild(info);
        row.appendChild(action);
        row.addEventListener("click", () => selectCrop(crop));
        list.appendChild(row);
      });
      statusEl.textContent = `切图完成 · ${payload.count || payload.crops.length} 张`;
      confirmButton.disabled = true;
      saveSliceState({ crops: payload });
      if (payload.crops.length === 1) {
        selectCrop(payload.crops[0]);
      }
    }

    file.addEventListener("change", () => {
      if (!file.files.length) return;
      hasPreview = false;
      currentCrops = [];
      selectedCrop = null;
      selectedActions.classList.remove("is-visible");
      clearSliceState();
      confirmButton.disabled = true;
      preview.innerHTML = '<span class="empty">自动标注会显示在这里</span>';
      list.innerHTML = '<span class="empty">切图列表会显示在这里</span>';
      runAnnotate();
    });

    async function runAnnotate() {
      if (!file.files.length) return;
      setBusy(true);
      statusEl.textContent = "正在自动标注";
      try {
        const response = await fetch("/api/slice-preview", { method: "POST", body: formData() });
        if (!response.ok) throw new Error((await response.json()).detail || "标注失败");
        showPreview(await response.json());
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      runAnnotate();
    });

    confirmButton.addEventListener("click", async () => {
      if (!file.files.length || !hasPreview) return;
      setBusy(true);
      statusEl.textContent = "正在生成切图";
      try {
        const response = await fetch("/api/slice-crops", { method: "POST", body: formData() });
        if (!response.ok) throw new Error((await response.json()).detail || "切图失败");
        renderCrops(await response.json());
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    });

    matteSelected.addEventListener("click", () => {
      if (!selectedCrop) return;
      sendToMatte(selectedCrop);
    });

    function restoreSliceState() {
      let state = null;
      try {
        state = JSON.parse(sessionStorage.getItem(SLICE_STATE_KEY) || "null");
      } catch (_) {
        state = null;
      }
      if (!state) return;
      if (state.settings) {
        minArea.value = state.settings.minArea || minArea.value;
        padding.value = state.settings.padding || padding.value;
      }
      if (state.preview) {
        showPreview(state.preview);
      }
      if (state.crops) {
        renderCrops(state.crops);
        const crop = currentCrops.find((item) => item.id === state.selectedCropId);
        if (crop) {
          selectCrop(crop);
        }
        statusEl.textContent = state.selectedCropId ? "已返回切图列表" : "切图已恢复";
      }
    }

    restoreSliceState();
  </script>
</body>
</html>"""


def _png_data_url(rgba: np.ndarray) -> str:
    encoded = base64.b64encode(_encode_png(rgba)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rgb_png_data_url(rgb: np.ndarray) -> str:
    encoded = base64.b64encode(_encode_rgb_png(rgb)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mask_png_data_url(mask: np.ndarray) -> str:
    arr = np.clip(mask.astype(np.float32), 0.0, 1.0)
    u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _json_safe_debug(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        summary: dict[str, object] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            summary.update(
                {
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                    "mean": float(np.mean(value)),
                }
            )
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe_debug(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_debug(v) for v in value]
    return value


def _candidate_payload(candidate: MatteCandidate, stem: str) -> dict[str, object]:
    debug = _json_safe_debug(candidate.debug)
    return {
        "id": candidate.id,
        "label": candidate.label,
        "kind": candidate.kind,
        "filename": f"{stem}_{candidate.id}.png",
        "rgba": _png_data_url(candidate.rgba),
        "selected": candidate.selected,
        "plan": debug.get("plan"),
        "regions": debug.get("regions", []),
        "operation_results": debug.get("operation_results", []),
        "debug": debug,
    }


def _slice_annotated_preview(image_rgb: np.ndarray, boxes: list[SliceBox]) -> np.ndarray:
    preview = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(preview)
    for box in boxes:
        x, y, w, h = box.bbox
        color = (255, 160, 0, 255)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=3)
        label = f"{box.id}"
        text_box = draw.textbbox((x, y), label)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        draw.rectangle((x, y, x + tw + 8, y + th + 6), fill=(25, 111, 90, 235))
        draw.text((x + 4, y + 3), label, fill=(255, 255, 255, 255))
    return np.asarray(preview, dtype=np.uint8)


def _cached_slice_result(
    image_rgb: np.ndarray,
    image_digest: str,
    *,
    min_area: int,
    padding: int,
) -> SliceResult:
    key = (image_digest, int(min_area), int(padding))
    with _SLICE_CACHE_LOCK:
        cached = _SLICE_CACHE.get(key)
        if cached is not None:
            _SLICE_CACHE.move_to_end(key)
            return cached

    h, w = image_rgb.shape[:2]
    pixels = h * w
    if pixels > _SLICE_WEB_MAX_PIXELS:
        # Web interaction only needs crop rectangles. For very large sheets,
        # full-resolution OKLab masking dominates latency; detect boxes on a
        # bounded preview image and map them back. This protects 4K/6K sheets
        # from multi-second duplicate work while keeping CLI/core slicing exact.
        scale = (_SLICE_WEB_MAX_PIXELS / float(pixels)) ** 0.5
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_min_area = max(1, int(round(min_area * scale * scale)))
        small_padding = max(1, int(round(padding * scale)))
        small_result = slice_image(small, min_area=small_min_area, padding=small_padding)
        boxes: list[SliceBox] = []
        mask = np.zeros((h, w), dtype=np.float32)
        for box in small_result.boxes:
            x, y, bw, bh = box.bbox
            x0 = max(0, int(np.floor(x / scale)) - padding)
            y0 = max(0, int(np.floor(y / scale)) - padding)
            x1 = min(w, int(np.ceil((x + bw) / scale)) + padding)
            y1 = min(h, int(np.ceil((y + bh) / scale)) + padding)
            mask[y0:y1, x0:x1] = 1.0
            boxes.append(SliceBox(id=box.id, bbox=(x0, y0, x1 - x0, y1 - y0), area=int(box.area / max(scale * scale, 1e-6))))
        result = SliceResult(background_color=small_result.background_color, foreground_mask=mask, boxes=boxes)
    else:
        result = slice_image(image_rgb, min_area=min_area, padding=padding)

    with _SLICE_CACHE_LOCK:
        _SLICE_CACHE[key] = result
        _SLICE_CACHE.move_to_end(key)
        while len(_SLICE_CACHE) > _SLICE_CACHE_MAX:
            _SLICE_CACHE.popitem(last=False)
    return result


def _slice_preview_payload(image_rgb: np.ndarray, stem: str, result: SliceResult) -> dict[str, object]:
    annotated = _slice_annotated_preview(image_rgb, result.boxes)
    payload = result.to_dict()
    payload.update(
        {
            "stem": stem,
            "annotated": _png_data_url(annotated),
            "boxes": [box.to_dict() for box in result.boxes],
        }
    )
    return payload


def _slice_crop_payloads(image_rgb: np.ndarray, stem: str, result: SliceResult) -> dict[str, object]:
    crops = []
    kind_counts: dict[str, int] = {}
    for box in result.boxes:
        crop = crop_slice(image_rgb, result.foreground_mask, box, transparent=False)
        prediction = classify_ui_slice(crop, box, image_rgb.shape[:2], result.foreground_mask)
        kind = prediction.kind if prediction.confidence >= 0.6 else "asset"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        name = f"{kind}_{kind_counts[kind]:03d}"
        x, y, w, h = box.bbox
        crops.append(
            {
                "id": name,
                "label": name,
                "kind": kind,
                "confidence": prediction.confidence,
                "filename": f"{name}_rgb.png",
                "rgb": _rgb_png_data_url(crop),
                "bbox": [x, y, w, h],
                "meta": f"{kind} {prediction.confidence:.2f} · {w}x{h}",
                "features": prediction.features,
            }
        )
    return {
        "background": list(result.background_color),
        "count": len(crops),
        "crops": crops,
    }


@app.post("/api/sam-mask")
def sam_mask_endpoint(
    file: Annotated[UploadFile, File()],
    threshold: Annotated[float, Form()] = 0.5,
    refine_iterations: Annotated[int, Form()] = 2,
) -> dict[str, object]:
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    if not 0 <= refine_iterations <= 5:
        raise HTTPException(status_code=400, detail="refine_iterations must be between 0 and 5")

    image = _load_upload_image(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    server_started_at = time.perf_counter()
    try:
        from .probe.comfyui_sam3_mask import ComfyUISAM3MaskClient

        result = ComfyUISAM3MaskClient().mask(
            image_rgb,
            threshold=threshold,
            refine_iterations=refine_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM3 mask failed: {e}") from e

    return {
        "backend": "comfy-sam3",
        "server_elapsed_sec": time.perf_counter() - server_started_at,
        "mask": _mask_png_data_url(result.mask),
        "debug": _json_safe_debug(result.debug),
    }


def _effective_backend(requested_backend: str, result: MatteResponse) -> str:
    auto_route = result.debug.get("auto_route") if isinstance(result.debug, dict) else None
    if requested_backend == "auto" and isinstance(auto_route, dict):
        selected = auto_route.get("selected_backend")
        if isinstance(selected, str) and selected:
            return selected
    return requested_backend


def _route_metadata(result: MatteResponse) -> dict[str, Any]:
    auto_route = result.debug.get("auto_route") if isinstance(result.debug, dict) else None
    if not isinstance(auto_route, dict):
        return {}
    reasons = auto_route.get("reasons")
    if not isinstance(reasons, list):
        reason = auto_route.get("reason")
        reasons = [reason] if isinstance(reason, str) and reason else []
    return {
        "route": auto_route.get("route"),
        "asset_kind": auto_route.get("asset_kind"),
        "parameter_profile": auto_route.get("parameter_profile"),
        "route_confidence": auto_route.get("confidence"),
        "route_reasons": reasons,
    }


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail="pymatting_bg_color must be R,G,B")
    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="pymatting_bg_color must contain integers") from e
    if any(c < 0 or c > 255 for c in rgb):
        raise HTTPException(status_code=400, detail="pymatting_bg_color channels must be between 0 and 255")
    return rgb  # type: ignore[return-value]


def _pymatting_kwargs(
    *,
    pymatting_method: str,
    pymatting_image_space: str,
    pymatting_bg_source: str,
    pymatting_bg_color: str,
    pymatting_bg_threshold: float,
    pymatting_fg_threshold: float,
    pymatting_boundary_band_px: int,
    pymatting_auto_adapt: bool,
    pymatting_cg_maxiter: int,
    pymatting_cg_rtol: float,
) -> dict[str, object]:
    method = pymatting_method.strip().lower()
    if method not in {"cf", "knn", "lbdm", "lkm", "rw", "sm"}:
        raise HTTPException(status_code=400, detail="pymatting_method must be cf, knn, lbdm, lkm, rw, or sm")
    if pymatting_image_space not in {"linear", "sRGB"}:
        raise HTTPException(status_code=400, detail="pymatting_image_space must be linear or sRGB")
    bg_source = pymatting_bg_source.strip().lower()
    if bg_source not in {"auto", "green", "blue", "custom"}:
        raise HTTPException(status_code=400, detail="pymatting_bg_source must be auto, green, blue, or custom")
    if not 0.0 <= pymatting_bg_threshold < pymatting_fg_threshold:
        raise HTTPException(status_code=400, detail="pymatting_bg_threshold must be >= 0 and less than pymatting_fg_threshold")
    if not 0 <= pymatting_boundary_band_px <= 16:
        raise HTTPException(status_code=400, detail="pymatting_boundary_band_px must be between 0 and 16")
    if not 1 <= pymatting_cg_maxiter <= 10000:
        raise HTTPException(status_code=400, detail="pymatting_cg_maxiter must be between 1 and 10000")
    if not 0.0 < pymatting_cg_rtol <= 0.01:
        raise HTTPException(status_code=400, detail="pymatting_cg_rtol must be between 0 and 0.01")
    return {
        "pymatting_method": method,
        "pymatting_image_space": pymatting_image_space,
        "pymatting_bg_source": bg_source,
        "pymatting_bg_color": _parse_rgb_triplet(pymatting_bg_color) if bg_source == "custom" else None,
        "pymatting_bg_threshold": pymatting_bg_threshold,
        "pymatting_fg_threshold": pymatting_fg_threshold,
        "pymatting_boundary_band_px": pymatting_boundary_band_px,
        "pymatting_auto_adapt": bool(pymatting_auto_adapt),
        "pymatting_cg_maxiter": pymatting_cg_maxiter,
        "pymatting_cg_rtol": pymatting_cg_rtol,
    }


@app.post("/api/matte")
def matte_endpoint(
    file: Annotated[UploadFile, File()],
    backend: Annotated[str, Form()] = "auto",
    shadow_enabled: Annotated[bool, Form()] = True,
    pymatting_method: Annotated[str, Form()] = "cf",
    pymatting_image_space: Annotated[str, Form()] = "linear",
    pymatting_bg_source: Annotated[str, Form()] = "auto",
    pymatting_bg_color: Annotated[str, Form()] = "0,200,0",
    pymatting_bg_threshold: Annotated[float, Form()] = 3.5,
    pymatting_fg_threshold: Annotated[float, Form()] = 30.0,
    pymatting_boundary_band_px: Annotated[int, Form()] = 2,
    pymatting_auto_adapt: Annotated[bool, Form()] = True,
    pymatting_cg_maxiter: Annotated[int, Form()] = 1000,
    pymatting_cg_rtol: Annotated[float, Form()] = 1e-6,
) -> Response:
    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {sorted(ALLOWED_BACKENDS)}")

    image = _load_upload_image(file)
    shadow_mode = "on" if shadow_enabled else "off"
    pymatting_params = _pymatting_kwargs(
        pymatting_method=pymatting_method,
        pymatting_image_space=pymatting_image_space,
        pymatting_bg_source=pymatting_bg_source,
        pymatting_bg_color=pymatting_bg_color,
        pymatting_bg_threshold=pymatting_bg_threshold,
        pymatting_fg_threshold=pymatting_fg_threshold,
        pymatting_boundary_band_px=pymatting_boundary_band_px,
        pymatting_auto_adapt=pymatting_auto_adapt,
        pymatting_cg_maxiter=pymatting_cg_maxiter,
        pymatting_cg_rtol=pymatting_cg_rtol,
    )
    try:
        result = matte_image(image, backend=backend, qa=False, shadow_mode=shadow_mode, **pymatting_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e

    effective_backend = _effective_backend(backend, result)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected_rgba = result.rgba
    local_ownership_used = False
    try:
        local_candidate = generate_local_ownership_candidate(
            image_rgb,
            result.rgba,
            result.background_color,
            backend=effective_backend,
            soft_mask=result.debug.get("soft_mask"),
            shadow_mode=shadow_mode,
        ) if effective_backend not in REMOTE_DIRECT_BACKENDS else None
    except Exception:
        local_candidate = None
    if local_candidate is not None:
        selected_rgba = local_candidate.rgba
        local_ownership_used = True

    png = _encode_png(selected_rgba)
    filename = (file.filename or "ermbg").rsplit(".", 1)[0] + "_rgba.png"
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ERMBG-Strategy": result.strategy_name,
            "X-ERMBG-Background": ",".join(str(c) for c in result.background_color),
            "X-ERMBG-Local-Ownership": "1" if local_ownership_used else "0",
        },
    )


@app.post("/api/matte-candidates")
def matte_candidates_endpoint(
    file: Annotated[UploadFile, File()],
    corridorkey_hint_mask: Annotated[UploadFile | None, File()] = None,
    backend: Annotated[str, Form()] = "auto",
    shadow_enabled: Annotated[bool, Form()] = True,
    corridorkey_gamma_space: Annotated[str, Form()] = "sRGB",
    corridorkey_despill_strength: Annotated[float, Form()] = 1.0,
    corridorkey_refiner_strength: Annotated[float, Form()] = 1.0,
    corridorkey_auto_despeckle: Annotated[str, Form()] = "On",
    corridorkey_despeckle_size: Annotated[int, Form()] = 400,
    corridorkey_auto_mask: Annotated[bool, Form()] = False,
    corridorkey_color_protection: Annotated[bool, Form()] = True,
    corridorkey_protection_bg_max: Annotated[float, Form()] = 12.0,
    corridorkey_protection_fg_min: Annotated[float, Form()] = 28.0,
    corridorkey_screen_mode: Annotated[str, Form()] = "auto",
    corridorkey_preset: Annotated[str, Form()] = "auto",
    corridorkey_hard_ui_hint_mode: Annotated[str, Form()] = "bbox_2px",
    pymatting_method: Annotated[str, Form()] = "cf",
    pymatting_image_space: Annotated[str, Form()] = "linear",
    pymatting_bg_source: Annotated[str, Form()] = "auto",
    pymatting_bg_color: Annotated[str, Form()] = "0,200,0",
    pymatting_bg_threshold: Annotated[float, Form()] = 3.5,
    pymatting_fg_threshold: Annotated[float, Form()] = 30.0,
    pymatting_boundary_band_px: Annotated[int, Form()] = 2,
    pymatting_auto_adapt: Annotated[bool, Form()] = True,
    pymatting_cg_maxiter: Annotated[int, Form()] = 1000,
    pymatting_cg_rtol: Annotated[float, Form()] = 1e-6,
) -> dict[str, object]:
    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {sorted(ALLOWED_BACKENDS)}")
    if corridorkey_gamma_space not in {"sRGB", "Linear"}:
        raise HTTPException(status_code=400, detail="corridorkey_gamma_space must be sRGB or Linear")
    if corridorkey_auto_despeckle not in {"On", "Off"}:
        raise HTTPException(status_code=400, detail="corridorkey_auto_despeckle must be On or Off")
    if corridorkey_screen_mode not in {"auto", "green", "blue"}:
        raise HTTPException(status_code=400, detail="corridorkey_screen_mode must be auto, green, or blue")
    if corridorkey_preset not in {"auto", "detail_safe", "spill_safe", "manual"}:
        raise HTTPException(status_code=400, detail="corridorkey_preset must be auto, detail_safe, spill_safe, or manual")
    if corridorkey_hard_ui_hint_mode not in {
        "all_white",
        "bbox_2px",
        "boundary_2px",
        "boundary_2px_shadow_safe",
        "boundary_2px_shadow_safe_edge_floor",
        "translucent_button",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "corridorkey_hard_ui_hint_mode must be all_white, bbox_2px, boundary_2px, "
                "boundary_2px_shadow_safe, boundary_2px_shadow_safe_edge_floor, or translucent_button"
            ),
        )
    if not 0.0 <= corridorkey_despill_strength <= 1.0:
        raise HTTPException(status_code=400, detail="corridorkey_despill_strength must be between 0 and 1")
    if not 0.0 <= corridorkey_refiner_strength <= 4.0:
        raise HTTPException(status_code=400, detail="corridorkey_refiner_strength must be between 0 and 4")
    if not 0 <= corridorkey_despeckle_size <= 4096:
        raise HTTPException(status_code=400, detail="corridorkey_despeckle_size must be between 0 and 4096")
    if corridorkey_protection_fg_min <= corridorkey_protection_bg_max:
        raise HTTPException(status_code=400, detail="corridorkey_protection_fg_min must be greater than corridorkey_protection_bg_max")
    shadow_mode = "on" if shadow_enabled else "off"
    pymatting_params = _pymatting_kwargs(
        pymatting_method=pymatting_method,
        pymatting_image_space=pymatting_image_space,
        pymatting_bg_source=pymatting_bg_source,
        pymatting_bg_color=pymatting_bg_color,
        pymatting_bg_threshold=pymatting_bg_threshold,
        pymatting_fg_threshold=pymatting_fg_threshold,
        pymatting_boundary_band_px=pymatting_boundary_band_px,
        pymatting_auto_adapt=pymatting_auto_adapt,
        pymatting_cg_maxiter=pymatting_cg_maxiter,
        pymatting_cg_rtol=pymatting_cg_rtol,
    )

    image = _load_upload_image(file)
    hint_mask = (
        _load_upload_image(corridorkey_hint_mask)
        if corridorkey_hint_mask is not None and not corridorkey_auto_mask
        else None
    )
    server_started_at = time.perf_counter()
    try:
        result = matte_image(
            image,
            backend=backend,
            qa=False,
            shadow_mode=shadow_mode,
            corridorkey_gamma_space=corridorkey_gamma_space,
            corridorkey_despill_strength=corridorkey_despill_strength,
            corridorkey_refiner_strength=corridorkey_refiner_strength,
            corridorkey_auto_despeckle=corridorkey_auto_despeckle,
            corridorkey_despeckle_size=corridorkey_despeckle_size,
            corridorkey_auto_mask=corridorkey_auto_mask,
            corridorkey_color_protection=corridorkey_color_protection,
            corridorkey_protection_bg_max=corridorkey_protection_bg_max,
            corridorkey_protection_fg_min=corridorkey_protection_fg_min,
            corridorkey_screen_mode=corridorkey_screen_mode,
            corridorkey_preset=corridorkey_preset,
            corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
            corridorkey_hint_mask=hint_mask,
            **pymatting_params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e

    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    effective_backend = _effective_backend(backend, result)
    if effective_backend in REMOTE_DIRECT_BACKENDS:
        if effective_backend == "comfy-corridorkey":
            direct_label = "远端 CorridorKey"
        elif effective_backend == "comfy-pymatting-known-b":
            direct_label = "远端 PyMatting Known-B"
        elif effective_backend == "comfy-rmbg":
            direct_label = "远端 RMBG"
        elif effective_backend == "passthrough":
            direct_label = "远端 Passthrough"
        else:
            direct_label = "PyMatting Known-B"
        candidates = [
            MatteCandidate(
                id="auto",
                label=direct_label,
                rgba=result.rgba,
                selected=True,
                debug={"remote": result.debug},
            )
        ]
    else:
        candidates = generate_matte_candidates(image_rgb, result.rgba, result.background_color)
    try:
        local_candidate = generate_local_ownership_candidate(
            image_rgb,
            result.rgba,
            result.background_color,
            backend=effective_backend,
            soft_mask=result.debug.get("soft_mask"),
            shadow_mode=shadow_mode,
        ) if effective_backend not in REMOTE_DIRECT_BACKENDS else None
    except Exception as e:
        local_candidate = None
        for candidate in candidates:
            candidate.debug["local_ownership_error"] = str(e)
    if local_candidate is not None:
        for candidate in candidates:
            candidate.selected = False
        candidates.append(local_candidate)
    return {
        "strategy": result.strategy_name,
        "background": list(result.background_color),
        "backend": effective_backend,
        "requested_backend": backend,
        **_route_metadata(result),
        "server_elapsed_sec": time.perf_counter() - server_started_at,
        "debug": _json_safe_debug(result.debug),
        "candidates": [_candidate_payload(candidate, stem) for candidate in candidates],
    }


@app.post("/api/slice")
def slice_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 2,
    transparent: Annotated[bool, Form()] = False,
) -> Response:
    image, image_digest = _load_upload_image_with_digest(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    try:
        result = _cached_slice_result(image_rgb, image_digest, min_area=min_area, padding=padding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slicing failed: {e}") from e

    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.slices.json", json.dumps(result.to_dict(), indent=2))
        for box in result.boxes:
            crop = crop_slice(image_rgb, result.foreground_mask, box, transparent=transparent)
            png = _encode_png(crop) if transparent else _encode_rgb_png(crop)
            suffix = "rgba" if transparent else "rgb"
            zf.writestr(f"{stem}_{box.id:03d}_{suffix}.png", png)

    filename = f"{stem}_slices.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ERMBG-Slice-Count": str(len(result.boxes)),
            "X-ERMBG-Background": ",".join(str(c) for c in result.background_color),
        },
    )


@app.post("/api/slice-preview")
def slice_preview_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 2,
) -> dict[str, object]:
    image, image_digest = _load_upload_image_with_digest(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    try:
        result = _cached_slice_result(image_rgb, image_digest, min_area=min_area, padding=padding)
        return _slice_preview_payload(image_rgb, stem, result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slice preview failed: {e}") from e


@app.post("/api/slice-crops")
def slice_crops_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 2,
) -> dict[str, object]:
    image, image_digest = _load_upload_image_with_digest(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    try:
        result = _cached_slice_result(image_rgb, image_digest, min_area=min_area, padding=padding)
        return _slice_crop_payloads(image_rgb, stem, result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slice crops failed: {e}") from e


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Missing file: {path.relative_to(PROJECT_ROOT)}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {path.relative_to(PROJECT_ROOT)}") from e


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _image_url(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = _resolve_project_path(path_value)
    if not path.exists() or path.suffix.lower() not in SERVABLE_IMAGE_SUFFIXES:
        return None
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return f"/eval/game/file/{quote(rel, safe='/')}"


def _candidate_result_items(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _candidate_tools(candidate: dict[str, object], fallback: list[str]) -> list[str]:
    plan = candidate.get("plan")
    if not isinstance(plan, dict):
        return fallback
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return fallback
    tools: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        tool = operation.get("tool")
        if isinstance(tool, str) and tool not in tools:
            tools.append(tool)
    return tools or fallback


def _candidate_regions(candidate_results: list[dict[str, object]]) -> list[dict[str, object]]:
    for candidate in candidate_results:
        regions = candidate.get("regions")
        if isinstance(regions, list):
            return [region for region in regions if isinstance(region, dict)]
    return []


def _game_sample_paths(case_id: str) -> dict[str, str]:
    sample_root = _game_sample_root()
    case_path = sample_root / case_id / "case.json"
    if not case_path.exists():
        for category in ("button", "icon", "character"):
            candidate = sample_root / category / case_id / "case.json"
            if candidate.exists():
                case_path = candidate
                break
    if case_path.exists():
        payload = _load_json(case_path)
        if isinstance(payload, dict):
            paths = {
                screen: path
                for screen in GAME_EVAL_SCREENS
                if isinstance(path := payload.get(screen), str)
            }
            if paths:
                return paths
    return {screen: f"{GAME_SAMPLE_REL.as_posix()}/{case_id}/{screen}.png" for screen in GAME_EVAL_SCREENS}


def _sample_screen_from_path(path_value: object) -> str | None:
    if not isinstance(path_value, str):
        return None
    stem = Path(path_value).stem.lower()
    if stem in {"white", "green", "blue"}:
        return stem
    return None


def _game_sample_ids() -> dict[str, str]:
    manifest = _load_json(_game_sample_manifest())
    if not isinstance(manifest, dict):
        return {}
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return {}
    sample_ids: dict[str, str] = {}
    for index, item in enumerate(cases, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        fallback = f"G{index:02d}"
        sample_id = item.get("sample_id")
        sample_ids[item["id"]] = sample_id if isinstance(sample_id, str) else fallback
    return sample_ids


def _game_eval_samples() -> list[dict[str, object]]:
    if not _game_sample_manifest().exists():
        return []
    manifest = _load_json(_game_sample_manifest())
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list):
        return []
    samples: list[dict[str, object]] = []
    for index, item in enumerate(cases, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        sample_id = item.get("sample_id")
        sample_id = sample_id if isinstance(sample_id, str) else f"G{index:02d}"
        sample_paths = _game_sample_paths(item["id"])
        thumb_url = (
            _image_url(sample_paths.get("green"))
            or _image_url(sample_paths.get("white"))
            or _image_url(sample_paths.get("blue"))
        )
        samples.append(
            {
                "sampleId": sample_id,
                "caseId": item["id"],
                "category": item.get("category", ""),
                "family": item.get("family", ""),
                "screen": item.get("screen", ""),
                "primaryAmbiguity": item.get("primary_ambiguity", ""),
                "thumbnailUrl": thumb_url,
                "defaultSelected": sample_id in FAST_GAME_EVAL_SAMPLE_IDS,
            }
        )
    return samples


def _game_eval_manifest_cases() -> list[dict[str, object]]:
    if not _game_sample_manifest().exists():
        return []
    manifest = _load_json(_game_sample_manifest())
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    return [case for case in cases if isinstance(case, dict)] if isinstance(cases, list) else []


def _game_eval_case_count(
    cases: list[dict[str, object]] | None = None,
    sample_ids: list[str] | None = None,
) -> int:
    selected = cases if cases is not None else _game_eval_manifest_cases()
    if not selected and sample_ids:
        return len(sample_ids)
    if sample_ids:
        wanted = set(sample_ids)
        selected = [case for case in selected if str(case.get("sample_id", "")) in wanted]
    return sum(
        1
        for case in selected
        if isinstance(case.get("input"), str)
        or any(isinstance(case.get(screen), str) for screen in GAME_EVAL_SCREENS)
    )


def _game_report_path(root: Path) -> Path | None:
    path = root / "local_ownership" / "eval_report.json"
    if path.exists():
        return path
    return None


def _game_vlm_root(root: Path) -> Path:
    report_path = _game_report_path(root)
    if report_path is not None:
        return report_path.parent
    return root / "local_ownership"


def _game_eval_partial_summary_paths(root: Path) -> list[Path]:
    local_root = root / "local_ownership"
    if not local_root.exists():
        return []
    return sorted(
        path
        for path in local_root.glob("*/*/summary.json")
        if path.is_file()
    )


def _solid_graphic_summary_path(root: Path) -> Path | None:
    path = root / "summary.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    batch = str(payload.get("batch", ""))
    if root.name.startswith(SOLID_GRAPHIC_EVAL_PREFIX) or batch.startswith(f"out/{SOLID_GRAPHIC_EVAL_PREFIX}"):
        return path
    if payload.get("solid_graphic_prepass") is True or isinstance(payload.get("strategy_pairs"), dict):
        return path
    return None


def _remote_backend_summary_path(root: Path) -> Path | None:
    path = root / "summary.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return None
    for item in runs:
        if isinstance(item, dict) and str(item.get("backend", "")).startswith("comfy-"):
            return path
    return None


def _game_eval_root_has_data(root: Path) -> bool:
    if _game_report_path(root) is not None:
        return True
    if _game_eval_partial_summary_paths(root):
        return True
    if _solid_graphic_summary_path(root) is not None:
        return True
    if _remote_backend_summary_path(root) is not None:
        return True
    return _game_matte_summary_path(root) is not None


def _game_eval_root_is_complete(root: Path) -> bool:
    report_path = _game_report_path(root)
    if report_path is None:
        solid_path = _solid_graphic_summary_path(root)
        if solid_path is None:
            comfy_path = _remote_backend_summary_path(root)
            if comfy_path is None:
                return False
            report = _load_json(comfy_path)
            if not isinstance(report, dict):
                return False
            runs = report.get("runs")
            return isinstance(runs, list) and len(runs) >= _game_eval_expected_case_count()
        report = _load_json(solid_path)
        if not isinstance(report, dict):
            return False
        rows = report.get("rows")
        case_count = report.get("case_count")
        try:
            return isinstance(rows, list) and len(rows) >= int(case_count or _game_eval_expected_case_count())
        except (TypeError, ValueError):
            return False
    report = _load_json(report_path)
    if not isinstance(report, dict):
        return False
    try:
        return int(report.get("case_count", 0)) >= 18
    except (TypeError, ValueError):
        return False


def _game_eval_root_sort_key(root: Path) -> tuple[float, str]:
    try:
        mtime = root.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, root.name)


def _game_eval_data_roots() -> list[Path]:
    out_root = PROJECT_ROOT / "out"
    roots = [
        path
        for path in out_root.iterdir()
        if path.is_dir() and _game_eval_root_has_data(path)
    ] if out_root.exists() else []
    if DEFAULT_GAME_EVAL_ROOT.exists() and DEFAULT_GAME_EVAL_ROOT not in roots and _game_eval_root_has_data(DEFAULT_GAME_EVAL_ROOT):
        roots.append(DEFAULT_GAME_EVAL_ROOT)
    return sorted(set(roots), key=_game_eval_root_sort_key, reverse=True)


def _game_eval_runs(selected_root: Path | None = None) -> list[dict[str, object]]:
    roots = _game_eval_data_roots()
    selected = (selected_root or _default_game_eval_root()).resolve()
    runs: list[dict[str, object]] = []
    for root in roots:
        runs.append(
            {
                "id": root.name,
                "label": root.name,
                "selected": root.resolve() == selected,
                "url": f"/eval/game?run={quote(root.name, safe='')}",
            }
        )
    return runs


def _is_valid_game_eval_run_id(run_id: str) -> bool:
    return bool(run_id) and "/" not in run_id and "\\" not in run_id and not run_id.startswith(".")


def _validate_game_eval_run_id(run_id: str) -> None:
    if not _is_valid_game_eval_run_id(run_id):
        raise HTTPException(status_code=404, detail="Game eval run not found.")


def _game_eval_run_path(run_id: str) -> Path:
    _validate_game_eval_run_id(run_id)
    root = (PROJECT_ROOT / "out" / run_id).resolve()
    if not _is_relative_to(root, (PROJECT_ROOT / "out").resolve()):
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    return root


def _next_game_eval_run_id(prefix: str = LOCAL_OWNERSHIP_EVAL_PREFIX) -> str:
    out_root = PROJECT_ROOT / "out"
    stamp = datetime.now().strftime("%Y%m%d")
    version_re = re.compile(rf"^{re.escape(prefix)}{stamp}_v(\d+)$")
    versions = []
    for path in out_root.glob(f"{prefix}{stamp}_v*"):
        match = version_re.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1
    return f"{prefix}{stamp}_v{version:03d}"


def _game_eval_expected_case_count() -> int:
    total = _game_eval_case_count()
    return total if total > 0 else FALLBACK_GAME_EVAL_EXPECTED_TOTAL


def _game_eval_batch_progress(
    root: Path,
    report_path: Path | None,
    *,
    prefer_report_total: bool = False,
    expected_total: int | None = None,
) -> dict[str, object]:
    total = int(expected_total) if expected_total is not None and expected_total > 0 else _game_eval_expected_case_count()
    completed = 0
    ok = 0
    errors = 0
    if report_path is not None:
        report = _load_json(report_path)
        rows = report.get("rows") if isinstance(report, dict) else None
        if isinstance(rows, list):
            completed = len(rows)
            ok = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "ok")
            errors = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "error")
        runs = report.get("runs") if isinstance(report, dict) else None
        if isinstance(runs, list):
            completed = len(runs)
            errors = sum(1 for row in runs if isinstance(row, dict) and row.get("status") == "error")
            ok = completed - errors
        if isinstance(report, dict):
            try:
                if isinstance(runs, list) and isinstance(report.get("run_count"), int):
                    report_total = int(report["run_count"])
                else:
                    report_total = int(report.get("case_count", 0))
                total = report_total if prefer_report_total and report_total > 0 else max(total, report_total)
            except (TypeError, ValueError):
                pass
    else:
        # The batch script writes per-case summaries immediately, while the
        # final eval_report.json appears only after every selected input
        # finishes. Counting these partial summaries keeps the UI visibly
        # alive during long local matting/ownership runs.
        for summary_path in _game_eval_partial_summary_paths(root):
            summary = _load_json(summary_path)
            if not isinstance(summary, dict):
                continue
            completed += 1
            if summary.get("status", "ok") == "error":
                errors += 1
            else:
                ok += 1
    percent = 0 if total <= 0 else round(min(100.0, completed * 100.0 / total), 1)
    return {
        "completed": completed,
        "total": total,
        "ok": ok,
        "errors": errors,
        "percent": percent,
        "reportPath": str(report_path.relative_to(PROJECT_ROOT)) if report_path is not None else None,
    }


def _game_eval_status_report_path(root: Path) -> Path | None:
    return (
        _game_report_path(root)
        or _solid_graphic_summary_path(root)
        or _remote_backend_summary_path(root)
        or _game_matte_summary_path(root)
    )


def _game_eval_batch_status(run_id: str) -> dict[str, object]:
    root = _game_eval_run_path(run_id)
    report_path = _game_eval_status_report_path(root)
    with _GAME_EVAL_JOBS_LOCK:
        job = _GAME_EVAL_JOBS.get(run_id)
    process = job.get("process") if isinstance(job, dict) else None
    running = isinstance(process, subprocess.Popen) and process.poll() is None
    returncode = process.poll() if isinstance(process, subprocess.Popen) and not running else None
    if running:
        status = "running"
    elif report_path is not None:
        status = "complete"
    elif returncode not in (None, 0):
        status = "error"
    else:
        status = "started" if root.exists() else "unknown"
    expected_total = job.get("expected_total") if isinstance(job, dict) else None
    progress = _game_eval_batch_progress(
        root,
        report_path,
        prefer_report_total=not running,
        expected_total=int(expected_total) if isinstance(expected_total, int) else None,
    )
    return {
        "runId": run_id,
        "status": status,
        "returnCode": returncode,
        "url": f"/eval/game?run={quote(run_id, safe='')}",
        "statusUrl": f"/eval/game/run/{quote(run_id, safe='')}/status",
        "hasReport": report_path is not None,
        "progress": progress,
    }


def _selected_game_eval_sample_ids(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    raw = payload.get("sample_ids")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="sample_ids must be a list.")
    sample_ids = [str(item).strip() for item in raw if str(item).strip()]
    known = {str(item["sampleId"]) for item in _game_eval_samples()}
    if known:
        invalid = sorted(set(sample_ids) - known)
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unknown sample_id: {', '.join(invalid)}")
    elif any(not re.fullmatch(r"[A-Z]\d{3}|G\d{2}", sample_id) for sample_id in sample_ids):
        raise HTTPException(status_code=400, detail="sample_ids must look like B001.")
    deduped: list[str] = []
    for sample_id in sample_ids:
        if sample_id not in deduped:
            deduped.append(sample_id)
    if raw and not deduped:
        raise HTTPException(status_code=400, detail="Select at least one sample.")
    return deduped


def _selected_game_eval_test_path(payload: dict[str, Any] | None) -> str:
    if not payload:
        return DEFAULT_GAME_EVAL_TEST_PATH
    raw = payload.get("test_path", payload.get("path", payload.get("backend")))
    if raw is None:
        return DEFAULT_GAME_EVAL_TEST_PATH
    selected = str(raw).strip().lower()
    backend_to_path = {
        str(config["backend"]): path_key
        for path_key, config in GAME_EVAL_TEST_PATHS.items()
    }
    selected = backend_to_path.get(selected, selected)
    if selected not in GAME_EVAL_TEST_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown test_path: {raw}")
    return selected


def _start_game_eval_batch(
    sample_ids: list[str] | None = None,
    test_path: str = DEFAULT_GAME_EVAL_TEST_PATH,
) -> dict[str, object]:
    selected_sample_ids = list(sample_ids or [])
    path_config = GAME_EVAL_TEST_PATHS.get(test_path, GAME_EVAL_TEST_PATHS[DEFAULT_GAME_EVAL_TEST_PATH])
    backend = str(path_config["backend"])
    run_id = _next_game_eval_run_id(str(path_config["prefix"]))
    out_dir = PROJECT_ROOT / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "web_batch.log"
    script_path = PROJECT_ROOT / "scripts" / "run_corridorkey_game_eval.py"
    command = [
        sys.executable,
        str(script_path),
        "--out-dir",
        str(out_dir),
        "--backend",
        backend,
    ]
    if selected_sample_ids:
        command.extend(["--sample-id", ",".join(selected_sample_ids)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "run_id": run_id,
        "command": command,
        "log": str(log_path.relative_to(PROJECT_ROOT)),
        "pid": process.pid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "test_path": test_path,
        "test_path_label": path_config["label"],
        "sample_ids": selected_sample_ids,
    }
    (out_dir / "web_launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False), encoding="utf-8")
    with _GAME_EVAL_JOBS_LOCK:
        _GAME_EVAL_JOBS[run_id] = {
            "process": process,
            "log": log_path,
            "backend": backend,
            "test_path": test_path,
            "sample_ids": selected_sample_ids,
            "expected_total": (
                _game_eval_case_count(sample_ids=selected_sample_ids)
                if selected_sample_ids
                else _game_eval_expected_case_count()
            ),
        }
    return _game_eval_batch_status(run_id)


def _default_game_eval_root() -> Path:
    roots = _game_eval_data_roots()
    if roots:
        return roots[0]
    return DEFAULT_GAME_EVAL_ROOT


def _game_eval_root(run: str | None = None) -> Path:
    if not run:
        root = _default_game_eval_root()
        if root.is_dir() and _game_eval_root_has_data(root):
            return root
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    root = _game_eval_run_path(run)
    if not root.is_dir() or not _game_eval_root_has_data(root):
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    return root


def _game_matte_summary_path(root: Path) -> Path | None:
    matte_root = root / "matte"
    candidates = [
        matte_root / "summary_shadow_rerun.json",
        matte_root / "summary.json",
    ]
    candidates.extend(sorted(matte_root.glob("summary*.json")))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _game_report_rows(root: Path = DEFAULT_GAME_EVAL_ROOT) -> list[dict[str, object]]:
    report_path = _game_report_path(root)
    if report_path is None:
        raise HTTPException(status_code=404, detail="Game eval report not found.")
    report = _load_json(report_path)
    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="Game eval report must be a JSON object.")

    rows = report.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=500, detail="Game eval report is missing rows.")
    return [row for row in rows if isinstance(row, dict)]


def _game_case_out_dir(row: dict[str, object], root: Path = DEFAULT_GAME_EVAL_ROOT) -> Path:
    case_id = str(row.get("case_id", "unknown"))
    out_dir_value = row.get("out_dir")
    if isinstance(out_dir_value, str):
        return _resolve_project_path(out_dir_value)
    screen = row.get("sample_screen")
    if isinstance(screen, str):
        return _game_vlm_root(root) / case_id / screen
    return _game_vlm_root(root) / case_id


def _case_matte_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    for value in (summary.get("rgba"), summary.get("matte"), summary.get("output")):
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in (f"{sample_screen}_rgba.png", "rgba.png"):
        url = _image_url(out_dir / name)
        if url:
            return url
    matches = sorted(out_dir.glob("*_rgba.png"))
    return _image_url(matches[0]) if matches else None


def _case_alpha_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    for value in (summary.get("alpha"), summary.get("mask")):
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in (f"{sample_screen}_alpha.png", "alpha.png", "mask.png"):
        url = _image_url(out_dir / name)
        if url:
            return url
    matches = sorted(out_dir.glob("*_alpha.png"))
    return _image_url(matches[0]) if matches else None


def _case_artifact_url(
    out_dir: Path,
    sample_screen: str,
    summary: dict[str, object],
    summary_keys: tuple[str, ...],
    filenames: tuple[str, ...],
) -> str | None:
    for key in summary_keys:
        value = summary.get(key)
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in filenames:
        url = _image_url(out_dir / name)
        if url:
            return url

    stemmed_suffixes = tuple(name.removeprefix(f"{sample_screen}_") for name in filenames)
    for suffix in stemmed_suffixes:
        matches = sorted(out_dir.glob(f"*_{suffix}"))
        if matches:
            url = _image_url(matches[0])
            if url:
                return url
    return None


def _case_mask_hint_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    return _case_artifact_url(
        out_dir,
        sample_screen,
        summary,
        ("hint", "mask_hint", "corridorkey_hint"),
        (f"{sample_screen}_corridorkey_hint.png", "corridorkey_hint.png"),
    )


def _case_corridorkey_raw_alpha_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    return _case_artifact_url(
        out_dir,
        sample_screen,
        summary,
        ("raw_alpha", "corridorkey_raw_alpha"),
        (f"{sample_screen}_corridorkey_raw_alpha.png", "corridorkey_raw_alpha.png"),
    )


def _case_foreground_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    return _case_artifact_url(
        out_dir,
        sample_screen,
        summary,
        ("foreground", "corridorkey_foreground"),
        (f"{sample_screen}_foreground.png", "foreground.png"),
    )


def _sibling_image_url(path_value: object, filename: str) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = _resolve_project_path(path_value)
    return _image_url(path.with_name(filename))


def _game_region_url(root: Path, case_id: str, sample_screen: str | None = None) -> str:
    base = f"/eval/game/regions/{quote(case_id, safe='')}"
    params: list[str] = []
    if sample_screen:
        params.append(f"screen={quote(sample_screen, safe='')}")
    if root.resolve() == DEFAULT_GAME_EVAL_ROOT.resolve():
        return f"{base}?{'&'.join(params)}" if params else base
    params.append(f"run={quote(root.name, safe='')}")
    return f"{base}?{'&'.join(params)}"


def _game_eval_data_from_matte_summary(root: Path) -> dict[str, object]:
    summary_path = _game_matte_summary_path(root)
    if summary_path is None:
        raise HTTPException(status_code=404, detail="Game eval summary not found.")
    payload = _load_json(summary_path)
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Game matte summary must be a JSON list.")

    sample_ids = _game_sample_ids()
    cases: list[dict[str, object]] = []
    for case_index, row in enumerate(item for item in payload if isinstance(item, dict)):
        case_id = str(row.get("id") or row.get("image") or row.get("case_id") or f"case_{case_index + 1}")
        sample_id = sample_ids.get(case_id, f"G{case_index + 1:02d}")
        out_dir = _resolve_project_path(str(row.get("out_dir"))) if isinstance(row.get("out_dir"), str) else root / "matte" / case_id
        input_path = row.get("input")
        active_screen = _sample_screen_from_path(input_path) or "green"
        sample_paths = _game_sample_paths(case_id)
        shadow_detected = bool(row.get("shadow_detected", False))
        shadow_pixels = int(row.get("shadow_pixels", 0) or 0)
        strategy = str(row.get("strategy", ""))
        matte_url = _case_matte_url(out_dir, active_screen, row)
        alpha_url = _case_alpha_url(out_dir, active_screen, row)
        mask_hint_url = _case_mask_hint_url(out_dir, active_screen, row)
        raw_alpha_url = _case_corridorkey_raw_alpha_url(out_dir, active_screen, row)
        foreground_url = _case_foreground_url(out_dir, active_screen, row)
        candidate = {
            "id": "matte",
            "label": "matte result",
            "selected": True,
            "tools": [strategy] if strategy else [],
            "reason": f"shadow={shadow_detected}, pixels={shadow_pixels}",
            "url": matte_url,
        }

        for sample_screen, sample_path in sample_paths.items():
            is_active_run = sample_screen == active_screen
            sample_code = f"{sample_id}-{sample_screen[:1].upper()}"
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": sample_code,
                    "sampleScreen": sample_screen,
                    "runStatus": "ran" if is_active_run else "not-run",
                    "category": "matte-rerun" if is_active_run else "",
                    "verdict": strategy if is_active_run else "not-run",
                    "expectedHit": shadow_detected if is_active_run else False,
                    "regionCount": shadow_pixels if is_active_run else 0,
                    "counts": {"shadow_pixels": shadow_pixels} if is_active_run else {},
                    "selectedTools": [strategy] if is_active_run and strategy else [],
                    "primaryAmbiguity": f"shadow mean={float(row.get('shadow_mean_alpha', 0.0) or 0.0):.3f}, p95={float(row.get('shadow_p95_alpha', 0.0) or 0.0):.3f}" if is_active_run else "",
                    "originalUrl": _image_url(sample_path),
                    "regionsUrl": None,
                    "alphaUrl": alpha_url if is_active_run else None,
                    "matteUrl": matte_url if is_active_run else None,
                    "maskHintUrl": mask_hint_url if is_active_run else None,
                    "corridorkeyRawAlphaUrl": raw_alpha_url if is_active_run else None,
                    "corridorkeyForegroundUrl": foreground_url if is_active_run else None,
                    "candidates": [candidate] if is_active_run and matte_url else [],
                }
            )

    return {
        "runId": root.name,
        "model": "matte rerun",
        "success": f"{sum(1 for item in cases if item.get('runStatus') == 'ran')}/{len(payload)}",
        "expectedHit": f"{sum(1 for item in payload if isinstance(item, dict) and item.get('shadow_detected'))}/{len(payload)}",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str((root / "matte").relative_to(PROJECT_ROOT)),
        "vlmRoot": "",
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _game_eval_data_from_partial_summaries(root: Path) -> dict[str, object]:
    summary_paths = _game_eval_partial_summary_paths(root)
    if not summary_paths:
        raise HTTPException(status_code=404, detail="Game eval summary not found.")

    sample_ids = _game_sample_ids()
    cases: list[dict[str, object]] = []
    ok_count = 0
    expected_role_hit_count = 0
    expected_role_required_count = 0
    for summary_path in summary_paths:
        row = _load_json(summary_path)
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or summary_path.parents[1].name)
        sample_screen = str(row.get("sample_screen") or summary_path.parent.name)
        sample_id = str(row.get("sample_id") or sample_ids.get(case_id, case_id))
        sample_paths = _game_sample_paths(case_id)
        sample_path = sample_paths.get(sample_screen, "")
        status = str(row.get("status", "ok"))
        is_ok = status != "error"
        ok_count += 1 if is_ok else 0
        if row.get("expected_role_hit") is not None:
            expected_role_required_count += 1
            expected_role_hit_count += 1 if row.get("expected_role_hit") is True else 0

        top_roles = [role for role in row.get("top_roles", []) if isinstance(role, str)]
        role_counts = row.get("role_counts") if isinstance(row.get("role_counts"), dict) else {}
        role_summary = ", ".join(
            f"{role}={count}"
            for role, count in sorted(role_counts.items())
            if isinstance(role, str)
        )
        preview_path = row.get("protected_rgba") or row.get("rgba")
        alpha_url = _sibling_image_url(preview_path, "alpha.png")
        artifact_dir = _resolve_project_path(preview_path).parent if isinstance(preview_path, str) and preview_path else summary_path.parent
        mask_hint_url = _case_mask_hint_url(artifact_dir, sample_screen, row)
        raw_alpha_url = _case_corridorkey_raw_alpha_url(artifact_dir, sample_screen, row)
        foreground_url = _case_foreground_url(artifact_dir, sample_screen, row)
        candidates = []
        if is_ok and preview_path:
            candidates.append(
                {
                    "id": "local_ownership",
                    "label": "local ownership",
                    "selected": True,
                    "tools": top_roles[:8],
                    "reason": role_summary or "Local ownership ranking.",
                    "url": _image_url(preview_path),
                }
            )

        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": str(row.get("sample_code") or f"{sample_id}-{sample_screen[:1].upper()}"),
                "sampleScreen": sample_screen,
                "runStatus": "ran" if is_ok else "error",
                "category": row.get("category", ""),
                "verdict": row.get("diagnosis_verdict", status),
                "expectedHit": bool(row.get("expected_role_hit")) if is_ok else False,
                "expectedAnyHit": bool(row.get("expected_role_hit")) if is_ok else False,
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "shadowPolicyRequired": False,
                "shadowPolicyHit": None,
                "shadowCandidateCount": 0,
                "regionCount": row.get("region_count", 0) if is_ok else 0,
                "counts": row.get("role_mask_pixels", {}) if is_ok else {},
                "selectedTools": top_roles if is_ok else [],
                "primaryAmbiguity": row.get("expected_role", row.get("error", "")),
                "originalUrl": _image_url(sample_path),
                "regionsUrl": _game_region_url(root, case_id, sample_screen) if is_ok else None,
                "alphaUrl": alpha_url if is_ok else None,
                "matteUrl": _image_url(preview_path) if is_ok else None,
                "maskHintUrl": mask_hint_url if is_ok else None,
                "corridorkeyRawAlphaUrl": raw_alpha_url if is_ok else None,
                "corridorkeyForegroundUrl": foreground_url if is_ok else None,
                "candidates": candidates,
            }
        )

    progress = _game_eval_batch_progress(root, None)
    return {
        "runId": root.name,
        "model": "local ownership (running)",
        "success": f"{ok_count}/{progress['total']}",
        "expectedHit": f"{expected_role_hit_count}/{expected_role_required_count}",
        "expectedAnyHit": f"{expected_role_hit_count}/{expected_role_required_count}",
        "harmfulTools": f"0/{len(cases)}",
        "shadowPolicyHit": "0/0",
        "sampleRows": len(cases),
        "reportPath": None,
        "matteRoot": str((root / "matte").relative_to(PROJECT_ROOT)),
        "vlmRoot": str((root / "local_ownership").relative_to(PROJECT_ROOT)),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _solid_graphic_artifact_url(branch: dict[str, object], field: str) -> str | None:
    value = branch.get(field)
    if not isinstance(value, str):
        return None
    path = Path(value)
    if not path.is_absolute() and len(path.parts) == 1 and isinstance(branch.get("dir"), str):
        path = Path(str(branch["dir"])) / path
    return _image_url(path)


def _solid_graphic_diff_url(root: Path, row: dict[str, object]) -> str | None:
    for branch_name in ("new", "old"):
        branch = row.get(branch_name)
        if isinstance(branch, dict) and isinstance(branch.get("dir"), str):
            candidate = _resolve_project_path(str(branch["dir"])).parent / "alpha_abs_diff.png"
            if candidate.exists():
                return _image_url(candidate)
    sample_id = str(row.get("sample_id", ""))
    case_id = str(row.get("case_id", ""))
    screen = str(row.get("screen", ""))
    if sample_id and case_id and screen:
        return _image_url(root / f"{sample_id}_{case_id}_{screen}" / "alpha_abs_diff.png")
    return None


def _solid_graphic_candidate_reason(branch: dict[str, object]) -> str:
    parts: list[str] = []
    if isinstance(branch.get("solid_confidence"), (int, float)):
        parts.append(f"confidence={float(branch['solid_confidence']):.3f}")
    if isinstance(branch.get("alpha_mean"), (int, float)):
        parts.append(f"alpha_mean={float(branch['alpha_mean']):.3f}")
    if isinstance(branch.get("alpha_soft_fraction"), (int, float)):
        parts.append(f"soft={float(branch['alpha_soft_fraction']):.3f}")
    if isinstance(branch.get("elapsed_sec"), (int, float)):
        parts.append(f"{float(branch['elapsed_sec']):.2f}s")
    return ", ".join(parts)


def _solid_graphic_diff_reason(diff: dict[str, object]) -> str:
    parts: list[str] = []
    labels = (
        ("mean_abs", "mean"),
        ("p95_abs", "p95"),
        ("max_abs", "max"),
        ("gt_05_fraction", ">0.05"),
        ("gt_25_fraction", ">0.25"),
    )
    for key, label in labels:
        value = diff.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}={float(value):.3f}")
    return ", ".join(parts)


def _game_eval_data_from_solid_graphic_summary(root: Path, summary_path: Path) -> dict[str, object]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Solid graphic summary must be a JSON object.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=500, detail="Solid graphic summary is missing rows.")

    cases: list[dict[str, object]] = []
    ok_count = 0
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "ok"))
        is_ok = status != "error"
        ok_count += 1 if is_ok else 0
        case_id = str(item.get("case_id") or f"case_{index:02d}")
        sample_id = str(item.get("sample_id") or f"G{index:02d}")
        screen = str(item.get("screen") or _sample_screen_from_path(item.get("input")) or "green")
        sample_code = f"{sample_id}-{screen[:1].upper()}"
        new_branch = item.get("new") if isinstance(item.get("new"), dict) else {}
        old_branch = item.get("old") if isinstance(item.get("old"), dict) else {}
        if not new_branch and isinstance(item.get("outputs"), dict):
            outputs = item["outputs"]
            alpha_stats = item.get("alpha") if isinstance(item.get("alpha"), dict) else {}
            solid_graphic = item.get("solid_graphic") if isinstance(item.get("solid_graphic"), dict) else {}
            new_branch = {
                "strategy": item.get("strategy", "solid_graphic"),
                "solid_confidence": solid_graphic.get("confidence"),
                "alpha_mean": alpha_stats.get("mean"),
                "alpha_soft_fraction": alpha_stats.get("soft_fraction"),
                "elapsed_sec": item.get("elapsed_sec"),
                "dir": outputs.get("case_dir"),
                "rgba": outputs.get("rgba"),
                "ownership_counts": item.get("ownership_counts", {}),
            }
        diff = item.get("alpha_diff") if isinstance(item.get("alpha_diff"), dict) else {}
        new_strategy = str(new_branch.get("strategy", "solid_graphic"))
        old_strategy = str(old_branch.get("strategy", "fallback"))

        candidates: list[dict[str, object]] = []
        new_url = _solid_graphic_artifact_url(new_branch, "rgba")
        new_alpha_url = _solid_graphic_artifact_url(new_branch, "alpha")
        new_foreground_url = _solid_graphic_artifact_url(new_branch, "foreground")
        new_mask_hint_url = _solid_graphic_artifact_url(new_branch, "hint")
        new_raw_alpha_url = _solid_graphic_artifact_url(new_branch, "raw_alpha")
        if new_url:
            candidates.append(
                {
                    "id": "new_solid_graphic" if old_branch else "solid_graphic",
                    "label": f"new {new_strategy}" if old_branch else new_strategy,
                    "selected": True,
                    "tools": [new_strategy],
                    "reason": _solid_graphic_candidate_reason(new_branch),
                    "url": new_url,
                }
            )
        old_url = _solid_graphic_artifact_url(old_branch, "rgba")
        if old_url:
            candidates.append(
                {
                    "id": "old_fallback",
                    "label": "old fallback",
                    "selected": False,
                    "tools": [old_strategy],
                    "reason": _solid_graphic_candidate_reason(old_branch),
                    "url": old_url,
                }
            )
        diff_url = _solid_graphic_diff_url(root, item)
        if diff_url:
            candidates.append(
                {
                    "id": "alpha_abs_diff",
                    "label": "alpha diff",
                    "selected": False,
                    "tools": ["alpha_abs_diff"],
                    "reason": _solid_graphic_diff_reason(diff),
                    "url": diff_url,
                }
            )

        ownership_counts = new_branch.get("ownership_counts")
        if not isinstance(ownership_counts, dict):
            ownership_counts = {}
        verdict = f"{new_strategy} vs {old_strategy}" if old_branch else new_strategy
        primary = str(item.get("primary_ambiguity", ""))
        diff_reason = _solid_graphic_diff_reason(diff)
        if diff_reason:
            primary = f"{primary} · diff {diff_reason}" if primary else f"diff {diff_reason}"

        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": sample_code,
                "sampleScreen": screen,
                "runStatus": "ran" if is_ok else "error",
                "category": "solid-graphic-compare",
                "verdict": verdict if is_ok else status,
                "expectedHit": is_ok,
                "expectedAnyHit": is_ok,
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "shadowPolicyRequired": False,
                "shadowPolicyHit": None,
                "shadowCandidateCount": 0,
                "regionCount": sum(int(value) for value in ownership_counts.values() if isinstance(value, int)),
                "counts": ownership_counts,
                "selectedTools": [new_strategy] if is_ok else [],
                "primaryAmbiguity": primary,
                "originalUrl": _image_url(item.get("input")),
                "regionsUrl": None,
                "alphaUrl": new_alpha_url,
                "matteUrl": new_url,
                "maskHintUrl": new_mask_hint_url,
                "corridorkeyRawAlphaUrl": new_raw_alpha_url,
                "corridorkeyForegroundUrl": new_foreground_url,
                "candidates": candidates if is_ok else [],
            }
        )

    case_count = int(payload.get("case_count", len(rows)) or len(rows))
    progress = _game_eval_batch_progress(root, summary_path, prefer_report_total=True)
    return {
        "runId": root.name,
        "model": "solid graphic comparison",
        "success": f"{ok_count}/{case_count}",
        "expectedHit": "n/a",
        "expectedAnyHit": "n/a",
        "harmfulTools": "0/0",
        "shadowPolicyHit": "0/0",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str(root.relative_to(PROJECT_ROOT)),
        "vlmRoot": GAME_SAMPLE_REL.as_posix(),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _case_id_from_comfy_run(item: dict[str, object], index: int) -> tuple[str, str, str]:
    metadata = item.get("case_metadata") if isinstance(item.get("case_metadata"), dict) else {}
    input_path = item.get("input")
    item_screen = item.get("sample_screen")
    screen = str(item_screen) if isinstance(item_screen, str) and item_screen else (_sample_screen_from_path(input_path) or "green")
    sample_id = str(metadata.get("sample_id") or "")
    case_id = str(metadata.get("id") or "")
    case_label = str(item.get("case") or "")
    if not sample_id and case_label:
        parts = case_label.split("_")
        if parts and re.fullmatch(r"[A-Z]\d{3}|G\d{2}", parts[0]):
            sample_id = parts[0]
    if not case_id and isinstance(input_path, str):
        try:
            case_id = Path(input_path).parent.name
        except Exception:
            case_id = ""
    if not sample_id:
        sample_id = f"S{index:03d}"
    if not case_id:
        case_id = case_label or f"case_{index:02d}"
    return sample_id, case_id, screen


def _game_eval_data_from_comfy_ermbg_summary(root: Path, summary_path: Path) -> dict[str, object]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Comfy ERMBG summary must be a JSON object.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise HTTPException(status_code=500, detail="Comfy ERMBG summary is missing runs.")

    cases: list[dict[str, object]] = []
    ok_count = 0
    for index, item in enumerate((run for run in runs if isinstance(run, dict)), start=1):
        status = str(item.get("status", "ok"))
        is_ok = status != "error"
        ok_count += 1 if is_ok else 0
        sample_id, case_id, screen = _case_id_from_comfy_run(item, index)
        metadata = item.get("case_metadata") if isinstance(item.get("case_metadata"), dict) else {}
        outputs = item.get("outputs") if isinstance(item.get("outputs"), dict) else {}
        metrics = item.get("quality_metrics") if isinstance(item.get("quality_metrics"), dict) else {}
        remote_debug = item.get("remote_debug") if isinstance(item.get("remote_debug"), dict) else {}
        timings = remote_debug.get("timings") if isinstance(remote_debug.get("timings"), dict) else {}
        strategy = str(item.get("backend") or "auto")
        elapsed = item.get("elapsed_sec_client")
        alpha_mean = metrics.get("alpha_mean")
        alpha_pixels = metrics.get("alpha_nonzero_pixels")
        reason_parts = []
        if isinstance(elapsed, (int, float)):
            reason_parts.append(f"{float(elapsed):.1f}s client")
        if isinstance(timings.get("total_sec"), (int, float)):
            reason_parts.append(f"{float(timings['total_sec']):.1f}s server")
        if isinstance(alpha_mean, (int, float)):
            reason_parts.append(f"alpha_mean={float(alpha_mean):.3f}")
        if isinstance(alpha_pixels, int):
            reason_parts.append(f"alpha_px={alpha_pixels}")
        candidate_url = _image_url(outputs.get("rgba"))
        alpha_url = _image_url(outputs.get("alpha")) or _sibling_image_url(outputs.get("rgba"), "alpha.png")
        mask_hint_url = _image_url(outputs.get("hint")) or _image_url(outputs.get("corridorkey_hint")) or _sibling_image_url(outputs.get("rgba"), "corridorkey_hint.png")
        raw_alpha_url = _image_url(outputs.get("raw_alpha")) or _image_url(outputs.get("corridorkey_raw_alpha")) or _sibling_image_url(outputs.get("rgba"), "corridorkey_raw_alpha.png")
        foreground_url = _image_url(outputs.get("foreground")) or _sibling_image_url(outputs.get("rgba"), "foreground.png")
        candidates = []
        if is_ok and candidate_url:
            candidates.append(
                {
                    "id": strategy.replace("-", "_"),
                    "label": strategy,
                    "selected": True,
                    "tools": [strategy],
                    "reason": ", ".join(reason_parts),
                    "url": candidate_url,
                }
            )
        sample_paths = _game_sample_paths(case_id)
        sample_path = sample_paths.get(screen) or str(item.get("input", ""))
        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": f"{sample_id}-{screen[:1].upper()}",
                "sampleScreen": screen,
                "runStatus": "ran" if is_ok else "error",
                "category": metadata.get("category", strategy),
                "verdict": strategy if is_ok else status,
                "expectedHit": is_ok,
                "expectedAnyHit": is_ok,
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "shadowPolicyRequired": False,
                "shadowPolicyHit": None,
                "shadowCandidateCount": 0,
                "regionCount": int(alpha_pixels) if isinstance(alpha_pixels, int) else 0,
                "counts": {"alpha_nonzero_pixels": alpha_pixels, "alpha_mean": alpha_mean},
                "selectedTools": [strategy] if is_ok else [],
                "primaryAmbiguity": metadata.get("primary_ambiguity", ""),
                "originalUrl": _image_url(sample_path),
                "regionsUrl": None,
                "alphaUrl": alpha_url if is_ok else None,
                "matteUrl": candidate_url if is_ok else None,
                "maskHintUrl": mask_hint_url if is_ok else None,
                "corridorkeyRawAlphaUrl": raw_alpha_url if is_ok else None,
                "corridorkeyForegroundUrl": foreground_url if is_ok else None,
                "candidates": candidates,
            }
        )

    progress = _game_eval_batch_progress(root, summary_path, prefer_report_total=True)
    return {
        "runId": root.name,
        "model": f"{str(payload.get('backend') or (runs[0].get('backend') if runs and isinstance(runs[0], dict) else 'auto'))} remote",
        "success": f"{ok_count}/{len(runs)}",
        "expectedHit": f"{ok_count}/{len(runs)}",
        "expectedAnyHit": f"{ok_count}/{len(runs)}",
        "harmfulTools": f"0/{len(runs)}",
        "shadowPolicyHit": "0/0",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str(root.relative_to(PROJECT_ROOT)),
        "vlmRoot": GAME_SAMPLE_REL.as_posix(),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _game_eval_data(root: Path = DEFAULT_GAME_EVAL_ROOT) -> dict[str, object]:
    report_path = _game_report_path(root)
    if report_path is None:
        if _game_eval_partial_summary_paths(root):
            return _game_eval_data_from_partial_summaries(root)
        solid_path = _solid_graphic_summary_path(root)
        if solid_path is not None:
            return _game_eval_data_from_solid_graphic_summary(root, solid_path)
        comfy_path = _remote_backend_summary_path(root)
        if comfy_path is not None:
            return _game_eval_data_from_comfy_ermbg_summary(root, comfy_path)
        data = _game_eval_data_from_matte_summary(root)
        data["runs"] = _game_eval_runs(root)
        data["selectedRun"] = root.name
        return data

    report = _load_json(report_path)
    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="Game eval report must be a JSON object.")

    rows = _game_report_rows(root)
    sample_ids = _game_sample_ids()

    cases: list[dict[str, object]] = []
    for case_index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id", "unknown"))
        sample_id = sample_ids.get(case_id, f"G{case_index:02d}")
        out_dir = _game_case_out_dir(row, root)
        summary_path = out_dir / "summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        if not isinstance(summary, dict):
            summary = {}

        fallback_tools = [tool for tool in row.get("selected_tools", []) if isinstance(tool, str)]
        top_roles = [role for role in row.get("top_roles", []) if isinstance(role, str)]
        if not fallback_tools and top_roles:
            fallback_tools = top_roles
        candidate_results_path = out_dir / "candidate_results.json"
        candidate_results = _candidate_result_items(candidate_results_path)
        candidate_paths = [path for path in summary.get("candidate_paths", []) if isinstance(path, str)]
        if not candidate_paths:
            selected_ids = [plan_id for plan_id in row.get("selected_plan_ids", []) if isinstance(plan_id, str)]
            candidate_paths = [str(out_dir / "candidates" / f"{plan_id}.png") for plan_id in selected_ids]
        if not candidate_paths:
            candidate_paths = [str(path) for path in sorted((out_dir / "candidates").glob("*.png"))]

        candidates: list[dict[str, object]] = []
        result_by_path: dict[str, dict[str, object]] = {}
        for candidate in candidate_results:
            candidate_path = candidate.get("path")
            if isinstance(candidate_path, str):
                result_by_path[_resolve_project_path(candidate_path).as_posix()] = candidate

        for index, candidate_path in enumerate(candidate_paths):
            resolved_candidate_path = _resolve_project_path(candidate_path)
            candidate = result_by_path.get(resolved_candidate_path.as_posix(), {})
            plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
            candidate_id = candidate.get("id") or (plan.get("id") if isinstance(plan, dict) else None) or resolved_candidate_path.stem
            label = candidate.get("label") or (plan.get("label") if isinstance(plan, dict) else None) or str(candidate_id)
            candidates.append(
                {
                    "id": str(candidate_id),
                    "label": str(label),
                    "selected": bool(candidate.get("selected", index == 0)),
                    "tools": _candidate_tools(candidate, fallback_tools),
                    "reason": str((plan.get("reason") if isinstance(plan, dict) else "") or row.get("selected_reason", "")),
                    "url": _image_url(resolved_candidate_path),
                }
            )
        if not candidates and isinstance(row.get("ownership"), list):
            role_counts = row.get("role_counts") if isinstance(row.get("role_counts"), dict) else {}
            role_summary = ", ".join(
                f"{role}={count}"
                for role, count in sorted(role_counts.items())
                if isinstance(role, str)
            )
            candidates.append(
                {
                    "id": "local_ownership",
                    "label": "local ownership",
                    "selected": True,
                    "tools": top_roles[:8],
                    "reason": role_summary or "Local local ownership ranking.",
                    "url": _image_url(row.get("protected_rgba") or row.get("rgba")),
                }
            )

        sample_paths = _game_sample_paths(case_id)
        row_screen = row.get("sample_screen")
        active_screen = (
            row_screen
            if isinstance(row_screen, str) and row_screen in sample_paths
            else _sample_screen_from_path(summary.get("input")) or "green"
        )
        screens = [active_screen] if isinstance(row_screen, str) else list(sample_paths)
        for sample_screen in screens:
            sample_path = sample_paths.get(sample_screen, str(summary.get("input", "")))
            is_active_run = sample_screen == active_screen
            sample_code = f"{sample_id}-{sample_screen[:1].upper()}"
            alpha_url = _case_alpha_url(out_dir, active_screen, summary) if is_active_run else None
            matte_url = (
                _image_url(summary.get("rgba") or row.get("protected_rgba") or row.get("rgba") or root / "matte" / case_id / "rgba.png")
                if is_active_run
                else None
            )
            mask_hint_url = _case_mask_hint_url(out_dir, active_screen, summary) if is_active_run else None
            raw_alpha_url = _case_corridorkey_raw_alpha_url(out_dir, active_screen, summary) if is_active_run else None
            foreground_url = _case_foreground_url(out_dir, active_screen, summary) if is_active_run else None
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": sample_code,
                    "sampleScreen": sample_screen,
                    "runStatus": "ran" if is_active_run else "not-run",
                    "category": row.get("category", ""),
                    "verdict": row.get("diagnosis_verdict", "") if is_active_run else "not-run",
                    "expectedHit": bool(row.get("expected_hit", row.get("expected_role_hit"))) if is_active_run else False,
                    "expectedAnyHit": bool(row.get("expected_any_hit", row.get("expected_hit", row.get("expected_role_hit")))) if is_active_run else False,
                    "harmfulToolSelected": bool(row.get("harmful_tool_selected")) if is_active_run else False,
                    "harmfulTools": row.get("harmful_tools", []) if is_active_run else [],
                    "shadowPolicyRequired": bool(row.get("shadow_policy_required")) if is_active_run else False,
                    "shadowPolicyHit": row.get("shadow_policy_hit") if is_active_run else None,
                    "shadowCandidateCount": row.get("shadow_candidate_count", 0) if is_active_run else 0,
                    "regionCount": row.get("region_count", 0) if is_active_run else 0,
                    "counts": row.get("counts", {}) if is_active_run else {},
                    "selectedTools": fallback_tools if is_active_run else [],
                    "primaryAmbiguity": row.get("primary_ambiguity", row.get("expected_role", "")),
                    "originalUrl": _image_url(sample_path),
                    "regionsUrl": _game_region_url(root, case_id, sample_screen) if is_active_run else None,
                    "alphaUrl": alpha_url,
                    "matteUrl": matte_url,
                    "maskHintUrl": mask_hint_url,
                    "corridorkeyRawAlphaUrl": raw_alpha_url,
                    "corridorkeyForegroundUrl": foreground_url,
                    "candidates": candidates if is_active_run else [],
                }
            )

    return {
        "runId": report.get("run_id", root.name),
        "model": report.get("model", ""),
        "success": f"{report.get('ok_count', 0)}/{report.get('case_count', len(cases))}",
        "expectedHit": f"{report.get('expected_tool_hit_count', report.get('expected_role_hit_count', 0))}/{report.get('case_count', len(cases))}",
        "expectedAnyHit": f"{report.get('expected_any_tool_hit_count', report.get('expected_tool_hit_count', report.get('expected_role_hit_count', 0)))}/{report.get('case_count', len(cases))}",
        "harmfulTools": f"{report.get('harmful_tool_selected_count', 0)}/{report.get('case_count', len(cases))}",
        "shadowPolicyHit": f"{report.get('shadow_policy_hit_count', 0)}/{report.get('shadow_policy_required_count', 0)}",
        "sampleRows": len(cases),
        "reportPath": str(report_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str((root / "matte").relative_to(PROJECT_ROOT)),
        "vlmRoot": str(_game_vlm_root(root).relative_to(PROJECT_ROOT)),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": _game_eval_batch_progress(root, report_path, prefer_report_total=True),
        "samples": _game_eval_samples(),
        "cases": cases,
    }


@app.post("/eval/game/run")
def start_game_eval_run(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, object]:
    return _start_game_eval_batch(
        sample_ids=_selected_game_eval_sample_ids(payload),
        test_path=_selected_game_eval_test_path(payload),
    )


@app.get("/eval/game/run/{run_id}/status")
def game_eval_run_status(run_id: str) -> dict[str, object]:
    return _game_eval_batch_status(run_id)


@app.get("/eval/game/file/{rel_path:path}")
def game_eval_file(rel_path: str) -> FileResponse:
    path = (PROJECT_ROOT / rel_path).resolve()
    allowed_roots = [PROJECT_ROOT / "out", PROJECT_ROOT / "samples"]
    if not any(_is_relative_to(path, root.resolve()) for root in allowed_roots):
        raise HTTPException(status_code=404, detail="File is outside eval output roots.")
    if not path.exists() or not path.is_file() or path.suffix.lower() not in SERVABLE_IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


def _draw_region_overlay(input_path: Path, regions: list[dict[str, object]]) -> bytes:
    image = Image.open(input_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    line_width = max(2, min(w, h) // 220)

    for region in regions:
        bbox = region.get("bbox_xyxy")
        kind = str(region.get("kind", "unknown"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + line_width)
        if y2 <= y1:
            y2 = min(h - 1, y1 + line_width)
        outline = REGION_BOX_COLORS.get(kind, (255, 255, 255, 235))
        fill = REGION_FILL_COLORS.get(kind, (255, 255, 255, 20))
        draw.rectangle((x1, y1, x2, y2), outline=outline, fill=fill, width=line_width)

    legend_items = [
        ("same_bg", REGION_BOX_COLORS["same_bg_enclosed_region"]),
        ("alpha_diff", REGION_BOX_COLORS["alpha_keyer_disagreement"]),
        ("hard_edge", REGION_BOX_COLORS["hard_edge_candidate"]),
    ]
    legend_pad = max(8, line_width * 3)
    row_h = 18
    legend_w = 142
    legend_h = legend_pad * 2 + row_h * len(legend_items)
    draw.rounded_rectangle(
        (legend_pad, legend_pad, legend_pad + legend_w, legend_pad + legend_h),
        radius=6,
        fill=(255, 255, 255, 210),
        outline=(18, 25, 22, 90),
        width=1,
    )
    y = legend_pad * 2
    for label, color in legend_items:
        draw.rectangle((legend_pad * 2, y + 3, legend_pad * 2 + 12, y + 15), fill=color)
        draw.text((legend_pad * 2 + 18, y), label, fill=(18, 25, 22, 255))
        y += row_h

    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/eval/game/regions/{case_id}")
def game_eval_regions(
    case_id: str,
    run: str | None = Query(default=None),
    screen: str | None = Query(default=None),
) -> Response:
    root = _game_eval_root(run)
    rows = _game_report_rows(root)
    row = next(
        (
            item
            for item in rows
            if item.get("case_id") == case_id
            and (screen is None or item.get("sample_screen") == screen)
        ),
        None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    out_dir = _game_case_out_dir(row, root)
    summary = _load_json(out_dir / "summary.json")
    if not isinstance(summary, dict):
        raise HTTPException(status_code=500, detail="Case summary must be a JSON object.")
    input_path_value = summary.get("input")
    if isinstance(input_path_value, str):
        input_path = _resolve_project_path(input_path_value)
    else:
        sample_paths = _game_sample_paths(case_id)
        input_path = _resolve_project_path(sample_paths.get(str(row.get("sample_screen", screen or "")), ""))
    if not _is_relative_to(input_path, (PROJECT_ROOT / "samples").resolve()) or not input_path.exists():
        raise HTTPException(status_code=404, detail="Case input image not found.")
    regions = _candidate_regions(_candidate_result_items(out_dir / "candidate_results.json"))
    if not regions:
        regions = [
            item["region"]
            for item in summary.get("ownership", row.get("ownership", []))
            if isinstance(item, dict) and isinstance(item.get("region"), dict)
        ]
    png = _draw_region_overlay(input_path, regions)
    return Response(content=png, media_type="image/png")


@app.get("/eval/game", response_class=HTMLResponse)
def game_eval_page(run: str | None = Query(default=None)) -> str:
    root = _game_eval_root(run)
    data = _game_eval_data(root)
    data_json = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG Game Eval</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17201c;
      background: #f4f6f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      padding: 10px 20px;
      border-bottom: 1px solid #d6ddd4;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
    }}
    h1 {{ flex: 0 0 auto; margin: 0; font-size: 18px; letter-spacing: 0; white-space: nowrap; }}
    nav {{ min-width: 0; display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 10px; font-size: 13px; color: #53615a; }}
    nav a {{ color: #196f5a; font-weight: 700; text-decoration: none; white-space: nowrap; }}
    #run-id {{
      min-width: 0;
      flex: 1 1 260px;
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .run-button {{
      min-height: 34px;
      padding: 0 12px;
      border: 0;
      border-radius: 6px;
      background: #176a56;
      color: #ffffff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      white-space: nowrap;
    }}
    .run-button:disabled {{ opacity: 0.58; cursor: progress; }}
    .run-status {{ flex: 0 1 160px; min-width: 92px; color: #53615a; font-size: 12px; font-weight: 800; }}
    .run-progress {{
      flex: 1 1 160px;
      width: 128px;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dce4d9;
    }}
    .run-progress-bar {{
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: #176a56;
      transition: width 180ms ease;
    }}
    .run-picker {{
      min-width: 0;
      flex: 1 1 420px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #53615a;
      font-weight: 800;
    }}
    .run-picker span {{ flex: 0 0 auto; white-space: nowrap; }}
    .run-picker select {{
      min-width: 0;
      width: 100%;
      min-height: 34px;
      padding: 0 30px 0 10px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
      font: inherit;
      font-weight: 700;
    }}
    main {{ width: min(1600px, 100%); margin: 0 auto; padding: 18px 20px 28px; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
      color: #53615a;
      font-size: 13px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border: 1px solid #d1d9cf;
      border-radius: 999px;
      background: #ffffff;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid #d6ddd4;
      border-radius: 8px;
      background: #ffffff;
    }}
    table {{
      width: max(100%, 1280px);
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
    }}
    th, td {{ border-bottom: 1px solid #e2e8df; vertical-align: top; }}
    th {{
      position: sticky;
      top: 0;
      z-index: 5;
      height: 40px;
      padding: 0 10px;
      background: #fbfcfa;
      color: #53615a;
      font-size: 12px;
      text-align: left;
      white-space: nowrap;
    }}
    td {{ padding: 10px; }}
    tr:last-child td {{ border-bottom: 0; }}
    .compare-col {{ width: 92px; }}
    .preview-col {{ width: 148px; }}
    .compare-button {{
      width: 100%;
      min-height: 34px;
      border: 1px solid #176a56;
      border-radius: 6px;
      background: #176a56;
      color: #ffffff;
      font: inherit;
      font-size: 13px;
      font-weight: 900;
      cursor: pointer;
    }}
    .compare-button:disabled {{
      border-color: #cbd5c8;
      background: #eef3ec;
      color: #758179;
      cursor: not-allowed;
    }}
    .case-name {{ font-size: 13px; font-weight: 800; overflow-wrap: anywhere; }}
    .sample-code {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      margin-bottom: 7px;
      padding: 0 8px;
      border-radius: 6px;
      color: #ffffff;
      background: #245f53;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0;
      white-space: nowrap;
    }}
    .sample-badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      margin-top: 7px;
      padding: 0 8px;
      border: 1px solid #cbd5c8;
      border-radius: 999px;
      color: #17201c;
      background: #f7f9f6;
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .case-meta {{ margin-top: 6px; display: grid; gap: 5px; color: #5f6c66; font-size: 12px; line-height: 1.35; }}
    .hit {{ color: #176a56; font-weight: 800; }}
    .miss {{ color: #a23d35; font-weight: 800; }}
    .pending {{ color: #6b6258; font-weight: 800; }}
    .tools {{ overflow-wrap: anywhere; }}
    .thumb-button {{
      position: relative;
      width: 100%;
      min-height: 92px;
      max-height: 220px;
      aspect-ratio: var(--thumb-ratio, 1 / 1);
      display: grid;
      place-items: center;
      padding: 6px;
      border: 1px solid #cad3c7;
      border-radius: 6px;
      cursor: zoom-in;
      overflow: hidden;
    }}
    .thumb-tag {{
      position: absolute;
      top: 6px;
      left: 6px;
      max-width: calc(100% - 12px);
      padding: 3px 6px;
      border-radius: 5px;
      background: rgba(12, 17, 15, 0.78);
      color: #ffffff;
      font-size: 11px;
      font-weight: 900;
      line-height: 1.2;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      pointer-events: none;
    }}
    .thumb-button img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      user-select: none;
      pointer-events: none;
    }}
    .thumb-button:focus-visible {{ outline: 3px solid rgba(25, 111, 90, 0.32); outline-offset: 2px; }}
    .bg-checker {{
      background-color: #edf1ea;
      background-image:
        linear-gradient(45deg, #cad3c7 25%, transparent 25%),
        linear-gradient(-45deg, #cad3c7 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #cad3c7 75%),
        linear-gradient(-45deg, transparent 75%, #cad3c7 75%);
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }}
    .bg-white {{ background: #ffffff; }}
    .bg-black {{ background: #101413; }}
    .bg-purple {{ background: #7c3aed; }}
    .bg-blue {{ background: #2563eb; }}
    /* Known green-screen reference for judging whether transparent shadows
       match the original source, without white/checker contrast bias. */
    .bg-green {{ background: #00c800; }}
    .candidate-label {{ margin-bottom: 7px; color: #53615a; font-size: 12px; font-weight: 800; overflow-wrap: anywhere; }}
    .selected-mark {{
      display: inline-flex;
      margin-left: 6px;
      color: #176a56;
      font-weight: 900;
    }}
    .empty-cell {{
      width: 100%;
      min-height: 92px;
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
      border: 1px dashed #cbd5c8;
      border-radius: 6px;
      color: #66736c;
      background: #f7f9f6;
      font-size: 12px;
      font-weight: 800;
    }}
    .modal {{
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      grid-template-rows: 56px 1fr;
      background: rgba(12, 17, 15, 0.94);
    }}
    .modal.is-open {{ display: grid; }}
    .modal-bar {{
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 0 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.14);
      color: #ffffff;
    }}
    .modal-title {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 800; }}
    .modal-actions {{ display: flex; align-items: center; gap: 8px; flex-shrink: 0; }}
    .swatch, .icon-button {{
      width: 34px;
      height: 34px;
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 6px;
      color: #ffffff;
      background: transparent;
      cursor: pointer;
    }}
    .swatch[aria-pressed="true"], .icon-button:focus-visible {{
      outline: 2px solid #ffffff;
      outline-offset: 2px;
    }}
    .icon-button {{ font-size: 18px; line-height: 1; }}
    .modal-stage {{
      min-height: 0;
      display: grid;
      place-items: center;
      overflow: hidden;
      touch-action: none;
      cursor: grab;
    }}
    .modal-stage.is-dragging {{ cursor: grabbing; }}
    .modal-stage img {{
      max-width: 86vw;
      max-height: 82vh;
      object-fit: contain;
      transform-origin: center center;
      will-change: transform;
      user-select: none;
      pointer-events: none;
    }}
    .compare-modal {{
      position: fixed;
      inset: 0;
      z-index: 70;
      display: none;
      grid-template-rows: 64px 1fr;
      background: rgba(0, 0, 0, 0.96);
      color: #ffffff;
    }}
    .compare-modal.is-open {{ display: grid; }}
    .compare-bar {{
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.14);
    }}
    .compare-picker {{
      min-width: 0;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #cdd5d0;
      font-size: 13px;
      font-weight: 900;
    }}
    .compare-picker select {{
      width: min(240px, 30vw);
      min-height: 36px;
      padding: 0 32px 0 10px;
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 6px;
      background: #111614;
      color: #ffffff;
      font: inherit;
      font-weight: 800;
    }}
    .compare-alpha {{
      min-width: 132px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #cdd5d0;
      font-size: 12px;
      font-weight: 900;
    }}
    .compare-alpha input {{
      width: 104px;
      accent-color: #ffffff;
      cursor: pointer;
    }}
    .compare-close {{
      position: absolute;
      top: 14px;
      right: 16px;
    }}
    .compare-stage {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      place-items: center;
      gap: 12px;
      overflow: hidden;
      padding: 24px;
    }}
    .compare-bg-row {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }}
    .compare-bg-button {{
      width: 34px;
      height: 34px;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 6px;
      cursor: pointer;
    }}
    .compare-bg-button[aria-pressed="true"], .compare-bg-button:focus-visible {{
      outline: 2px solid #ffffff;
      outline-offset: 2px;
    }}
    .compare-frame {{
      position: relative;
      width: min(calc(100vw - 48px), calc(100vh - 172px));
      aspect-ratio: 1 / 1;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      background: #070908;
      cursor: ew-resize;
    }}
    .compare-frame.bg-checker {{ background-color: #edf1ea; }}
    .compare-frame.bg-white {{ background: #ffffff; }}
    .compare-frame.bg-black {{ background: #101413; }}
    .compare-frame.bg-green {{ background: #00c800; }}
    .compare-frame.bg-purple {{ background: #7c3aed; }}
    .compare-frame.bg-blue {{ background: #2563eb; }}
    .compare-frame img {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      user-select: none;
      pointer-events: none;
    }}
    .compare-img-two {{
      clip-path: inset(0 50% 0 0);
    }}
    .compare-divider {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 2px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.28);
      transform: translateX(-1px);
      pointer-events: none;
    }}
    .compare-empty {{
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      color: #aab3ae;
      font-size: 13px;
      font-weight: 900;
      background: #070908;
    }}
    .compare-frame.is-empty .compare-empty {{ display: grid; }}
    .compare-frame.is-empty img, .compare-frame.is-empty .compare-divider {{ display: none; }}
    .eval-panel {{
      position: fixed;
      inset: 0;
      z-index: 60;
      display: none;
      place-items: center;
      padding: 20px;
      background: rgba(12, 17, 15, 0.58);
    }}
    .eval-panel.is-open {{ display: grid; }}
    .eval-dialog {{
      width: min(720px, 100%);
      max-height: min(760px, calc(100vh - 40px));
      display: grid;
      grid-template-rows: auto auto auto auto 1fr auto;
      overflow: hidden;
      border: 1px solid #d6ddd4;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 18px 52px rgba(20, 31, 26, 0.22);
    }}
    .eval-dialog header {{
      position: static;
      min-height: 54px;
      padding: 0 16px;
      border-bottom: 1px solid #e2e8df;
      background: #ffffff;
      backdrop-filter: none;
    }}
    .eval-dialog h2 {{ margin: 0; font-size: 16px; letter-spacing: 0; }}
    .eval-tools {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      border-bottom: 1px solid #e2e8df;
    }}
    .eval-tools button, .eval-actions button {{
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }}
    .eval-tools .selection-count {{ margin-left: auto; color: #53615a; font-size: 12px; font-weight: 800; }}
    .path-tools, .screen-tools {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px 16px;
      border-bottom: 1px solid #e2e8df;
      color: #53615a;
      font-size: 13px;
      font-weight: 800;
    }}
    .path-tools label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
    }}
    .path-tools select {{
      min-width: 180px;
      min-height: 34px;
      padding: 0 10px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
      font: inherit;
      font-weight: 800;
    }}
    .screen-option {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 0 10px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
    }}
    .screen-option input {{ width: 15px; height: 15px; min-height: 0; margin: 0; }}
    .sample-list {{
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(218px, 1fr));
      align-content: start;
      gap: 10px;
      overflow: auto;
      padding: 12px 16px;
      background: #f7faf6;
    }}
    .sample-group {{
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 24px;
      margin-top: 4px;
      color: #53615a;
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .sample-group:first-child {{ margin-top: 0; }}
    .sample-option {{
      display: grid;
      grid-template-columns: 22px 54px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      min-height: 74px;
      padding: 8px;
      border: 1px solid #dde6da;
      border-radius: 7px;
      background: #ffffff;
      color: #17201c;
      font-size: 13px;
      font-weight: 700;
    }}
    .sample-option:hover {{ border-color: #b8c8b4; background: #fbfdfb; }}
    .sample-option input {{ width: 16px; height: 16px; min-height: 0; margin: 0; }}
    .sample-thumb {{
      width: 54px;
      height: 54px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid #cbd5c8;
      border-radius: 6px;
      background: #00c800;
    }}
    .sample-thumb img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .sample-meta {{ min-width: 0; display: grid; gap: 2px; }}
    .sample-code {{ margin: 0; font-weight: 900; line-height: 1.1; }}
    .sample-case, .sample-family {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #617068;
      font-size: 11px;
      line-height: 1.15;
    }}
    .eval-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding: 12px 16px;
      border-top: 1px solid #e2e8df;
    }}
    .eval-actions .primary {{
      border-color: #176a56;
      background: #176a56;
      color: #ffffff;
    }}
    .eval-actions .primary:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    @media (max-width: 980px) {{
      header {{
        position: static;
        padding: 10px 16px;
      }}
      nav {{ width: 100%; gap: 8px; }}
      #run-id {{ flex: 1 1 100%; max-width: 100%; }}
      .run-picker {{ flex: 1 1 320px; width: auto; min-width: 0; }}
      .run-picker select {{ width: 100%; }}
      .run-button {{ flex: 0 0 auto; }}
      .run-status {{ flex: 1 1 180px; }}
      .run-progress {{ flex: 1 1 160px; }}
      main {{ padding: 14px 12px 22px; }}
      .modal-bar {{ min-height: 92px; align-items: flex-start; flex-direction: column; padding: 10px 12px; }}
      .modal {{ grid-template-rows: auto 1fr; }}
      .compare-modal {{ grid-template-rows: auto 1fr; }}
      .compare-bar {{ justify-content: flex-start; flex-wrap: wrap; padding-right: 58px; }}
      .compare-picker select {{ width: min(220px, 62vw); }}
      .compare-alpha {{ min-width: 124px; }}
    }}
    @media (max-width: 560px) {{
      .run-picker {{ flex-basis: 100%; }}
      .run-progress, .run-status {{ flex-basis: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ERMBG Game Eval</h1>
    <nav>
      <span id="run-id"></span>
      <label class="run-picker" for="run-select">
        <span>批次</span>
        <select id="run-select" aria-label="选择测试批次"></select>
      </label>
      <button class="run-button" type="button" id="start-full-eval">启动测试</button>
      <div class="run-progress" role="progressbar" aria-label="测试进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <div class="run-progress-bar" id="batch-progress"></div>
      </div>
      <span class="run-status" id="batch-status" aria-live="polite"></span>
      <a href="/">上传页</a>
    </nav>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <div class="table-wrap">
      <table aria-label="game eval result table">
        <thead>
          <tr>
            <th class="compare-col">比较</th>
            <th class="preview-col">原图</th>
            <th class="preview-col">alpha mask</th>
            <th class="preview-col">白底</th>
            <th class="preview-col">黑底</th>
            <th class="preview-col">透明底</th>
            <th class="preview-col">绿底</th>
            <th class="preview-col">紫底</th>
            <th class="preview-col">蓝底</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </main>
  <div class="eval-panel" id="eval-panel" aria-hidden="true">
    <section class="eval-dialog" role="dialog" aria-modal="true" aria-labelledby="eval-dialog-title">
      <header>
        <h2 id="eval-dialog-title">选择测试样本</h2>
      </header>
      <div class="eval-tools">
        <button type="button" id="select-all-samples">全选</button>
        <button type="button" id="clear-all-samples">取消全选</button>
        <span class="selection-count" id="selection-count"></span>
      </div>
      <div class="path-tools" aria-label="选择测试路径">
        <label for="eval-test-path">测试路径
          <select id="eval-test-path" name="eval-test-path">
            <option value="auto" selected>Auto RouteMatte</option>
            <option value="corridorkey">CorridorKey</option>
            <option value="rmbg">RMBG</option>
          </select>
        </label>
      </div>
      <div class="sample-list" id="sample-list"></div>
      <div class="eval-actions">
        <button type="button" id="cancel-eval-panel">取消</button>
        <button class="primary" type="button" id="confirm-start-eval">开始测试</button>
      </div>
    </section>
  </div>
  <div class="compare-modal" id="compare-modal" aria-hidden="true">
    <div class="compare-bar">
      <label class="compare-alpha" for="compare-alpha-one">Alpha 1
        <input id="compare-alpha-one" type="range" min="0" max="100" value="100" step="1">
      </label>
      <label class="compare-picker" for="compare-view-one">视图1
        <select id="compare-view-one"></select>
      </label>
      <label class="compare-picker" for="compare-view-two">视图2
        <select id="compare-view-two"></select>
      </label>
      <label class="compare-alpha" for="compare-alpha-two">Alpha 2
        <input id="compare-alpha-two" type="range" min="0" max="100" value="100" step="1">
      </label>
      <button class="icon-button compare-close" type="button" id="close-compare" title="关闭" aria-label="关闭">×</button>
    </div>
    <div class="compare-stage" id="compare-stage">
      <div class="compare-bg-row" aria-label="比较背景色">
        <button class="compare-bg-button bg-black" type="button" data-bg="black" title="黑底" aria-label="黑底"></button>
        <button class="compare-bg-button bg-white" type="button" data-bg="white" title="白底" aria-label="白底"></button>
        <button class="compare-bg-button bg-checker" type="button" data-bg="checker" title="透明底" aria-label="透明底"></button>
        <button class="compare-bg-button bg-green" type="button" data-bg="green" title="绿底" aria-label="绿底"></button>
        <button class="compare-bg-button bg-purple" type="button" data-bg="purple" title="紫底" aria-label="紫底"></button>
        <button class="compare-bg-button bg-blue" type="button" data-bg="blue" title="蓝底" aria-label="蓝底"></button>
      </div>
      <div class="compare-frame bg-black" id="compare-frame">
        <img class="compare-img-one" id="compare-img-one" alt="">
        <img class="compare-img-two" id="compare-img-two" alt="">
        <div class="compare-divider" id="compare-divider"></div>
        <div class="compare-empty" id="compare-empty">没有可比较的图片</div>
      </div>
    </div>
  </div>
  <div class="modal" id="modal" aria-hidden="true">
    <div class="modal-bar">
      <div class="modal-title" id="modal-title"></div>
      <div class="modal-actions" aria-label="preview controls">
        <button class="swatch bg-checker" type="button" data-bg="checker" title="透明底" aria-label="透明底"></button>
        <button class="swatch bg-white" type="button" data-bg="white" title="白底" aria-label="白底"></button>
        <button class="swatch bg-black" type="button" data-bg="black" title="黑底" aria-label="黑底"></button>
        <button class="swatch bg-green" type="button" data-bg="green" title="绿幕参照" aria-label="绿幕参照"></button>
        <button class="swatch bg-purple" type="button" data-bg="purple" title="紫底" aria-label="紫底"></button>
        <button class="swatch bg-blue" type="button" data-bg="blue" title="蓝底" aria-label="蓝底"></button>
        <button class="icon-button" type="button" id="reset-preview" title="重置视图" aria-label="重置视图">↺</button>
        <button class="icon-button" type="button" id="close-modal" title="关闭" aria-label="关闭">×</button>
      </div>
    </div>
    <div class="modal-stage bg-checker" id="modal-stage">
      <img id="modal-img" alt="">
    </div>
  </div>
  <script>
    const data = {data_json};
    const backgrounds = ["checker", "white", "black", "green", "purple", "blue"];
    const previewColumns = [
      {{ label: "原图", urlKey: "originalUrl", bg: "checker" }},
      {{ label: "alpha mask", urlKey: "alphaUrl", bg: "white" }},
      {{ label: "白底", urlKey: "matteUrl", bg: "white" }},
      {{ label: "黑底", urlKey: "matteUrl", bg: "black" }},
      {{ label: "透明底", urlKey: "matteUrl", bg: "checker" }},
      {{ label: "绿底", urlKey: "matteUrl", bg: "green" }},
      {{ label: "紫底", urlKey: "matteUrl", bg: "purple" }},
      {{ label: "蓝底", urlKey: "matteUrl", bg: "blue" }},
    ];
    const compareOptions = [
      {{ label: "原图", urlKey: "originalUrl" }},
      {{ label: "Mask Hint", urlKey: "maskHintUrl" }},
      {{ label: "corridorkey Raw Alpha", urlKey: "corridorkeyRawAlphaUrl" }},
      {{ label: "corridorkey Forground", urlKey: "corridorkeyForegroundUrl" }},
      {{ label: "输出 Alpha", urlKey: "alphaUrl" }},
    ];
    const rowsEl = document.getElementById("rows");
    const summaryEl = document.getElementById("summary");
    const runIdEl = document.getElementById("run-id");
    const runSelect = document.getElementById("run-select");
    const startFullEvalButton = document.getElementById("start-full-eval");
    const batchProgress = document.getElementById("batch-progress");
    const batchProgressRoot = batchProgress.parentElement;
    const batchStatusEl = document.getElementById("batch-status");
    const evalPanel = document.getElementById("eval-panel");
    const sampleList = document.getElementById("sample-list");
    const testPathSelect = document.getElementById("eval-test-path");
    const selectAllSamplesButton = document.getElementById("select-all-samples");
    const clearAllSamplesButton = document.getElementById("clear-all-samples");
    const cancelEvalPanelButton = document.getElementById("cancel-eval-panel");
    const confirmStartEvalButton = document.getElementById("confirm-start-eval");
    const selectionCountEl = document.getElementById("selection-count");
    const modal = document.getElementById("modal");
    const modalStage = document.getElementById("modal-stage");
    const modalImg = document.getElementById("modal-img");
    const modalTitle = document.getElementById("modal-title");
    const closeModalButton = document.getElementById("close-modal");
    const resetPreviewButton = document.getElementById("reset-preview");
    const swatches = Array.from(document.querySelectorAll(".swatch"));
    const compareModal = document.getElementById("compare-modal");
    const compareStage = document.getElementById("compare-stage");
    const compareFrame = document.getElementById("compare-frame");
    const compareViewOne = document.getElementById("compare-view-one");
    const compareViewTwo = document.getElementById("compare-view-two");
    const compareAlphaOne = document.getElementById("compare-alpha-one");
    const compareAlphaTwo = document.getElementById("compare-alpha-two");
    const compareImgOne = document.getElementById("compare-img-one");
    const compareImgTwo = document.getElementById("compare-img-two");
    const compareDivider = document.getElementById("compare-divider");
    const closeCompareButton = document.getElementById("close-compare");
    const compareBgButtons = Array.from(document.querySelectorAll(".compare-bg-button"));
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let dragStart = null;
    let activeBatchStatusUrl = "";
    let activeCompareCase = null;
    let comparePosition = 0.5;
    let compareAlphaOneValue = 1;
    let compareAlphaTwoValue = 1;

    function text(value) {{
      return value === null || value === undefined || value === "" ? "—" : String(value);
    }}

    function setBackground(element, bg) {{
      element.classList.remove(...backgrounds.map((name) => `bg-${{name}}`));
      element.classList.add(`bg-${{bg}}`);
    }}

    function countsText(counts) {{
      if (!counts || typeof counts !== "object") return "";
      return Object.entries(counts).map(([key, value]) => `${{key}}=${{value}}`).join(", ");
    }}

    function makePreview(src, label, bg, tag = "") {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      setBackground(button, bg);
      button.title = label;
      const img = document.createElement("img");
      img.src = src;
      img.alt = label;
      img.onload = () => {{
        if (img.naturalWidth > 0 && img.naturalHeight > 0) {{
          button.style.setProperty("--thumb-ratio", img.naturalWidth + " / " + img.naturalHeight);
        }}
      }};
      button.appendChild(img);
      if (tag) {{
        const badge = document.createElement("span");
        badge.className = "thumb-tag";
        badge.textContent = tag;
        button.appendChild(badge);
      }}
      button.addEventListener("click", () => openModal(src, label, bg));
      return button;
    }}

    function availableCompareViews(caseItem) {{
      return compareOptions
        .map((option) => ({{ ...option, url: caseItem[option.urlKey] || "" }}))
        .filter((option) => option.url);
    }}

    function populateCompareSelect(select, caseItem, preferredKey, fallbackViews) {{
      select.innerHTML = "";
      compareOptions.forEach((option) => {{
        const choice = document.createElement("option");
        choice.value = option.urlKey;
        choice.textContent = option.label;
        choice.disabled = !caseItem[option.urlKey];
        select.appendChild(choice);
      }});
      const preferred = compareOptions.find((option) => option.urlKey === preferredKey && caseItem[option.urlKey]);
      select.value = preferred ? preferred.urlKey : (fallbackViews[0] ? fallbackViews[0].urlKey : compareOptions[0].urlKey);
    }}

    function updateCompareImages() {{
      if (!activeCompareCase) return;
      const first = compareOptions.find((option) => option.urlKey === compareViewOne.value);
      const second = compareOptions.find((option) => option.urlKey === compareViewTwo.value);
      const firstUrl = first ? activeCompareCase[first.urlKey] : "";
      const secondUrl = second ? activeCompareCase[second.urlKey] : "";
      compareFrame.classList.toggle("is-empty", !firstUrl || !secondUrl);
      compareImgOne.src = firstUrl || "";
      compareImgTwo.src = secondUrl || "";
      compareImgOne.alt = first ? first.label : "";
      compareImgTwo.alt = second ? second.label : "";
      updateComparePosition(comparePosition);
      updateCompareAlpha();
    }}

    function updateCompareAlpha() {{
      compareAlphaOneValue = Math.max(0, Math.min(1, Number(compareAlphaOne.value) / 100));
      compareAlphaTwoValue = Math.max(0, Math.min(1, Number(compareAlphaTwo.value) / 100));
      compareImgOne.style.opacity = String(compareAlphaOneValue);
      compareImgTwo.style.opacity = String(compareAlphaTwoValue);
    }}

    function resetCompareAlpha() {{
      compareAlphaOne.value = "100";
      compareAlphaTwo.value = "100";
      updateCompareAlpha();
    }}

    function setCompareBackground(bg) {{
      setBackground(compareFrame, bg);
      compareBgButtons.forEach((button) => {{
        button.setAttribute("aria-pressed", String(button.dataset.bg === bg));
      }});
    }}

    function updateComparePosition(value) {{
      comparePosition = Math.max(0, Math.min(1, value));
      const pct = comparePosition * 100;
      compareImgTwo.style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
      compareDivider.style.left = `${{pct}}%`;
    }}

    function updateCompareFromPointer(event) {{
      if (!compareModal.classList.contains("is-open")) return;
      const rect = compareFrame.getBoundingClientRect();
      if (!rect.width) return;
      updateComparePosition((event.clientX - rect.left) / rect.width);
    }}

    function openCompare(caseItem) {{
      activeCompareCase = caseItem;
      const views = availableCompareViews(caseItem);
      populateCompareSelect(compareViewOne, caseItem, "maskHintUrl", views);
      populateCompareSelect(compareViewTwo, caseItem, "corridorkeyRawAlphaUrl", views.slice(1).length ? views.slice(1) : views);
      if (compareViewOne.value === compareViewTwo.value && views.length > 1) {{
        const alternate = views.find((view) => view.urlKey !== compareViewOne.value);
        if (alternate) compareViewTwo.value = alternate.urlKey;
      }}
      updateComparePosition(0.5);
      resetCompareAlpha();
      setCompareBackground("black");
      updateCompareImages();
      compareModal.classList.add("is-open");
      compareModal.setAttribute("aria-hidden", "false");
    }}

    function closeCompare() {{
      compareModal.classList.remove("is-open");
      compareModal.setAttribute("aria-hidden", "true");
      activeCompareCase = null;
      compareImgOne.removeAttribute("src");
      compareImgTwo.removeAttribute("src");
    }}

    function renderRunSelect() {{
      const runs = Array.isArray(data.runs) ? data.runs : [];
      runSelect.innerHTML = "";
      runs.forEach((run) => {{
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = run.label || run.id;
        option.selected = run.selected === true || run.id === data.selectedRun;
        runSelect.appendChild(option);
      }});
      runSelect.disabled = runs.length <= 1;
    }}

    function renderRows() {{
      renderRunSelect();
      runIdEl.textContent = data.runId || "game eval";
      setBatchProgress(data.progress);
      if (data.progress) {{
        setBatchStatus(`当前：${{progressText(data)}}`, false);
      }}
      summaryEl.innerHTML = "";
      [
        `model: ${{text(data.model)}}`,
        `success: ${{text(data.success)}}`,
        `expected hit: ${{text(data.expectedHit)}}`,
        `any hit: ${{text(data.expectedAnyHit)}}`,
        `harmful tools: ${{text(data.harmfulTools)}}`,
        `shadow policy: ${{text(data.shadowPolicyHit)}}`,
        `sample rows: ${{text(data.sampleRows)}}`,
        `report: ${{text(data.reportPath)}}`,
        `samples: ${{text(data.vlmRoot)}}`,
        `matte: ${{text(data.matteRoot)}}`,
      ].forEach((item) => {{
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = item;
        summaryEl.appendChild(pill);
      }});

      rowsEl.innerHTML = "";
      data.cases.forEach((caseItem) => {{
        const row = document.createElement("tr");
        const compareCell = document.createElement("td");
        const compareButton = document.createElement("button");
        const compareViews = availableCompareViews(caseItem);
        compareButton.type = "button";
        compareButton.className = "compare-button";
        compareButton.textContent = "比较";
        compareButton.disabled = compareViews.length < 2;
        compareButton.title = compareViews.length < 2 ? "至少需要两张图" : `${{caseItem.sampleCode || ""}} · ${{caseItem.caseId || ""}} · 比较`;
        compareButton.addEventListener("click", () => openCompare(caseItem));
        compareCell.appendChild(compareButton);
        row.appendChild(compareCell);
        previewColumns.forEach((column) => {{
          const cell = document.createElement("td");
          const previewUrl = caseItem[column.urlKey] || "";
          if (previewUrl) {{
            cell.appendChild(
              makePreview(
                previewUrl,
                `${{caseItem.sampleCode || ""}} · ${{caseItem.caseId || ""}} · ${{column.label}}`,
                column.bg,
                column.urlKey === "originalUrl" ? `${{caseItem.sampleCode || ""}}` : "",
              ),
            );
          }} else {{
            const empty = document.createElement("div");
            empty.className = "empty-cell";
            empty.textContent = "—";
            cell.appendChild(empty);
          }}
          row.appendChild(cell);
        }});
        rowsEl.appendChild(row);
      }});
    }}

    function setBatchStatus(message, isRunning = false) {{
      batchStatusEl.textContent = message || "";
      startFullEvalButton.disabled = isRunning;
    }}

    function setBatchProgress(progress) {{
      const pct = progress && Number.isFinite(Number(progress.percent)) ? Number(progress.percent) : 0;
      const clamped = Math.max(0, Math.min(100, pct));
      batchProgress.style.width = `${{clamped}}%`;
      batchProgressRoot.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }}

    function progressText(status) {{
      const progress = status && status.progress ? status.progress : {{}};
      const completed = Number.isFinite(Number(progress.completed)) ? Number(progress.completed) : 0;
      const total = Number.isFinite(Number(progress.total)) ? Number(progress.total) : 18;
      const ok = Number.isFinite(Number(progress.ok)) ? Number(progress.ok) : 0;
      const errors = Number.isFinite(Number(progress.errors)) ? Number(progress.errors) : 0;
      return `${{completed}}/${{total}} · ok ${{ok}} · error ${{errors}}`;
    }}

    async function pollBatchStatus() {{
      if (!activeBatchStatusUrl) return;
      try {{
        const response = await fetch(activeBatchStatusUrl);
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const status = await response.json();
        setBatchProgress(status.progress);
        if (status.status === "complete" && status.hasReport && status.url) {{
          setBatchStatus(`完成：${{progressText(status)}}`, false);
          window.location.href = status.url;
          return;
        }}
        if (status.status === "error") {{
          setBatchStatus(`失败：${{progressText(status)}}`, false);
          activeBatchStatusUrl = "";
          return;
        }}
        setBatchStatus(`运行中：${{progressText(status)}}`, true);
        window.setTimeout(pollBatchStatus, 5000);
      }} catch (error) {{
        setBatchStatus("状态读取失败", false);
        activeBatchStatusUrl = "";
      }}
    }}

    function sampleCheckboxes() {{
      return Array.from(sampleList.querySelectorAll('input[type="checkbox"]'));
    }}

    function selectedSampleIds() {{
      return sampleCheckboxes().filter((input) => input.checked).map((input) => input.value);
    }}

    function selectedTestPath() {{
      return testPathSelect ? testPathSelect.value : "corridorkey";
    }}

    function updateSelectionCount() {{
      const selected = selectedSampleIds().length;
      const total = sampleCheckboxes().length;
      selectionCountEl.textContent = `${{selected}}/${{total}} samples`;
      confirmStartEvalButton.disabled = selected === 0;
    }}

    function renderSampleList() {{
      const samples = Array.isArray(data.samples) ? data.samples : [];
      sampleList.innerHTML = "";
      const labels = {{ button: "Button", icon: "Icon / Effect", character: "Character" }};
      const grouped = samples.reduce((acc, sample) => {{
        const key = sample.category || "other";
        if (!acc.has(key)) acc.set(key, []);
        acc.get(key).push(sample);
        return acc;
      }}, new Map());
      grouped.forEach((groupSamples, category) => {{
        const group = document.createElement("div");
        group.className = "sample-group";
        group.textContent = `${{labels[category] || category}} · ${{groupSamples.length}}`;
        sampleList.appendChild(group);
        groupSamples.forEach((sample) => {{
        const label = document.createElement("label");
        label.className = "sample-option";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = sample.sampleId;
        checkbox.checked = sample.defaultSelected === true;
        checkbox.addEventListener("change", updateSelectionCount);
        const thumb = document.createElement("span");
        thumb.className = "sample-thumb";
        if (sample.thumbnailUrl) {{
          const image = document.createElement("img");
          image.src = sample.thumbnailUrl;
          image.alt = sample.sampleId || "";
          thumb.appendChild(image);
        }}
        const code = document.createElement("span");
        code.className = "sample-meta";
        const codeText = document.createElement("span");
        codeText.className = "sample-code";
        codeText.textContent = `${{sample.sampleId || ""}} · ${{sample.screen || ""}}`;
        const caseText = document.createElement("span");
        caseText.className = "sample-case";
        caseText.textContent = sample.caseId || "";
        const familyText = document.createElement("span");
        familyText.className = "sample-family";
        familyText.textContent = sample.family || sample.primaryAmbiguity || "";
        code.appendChild(codeText);
        code.appendChild(caseText);
        code.appendChild(familyText);
        label.title = `${{sample.sampleId || ""}} · ${{sample.caseId || ""}}`;
        label.appendChild(checkbox);
        label.appendChild(thumb);
        label.appendChild(code);
        sampleList.appendChild(label);
      }});
      }});
      updateSelectionCount();
    }}

    function openEvalPanel() {{
      renderSampleList();
      evalPanel.classList.add("is-open");
      evalPanel.setAttribute("aria-hidden", "false");
    }}

    function closeEvalPanel() {{
      evalPanel.classList.remove("is-open");
      evalPanel.setAttribute("aria-hidden", "true");
    }}

    function setAllSamples(checked) {{
      sampleCheckboxes().forEach((input) => {{
        input.checked = checked;
      }});
      updateSelectionCount();
    }}

    async function startSelectedEval() {{
      const sampleIds = selectedSampleIds();
      const testPath = selectedTestPath();
      if (!sampleIds.length) return;
      setBatchStatus("启动中", true);
      setBatchProgress({{ percent: 0 }});
      try {{
        closeEvalPanel();
        const response = await fetch("/eval/game/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ sample_ids: sampleIds, test_path: testPath }}),
        }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const payload = await response.json();
        activeBatchStatusUrl = payload.statusUrl || "";
        setBatchProgress(payload.progress);
        setBatchStatus(`运行中：${{progressText(payload)}}`, true);
        window.setTimeout(pollBatchStatus, 5000);
      }} catch (error) {{
        setBatchStatus("启动失败", false);
      }}
    }}

    function clampScale(value) {{
      return Math.min(16, Math.max(0.1, value));
    }}

    function applyTransform() {{
      modalImg.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
    }}

    function resetTransform() {{
      scale = 1;
      panX = 0;
      panY = 0;
      dragStart = null;
      applyTransform();
    }}

    function setModalBackground(bg) {{
      setBackground(modalStage, bg);
      swatches.forEach((swatch) => swatch.setAttribute("aria-pressed", String(swatch.dataset.bg === bg)));
    }}

    function openModal(src, label, bg) {{
      modalImg.src = src;
      modalImg.alt = label;
      modalTitle.textContent = label;
      setModalBackground(bg || "checker");
      resetTransform();
      modal.classList.add("is-open");
      modal.setAttribute("aria-hidden", "false");
    }}

    function closeModal() {{
      modal.classList.remove("is-open");
      modal.setAttribute("aria-hidden", "true");
      modalImg.removeAttribute("src");
    }}

    modalStage.addEventListener("wheel", (event) => {{
      if (!modal.classList.contains("is-open")) return;
      event.preventDefault();
      const rect = modalStage.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const pointerX = event.clientX - centerX;
      const pointerY = event.clientY - centerY;
      const previousScale = scale;
      const factor = event.deltaY < 0 ? 1.14 : 1 / 1.14;
      scale = clampScale(scale * factor);
      panX = pointerX - ((pointerX - panX) * scale) / previousScale;
      panY = pointerY - ((pointerY - panY) * scale) / previousScale;
      applyTransform();
    }}, {{ passive: false }});

    modalStage.addEventListener("pointerdown", (event) => {{
      if (!modal.classList.contains("is-open")) return;
      dragStart = {{ pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX, panY }};
      modalStage.setPointerCapture(event.pointerId);
      modalStage.classList.add("is-dragging");
    }});

    modalStage.addEventListener("pointermove", (event) => {{
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      panX = dragStart.panX + event.clientX - dragStart.x;
      panY = dragStart.panY + event.clientY - dragStart.y;
      applyTransform();
    }});

    function endDrag(event) {{
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      dragStart = null;
      modalStage.classList.remove("is-dragging");
    }}

    modalStage.addEventListener("pointerup", endDrag);
    modalStage.addEventListener("pointercancel", endDrag);
    modalStage.addEventListener("dblclick", resetTransform);
    closeModalButton.addEventListener("click", closeModal);
    resetPreviewButton.addEventListener("click", resetTransform);
    swatches.forEach((swatch) => swatch.addEventListener("click", () => setModalBackground(swatch.dataset.bg)));
    closeCompareButton.addEventListener("click", closeCompare);
    compareViewOne.addEventListener("change", updateCompareImages);
    compareViewTwo.addEventListener("change", updateCompareImages);
    compareAlphaOne.addEventListener("input", updateCompareAlpha);
    compareAlphaTwo.addEventListener("input", updateCompareAlpha);
    compareBgButtons.forEach((button) => button.addEventListener("click", () => setCompareBackground(button.dataset.bg)));
    compareFrame.addEventListener("pointermove", updateCompareFromPointer);
    compareFrame.addEventListener("pointerdown", (event) => {{
      updateCompareFromPointer(event);
      compareFrame.setPointerCapture(event.pointerId);
    }});
    compareModal.addEventListener("click", (event) => {{
      if (event.target === compareModal) closeCompare();
    }});
    startFullEvalButton.addEventListener("click", openEvalPanel);
    selectAllSamplesButton.addEventListener("click", () => setAllSamples(true));
    clearAllSamplesButton.addEventListener("click", () => setAllSamples(false));
    if (testPathSelect) testPathSelect.addEventListener("change", updateSelectionCount);
    cancelEvalPanelButton.addEventListener("click", closeEvalPanel);
    confirmStartEvalButton.addEventListener("click", startSelectedEval);
    evalPanel.addEventListener("click", (event) => {{
      if (event.target === evalPanel) closeEvalPanel();
    }});
    runSelect.addEventListener("change", () => {{
      if (runSelect.value) {{
        window.location.href = `/eval/game?run=${{encodeURIComponent(runSelect.value)}}`;
      }}
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape") closeModal();
      if (event.key === "Escape") closeCompare();
      if (event.key === "Escape") closeEvalPanel();
    }});

    renderRows();
  </script>
</body>
</html>"""


def main() -> None:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - exercised only without web extra
        raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

    uvicorn.run("ermbg.web:app", host="127.0.0.1", port=7860, reload=False)


__all__ = ["app", "main"]
