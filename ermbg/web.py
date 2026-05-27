"""Small web UI for ERMBG.

The service keeps the browser flow intentionally narrow: upload one image,
run ``matte_image``, preview the returned RGBA PNG, and download it.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw

try:
    from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, Response
except ImportError as e:  # pragma: no cover - exercised only without web extra
    raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

from .api import matte_image
from .candidates import MatteCandidate, generate_matte_candidates
from .local_ownership import generate_local_ownership_candidate
from .slicer import SliceBox, classify_ui_slice, crop_slice, slice_image

ALLOWED_BACKENDS = {"grabcut", "auto", "birefnet"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME_EVAL_ROOT = PROJECT_ROOT / "out" / "vlm_eval_game_qwen_gw_v009_display_safe_20260527"
GAME_EVAL_PREFIX = "vlm_eval_game_qwen_gw_v"
LOCAL_OWNERSHIP_EVAL_PREFIX = "local_ownership_"
GAME_EVAL_RUN_PREFIXES = (
    GAME_EVAL_PREFIX,
    LOCAL_OWNERSHIP_EVAL_PREFIX,
)
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


def _encode_png(rgba: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _load_upload_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from e

    return image.convert("RGBA" if image.mode == "RGBA" else "RGB")


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
    body { margin: 0; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }
    .header-actions { min-width: 0; display: flex; align-items: center; gap: 14px; }
    .nav-link { color: #196f5a; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    main { width: min(1120px, 100%); margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 320px 1fr; gap: 24px; align-items: start; }
    form, .preview { background: #ffffff; border: 1px solid #d9dfd7; border-radius: 8px; }
    form { min-width: 0; padding: 16px; display: grid; gap: 12px; }
    label { display: grid; gap: 8px; font-size: 13px; font-weight: 600; color: #47524c; }
    input, select, button { width: 100%; min-height: 40px; border-radius: 6px; border: 1px solid #b8c1b7; background: #ffffff; color: #1c2320; font: inherit; }
    input[type="file"] { padding: 8px; }
    button, a.download { display: inline-flex; align-items: center; justify-content: center; min-height: 42px; border: 0; border-radius: 6px; background: #196f5a; color: #ffffff; text-decoration: none; font-weight: 700; cursor: pointer; }
    button:disabled, a.download[aria-disabled="true"] { opacity: 0.55; cursor: not-allowed; pointer-events: none; }
    .source-preview { display: none; gap: 10px; }
    .source-preview.is-visible { display: grid; }
    .source-frame { width: 100%; aspect-ratio: 4 / 3; min-height: 148px; display: grid; place-items: center; overflow: hidden; border: 1px solid #d9dfd7; border-radius: 6px; background-color: #eef2ec; background-image: linear-gradient(45deg, #d7dfd4 25%, transparent 25%), linear-gradient(-45deg, #d7dfd4 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d7dfd4 75%), linear-gradient(-45deg, transparent 75%, #d7dfd4 75%); background-position: 0 0, 0 10px, 10px -10px, -10px 0; background-size: 20px 20px; }
    .source-frame img { display: block; width: 100%; height: 100%; object-fit: contain; object-position: center; }
    .source-meta { min-height: auto; font-size: 12px; line-height: 1.4; color: #5d6862; overflow-wrap: anywhere; }
    .preview { min-height: 520px; display: grid; grid-template-rows: 48px 1fr 104px 56px; overflow: hidden; }
    .preview-bar, .preview-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 16px; border-bottom: 1px solid #d9dfd7; }
    .preview-actions { border-top: 1px solid #d9dfd7; border-bottom: 0; }
    .tabs { display: inline-flex; align-items: center; gap: 4px; padding: 3px; border: 1px solid #cfd7cc; border-radius: 6px; background: #f7f9f6; flex-shrink: 0; }
    .tab { width: auto; min-height: 30px; padding: 0 10px; border: 0; border-radius: 4px; background: transparent; color: #47524c; font-size: 12px; font-weight: 700; }
    .tab[aria-selected="true"] { background: #196f5a; color: #ffffff; }
    .status { font-size: 13px; color: #5d6862; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .canvas, .candidate-thumb { background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); }
    .canvas { min-height: 416px; display: grid; place-items: center; padding: 16px; overflow: hidden; touch-action: none; background-position: 0 0, 0 12px, 12px -12px, -12px 0; background-size: 24px 24px; }
    .canvas.has-image { cursor: grab; }
    .canvas.is-dragging { cursor: grabbing; }
    .canvas.bg-white { background: #ffffff; }
    .canvas.bg-black { background: #111514; }
    .canvas.bg-gray { background: #aeb7b1; }
    .canvas.bg-green { background: #00c853; }
    .canvas.bg-blue { background: #4aa3ff; }
    img { max-width: 100%; max-height: 68vh; object-fit: contain; image-rendering: auto; }
    .result-image { transform-origin: center center; user-select: none; pointer-events: none; will-change: transform; }
    .empty { color: #6a746f; font-size: 14px; }
    .candidate-panel { min-height: 104px; display: grid; grid-template-columns: auto 1fr; align-items: center; gap: 12px; padding: 12px 16px; border-top: 1px solid #d9dfd7; background: #fbfcfa; }
    .candidate-title { font-size: 12px; font-weight: 800; color: #47524c; white-space: nowrap; }
    .candidate-list { min-width: 0; display: flex; gap: 8px; overflow-x: auto; padding: 2px; }
    .candidate-tab { width: 92px; min-width: 92px; min-height: 76px; display: grid; grid-template-rows: 48px auto; gap: 5px; padding: 5px; border: 1px solid #cfd7cc; border-radius: 6px; background: #ffffff; color: #47524c; cursor: pointer; }
    .candidate-tab[aria-selected="true"] { border-color: #196f5a; box-shadow: 0 0 0 2px rgba(25, 111, 90, 0.18); color: #1c2320; }
    .candidate-thumb { width: 100%; height: 48px; display: grid; place-items: center; overflow: hidden; border-radius: 4px; background-position: 0 0, 0 6px, 6px -6px, -6px 0; background-size: 12px 12px; }
    .candidate-thumb img { width: 100%; height: 100%; object-fit: contain; }
    .candidate-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; font-weight: 800; line-height: 1.1; }
    @media (max-width: 760px) { header { padding: 0 16px; } main { grid-template-columns: 1fr; padding: 16px; } .preview { min-height: 420px; grid-template-rows: auto 1fr 104px 56px; } .preview-bar { min-height: 84px; align-items: stretch; flex-direction: column; justify-content: center; padding: 10px 16px; } .tabs { width: 100%; overflow-x: auto; } .canvas { min-height: 312px; } .candidate-panel { grid-template-columns: 1fr; align-items: stretch; gap: 8px; } }
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
      <div class="source-preview" id="source-preview" aria-live="polite">
        <div class="source-frame" id="source-frame"><span class="empty">选择图片后显示预览</span></div>
        <div class="source-meta" id="source-meta">未选择图片</div>
      </div>
      <label>后端<select id="backend" name="backend"><option value="grabcut">grabcut</option><option value="auto" selected>auto</option><option value="birefnet">birefnet</option></select></label>
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
      <div class="canvas" id="canvas"><span class="empty">结果会显示在这里</span></div>
      <div class="candidate-panel" aria-label="候选结果">
        <span class="candidate-title">候选</span>
        <div class="candidate-list" id="candidate-list" role="tablist" aria-label="候选缩略图"><span class="empty">候选会显示在这里</span></div>
      </div>
      <div class="preview-actions"><span class="status" id="meta">RGBA PNG</span><a class="download" id="download" aria-disabled="true" download="ermbg_rgba.png">下载 PNG</a></div>
    </section>
  </main>
  <script>
    const form = document.getElementById("matte-form");
    const file = document.getElementById("file");
    const backend = document.getElementById("backend");
    const submit = document.getElementById("submit");
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

    function humanSize(bytes) { if (bytes < 1024) return `${bytes} B`; if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`; return `${(bytes / 1024 / 1024).toFixed(2)} MB`; }
    function setBusy(isBusy) { submit.disabled = isBusy; file.disabled = isBusy; backend.disabled = isBusy; submit.textContent = isBusy ? "处理中" : "抠图"; }
    function setPreviewBackground(mode) { canvas.classList.remove("bg-white", "bg-black", "bg-gray", "bg-green", "bg-blue"); if (mode !== "checker") canvas.classList.add(`bg-${mode}`); tabs.forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.bg === mode))); }
    function resetPreviewTransform() { previewScale = 1; previewPanX = 0; previewPanY = 0; dragStart = null; applyPreviewTransform(); }
    function applyPreviewTransform() { if (resultImage) resultImage.style.transform = `translate(${previewPanX}px, ${previewPanY}px) scale(${previewScale})`; }
    function resetResult() { candidates.forEach((candidate) => { if (candidate.revoke) URL.revokeObjectURL(candidate.url); }); candidates = []; activeCandidateIndex = -1; resultImage = null; resetPreviewTransform(); canvas.innerHTML = '<span class="empty">结果会显示在这里</span>'; canvas.classList.remove("has-image", "is-dragging"); candidateList.innerHTML = '<span class="empty">候选会显示在这里</span>'; metaEl.textContent = "RGBA PNG"; download.removeAttribute("href"); download.setAttribute("aria-disabled", "true"); }
    function renderCandidateTabs() { candidateList.innerHTML = ""; if (!candidates.length) { candidateList.innerHTML = '<span class="empty">候选会显示在这里</span>'; return; } candidates.forEach((candidate, index) => { const button = document.createElement("button"); button.className = "candidate-tab"; button.type = "button"; button.role = "tab"; button.setAttribute("aria-selected", String(index === activeCandidateIndex)); button.dataset.index = String(index); button.title = candidate.label; const thumb = document.createElement("span"); thumb.className = "candidate-thumb"; const img = document.createElement("img"); img.src = candidate.url; img.alt = `${candidate.label} 缩略图`; thumb.appendChild(img); const label = document.createElement("span"); label.className = "candidate-name"; label.textContent = candidate.label; button.appendChild(thumb); button.appendChild(label); button.addEventListener("click", () => setActiveCandidate(index)); candidateList.appendChild(button); }); }
    function setActiveCandidate(index) { if (index < 0 || index >= candidates.length) return; const candidate = candidates[index]; activeCandidateIndex = index; resetPreviewTransform(); canvas.innerHTML = ""; const img = document.createElement("img"); img.src = candidate.url; img.alt = candidate.label; img.draggable = false; img.className = "result-image"; resultImage = img; canvas.classList.add("has-image"); canvas.appendChild(img); applyPreviewTransform(); download.href = candidate.url; download.download = candidate.downloadName; download.setAttribute("aria-disabled", "false"); metaEl.textContent = candidate.meta; renderCandidateTabs(); }
    function setCandidatePayloads(payload, name) { resetResult(); const stem = name.replace(/\\.[^.]+$/, ""); candidates = (payload.candidates || []).map((candidate, index) => ({ url: candidate.rgba, revoke: false, label: candidate.label || `候选 ${index + 1}`, selected: candidate.selected === true, meta: `候选 ${index + 1} / ${payload.candidates.length} · ${candidate.kind || "RGBA PNG"}`, downloadName: candidate.filename || `${stem}_${candidate.id || `candidate_${index + 1}`}.png` })); if (!candidates.length) throw new Error("没有可显示的候选结果"); const selectedIndex = candidates.findIndex((candidate) => candidate.selected); setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0); }
    function dataUrlToFile(dataUrl, filename) { const [header, base64] = dataUrl.split(","); const mime = (header.match(/data:(.*);base64/) || [])[1] || "image/png"; const binary = atob(base64); const bytes = new Uint8Array(binary.length); for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i); return new File([bytes], filename, { type: mime }); }
    function loadPendingSlice() { const raw = sessionStorage.getItem("ermbgPendingSlice"); if (!raw) return; sessionStorage.removeItem("ermbgPendingSlice"); try { const pending = JSON.parse(raw); const sliceFile = dataUrlToFile(pending.rgb, pending.filename || "slice.png"); const transfer = new DataTransfer(); transfer.items.add(sliceFile); file.files = transfer.files; sourcePreview.classList.add("is-visible"); sourceFrame.innerHTML = ""; const img = document.createElement("img"); img.src = pending.rgb; img.alt = "切图预览"; sourceFrame.appendChild(img); sourceMeta.textContent = `${sliceFile.name} · ${pending.meta || "来自切图"}`; statusEl.textContent = "已载入切图，可直接抠图"; } catch (error) { statusEl.textContent = "切图载入失败"; } }

    file.addEventListener("change", () => { resetResult(); statusEl.textContent = "等待抠图"; strategyEl.textContent = backend.value; if (sourceUrl) URL.revokeObjectURL(sourceUrl); if (!file.files.length) { sourceUrl = null; sourcePreview.classList.remove("is-visible"); sourceFrame.innerHTML = '<span class="empty">选择图片后显示预览</span>'; sourceMeta.textContent = "未选择图片"; return; } const selected = file.files[0]; sourceUrl = URL.createObjectURL(selected); sourcePreview.classList.add("is-visible"); sourceFrame.innerHTML = ""; const img = document.createElement("img"); img.src = sourceUrl; img.alt = "上传图片预览"; img.onload = () => { sourceMeta.textContent = `${selected.name} · ${img.naturalWidth}x${img.naturalHeight} · ${humanSize(selected.size)}`; }; img.onerror = () => { sourceMeta.textContent = `${selected.name} · 无法预览 · ${humanSize(selected.size)}`; }; sourceFrame.appendChild(img); });
    tabs.forEach((tab) => tab.addEventListener("click", () => setPreviewBackground(tab.dataset.bg)));
    canvas.addEventListener("wheel", (event) => { if (!resultImage) return; event.preventDefault(); const rect = canvas.getBoundingClientRect(); const centerX = rect.left + rect.width / 2; const centerY = rect.top + rect.height / 2; const pointerX = event.clientX - centerX; const pointerY = event.clientY - centerY; const previousScale = previewScale; const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12; previewScale = Math.min(8, Math.max(0.2, previewScale * factor)); previewPanX = pointerX - ((pointerX - previewPanX) * previewScale) / previousScale; previewPanY = pointerY - ((pointerY - previewPanY) * previewScale) / previousScale; applyPreviewTransform(); }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => { if (!resultImage) return; dragStart = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: previewPanX, panY: previewPanY }; canvas.setPointerCapture(event.pointerId); canvas.classList.add("is-dragging"); });
    canvas.addEventListener("pointermove", (event) => { if (!dragStart || dragStart.pointerId !== event.pointerId) return; previewPanX = dragStart.panX + event.clientX - dragStart.x; previewPanY = dragStart.panY + event.clientY - dragStart.y; applyPreviewTransform(); });
    function endDrag(event) { if (!dragStart || dragStart.pointerId !== event.pointerId) return; dragStart = null; canvas.classList.remove("is-dragging"); }
    canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag); canvas.addEventListener("dblclick", () => resetPreviewTransform());
    form.addEventListener("submit", async (event) => { event.preventDefault(); if (!file.files.length) return; const formData = new FormData(); formData.append("file", file.files[0]); formData.append("backend", backend.value); setBusy(true); statusEl.textContent = "正在抠图"; strategyEl.textContent = backend.value; try { const response = await fetch("/api/matte-candidates", { method: "POST", body: formData }); if (!response.ok) { let message = "处理失败"; try { const payload = await response.json(); message = payload.detail || message; } catch (_) {} throw new Error(message); } const payload = await response.json(); setCandidatePayloads(payload, file.files[0].name); const strategy = payload.strategy || "done"; const bg = Array.isArray(payload.background) ? payload.background.join(",") : ""; statusEl.textContent = "完成"; strategyEl.textContent = bg ? `${strategy} · ${bg}` : strategy; } catch (error) { statusEl.textContent = error.message; } finally { setBusy(false); } });
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
      overflow: hidden;
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
      width: 100%;
      height: 100%;
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
          <option value="grabcut">grabcut</option>
          <option value="auto" selected>auto</option>
          <option value="birefnet">birefnet</option>
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
        setCandidatePayloads(payload, crop.filename);
        const strategy = payload.strategy || "done";
        const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
        statusEl.textContent = "完成";
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
      resetResult();
      statusEl.textContent = task.value === "slice" ? "等待切图标注" : "等待抠图";
      strategyEl.textContent = backend.value;
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
      if (!file.files.length) {
        sourceUrl = null;
        sourcePreview.classList.remove("is-visible");
        sourceFrame.innerHTML = '<span class="empty">选择图片后显示预览</span>';
        sourceMeta.textContent = "未选择图片";
        return;
      }

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
          setCandidatePayloads(payload, file.files[0].name);
          const strategy = payload.strategy || "done";
          const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
          statusEl.textContent = "完成";
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
      hasPreview = false;
      currentCrops = [];
      selectedCrop = null;
      selectedActions.classList.remove("is-visible");
      clearSliceState();
      confirmButton.disabled = true;
      preview.innerHTML = '<span class="empty">自动标注会显示在这里</span>';
      list.innerHTML = '<span class="empty">切图列表会显示在这里</span>';
      if (file.files.length) {
        runAnnotate();
      } else {
        statusEl.textContent = "等待上传";
      }
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


def _candidate_payload(candidate: MatteCandidate, stem: str) -> dict[str, object]:
    debug = candidate.debug
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


def _slice_preview_payload(image_rgb: np.ndarray, stem: str, min_area: int, padding: int) -> dict[str, object]:
    result = slice_image(image_rgb, min_area=min_area, padding=padding)
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


def _slice_crop_payloads(image_rgb: np.ndarray, stem: str, min_area: int, padding: int) -> dict[str, object]:
    result = slice_image(image_rgb, min_area=min_area, padding=padding)
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


@app.post("/api/matte")
def matte_endpoint(
    file: Annotated[UploadFile, File()],
    backend: Annotated[str, Form()] = "grabcut",
) -> Response:
    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {sorted(ALLOWED_BACKENDS)}")

    image = _load_upload_image(file)
    try:
        result = matte_image(image, backend=backend, qa=False, shadow_mode="off")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e

    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected_rgba = result.rgba
    local_ownership_used = False
    try:
        local_candidate = generate_local_ownership_candidate(
            image_rgb,
            result.rgba,
            result.background_color,
            backend=backend,
            soft_mask=result.debug.get("soft_mask"),
            shadow_mode="off",
        )
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
    backend: Annotated[str, Form()] = "grabcut",
) -> dict[str, object]:
    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {sorted(ALLOWED_BACKENDS)}")

    image = _load_upload_image(file)
    try:
        result = matte_image(image, backend=backend, qa=False, shadow_mode="off")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e

    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    candidates = generate_matte_candidates(image_rgb, result.rgba, result.background_color)
    try:
        local_candidate = generate_local_ownership_candidate(
            image_rgb,
            result.rgba,
            result.background_color,
            backend=backend,
            soft_mask=result.debug.get("soft_mask"),
            shadow_mode="off",
        )
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
        "candidates": [_candidate_payload(candidate, stem) for candidate in candidates],
    }


@app.post("/api/slice")
def slice_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 2,
    transparent: Annotated[bool, Form()] = False,
) -> Response:
    image = _load_upload_image(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    try:
        result = slice_image(image_rgb, min_area=min_area, padding=padding)
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
    image = _load_upload_image(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    try:
        return _slice_preview_payload(image_rgb, stem, min_area=min_area, padding=padding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slice preview failed: {e}") from e


@app.post("/api/slice-crops")
def slice_crops_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 2,
) -> dict[str, object]:
    image = _load_upload_image(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    try:
        return _slice_crop_payloads(image_rgb, stem, min_area=min_area, padding=padding)
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
    case_path = PROJECT_ROOT / "samples" / "vlm_eval_game" / case_id / "case.json"
    if case_path.exists():
        payload = _load_json(case_path)
        if isinstance(payload, dict):
            paths = {
                variant: path
                for variant in ("white", "green")
                if isinstance(path := payload.get(variant), str)
            }
            if paths:
                return paths
    return {
        "white": f"samples/vlm_eval_game/{case_id}/white.png",
        "green": f"samples/vlm_eval_game/{case_id}/green.png",
    }


def _sample_variant_from_path(path_value: object) -> str | None:
    if not isinstance(path_value, str):
        return None
    stem = Path(path_value).stem.lower()
    if stem in {"white", "green"}:
        return stem
    return None


def _game_sample_ids() -> dict[str, str]:
    manifest_path = PROJECT_ROOT / "samples" / "vlm_eval_game" / "manifest.json"
    manifest = _load_json(manifest_path)
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
    manifest_path = PROJECT_ROOT / "samples" / "vlm_eval_game" / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = _load_json(manifest_path)
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list):
        return []
    samples: list[dict[str, object]] = []
    for index, item in enumerate(cases, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        sample_id = item.get("sample_id")
        sample_id = sample_id if isinstance(sample_id, str) else f"G{index:02d}"
        samples.append(
            {
                "sampleId": sample_id,
                "caseId": item["id"],
                "category": item.get("category", ""),
                "primaryAmbiguity": item.get("primary_ambiguity", ""),
            }
        )
    return samples


def _game_report_path(root: Path) -> Path | None:
    for name in ("vlm_qwen", "vlm_openai", "local_ownership"):
        path = root / name / "eval_report.json"
        if path.exists():
            return path
    return None


def _game_vlm_root(root: Path) -> Path:
    report_path = _game_report_path(root)
    if report_path is not None:
        return report_path.parent
    return root / "vlm_qwen"


def _game_eval_root_has_data(root: Path) -> bool:
    if _game_report_path(root) is not None:
        return True
    return _game_matte_summary_path(root) is not None


def _game_eval_root_is_complete(root: Path) -> bool:
    report_path = _game_report_path(root)
    if report_path is None:
        return False
    report = _load_json(report_path)
    if not isinstance(report, dict):
        return False
    try:
        return int(report.get("case_count", 0)) >= 18
    except (TypeError, ValueError):
        return False


def _game_eval_runs(selected_root: Path | None = None) -> list[dict[str, object]]:
    out_root = PROJECT_ROOT / "out"
    roots = [
        path
        for prefix in GAME_EVAL_RUN_PREFIXES
        for path in sorted(out_root.glob(f"{prefix}*"))
        if path.is_dir() and _game_eval_root_has_data(path)
    ]
    if DEFAULT_GAME_EVAL_ROOT.exists() and DEFAULT_GAME_EVAL_ROOT not in roots and _game_eval_root_has_data(DEFAULT_GAME_EVAL_ROOT):
        roots.insert(0, DEFAULT_GAME_EVAL_ROOT)

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


def _validate_game_eval_run_id(run_id: str) -> None:
    if (
        "/" in run_id
        or "\\" in run_id
        or run_id.startswith(".")
        or not any(run_id.startswith(prefix) for prefix in GAME_EVAL_RUN_PREFIXES)
    ):
        raise HTTPException(status_code=404, detail="Game eval run not found.")


def _game_eval_run_path(run_id: str) -> Path:
    _validate_game_eval_run_id(run_id)
    root = (PROJECT_ROOT / "out" / run_id).resolve()
    if not _is_relative_to(root, (PROJECT_ROOT / "out").resolve()):
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    return root


def _next_game_eval_run_id() -> str:
    out_root = PROJECT_ROOT / "out"
    version_re = re.compile(rf"^{re.escape(GAME_EVAL_PREFIX)}(\d+)")
    versions = []
    for path in out_root.glob(f"{GAME_EVAL_PREFIX}*"):
        match = version_re.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{GAME_EVAL_PREFIX}{version:03d}_display_safe_{stamp}"


def _game_eval_expected_case_count() -> int:
    manifest_path = PROJECT_ROOT / "samples" / "vlm_eval_game" / "manifest.json"
    if not manifest_path.exists():
        return 18
    manifest = _load_json(manifest_path)
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if isinstance(cases, list) and cases:
        return len(cases) * 2
    return 18


def _game_eval_batch_progress(
    root: Path,
    report_path: Path | None,
    *,
    prefer_report_total: bool = False,
    expected_total: int | None = None,
) -> dict[str, object]:
    del root
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
        if isinstance(report, dict):
            try:
                report_total = int(report.get("case_count", 0))
                total = report_total if prefer_report_total and report_total > 0 else max(total, report_total)
            except (TypeError, ValueError):
                pass
    percent = 0 if total <= 0 else round(min(100.0, completed * 100.0 / total), 1)
    return {
        "completed": completed,
        "total": total,
        "ok": ok,
        "errors": errors,
        "percent": percent,
        "reportPath": str(report_path.relative_to(PROJECT_ROOT)) if report_path is not None else None,
    }


def _game_eval_batch_status(run_id: str) -> dict[str, object]:
    root = _game_eval_run_path(run_id)
    report_path = _game_report_path(root)
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
    elif any(not re.fullmatch(r"G\d{2}", sample_id) for sample_id in sample_ids):
        raise HTTPException(status_code=400, detail="sample_ids must look like G01.")
    deduped: list[str] = []
    for sample_id in sample_ids:
        if sample_id not in deduped:
            deduped.append(sample_id)
    if raw and not deduped:
        raise HTTPException(status_code=400, detail="Select at least one sample.")
    return deduped


def _start_game_eval_batch(sample_ids: list[str] | None = None) -> dict[str, object]:
    selected_sample_ids = list(sample_ids or [])
    run_id = _next_game_eval_run_id()
    out_dir = PROJECT_ROOT / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "web_batch.log"
    script_path = PROJECT_ROOT / "scripts" / "09_game_eval_qwen_batch.py"
    command = [
        sys.executable,
        str(script_path),
        "--out-dir",
        str(out_dir),
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
        "sample_ids": selected_sample_ids,
    }
    (out_dir / "web_launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False), encoding="utf-8")
    with _GAME_EVAL_JOBS_LOCK:
        _GAME_EVAL_JOBS[run_id] = {
            "process": process,
            "log": log_path,
            "sample_ids": selected_sample_ids,
            "expected_total": len(selected_sample_ids) * 2 if selected_sample_ids else _game_eval_expected_case_count(),
        }
    return _game_eval_batch_status(run_id)


def _default_game_eval_root() -> Path:
    roots = [
        path
        for prefix in GAME_EVAL_RUN_PREFIXES
        for path in sorted((PROJECT_ROOT / "out").glob(f"{prefix}*"), reverse=True)
        if path.is_dir() and _game_eval_root_has_data(path)
    ]
    complete_roots = [root for root in roots if _game_eval_root_is_complete(root)]
    if complete_roots:
        return complete_roots[0]
    if roots:
        return roots[0]
    if DEFAULT_GAME_EVAL_ROOT.is_dir() and _game_eval_root_has_data(DEFAULT_GAME_EVAL_ROOT):
        return DEFAULT_GAME_EVAL_ROOT
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
    variant = row.get("sample_variant")
    if isinstance(variant, str):
        return _game_vlm_root(root) / case_id / variant
    return _game_vlm_root(root) / case_id


def _case_matte_url(out_dir: Path, sample_variant: str, summary: dict[str, object]) -> str | None:
    for value in (summary.get("rgba"), summary.get("matte"), summary.get("output")):
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in (f"{sample_variant}_rgba.png", "rgba.png"):
        url = _image_url(out_dir / name)
        if url:
            return url
    matches = sorted(out_dir.glob("*_rgba.png"))
    return _image_url(matches[0]) if matches else None


def _game_region_url(root: Path, case_id: str, sample_variant: str | None = None) -> str:
    base = f"/eval/game/regions/{quote(case_id, safe='')}"
    params: list[str] = []
    if sample_variant:
        params.append(f"variant={quote(sample_variant, safe='')}")
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
        active_variant = _sample_variant_from_path(input_path) or "green"
        sample_paths = _game_sample_paths(case_id)
        shadow_detected = bool(row.get("shadow_detected", False))
        shadow_pixels = int(row.get("shadow_pixels", 0) or 0)
        strategy = str(row.get("strategy", ""))
        matte_url = _case_matte_url(out_dir, active_variant, row)
        candidate = {
            "id": "matte",
            "label": "matte result",
            "selected": True,
            "tools": [strategy] if strategy else [],
            "reason": f"shadow={shadow_detected}, pixels={shadow_pixels}",
            "url": matte_url,
        }

        for sample_variant, sample_path in sample_paths.items():
            is_active_run = sample_variant == active_variant
            sample_code = f"{sample_id}-{sample_variant[:1].upper()}"
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": sample_code,
                    "sampleVariant": sample_variant,
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
                    "matteUrl": matte_url if is_active_run else None,
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


def _game_eval_data(root: Path = DEFAULT_GAME_EVAL_ROOT) -> dict[str, object]:
    report_path = _game_report_path(root)
    if report_path is None:
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
        row_variant = row.get("sample_variant")
        active_variant = (
            row_variant
            if isinstance(row_variant, str) and row_variant in sample_paths
            else _sample_variant_from_path(summary.get("input")) or "green"
        )
        variants = [active_variant] if isinstance(row_variant, str) else list(sample_paths)
        for sample_variant in variants:
            sample_path = sample_paths.get(sample_variant, str(summary.get("input", "")))
            is_active_run = sample_variant == active_variant
            sample_code = f"{sample_id}-{sample_variant[:1].upper()}"
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": sample_code,
                    "sampleVariant": sample_variant,
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
                    "regionsUrl": _game_region_url(root, case_id, sample_variant) if is_active_run else None,
                    "matteUrl": _image_url(summary.get("rgba") or row.get("protected_rgba") or row.get("rgba") or root / "matte" / case_id / "rgba.png")
                    if is_active_run
                    else None,
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
    return _start_game_eval_batch(_selected_game_eval_sample_ids(payload))


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
    variant: str | None = Query(default=None),
) -> Response:
    root = _game_eval_root(run)
    rows = _game_report_rows(root)
    row = next(
        (
            item
            for item in rows
            if item.get("case_id") == case_id
            and (variant is None or item.get("sample_variant") == variant)
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
    if not isinstance(input_path_value, str):
        raise HTTPException(status_code=404, detail="Case input image not found.")
    input_path = _resolve_project_path(input_path_value)
    if not _is_relative_to(input_path, (PROJECT_ROOT / "samples").resolve()) or not input_path.exists():
        raise HTTPException(status_code=404, detail="Case input image not found.")
    regions = _candidate_regions(_candidate_result_items(out_dir / "candidate_results.json"))
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
    main {{ width: min(1760px, 100%); margin: 0 auto; padding: 18px 20px 28px; }}
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
      width: max(100%, 1220px);
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
    .case-col {{ width: 260px; }}
    .original-col {{ width: 145px; }}
    .regions-col {{ width: 155px; }}
    .preview-col {{ width: 132px; }}
    th.case-col, td:first-child {{
      position: sticky;
      left: 0;
      z-index: 4;
      background: #ffffff;
      box-shadow: 1px 0 0 #e2e8df;
    }}
    th.case-col {{ z-index: 6; background: #fbfcfa; }}
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
      grid-template-rows: auto auto 1fr auto;
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
    .sample-list {{ min-height: 0; overflow: auto; padding: 8px 16px; }}
    .sample-option {{
      display: grid;
      grid-template-columns: 22px 72px 1fr;
      gap: 10px;
      align-items: start;
      min-height: 44px;
      padding: 10px 0;
      border-bottom: 1px solid #edf1ea;
      color: #17201c;
      font-size: 13px;
      font-weight: 700;
    }}
    .sample-option:last-child {{ border-bottom: 0; }}
    .sample-option input {{ width: 16px; height: 16px; min-height: 0; margin: 2px 0 0; }}
    .sample-detail {{ color: #5f6c66; font-size: 12px; font-weight: 600; line-height: 1.35; overflow-wrap: anywhere; }}
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
            <th class="case-col">case</th>
            <th class="original-col">original</th>
            <th class="regions-col">regions</th>
            <th class="preview-col">checker</th>
            <th class="preview-col">white</th>
            <th class="preview-col">black</th>
            <th class="preview-col">purple</th>
            <th class="preview-col">green ref</th>
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
      <div class="sample-list" id="sample-list"></div>
      <div class="eval-actions">
        <button type="button" id="cancel-eval-panel">取消</button>
        <button class="primary" type="button" id="confirm-start-eval">开始测试</button>
      </div>
    </section>
  </div>
  <div class="modal" id="modal" aria-hidden="true">
    <div class="modal-bar">
      <div class="modal-title" id="modal-title"></div>
      <div class="modal-actions" aria-label="preview controls">
        <button class="swatch bg-checker" type="button" data-bg="checker" title="棋盘背景" aria-label="棋盘背景"></button>
        <button class="swatch bg-white" type="button" data-bg="white" title="白底" aria-label="白底"></button>
        <button class="swatch bg-black" type="button" data-bg="black" title="黑底" aria-label="黑底"></button>
        <button class="swatch bg-purple" type="button" data-bg="purple" title="紫底" aria-label="紫底"></button>
        <button class="swatch bg-green" type="button" data-bg="green" title="绿幕参照" aria-label="绿幕参照"></button>
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
    const backgrounds = ["checker", "white", "black", "purple", "green"];
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
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let dragStart = null;
    let activeBatchStatusUrl = "";

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

    function makePreview(src, label, bg) {{
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
      button.addEventListener("click", () => openModal(src, label, bg));
      return button;
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
        `vlm: ${{text(data.vlmRoot)}}`,
        `matte: ${{text(data.matteRoot)}}`,
      ].forEach((item) => {{
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = item;
        summaryEl.appendChild(pill);
      }});

      rowsEl.innerHTML = "";
      data.cases.forEach((caseItem) => {{
        const candidates = caseItem.candidates && caseItem.candidates.length ? caseItem.candidates : [null];

        candidates.forEach((candidate, candidateIndex) => {{
          const row = document.createElement("tr");
          const caseCell = document.createElement("td");
          const sampleCode = document.createElement("div");
          sampleCode.className = "sample-code";
          sampleCode.textContent = caseItem.sampleCode || "G??";
          const title = document.createElement("div");
          title.className = "case-name";
          title.textContent = caseItem.caseId;
          const sampleBadge = document.createElement("div");
          sampleBadge.className = "sample-badge";
          sampleBadge.textContent = `${{caseItem.sampleVariant || "sample"}} · ${{caseItem.runStatus || "unknown"}}`;
          const candidateLabel = document.createElement("div");
          candidateLabel.className = "candidate-label";
          candidateLabel.textContent = candidate
            ? candidate.label || candidate.id || `candidate ${{candidateIndex + 1}}`
            : "not run";
          if (candidate && candidate.selected) {{
            const selected = document.createElement("span");
            selected.className = "selected-mark";
            selected.textContent = "selected";
            candidateLabel.appendChild(selected);
          }}
          const meta = document.createElement("div");
          meta.className = "case-meta";
          const hitClass = caseItem.runStatus === "not-run" ? "pending" : (caseItem.expectedHit ? "hit" : "miss");
          const hitText = caseItem.runStatus === "not-run" ? "not run" : `expected ${{caseItem.expectedHit ? "hit" : "miss"}}`;
          const harmfulText = caseItem.harmfulToolSelected ? ` · harmful: ${{(caseItem.harmfulTools || []).join(", ")}}` : "";
          const shadowText = caseItem.shadowPolicyRequired
            ? ` · shadow: ${{caseItem.shadowPolicyHit ? "hit" : "miss"}} (${{text(caseItem.shadowCandidateCount)}})`
            : "";
          meta.innerHTML = `
            <span>verdict: ${{text(caseItem.verdict)}} · <span class="${{hitClass}}">${{hitText}}</span></span>
            <span>regions: ${{text(caseItem.regionCount)}}</span>
            <span>${{text(caseItem.primaryAmbiguity)}}</span>
            <span>${{countsText(caseItem.counts)}}</span>
            <span class="tools">tools: ${{candidate ? (candidate.tools || []).join(", ") || "—" : "—"}}${{harmfulText}}${{shadowText}}</span>
          `;
          caseCell.appendChild(sampleCode);
          caseCell.appendChild(title);
          caseCell.appendChild(sampleBadge);
          caseCell.appendChild(candidateLabel);
          caseCell.appendChild(meta);
          row.appendChild(caseCell);

          const originalCell = document.createElement("td");
          if (caseItem.originalUrl) {{
            originalCell.appendChild(makePreview(caseItem.originalUrl, `${{caseItem.sampleCode}} · ${{caseItem.caseId}} original`, "checker"));
          }}
          row.appendChild(originalCell);

          const regionsCell = document.createElement("td");
          if (caseItem.regionsUrl) {{
            regionsCell.appendChild(makePreview(caseItem.regionsUrl, `${{caseItem.sampleCode}} · ${{caseItem.caseId}} regions`, "checker"));
          }} else {{
            const empty = document.createElement("div");
            empty.className = "empty-cell";
            empty.textContent = "not run";
            regionsCell.appendChild(empty);
          }}
          row.appendChild(regionsCell);

          backgrounds.forEach((bg) => {{
            const cell = document.createElement("td");
            const previewUrl = candidate && candidate.url ? candidate.url : (caseItem.runStatus === "ran" ? caseItem.matteUrl : "");
            if (previewUrl) {{
              cell.appendChild(
                makePreview(
                  previewUrl,
                  `${{caseItem.sampleCode}} · ${{caseItem.caseId}} · ${{caseItem.sampleVariant}} · ${{candidate ? (candidate.label || candidate.id) : "matte"}} · ${{bg}}`,
                  bg,
                ),
              );
            }} else {{
              const empty = document.createElement("div");
              empty.className = "empty-cell";
              empty.textContent = "not run";
              cell.appendChild(empty);
            }}
            row.appendChild(cell);
          }});
          rowsEl.appendChild(row);
        }});
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

    function updateSelectionCount() {{
      const selected = selectedSampleIds().length;
      const total = sampleCheckboxes().length;
      selectionCountEl.textContent = `${{selected}}/${{total}}`;
      confirmStartEvalButton.disabled = selected === 0;
    }}

    function renderSampleList() {{
      const samples = Array.isArray(data.samples) ? data.samples : [];
      sampleList.innerHTML = "";
      samples.forEach((sample) => {{
        const label = document.createElement("label");
        label.className = "sample-option";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = sample.sampleId;
        checkbox.checked = true;
        checkbox.addEventListener("change", updateSelectionCount);
        const code = document.createElement("span");
        code.textContent = sample.sampleId;
        const detail = document.createElement("span");
        detail.className = "sample-detail";
        detail.textContent = `${{sample.caseId || ""}} · ${{sample.category || ""}} · ${{sample.primaryAmbiguity || ""}}`;
        label.appendChild(checkbox);
        label.appendChild(code);
        label.appendChild(detail);
        sampleList.appendChild(label);
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
      if (!sampleIds.length) return;
      setBatchStatus("启动中", true);
      setBatchProgress({{ percent: 0 }});
      try {{
        closeEvalPanel();
        const response = await fetch("/eval/game/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ sample_ids: sampleIds }}),
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
    startFullEvalButton.addEventListener("click", openEvalPanel);
    selectAllSamplesButton.addEventListener("click", () => setAllSamples(true));
    clearAllSamplesButton.addEventListener("click", () => setAllSamples(false));
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
