"""Small web UI for ERMBG.

The service keeps the browser flow intentionally narrow: upload one image,
run ``matte_image``, preview the returned RGBA PNG, and download it.
"""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, Response
except ImportError as e:  # pragma: no cover - exercised only without web extra
    raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

from .api import matte_image

ALLOWED_BACKENDS = {"grabcut", "auto", "birefnet"}

app = FastAPI(title="ERMBG Web", version="0.1.0")


def _encode_png(rgba: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
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
    label {
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
    button:disabled, a.download[aria-disabled="true"] {
      opacity: 0.55;
      cursor: not-allowed;
      pointer-events: none;
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
      grid-template-rows: 48px 1fr 56px;
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
    @media (max-width: 760px) {
      header { padding: 0 16px; }
      main {
        grid-template-columns: 1fr;
        padding: 16px;
      }
      .preview {
        min-height: 420px;
        grid-template-rows: auto 1fr 56px;
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
    }
  </style>
</head>
<body>
  <header>
    <h1>ERMBG</h1>
    <span class="status" id="strategy">就绪</span>
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
      <label>
        后端
        <select id="backend" name="backend">
          <option value="grabcut">grabcut</option>
          <option value="auto" selected>auto</option>
          <option value="birefnet">birefnet</option>
        </select>
      </label>
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
      <div class="preview-actions">
        <span class="status" id="meta">RGBA PNG</span>
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
    const canvas = document.getElementById("canvas");
    const download = document.getElementById("download");
    const sourcePreview = document.getElementById("source-preview");
    const sourceFrame = document.getElementById("source-frame");
    const sourceMeta = document.getElementById("source-meta");
    const tabs = Array.from(document.querySelectorAll(".tab"));
    let resultUrl = null;
    let sourceUrl = null;
    let resultImage = null;
    let previewScale = 1;
    let previewPanX = 0;
    let previewPanY = 0;
    let dragStart = null;

    function humanSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }

    function setBusy(isBusy) {
      submit.disabled = isBusy;
      file.disabled = isBusy;
      backend.disabled = isBusy;
      submit.textContent = isBusy ? "处理中" : "抠图";
    }

    function setPreviewBackground(mode) {
      canvas.classList.remove("bg-white", "bg-black", "bg-gray", "bg-green", "bg-blue");
      if (mode !== "checker") canvas.classList.add(`bg-${mode}`);
      tabs.forEach((tab) => {
        tab.setAttribute("aria-selected", String(tab.dataset.bg === mode));
      });
    }

    function resetResult() {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      resultUrl = null;
      resultImage = null;
      resetPreviewTransform();
      canvas.innerHTML = '<span class="empty">结果会显示在这里</span>';
      canvas.classList.remove("has-image", "is-dragging");
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

    function setDownload(blob, name) {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      resultUrl = URL.createObjectURL(blob);
      resetPreviewTransform();
      canvas.innerHTML = "";
      const img = document.createElement("img");
      img.src = resultUrl;
      img.alt = "ERMBG result";
      img.draggable = false;
      img.className = "result-image";
      resultImage = img;
      canvas.classList.add("has-image");
      canvas.appendChild(img);
      applyPreviewTransform();
      download.href = resultUrl;
      download.download = name.replace(/\\.[^.]+$/, "") + "_rgba.png";
      download.setAttribute("aria-disabled", "false");
    }

    file.addEventListener("change", () => {
      resetResult();
      statusEl.textContent = "等待抠图";
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

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!file.files.length) return;
      const formData = new FormData();
      formData.append("file", file.files[0]);
      formData.append("backend", backend.value);
      setBusy(true);
      statusEl.textContent = "正在抠图";
      strategyEl.textContent = backend.value;
      try {
        const response = await fetch("/api/matte", { method: "POST", body: formData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        const blob = await response.blob();
        setDownload(blob, file.files[0].name);
        const strategy = response.headers.get("x-ermbg-strategy") || "done";
        const bg = response.headers.get("x-ermbg-background") || "";
        statusEl.textContent = "完成";
        strategyEl.textContent = bg ? `${strategy} · ${bg}` : strategy;
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    });
  </script>
</body>
</html>"""


@app.post("/api/matte")
def matte_endpoint(
    file: Annotated[UploadFile, File()],
    backend: Annotated[str, Form()] = "grabcut",
) -> Response:
    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {sorted(ALLOWED_BACKENDS)}")

    image = _load_upload_image(file)
    try:
        result = matte_image(image, backend=backend, qa=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e

    png = _encode_png(result.rgba)
    filename = (file.filename or "ermbg").rsplit(".", 1)[0] + "_rgba.png"
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ERMBG-Strategy": result.strategy_name,
            "X-ERMBG-Background": ",".join(str(c) for c in result.background_color),
        },
    )


def main() -> None:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - exercised only without web extra
        raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

    uvicorn.run("ermbg.web:app", host="127.0.0.1", port=7860, reload=False)


__all__ = ["app", "main"]
