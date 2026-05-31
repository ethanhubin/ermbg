"""Tests for the ERMBG web service."""

from __future__ import annotations

import base64
import json
import os
import zipfile
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from ermbg.api import MatteResponse
from ermbg.candidates import MatteCandidate
from ermbg.web import app


def _png_bytes() -> bytes:
    img = np.full((16, 16, 3), [0, 200, 0], dtype=np.uint8)
    img[5:11, 5:11] = [220, 30, 30]
    buf = BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _mask_png_bytes() -> bytes:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    buf = BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _ring_png_bytes() -> bytes:
    h, w = 64, 64
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    img[(r <= 22) & (r >= 9)] = (230, 0, 0)
    buf = BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_index_serves_upload_ui():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "file" in response.text
    assert "api/matte-candidates" in response.text
    assert "source-preview" in response.text
    assert "candidate-list" in response.text
    assert 'id="preview-panel"' in response.text
    assert 'aria-selected="true" data-view="mask">遮罩</button>' in response.text
    assert 'aria-selected="false" data-bg="checker">棋盘</button>' in response.text
    assert 'id="mask-toolbar"' in response.text
    assert response.text.index('id="preview-panel"') < response.text.index('id="source-preview"')
    assert 'id="source-preview"' not in response.text.split('<label class="inline-label">后端')[0]
    assert 'href="/slice">切图</a>' in response.text
    assert '"/api/slice-preview"' not in response.text
    assert '"/api/slice-crops"' not in response.text
    assert "confirm-slices" not in response.text
    assert "候选缩略图" in response.text
    assert 'href="/eval/game"' in response.text
    assert 'role="tablist"' in response.text
    assert ".source-frame img { position: absolute; z-index: 1; left: 50%; top: 50%; display: block; width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: contain;" in response.text
    assert "translate(-50%, -50%) translate(${maskPanX}px, ${maskPanY}px) scale(${maskScale})" in response.text
    assert ".mask-overlay { position: absolute; z-index: 2;" in response.text
    assert ".source-frame { position: relative; width: 100%; aspect-ratio: 4 / 3; max-height: 360px; min-height: 148px; display: grid; place-items: center; border:" in response.text
    assert "height: calc(100vh - 56px)" in response.text
    assert ".preview { min-height: 0; height: 100%; display: grid; grid-template-rows: 48px auto minmax(0, 1fr) 104px 56px;" in response.text
    assert ".candidate-panel { height: 104px; min-height: 104px; max-height: 104px;" in response.text
    assert ".preview-actions { height: 56px; min-height: 56px; max-height: 56px;" in response.text
    assert ".preview-statuses { min-width: 0; flex: 1 1 auto; overflow: hidden; }" in response.text
    assert ".preview-statuses .status { display: block; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }" in response.text
    assert ".preview.is-mask-mode { grid-template-rows: 48px auto minmax(0, 1fr) 0 56px; }" in response.text
    assert ".preview.is-mask-mode .candidate-panel { display: none;" in response.text
    assert ".canvas { min-height: 0; height: 100%;" in response.text
    assert ".canvas.is-mask-mode { padding: 0; }" in response.text
    assert ".canvas.is-mask-mode .source-frame { border: 0; border-radius: 0; }" in response.text
    assert ".canvas img { max-width: 100%; max-height: 100%; }" in response.text
    assert "let maskScale = 1;" in response.text
    assert "applyMaskTransform()" in response.text
    assert 'if (activeView === "mask") { if (!sourceImage) return; event.preventDefault();' in response.text
    assert 'file.addEventListener("change", () => { if (!file.files.length) return; resetResult();' in response.text
    assert 'sourceFrame.appendChild(img); img.src = sourceUrl;' in response.text
    assert 'sourceFrame.appendChild(img); img.src = pending.rgb;' in response.text
    assert 'data-bg="checker"' in response.text
    assert 'data-bg="black"' in response.text
    assert '<option value="auto" selected>auto</option>' in response.text
    assert '<option value="comfy-corridorkey">comfy-corridorkey</option>' in response.text
    assert '<option value="comfy-ermbg">comfy-ermbg</option>' in response.text
    assert 'id="corridorkey-settings" open' in response.text
    assert "<summary>[设置]</summary>" in response.text
    assert '<input id="ck-screen-mode" name="corridorkey_screen_mode" type="hidden" value="auto">' in response.text
    assert "幕布<select" not in response.text
    assert 'name="corridorkey_preset"' in response.text
    assert 'name="corridorkey_despill_strength"' in response.text
    assert 'name="corridorkey_auto_mask"' in response.text
    assert 'id="sam-mask-button"' in response.text
    assert '"/api/sam-mask"' in response.text
    assert 'id="mask-brush-mode"' not in response.text
    assert 'data-mask-mode="keep">保留</button>' in response.text
    assert 'data-mask-mode="erase">擦除</button>' in response.text
    assert 'setMaskBrushMode(button.dataset.maskMode)' in response.text
    assert 'id="mask-brush-size"' in response.text
    assert 'id="mask-reset-button"' not in response.text
    assert 'id="mask-clear-button"' in response.text
    assert '<button id="sam-mask-button" type="button">Sam3</button>' in response.text
    assert '<button id="mask-clear-button" type="button">清空</button>' in response.text
    assert 'id="status">等待上传</span>' in response.text
    assert 'id="sam-mask-status"' not in response.text
    assert 'const samMaskStatus = statusEl;' in response.text
    assert "loadMaskOverlay(payload.mask)" in response.text
    assert 'autoMask.addEventListener("change", () => { if (autoMask.checked) generateSamMask(); });' in response.text
    assert 'samMaskButton.addEventListener("click", () => generateSamMask())' in response.text
    assert 'setPreviewView("mask")' in response.text
    assert 'let activeView = "mask";' in response.text
    assert "maskToolbarControls" in response.text
    assert "edited_hint_mask.png" in response.text
    assert "function exportHintMaskFile()" in response.text
    assert "const value = pixels.data[i + 3] > 8 ? 255 : 0;" in response.text
    assert 'const hintMaskFile = backend.value === "comfy-corridorkey" ? await exportHintMaskFile() : null;' in response.text
    assert 'formData.append("corridorkey_hint_mask", hintMaskFile)' in response.text
    assert 'name="corridorkey_protection_bg_max"' in response.text
    assert "syncBackendSettings()" in response.text
    assert 'canvas.addEventListener("wheel"' in response.text
    assert 'canvas.addEventListener("pointerdown"' in response.text
    assert "selected: candidate.selected === true" in response.text
    assert "setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0)" in response.text
    assert "formatElapsed(performance.now() - startedAt)" in response.text
    assert "server_elapsed_sec" in response.text
    assert "client ${elapsed}" in response.text
    assert "payload.backend || backend.value" in response.text
    assert 'backend.value = "auto"' in response.text


def test_slice_page_serves_slice_mode_entry():
    client = TestClient(app)
    response = client.get("/slice")
    assert response.status_code == 200
    assert "ERMBG 切图" in response.text
    assert 'href="/">返回抠图</a>' in response.text
    assert '"/api/slice-preview"' in response.text
    assert '"/api/slice-crops"' in response.text
    assert 'sessionStorage.setItem("ermbgPendingSlice"' in response.text
    assert 'const SLICE_STATE_KEY = "ermbgSliceWorkspace"' in response.text
    assert "restoreSliceState()" in response.text
    assert ".thumb img { display: block; width: 100%; height: 100%; max-width: 100%; max-height: 100%; object-fit: contain;" in response.text
    assert "grid-template-columns: 64px minmax(0, 1fr) 52px" in response.text
    assert ".thumb { width: 64px; height: 64px;" in response.text
    assert "grid-template-rows: auto auto auto minmax(0, 1fr) auto" in response.text
    assert "scrollbar-gutter: stable" in response.text
    assert ".row:hover { background: #f3f7f1; }" in response.text
    assert ".row[aria-selected=\"true\"] { background: #d7eadf; }" in response.text
    assert ".row[aria-selected=\"true\"] .row-action { visibility: visible; }" in response.text
    assert "overflow-x: hidden" in response.text
    assert 'action.className = "row-action"' in response.text
    assert 'file.addEventListener("change", () => {\n      if (!file.files.length) return;' in response.text


def test_game_eval_page_serves_result_table():
    client = TestClient(app)
    response = client.get("/eval/game")
    assert response.status_code == 200
    assert "ERMBG Game Eval" in response.text
    assert 'id="run-select"' in response.text
    assert 'id="start-full-eval"' in response.text
    assert 'id="eval-panel"' in response.text
    assert 'id="sample-list"' in response.text
    assert "选择测试样本" in response.text
    assert "选择测试路径" in response.text
    assert 'id="eval-test-path"' in response.text
    assert '<option value="auto" selected>Auto</option>' in response.text
    assert '<option value="corridorkey">CorridorKey</option>' in response.text
    assert '<option value="ermbg">ERMBG</option>' in response.text
    assert '<option value="rmbg">RMBG</option>' in response.text
    assert "全选" in response.text
    assert "取消全选" in response.text
    assert "选择测试变体" not in response.text
    assert 'name="eval-variant"' not in response.text
    assert "selectedVariants()" not in response.text
    assert "variantInputs" not in response.text
    assert "selectedTestPath()" in response.text
    assert 'role="progressbar"' in response.text
    assert 'id="batch-progress"' in response.text
    assert '"sampleRows":' in response.text
    assert '"sampleId": "B001"' in response.text
    assert '"sampleId": "I001"' in response.text
    assert '"thumbnailUrl": "/eval/game/file/samples/corridorkey_semantic/' in response.text
    assert '"defaultSelected": true' in response.text
    assert '"defaultSelected": false' in response.text
    assert '"sampleVariant": "green"' in response.text
    assert '"runStatus": "ran"' in response.text
    assert '"progress": {' in response.text
    assert "<th class=\"regions-col\">regions</th>" not in response.text
    for heading in ("原图", "alpha mask", "白底", "黑底", "透明底", "绿底", "紫底", "蓝底"):
        assert f"<th class=\"preview-col\">{heading}</th>" in response.text
    assert "<th class=\"preview-col\">gray</th>" not in response.text
    assert 'data-bg="green"' in response.text
    assert 'data-bg="blue"' in response.text
    assert 'className = "sample-group"' in response.text
    assert 'className = "sample-meta"' in response.text
    assert 'className = "sample-thumb"' in response.text
    assert 'label.title = `${sample.sampleId || ""} · ${sample.caseId || ""}`' in response.text
    assert 'detail.textContent = `${sample.caseId || ""}' not in response.text
    assert "modalStage.addEventListener(\"wheel\"" in response.text
    assert "modalStage.addEventListener(\"pointerdown\"" in response.text


def test_game_eval_start_run_creates_new_batch(monkeypatch, tmp_path):
    import ermbg.web as web

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web.subprocess, "Popen", FakePopen)
    web._GAME_EVAL_JOBS.clear()

    client = TestClient(app)
    response = client.post("/eval/game/run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runId"].startswith("auto_")
    assert "_web_" not in payload["runId"]
    assert payload["runId"].endswith("_v001")
    assert payload["status"] == "running"
    assert payload["progress"]["completed"] == 0
    assert payload["progress"]["total"] == 83
    assert (tmp_path / "out" / payload["runId"] / "web_launch.json").exists()

    status_response = client.get(payload["statusUrl"])
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "running"
    assert status_payload["progress"]["percent"] == 0


def test_game_eval_expected_count_tracks_manifest():
    import ermbg.web as web

    manifest = json.loads((web.PROJECT_ROOT / "samples" / "corridorkey_semantic" / "manifest.json").read_text())

    assert web._game_eval_expected_case_count() == manifest["case_count"] == 83


def test_game_eval_start_run_accepts_selected_samples(monkeypatch, tmp_path):
    import ermbg.web as web

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web.subprocess, "Popen", FakePopen)
    web._GAME_EVAL_JOBS.clear()

    client = TestClient(app)
    response = client.post("/eval/game/run", json={"sample_ids": ["B003", "B005"], "variants": ["white"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["progress"]["total"] == 2
    process = web._GAME_EVAL_JOBS[payload["runId"]]["process"]
    assert "run_corridorkey_game_eval.py" in process.command[1]
    assert "--backend" in process.command
    assert "auto" in process.command
    assert "--sample-id" in process.command
    assert "B003,B005" in process.command
    assert "--variants" not in process.command
    launch = tmp_path / "out" / payload["runId"] / "web_launch.json"
    launch_payload = json.loads(launch.read_text(encoding="utf-8"))
    assert launch_payload["backend"] == "auto"
    assert launch_payload["test_path"] == "auto"
    assert launch_payload["sample_ids"] == ["B003", "B005"]
    assert "variants" not in launch_payload


def test_game_eval_start_run_accepts_test_path(monkeypatch, tmp_path):
    import ermbg.web as web

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web.subprocess, "Popen", FakePopen)
    web._GAME_EVAL_JOBS.clear()

    client = TestClient(app)
    response = client.post("/eval/game/run", json={"sample_ids": ["B002"], "test_path": "ermbg"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["runId"].startswith("ermbg_")
    assert payload["progress"]["total"] == 1
    process = web._GAME_EVAL_JOBS[payload["runId"]]["process"]
    assert "--backend" in process.command
    assert "comfy-ermbg" in process.command
    launch = tmp_path / "out" / payload["runId"] / "web_launch.json"
    launch_payload = json.loads(launch.read_text(encoding="utf-8"))
    assert launch_payload["backend"] == "comfy-ermbg"
    assert launch_payload["test_path"] == "ermbg"
    assert launch_payload["test_path_label"] == "ERMBG"


def test_game_eval_runs_order_by_mtime_and_default_latest(monkeypatch, tmp_path):
    import ermbg.web as web

    out_root = tmp_path / "out"
    older = out_root / "local_ownership_20260528_v001"
    newer = out_root / "local_ownership_20260529_v001"
    for root in (older, newer):
        report_dir = root / "local_ownership"
        report_dir.mkdir(parents=True)
        (report_dir / "eval_report.json").write_text(
            json.dumps({"run_id": root.name, "case_count": 0, "ok_count": 0, "rows": []}),
            encoding="utf-8",
        )
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web, "DEFAULT_GAME_EVAL_ROOT", older)

    runs = web._game_eval_runs()

    assert [item["id"] for item in runs[:2]] == [newer.name, older.name]
    assert web._default_game_eval_root() == newer
    assert runs[0]["selected"] is True


def test_game_eval_status_recognizes_corridorkey_summary(monkeypatch, tmp_path):
    import ermbg.web as web

    run_root = tmp_path / "out" / "corridorkey_20260529_v001"
    run_root.mkdir(parents=True)
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "backend": "comfy-corridorkey",
                "run_count": 2,
                "ok_count": 2,
                "runs": [
                    {"status": "ok", "backend": "comfy-corridorkey"},
                    {"status": "ok", "backend": "comfy-corridorkey"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    web._GAME_EVAL_JOBS.clear()

    status = web._game_eval_batch_status("corridorkey_20260529_v001")

    assert status["status"] == "complete"
    assert status["hasReport"] is True
    assert status["progress"]["completed"] == 2
    assert status["progress"]["total"] == 2


def test_game_eval_running_progress_counts_partial_summaries(monkeypatch, tmp_path):
    import ermbg.web as web

    run_root = tmp_path / "out" / "local_ownership_v001_web_20260527"
    summary_dir = run_root / "local_ownership" / "ui_hard_button_soft_shadow" / "green"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.json").write_text(
        json.dumps({"status": "ok"}, indent=2),
        encoding="utf-8",
    )
    error_dir = run_root / "local_ownership" / "ui_hard_button_soft_shadow" / "white"
    error_dir.mkdir(parents=True)
    (error_dir / "summary.json").write_text(
        json.dumps({"status": "error"}, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    web._GAME_EVAL_JOBS.clear()

    progress = web._game_eval_batch_progress(
        run_root,
        None,
        expected_total=6,
    )

    assert progress["completed"] == 2
    assert progress["ok"] == 1
    assert progress["errors"] == 1
    assert progress["percent"] == 33.3
    assert web._game_eval_root_has_data(run_root) is True


def test_game_eval_page_renders_running_partial_summary(monkeypatch, tmp_path):
    import ermbg.web as web

    sample_root = tmp_path / "samples" / "corridorkey_semantic" / "button" / "button_soft_shadow"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 200, 0)).save(sample_root / "green.png")
    manifest = {
        "cases": [
            {
                "id": "button_soft_shadow",
                "sample_id": "B002",
                "category": "button",
                "green": "samples/corridorkey_semantic/button/button_soft_shadow/green.png",
            }
        ]
    }
    (tmp_path / "samples" / "corridorkey_semantic" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    run_root = tmp_path / "out" / "local_ownership_v001_web_20260527"
    summary_dir = run_root / "local_ownership" / "button_soft_shadow" / "green"
    summary_dir.mkdir(parents=True)
    matte_dir = run_root / "matte" / "button_soft_shadow" / "green"
    matte_dir.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(matte_dir / "rgba.png")
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_id": "B002",
                "sample_code": "B002-G",
                "case_id": "button_soft_shadow",
                "sample_variant": "green",
                "expected_role_hit": True,
                "expected_role": "shadow_like_layer",
                "rgba": "out/local_ownership_v001_web_20260527/matte/button_soft_shadow/green/rgba.png",
                "top_roles": ["shadow_like_layer"],
                "role_counts": {"shadow_like_layer": 1},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web, "DEFAULT_GAME_EVAL_ROOT", run_root)
    web._GAME_EVAL_JOBS.clear()

    client = TestClient(app)
    response = client.get("/eval/game?run=local_ownership_v001_web_20260527")

    assert response.status_code == 200
    assert "local ownership (running)" in response.text
    assert '"sampleCode": "B002-G"' in response.text
    assert '"percent": 100.0' in response.text


def test_game_eval_page_renders_solid_graphic_compare_batch(monkeypatch, tmp_path):
    import ermbg.web as web

    sample_root = tmp_path / "samples" / "corridorkey_semantic" / "button" / "ui_panel"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 200, 0)).save(sample_root / "green.png")

    run_root = tmp_path / "out" / "solid_graphic_game9_compare_20260527"
    case_root = run_root / "B005_ui_panel_green"
    new_root = case_root / "new_solid_graphic"
    old_root = case_root / "old_fallback"
    new_root.mkdir(parents=True)
    old_root.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(new_root / "green_rgba.png")
    Image.new("RGBA", (8, 8), (220, 0, 0, 255)).save(old_root / "green_rgba.png")
    Image.new("RGB", (8, 8), (20, 20, 20)).save(case_root / "alpha_abs_diff.png")
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "batch": "out/solid_graphic_game9_compare_20260527",
                "case_count": 1,
                "rows": [
                    {
                        "sample_id": "B005",
                        "case_id": "ui_panel",
                        "variant": "green",
                        "input": "samples/corridorkey_semantic/button/ui_panel/green.png",
                        "primary_ambiguity": "same_bg_enclosed_region",
                        "status": "ok",
                        "new": {
                            "strategy": "solid_bg_graphic",
                            "solid_confidence": 0.94,
                            "alpha_mean": 0.4,
                            "alpha_soft_fraction": 0.02,
                            "dir": "out/solid_graphic_game9_compare_20260527/B005_ui_panel_green/new_solid_graphic",
                            "rgba": "green_rgba.png",
                            "ownership_counts": {"opaque_subject": 64},
                        },
                        "old": {
                            "strategy": "saturated_bg",
                            "alpha_mean": 0.3,
                            "alpha_soft_fraction": 0.08,
                            "dir": "out/solid_graphic_game9_compare_20260527/B005_ui_panel_green/old_fallback",
                            "rgba": "green_rgba.png",
                        },
                        "alpha_diff": {"mean_abs": 0.24, "p95_abs": 0.8, "max_abs": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web, "DEFAULT_GAME_EVAL_ROOT", run_root)

    client = TestClient(app)
    response = client.get("/eval/game?run=solid_graphic_game9_compare_20260527")

    assert response.status_code == 200
    assert "solid_graphic_game9_compare_20260527" in response.text
    assert "solid graphic comparison" in response.text
    assert "new solid_bg_graphic" in response.text
    assert "old fallback" in response.text
    assert "alpha diff" in response.text
    assert "/eval/game/file/out/solid_graphic_game9_compare_20260527/B005_ui_panel_green/new_solid_graphic/green_rgba.png" in response.text
    assert "/eval/game/file/out/solid_graphic_game9_compare_20260527/B005_ui_panel_green/alpha_abs_diff.png" in response.text


def test_game_eval_page_renders_comfy_ermbg_batch(monkeypatch, tmp_path):
    import ermbg.web as web

    sample_root = tmp_path / "samples" / "corridorkey_semantic" / "button" / "ui_panel"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 200, 0)).save(sample_root / "green.png")
    Image.new("RGB", (8, 8), (255, 255, 255)).save(sample_root / "white.png")
    (tmp_path / "samples" / "corridorkey_semantic" / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "sample_id": "B005",
                        "id": "ui_panel",
                        "category": "button",
                        "green": "samples/corridorkey_semantic/button/ui_panel/green.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_root = tmp_path / "out" / "comfy_full_test_20260529"
    case_root = run_root / "B005_green_remote"
    case_root.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(case_root / "rgba.png")
    Image.new("L", (8, 8), 255).save(case_root / "alpha.png")
    Image.new("RGB", (8, 8), (200, 200, 200)).save(case_root / "contact_sheet.png")
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "case": "B005_green",
                        "phase": "remote",
                        "backend": "comfy-ermbg",
                        "input": str(sample_root / "green.png"),
                        "elapsed_sec_client": 6.2,
                        "outputs": {
                            "rgba": "out/comfy_full_test_20260529/B005_green_remote/rgba.png",
                            "alpha": "out/comfy_full_test_20260529/B005_green_remote/alpha.png",
                            "contact_sheet": "out/comfy_full_test_20260529/B005_green_remote/contact_sheet.png",
                        },
                        "remote_debug": {"timings": {"total_sec": 5.1}},
                        "quality_metrics": {"alpha_mean": 0.42, "alpha_nonzero_pixels": 64},
                        "case_metadata": {
                            "sample_id": "B005",
                            "id": "ui_panel",
                            "category": "button",
                            "primary_ambiguity": "remote production smoke",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web, "DEFAULT_GAME_EVAL_ROOT", run_root)

    runs = web._game_eval_runs(run_root)
    assert any(item["id"] == "comfy_full_test_20260529" for item in runs)

    client = TestClient(app)
    response = client.get("/eval/game?run=comfy_full_test_20260529")

    assert response.status_code == 200
    assert "comfy_full_test_20260529" in response.text
    assert "comfy-ermbg remote" in response.text
    assert "comfy-ermbg" in response.text
    assert "contact sheet" not in response.text
    assert "/eval/game/file/out/comfy_full_test_20260529/B005_green_remote/rgba.png" in response.text
    assert "/eval/game/file/out/comfy_full_test_20260529/B005_green_remote/alpha.png" in response.text
    assert "/eval/game/file/out/comfy_full_test_20260529/B005_green_remote/contact_sheet.png" not in response.text


def test_game_eval_file_serves_eval_image():
    client = TestClient(app)
    response = client.get(
        "/eval/game/file/samples/corridorkey_semantic/sheets/full_samples_v1_sheet.jpg"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(response.content)).mode == "RGB"


def test_game_eval_regions_serves_bbox_overlay(monkeypatch, tmp_path):
    import ermbg.web as web

    sample_root = tmp_path / "samples" / "corridorkey_semantic" / "button" / "ui_panel"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (16, 16), (0, 200, 0)).save(sample_root / "green.png")
    run_root = tmp_path / "out" / "local_ownership_test_regions"
    local_dir = run_root / "local_ownership" / "ui_panel" / "green"
    local_dir.mkdir(parents=True)
    (local_dir / "summary.json").write_text(
        json.dumps(
            {
                "input": "samples/corridorkey_semantic/button/ui_panel/green.png",
                "ownership": [
                    {
                        "region": {
                            "kind": "hard_edge_candidate",
                            "bbox_xyxy": [3, 3, 12, 12],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "local_ownership" / "eval_report.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "case_count": 1,
                "ok_count": 1,
                "rows": [
                    {
                        "status": "ok",
                        "case_id": "ui_panel",
                        "sample_id": "B005",
                        "sample_variant": "green",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web, "DEFAULT_GAME_EVAL_ROOT", run_root)

    client = TestClient(app)
    response = client.get("/eval/game/regions/ui_panel?variant=green&run=local_ownership_test_regions")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(BytesIO(response.content))
    assert image.mode == "RGBA"
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_matte_endpoint_returns_png(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        assert kwargs["shadow_mode"] == "on"
        del image, backend, qa, kwargs
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., 0] = 220
        rgba[..., 3] = 255
        return MatteResponse(
            rgba=rgba,
            alpha=np.ones((8, 8), dtype=np.float32),
            foreground_srgb=rgba[..., :3],
            strategy_name="saturated_bg",
            background_color=(0, 200, 0),
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "grabcut"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-ermbg-strategy"] == "saturated_bg"
    assert response.headers["x-ermbg-background"] == "0,200,0"
    assert response.headers["x-ermbg-local-ownership"] == "0"
    assert Image.open(BytesIO(response.content)).mode == "RGBA"


def test_matte_endpoint_returns_local_ownership_png_when_available(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        assert kwargs["shadow_mode"] == "on"
        del backend, qa, kwargs
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 60
        return MatteResponse(
            rgba=rgba,
            alpha=rgba[..., 3].astype(np.float32) / 255.0,
            foreground_srgb=rgba[..., :3],
            strategy_name="saturated_bg",
            background_color=(0, 200, 0),
        )

    def fake_local_candidate(image_rgb, base_rgba, background_color, backend="auto", **kwargs):
        assert kwargs["shadow_mode"] == "on"
        del image_rgb, base_rgba, background_color, backend, kwargs
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[..., :3] = (10, 20, 30)
        rgba[..., 3] = 180
        return MatteCandidate(id="local_ownership", label="Local Ownership", rgba=rgba, selected=True)

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)
    monkeypatch.setattr(web, "generate_local_ownership_candidate", fake_local_candidate)

    client = TestClient(app)
    response = client.post(
        "/api/matte",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "grabcut"},
    )

    assert response.status_code == 200
    assert response.headers["x-ermbg-local-ownership"] == "1"
    rgba = np.asarray(Image.open(BytesIO(response.content)).convert("RGBA"))
    assert rgba[0, 0].tolist() == [10, 20, 30, 180]


def test_matte_candidates_endpoint_returns_candidate_json(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        assert kwargs["shadow_mode"] == "on"
        del backend, qa, kwargs
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 1] = 180
        rgba[..., 3] = 255
        return MatteResponse(
            rgba=rgba,
            alpha=np.ones((h, w), dtype=np.float32),
            foreground_srgb=rgba[..., :3],
            strategy_name="white_bg",
            background_color=(255, 255, 255),
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "grabcut"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "white_bg"
    assert payload["background"] == [255, 255, 255]
    assert payload["candidates"][0]["id"] == "auto"
    assert payload["candidates"][0]["label"] == "自动结果"
    assert payload["candidates"][0]["regions"] == []
    assert payload["candidates"][0]["operation_results"] == []
    assert payload["candidates"][0]["plan"] is None
    data_url = payload["candidates"][0]["rgba"]
    assert data_url.startswith("data:image/png;base64,")
    png = base64.b64decode(data_url.split(",", 1)[1])
    assert Image.open(BytesIO(png)).mode == "RGBA"


def test_matte_candidates_endpoint_serializes_comfy_ermbg_debug(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        assert backend == "comfy-ermbg"
        assert kwargs["shadow_mode"] == "on"
        del qa, kwargs
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        return MatteResponse(
            rgba=rgba,
            alpha=np.ones((h, w), dtype=np.float32),
            foreground_srgb=rgba[..., :3],
            strategy_name="comfy_ermbg",
            background_color=(0, 200, 0),
            debug={"soft_mask": np.ones((h, w), dtype=np.float32), "prompt_id": "prompt-1"},
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "comfy-ermbg"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "comfy_ermbg"
    assert payload["backend"] == "comfy-ermbg"
    assert isinstance(payload["server_elapsed_sec"], float)
    assert payload["debug"]["prompt_id"] == "prompt-1"
    assert [(c["id"], c["label"], c["selected"]) for c in payload["candidates"]] == [
        ("auto", "远端 ERMBG", True)
    ]
    assert payload["candidates"][0]["debug"]["remote"]["prompt_id"] == "prompt-1"
    assert payload["candidates"][0]["debug"]["remote"]["soft_mask"]["shape"] == [16, 16]


def test_matte_candidates_endpoint_uses_auto_selected_remote_backend(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        assert backend == "auto"
        assert kwargs["shadow_mode"] == "on"
        del qa, kwargs
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        return MatteResponse(
            rgba=rgba,
            alpha=np.ones((h, w), dtype=np.float32),
            foreground_srgb=rgba[..., :3],
            strategy_name="comfy_corridorkey",
            background_color=(0, 200, 0),
            debug={
                "soft_mask": np.ones((h, w), dtype=np.float32),
                "auto_route": {"selected_backend": "comfy-corridorkey", "reason": "green_screen"},
            },
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "auto"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "comfy-corridorkey"
    assert payload["requested_backend"] == "auto"
    assert [(c["id"], c["label"], c["selected"]) for c in payload["candidates"]] == [
        ("auto", "远端 CorridorKey", True)
    ]


def test_matte_candidates_endpoint_passes_corridorkey_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        del qa
        captured.update(kwargs)
        assert backend == "comfy-corridorkey"
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        return MatteResponse(
            rgba=rgba,
            alpha=np.ones((h, w), dtype=np.float32),
            foreground_srgb=rgba[..., :3],
            strategy_name="comfy_corridorkey",
            background_color=(0, 200, 0),
            debug={"prompt_id": "prompt-ck", "color_protection": {"enabled": True}},
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={
            "backend": "comfy-corridorkey",
            "corridorkey_gamma_space": "Linear",
            "corridorkey_despill_strength": "0.25",
            "corridorkey_refiner_strength": "1.5",
            "corridorkey_auto_despeckle": "Off",
            "corridorkey_despeckle_size": "64",
            "corridorkey_auto_mask": "false",
            "corridorkey_color_protection": "false",
            "corridorkey_protection_bg_max": "6",
            "corridorkey_protection_fg_min": "14",
            "corridorkey_screen_mode": "blue",
            "corridorkey_preset": "manual",
        },
    )

    assert response.status_code == 200
    assert captured["shadow_mode"] == "on"
    assert captured["corridorkey_gamma_space"] == "Linear"
    assert captured["corridorkey_despill_strength"] == 0.25
    assert captured["corridorkey_refiner_strength"] == 1.5
    assert captured["corridorkey_auto_despeckle"] == "Off"
    assert captured["corridorkey_despeckle_size"] == 64
    assert captured["corridorkey_auto_mask"] is False
    assert captured["corridorkey_color_protection"] is False
    assert captured["corridorkey_protection_bg_max"] == 6.0
    assert captured["corridorkey_protection_fg_min"] == 14.0
    assert captured["corridorkey_screen_mode"] == "blue"
    assert captured["corridorkey_preset"] == "manual"
    payload = response.json()
    assert payload["candidates"][0]["label"] == "远端 CorridorKey"


def test_matte_candidates_endpoint_accepts_corridorkey_hint_mask(monkeypatch):
    captured: dict[str, object] = {}

    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        del image, qa
        captured.update(kwargs)
        assert backend == "comfy-corridorkey"
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return MatteResponse(
            rgba=rgba,
            alpha=np.ones((16, 16), dtype=np.float32),
            foreground_srgb=rgba[..., :3],
            strategy_name="comfy_corridorkey",
            background_color=(0, 200, 0),
            debug={"prompt_id": "prompt-ck"},
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={
            "file": ("input.png", _png_bytes(), "image/png"),
            "corridorkey_hint_mask": ("mask.png", _mask_png_bytes(), "image/png"),
        },
        data={"backend": "comfy-corridorkey"},
    )

    assert response.status_code == 200
    assert captured["corridorkey_hint_mask"] is not None


def test_sam_mask_endpoint_returns_mask_payload(monkeypatch):
    class _FakeSAM3Result:
        def __init__(self):
            self.mask = np.zeros((16, 16), dtype=np.float32)
            self.mask[4:12, 4:12] = 1.0
            self.debug = {"prompt_id": "prompt-sam3", "mask": {"mean": float(self.mask.mean())}}

    class _FakeSAM3Client:
        def mask(self, image_srgb, **kwargs):
            assert image_srgb.shape == (16, 16, 3)
            assert kwargs["threshold"] == 0.4
            assert kwargs["refine_iterations"] == 1
            return _FakeSAM3Result()

    import ermbg.probe.comfyui_sam3_mask as sam3_mod

    monkeypatch.setattr(sam3_mod, "ComfyUISAM3MaskClient", lambda: _FakeSAM3Client())

    client = TestClient(app)
    response = client.post(
        "/api/sam-mask",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"threshold": "0.4", "refine_iterations": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "comfy-sam3"
    assert payload["mask"].startswith("data:image/png;base64,")
    assert payload["debug"]["prompt_id"] == "prompt-sam3"


def test_matte_candidates_endpoint_returns_same_color_hole_candidates(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        del backend, qa, kwargs
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
        ring = (r <= 22) & (r >= 9)
        hole = r < 9
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[ring, :3] = rgb[ring]
        rgba[ring, 3] = 255
        rgba[hole, :3] = 255
        rgba[hole, 3] = 0
        return MatteResponse(
            rgba=rgba,
            alpha=rgba[..., 3].astype(np.float32) / 255.0,
            foreground_srgb=rgba[..., :3],
            strategy_name="white_bg",
            background_color=(255, 255, 255),
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("ring.png", _ring_png_bytes(), "image/png")},
        data={"backend": "grabcut"},
    )

    assert response.status_code == 200
    payload = response.json()
    ids = [candidate["id"] for candidate in payload["candidates"]]
    assert ids == ["transparent_hole", "same_color_marking"]
    assert payload["candidates"][0]["plan"]["operations"][0]["tool"] == "preserve_hole"
    assert payload["candidates"][0]["regions"][0]["kind"] == "same_bg_enclosed_region"
    assert payload["candidates"][0]["regions"][0]["evidence_kind"] == "same_bg_low_alpha_enclosed"
    assert payload["candidates"][1]["operation_results"][0]["tool"] == "fill_same_color_region"
    filled_url = payload["candidates"][1]["rgba"]
    filled_png = base64.b64decode(filled_url.split(",", 1)[1])
    filled = np.asarray(Image.open(BytesIO(filled_png)).convert("RGBA"))
    assert filled[32, 32, 3] == 255
    assert filled[32, 32, :3].tolist() == [255, 255, 255]


def test_matte_candidates_endpoint_selects_local_ownership_candidate(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False, **kwargs):
        del backend, qa, kwargs
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 80
        return MatteResponse(
            rgba=rgba,
            alpha=rgba[..., 3].astype(np.float32) / 255.0,
            foreground_srgb=rgba[..., :3],
            strategy_name="saturated_bg",
            background_color=(0, 200, 0),
        )

    def fake_local_candidate(image_rgb, base_rgba, background_color, backend="auto", **kwargs):
        del image_rgb, base_rgba, background_color, backend, kwargs
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[..., :3] = (10, 20, 30)
        rgba[..., 3] = 180
        return MatteCandidate(
            id="local_ownership",
            label="Local Ownership",
            rgba=rgba,
            selected=True,
            debug={"local_ownership": {"role_mask_pixels": {"subject_soft_layer": 64}}},
        )

    import ermbg.web as web

    monkeypatch.setattr(web, "matte_image", fake_matte_image)
    monkeypatch.setattr(web, "generate_local_ownership_candidate", fake_local_candidate)

    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "grabcut"},
    )

    assert response.status_code == 200
    payload = response.json()
    selected = [candidate for candidate in payload["candidates"] if candidate["selected"]]
    assert [candidate["id"] for candidate in selected] == ["local_ownership"]
    assert payload["candidates"][-1]["debug"]["local_ownership"]["role_mask_pixels"]["subject_soft_layer"] == 64


def test_slice_endpoint_returns_zip_of_rectangular_crops():
    img = np.full((48, 72, 3), [0, 200, 0], dtype=np.uint8)
    img[8:22, 8:24] = [240, 30, 30]
    img[25:42, 44:64] = [20, 40, 220]
    buf = BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")

    client = TestClient(app)
    response = client.post(
        "/api/slice",
        files={"file": ("sheet.png", buf.getvalue(), "image/png")},
        data={"min_area": "50", "padding": "1"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-ermbg-slice-count"] == "2"
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["sheet.slices.json", "sheet_001_rgb.png", "sheet_002_rgb.png"]
        report = json.loads(zf.read("sheet.slices.json"))
        assert report["background_color"] == [0, 200, 0]
        assert report["count"] == 2
        assert Image.open(BytesIO(zf.read("sheet_001_rgb.png"))).mode == "RGB"


def test_slice_preview_endpoint_returns_annotated_boxes():
    img = np.full((48, 72, 3), [0, 200, 0], dtype=np.uint8)
    img[8:22, 8:24] = [240, 30, 30]
    img[25:42, 44:64] = [20, 40, 220]
    buf = BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")

    client = TestClient(app)
    response = client.post(
        "/api/slice-preview",
        files={"file": ("sheet.png", buf.getvalue(), "image/png")},
        data={"min_area": "50", "padding": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["background_color"] == [0, 200, 0]
    assert payload["annotated"].startswith("data:image/png;base64,")
    png = base64.b64decode(payload["annotated"].split(",", 1)[1])
    assert Image.open(BytesIO(png)).mode == "RGBA"


def test_slice_crops_endpoint_returns_list_payload():
    img = np.full((48, 72, 3), [0, 200, 0], dtype=np.uint8)
    img[8:22, 8:24] = [240, 30, 30]
    img[25:42, 44:64] = [20, 40, 220]
    buf = BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")

    client = TestClient(app)
    response = client.post(
        "/api/slice-crops",
        files={"file": ("sheet.png", buf.getvalue(), "image/png")},
        data={"min_area": "50", "padding": "1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["crops"][0]["filename"] == "icon_001_rgb.png"
    assert payload["crops"][0]["label"] == "icon_001"
    assert payload["crops"][0]["kind"] == "icon"
    assert "confidence" in payload["crops"][0]
    assert "features" in payload["crops"][0]
    assert payload["crops"][0]["rgb"].startswith("data:image/png;base64,")
    png = base64.b64decode(payload["crops"][0]["rgb"].split(",", 1)[1])
    assert Image.open(BytesIO(png)).mode == "RGB"


def test_slice_preview_and_crops_reuse_cached_slice_result(monkeypatch):
    import ermbg.web as web

    img = np.full((48, 72, 3), [0, 200, 0], dtype=np.uint8)
    img[8:22, 8:24] = [240, 30, 30]
    buf = BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    image_bytes = buf.getvalue()

    calls = {"count": 0}
    real_slice_image = web.slice_image

    def counted_slice_image(*args, **kwargs):
        calls["count"] += 1
        return real_slice_image(*args, **kwargs)

    with web._SLICE_CACHE_LOCK:
        web._SLICE_CACHE.clear()
    monkeypatch.setattr(web, "slice_image", counted_slice_image)

    client = TestClient(app)
    for endpoint in ("/api/slice-preview", "/api/slice-crops"):
        response = client.post(
            endpoint,
            files={"file": ("sheet.png", image_bytes, "image/png")},
            data={"min_area": "50", "padding": "1"},
        )
        assert response.status_code == 200

    assert calls["count"] == 1


def test_matte_endpoint_rejects_unknown_backend():
    client = TestClient(app)
    response = client.post(
        "/api/matte",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "unknown"},
    )
    assert response.status_code == 400


def test_matte_candidates_endpoint_rejects_unknown_backend():
    client = TestClient(app)
    response = client.post(
        "/api/matte-candidates",
        files={"file": ("input.png", _png_bytes(), "image/png")},
        data={"backend": "unknown"},
    )
    assert response.status_code == 400
