"""Tests for the ERMBG web service."""

from __future__ import annotations

import base64
import json
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
    assert 'href="/slice">切图</a>' in response.text
    assert '"/api/slice-preview"' not in response.text
    assert '"/api/slice-crops"' not in response.text
    assert "confirm-slices" not in response.text
    assert "候选缩略图" in response.text
    assert 'href="/eval/game"' in response.text
    assert 'role="tablist"' in response.text
    assert ".source-frame img { display: block; width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: contain;" in response.text
    assert ".source-frame { width: 100%; aspect-ratio: 4 / 3; min-height: 148px; display: grid; place-items: center; border:" in response.text
    assert 'file.addEventListener("change", () => { if (!file.files.length) return; resetResult();' in response.text
    assert 'data-bg="checker"' in response.text
    assert 'data-bg="black"' in response.text
    assert '<option value="comfy-ermbg" selected>comfy-ermbg</option>' in response.text
    assert 'canvas.addEventListener("wheel"' in response.text
    assert 'canvas.addEventListener("pointerdown"' in response.text
    assert "selected: candidate.selected === true" in response.text
    assert "setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0)" in response.text
    assert "formatElapsed(performance.now() - startedAt)" in response.text
    assert "server_elapsed_sec" in response.text
    assert "client ${elapsed}" in response.text
    assert "payload.backend || backend.value" in response.text
    assert 'backend.value = "comfy-ermbg"' in response.text


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
    assert "local_ownership_full_20260527" in response.text
    assert 'id="run-select"' in response.text
    assert 'id="start-full-eval"' in response.text
    assert 'id="eval-panel"' in response.text
    assert 'id="sample-list"' in response.text
    assert "选择测试样本" in response.text
    assert "全选" in response.text
    assert "取消全选" in response.text
    assert 'role="progressbar"' in response.text
    assert 'id="batch-progress"' in response.text
    assert "ui_glass_button_soft_shadow" in response.text
    assert '"sampleRows": 18' in response.text
    assert '"sampleId": "G01"' in response.text
    assert '"sampleId": "G02"' in response.text
    assert '"defaultSelected": true' in response.text
    assert '"defaultSelected": false' in response.text
    assert '"sampleCode": "G01-W"' in response.text
    assert '"sampleCode": "G01-G"' in response.text
    assert '"sampleVariant": "white"' in response.text
    assert '"sampleVariant": "green"' in response.text
    assert '"runStatus": "ran"' in response.text
    assert '"progress": {' in response.text
    assert '"/eval/game/regions/ui_hard_button_no_shadow?variant=green' in response.text
    assert "local_ownership" in response.text
    assert "<th class=\"regions-col\">regions</th>" in response.text
    assert "<th class=\"preview-col\">purple</th>" in response.text
    assert "<th class=\"preview-col\">green ref</th>" in response.text
    assert "<th class=\"preview-col\">gray</th>" not in response.text
    assert "<th class=\"preview-col\">blue</th>" not in response.text
    assert 'data-bg="green"' in response.text
    assert "modalStage.addEventListener(\"wheel\"" in response.text
    assert "modalStage.addEventListener(\"pointerdown\"" in response.text
    assert "/eval/game/file/out/local_ownership_" in response.text


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
    assert payload["runId"].startswith("local_ownership_v001_web_")
    assert payload["status"] == "running"
    assert payload["progress"]["completed"] == 0
    assert payload["progress"]["total"] == 18
    assert (tmp_path / "out" / payload["runId"] / "web_launch.json").exists()

    status_response = client.get(payload["statusUrl"])
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "running"
    assert status_payload["progress"]["percent"] == 0


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
    response = client.post("/eval/game/run", json={"sample_ids": ["G03", "G05"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["progress"]["total"] == 4
    process = web._GAME_EVAL_JOBS[payload["runId"]]["process"]
    assert "10_local_ownership_batch.py" in process.command[1]
    assert "--sample-id" in process.command
    assert "G03,G05" in process.command
    launch = tmp_path / "out" / payload["runId"] / "web_launch.json"
    assert json.loads(launch.read_text(encoding="utf-8"))["sample_ids"] == ["G03", "G05"]


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

    sample_root = tmp_path / "samples" / "vlm_eval_game" / "ui_hard_button_soft_shadow"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 200, 0)).save(sample_root / "green.png")
    manifest = {
        "cases": [
            {
                "id": "ui_hard_button_soft_shadow",
                "sample_id": "G02",
                "green": "samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png",
            }
        ]
    }
    (tmp_path / "samples" / "vlm_eval_game" / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    run_root = tmp_path / "out" / "local_ownership_v001_web_20260527"
    summary_dir = run_root / "local_ownership" / "ui_hard_button_soft_shadow" / "green"
    summary_dir.mkdir(parents=True)
    matte_dir = run_root / "matte" / "ui_hard_button_soft_shadow" / "green"
    matte_dir.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(matte_dir / "rgba.png")
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_id": "G02",
                "sample_code": "G02-G",
                "case_id": "ui_hard_button_soft_shadow",
                "sample_variant": "green",
                "expected_role_hit": True,
                "expected_role": "shadow_like_layer",
                "rgba": "out/local_ownership_v001_web_20260527/matte/ui_hard_button_soft_shadow/green/rgba.png",
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
    assert '"sampleCode": "G02-G"' in response.text
    assert '"percent": 50.0' in response.text


def test_game_eval_page_renders_solid_graphic_compare_batch(monkeypatch, tmp_path):
    import ermbg.web as web

    sample_root = tmp_path / "samples" / "vlm_eval_game" / "ui_panel"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 200, 0)).save(sample_root / "green.png")

    run_root = tmp_path / "out" / "solid_graphic_game9_compare_20260527"
    case_root = run_root / "G05_ui_panel_green"
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
                        "sample_id": "G05",
                        "case_id": "ui_panel",
                        "variant": "green",
                        "input": "samples/vlm_eval_game/ui_panel/green.png",
                        "primary_ambiguity": "same_bg_enclosed_region",
                        "status": "ok",
                        "new": {
                            "strategy": "solid_bg_graphic",
                            "solid_confidence": 0.94,
                            "alpha_mean": 0.4,
                            "alpha_soft_fraction": 0.02,
                            "dir": "out/solid_graphic_game9_compare_20260527/G05_ui_panel_green/new_solid_graphic",
                            "rgba": "green_rgba.png",
                            "ownership_counts": {"opaque_subject": 64},
                        },
                        "old": {
                            "strategy": "saturated_bg",
                            "alpha_mean": 0.3,
                            "alpha_soft_fraction": 0.08,
                            "dir": "out/solid_graphic_game9_compare_20260527/G05_ui_panel_green/old_fallback",
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
    assert "/eval/game/file/out/solid_graphic_game9_compare_20260527/G05_ui_panel_green/new_solid_graphic/green_rgba.png" in response.text
    assert "/eval/game/file/out/solid_graphic_game9_compare_20260527/G05_ui_panel_green/alpha_abs_diff.png" in response.text


def test_game_eval_page_renders_comfy_ermbg_batch(monkeypatch, tmp_path):
    import ermbg.web as web

    sample_root = tmp_path / "samples" / "vlm_eval_game" / "ui_panel"
    sample_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 200, 0)).save(sample_root / "green.png")
    Image.new("RGB", (8, 8), (255, 255, 255)).save(sample_root / "white.png")
    (tmp_path / "samples" / "vlm_eval_game" / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "sample_id": "G05",
                        "id": "ui_panel",
                        "category": "ui",
                        "green": "samples/vlm_eval_game/ui_panel/green.png",
                        "white": "samples/vlm_eval_game/ui_panel/white.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_root = tmp_path / "out" / "comfy_full_test_20260529"
    case_root = run_root / "G05_green_remote"
    case_root.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(case_root / "rgba.png")
    Image.new("RGB", (8, 8), (200, 200, 200)).save(case_root / "contact_sheet.png")
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "case": "G05_green",
                        "phase": "remote",
                        "backend": "comfy-ermbg",
                        "input": str(sample_root / "green.png"),
                        "elapsed_sec_client": 6.2,
                        "outputs": {
                            "rgba": "out/comfy_full_test_20260529/G05_green_remote/rgba.png",
                            "contact_sheet": "out/comfy_full_test_20260529/G05_green_remote/contact_sheet.png",
                        },
                        "remote_debug": {"timings": {"total_sec": 5.1}},
                        "quality_metrics": {"alpha_mean": 0.42, "alpha_nonzero_pixels": 64},
                        "case_metadata": {
                            "sample_id": "G05",
                            "id": "ui_panel",
                            "category": "ui",
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

    client = TestClient(app)
    response = client.get("/eval/game?run=comfy_full_test_20260529")

    assert response.status_code == 200
    assert "comfy_full_test_20260529" in response.text
    assert "comfy-ermbg remote" in response.text
    assert "comfy-ermbg" in response.text
    assert "contact sheet" in response.text
    assert "/eval/game/file/out/comfy_full_test_20260529/G05_green_remote/rgba.png" in response.text
    assert "/eval/game/file/out/comfy_full_test_20260529/G05_green_remote/contact_sheet.png" in response.text


def test_game_eval_file_serves_eval_image():
    client = TestClient(app)
    response = client.get(
        "/eval/game/file/out/local_ownership_full_20260527/local_ownership/role_sheet.png"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(BytesIO(response.content)).mode == "RGB"


def test_game_eval_regions_serves_bbox_overlay():
    client = TestClient(app)
    response = client.get("/eval/game/regions/ui_glass_button_soft_shadow?variant=green")
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
