"""Tests for the ERMBG web service."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from ermbg.api import MatteResponse
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
    assert "候选缩略图" in response.text
    assert 'role="tablist"' in response.text
    assert 'data-bg="checker"' in response.text
    assert 'data-bg="black"' in response.text
    assert '<option value="auto" selected>auto</option>' in response.text
    assert 'canvas.addEventListener("wheel"' in response.text
    assert 'canvas.addEventListener("pointerdown"' in response.text
    assert "selected: candidate.selected === true" in response.text
    assert "setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0)" in response.text


def test_matte_endpoint_returns_png(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False):
        del image, backend, qa
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
    assert Image.open(BytesIO(response.content)).mode == "RGBA"


def test_matte_candidates_endpoint_returns_candidate_json(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False):
        del backend, qa
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


def test_matte_candidates_endpoint_returns_same_color_hole_candidates(monkeypatch):
    def fake_matte_image(image, backend="auto", qa=False):
        del backend, qa
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
