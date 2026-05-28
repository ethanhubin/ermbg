"""Tests for the high-level Python API (ermbg.api)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ermbg import MatteResponse, classify_image, matte_image

pytestmark = pytest.mark.core


def _solid_green_with_red_subject(h=128, w=128):
    img = np.full((h, w, 3), [0, 200, 0], dtype=np.uint8)
    img[40:90, 40:90] = (220, 30, 30)
    return img


# Use grabcut for tests so we don't need to download BiRefNet weights
@pytest.fixture
def _force_grabcut(monkeypatch):
    """matte_image asks build_segmenter(backend="auto") which prefers BiRefNet.
    For unit tests we want a fast deterministic backend. Patch the default."""
    import ermbg.api as api

    real = api.build_segmenter

    def stub(backend="auto", **kwargs):
        return real(backend="grabcut")

    monkeypatch.setattr(api, "build_segmenter", stub)


def test_classify_image_from_ndarray():
    img = _solid_green_with_red_subject()
    s = classify_image(img)
    assert s.bg_type == "saturated"


def test_classify_image_from_path(tmp_path):
    img = _solid_green_with_red_subject()
    p = tmp_path / "in.png"
    Image.fromarray(img).save(p)
    s = classify_image(p)
    assert s.bg_type == "saturated"


def test_classify_image_from_pil():
    img = _solid_green_with_red_subject()
    s = classify_image(Image.fromarray(img))
    assert s.bg_type == "saturated"


def test_matte_image_ndarray_returns_response(_force_grabcut):
    img = _solid_green_with_red_subject()
    r = matte_image(img)
    assert isinstance(r, MatteResponse)
    assert r.rgba.shape == (128, 128, 4)
    assert r.rgba.dtype == np.uint8
    assert r.alpha.shape == (128, 128)
    assert r.foreground_srgb.shape == (128, 128, 3)
    assert r.strategy_name == "solid_bg_graphic"
    assert r.output_dir is None


def test_matte_image_solid_graphic_prepass_skips_segmenter(monkeypatch):
    import ermbg.api as api

    def fail_build_segmenter(**kwargs):
        raise AssertionError("solid graphic prepass should not build a segmenter")

    monkeypatch.setattr(api, "build_segmenter", fail_build_segmenter)

    img = _solid_green_with_red_subject()
    r = matte_image(img)

    assert r.strategy_name == "solid_bg_graphic"
    assert r.report["strategy"]["name"] == "solid_bg_graphic"
    assert r.alpha[44:86, 44:86].mean() > 0.99


def test_matte_image_reuses_cached_segmenter(monkeypatch):
    import ermbg.api as api

    class _CountingSegmenter:
        def __init__(self):
            self.calls = 0

        def segment(self, image, object_prompt=None):
            self.calls += 1
            alpha = np.zeros(image.shape[:2], dtype=np.float32)
            alpha[32:96, 32:96] = 1.0
            return alpha

    built: list[_CountingSegmenter] = []

    def stub_build_segmenter(**kwargs):
        assert "url" not in kwargs
        seg = _CountingSegmenter()
        built.append(seg)
        return seg

    api._SEGMENTER_CACHE.clear()
    monkeypatch.setattr(api, "build_segmenter", stub_build_segmenter)

    img = _solid_green_with_red_subject()
    matte_image(img, backend="grabcut", matting_model="cache-test", solid_graphic_prepass=False)
    matte_image(img, backend="grabcut", matting_model="cache-test", solid_graphic_prepass=False)

    assert len(built) == 1
    assert built[0].calls == 2


def test_matte_image_cache_separates_comfy_urls(monkeypatch):
    import ermbg.api as api

    class _Segmenter:
        def segment(self, image, object_prompt=None):
            alpha = np.zeros(image.shape[:2], dtype=np.float32)
            alpha[32:96, 32:96] = 1.0
            return alpha

    built_urls: list[str] = []

    def stub_build_segmenter(**kwargs):
        built_urls.append(kwargs["url"])
        return _Segmenter()

    api._SEGMENTER_CACHE.clear()
    monkeypatch.setattr(api, "build_segmenter", stub_build_segmenter)

    img = _solid_green_with_red_subject()
    matte_image(img, backend="comfy-rmbg", comfy_url="http://comfy-a.invalid")
    matte_image(img, backend="comfy-rmbg", comfy_url="http://comfy-b.invalid")
    matte_image(img, backend="comfy-rmbg", comfy_url="http://comfy-a.invalid")

    assert built_urls == ["http://comfy-a.invalid", "http://comfy-b.invalid"]


def test_matte_image_comfy_ermbg_uses_remote_full_pipeline(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_ermbg_matte as remote_mod

    def fail_build_segmenter(**kwargs):
        raise AssertionError("comfy-ermbg should not build a local segmenter")

    class _FakeRemoteResult:
        def __init__(self):
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., :3] = (10, 20, 30)
            self.rgba[..., 3] = 255
            self.alpha = np.ones((128, 128), dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.debug = {"prompt_id": "prompt-1"}

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            assert image_srgb.shape == (128, 128, 3)
            assert kwargs["shadow_mode"] == "off"
            return _FakeRemoteResult()

    monkeypatch.setattr(api, "build_segmenter", fail_build_segmenter)
    monkeypatch.setattr(remote_mod, "ComfyUIErmbgMatteClient", _FakeClient)

    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="comfy-ermbg", shadow_mode="off")

    assert r.strategy_name == "comfy_ermbg"
    assert r.rgba[0, 0].tolist() == [10, 20, 30, 255]
    assert r.report["strategy"]["name"] == "comfy_ermbg"
    assert r.debug["prompt_id"] == "prompt-1"


def test_matte_image_writes_files_when_output_dir_given(_force_grabcut, tmp_path):
    img = _solid_green_with_red_subject()
    p = tmp_path / "in.png"
    Image.fromarray(img).save(p)
    out = tmp_path / "out"
    r = matte_image(p, output_dir=out)
    assert r.output_dir == out
    assert (out / "in_rgba.png").exists()
    assert (out / "in_alpha.png").exists()
    assert (out / "in_shadow.png").exists()
    assert (out / "in_foreground.png").exists()
    assert (out / "in.report.json").exists()


def test_matte_image_qa_adds_metrics_to_report(_force_grabcut, tmp_path):
    img = _solid_green_with_red_subject()
    out = tmp_path / "out"
    r = matte_image(img, output_dir=out, qa=True)
    assert "qa" in r.report
    assert "edge_halo_score_mean" in r.report["qa"]
    assert (out / "matte_qa").exists()


def test_matte_image_rgba_input_passthrough(_force_grabcut):
    """A clean RGBA input should route to passthrough — no matting net."""
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    a = np.clip((40.0 - rad) / 4.0, 0.0, 1.0).astype(np.float32)
    F = np.array([220, 30, 30], dtype=np.float32)
    rgb = (a[..., None] * F).astype(np.uint8)
    rgba = np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)])
    r = matte_image(rgba)
    assert r.strategy_name == "rgba_passthrough"


def test_matte_image_rejects_bad_dtype(_force_grabcut):
    bad = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        matte_image(bad)
