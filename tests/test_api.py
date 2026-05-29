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
    r = matte_image(img, backend="grabcut")
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
    r = matte_image(img, backend="grabcut")

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


def test_matte_image_auto_routes_green_screen_to_corridorkey(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., :3] = (220, 30, 30)
            self.rgba[..., 3] = 255
            self.alpha = np.ones((128, 128), dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros((128, 128), dtype=np.float32)
            self.debug = {"prompt_id": "prompt-auto-corridor", "hint": {"source": "all_white_alpha_hint"}}

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    result = matte_image(_solid_green_with_red_subject(), backend="auto")

    assert result.strategy_name == "comfy_corridorkey"
    assert result.debug["auto_route"]["selected_backend"] == "comfy-corridorkey"
    assert result.debug["auto_route"]["reason"] == "green_screen"


def test_matte_image_auto_routes_unknown_background_to_rmbg(monkeypatch):
    import ermbg.api as api

    captured = {}

    class _Segmenter:
        def segment(self, image, object_prompt=None):
            captured["image_shape"] = image.shape
            alpha = np.zeros(image.shape[:2], dtype=np.float32)
            alpha[32:96, 32:96] = 1.0
            return alpha

    def fake_build_segmenter(**kwargs):
        captured["backend"] = kwargs["backend"]
        return _Segmenter()

    api._SEGMENTER_CACHE.clear()
    monkeypatch.setattr(api, "build_segmenter", fake_build_segmenter)
    image = np.full((128, 128, 3), 180, dtype=np.uint8)
    image[32:96, 32:96] = (20, 40, 180)

    result = matte_image(image, backend="auto", solid_graphic_prepass=False)

    assert captured["backend"] == "comfy-rmbg"
    assert result.report["auto_route"]["selected_backend"] == "comfy-rmbg"
    assert result.report["auto_route"]["reason"] == "unknown_background"


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


def test_matte_image_comfy_corridorkey_uses_remote_pipeline(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    def fail_build_segmenter(**kwargs):
        raise AssertionError("comfy-corridorkey should not build a local segmenter")

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., :3] = (30, 20, 10)
            self.rgba[..., 3] = 255
            self.alpha = np.ones((128, 128), dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = np.ones((128, 128), dtype=np.float32)
            self.color_protection_alpha = np.zeros((128, 128), dtype=np.float32)
            self.debug = {"prompt_id": "prompt-corridor", "hint": {"source": "all_white_alpha_hint"}}

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            assert image_srgb.shape == (128, 128, 3)
            assert kwargs["background_color"] == (0, 200, 0)
            assert kwargs["gamma_space"] == "sRGB"
            assert kwargs["despill_strength"] == 1.0
            assert kwargs["refiner_strength"] == 1.0
            assert kwargs["auto_despeckle"] == "On"
            assert kwargs["despeckle_size"] == 400
            assert kwargs["hint_source"] == "all_white_alpha_hint"
            assert kwargs["hint_alpha"].shape == (128, 128)
            assert np.all(kwargs["hint_alpha"] == 1.0)
            assert kwargs["apply_color_protection"] is True
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", fail_build_segmenter)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="comfy-corridorkey", shadow_mode="on")

    assert r.strategy_name == "comfy_corridorkey"
    assert r.rgba[0, 0].tolist() == [30, 20, 10, 255]
    assert r.report["strategy"]["name"] == "comfy_corridorkey"
    assert r.debug["prompt_id"] == "prompt-corridor"
    assert "corridorkey_hint" in r.debug
    assert r.debug["corridorkey_hint"].mean() == 1.0
    assert r.debug["corridorkey_analysis"]["screen_mode"] == "green"
    assert r.report["strategy"]["bg_type"] == "saturated_green"


def test_matte_image_comfy_corridorkey_patches_shadow_below_subject(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod
    from ermbg import io as ermbg_io

    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (128, 128, 3)).copy()
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:78, 44:84] = 1.0
    shadow = np.zeros((128, 128), dtype=np.float32)
    shadow[76:94, 36:96] = 0.38
    shadow[subject > 0] = 0.0
    bg_lin = ermbg_io.srgb_to_linear(np.broadcast_to(bg, image.shape))
    image = ermbg_io.linear_to_srgb_u8((1.0 - shadow[..., None]) * bg_lin)
    image[subject > 0] = (230, 40, 40)

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.alpha = subject.copy()
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., :3] = (230, 40, 40)
            self.rgba[..., 3] = (self.alpha * 255).astype(np.uint8)
            self.foreground_srgb = self.rgba[..., :3].copy()
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros((128, 128), dtype=np.float32)
            self.debug = {"prompt_id": "prompt-corridor", "hint": {"source": "all_white_alpha_hint"}}

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    result = matte_image(image, backend="comfy-corridorkey", shadow_mode="on")

    assert np.allclose(result.debug["subject_alpha"], subject)
    assert result.debug["shadow"]["source"] == "corridorkey_shadow_patch"
    assert result.debug["shadow"]["detected"] is True
    assert result.debug["shadow_alpha"][82:90, 42:90].mean() > 0.10
    assert result.alpha[82:90, 42:90].mean() > 0.10
    assert result.alpha[40:70, 50:78].mean() == 1.0
    assert result.rgba[50, 60].tolist() == [230, 40, 40, 255]


def test_matte_image_comfy_corridorkey_skips_patch_when_shadow_preserved(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod
    from ermbg import io as ermbg_io

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:78, 44:84] = 1.0
    shadow = np.zeros((128, 128), dtype=np.float32)
    shadow[76:94, 36:96] = 0.38
    shadow[subject > 0] = 0.0
    image = ermbg_io.linear_to_srgb_u8(
        (1.0 - shadow[..., None]) * ermbg_io.srgb_to_linear(np.broadcast_to(bg, (128, 128, 3)))
    )
    image[subject > 0] = (230, 40, 40)
    preserved_alpha = np.maximum(subject, shadow * 0.62).astype(np.float32)

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.alpha = preserved_alpha.copy()
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., :3] = (230, 40, 40)
            self.rgba[..., 3] = (self.alpha * 255).astype(np.uint8)
            self.foreground_srgb = self.rgba[..., :3].copy()
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros((128, 128), dtype=np.float32)
            self.debug = {"prompt_id": "prompt-corridor", "hint": {"source": "all_white_alpha_hint"}}

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    result = matte_image(image, backend="comfy-corridorkey", shadow_mode="on")

    assert np.allclose(result.alpha, preserved_alpha)
    assert result.debug["shadow"]["source"] == "corridorkey_shadow_patch"
    assert result.debug["shadow"]["detected"] is True
    assert result.debug["shadow"]["applied"] is False
    assert result.debug["shadow"]["patch_gate"]["missing_in_corridorkey"] is False
    assert result.debug["shadow_alpha"].max() == 0.0


def test_corridorkey_shadow_patch_gate_filters_preserved_subject_components():
    import ermbg.api as api

    subject = np.zeros((128, 128), dtype=np.float32)
    shadow_display = np.zeros((128, 128), dtype=np.float32)

    shadow_display[82:100, 32:96] = 0.30
    shadow_display[28:46, 44:92] = 0.36
    subject[28:46, 44:92] = 0.58

    filtered, gate = api._corridorkey_shadow_patch_gate(
        subject,
        shadow_display,
        {"detected": True, "accepted_components": 2},
    )

    assert gate["apply"] is True
    assert gate["missing_in_corridorkey"] is True
    assert gate["kept_components"] == 1
    assert filtered[86:96, 40:88].mean() > 0.20
    assert filtered[32:42, 50:86].max() == 0.0
    rejected = [item for item in gate["component_details"] if not item["apply"]]
    assert rejected
    assert rejected[0]["missing_in_corridorkey"] is False


def test_corridorkey_shadow_patch_gate_rejects_broad_vertical_background_wash():
    import ermbg.api as api

    subject = np.zeros((128, 128), dtype=np.float32)
    shadow_display = np.zeros((128, 128), dtype=np.float32)
    shadow_display[8:120, 18:110] = 0.065

    filtered, gate = api._corridorkey_shadow_patch_gate(
        subject,
        shadow_display,
        {"detected": True, "accepted_components": 1},
    )

    assert gate["apply"] is False
    assert gate["missing_in_corridorkey"] is False
    assert gate["reason"] == "shadow candidates rejected as broad background wash or vertical subject residue"
    assert gate["rejected_missing_shape_components"] == 1
    assert gate["component_details"][0]["broad_low_contrast_wash"] is True
    assert gate["component_details"][0]["shadow_like_shape"] is False
    assert filtered.max() == 0.0


def test_matte_image_comfy_corridorkey_can_use_all_white_hint(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., 3] = 255
            self.alpha = np.ones((128, 128), dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = np.ones((128, 128), dtype=np.float32)
            self.color_protection_alpha = np.zeros((128, 128), dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": "all_white_alpha_hint", "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            assert image_srgb.shape == (128, 128, 3)
            assert kwargs["hint_source"] == "all_white_alpha_hint"
            assert kwargs["hint_alpha"].shape == (128, 128)
            assert np.all(kwargs["hint_alpha"] == 1.0)
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="comfy-corridorkey", corridorkey_auto_mask=False)

    assert r.debug["corridorkey_hint"].mean() == 1.0
    assert r.debug["hint"]["source"] == "all_white_alpha_hint"


def test_matte_image_comfy_corridorkey_accepts_hint_mask(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            self.rgba[..., 3] = 255
            self.alpha = np.ones((128, 128), dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = np.ones((128, 128), dtype=np.float32)
            self.color_protection_alpha = np.zeros((128, 128), dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": "provided_corridorkey_hint_mask", "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            assert kwargs["hint_source"] == "provided_corridorkey_hint_mask"
            assert kwargs["hint_alpha"].shape == (128, 128)
            assert 0.20 < float(kwargs["hint_alpha"].mean()) < 0.30
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    img = _solid_green_with_red_subject()
    hint = np.zeros((128, 128), dtype=np.uint8)
    hint[32:96, 32:96] = 255
    r = matte_image(img, backend="comfy-corridorkey", corridorkey_hint_mask=hint)

    assert r.debug["hint"]["source"] == "provided_corridorkey_hint_mask"
    assert 0.20 < float(r.debug["corridorkey_hint"].mean()) < 0.30


def test_matte_image_comfy_corridorkey_blue_analysis_metadata(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            self.rgba[..., 3] = 255
            self.alpha = np.ones((64, 64), dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = np.ones((64, 64), dtype=np.float32)
            self.color_protection_alpha = np.zeros((64, 64), dtype=np.float32)
            self.debug = {"prompt_id": "prompt-blue", "hint": {"source": "all_white_alpha_hint"}}

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            assert image_srgb.shape == (64, 64, 3)
            assert kwargs["background_color"] == (0, 0, 255)
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    img = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
    img[18:46, 18:46] = (220, 30, 30)
    r = matte_image(img, backend="comfy-corridorkey")

    assert r.background_color == (0, 0, 255)
    assert r.report["strategy"]["bg_type"] == "saturated_blue"
    assert r.report["strategy"]["image_type"] == "ai_blue_asset"
    assert r.debug["corridorkey_analysis"]["screen_mode"] == "blue"


def test_matte_image_writes_files_when_output_dir_given(_force_grabcut, tmp_path):
    img = _solid_green_with_red_subject()
    p = tmp_path / "in.png"
    Image.fromarray(img).save(p)
    out = tmp_path / "out"
    r = matte_image(p, backend="grabcut", output_dir=out)
    assert r.output_dir == out
    assert (out / "in_rgba.png").exists()
    assert (out / "in_alpha.png").exists()
    assert (out / "in_shadow.png").exists()
    assert (out / "in_foreground.png").exists()
    assert (out / "in.report.json").exists()


def test_matte_image_qa_adds_metrics_to_report(_force_grabcut, tmp_path):
    img = _solid_green_with_red_subject()
    out = tmp_path / "out"
    r = matte_image(img, backend="grabcut", output_dir=out, qa=True)
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
