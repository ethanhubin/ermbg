"""Tests for the high-level Python API (ermbg.api)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ermbg import MatteResponse, classify_image, matte_image

pytestmark = pytest.mark.core


def _solid_green_with_red_subject(h=128, w=128):
    img = np.full((h, w, 3), [0, 200, 0], dtype=np.uint8)
    img[40:90, 40:90] = (220, 30, 30)
    return img


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


def test_matte_image_ndarray_returns_response():
    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="pymatting-known-b")
    assert isinstance(r, MatteResponse)
    assert r.rgba.shape == (128, 128, 4)
    assert r.rgba.dtype == np.uint8
    assert r.alpha.shape == (128, 128)
    assert r.foreground_srgb.shape == (128, 128, 3)
    assert r.strategy_name == "pymatting_known_b"
    assert r.output_dir is None


def test_matte_image_rejects_removed_legacy_backend():
    img = _solid_green_with_red_subject()
    with pytest.raises(ValueError, match="removed"):
        matte_image(img, backend="grabcut")


def test_matte_image_pymatting_known_b_backend_skips_segmenter(monkeypatch):
    import ermbg.api as api

    def fail_build_segmenter(**kwargs):
        raise AssertionError("pymatting-known-b should not build a segmenter")

    monkeypatch.setattr(api, "build_segmenter", fail_build_segmenter)

    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="pymatting-known-b")

    assert r.strategy_name == "pymatting_known_b"
    assert r.report["strategy"]["name"] == "pymatting_known_b"
    assert r.background_color == (0, 200, 0)
    assert r.debug["pymatting_known_b"]["pymatting"]["method"] == "cf"
    assert r.alpha[44:86, 44:86].mean() > 0.99


def test_matte_image_pymatting_known_b_accepts_parameters():
    img = _solid_green_with_red_subject()
    r = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_method="knn",
        pymatting_image_space="sRGB",
        pymatting_bg_source="custom",
        pymatting_bg_color=(0, 200, 0),
        pymatting_bg_threshold=4.5,
        pymatting_fg_threshold=28.0,
        pymatting_boundary_band_px=3,
        pymatting_auto_adapt=False,
        pymatting_cg_maxiter=1500,
        pymatting_cg_rtol=1e-5,
    )

    params = r.debug["pymatting_known_b"]["parameters"]
    assert r.strategy_name == "pymatting_known_b"
    assert r.background_color == (0, 200, 0)
    assert params["method"] == "knn"
    assert params["image_space"] == "sRGB"
    assert params["bg_source"] == "custom"
    assert params["bg_threshold"] == 4.5
    assert params["fg_threshold"] == 28.0
    assert params["boundary_band_px"] == 3
    assert params["auto_adapt"] is False
    assert params["cg_maxiter"] == 1500
    assert params["cg_rtol"] == 1e-5


def test_matte_image_pymatting_known_b_auto_background_falls_back_when_unstable():
    h = w = 64
    yy = np.linspace(0.0, 24.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 24.0, w, dtype=np.float32)[None, :]
    gray = 154.0 + (xx + yy) * 0.5
    img = np.dstack([gray, gray, gray + 2.0]).astype(np.uint8)
    img[20:46, 22:42] = (220, 40, 30)

    r = matte_image(img, backend="pymatting-known-b", shadow_mode="off")

    background = r.debug["pymatting_known_b"]["background"]
    params = r.debug["pymatting_known_b"]["parameters"]
    assert r.strategy_name == "pymatting_known_b"
    assert background["source"] == "auto_fallback_best_effort"
    assert background["auto_background"]["accepted"] is False
    assert background["auto_background"]["reason"] == "corner/background border is unstable"
    assert params["requested_bg_source"] == "auto"
    assert params["bg_source"] == "custom"
    assert params["auto_adapt"] is False


def test_matte_image_pymatting_known_b_recovers_neutral_ui_shadow():
    img = np.full((128, 128, 3), [0, 200, 0], dtype=np.uint8)
    img[72:98, 24:104] = [0, 120, 0]
    img[40:82, 28:100] = [240, 30, 30]

    r = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(0, 200, 0),
        pymatting_fg_threshold=24.0,
        shadow_mode="on",
    )

    assert r.debug["shadow"]["source"] == "pymatting_known_b_shadow_patch"
    assert r.debug["shadow"]["applied"] is True
    assert r.debug["shadow"]["method"] == "unknown_domain_bidirectional_same_background_reconstruction"
    assert r.debug["trimap_u8"][90, 64] == 128
    assert r.debug["shadow"]["objective_shadow"]["mean_abs_error_after_u8"] < 1.0
    assert r.debug["shadow_alpha"][90, 64] > 0.20
    assert r.alpha[90, 64] > 0.20
    assert tuple(r.rgba[90, 64, :3]) == (0, 0, 0)


def test_pymatting_known_b_shadow_patch_reduces_overdark_raw_subject_alpha():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((48, 64, 3), bg, dtype=np.uint8)
    repair_domain = np.zeros((48, 64), dtype=bool)
    repair_domain[16:32, 20:44] = True
    image[repair_domain] = (0, 180, 0)

    subject_alpha = np.zeros((48, 64), dtype=np.float32)
    subject_alpha[repair_domain] = 0.50
    foreground = np.zeros((48, 64, 3), dtype=np.uint8)

    alpha, rgba_rgb, shadow_alpha, _, info = api._pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        repair_domain=repair_domain,
    )

    assert info["applied"] is True
    assert info["subject_alpha_reduced_pixels"] == int(repair_domain.sum())
    assert info["shadow_pixels"] == 0
    assert shadow_alpha[repair_domain].max() == 0.0
    assert np.allclose(alpha[repair_domain].mean(), 0.10, atol=0.01)


def test_pymatting_known_b_shadow_patch_prefers_source_shadow_over_screen_colored_foreground():
    import ermbg.api as api

    bg = np.array([3, 203, 6], dtype=np.uint8)
    image = np.full((48, 64, 3), bg, dtype=np.uint8)
    repair_domain = np.zeros((48, 64), dtype=bool)
    repair_domain[16:32, 20:44] = True
    image[repair_domain] = (3, 150, 2)

    subject_alpha = np.zeros((48, 64), dtype=np.float32)
    subject_alpha[repair_domain] = 0.55
    foreground = np.zeros((48, 64, 3), dtype=np.uint8)
    foreground[repair_domain] = (3, 61, 0)

    alpha, rgba_rgb, shadow_alpha, _, info = api._pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        repair_domain=repair_domain,
    )

    assert info["applied"] is True
    assert info["shadow_pixels"] == int(repair_domain.sum())
    assert info["subject_alpha_reduced_pixels"] == 0
    assert info["objective_shadow"]["source_shadow_written_pixels"] == int(repair_domain.sum())
    assert float(np.median(shadow_alpha[repair_domain])) > 0.20
    assert np.all(rgba_rgb[repair_domain] == 0)
    assert float(np.median(alpha[repair_domain])) > 0.20
    replay = (
        alpha[..., None] * rgba_rgb.astype(np.float32)
        + (1.0 - alpha[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
    )
    # A single display-space black alpha cannot fit all quantized sRGB
    # channels exactly, but it must be a close replay of the known-B darkening.
    assert np.abs(replay[repair_domain] - image[repair_domain].astype(np.float32)).mean() < 1.5


def test_pymatting_known_b_shadow_patch_extends_source_shadow_to_connected_screen_residue():
    import ermbg.api as api

    bg = np.array([3, 178, 10], dtype=np.uint8)
    image = np.full((48, 64, 3), bg, dtype=np.uint8)
    repair_domain = np.zeros((48, 64), dtype=bool)
    seed = np.zeros((48, 64), dtype=bool)
    residue = np.zeros((48, 64), dtype=bool)
    seed[16:32, 18:30] = True
    residue[16:32, 30:42] = True
    repair_domain |= seed | residue
    image[seed] = (3, 150, 2)
    image[residue] = (6, 145, 10)

    subject_alpha = np.zeros((48, 64), dtype=np.float32)
    subject_alpha[seed] = 0.55
    subject_alpha[residue] = 0.61
    foreground = np.zeros((48, 64, 3), dtype=np.uint8)
    foreground[seed] = (3, 61, 0)
    foreground[residue] = (8, 115, 10)

    alpha, rgba_rgb, shadow_alpha, _, info = api._pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        repair_domain=repair_domain,
    )

    assert info["applied"] is True
    assert info["subject_alpha_reduced_pixels"] == 0
    assert info["objective_shadow"]["source_shadow_seed_written_pixels"] == int(seed.sum())
    assert info["objective_shadow"]["source_shadow_connected_written_pixels"] == int(residue.sum())
    assert info["objective_shadow"]["source_shadow_written_pixels"] == int(repair_domain.sum())
    assert np.all(rgba_rgb[repair_domain] == 0)
    assert float(np.median(alpha[residue])) > 0.15
    assert float(np.median(shadow_alpha[residue])) > 0.15


def test_matte_image_comfy_pymatting_known_b_uses_remote_node(monkeypatch):
    import ermbg.probe.comfyui_pymatting_known_b as remote_module

    captured = {}

    class FakeClient:
        def __init__(self, url):
            captured["url"] = url

        def matte(self, image_srgb, **kwargs):
            captured["shape"] = image_srgb.shape
            captured["kwargs"] = kwargs
            alpha = np.zeros(image_srgb.shape[:2], dtype=np.float32)
            alpha[40:90, 40:90] = 1.0
            fg = image_srgb.copy()
            rgba = np.dstack([fg, (alpha * 255 + 0.5).astype(np.uint8)])
            return remote_module.ComfyPyMattingKnownBResult(
                rgba=rgba,
                alpha=alpha,
                foreground_srgb=fg,
                trimap_u8=(alpha * 255 + 0.5).astype(np.uint8),
                debug={"prompt_id": "fake-prompt", "backend": "comfy-pymatting-known-b"},
            )

    monkeypatch.setattr(remote_module, "ComfyUIPyMattingKnownBClient", FakeClient)

    img = _solid_green_with_red_subject()
    r = matte_image(
        img,
        backend="comfy-pymatting-known-b",
        comfy_url="http://example.invalid:8000",
        pymatting_method="knn",
        pymatting_image_space="sRGB",
        pymatting_bg_source="custom",
        pymatting_bg_color=(0, 200, 0),
        pymatting_bg_threshold=4.5,
        pymatting_fg_threshold=28.0,
        pymatting_boundary_band_px=3,
        pymatting_auto_adapt=False,
        pymatting_cg_maxiter=1500,
        pymatting_cg_rtol=1e-5,
    )

    assert r.strategy_name == "comfy_pymatting_known_b"
    assert r.report["strategy"]["name"] == "comfy_pymatting_known_b"
    assert r.background_color == (0, 200, 0)
    assert captured["url"] == "http://example.invalid:8000"
    assert captured["kwargs"]["method"] == "knn"
    assert captured["kwargs"]["image_space"] == "sRGB"
    assert captured["kwargs"]["bg_source"] == "custom"
    assert captured["kwargs"]["bg_color"] == (0, 200, 0)
    assert captured["kwargs"]["bg_threshold"] == 4.5
    assert captured["kwargs"]["fg_threshold"] == 28.0
    assert captured["kwargs"]["boundary_band_px"] == 3
    assert captured["kwargs"]["auto_adapt"] is False
    assert captured["kwargs"]["cg_maxiter"] == 1500
    assert captured["kwargs"]["cg_rtol"] == 1e-5
    assert r.debug["pymatting_known_b"]["remote"]["prompt_id"] == "fake-prompt"


def test_matte_image_auto_routes_square_green_screen_icon_to_corridorkey(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_route_matte as remote_mod

    class _FakeClient:
        def __init__(self, url, poll_interval=0.05):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            auto_route = {
                "selected_backend": "comfy-corridorkey",
                "route": "corridorkey",
                "asset_kind": "icon",
            }
            rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            rgba[..., :3] = (220, 30, 30)
            rgba[..., 3] = 255
            alpha = np.ones((128, 128), dtype=np.float32)
            return remote_mod.ComfyRouteMatteResult(
                rgba=rgba,
                alpha=alpha,
                foreground_srgb=rgba[..., :3],
                background_color=(0, 200, 0),
                strategy_name="comfy_corridorkey",
                report={"auto_route": auto_route, "strategy": {"name": "comfy_corridorkey"}},
                debug={"auto_route": auto_route, "backend": "comfy-corridorkey"},
            )

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(remote_mod, "ComfyUIRouteMatteClient", _FakeClient)

    result = matte_image(_solid_green_with_red_subject(), backend="auto")

    assert result.strategy_name == "comfy_corridorkey"
    assert result.debug["auto_route"]["selected_backend"] == "comfy-corridorkey"
    assert result.debug["auto_route"]["route"] == "corridorkey"
    assert result.debug["auto_route"]["asset_kind"] == "icon"


def test_matte_image_auto_routes_hard_button_to_comfy_pymatting(monkeypatch):
    import ermbg.probe.comfyui_route_matte as remote_module

    captured = {}

    class FakeClient:
        def __init__(self, url, poll_interval=0.05):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            h, w = image_srgb.shape[:2]
            alpha = np.zeros((h, w), dtype=np.float32)
            alpha[24:104, 56:200] = 1.0
            auto_route = {
                "selected_backend": "comfy-pymatting-known-b",
                "route": "pymatting_known_b",
                "asset_kind": "button",
            }
            return remote_module.ComfyRouteMatteResult(
                rgba=np.dstack([image_srgb, (alpha * 255).astype(np.uint8)]),
                alpha=alpha,
                foreground_srgb=image_srgb.copy(),
                background_color=(0, 200, 0),
                strategy_name="comfy_pymatting_known_b",
                report={"auto_route": auto_route, "strategy": {"name": "comfy_pymatting_known_b"}},
                debug={"auto_route": auto_route, "backend": "comfy-pymatting-known-b"},
            )

    monkeypatch.setattr(remote_module, "ComfyUIRouteMatteClient", FakeClient)
    path = (
        Path(__file__).resolve().parents[1]
        / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    )

    result = matte_image(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), backend="auto")

    assert result.strategy_name == "comfy_pymatting_known_b"
    assert result.debug["auto_route"]["selected_backend"] == "comfy-pymatting-known-b"
    assert result.debug["auto_route"]["route"] == "pymatting_known_b"
    assert result.debug["auto_route"]["asset_kind"] == "button"
    assert captured["pymatting_bg_source"] == "auto"
    assert captured["pymatting_bg_color"] is None


def test_matte_image_auto_routes_stable_non_green_blue_background_to_comfy_pymatting(monkeypatch):
    import ermbg.probe.comfyui_route_matte as remote_module

    captured = {}

    class FakeClient:
        def __init__(self, url, poll_interval=0.05):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            h, w = image_srgb.shape[:2]
            alpha = np.zeros((h, w), dtype=np.float32)
            alpha[32:96, 32:96] = 1.0
            auto_route = {
                "selected_backend": "comfy-pymatting-known-b",
                "route": "pymatting_known_b",
                "asset_kind": "button",
            }
            return remote_module.ComfyRouteMatteResult(
                rgba=np.dstack([image_srgb, (alpha * 255).astype(np.uint8)]),
                alpha=alpha,
                foreground_srgb=image_srgb.copy(),
                background_color=(180, 180, 180),
                strategy_name="comfy_pymatting_known_b",
                report={"auto_route": auto_route, "strategy": {"name": "comfy_pymatting_known_b"}},
                debug={"auto_route": auto_route, "backend": "comfy-pymatting-known-b"},
            )

    monkeypatch.setattr(remote_module, "ComfyUIRouteMatteClient", FakeClient)
    image = np.full((128, 128, 3), 180, dtype=np.uint8)
    image[32:96, 32:96] = (20, 40, 180)

    result = matte_image(image, backend="auto", solid_graphic_prepass=False)

    assert result.strategy_name == "comfy_pymatting_known_b"
    assert result.report["auto_route"]["selected_backend"] == "comfy-pymatting-known-b"
    assert result.report["auto_route"]["route"] == "pymatting_known_b"
    assert captured["pymatting_bg_source"] == "auto"


def test_matte_image_auto_routes_unstable_unknown_background_to_pymatting_fallback(monkeypatch):
    import ermbg.probe.comfyui_route_matte as remote_mod

    captured = {}

    class _FakePyMattingFallback:
        def __init__(self, url, poll_interval=0.05):
            captured["url"] = url

        def matte(self, image_srgb, **kwargs):
            captured["image_shape"] = image_srgb.shape
            auto_route = {
                "selected_backend": "comfy-pymatting-known-b",
                "route": "pymatting_fallback",
                "asset_kind": "unknown_fallback",
            }
            rgba = np.dstack([image_srgb, np.full(image_srgb.shape[:2], 255, dtype=np.uint8)])
            alpha = np.ones(image_srgb.shape[:2], dtype=np.float32)
            return remote_mod.ComfyRouteMatteResult(
                rgba=rgba,
                alpha=alpha,
                foreground_srgb=image_srgb.copy(),
                background_color=(0, 200, 0),
                strategy_name="comfy_pymatting_known_b",
                report={"auto_route": auto_route, "strategy": {"name": "comfy_pymatting_known_b"}},
                debug={"auto_route": auto_route, "backend": "comfy-pymatting-known-b"},
            )

    monkeypatch.setattr(remote_mod, "ComfyUIRouteMatteClient", _FakePyMattingFallback)
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    image[32:96, 32:96] = (20, 40, 180)

    result = matte_image(image, backend="auto", solid_graphic_prepass=False)

    assert captured["image_shape"] == (128, 128, 3)
    assert result.report["auto_route"]["selected_backend"] == "comfy-pymatting-known-b"
    assert result.report["auto_route"]["route"] == "pymatting_fallback"


def test_matte_image_comfy_rmbg_uses_remote_full_pipeline(monkeypatch):
    import ermbg.probe.comfyui_rmbg as remote_mod

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb):
            assert image_srgb.shape == (128, 128, 3)
            rgba = np.zeros((128, 128, 4), dtype=np.uint8)
            rgba[..., :3] = (10, 20, 30)
            rgba[..., 3] = 255
            return rgba

    monkeypatch.setattr(remote_mod, "ComfyUIRembgBaseline", _FakeClient)

    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="comfy-rmbg", shadow_mode="off")

    assert r.strategy_name == "comfy_rmbg"
    assert r.rgba[0, 0].tolist() == [10, 20, 30, 255]
    assert r.report["strategy"]["name"] == "comfy_rmbg"
    assert r.debug["backend"] == "comfy-rmbg"


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
            assert kwargs["color_protection_bg_max"] == 12.0
            assert kwargs["color_protection_fg_min"] == 28.0
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


def test_matte_image_comfy_corridorkey_explicit_color_protection_false_overrides_auto(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, apply_color_protection):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.rgba[..., 3] = 255
            self.alpha = np.ones(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": "known_bg_hard_ui_bbox_2px_hint"},
                "settings": {"apply_color_protection": apply_color_protection},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["apply_color_protection"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    result = matte_image(
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8),
        backend="comfy-corridorkey",
        corridorkey_preset="auto",
        corridorkey_color_protection=False,
    )

    assert result.debug["corridorkey_analysis"]["recommended_settings"]["color_protection"] is True
    assert captured["apply_color_protection"] is False
    assert result.report["strategy"]["extras"]["settings"]["apply_color_protection"] is False


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


def test_corridorkey_shadow_patch_gate_replaces_under_reconstructed_hard_shadow():
    import ermbg.api as api

    subject = np.zeros((128, 128), dtype=np.float32)
    shadow_display = np.zeros((128, 128), dtype=np.float32)
    shadow_display[78:96, 32:104] = 0.30
    subject[78:96, 32:104] = 0.12

    filtered, gate = api._corridorkey_shadow_patch_gate(
        subject,
        shadow_display,
        {"detected": True, "accepted_components": 1},
    )

    assert gate["apply"] is True
    assert gate["missing_in_corridorkey"] is True
    assert gate["component_details"][0]["under_reconstructed_shadow"] is True
    assert filtered[82:92, 40:96].mean() > 0.25

    preserved_subject = np.zeros((128, 128), dtype=np.float32)
    preserved_subject[78:96, 32:104] = 0.19
    preserved_filtered, preserved_gate = api._corridorkey_shadow_patch_gate(
        preserved_subject,
        shadow_display,
        {"detected": True, "accepted_components": 1},
    )

    assert preserved_gate["apply"] is False
    assert preserved_gate["missing_in_corridorkey"] is False
    assert preserved_filtered.max() == 0.0


def test_corridorkey_shadow_patch_removes_weak_ck_residue_in_patched_shadow():
    import ermbg.api as api
    from ermbg import io as ermbg_io

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:76, 44:92] = 1.0
    shadow = np.zeros((128, 128), dtype=np.float32)
    shadow[80:96, 32:112] = 0.42
    image = ermbg_io.linear_to_srgb_u8(
        (1.0 - shadow[..., None]) * ermbg_io.srgb_to_linear(np.broadcast_to(bg, (128, 128, 3)))
    )
    image[subject > 0] = (230, 40, 40)
    ck_alpha = np.maximum(subject, shadow * 0.20).astype(np.float32)
    protected_edge = np.zeros((128, 128), dtype=bool)
    protected_edge[80:96, 32:40] = True
    ck_alpha[protected_edge] = 0.62
    foreground = np.zeros((128, 128, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)

    alpha, _, shadow_alpha, _, info = api._corridorkey_shadow_patch(
        image,
        subject_alpha=ck_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        shadow_mode="on",
    )

    patch_region = shadow > 0
    low_shadow_region = patch_region & ~protected_edge
    assert info["applied"] is True
    assert info["patch_gate"]["component_details"][0]["under_reconstructed_shadow"] is True
    assert info["patch_gate"]["corridorkey_shadow_residue_pixels_removed"] > 0
    assert shadow_alpha[protected_edge].max() == 0.0
    assert np.allclose(alpha[protected_edge], ck_alpha[protected_edge])
    assert shadow_alpha[low_shadow_region].mean() > ck_alpha[low_shadow_region].mean()
    assert np.allclose(alpha[low_shadow_region].mean(), shadow_alpha[low_shadow_region].mean(), atol=0.03)


def test_near_subject_shadow_bridge_rejects_outline_scale_expansion():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    subject = np.zeros((96, 128), dtype=np.float32)
    foreground = np.zeros((96, 128, 3), dtype=np.uint8)
    shadow = np.zeros((96, 128), dtype=np.float32)

    core = np.zeros((96, 128), dtype=bool)
    core[28:60, 26:102] = True
    shadow_seed = np.zeros((96, 128), dtype=bool)
    shadow_seed[62:64, 26:102] = True
    gap = np.zeros((96, 128), dtype=bool)
    gap[60:62, 26:102] = True

    subject[core] = 1.0
    foreground[core] = (245, 180, 32)
    image[core] = (245, 180, 32)
    shadow[shadow_seed] = 0.30
    image[shadow_seed] = ((1.0 - shadow[shadow_seed, None]) * bg.reshape(1, 3) + 0.5).astype(np.uint8)

    refined, info = api._refine_near_subject_shadow_from_source_pixels(
        shadow,
        subject,
        image,
        tuple(int(c) for c in bg),
        foreground,
    )

    # Mechanism: contact-gap bridging is only a seam repair. If the would-be
    # bridge is larger than the accepted near-subject repair support, it would
    # expand the whole cast shadow along the UI outline, so it must be reported
    # and rejected instead of being written into the shadow alpha.
    assert info["contact_gap_bridge_rejected_as_expansion"] is True
    assert info["contact_gap_bridge_pixels"] == 0
    assert info["rejected_contact_gap_bridge_pixels"] >= int(gap.sum() * 0.8)
    assert refined[gap].max() == 0.0


def test_corridorkey_shadow_patch_uses_source_pixels_as_reprojection_target():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:74, 44:92] = 1.0
    target_display_shadow = np.zeros((128, 128), dtype=np.float32)
    horizontal = np.linspace(0.22, 0.32, 80, dtype=np.float32)
    target_display_shadow[80:96, 32:112] = horizontal[None, :]

    image = (
        (1.0 - target_display_shadow[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
        + 0.5
    ).astype(np.uint8)
    image[subject > 0] = (230, 40, 40)
    ck_alpha = np.maximum(subject, target_display_shadow * 0.28).astype(np.float32)
    foreground = np.zeros((128, 128, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)

    _, _, shadow_alpha, _, info = api._corridorkey_shadow_patch(
        image,
        subject_alpha=ck_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        shadow_mode="on",
    )

    shadow_region = target_display_shadow > 0.0
    reprojection = info["patch_gate"]["source_reprojection"]
    assert info["applied"] is True
    assert reprojection["enabled"] is True
    assert reprojection["mean_abs_error_after_u8"] < reprojection["mean_abs_error_before_u8"]
    assert np.mean(np.abs(shadow_alpha[shadow_region] - target_display_shadow[shadow_region])) < 0.012


def test_corridorkey_shadow_patch_preserves_subject_antialiasing_at_contact_edge():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:78, 44:92] = 1.0
    subject[78, 44:92] = 0.20
    shadow = np.zeros((128, 128), dtype=np.float32)
    shadow[78:96, 32:112] = 0.30
    image = (
        (1.0 - shadow[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
        + 0.5
    ).astype(np.uint8)
    image[subject >= 1.0] = (230, 40, 40)
    image[subject == 0.20] = (184, 168, 0)
    ck_alpha = np.maximum(subject, shadow * 0.28).astype(np.float32)
    foreground = np.zeros((128, 128, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)

    alpha, rgba_rgb, shadow_alpha, _, info = api._corridorkey_shadow_patch(
        image,
        subject_alpha=ck_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        shadow_mode="on",
    )

    contact_edge = np.zeros((128, 128), dtype=bool)
    contact_edge[78, 44:92] = True
    exterior_shadow = np.zeros((128, 128), dtype=bool)
    exterior_shadow[88:94, 44:92] = True
    assert info["applied"] is True
    assert info["patch_gate"]["corridorkey_subject_edge_pixels_preserved"] >= int(contact_edge.sum())
    assert np.all(alpha[contact_edge] > shadow_alpha[contact_edge])
    assert rgba_rgb[contact_edge, 0].mean() > 40.0
    assert np.allclose(alpha[exterior_shadow].mean(), shadow_alpha[exterior_shadow].mean(), atol=0.03)


def test_corridorkey_shadow_patch_reprojects_near_subject_region():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((64, 96), dtype=np.float32)
    subject[18:34, 24:72] = 1.0
    subject[34, 24:72] = 0.18
    shadow = np.zeros((64, 96), dtype=np.float32)
    shadow[36:44, 20:76] = 0.32
    source_shadow = np.zeros((64, 96), dtype=np.float32)
    source_shadow[34:44, 20:76] = 0.32
    foreground = np.zeros((64, 96, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)
    image = (
        subject[..., None] * foreground.astype(np.float32)
        + (1.0 - subject[..., None]) * (1.0 - source_shadow[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
        + 0.5
    ).astype(np.uint8)

    repaired, info = api._refine_near_subject_shadow_from_source_pixels(
        shadow,
        subject,
        image,
        tuple(int(c) for c in bg),
        foreground,
    )

    gap = np.zeros_like(subject, dtype=bool)
    gap[35, 24:72] = True
    subject_edge = np.zeros_like(subject, dtype=bool)
    subject_edge[34, 24:72] = True
    assert info["repair_pixels"] > 0
    assert info["source_added_pixels"] >= int(gap.sum())
    assert info["source_reproject_pixels"] > 0
    assert info["mean_abs_error_after_u8"] < info["mean_abs_error_before_u8"]
    assert repaired[gap].mean() > 0.20
    assert repaired[subject_edge].mean() > 0.20


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


def test_matte_image_comfy_corridorkey_hard_hint_requires_auto_mask(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.ones(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=False,
        corridorkey_hard_ui_hint_mode="boundary_2px_shadow_safe_edge_floor",
    )

    # Mechanism: hard-UI hint is an automatic-mask strategy. If automatic
    # hinting is off, the selected strategy is inert and CorridorKey receives
    # the all-white control hint.
    assert captured["hint_source"] == "all_white_alpha_hint"
    assert np.all(captured["hint_alpha"] == 1.0)
    assert result.debug["hard_ui_hint"]["solid_interior_pixels"] == 0


def test_matte_image_comfy_corridorkey_all_white_is_auto_mask_strategy(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.ones(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_hard_ui_hint_mode="all_white",
    )

    assert captured["hint_source"] == "all_white_alpha_hint"
    assert np.all(captured["hint_alpha"] == 1.0)


def test_matte_image_comfy_corridorkey_auto_glass_uses_all_white_hint(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.ones(hint_alpha.shape, dtype=np.float32) * 0.5
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_real_glass_green_bg_yellow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_hard_ui_hint_mode="bbox_2px",
    )

    assert captured["hint_source"] == "glass_all_white_corridorkey_hint"
    assert captured["execution_profile"] == "corridorkey-transparent-button"
    assert np.all(captured["hint_alpha"] == 1.0)
    assert captured["apply_color_protection"] is False
    assert result.debug["hard_ui_hint"]["solid_interior_pixels"] == 0


def test_matte_image_comfy_corridorkey_key_color_material_passes_hint_supported_protection(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, shape, hint_alpha, hint_source):
            self.rgba = np.zeros((*shape, 4), dtype=np.uint8)
            self.alpha = np.ones(shape, dtype=np.float32) * 0.5
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = np.zeros(shape, dtype=np.float32) if hint_alpha is None else hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(self.hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(image_srgb.shape[:2], kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/icon/icon_icon_a03_hard_boundary_weak_contrast/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
    )

    assert result.debug["corridorkey_analysis"]["parameter_profile"] == "key_color_material"
    assert captured["protect_hint_supported_material"] is True
    assert captured["apply_color_protection"] is True


def test_matte_image_comfy_corridorkey_composite_character_uses_all_white_no_protection(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, shape, hint_alpha, hint_source):
            self.rgba = np.zeros((*shape, 4), dtype=np.uint8)
            self.alpha = np.ones(shape, dtype=np.float32) * 0.5
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = np.zeros(shape, dtype=np.float32) if hint_alpha is None else hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(self.hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(image_srgb.shape[:2], kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = (
        Path(__file__).resolve().parents[1]
        / "samples/corridorkey_semantic/character/character_char_a06_pale_hair_translucent_sleeves_white_glow_blue/blue.png"
    )
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_color_protection=True,
    )

    assert result.debug["corridorkey_analysis"]["parameter_profile"] == "composite_character_corridor_only"
    assert captured["hint_source"] == "character_all_white_corridorkey_hint"
    assert captured["execution_profile"] == "corridorkey-character"
    assert np.all(captured["hint_alpha"] == 1.0)
    assert captured["apply_color_protection"] is False
    assert captured["protect_hint_supported_material"] is False


def test_matte_image_comfy_corridorkey_explicit_translucent_hint_forces_glass_settings(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.ones(hint_alpha.shape, dtype=np.float32) * 0.5
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_real_glass_blue_bg_yellow/blue.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_hard_ui_hint_mode="translucent_button",
    )

    assert captured["hint_source"] == "glass_all_white_corridorkey_hint"
    assert np.all(captured["hint_alpha"] == 1.0)
    assert captured["apply_color_protection"] is False
    assert captured["refiner_strength"] == 1.15
    assert captured["auto_despeckle"] == "Off"
    assert captured["despeckle_size"] == 64
    assert result.debug["hard_ui_hint"]["forced_translucent_settings"] is True


def test_matte_image_comfy_corridorkey_uses_hard_ui_hint_for_opaque_buttons(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.rgba[..., 3] = 255
            self.alpha = np.ones(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = self.rgba[..., :3]
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": "known_bg_hard_ui_bbox_2px_hint", "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    r = matte_image(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), backend="comfy-corridorkey", corridorkey_auto_mask=True)

    assert captured["hint_source"] == "known_bg_hard_ui_bbox_2px_hint"
    assert captured["hint_alpha"].shape == (128, 256)
    assert 0.20 < float(captured["hint_alpha"].mean()) < 0.35
    assert r.debug["hint"]["source"] == "known_bg_hard_ui_bbox_2px_hint"


def test_matte_image_comfy_corridorkey_boundary_hint_restores_hard_ui_interior(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod
    from ermbg.probe.comfyui_corridorkey import build_hard_ui_solid_interior_mask

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = np.zeros((*hint_alpha.shape, 3), dtype=np.uint8)
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_hard_ui_hint_mode="boundary_2px",
    )
    interior = build_hard_ui_solid_interior_mask(image, (0, 200, 0))

    assert captured["hint_source"] == "known_bg_hard_ui_boundary_2px_hint"
    assert captured["hint_alpha"].mean() < 0.10
    assert result.debug["hard_ui_hint"]["mode"] == "boundary_2px"
    assert result.debug["hard_ui_hint"]["solid_interior_pixels"] == int(interior.sum())
    assert result.debug["subject_alpha"][interior].min() == 1.0
    assert np.all(result.foreground_srgb[interior] == image[interior])


def test_matte_image_comfy_corridorkey_shadow_safe_boundary_hint_excludes_shadow_interior(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod
    from ermbg.probe.comfyui_corridorkey import (
        build_hard_ui_shadow_safe_solid_interior_mask,
        build_hard_ui_solid_interior_mask,
    )

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = np.zeros((*hint_alpha.shape, 3), dtype=np.uint8)
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_b_unoutlined_hard_heavy_shadow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_hard_ui_hint_mode="boundary_2px_shadow_safe",
    )
    base_interior = build_hard_ui_solid_interior_mask(image, (0, 200, 0))
    safe_interior, safe_info = build_hard_ui_shadow_safe_solid_interior_mask(image, (0, 200, 0))

    assert captured["hint_source"] == "known_bg_hard_ui_boundary_2px_shadow_safe_hint"
    assert result.debug["hard_ui_hint"]["mode"] == "boundary_2px_shadow_safe"
    assert result.debug["hard_ui_hint"]["solid_interior_pixels"] == int(safe_interior.sum())
    assert result.debug["hard_ui_hint"]["shadow_excluded_interior_pixels"] == safe_info["shadow_excluded_interior_pixels"]
    assert int(safe_interior.sum()) < int(base_interior.sum())
    assert result.debug["subject_alpha"][safe_interior].min() == 1.0


def test_matte_image_comfy_corridorkey_edge_floor_lifts_unoutlined_material_band(monkeypatch):
    import ermbg.api as api
    import ermbg.probe.comfyui_corridorkey as remote_mod
    from ermbg.probe.comfyui_corridorkey import (
        build_hard_ui_shadow_safe_material_alpha_floor,
        build_hard_ui_solid_interior_mask,
    )

    captured = {}

    class _FakeRemoteResult:
        def __init__(self, hint_alpha, hint_source):
            self.rgba = np.zeros((*hint_alpha.shape, 4), dtype=np.uint8)
            self.alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.foreground_srgb = np.zeros((*hint_alpha.shape, 3), dtype=np.uint8)
            self.hint_alpha = hint_alpha
            self.raw_alpha = self.alpha.copy()
            self.color_protection_alpha = np.zeros(hint_alpha.shape, dtype=np.float32)
            self.debug = {
                "prompt_id": "prompt-corridor",
                "hint": {"source": hint_source, "mean": float(hint_alpha.mean())},
            }

    class _FakeClient:
        def __init__(self, url):
            self.url = url

        def matte(self, image_srgb, **kwargs):
            captured.update(kwargs)
            return _FakeRemoteResult(kwargs["hint_alpha"], kwargs["hint_source"])

    monkeypatch.setattr(api, "build_segmenter", lambda **kwargs: None)
    monkeypatch.setattr(remote_mod, "ComfyUICorridorKeyClient", _FakeClient)

    path = Path(__file__).resolve().parents[1] / "samples/corridorkey_semantic/button/button_green_yellow_b_unoutlined_no_shadow/green.png"
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = matte_image(
        image,
        backend="comfy-corridorkey",
        corridorkey_auto_mask=True,
        corridorkey_hard_ui_hint_mode="boundary_2px_shadow_safe_edge_floor",
    )
    interior = build_hard_ui_solid_interior_mask(image, (0, 200, 0))
    floor, floor_info = build_hard_ui_shadow_safe_material_alpha_floor(image, (0, 200, 0))
    edge_floor = (floor > 0.0) & ~interior

    assert captured["hint_source"] == "known_bg_hard_ui_boundary_2px_shadow_safe_edge_floor_hint"
    assert result.debug["hard_ui_hint"]["mode"] == "boundary_2px_shadow_safe_edge_floor"
    assert result.debug["hard_ui_hint"]["material_floor_pixels"] == floor_info["material_floor_pixels"]
    assert result.debug["hard_ui_hint"]["material_floor_lift_pixels"] == int((floor > 0.0).sum())
    assert edge_floor.sum() > 100
    assert np.all(result.debug["subject_alpha"][edge_floor] >= floor[edge_floor])


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


def test_matte_image_writes_files_when_output_dir_given(tmp_path):
    img = _solid_green_with_red_subject()
    p = tmp_path / "in.png"
    Image.fromarray(img).save(p)
    out = tmp_path / "out"
    r = matte_image(p, backend="pymatting-known-b", output_dir=out)
    assert r.output_dir == out
    assert (out / "in_rgba.png").exists()
    assert (out / "in_alpha.png").exists()
    assert (out / "in_shadow.png").exists()
    assert (out / "in_foreground.png").exists()
    assert (out / "in.report.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "ermbg.run.v1"
    assert manifest["outputs"]["rgba"] == "in_rgba.png"
    assert manifest["outputs"]["alpha"] == "in_alpha.png"
    assert manifest["request"]["backend"] == "pymatting-known-b"
    assert manifest["report"] == "in.report.json"


def test_matte_image_qa_adds_metrics_to_report(tmp_path):
    img = _solid_green_with_red_subject()
    out = tmp_path / "out"
    r = matte_image(img, backend="pymatting-known-b", output_dir=out, qa=True)
    assert "qa" in r.report
    assert "edge_halo_score_mean" in r.report["qa"]
    assert (out / "matte_qa").exists()


def test_matte_image_rgba_input_passthrough(monkeypatch):
    """A clean RGBA input should route to passthrough — no matting net."""
    import ermbg.probe.comfyui_route_matte as remote_module

    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    a = np.clip((40.0 - rad) / 4.0, 0.0, 1.0).astype(np.float32)
    F = np.array([220, 30, 30], dtype=np.float32)
    rgb = (a[..., None] * F).astype(np.uint8)
    rgba = np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)])

    class FakeClient:
        def __init__(self, url, poll_interval=0.05):
            pass

        def matte(self, image_srgb, **kwargs):
            auto_route = {
                "selected_backend": "passthrough",
                "route": "rgba_passthrough",
                "asset_kind": "rgba",
            }
            alpha = kwargs["source_alpha"]
            out = np.dstack([image_srgb, (alpha * 255 + 0.5).astype(np.uint8)])
            return remote_module.ComfyRouteMatteResult(
                rgba=out,
                alpha=alpha,
                foreground_srgb=image_srgb,
                background_color=(0, 0, 0),
                strategy_name="rgba_passthrough",
                report={"auto_route": auto_route, "strategy": {"name": "rgba_passthrough"}},
                debug={"auto_route": auto_route, "backend": "passthrough"},
            )

    monkeypatch.setattr(remote_module, "ComfyUIRouteMatteClient", FakeClient)
    r = matte_image(rgba)
    assert r.strategy_name == "rgba_passthrough"


def test_matte_image_rejects_bad_dtype():
    bad = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        matte_image(bad)
