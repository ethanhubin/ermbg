"""Smoke tests for the ComfyUI node wrappers.

These don't run inside ComfyUI; they just check our IMAGE↔ndarray plumbing
and that the node returns the right shapes/types. The wrapped pipeline is
exercised by tests/test_api.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

# comfy_nodes/ is meant to be dropped under ComfyUI/custom_nodes, not installed
# as a package. For tests, add the repo root so the import resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfy_nodes.ermbg_nodes import (  # noqa: E402
    ConvertMasksToImages,
    ErmbgClassify,
    ErmbgPyMattingKnownB,
    ErmbgRouteStrategy,
    _LocalCorridorKeyClient,
    _dev_reload_ermbg_modules,
    _mask_to_numpy,
)


def _green_with_red_subject(h=128, w=128):
    img = np.full((h, w, 3), [0, 200, 0], dtype=np.uint8)
    img[40:90, 40:90] = (220, 30, 30)
    return img


def _to_comfy_image(arr_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr_uint8.astype(np.float32) / 255.0).unsqueeze(0)


def test_classify_node_returns_strings():
    img = _to_comfy_image(_green_with_red_subject())
    node = ErmbgClassify()
    bg, image_type, payload_json = node.run(img)
    assert bg == "saturated"
    assert image_type in ("graphic", "photo")
    assert "saturated_bg" in payload_json


def test_route_strategy_node_runs_server_side_router():
    path = (
        Path(__file__).resolve().parent.parent
        / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
    )
    img = _to_comfy_image(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    node = ErmbgRouteStrategy()
    backend, route, asset_kind, payload_json = node.run(
        image=img,
        screen_mode="auto",
        preset="auto",
        fallback_bg_color="0,200,0",
    )
    assert backend == "comfy-pymatting-known-b"
    assert route == "pymatting_known_b"
    assert asset_kind == "button"
    assert '"selected_backend": "comfy-pymatting-known-b"' in payload_json


def test_dev_reload_is_opt_in(monkeypatch):
    monkeypatch.delenv("ERMBG_DEV_RELOAD", raising=False)
    assert _dev_reload_ermbg_modules() == ""

    monkeypatch.setenv("ERMBG_DEV_RELOAD", "1")
    note = _dev_reload_ermbg_modules()
    assert note.startswith("dev_reload=")
    assert "router" in note


def test_pymatting_known_b_node_returns_outputs():
    img = _to_comfy_image(_green_with_red_subject(64, 64))
    node = ErmbgPyMattingKnownB()
    fg, alpha, summary, rgba_rgb, trimap = node.run(
        image=img,
        method="cf",
        image_space="linear",
        bg_source="custom",
        bg_color="0,200,0",
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        auto_adapt=True,
        cg_maxiter=1000,
        cg_rtol=1e-6,
    )

    assert fg.shape == (1, 64, 64, 3)
    assert fg.dtype == torch.float32
    assert alpha.shape == (1, 64, 64)
    assert rgba_rgb.shape == (1, 64, 64, 3)
    assert trimap.shape == (1, 64, 64, 3)
    assert "pymatting_known_b" in summary
    assert "method=cf" in summary
    assert "auto=True" in summary
    assert float(alpha.max()) == pytest.approx(1.0)


def test_comfy_pymatting_known_b_client_renders_workflow():
    from ermbg.probe.comfyui_pymatting_known_b import ComfyUIPyMattingKnownBClient

    client = ComfyUIPyMattingKnownBClient(url="http://example.invalid")
    workflow = client._render_workflow(
        input_image="input.png",
        method="cf",
        image_space="linear",
        bg_source="custom",
        bg_color="0,200,0",
        bg_threshold=3.5,
        fg_threshold=24.0,
        boundary_band_px=2,
        auto_adapt=True,
        cg_maxiter=1000,
        cg_rtol=1e-6,
        filename_prefix="pm",
    )

    assert workflow["20"]["class_type"] == "ErmbgPyMattingKnownB"
    assert workflow["20"]["inputs"]["image"] == ["10", 0]
    assert workflow["20"]["inputs"]["bg_threshold"] == pytest.approx(3.5)
    assert workflow["20"]["inputs"]["boundary_band_px"] == 2
    assert workflow["20"]["inputs"]["auto_adapt"] is True
    assert workflow["50"]["class_type"] == "SaveImage"
    assert workflow["60"]["inputs"]["images"] == ["20", 4]


def test_mask_to_numpy_accepts_batched_and_unbatched_masks():
    mask_2d = torch.ones((32, 48), dtype=torch.float32)
    mask_3d = mask_2d.unsqueeze(0)
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(-1)

    assert _mask_to_numpy(mask_2d).shape == (32, 48)
    assert _mask_to_numpy(mask_3d).shape == (32, 48)
    assert _mask_to_numpy(mask_4d).shape == (32, 48)


def test_convert_masks_to_images_node():
    mask = torch.ones((1, 16, 20), dtype=torch.float32) * 0.5
    images, = ConvertMasksToImages().run(mask)
    assert images.shape == (1, 16, 20, 3)
    assert images.dtype == torch.float32
    assert float(images.min()) == pytest.approx(0.5)


def test_local_corridorkey_client_reuses_loaded_comfy_node(monkeypatch):
    calls = []

    class FakeCorridorKeyNode:
        def run(
            self,
            image,
            mask,
            gamma_space,
            screen_color,
            despill_strength,
            refiner_strength,
            auto_despeckle,
            despeckle_size,
            unique_id=None,
        ):
            calls.append(
                {
                    "gamma_space": gamma_space,
                    "screen_color": screen_color,
                    "refiner_strength": refiner_strength,
                    "auto_despeckle": auto_despeckle,
                    "despeckle_size": despeckle_size,
                    "unique_id": unique_id,
                    "mask_mean": float(mask.mean()),
                }
            )
            alpha = torch.ones_like(mask) * 0.25
            return image, alpha, image, image

    class FakeImageToMaskNode:
        FUNCTION = "image_to_mask"

        def image_to_mask(self, image, channel):
            assert channel == "red"
            return (torch.ones(image.shape[:3], dtype=torch.float32) * 0.30,)

    fake_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "CorridorKey": FakeCorridorKeyNode,
            "ImageToMask": FakeImageToMaskNode,
        }
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_module)
    monkeypatch.setattr(_LocalCorridorKeyClient, "_corridorkey_node", None)

    image = _green_with_red_subject(16, 16)
    result = _LocalCorridorKeyClient().matte(
        image,
        hint_alpha=np.ones((16, 16), dtype=np.float32),
        gamma_space="sRGB",
        screen_color="green",
        refiner_strength=1.15,
        auto_despeckle="Off",
        despeckle_size=64,
        apply_color_protection=False,
    )

    assert calls == [
        {
            "gamma_space": "sRGB",
            "screen_color": "green",
            "refiner_strength": pytest.approx(1.15),
            "auto_despeckle": "Off",
            "despeckle_size": 64,
            "unique_id": None,
            "mask_mean": pytest.approx(0.30),
        }
    ]
    assert result.debug["settings"]["runner"] == "loaded_comfy_node"
    assert result.debug["settings"]["runner_module"] == FakeCorridorKeyNode.__module__
    assert result.debug["corridorkey_mask"]["convention"] == "comfy_image_to_mask_node"
    assert result.debug["corridorkey_mask"]["source_node"] == (
        f"{FakeImageToMaskNode.__module__}.{FakeImageToMaskNode.__name__}"
    )
    assert float(result.alpha.mean()) == pytest.approx(0.25)


def test_local_corridorkey_client_forwards_trusted_material_hint(monkeypatch):
    import ermbg.probe.comfyui_corridorkey as corridorkey_mod

    captured = {}

    class FakeCorridorKeyNode:
        def run(
            self,
            image,
            mask,
            gamma_space,
            screen_color,
            despill_strength,
            refiner_strength,
            auto_despeckle,
            despeckle_size,
            unique_id=None,
        ):
            return image, torch.zeros_like(mask), image, image

    def fake_apply_key_color_protection(
        *,
        image_srgb,
        foreground_srgb,
        alpha,
        background_color,
        thresholds,
        trusted_material_alpha=None,
    ):
        captured["trusted_material_alpha"] = trusted_material_alpha
        floor = (
            np.clip(trusted_material_alpha.astype(np.float32), 0.0, 1.0)
            if trusted_material_alpha is not None
            else np.zeros_like(alpha, dtype=np.float32)
        )
        return foreground_srgb, np.maximum(alpha, floor), floor, {"trusted_material_pixels": int((floor > 0).sum())}

    fake_module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"CorridorKey": FakeCorridorKeyNode})
    monkeypatch.setitem(sys.modules, "nodes", fake_module)
    monkeypatch.setattr(_LocalCorridorKeyClient, "_corridorkey_node", None)
    monkeypatch.setattr(corridorkey_mod, "apply_key_color_protection", fake_apply_key_color_protection)

    image = _green_with_red_subject(16, 16)
    hint = np.zeros((16, 16), dtype=np.float32)
    hint[4:12, 5:11] = 0.75

    result = _LocalCorridorKeyClient().matte(
        image,
        hint_alpha=hint,
        apply_color_protection=True,
        protect_hint_supported_material=True,
        execution_profile="corridorkey-shaped-icon",
    )

    assert np.array_equal(captured["trusted_material_alpha"], hint)
    assert result.debug["settings"]["protect_hint_supported_material"] is True
    assert float(result.alpha.max()) == pytest.approx(0.75)
