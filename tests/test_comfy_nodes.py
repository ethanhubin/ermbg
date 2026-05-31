"""Smoke tests for the ComfyUI node wrappers.

These don't run inside ComfyUI; they just check our IMAGE↔ndarray plumbing
and that the node returns the right shapes/types. The wrapped pipeline is
exercised by tests/test_api.py.
"""

from __future__ import annotations

import sys
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
