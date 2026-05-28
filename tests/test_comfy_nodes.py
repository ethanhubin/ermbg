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

torch = pytest.importorskip("torch")

# comfy_nodes/ is meant to be dropped under ComfyUI/custom_nodes, not installed
# as a package. For tests, add the repo root so the import resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from comfy_nodes.ermbg_nodes import ConvertMasksToImages, ErmbgAutoMatte, ErmbgClassify, _dev_reload_ermbg_modules, _mask_to_numpy  # noqa: E402


def _green_with_red_subject(h=128, w=128):
    img = np.full((h, w, 3), [0, 200, 0], dtype=np.uint8)
    img[40:90, 40:90] = (220, 30, 30)
    return img


def _to_comfy_image(arr_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr_uint8.astype(np.float32) / 255.0).unsqueeze(0)


@pytest.fixture
def _force_grabcut(monkeypatch):
    """Force the matting backend to grabcut so tests don't download BiRefNet."""
    import ermbg.api as api

    real = api.build_segmenter
    monkeypatch.setattr(api, "build_segmenter", lambda backend="auto", **kw: real(backend="grabcut"))


def test_classify_node_returns_strings():
    img = _to_comfy_image(_green_with_red_subject())
    node = ErmbgClassify()
    bg, image_type, payload_json = node.run(img)
    assert bg == "saturated"
    assert image_type in ("graphic", "photo")
    assert "saturated_bg" in payload_json


def test_automatte_returns_image_mask_summary(_force_grabcut):
    img = _to_comfy_image(_green_with_red_subject())
    node = ErmbgAutoMatte()
    fg, alpha, summary, rgba_rgb = node.run(
        image=img,
        despill="auto (router decides)",
        use_keyer="auto (router decides)",
        bg_color="0,200,0",
        matting_model="ZhengPeng7/BiRefNet-matting",
        shadow_mode="off",
    )
    # IMAGE convention: [B, H, W, C] float
    assert fg.shape == (1, 128, 128, 3)
    assert fg.dtype == torch.float32
    assert rgba_rgb.shape == (1, 128, 128, 3)
    assert rgba_rgb.dtype == torch.float32
    # MASK convention: [B, H, W] float
    assert alpha.shape == (1, 128, 128)
    assert "saturated_bg" in summary


def test_dev_reload_is_opt_in(monkeypatch):
    monkeypatch.delenv("ERMBG_DEV_RELOAD", raising=False)
    assert _dev_reload_ermbg_modules() == ""

    monkeypatch.setenv("ERMBG_DEV_RELOAD", "1")
    note = _dev_reload_ermbg_modules()
    assert note.startswith("dev_reload=")
    assert "matting" in note


def test_automatte_with_source_mask_passes_through(_force_grabcut):
    """A clean RGBA-style source mask should let the node use passthrough."""
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    a = np.clip((40.0 - rad) / 4.0, 0.0, 1.0).astype(np.float32)
    F = np.array([220, 30, 30], dtype=np.float32)
    rgb = (a[..., None] * F).astype(np.uint8)

    img = _to_comfy_image(rgb)
    mask = torch.from_numpy(a).unsqueeze(0)

    node = ErmbgAutoMatte()
    _, _, summary, _ = node.run(
        image=img,
        despill="auto (router decides)",
        use_keyer="auto (router decides)",
        bg_color="0,200,0",
        matting_model="ZhengPeng7/BiRefNet-matting",
        shadow_mode="off",
        source_mask=mask,
    )
    assert "rgba_passthrough" in summary


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
