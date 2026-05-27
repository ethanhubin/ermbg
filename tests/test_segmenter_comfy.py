"""Tests for ComfyUI-backed segmenter adapters."""

from __future__ import annotations

import numpy as np
import pytest

from ermbg.segmenter import build_segmenter

pytestmark = pytest.mark.core


def test_comfy_rmbg_segmenter_returns_remote_alpha(monkeypatch):
    import ermbg.probe.comfyui_rmbg as comfyui_rmbg

    class _FakeComfyRembg:
        def __init__(self, url, timeout, poll_interval):
            self.url = url
            self.timeout = timeout
            self.poll_interval = poll_interval

        def matte(self, image_srgb):
            h, w = image_srgb.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 3] = 64
            rgba[2:6, 3:7, 3] = 255
            return rgba

    monkeypatch.setattr(comfyui_rmbg, "ComfyUIRembgBaseline", _FakeComfyRembg)

    seg = build_segmenter(backend="comfy-rmbg", url="http://comfy.invalid", timeout=12.0)
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    alpha = seg.segment(image)

    assert alpha.shape == (10, 12)
    assert alpha.dtype == np.float32
    assert np.isclose(alpha[0, 0], 64 / 255.0)
    assert alpha[3, 4] == 1.0
