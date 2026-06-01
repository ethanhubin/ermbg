from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

import ermbg.direct_worker_client as client_mod


def _rgba_png_base64() -> str:
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_direct_worker_client_omits_unspecified_corridorkey_overrides(monkeypatch):
    captured = {}

    def fake_post(url, *, files, data, timeout):
        captured["url"] = url
        captured["data"] = data
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "rgba_png_base64": _rgba_png_base64(),
                "background": [0, 200, 0],
                "execution_backend": "direct-corridorkey",
            },
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    client_mod.matte_image_direct_worker(np.zeros((2, 3, 3), dtype=np.uint8), direct_worker_url="http://worker.test")

    assert captured["url"] == "http://worker.test/matte"
    assert "corridorkey_auto_mask" not in captured["data"]
    assert "corridorkey_protection_bg_max" not in captured["data"]
    assert "corridorkey_gamma_space" not in captured["data"]


def test_direct_worker_client_sends_explicit_corridorkey_overrides(monkeypatch):
    captured = {}

    def fake_post(url, *, files, data, timeout):
        captured["data"] = data
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "rgba_png_base64": _rgba_png_base64(),
                "background": [0, 200, 0],
                "execution_backend": "direct-corridorkey",
            },
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    client_mod.matte_image_direct_worker(
        np.zeros((2, 3, 3), dtype=np.uint8),
        direct_worker_url="http://worker.test",
        corridorkey_auto_mask=False,
        corridorkey_protection_bg_max=6.0,
        corridorkey_gamma_space="Linear",
    )

    assert captured["data"]["corridorkey_auto_mask"] == "false"
    assert captured["data"]["corridorkey_protection_bg_max"] == "6.0"
    assert captured["data"]["corridorkey_gamma_space"] == "Linear"
