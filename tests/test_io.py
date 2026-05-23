from __future__ import annotations

import numpy as np

from ermbg import io


def test_srgb_linear_roundtrip():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    lin = io.srgb_to_linear(src)
    assert lin.dtype == np.float32
    back = io.linear_to_srgb_u8(lin)
    # sRGB -> linear -> sRGB roundtrip should be exact within +/-1.
    assert np.max(np.abs(back.astype(int) - src.astype(int))) <= 1


def test_save_load_rgb(tmp_path):
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    img[..., 0] = 255
    p = tmp_path / "x.png"
    io.save_rgb(p, img)
    out = io.load_rgb(p)
    assert out.shape == img.shape
    assert (out == img).all()


def test_save_mask_float(tmp_path):
    m = np.linspace(0, 1, 32, dtype=np.float32).reshape(4, 8)
    p = tmp_path / "m.png"
    io.save_mask(p, m)
    out = io.load_rgb(p)  # PIL returns RGB; the L channel was duplicated
    assert out.shape == (4, 8, 3)
