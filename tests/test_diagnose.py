"""Tests for the BackgroundDiagnoser using fixture images and synthetic cases."""

from __future__ import annotations

import numpy as np

from ermbg import io
from ermbg.diagnose import BackgroundDiagnoser


def _make_clean_solid_bg_image(h=160, w=160):
    """Bright disc on a uniform near-black background."""
    bg = np.full((h, w, 3), 8, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    alpha = np.clip((50.0 - r) / 8.0, 0.0, 1.0)
    fg = np.array([220, 60, 80], dtype=np.float32)
    img = bg.astype(np.float32)
    img = alpha[..., None] * fg + (1 - alpha[..., None]) * img
    img = np.clip(img, 0, 255).astype(np.uint8)
    soft = alpha.astype(np.float32)
    return img, soft


def _make_noisy_bg_image(h=160, w=160, sigma=20):
    img, soft = _make_clean_solid_bg_image(h, w)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, sigma, size=img.shape)
    bg_pixels = soft < 0.05
    img = img.astype(np.float32)
    img[bg_pixels] = np.clip(img[bg_pixels] + noise[bg_pixels], 0, 255)
    return img.astype(np.uint8), soft


def test_clean_solid_bg_diagnoses_ready():
    img, soft = _make_clean_solid_bg_image()
    rep = BackgroundDiagnoser().diagnose(img, soft)
    assert rep.purity_passed
    assert rep.edge_contrast_passed
    assert rep.verdict == "ready"


def test_noisy_bg_marked_not_pure():
    img, soft = _make_noisy_bg_image()
    rep = BackgroundDiagnoser().diagnose(img, soft)
    assert not rep.purity_passed
    assert rep.verdict == "not-pure-bg"


def test_low_contrast_subject_marked_risky():
    """Subject color very close to bg color but bg itself is uniform -> risky."""
    h, w = 160, 160
    bg = np.full((h, w, 3), 50, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h // 2, w // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    alpha = np.clip((50.0 - r) / 8.0, 0.0, 1.0)
    fg = np.array([55, 55, 55], dtype=np.float32)  # nearly identical to bg
    img = alpha[..., None] * fg + (1 - alpha[..., None]) * bg.astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    rep = BackgroundDiagnoser().diagnose(img, alpha.astype(np.float32))
    assert rep.purity_passed
    assert not rep.edge_contrast_passed
    assert rep.verdict == "risky"
