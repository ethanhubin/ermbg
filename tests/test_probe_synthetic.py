from __future__ import annotations

import numpy as np

from ermbg.probe.synthetic import SyntheticProbeGenerator


def test_synthetic_paints_background(synth_image):
    image, mask = synth_image
    gen = SyntheticProbeGenerator()
    bg = (255, 255, 255)
    out = gen.generate(image, mask, bg)
    assert out.shape == image.shape and out.dtype == np.uint8

    # Far-from-subject pixels (mask < 0.05) should be near white.
    bg_pixels = out[mask < 0.05]
    assert bg_pixels.size > 0
    assert (bg_pixels.astype(int) > 240).all()


def test_synthetic_preserves_subject_inside(synth_image):
    image, mask = synth_image
    gen = SyntheticProbeGenerator()
    out = gen.generate(image, mask, (10, 200, 30))
    inside = mask >= 0.999
    diff = np.abs(out[inside].astype(int) - image[inside].astype(int))
    # alpha == 1 pixels must be exactly the original (within rounding).
    assert diff.max() <= 1
