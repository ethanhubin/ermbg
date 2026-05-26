"""Smoke test: run the segmenter on one image and dump mask + rough trimap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling 'ermbg' importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg import io
from ermbg.segmenter import build_segmenter, make_bands


def main(input_path: str, out_dir: str = "samples/legacy/outputs/smoke", backend: str = "auto") -> None:
    image = io.load_rgb(input_path)
    seg = build_segmenter(backend=backend)
    soft = seg.segment(image)
    bands = make_bands(soft)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    io.save_mask(out / f"{stem}_mask.png", soft)
    trimap = np.zeros_like(soft, dtype=np.uint8)
    trimap[bands.inner] = 255
    trimap[bands.unknown_band] = 128
    io.save_mask(out / f"{stem}_trimap.png", trimap)
    print(f"Saved mask + trimap under {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/01_smoke_segment.py <image> [out_dir] [backend]")
        sys.exit(1)
    main(*sys.argv[1:])
