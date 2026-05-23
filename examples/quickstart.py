"""Quickstart for the ERMBG Python API.

Run::

    .venv/bin/python examples/quickstart.py samples/inputs/11.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from ermbg import classify_image, matte_image


def main(path: str = "samples/inputs/11.png") -> None:
    p = Path(path)

    # 1) Fast preview: which strategy will be applied? No matting net.
    s = classify_image(p)
    print(f"strategy:   {s.name}")
    print(f"bg_type:    {s.bg_type}")
    print(f"image_type: {s.image_type}")
    print(f"keyer:      {s.keyer_mode}")
    print(f"despill:    {s.despill}")
    print(f"notes:      {s.notes}")
    print()

    # 2) End-to-end: matte + multi-background QA, write everything to ./out/
    r = matte_image(p, output_dir="out", qa=True)
    print(f"rgba shape:      {r.rgba.shape}")
    print(f"strategy applied: {r.strategy_name}")
    print(f"measured B:       {r.background_color}")
    if "qa" in r.report:
        qa = r.report["qa"]
        print(f"recomp_err:       {qa['recomposition_error_on_observed_bg']:.4f}")
        print(f"halo_mean:        {qa['edge_halo_score_mean']:.3f}")
        print(f"thin_keep:        {qa['thin_structure_preservation']}")
    print(f"\nFiles written under: {r.output_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/inputs/11.png")
