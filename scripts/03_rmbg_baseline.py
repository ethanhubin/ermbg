"""Run the ComfyUI-rembg baseline on a set of inputs and emit the same QA
metrics + multi-bg composites our pipeline produces, so the two can be compared
side-by-side.

Usage:
    .venv/bin/python scripts/03_rmbg_baseline.py --input samples/legacy/inputs/6.png \
        --out samples/legacy/outputs/rmbg_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from ermbg import io
from ermbg.comfy import DEFAULT_COMFY_URL
from ermbg.probe.comfyui_rmbg import ComfyUIRembgBaseline
from ermbg.qa import run_qa
from ermbg.segmenter import build_segmenter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, action="append",
                   help="Input image; pass multiple times")
    p.add_argument("--out", type=Path, required=True, help="Output root")
    p.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    args = p.parse_args()

    rmbg = ComfyUIRembgBaseline(url=args.comfy_url)
    seg = build_segmenter(backend="auto")

    args.out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for ip in args.input:
        logger.info(f"=== {ip.stem} ===")
        image = io.load_rgb(ip)
        rgba = rmbg.matte(image)

        # Use BiRefNet-matting alpha as the "soft mask" reference for QA's
        # thin-structure-preservation metric (same convention as our pipeline).
        soft = seg.segment(image)

        case = args.out / ip.stem
        case.mkdir(parents=True, exist_ok=True)
        io.save_rgba(case / "rgba.png", rgba)
        io.save_mask(case / "alpha.png", rgba[..., 3])

        # rembg returns RGB+alpha; we don't have B, so use the diagnoser-measured one.
        from ermbg.diagnose import BackgroundDiagnoser
        diag = BackgroundDiagnoser().diagnose(image, soft)

        qa_metrics = run_qa(
            image_srgb=image,
            rgba=rgba,
            soft_mask=soft,
            background_color=diag.background_color,
            out_dir=case / "qa",
        )
        (case / "qa" / "report.json").write_text(json.dumps(qa_metrics, indent=2), encoding="utf-8")
        (case / "report.json").write_text(json.dumps({
            "tool": "comfyui-rembg-isnet-general-use",
            "background_color": list(diag.background_color),
            "qa": qa_metrics,
        }, indent=2), encoding="utf-8")

        h = qa_metrics["edge_halo_score_per_bg"]
        summary.append({
            "image": ip.stem,
            "halo_mean": qa_metrics["edge_halo_score_mean"],
            "halo_white": h.get("white"),
            "halo_magenta": h.get("magenta"),
            "halo_cyan": h.get("cyan"),
            "alpha_n95": qa_metrics["alpha_noise_p95"],
            "thin": qa_metrics["thin_structure_preservation"],
            "recomp": qa_metrics["recomposition_error_on_observed_bg"],
        })

    print()
    print(f'{"image":<6} {"halo_mean":<10} {"halo_white":<11} {"halo_magenta":<13} {"halo_cyan":<10} {"alpha_n95":<10} {"thin":<6} {"recomp"}')
    for r in summary:
        print(f'{r["image"]:<6} {r["halo_mean"]:<10.2f} {r["halo_white"]:<11.2f} {r["halo_magenta"]:<13.2f} {r["halo_cyan"]:<10.2f} {r["alpha_n95"]:<10.3f} {r["thin"]:<6.2f} {r["recomp"]:.4f}')


if __name__ == "__main__":
    main()
