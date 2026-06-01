"""ERMBG command-line interface.

Phase-1 (revised) commands:

  segment    coarse subject segmentation + rough trimap
  diagnose   single-image background diagnosis (B, purity, edge_q10, risk map)
  matte      end-to-end analytic matting -> RGBA + QA on multiple backgrounds
  phase1     batch: diagnose + matte + QA over a directory
  probe      (legacy) generate one probe via synthetic / sdxl / comfyui / openai

The probe-generation backends are kept around for later experiments but are not
part of the main matting pipeline anymore.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from . import io
from .api import matte_image
from .comfy import DEFAULT_COMFY_URL
from .diagnose import BackgroundDiagnoser
from .probe.generator import PROBE_COLORS
from .probe.synthetic import SyntheticProbeGenerator
from .segmenter import build_segmenter, make_bands
from .slicer import save_slices, slice_image

app = typer.Typer(add_completion=False, help="ERMBG: clean transparent matting toolkit.")


def _load_object_prompt(json_path: Path) -> str | None:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("object_prompt")
    except Exception as e:
        logger.warning(f"Could not parse {json_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


@app.command("slice")
def slice_command(
    input_path: Path = typer.Argument(..., help="Input image path with separated subjects"),
    out_dir: Path = typer.Option(Path("samples/outputs/slices"), help="Output directory"),
    bg_color: str | None = typer.Option(
        None,
        "--bg-color",
        help="Optional known background as 'R,G,B'. Default: auto-estimate from image border.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="Optional OKLab distance threshold. Default: auto from border noise.",
    ),
    min_area: int = typer.Option(64, help="Ignore connected foreground regions smaller than this many pixels."),
    padding: int = typer.Option(2, help="Padding around each exported rectangle, in pixels."),
    transparent: bool = typer.Option(
        False,
        "--transparent/--no-transparent",
        help="Export each rectangle as RGBA with background masked out.",
    ),
):
    """Auto-detect solid background and rectangle-slice separated subjects."""
    bg_tuple = None
    if bg_color is not None:
        bg_tuple = tuple(int(c) for c in bg_color.split(","))
        if len(bg_tuple) != 3:
            raise typer.BadParameter(f"--bg-color must be 'R,G,B', got {bg_color!r}")

    image = io.load_rgb(input_path)
    result = slice_image(
        image,
        background_color=bg_tuple,
        distance_threshold=threshold,
        min_area=min_area,
        padding=padding,
    )
    paths = save_slices(
        image,
        result,
        out_dir,
        stem=input_path.stem,
        transparent=transparent,
    )
    logger.info(
        f"Saved {len(paths)} slice(s) to {out_dir}; "
        f"background={result.background_color}"
    )


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------


@app.command()
def segment(
    input_path: Path = typer.Argument(..., help="Input image path"),
    out_dir: Path = typer.Option(Path("samples/outputs/smoke"), help="Output directory"),
    backend: str = typer.Option("comfy-rmbg", help="comfy-rmbg"),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL for --backend comfy-rmbg"),
):
    """Run coarse subject segmentation + build a rough trimap."""
    image = io.load_rgb(input_path)
    seg = build_segmenter(backend=backend, url=comfy_url)
    soft = seg.segment(image)
    bands = make_bands(soft)

    stem = input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    io.save_mask(out_dir / f"{stem}_mask.png", soft)
    trimap_vis = np.zeros(soft.shape, dtype=np.uint8)
    trimap_vis[bands.inner] = 255
    trimap_vis[bands.unknown_band] = 128
    io.save_mask(out_dir / f"{stem}_trimap.png", trimap_vis)
    logger.info(f"Saved {out_dir / f'{stem}_mask.png'} and {stem}_trimap.png")


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


@app.command()
def diagnose(
    input_path: Path = typer.Argument(..., help="Input image path"),
    out_dir: Path = typer.Option(Path("samples/outputs/diagnose"), help="Output directory"),
    backend: str = typer.Option("comfy-rmbg", help="comfy-rmbg"),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL for --backend comfy-rmbg"),
):
    """Background diagnosis: is the image suitable for direct analytic matting?"""
    image = io.load_rgb(input_path)
    seg = build_segmenter(backend=backend, url=comfy_url)
    mask = seg.segment(image)
    report = BackgroundDiagnoser().diagnose(image, mask)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    (out_dir / f"{stem}.report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    io.save_mask(out_dir / f"{stem}_mask.png", mask)
    if report.risk_map is not None:
        io.save_mask(out_dir / f"{stem}_risk.png", report.risk_map)
    logger.info(f"verdict={report.verdict}  B={report.background_color}  purity_sigma={report.purity_sigma:.2f}  edge_q10={report.edge_contrast_q10:.2f}")


# ---------------------------------------------------------------------------
# matte (end-to-end)
# ---------------------------------------------------------------------------


@app.command()
def matte(
    input_path: Path = typer.Argument(..., help="Input image path (clean solid-bg)"),
    out_dir: Path = typer.Option(Path("samples/outputs/matte"), help="Output directory"),
    backend: str = typer.Option("auto", help="auto | comfy-rmbg | comfy-corridorkey | pymatting-known-b | comfy-pymatting-known-b"),
    shadow_mode: str = typer.Option("on", help="on | auto | off. Use off for faster previews without shadow recovery."),
    bg_color: str = typer.Option(
        "0,200,0",
        help="Composite background for transparent inputs, as 'R,G,B' (default green screen)",
    ),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL for remote backends"),
    qa: bool = typer.Option(True, help="Composite to multiple backgrounds and score"),
):
    """End-to-end analytic matting: produce RGBA + alpha + clean foreground + QA report."""
    bg_tuple = tuple(int(c) for c in bg_color.split(","))
    if len(bg_tuple) != 3:
        raise typer.BadParameter(f"--bg-color must be 'R,G,B', got {bg_color!r}")
    if shadow_mode not in {"on", "off", "auto"}:
        raise typer.BadParameter("--shadow-mode must be on, auto, or off")

    if backend in {"birefnet", "grabcut", "comfy-ermbg"}:
        raise typer.BadParameter("legacy full-matting backends were removed; use auto or a routed backend")
    response = matte_image(
        input_path,
        output_dir=out_dir,
        qa=qa,
        backend=backend,
        bg_color=bg_tuple,
        shadow_mode=shadow_mode,
        comfy_url=comfy_url,
    )
    logger.info(f"Saved {response.strategy_name} matte to {response.output_dir}")


# ---------------------------------------------------------------------------
# phase1 batch (diagnose + matte over a directory)
# ---------------------------------------------------------------------------


@app.command()
def phase1(
    input_dir: Path = typer.Option(..., help="Directory of input images"),
    out_dir: Path = typer.Option(Path("samples/outputs/phase1"), help="Output root"),
    backend: str = typer.Option("auto"),
    input_size: int = typer.Option(1024, help="Deprecated compatibility option."),
    shadow_mode: str = typer.Option("on", help="on | auto | off. Use off for faster previews without shadow recovery."),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL for --backend comfy-rmbg"),
    matte_only_when_ready: bool = typer.Option(
        False,
        help="If true, only run matting when diagnose verdict='ready'. Default: always matte.",
    ),
):
    """Batch: diagnose + matte all images in input_dir, write a summary."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    inputs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    if not inputs:
        logger.error(f"No images in {input_dir}")
        return
    if shadow_mode not in {"on", "off", "auto"}:
        raise typer.BadParameter("--shadow-mode must be on, auto, or off")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for p in inputs:
        logger.info(f"=== {p.stem} ===")
        case_dir = out_dir / p.stem
        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = matte_image(p, output_dir=case_dir, qa=True, backend=backend, shadow_mode=shadow_mode, comfy_url=comfy_url)
        except Exception as e:
            logger.exception(f"matte failed for {p.stem}: {e}")
            summary.append({"image": p.stem, "error": str(e)})
            continue
        row = {
            "image": p.stem,
            "background_color": list(result.background_color),
            "strategy": result.strategy_name,
            "backend": result.debug.get("auto_route", {}).get("selected_backend", backend),
        }
        row.update(result.report.get("qa", {}))
        summary.append(row)

    _write_summary(out_dir / "summary.md", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote {out_dir / 'summary.md'}")


def _write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Phase 1 Matting Summary",
        "",
        "| image | backend | strategy | bg | recomp_err | halo_mean | α_noise | thin_keep |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        bg = r.get("background_color", "")
        lines.append(
            "| {img} | {backend} | {strategy} | {bg} | {re} | {hm} | {an} | {tk} |".format(
                img=r["image"],
                backend=r.get("backend", "-"),
                strategy=r.get("strategy", "-"),
                bg=tuple(bg) if bg else "-",
                re=f"{r.get('recomposition_error_on_observed_bg', float('nan')):.4f}" if "recomposition_error_on_observed_bg" in r else "-",
                hm=f"{r.get('edge_halo_score_mean', float('nan')):.2f}" if "edge_halo_score_mean" in r else "-",
                an=f"{r.get('alpha_noise_p95', float('nan')):.3f}" if "alpha_noise_p95" in r else "-",
                tk=f"{r.get('thin_structure_preservation', float('nan')):.2f}" if "thin_structure_preservation" in r else "-",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# probe (legacy: kept for development of probe backends)
# ---------------------------------------------------------------------------


@app.command()
def probe(
    input_path: Path = typer.Argument(..., help="Input image path"),
    color: str = typer.Option("white", help=f"Probe color name. One of {list(PROBE_COLORS)}"),
    generator: str = typer.Option("synthetic", help="synthetic | sdxl | comfyui | openai"),
    out_dir: Path = typer.Option(Path("samples/outputs/probes"), help="Output directory"),
    backend: str = typer.Option("auto", help="Segmenter backend"),
    seed: int = typer.Option(42, help="Random seed"),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL"),
):
    """(Legacy) generate one probe image. Not part of the main matting pipeline."""
    image = io.load_rgb(input_path)
    seg = build_segmenter(backend=backend)
    mask = seg.segment(image)

    if color not in PROBE_COLORS:
        raise typer.BadParameter(f"Unknown color {color!r}; choose from {list(PROBE_COLORS)}")
    bg = PROBE_COLORS[color]

    if generator == "synthetic":
        gen = SyntheticProbeGenerator()
        probe_img = gen.generate(image, mask, bg, seed=seed)
    elif generator == "sdxl":
        from .probe.sdxl_inpaint import SDXLInpaintProbeGenerator
        gen = SDXLInpaintProbeGenerator()
        probe_img = gen.generate(image, mask, bg, seed=seed)
    elif generator == "comfyui":
        from .probe.comfyui import ComfyUIProbeGenerator
        prompt = _load_object_prompt(input_path.with_suffix(".json"))
        gen = ComfyUIProbeGenerator(url=comfy_url)
        probe_img = gen.generate(image, mask, bg, seed=seed, object_prompt=prompt)
    elif generator == "openai":
        from .probe.openai_image import OpenAIImageProbeGenerator
        gen = OpenAIImageProbeGenerator()
        probe_img = gen.generate(image, mask, bg, seed=seed)
    else:
        raise typer.BadParameter(f"Unknown generator {generator!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_{generator}_{color}.png"
    io.save_rgb(out_path, probe_img)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    app()
