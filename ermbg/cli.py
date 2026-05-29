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
from PIL import Image

from . import io
from .api import matte_image
from .comfy import DEFAULT_COMFY_URL
from .diagnose import BackgroundDiagnoser
from .matting import matte as run_matte
from .probe.generator import PROBE_COLORS
from .probe.synthetic import SyntheticProbeGenerator
from .qa import run_qa
from .segmenter import build_segmenter, make_bands
from .slicer import save_slices, slice_image

app = typer.Typer(add_completion=False, help="ERMBG: clean transparent matting toolkit.")


def _load_object_prompt(json_path: Path) -> str | None:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
        return data.get("object_prompt")
    except Exception as e:
        logger.warning(f"Could not parse {json_path}: {e}")
        return None


def _load_subject_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load an ownership mask as H×W float32 [0,1]."""
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape != shape:
        raise typer.BadParameter(
            f"--subject-mask shape must match input image {shape}, got {mask.shape}"
        )
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


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
    backend: str = typer.Option("auto", help="auto | birefnet | grabcut | comfy-rmbg | comfy-ermbg"),
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
    backend: str = typer.Option("auto", help="auto | birefnet | grabcut | comfy-rmbg"),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL for --backend comfy-rmbg"),
):
    """Background diagnosis: is the image suitable for direct analytic matting?"""
    image = io.load_rgb(input_path)
    seg = build_segmenter(backend=backend, url=comfy_url)
    mask = seg.segment(image)
    report = BackgroundDiagnoser().diagnose(image, mask)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    (out_dir / f"{stem}.report.json").write_text(json.dumps(report.to_dict(), indent=2))
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
    backend: str = typer.Option("auto", help="auto | birefnet | grabcut | comfy-rmbg"),
    matting_model: str = typer.Option(
        "ZhengPeng7/BiRefNet-matting",
        help="HF model id for the matting segmenter (default: BiRefNet-matting)",
    ),
    input_size: int = typer.Option(1024, help="BiRefNet square input size; lower values trade quality for speed."),
    despill: str = typer.Option(
        "auto",
        help="auto | unmix | chroma_cap | local_borrow | closed_form | none. Overrides router.",
    ),
    use_keyer: bool = typer.Option(
        True,
        "--keyer/--no-keyer",
        help="Run chromatic-key on top of matting α to recover small components missed by the matting net (auto-skipped when B is near-grey)",
    ),
    shadow_mode: str = typer.Option("on", help="on | auto | off. Use off for faster previews without shadow recovery."),
    bg_color: str = typer.Option(
        "0,200,0",
        help="Composite background for transparent inputs, as 'R,G,B' (default green screen)",
    ),
    subject_mask: Path | None = typer.Option(
        None,
        "--subject-mask",
        help="Optional HxW ownership mask. Used only to repair subject-owned low-alpha holes.",
    ),
    vlm_prior: bool = typer.Option(
        False,
        "--vlm-prior/--no-vlm-prior",
        help="Use a VLM provider to classify semantic prior regions before despill.",
    ),
    vlm_provider: str = typer.Option("openai", help="openai | comfy-qwen"),
    vlm_model: str = typer.Option("gpt-4o-mini", help="Vision model for --vlm-prior"),
    vlm_prior_mode: str = typer.Option(
        "shadow",
        help="shadow | material | all. Default keeps VLM focused on owned shadow constraints.",
    ),
    comfy_url: str = typer.Option(DEFAULT_COMFY_URL, help="ComfyUI server URL for --backend comfy-rmbg/comfy-ermbg/comfy-corridorkey or --vlm-provider comfy-qwen"),
    legacy_analytic_alpha: bool = typer.Option(
        False, "--legacy-analytic-alpha", help="Run the old trimap+projection+guided-filter pipeline."
    ),
    qa: bool = typer.Option(True, help="Composite to multiple backgrounds and score"),
):
    """End-to-end analytic matting: produce RGBA + alpha + clean foreground + QA report."""
    bg_tuple = tuple(int(c) for c in bg_color.split(","))
    if len(bg_tuple) != 3:
        raise typer.BadParameter(f"--bg-color must be 'R,G,B', got {bg_color!r}")
    if shadow_mode not in {"on", "off", "auto"}:
        raise typer.BadParameter("--shadow-mode must be on, auto, or off")

    if backend in {"comfy-ermbg", "comfy-corridorkey"}:
        response = matte_image(
            input_path,
            output_dir=out_dir,
            qa=qa,
            matting_model=matting_model,
            backend=backend,
            bg_color=bg_tuple,
            despill=despill if despill != "auto" else None,
            use_keyer=False if not use_keyer else None,
            subject_mask=subject_mask,
            shadow_mode=shadow_mode,
            comfy_url=comfy_url,
        )
        logger.info(f"Saved remote {backend} matte to {response.output_dir}")
        return

    image, source_alpha = io.load_image_with_alpha(input_path)
    subject_support = _load_subject_mask(subject_mask, image.shape[:2]) if subject_mask is not None else None
    object_prompt = _load_object_prompt(input_path.with_suffix(".json"))
    seg = build_segmenter(
        backend=backend,
        model_id=matting_model,
        input_size=input_size,
        url=comfy_url,
    )

    # If the source has alpha but the router decides to RE-matte (not pass-through),
    # the matting net needs to see RGB on a known constant background, otherwise
    # transparent-region junk biases the segmentation. Pre-composite onto bg_tuple;
    # the router still sees source_alpha and can pick passthrough independently.
    if source_alpha is not None:
        from .router import classify_strategy
        strat_preview = classify_strategy(image, source_alpha=source_alpha)
        if not strat_preview.passthrough:
            image = io.load_rgb(input_path, background=bg_tuple)

    semantic_prior = None
    if vlm_prior:
        from .shadow import estimate_shadow_alpha
        from .vlm_semantic import (
            ComfyQwenVLMSemanticPriorClient,
            OpenAIVLMSemanticPriorClient,
            build_vlm_semantic_request,
            extract_shadow_candidate_regions,
            extract_subject_material_candidate_regions,
        )

        soft_preview = seg.segment(image, object_prompt=object_prompt)
        diag_preview = BackgroundDiagnoser().diagnose(image, soft_preview)
        B_preview = np.array(diag_preview.background_color, dtype=np.uint8)
        shadow_preview, _ = estimate_shadow_alpha(image, soft_preview, B_preview)
        mode = vlm_prior_mode.strip().lower()
        if mode not in {"shadow", "material", "all"}:
            raise typer.BadParameter("--vlm-prior-mode must be shadow, material, or all")
        regions = []
        if mode in {"shadow", "all"}:
            regions.extend(
                extract_shadow_candidate_regions(
                    image,
                    soft_preview,
                    B_preview,
                    shadow_alpha=shadow_preview,
                )
            )
        if mode in {"material", "all"}:
            regions.extend(
                extract_subject_material_candidate_regions(
                    image,
                    soft_preview,
                    B_preview,
                    shadow_alpha=shadow_preview,
                )
            )
        logger.info(f"vlm-prior: mode={mode} found {len(regions)} candidate region(s)")
        if regions:
            request = build_vlm_semantic_request(
                image_srgb=image,
                subject_alpha=soft_preview,
                background_color=tuple(int(c) for c in B_preview),
                regions=regions,
                shadow_alpha=shadow_preview,
            )
            if vlm_provider == "openai":
                client = OpenAIVLMSemanticPriorClient(
                    model=vlm_model,
                    env_path=Path(".env"),
                )
            elif vlm_provider == "comfy-qwen":
                client = ComfyQwenVLMSemanticPriorClient(
                    url=comfy_url,
                    model=vlm_model if vlm_model != "gpt-4o-mini" else "Qwen3-VL-4B-Instruct-FP8",
                )
            else:
                raise typer.BadParameter("--vlm-provider must be openai or comfy-qwen")
            semantic_prior = client.classify_request(request, regions, image.shape[:2])
            prior_dict = semantic_prior.to_dict()
            logger.info(
                "vlm-prior: shadow_allowed="
                f"{prior_dict['shadow_allowed']} shadow_ownership_pixels="
                f"{prior_dict['shadow_ownership_pixels']} subject_material_pixels="
                f"{prior_dict['subject_material_pixels']}"
            )

    result = run_matte(
        image,
        source_alpha=source_alpha,
        object_prompt=object_prompt,
        segmenter=seg,
        despill=despill if despill != "auto" else None,
        use_keyer=False if not use_keyer else None,
        subject_support=subject_support,
        semantic_prior=semantic_prior,
        soft_mask=soft_preview if vlm_prior else None,
        shadow_mode=shadow_mode,
        legacy_analytic_alpha=legacy_analytic_alpha,
    )

    stem = input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    io.save_rgba(out_dir / f"{stem}_rgba.png", result.rgba)
    io.save_mask(out_dir / f"{stem}_alpha.png", result.alpha)
    io.save_mask(out_dir / f"{stem}_shadow.png", result.debug["shadow_alpha"])
    io.save_rgb(out_dir / f"{stem}_foreground.png", result.foreground_srgb)
    io.save_mask(out_dir / f"{stem}_trimap.png", result.debug["trimap_u8"])

    metrics_payload = {
        "diagnosis": result.diagnosis.to_dict() if result.diagnosis is not None else None,
        "background_color": list(result.background_color),
        "despill_method": result.debug.get("despill_method"),
        "matting_model": matting_model,
        "keyer": result.debug.get("keyer", {}),
        "shadow": result.debug.get("shadow", {}),
        "semantic_prior": result.debug.get("semantic_prior", {}),
        "strategy": result.debug.get("strategy", {}),
    }

    if qa:
        qa_dir = out_dir / f"{stem}_qa"
        qa_metrics = run_qa(
            image_srgb=image,
            rgba=result.rgba,
            soft_mask=result.debug["soft_mask"],
            background_color=result.background_color,
            out_dir=qa_dir,
        )
        metrics_payload["qa"] = qa_metrics
        (qa_dir / "report.json").write_text(json.dumps(qa_metrics, indent=2))

    (out_dir / f"{stem}.report.json").write_text(json.dumps(metrics_payload, indent=2))
    verdict = result.diagnosis.verdict if result.diagnosis is not None else "passthrough"
    logger.info(f"Saved RGBA + QA to {out_dir}; verdict={verdict}")


# ---------------------------------------------------------------------------
# phase1 batch (diagnose + matte over a directory)
# ---------------------------------------------------------------------------


@app.command()
def phase1(
    input_dir: Path = typer.Option(..., help="Directory of input images"),
    out_dir: Path = typer.Option(Path("samples/outputs/phase1"), help="Output root"),
    backend: str = typer.Option("auto"),
    input_size: int = typer.Option(1024, help="BiRefNet square input size; lower values trade quality for speed."),
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

    seg = build_segmenter(backend=backend, input_size=input_size, url=comfy_url)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for p in inputs:
        logger.info(f"=== {p.stem} ===")
        image = io.load_rgb(p)
        case_dir = out_dir / p.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        io.save_rgb(case_dir / "original.png", image)

        soft = seg.segment(image)
        diag = BackgroundDiagnoser().diagnose(image, soft)
        (case_dir / "diagnose.json").write_text(json.dumps(diag.to_dict(), indent=2))
        if diag.risk_map is not None:
            io.save_mask(case_dir / "risk.png", diag.risk_map)

        row = {
            "image": p.stem,
            "background_color": list(diag.background_color),
            "purity_sigma": diag.purity_sigma,
            "edge_q10": diag.edge_contrast_q10,
            "verdict": diag.verdict,
        }

        if matte_only_when_ready and diag.verdict != "ready":
            logger.info(f"  skipping matte (verdict={diag.verdict})")
            row["matte_skipped"] = True
            summary.append(row)
            continue

        try:
            result = run_matte(image, segmenter=seg, soft_mask=soft, shadow_mode=shadow_mode)
        except Exception as e:
            logger.exception(f"matte failed for {p.stem}: {e}")
            row["error"] = str(e)
            summary.append(row)
            continue

        io.save_rgba(case_dir / "rgba.png", result.rgba)
        io.save_mask(case_dir / "alpha.png", result.alpha)
        io.save_mask(case_dir / "shadow.png", result.debug["shadow_alpha"])
        io.save_rgb(case_dir / "foreground.png", result.foreground_srgb)
        io.save_mask(case_dir / "trimap.png", result.debug["trimap_u8"])

        qa_metrics = run_qa(
            image_srgb=image,
            rgba=result.rgba,
            soft_mask=result.debug["soft_mask"],
            background_color=result.background_color,
            out_dir=case_dir / "qa",
        )
        (case_dir / "qa" / "report.json").write_text(json.dumps(qa_metrics, indent=2))
        row.update(qa_metrics)
        summary.append(row)

    _write_summary(out_dir / "summary.md", summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote {out_dir / 'summary.md'}")


def _write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Phase 1 Matting Summary",
        "",
        "| image | bg | purity_σ | edge_q10 | verdict | recomp_err | halo_mean | α_noise | thin_keep |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        bg = r.get("background_color", "")
        lines.append(
            "| {img} | {bg} | {ps:.2f} | {eq:.2f} | {v} | {re} | {hm} | {an} | {tk} |".format(
                img=r["image"],
                bg=tuple(bg) if bg else "-",
                ps=r["purity_sigma"],
                eq=r["edge_q10"],
                v=r["verdict"],
                re=f"{r.get('recomposition_error_on_observed_bg', float('nan')):.4f}" if "recomposition_error_on_observed_bg" in r else "-",
                hm=f"{r.get('edge_halo_score_mean', float('nan')):.2f}" if "edge_halo_score_mean" in r else "-",
                an=f"{r.get('alpha_noise_p95', float('nan')):.3f}" if "alpha_noise_p95" in r else "-",
                tk=f"{r.get('thin_structure_preservation', float('nan')):.2f}" if "thin_structure_preservation" in r else "-",
            )
        )
    path.write_text("\n".join(lines) + "\n")


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
