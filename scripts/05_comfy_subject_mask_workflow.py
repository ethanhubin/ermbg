"""Prepare or run the ComfyUI CLIPSeg -> ERMBG subject-mask workflow.

Default mode is dry-run: it writes the rendered ComfyUI workflow JSON without
touching the remote server. With ``--submit`` it uploads the image, queues the
workflow, waits for completion by default, and downloads foreground / alpha /
subject-mask artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ermbg import io
from ermbg.probe.comfyui_subject_mask import (
    ComfyUISubjectMaskWorkflow,
    render_clipseg_ermbg_workflow,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Local input image")
    p.add_argument("--prompt", required=True, help="Prompt describing the full subject to keep")
    p.add_argument("--out", type=Path, required=True, help="Rendered workflow JSON path")
    p.add_argument("--filename-prefix", default="ermbg_subject", help="ComfyUI SaveImage prefix")
    p.add_argument("--clipseg-model", default="CIDAS/clipseg-rd64-refined")
    p.add_argument("--matting-model", default="ZhengPeng7/BiRefNet-matting")
    p.add_argument("--bg-color", default="0,200,0")
    p.add_argument("--comfy-url", default="http://192.168.0.8:8000")
    p.add_argument("--submit", action="store_true", help="Upload and queue on ComfyUI")
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="When used with --submit, queue the workflow but do not wait/download outputs.",
    )
    p.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Where completed ComfyUI outputs are downloaded. Defaults next to --out.",
    )
    args = p.parse_args()

    server_image = f"DRY_RUN_{args.input.name}"
    runner = None
    if args.submit:
        runner = ComfyUISubjectMaskWorkflow(url=args.comfy_url)
        server_image = runner.upload_image(io.load_rgb(args.input), name=args.input.name)

    workflow = render_clipseg_ermbg_workflow(
        input_image=server_image,
        subject_prompt=args.prompt,
        filename_prefix=args.filename_prefix,
        clipseg_model=args.clipseg_model,
        matting_model=args.matting_model,
        bg_color=args.bg_color,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(workflow, indent=2))
    print(f"Wrote {args.out}")

    if runner is not None:
        prompt_id = runner.queue(workflow)
        print(f"Queued prompt_id={prompt_id}")
        if args.no_wait:
            summary = {
                "input": str(args.input),
                "workflow": str(args.out),
                "prompt_id": prompt_id,
                "status": "queued",
                "downloads": [],
            }
            (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
            return

        history = runner.wait(prompt_id)
        download_dir = args.download_dir or args.out.with_suffix("")
        downloads = runner.download_images(history, download_dir)
        summary = {
            "input": str(args.input),
            "workflow": str(args.out),
            "prompt_id": prompt_id,
            "status": history.get("status", {}),
            "downloads": downloads,
        }
        summary_path = download_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Downloaded {len(downloads)} image(s) to {download_dir}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
