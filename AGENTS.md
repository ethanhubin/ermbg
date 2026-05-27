# ERMBG · Engineering Contract

## Available Infrastructure

### ComfyUI Server (REMOTE — preferred over local model loads)

A long-running ComfyUI server is reachable at **`http://192.168.0.8:8000`**.

- **Hardware**: Windows + RTX 4090 (24 GB VRAM), 64 GB RAM, ComfyUI 0.21.1
- **Always running** — do **not** propose installing diffusers / SDXL / FLUX / RMBG models locally on the Mac. The Mac is for orchestration (CLI, lightweight CV/numpy, BiRefNet-matting via MPS) only. Heavy generation/inference goes to ComfyUI.
- **Mac local model budget**: BiRefNet-matting (≈1 GB MPS) yes; SDXL (16+ GB MPS) no — already proven OOM on Phase 1.

**Confirmed installed nodes / models** (full list cached in `/tmp/comfy_object_info.json` after `curl -s http://192.168.0.8:8000/object_info > /tmp/comfy_object_info.json`):

- Generation backends: Qwen-Image-Edit 2511 fp8, Flux Dev fp8, Flux 2 Klein 9b, FLUX schnell, Z-Image-Turbo
- Background removal: `Image Rembg` (isnet-general-use / u2net / u2netp / silueta / isnet-anime), `BriaRemoveImageBackground`, `RemoveBackground`, `LayerMask: RmBgUltra V2`
- IPAdapter / ControlNet / many LoRAs available
- VAE: qwen_image_vae, flux2-vae, flux ultra vae

**HTTP API** the Python client uses ([ermbg/probe/comfyui.py](ermbg/probe/comfyui.py) is the reference implementation):

```
POST /upload/image          (multipart with name + overwrite=true)
POST /prompt                (workflow JSON, returns prompt_id)
GET  /history/<prompt_id>   (poll until status.completed)
GET  /view?filename=...&subfolder=...&type=output
```

Workflow templates live in `ermbg/probe/comfyui_*.json`, with `${variable}` placeholders filled by `string.Template`.

### When to use ComfyUI vs local

| Task | Where |
|---|---|
| BiRefNet-matting (1 GB) | Local MPS (`ermbg.segmenter.BiRefNetSegmenter`) — already wired |
| BRIA RMBG-2.0 (gated) | ComfyUI (`BriaRemoveImageBackground` node) |
| RMBG-1.4 / IS-Net family | ComfyUI (`Image Rembg` with `isnet-general-use`) |
| SDXL / FLUX / Qwen-Edit generation | ComfyUI (always) |
| Pure numpy / OpenCV / scipy ops | Local |

### OpenAI API

`OPENAI_API_KEY` lives in `.env` (gitignored). Used only when a task explicitly needs `gpt-image-1` cloud editing — already verified to repaint subjects (IoU=0.85), so prefer ComfyUI for image generation.

---

## Project structure

```
ermbg/
  segmenter.py        BiRefNet (default: ZhengPeng7/BiRefNet-matting)
  matting.py          end-to-end pipeline
  diagnose.py         background diagnoser (B, purity, edge_q10)
  trimap.py           trimap construction (legacy path only)
  alpha.py            projection / per-channel / guided filter (legacy path)
  foreground.py       KNN F_ref (used by despill.local_borrow)
  recover.py          legacy decontamination (deprecated)
  despill.py          chroma_cap | local_borrow | closed_form | none
  lightwrap.py        edge halo suppression (Brinkmann light wrap)
  qa.py               composite to 6 backgrounds, score halos
  cli.py              segment / diagnose / matte / phase1 / probe
  probe/
    generator.py      backend protocol
    synthetic.py      mask-and-paste baseline
    sdxl_inpaint.py   diffusers SDXL inpainting (OOM on Mac)
    comfyui.py        Qwen-Edit via remote ComfyUI
    openai_image.py   gpt-image-1 via OpenAI API
    comfyui_*.json    workflow templates
    prompts.py        GREEN_SCREEN_RGB / GREEN_SCREEN_PROMPT
samples/vlm_eval/            AI-generated VLM planner eval cases
samples/legacy/inputs/       3.png 4.png ... 8.png + optional *.json prompts
samples/legacy/outputs/      archived matte_* and green_* output trees
tests/                pytest, 22 tests passing
```

## Conventions

- All code in **linear RGB** internally; convert at I/O boundary via `ermbg.io.{srgb_to_linear, linear_to_srgb_u8}`.
- Color distance work in **OKLab** via `ermbg.colorspace.oklab_distance`.
- Soft mask / alpha is **float32 [0, 1]**, H×W. RGBA outputs are **uint8 sRGB** with alpha in last channel.
- Tests must keep passing (`pytest -q`). When adding modules, add a smoke test.
- The python venv is **`.venv/`** (managed by uv, Python 3.12). All `.venv/bin/ermbg`, `.venv/bin/pytest`, `.venv/bin/python` commands.

### AI / algorithm tuning contract

- Any algorithm-detail adjustment made by an AI agent, especially heuristic thresholds, confidence gates, falloff widths, area ratios, or display/export remapping constants, must include a nearby code comment explaining the intent and the failure mode it protects against.
- Comments must distinguish broad invariants from empirical values. If a value is experience-driven, say what observable signal it keys on and which class of samples motivated it; do not leave a naked magic number.
- Do not encode sample IDs, file names, or one-off coordinates as fixes unless explicitly requested. Prefer feature-based rules and document why the rule should generalize.
- When changing visual/matting behavior, write or update a focused test that captures the intended class of failure, not just the current sample.

### Test / eval batch convention

- All sample tests, visual regressions, local-ownership/game eval reruns, probe comparisons, and one-off debugging runs that produce artifacts must write into a **batch directory** under `out/`.
- Do not write new eval artifacts directly into loose ad-hoc paths. Use a stable batch id such as `out/local_ownership_<purpose>_<YYYYMMDD>/` or an explicit user-provided batch name.
- Each batch should be self-contained and browsable: keep per-case outputs under the batch root and write a machine-readable summary (`summary.json`, `summary_*.json`, or `eval_report.json`) at a predictable location.
- Web/debug tooling should discover and browse batches rather than hard-code a single result directory. New test flows should automatically register their outputs through the batch summary.
- When re-running a specific sample such as `G02-G`, still run it through the same batch flow and record the selected sample id / variant in the summary; do not create orphan outputs.

## Default decisions

- **Matting model**: `ZhengPeng7/BiRefNet-matting` (MIT, matting-trained)
- **Background convention**: green-screen RGB (0, 200, 0) — see `ermbg.probe.prompts.GREEN_SCREEN_PROMPT`
- **Despill default**: `chroma_cap` (auto-degrades to `local_borrow` when B has no dominant channel)
- **QA backgrounds**: black / white / grey / cyan / magenta / checker, plus a `_lightwrap` variant for each

### Current local ownership contract

- Current active route is documented in `docs/local-ownership.md`.
- Default direction is local, deterministic ownership scoring from measurable known-background evidence.
- Current flow is:
  1. local matte and known-background diagnosis compute `B`, alpha, and foreground/debug outputs;
  2. local evidence extractors produce risk/debug regions;
  3. `ermbg.ownership.rank_regions_ownership()` ranks each region as `hole`, `opaque_subject`, `subject_soft_layer`, `shadow_like_layer`, or `conservative_unknown`;
  4. `ermbg.ownership.resolve_execution_masks()` performs global arbitration before any role becomes an execution mask;
  5. `ermbg.matting.matte()` uses `subject_material_mask` only as a protective constraint, restoring soft-layer alpha after destructive keyer/repair changes;
  6. shadow opacity is still measured locally from `C_linear ~= scale * B_linear`.
- Do not route ownership ambiguity to model planning by default. First inspect local signals, role scores, execution masks, and whether foreground/color recovery is the real failure.
- G02/G04/G06 green+white are the current fast target set. Keep G03 out of the fast loop while this branch is focused on the local ownership path.
- Archived model-planning and G02 single-sample documents under `docs/archive/` are historical context, not the active plan.
