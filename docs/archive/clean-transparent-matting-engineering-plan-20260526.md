# ERMBG · Current Engineering State

Date: 2026-05-26

This file is the short handoff entry point. Longer reasoning and sample-specific
analysis live under `docs/`.

## Handoff Summary

- Core contract: AI generation should use a known constant background; ERMBG
  then recovers clean RGBA from that known-background image.
- Default matting model: `ZhengPeng7/BiRefNet-matting`.
- Default background convention: green screen RGB `(0, 200, 0)`.
- Default pipeline is automatic: router chooses passthrough / chromatic /
  luminance / noisy handling.
- Shadow support is now part of the main matte path. Known-background scalar
  darkening is measured locally; VLM only supplies semantic constraints.
- `--vlm-prior` defaults to `--vlm-prior-mode shadow`.
- Comfy Qwen provider is wired through the remote ComfyUI server at
  `http://192.168.0.8:8000`, defaulting to `Qwen3-VL-4B-Instruct-FP8`.
- Full test suite last passed with `.venv/bin/pytest -q` at 154 tests.

Current focus:

- G02 soft shadow handoff:
  [docs/g02-soft-shadow-analysis.md](docs/g02-soft-shadow-analysis.md)
- Known-B candidate planning:
  [docs/known-b-candidate-matting.md](docs/known-b-candidate-matting.md)

## Core Contract

ERMBG assumes two stages:

1. **Generation stage**
   AI image generation should use a controllable constant background. Avoid
   choosing a background color that collides with important subject colors.

2. **Matting stage**
   ERMBG receives RGB/PNG input, estimates or uses known background color `B`,
   and produces RGBA plus QA/debug artifacts. Users should not need to tune
   thresholds for normal cases.

Known `B` is the core constraint:

```text
C = alpha * F + (1 - alpha) * B    in linear RGB
```

It turns despill and QA from guessing into measured reconstruction.

## Current Pipeline

```text
image (sRGB uint8 + optional source alpha)
  -> router.classify_strategy
  -> optional passthrough for clean RGBA
  -> BiRefNetSegmenter / GrabCut fallback
  -> BackgroundDiagnoser estimates B, purity, edge contrast
  -> optional VLM semantic prior and shadow pre-pass
  -> key alpha + guarded merge/repair
  -> despill / foreground recovery in linear RGB
  -> final shadow pass and subject+shadow composite
  -> RGBA, alpha, shadow, foreground, trimap, QA report
```

Important boundary:

```text
VLM: semantic constraints for subject / owned shadow / material candidates
CV: pixel membership, alpha, foreground color, and shadow strength
```

VLM does not generate alpha or RGBA.

## Defaults

- **Model**: `ZhengPeng7/BiRefNet-matting`
- **Router strategies**:
  - `rgba_passthrough`: reuse clean source alpha
  - `saturated`: chromatic key + `auto` despill
  - `white` / `black`: luminance key + `unmix`
  - `grey`: luminance key + `local_borrow`
  - `noisy`: no keyer + `local_borrow`
- **Despill**:
  - `auto`: unmix plus chroma cap on saturated backgrounds
  - `unmix`: closed-form known-B foreground recovery
  - `chroma_cap`, `local_borrow`, `closed_form`, `none`
- **Shadow**: known-B scalar darkening:
  `C_linear ~= scale * B_linear`, `shadow_alpha ~= 1 - scale`
- **VLM provider**: `openai` or `comfy-qwen`
- **QA backgrounds**: black, white, grey, cyan, magenta, checker, plus
  lightwrap variants

## Main Modules

| File | Role |
|---|---|
| [ermbg/matting.py](ermbg/matting.py) | End-to-end matte pipeline |
| [ermbg/router.py](ermbg/router.py) | Strategy classification |
| [ermbg/segmenter.py](ermbg/segmenter.py) | BiRefNet / GrabCut segmenters |
| [ermbg/diagnose.py](ermbg/diagnose.py) | Background diagnosis |
| [ermbg/keyer.py](ermbg/keyer.py) | Chromatic/luminance keying and guarded merge |
| [ermbg/despill.py](ermbg/despill.py) | Foreground recovery and despill |
| [ermbg/shadow.py](ermbg/shadow.py) | Known-B shadow extraction and composition |
| [ermbg/vlm_semantic.py](ermbg/vlm_semantic.py) | Shadow/material semantic priors; OpenAI and Comfy Qwen clients |
| [ermbg/planner.py](ermbg/planner.py) | Evidence regions, tool catalog, candidate plans |
| [ermbg/candidates.py](ermbg/candidates.py) | Same-color ambiguity candidates |
| [ermbg/qa.py](ermbg/qa.py) | Multi-background QA |
| [ermbg/cli.py](ermbg/cli.py) | CLI entry points |
| [ermbg/api.py](ermbg/api.py) | Python API |
| [ermbg/web.py](ermbg/web.py) | FastAPI UI/API |

## CLI

```bash
# End-to-end matte, router chooses strategy
.venv/bin/ermbg matte samples/legacy/inputs/3.png

# Manual despill override
.venv/bin/ermbg matte samples/legacy/inputs/3.png --despill unmix

# Disable keyer
.venv/bin/ermbg matte samples/legacy/inputs/3.png --no-keyer

# Shadow-only VLM semantic prior via Comfy Qwen
.venv/bin/ermbg matte samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png \
  --vlm-prior \
  --vlm-provider comfy-qwen \
  --vlm-model Qwen3-VL-4B-Instruct-FP8

# Explicit same-color material protection, separate from shadow acceptance
.venv/bin/ermbg matte samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png \
  --vlm-prior \
  --vlm-prior-mode material

# Diagnose only
.venv/bin/ermbg diagnose samples/legacy/inputs/3.png
```

Output files: `*_rgba.png`, `*_alpha.png`, `*_shadow.png`,
`*_foreground.png`, `*_trimap.png`, `*.report.json`, and `*_qa/on_*.png`.

Useful report fields: `diagnosis.background_color`, `strategy`, `keyer`,
`shadow.*`, `semantic_prior.*`, `qa.recomposition_error_on_observed_bg`, and
`qa.edge_halo_score_mean`.

## Tests

```bash
# Full suite
.venv/bin/pytest -q

# Core suite for daily work
.venv/bin/pytest -q -m core

# Shadow / semantic-prior focused checks
.venv/bin/pytest tests/test_shadow.py tests/test_vlm_semantic.py -q
```

Key coverage: router/source-alpha hygiene, keyer/despill, known-B diagnosis and
QA, candidate planning/executor, shadow extraction, VLM prior constraints,
OpenAI/Comfy Qwen mocks, API/CLI/web, and Comfy node smoke tests.

## Current Focus

### G02 Soft Shadow

G02 sample:

```text
samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png
```

Current status:

- shadow-only Qwen run accepts one owned shadow candidate;
- `subject_material_pixels = 0`, intentionally avoiding the green-panel issue;
- `shadow_pixels = 101189`;
- `bbox_xyxy = [155, 756, 1088, 1112]`;
- `shadow_mean_alpha = 0.2703`;
- `shadow_p95_alpha = 0.5221`.

Do not use G02 green panel color recovery as the shadow success metric. Same
color material collision should be solved upstream by selecting a better
generation background.

Details and artifacts:
[docs/g02-soft-shadow-analysis.md](docs/g02-soft-shadow-analysis.md)

### Known-B Candidate Planning

The same-color interior ambiguity path is a finite local-tool planning loop:

```text
local CV evidence
  -> EvidenceRegion / RiskRegion
  -> PlannerPromptBundle + ToolCatalog
  -> RulePlannerClient / future VLM planner
  -> CandidatePlan[]
  -> PlanExecutor
  -> MatteCandidate[] + API/UI/debug overlay
```

Details:
[docs/known-b-candidate-matting.md](docs/known-b-candidate-matting.md)

## Infrastructure

- Local Mac: Python 3.12, uv-managed `.venv/`, BiRefNet-matting on MPS.
- Remote ComfyUI: `http://192.168.0.8:8000`, Windows + RTX 4090.
- ComfyUI ERMBG nodes: `ErmbgAutoMatte`, `ErmbgClassify`; install notes in
  [DEPLOY.md](DEPLOY.md).
- OpenAI key: `.env`, used only on explicit OpenAI provider paths.
- OpenClaw integration:
  [integrations/openclaw/README.md](integrations/openclaw/README.md)

## Docs Index

- [docs/g02-soft-shadow-analysis.md](docs/g02-soft-shadow-analysis.md):
  current soft-shadow analysis, Qwen provider handoff, artifacts, next steps.
- [docs/known-b-candidate-matting.md](docs/known-b-candidate-matting.md):
  candidate planning, EvidenceRegion policy mapping, VLM planner contract.
- [docs/archive/sample-12-white-bg-panel-analysis.md](docs/archive/sample-12-white-bg-panel-analysis.md):
  archived sample-12 investigation, no longer current design source.
- [DEPLOY.md](DEPLOY.md): ComfyUI deployment.
- [integrations/openclaw/README.md](integrations/openclaw/README.md):
  OpenClaw integration.

## Guiding Judgment

```text
clean transparent matte =
  accurate alpha
  + clean foreground RGB
  + controlled background-contamination removal
  + optional owned shadow matte when source pixels support it
```

The user should be able to provide an image and let the system choose the path.
