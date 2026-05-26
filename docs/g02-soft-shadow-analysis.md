# G02-G Soft Shadow Analysis And Plan

Date: 2026-05-26

## Context

G02-G maps to:

```text
samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png
```

It is a green-screen UI asset with a soft, owned shadow under the subject. The
initial ERMBG path treated the subject reasonably, but did not preserve the
shadow. A first shadow detector proved that shadow preservation is feasible, but
the shadow edge transition was too hard.

Several experimental batches were produced:

```text
out/vlm_eval_game_shadow_rerun_20260526
out/vlm_eval_game_shadow_softedge_20260526
out/vlm_eval_game_shadow_field_20260526
out/vlm_eval_game_shadow_layers_20260526
out/vlm_eval_game_shadow_layers2_20260526
```

The later `layers/layers2` experiments are not considered the final design
direction. They improved some numeric smoothness metrics but damaged the more
important visual requirement: the contact between the subject and the shadow
must remain continuous.

## Observed Failure

The original problem was narrow:

- The real shadow already exists in the source image.
- The detector can find much of it.
- The weak point is the soft terminal transition from shadow to clean
  background.

The wrong repair direction was to reshape the whole shadow into contact/cast
layers. That added new artifacts: the contact region can become disconnected or
over-processed. For a production matte tool, a broken contact shadow is worse
than a slightly rough terminal fade.

## Pixel Evidence

A subject-aware diagnostic run was written to:

```text
out/analysis_g02_shadow_features_subject_aware
```

Key artifact:

```text
out/analysis_g02_shadow_features_subject_aware/subject_aware_overlay.png
```

The complete shadow support candidate is detectable as one connected region:

```text
bbox_xyxy: [148, 756, 1090, 1114]
support pixels: 104852
```

The strong seed is:

```text
bbox_xyxy: [186, 798, 1069, 1095]
seed pixels: 83079
```

The shadow pixels are not arbitrary dark pixels. In linear RGB they are very
close to the known green background scaled darker:

```text
C_linear ~= scale * B_linear
shadow_strength = 1 - scale
```

For the connected support:

```text
shadow_strength p1/p50/p95/p99:
0.0109 / 0.2616 / 0.5221 / 0.5837

reconstruction_error p50/p95/p99:
0.00051 / 0.00092 / 0.00122
```

The very low reconstruction error means these pixels are strong physical
evidence of a known-background shadow.

## Important Pitfall

Color evidence alone is not enough. Dark subject pixels can also satisfy:

```text
C_linear ~= scale * B_linear
```

because black is close to `0 * B`. A color-only detector incorrectly finds dark
subject interiors as strong shadow evidence.

The shadow detector therefore needs three kinds of evidence:

```text
known-B scalar darkening evidence
+ subject ownership exclusion
+ local geometry / connectedness to valid shadow seed
```

## Role Of VLM

ERMBG already has a VLM planning path for subject recognition and region
identification. For shadow preservation, VLM should be used as a semantic prior,
not as a pixel alpha generator.

VLM should provide:

- subject / owned-region proposal;
- optional owned-shadow or contact-shadow proposal;
- rough search-region constraints for where shadow evidence is plausible.

The local CV path should still decide pixel membership from measured evidence:

```text
VLM: where subject-owned content and shadow are semantically plausible
CV: which pixels actually match known-background scalar darkening
```

This prevents both failure modes:

- pure CV mistaking dark subject pixels for shadows;
- VLM hallucinating a shadow where the pixels do not support one.

## Current Direction

Do not continue tuning fixed falloff values for G02-G. The next production
direction should be:

```text
1. Use VLM / subject_mask / BiRefNet to define subject ownership constraints.
2. Estimate known background B.
3. Compute scalar darkening evidence:
   scale = dot(C, B) / dot(B, B)
   strength = 1 - scale
   err = rms(C - scale * B)
4. Build strong shadow seeds outside subject ownership:
   strength above strong threshold
   err below strict threshold
   near subject or inside VLM shadow search region
5. Grow support only through loose scalar-darkening evidence connected to seeds.
6. Preserve measured strength inside the accepted support.
7. Only feather the uncertain open boundary, not the subject-contact boundary.
```

The key production rule:

```text
Do not regenerate or globally blur the shadow.
The source image already contains the shadow; preserve its measured pixels.
```

## Boundary Feathering Principle

The falloff should be adaptive and local:

- subject-side/contact boundary: protected, no forced fade-out;
- interior shadow field: preserve measured strength;
- open terminal boundary: light feather only;
- feather width: estimated from support geometry / evidence confidence, with
  conservative clamps, not tuned to G02.

A possible implementation shape:

```text
raw_shadow = strength * confidence
support = connected_loose_evidence_from_strong_seed

boundary = support boundary
subject_contact = pixels close to subject ownership boundary
open_boundary = boundary excluding subject_contact side

falloff = 1 except near open_boundary
final_shadow_alpha = raw_shadow * falloff
```

## Acceptance Criteria

For G02-G:

- contact between subject and shadow remains continuous;
- terminal shadow edge fades smoothly enough to avoid a visible hard cutoff;
- subject dark interiors are not classified as shadow;
- shadow alpha mainly follows measured source-pixel darkening;
- no G02-specific fixed parameter tuning.

For generalization:

- works from VLM subject/region priors when available;
- degrades conservatively without VLM, using BiRefNet subject alpha and
  connected known-B evidence;
- fixed numbers act only as broad safety clamps;
- all artifact-producing quick tests write to a batch directory under `out/`.

## Implementation Checkpoint

The shadow extractor now has a front-half semantic prior contract:

```text
ShadowPrior.subject_mask
ShadowPrior.shadow_search_mask
ShadowPrior.shadow_ownership_mask
ShadowPrior.shadow_allowed
```

This is intentionally a constraint interface, not an alpha generator. VLM or a
planner may say where subject ownership, owned shadow, and plausible search
regions are. The local CV path still computes scalar darkening, connected
support, and final opacity from measured pixels.

The current code path uses this order:

```text
ShadowPrior / subject_support
  + BiRefNet alpha
  + known-B scalar darkening evidence
  + connected seed-to-support selection
  -> measured shadow strength
  -> feather open boundary only
```

New regression tests cover:

- semantic subject prior blocking dark subject interiors from becoming shadow;
- semantic shadow search prior blocking unrelated scalar-darkening patches;
- planner/VLM region kinds mapping into `ShadowPrior`;
- G02-style scalar-darkening shadow still compositing behind the subject.

Latest single-case batch:

```text
out/vlm_eval_game_shadow_vlmprior_20260526
```

G02-G result:

```text
shadow_pixels = 101189
bbox_xyxy = [155, 756, 1088, 1112]
shadow_mean_alpha = 0.2703
shadow_p95_alpha = 0.5221
```

## Scope Correction: G02 Same-Color Material

G02 also exposes a green-subject-material problem: the central green panel is
too close to the green screen, so despill can treat intentional material color
as background contamination. That is a separate source-generation issue. In the
intended AI-generation flow, we should avoid it upstream by choosing a
background color that does not collide with the subject palette.

For the shadow track, do not use the panel-color improvement as the main success
signal. The relevant checks are:

- VLM confirms whether CV shadow candidates are owned contact/cast shadows;
- CV still measures strength from known-background scalar darkening;
- contact and soft tail remain continuous when composited back to green;
- dark subject/body/material regions are not turned into shadow.

`--vlm-prior` now defaults to this shadow-only mode. Same-color material
protection remains available as an explicit semantic mode, but it should not be
mixed into G02 shadow acceptance unless we are testing foreground color recovery.

Latest shadow-only Comfy Qwen batch:

```text
out/vlm_eval_game_shadow_qwen_shadowonly_20260526
```

G02-G shadow-only result:

```text
provider = comfy-qwen:Qwen3-VL-4B-Instruct-FP8
vlm_mode = shadow
subject_material_pixels = 0
shadow_ownership_pixels = 89030
shadow_allowed = true
shadow_pixels = 101189
bbox_xyxy = [155, 756, 1088, 1112]
shadow_mean_alpha = 0.2703
shadow_p95_alpha = 0.5221
```

The shadow metrics match the earlier material-protected run; the non-shadow
foreground color metrics intentionally get worse because that run no longer asks
VLM to protect the same-color green panel.

## Handoff Snapshot

Status as of 2026-05-26:

- `ermbg/shadow.py` implements known-background scalar-darkening shadow
  extraction.
- `ShadowPrior` is wired into `ermbg/matting.py` before final compositing.
- `ermbg/vlm_semantic.py` implements OpenAI and Comfy Qwen semantic-prior
  clients.
- `--vlm-prior` defaults to `--vlm-prior-mode shadow`.
- Comfy Qwen provider is live through `Qwen3_VQA` on
  `http://192.168.0.8:8000`.
- Full test suite passes with `.venv/bin/pytest -q`.

Important code paths:

```text
ermbg/shadow.py
  ShadowPrior
  estimate_shadow_alpha
  composite_subject_with_shadow
  shadow_prior_from_regions

ermbg/vlm_semantic.py
  extract_shadow_candidate_regions
  extract_subject_material_candidate_regions
  build_vlm_semantic_request
  OpenAIVLMSemanticPriorClient
  ComfyQwenVLMSemanticPriorClient

ermbg/matting.py
  matte(..., semantic_prior=...)
  shadow pre-pass protects shadow from keyer raising
  final pass composites measured shadow behind subject

ermbg/cli.py / ermbg/api.py
  --vlm-prior
  --vlm-provider openai|comfy-qwen
  --vlm-prior-mode shadow|material|all
```

Reproduce the current G02 shadow-only run:

```bash
.venv/bin/ermbg matte samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png \
  --out-dir out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/ui_hard_button_soft_shadow \
  --backend auto \
  --vlm-prior \
  --vlm-provider comfy-qwen \
  --vlm-model Qwen3-VL-4B-Instruct-FP8
```

Key artifacts:

```text
out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/ui_hard_button_soft_shadow/green_rgba.png
out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/ui_hard_button_soft_shadow/green_shadow.png
out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/ui_hard_button_soft_shadow/green.report.json
out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/g02_recompose_analysis/whole_recompose_comparison.png
out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/g02_recompose_analysis/shadow_crop_comparison.png
out/vlm_eval_game_shadow_qwen_shadowonly_20260526/matte/g02_recompose_analysis/summary.json
```

The most useful report fields for handoff:

```text
report.semantic_prior.shadow_allowed
report.semantic_prior.shadow_ownership_pixels
report.semantic_prior.subject_material_pixels
report.shadow.detected
report.shadow.pixels
report.shadow.bbox_xyxy
report.shadow.mean_alpha
report.shadow.p95_alpha
report.keyer.shadow_protected_pixels
```

## Current Behavior Notes

The shadow-only path is intentionally conservative:

- VLM receives only measured shadow candidates by default.
- VLM can accept a candidate as `shadow`, reject it as `background`, or mark it
  as `subject`.
- Accepted VLM regions do not define opacity; they only constrain where
  `estimate_shadow_alpha()` may interpret scalar darkening as owned shadow.
- If VLM rejects all shadow candidates by setting `shadow_allowed=false`, local
  shadow extraction returns no shadow.
- Material protection is still available with `--vlm-prior-mode material` or
  `all`, but should be treated as a separate foreground-color recovery track.

Known implementation tradeoffs:

- `matte_image(vlm_prior=True)` and CLI `--vlm-prior` currently run an extra
  preview segmentation before the final matte. This is correct but inefficient.
- The Comfy Qwen client uses a contact sheet and `PreviewAny` text extraction.
  It is robust enough for the current node shape, but future Comfy node changes
  may require parser updates.
- `Qwen3_VQA` rejects temperature `0.0`, so the client clamps to at least
  `0.01` and defaults to `0.1`.
- The G02 green-panel issue is deliberately not fixed by shadow-only mode.

## Suggested Next Steps

1. Run a small shadow-only batch across the other `samples/vlm_eval_game`
   shadow cases and compare `shadow.detected`, `shadow.pixels`, and visual
   contact continuity.
2. Add a batch-level eval summary for shadow metrics so we do not inspect each
   report manually.
3. Add one negative sample where a dark subject interior matches scalar
   darkening; verify Qwen can classify the candidate as `subject` or
   `background`.
4. Consider reusing the preview BiRefNet alpha in the final matte call to avoid
   double segmentation.
