# CorridorKey Game UI Workflow Plan

This is the current development plan for game UI assets. The mainline is moving
from ERMBG-owned detail matting toward CorridorKey as the mature detail keyer,
with ERMBG providing analysis, parameter selection, Web tooling, QA, and
fallbacks.

## Summary

- Route game UI assets through ERMBG strategy first.
- Remove the old `comfy-ermbg`/AutoMatte full-matting backend from active routes.
- Route unknown backgrounds to PyMatting fallback; auto no longer invokes RMBG.
- Add a lightweight local analysis layer before remote CorridorKey execution.
- Support green-screen first, then add blue-screen support only after verifying
  the concrete Comfy/CorridorKey route.
- Preserve missing cast/contact shadows with a local `ShadowPatch` layer under
  the CorridorKey subject layer.
- Add Web-side fallback controls: SAM3 auto mask plus simple manual mask edits.

## Mainline Architecture

```text
uploaded image
  -> local screen/color analysis
  -> if green/blue: recommended CorridorKey settings
      -> optional SAM3/user hint mask
      -> remote comfy-corridorkey
      -> local ShadowPatch gate + shadow layer composite
      -> color protection / QA / Web candidates
      -> RGBA game UI asset
  -> if unknown: remote PyMatting fallback
```

The local analysis layer should be deterministic and cheap. It should not run
heavy models on the Mac. Heavy segmentation or model inference stays on the
remote ComfyUI server.

## Screen And Color Analysis

Add a `corridorkey_analyze_asset()` style entrypoint that returns:

- `screen_mode`: `green`, `blue`, or `unknown`.
- `background_color`: measured sRGB key color.
- `background_confidence`: confidence that the image is a known screen asset.
- `purity_sigma`: how stable the background is across trusted regions.
- `subject_key_color_risk`: whether subject material is close to the key color.
- `recommended_settings`: CorridorKey gamma, despill, refiner, despeckle, and
  color-protection thresholds.

The analysis should use observable signals rather than sample-specific rules:

- trusted corners and border bands for background candidates;
- OKLab distance to compare green and blue screen hypotheses;
- background variance/purity to decide whether auto mode is safe;
- subject/key-color overlap risk to avoid erasing green or blue subject details;
- connected-component scale to avoid despeckle removing small UI ornaments.

## Parameter Adaptation

Recommended defaults:

- Pure green screen, low subject key-color risk: standard CorridorKey settings.
- Subject contains green/blue-like materials: reduce aggressive despill/refiner
  behavior and rely more on color protection.
- Small icons, slots, sparkles, or thin UI decorations: lower `despeckle_size` or
  disable auto despeckle.
- Glass, glow, transparency, or soft gradients: preserve soft hints and avoid
  hard ownership masks.

The report should record both the selected values and why they were selected,
so Web results and batch summaries are debuggable.

## ShadowPatch Layer

`ShadowPatch` is the accepted shadow strategy for the CorridorKey game UI path.
It is not an ERMBG fallback and it does not edit the CorridorKey subject layer.
It only runs for green/blue known-screen assets that have already routed to
CorridorKey; unknown backgrounds skip this path and go to PyMatting fallback.
The layer stack is:

```text
shadow layer     measured locally from known-background scalar darkening
subject layer    remote CorridorKey RGBA/alpha, kept as the owner of hard edges
```

The exported `rgba.png` is the flattened result of compositing the shadow layer
below the CorridorKey subject layer. Debug outputs keep the layers separate:

- `corridorkey_subject_rgba.png`
- `corridorkey_subject_alpha.png`
- `shadow_layer.png`
- `shadow.png`
- `shadow_physical.png`

The trigger is intentionally conservative:

- first detect a coherent known-background shadow candidate from
  `C_linear ~= scale * B_linear`;
- require high-confidence shadow evidence: enough visible support, accepted
  connected components, and non-trivial measured display opacity;
- require that CorridorKey did not already preserve the same shadow region as
  alpha. If CorridorKey alpha is already comparable to the measured shadow
  support, the patch is skipped to avoid double-darkening.

Once the trigger passes, extraction is intentionally broader than the generic
shadow path. The purpose is to cover the whole soft tail and contact region;
any overlap with the subject is harmless because final compositing places the
unchanged CorridorKey subject above the shadow layer.

Do not use color protection or hint masks to recover shadows. Color protection
is for protecting near-key subject material and can easily misclassify shadows,
while hint masks introduce subject-shadow contact artifacts. ShadowPatch should
remain a measured known-background post-process with explicit debug metrics in
`report["shadow"]["patch_gate"]`.

## Blue-Screen Support

Do not treat blue support as merely changing `bg_color`. The current
`comfy-corridorkey` wrapper now passes explicit `screen_mode` through the Comfy
workflow and blue samples are part of the active full eval, but blue-screen
semantics should stay scoped to problems green screen cannot cover.

The B016-B030 blue button block was changed on 2026-05-31 from yellow buttons to
green buttons on blue screen. Rationale: yellow/orange UI can be evaluated and
fixed on green screen; blue screen should add coverage for green subject
material, which is exactly the family green screen cannot separate cleanly.

The yellow-on-blue investigation is still useful as a diagnosis of CorridorKey's
limits: on unoutlined yellow buttons over blue, CorridorKey can decompose blue
background darkening into dirty yellow foreground plus partial alpha. That is a
model decomposition weakness, not a core ShadowPatch failure, and those samples
are no longer active B016-B030 targets.

Latest full baseline:

```text
out/corridorkey_full_blue_green_baseline_20260531/summary.json
```

Result: 83/83 completed successfully. B016-B030 all ran successfully as
blue-screen green-button samples.

Background detection should live in the ERMBG/Mac layer because the Web UI,
batch scripts, reports, and Comfy workflow selection all need the same decision.
The Comfy wrapper should receive explicit `screen_mode`, `background_color`, and
settings rather than re-guessing them independently.

## Web UI Fallbacks

The Web UI should become an operator surface for the CorridorKey path:

- Default backend for game UI work: `comfy-corridorkey`.
- Show the analysis result: screen mode, measured background, confidence, and
  selected preset.
- Provide manual overrides for screen mode, despill, refiner, despeckle, and
  color protection.
- Add SAM3 auto mask through the remote ComfyUI `SAM3_Detect` node using the
  installed `sam3.1_multiplex_fp16` checkpoint.
- Add simple manual mask editing: keep brush, erase brush, clear, reset to SAM,
  and rerun.

The SAM/manual mask should not directly replace final alpha. It should act as a
coarse CorridorKey hint or protection input, preserving CorridorKey as the detail
matting engine.

## Testing And Verification

Offline tests:

- green, blue, and unknown background classification;
- subject key-color risk changes recommended parameters;
- small UI components do not get removed by default despeckle settings;
- blue-screen metadata never reports green-only strategy names;
- mask inputs validate shape, empty masks, full masks, and edited masks.

Batch tests:

- existing game UI green samples;
- approved green/blue screen samples from the current manifest;
- subject materials near green/blue;
- glass, glow, transparent gradients, thin outlines, and small ornaments.
- ShadowPatch hit scan across all game-eval samples; inspect
  `shadowpatch_hits.json` and final Web results for every applied case.

Remote/Web verification:

- run direct `comfy-corridorkey` smoke through ComfyUI;
- run SAM3 mask smoke through ComfyUI;
- run real HTTP `/api/matte-candidates` smoke through `127.0.0.1:7860`;
- save batch summaries under `out/` with selected screen mode, settings, timing,
  and quality metrics.

## Relationship To Existing ERMBG Work

The previous solid-background/local-ownership work remains useful as fallback
and QA infrastructure, but it is no longer the primary detail-matting roadmap
for game UI assets. ERMBG should focus on the orchestration layer around
CorridorKey: input analysis, parameter adaptation, mask hints, local
ShadowPatch, diagnostics, batch evaluation, and Web controls. Unknown-background
fallback is RMBG until ERMBG has a separately validated specialty route.
