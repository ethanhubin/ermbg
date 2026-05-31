# CorridorKey Game UI Workflow Plan

This is the current development plan for game UI assets. The mainline has moved
to ERMBG-owned routing with profile-specific execution: ERMBG decides the final
path and parameters, then ComfyUI executes PyMatting Known-B, CorridorKey,
passthrough, or PyMatting fallback inside the remote process.

## Summary

- Route game UI assets through ERMBG strategy first and emit a final
  `execution_profile` before matting.
- Remove the old `comfy-ermbg`/AutoMatte full-matting backend from active routes.
- Route unknown backgrounds to PyMatting fallback; auto no longer invokes RMBG.
- Run route analysis, parameter selection, CorridorKey/PyMatting execution,
  ShadowPatch, and metadata generation in the remote `ErmbgRouteMatte` node.
- Use CorridorKey for green/blue known-screen icon, character, glass, and
  translucent button profiles.
- Use PyMatting Known-B for deterministic hard buttons and unknown/unstable
  fallback.

## Mainline Architecture

```text
uploaded image
  -> remote ErmbgRouteMatte
      -> screen/color analysis
      -> final execution_profile + parameters
      -> PyMatting Known-B, CorridorKey, passthrough, or PyMatting fallback
      -> ShadowPatch / QA / route metadata
      -> RGBA game UI asset
```

The Mac side should only upload the image, submit the workflow, poll ComfyUI,
and download the result images and metadata. It should not duplicate route
analysis or perform post-matting repair for the production auto path.

## Screen And Color Analysis

The route analyzer returns:

- `screen_mode`: `green`, `blue`, or `unknown`.
- `background_color`: measured sRGB key color.
- `background_confidence`: confidence that the image is a known screen asset.
- `purity_sigma`: how stable the background is across trusted regions.
- `subject_key_color_risk`: whether subject material is close to the key color.
- `execution_profile`: final execution profile, for example
  `corridorkey-transparent-button`, `corridorkey-character`, or
  `pymatting-hard-button`.
- `recommended_settings`: CorridorKey/PyMatting gamma, thresholds, despill,
  refiner, despeckle, hint, color-protection, and ShadowPatch settings.

The analysis should use observable signals rather than sample-specific rules:

- trusted corners and border bands for background candidates;
- OKLab distance to compare green and blue screen hypotheses;
- background variance/purity to decide whether auto mode is safe;
- subject/key-color overlap risk to avoid erasing green or blue subject details;
- connected-component scale to avoid despeckle removing small UI ornaments.

## Execution Profiles

The execution profile is the production contract. CorridorKey semantic profiles
can inform routing, but execution must not re-infer the asset family after the
router has made the decision.

| Execution profile | Path | Notes |
|---|---|---|
| `corridorkey-character` | CorridorKey | full-frame character control, color protection off |
| `corridorkey-transparent-button` | CorridorKey | full-frame glass control, color protection off |
| `corridorkey-effect-icon` | CorridorKey | full-frame effect control for additive or soft-alpha icons |
| `corridorkey-shaped-icon` | CorridorKey | shaped hint for icon material that still needs protection |
| `pymatting-hard-button` | PyMatting Known-B | deterministic hard UI and button families |
| `pymatting-known-bg` | PyMatting Known-B | stable known-background graphic fallback |
| `pymatting-fallback` | PyMatting Known-B | unknown or unstable background fallback |

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
shadow layer     measured from known-background scalar darkening
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

Historical direct CorridorKey blue/green baseline:

```text
out/corridorkey_full_blue_green_baseline_20260531/summary.json
```

Result: 83/83 completed successfully. This predates the final 85-sample
RouteMatte auto contract, but remains useful as a direct CorridorKey reference.
B016-B030 all ran successfully as blue-screen green-button samples.

Latest full RouteMatte baseline:

```text
out/auto_routematte_routefix_20260531/summary.json
out/auto_routematte_routefix_20260531/timing_report.md
```

Result: 85/85 completed successfully through Web/API `backend=auto`, which
submits the remote `ErmbgRouteMatte` node. The active set is 56 buttons, 20
icon/effect samples, and 9 character samples. Route distribution was 37
PyMatting Known-B cases and 48 CorridorKey cases.

The targeted execution-profile verification is:

```text
out/verify_route_profiles_character_glass_icon_20260531/summary.json
```

It verifies B046-B049 as `corridorkey-transparent-button`, I011-I012 as
`corridorkey-effect-icon`, I019-I020 as `corridorkey-shaped-icon`, and
C001-C009 as `corridorkey-character`.

## Web UI And Debug Controls

The Web UI should surface the route decision rather than force users to reason
about raw backends:

- Default backend for game UI work: `auto`, which submits `ErmbgRouteMatte`.
- Show `requested_backend`, selected backend, route, asset kind,
  `execution_profile`, parameter profile, measured background, confidence,
  server elapsed time, and route reasons.
- Manual/debug controls may still target `comfy-corridorkey` or
  `comfy-pymatting-known-b`, but production quality audits should start with
  `backend=auto`.
- Mask editing remains a debug/operator aid. It should feed a coarse hint or
  protection signal, not replace the final detail alpha directly.

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

- run direct `ErmbgRouteMatte` smoke through ComfyUI;
- run focused direct `comfy-corridorkey` and `comfy-pymatting-known-b` smokes
  when a profile-specific backend is changed;
- run real HTTP `/api/matte-candidates` smoke through `127.0.0.1:7860`;
- save batch summaries under `out/` with selected backend, route,
  `execution_profile`, settings, timing, and quality metrics.

## Relationship To Existing ERMBG Work

The previous solid-background/local-ownership work remains useful as fallback
and QA infrastructure, but it is no longer the primary detail-matting roadmap
for game UI assets. ERMBG should focus on the orchestration layer around
CorridorKey and PyMatting: input analysis, profile selection, parameter
adaptation, mask hints, ShadowPatch, diagnostics, batch evaluation, and Web
controls. Unknown-background fallback is PyMatting Known-B with a configured
fallback background, not RMBG.
