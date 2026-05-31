# Solid Background Graphic Main Path

## Why This Is Next

Most near-term ERMBG inputs are expected to be known solid-background graphics:
small UI icons, generated game assets, product-like buttons, badges, panels, and
other crisp foreground objects on a flat green/white/black/gray screen.

For this workload, the current production path is inverted. It runs a general
matting model first, then repairs the model with known-background keying,
shadow cleanup, foreground stabilization, and display-alpha filtering. That is
appropriate as a fallback for photographs, hair, fur, smoke, or unknown
backgrounds, but it overcomplicates the common case where the strongest signal
is already deterministic: known background color plus exterior topology.

The next development phase should promote `solid_bg_graphic` to the primary
path for high-confidence solid-background graphics. BiRefNet/full matting should
be the fallback, not the first authority.

## Design Principle

Decide ownership before alpha.

The path should classify connected regions into semantic ownership roles, then
compute alpha/foreground per role:

- `external_background`: flood-filled known background reachable from the image
  border.
- `opaque_subject`: non-background material connected to the subject core.
- `subject_hole`: known-background-colored regions enclosed by subject topology.
- `soft_subject_layer`: glow, antialiasing, translucent material, or glass owned
  by the subject.
- `shadow_layer`: exterior scalar darkening where `C_linear ~= scale * B_linear`.
- `unknown_fallback`: regions whose ownership cannot be proved analytically.

This avoids forcing one alpha rule to cover subject material, holes, shadows,
and transparent effects.

## Proposed Analytic Path

1. Confirm solid background.
   - Measure corners and exterior samples.
   - Require stable known `B` and a dominant exterior background component.
   - Reject or fallback if the background is unstable, textured, or gradients
     cannot be explained by scalar darkening.

2. Build exterior background topology.
   - Flood fill from image borders through pixels close to known `B`.
   - Include exterior scalar darkening as background-side evidence when it
     matches `C_linear ~= scale * B_linear`.
   - Do not classify enclosed known-background-colored areas as exterior simply
     because their color matches `B`.

3. Build subject ownership.
   - Subject core is non-background material not reachable from exterior.
   - Subject interior should be opaque unless classified as a soft subject layer
     or transparent hole.
   - Subject components should be connected/topologically anchored, not chosen
     by isolated pixel thresholds.

4. Resolve holes.
   - A known-background-colored region enclosed by subject topology is a
     candidate hole.
   - Transparent holes remain alpha `0` only when topology proves they are
     enclosed openings, not when they are same-colored decoration/highlight.
   - Internal subject material that resembles background color must remain
     subject unless it is connected to exterior background or classified as a
     true hole.
   - Interior regions may be proved by exact `B` or by strict background-color
     family darkening. The exact-`B` part is transparent hole; the darkened
     same-family part is hole-side neutral shadow. Exterior-reachable
     background-family darkening is excluded before this proof so outer shadow
     does not masquerade as an internal hole.
   - Once an interior hole is proved, it becomes a local background seed. Its
     edge, antialiasing, residual background color, and scalar darkening should
     use the same known-background evidence model as the outer edge. This is the
     inner-hole rule: hole-side darkening is neutral shadow, not green
     foreground.
   - This must be separated from connected internal background-colored material:
     such material stays subject-owned unless topology proves a true opening or
     the candidate-selection flow exposes an alternate translucent
     interpretation.

5. Resolve soft subject layers.
   - Antialiasing, glow, and translucent material belong to subject when they
     are connected to or constrained by subject topology and do not match the
     scalar-background shadow model.
   - Estimate alpha from known `B` only inside narrow ownership-supported
     boundary/soft-layer regions.
   - Low-alpha soft pixels that are still near-pure background color family are
     background leaks, not translucent subject material. Drop their alpha rather
     than exporting green wisps on dark/new backgrounds.
   - Stabilize foreground color from subject material; do not trust raw
     inverse-composited RGB where background contribution dominates.

6. Resolve glass/translucent buttons.
   - Opaque frame, icon, text, and highlights are subject material.
   - Glass body is a `soft_subject_layer`, not a background hole.
   - Background-colored pixels inside the button are transparent only if
     topology proves an opening; otherwise they remain subject material with
     lower alpha or recovered foreground color.

7. Resolve shadows separately.
   - Shadow is an exterior scalar-darkening layer, not subject alpha.
   - Shadow must be coherent as a connected low-frequency field; detached
     display-visible islands should be rejected by area/topology evidence.
   - Keep clean foreground RGB separate from the RGB companion used for final
     shadow-preserving RGBA.
   - Visual rule: compositing the exported RGBA shadow layer back over the
     original known background should preserve source luminance, not source
     chroma. Exact RGB reconstruction over saturated green can encode green
     into the shadow layer; reusable RGBA should prefer neutral darkening with
     brightness consistency.

8. Fallback only when the analytic path cannot prove ownership.
   - Examples: photographic subject, hair/fur, smoke, complex transparency,
     unknown/mixed background, or insufficient color separation.
   - Fallback should be explicit in debug output so regressions can reveal
     whether the router chose the wrong path or the analytic path failed.

## Required Outputs

The API and Comfy path should preserve separate output semantics:

- `foreground`: clean subject-color layer for inspection/decontamination.
- `alpha`: final display alpha.
- `rgba`: final transparent result, allowed to include shadow compositing.
- `rgba_rgb` internally/for Comfy: RGB companion paired with `alpha` to preserve
  shadows in the final RGBA.
- debug ownership masks: at least exterior background, opaque subject, hole,
  soft subject, shadow, and fallback/unknown.

## Candidate Selection TODO

When multiple ownership/color interpretations are all inside their confidence
range, the system should not make a one-way cut. It should choose the
highest-confidence interpretation as the default, but keep the other plausible
interpretations as explicit candidates for user selection.

This is required for cases such as internal background-colored material where
both of these explanations may be valid:

- protect the whole connected region as subject material;
- solve the whole connected region as a smooth translucent known-background
  layer.

Implementation requirements:

1. Move ambiguous solid-graphic decisions from pixel-local gates to
   component/region-level hypotheses.
2. Rank hypotheses by local evidence and confidence; use the highest score as
   the default output.
3. Preserve alternate high-confidence hypotheses as named candidates with
   debug reasons, masks, and output RGBA/alpha/foreground artifacts.
4. Extend the Web/API flow so these candidates are returned and selectable
   instead of being hidden in one final matte.
5. Add eval coverage where the same sample exposes at least two plausible
   interpretations, and ensure the UI can compare them side by side.

## Perceptual Local-Diffusion TODO

Known background color is the physical anchor for solving `C = aF + (1-a)B`,
but it is not enough as the only execution threshold. Glass/translucent samples
show the failure mode: tiny hue/luma changes that are almost invisible on the
original green screen can become obvious purple/black/grey speckles after
inverse compositing.

For soft glass, glow, antialiasing, holes, and shadows, thresholds should be
anchored to human-visible local continuity:

- Keep the known background color as the global proof that a region is
  background-family and physically solvable.
- Diffuse tolerance from nearby high-confidence background-family pixels inside
  the same soft/hole context, using perceptual OKLab distance rather than raw
  RGB absolute difference.
- Allow the local threshold to be wider than the global background threshold
  only when local continuity, ownership context, and background-family channel
  direction all agree.
- Do not let gradual subject color changes make unrelated purple/blue material
  transparent; candidate pixels still need the known-background-family channel
  evidence and local soft/hole context.

Current implementation status:

- `solid_graphic` now has a local OKLab continuity gate for background-family
  glass speckles. It catches pixels that are not quite close enough to the flat
  screen color globally, but are perceptually continuous with neighboring
  screen-colored glass pixels.
- Regression coverage includes a false-hue guard so inverse solving cannot turn
  near-background source pixels into exported purple foreground.

Conversation summary, 2026-05-28:

- The problem is not that the source pixels have no mathematical difference;
  the problem is that the difference is below practical human visibility on the
  original green background, then becomes visible because inverse compositing
  amplifies it into exported foreground.
- A pure global "distance to background green" threshold is insufficient. The
  eye reads local continuity, so the algorithm must consider nearby pixels in
  the same soft/hole/glass context.
- The correct rule is two-stage:
  1. known background color proves the physical equation and background-family
     channel direction;
  2. local perceptual continuity decides whether a slightly wider threshold is
     allowed.
- This applies to:
  - glass interiors with lifted green/cyan scatter that should not become
    grey/black dirt;
  - low-alpha soft layers where the solved foreground flips to purple/blue even
    though the source pixel is visually continuous with the screen color;
  - internal holes or hole-adjacent soft regions, because a hole is local
    background and should use the same evidence model as the outer edge;
  - inner-hole shadows, where darkened background-family pixels inside a proved
    hole should export as neutral luminance shadow rather than subject-owned
    green/grey foreground;
  - soft shadows/glow only when ownership already proves the region belongs to
    background-family evidence rather than subject color.
- This must not apply to:
  - real subject-owned green material such as connected internal UI material;
  - broad true color gradients where green transitions into blue or magenta
    subject material;
  - colored glow unless local ownership and background-family channel direction
    both support removing a specific residual.

Implementation notes:

- Local diffusion uses OKLab, not raw RGB, because the threshold is about
  visibility rather than arithmetic channel difference.
- Seed pixels remain globally close to the known background. Neighbor pixels
  can only inherit tolerance from those seeds inside a matching soft/hole
  context.
- Hole seeds are topology-gated, not color-gated. Exact-B pixels and strict
  background-family darkening only become hole evidence after the basin is
  enclosed by subject topology and not reachable from the exterior.
- The local gate still requires background-family channel direction, so it
  cannot diffuse through a smooth subject gradient and make purple material
  transparent.
- Inverse-solved foreground is treated as a suspect signal when it invents hue:
  a source pixel that is visually background-family should not be exported as
  purple/blue/black just because the algebra is under-constrained at low alpha.

Open follow-up:

- Promote ambiguous component decisions into explicit candidates. When both
  "protect as subject material" and "solve as smooth translucent layer" are
  plausible, use the highest-confidence interpretation by default but expose
  the alternate result in Web/API instead of hard-coding one final matte.
- Add Web/API candidate support for ambiguous internal same-screen-color
  regions: default to the highest-confidence ownership, but expose "protect as
  subject material" versus "solve as smooth translucent known-background layer"
  when both fit the evidence.

## Tests To Add First

Add synthetic mechanism tests before replacing production routing:

- Crisp icon on green with subject touching borders.
- White/black/green/gray solid backgrounds.
- Enclosed transparent hole on known background.
- Scalar darkening inside an enclosed hole that must become neutral shadow, not
  subject-owned green foreground.
- Dark same-background-family interior holes even when no exact-`B` seed exists.
- Same-color internal decoration that must not become a hole.
- Exterior soft shadow that must stay out of subject alpha.
- Known-background shadow RGBA whose neutral shadow layer preserves source
  luminance over the original background after 8-bit export.
- Detached scalar-darkening specks that must be rejected.
- Semi-transparent glow around subject that must remain subject-owned.
- Low-alpha background-green leaks inside a soft layer that must be removed
  without touching high-alpha same-hue subject material.
- Glass button where the interior is translucent subject material, not a hole.
- Photo or hair-like sample that must fallback to the existing matting path.

Use the current `samples/corridorkey_semantic/manifest.json` cases as the
default production regression surface. Add real user failures under
`samples/regression/<case_id>/` only when they introduce a mechanism not already
covered by the semantic set, and do not tune the analytic path around any single
file.

## Migration Plan

1. Implement `ermbg/solid_graphic.py` as an isolated analytic engine returning
   alpha, foreground, ownership masks, and a confidence/fallback reason.
2. Add focused synthetic tests for ownership roles before wiring it into Web.
3. Teach the router to choose `solid_bg_graphic` only for high-confidence
   stable solid backgrounds and graphic-like assets.
4. In `matting.py`, try `solid_bg_graphic` before building/running the
   segmenter; fallback to the existing path when confidence is insufficient.
5. Update Comfy/Web debug summaries to expose the chosen path and ownership
   masks.
6. Run local tests, direct remote `backend="comfy-ermbg"` regression, and real
   Web HTTP smoke.

## Pixel-Patch Cleanup TODO

The current BiRefNet-first production path contains many pixel-level repair
rules that were added to compensate for the model being the first authority on
known solid-background graphics. These rules should not be carried forward as
the main design. They should be converted into a cleanup ledger and retired
only after the analytic path proves the corresponding failure mechanism.

Track each old repair with this table shape:

```text
old repair -> failure mechanism -> solid_graphic ownership role -> tests/eval coverage -> delete condition
```

Repairs that should migrate into `solid_bg_graphic` ownership semantics:

- Saturated known-background low-alpha interior repair.
- Saturated opaque-interior alpha snap.
- Saturated hard-edge key resolve / alpha raise-lower.
- White/black known-background hole repair for hard graphics.
- Exterior scalar-darkening reclassification.
- Same-background-color hole vs same-color subject decoration.
- Glow/glass/translucent subject material protection.

Current cleanup status:

- Saturated hard-edge key resolve / alpha raise-lower: main-path coverage added.
  `solid_graphic` now treats scalar-looking green-screen contour pixels that are
  glued to strong subject material as `soft_subject_layer`, not `shadow_layer`.
  The old `saturated_hard_edge_key_resolve` remains only for fallback/injected
  segmenter flows until real batch coverage confirms it can be narrowed further
  or removed.

Repairs that should remain temporarily in the fallback path:

- Photographic, hair, fur, smoke, and ambiguous soft-matte handling.
- Generic foreground export stabilization for weak alpha regions.
- Generic shadow estimation for non-analytic fallback results.
- Source-alpha hygiene and dirty-RGBA rematting.
- Semantic-prior subject/material/shadow protection.

Deletion order:

1. Add a focused synthetic test for the failure mechanism in `solid_graphic`.
2. Add or rerun a real regression/eval batch when the failure came from a user
   sample.
3. Confirm high-confidence solid-background graphics choose `solid_bg_graphic`
   and ambiguous/photo cases still fallback.
4. Remove or narrow the old pixel-level repair from the BiRefNet path.
5. Keep a fallback regression proving the removed repair was not still needed
   for photographic/ambiguous inputs.

Do not delete all old repair code in one sweep. Retire one mechanism at a time,
with tests proving the new ownership role has replaced the old patch.

## Non-Goals

- Do not add sample-id or coordinate-specific rules.
- Do not continue piling display cleanup rules onto the fallback path for cases
  that should be solved analytically.
- Do not remove the current BiRefNet path; it remains necessary for genuinely
  photographic or ambiguous images.
