# Sample 12 White Background Panel Analysis

Date: 2026-05-25

## Context

`samples/inputs/12.png` is a white-background graphic asset with a pale green
panel and vine leaves around the frame. During a `white_bg` run, the upper-right
area where leaves overlap the pale panel showed a visible cut-out: part of the
panel was treated as background/transparent.

The first attempted repair was to let the luminance keyer raise low-alpha pixels
inside an already represented foreground component. That filled the upper-right
panel hole, but it also made the overall boundary visibly dirty on dark
backgrounds, with obvious white-background residue.

That repair should not be considered a valid fix.

## Root Cause

There are two different failure modes that were accidentally mixed together.

1. The original panel hole is a matting-model recall problem.

   The asset uses a white background, while the panel is also bright and low
   contrast. In the upper-right overlap area, BiRefNet-matting assigns low alpha
   to some pale panel pixels near the leaves. The luminance keyer sees much of
   that panel area as foreground, but the existing keyer merge only restores
   separately missed connected components. It does not repair holes inside a
   large component that the matting model already mostly found.

2. The attempted fill created foreground-color contamination.

   Raising alpha across the whole keyer component also raises alpha on external
   antialiasing and soft edge pixels. Those pixels still contain the original
   white-background mixture in `C`. Once their alpha is raised too high,
   `unmix` can no longer remove enough background:

   ```text
   F = (C - (1 - alpha) * B) / alpha
   ```

   If the true edge alpha should be around 0.3 but is forced toward 0.9 or 1.0,
   the formula subtracts almost none of the white background. The resulting
   foreground keeps the white mixture, which appears as white edge residue on
   black or saturated QA backgrounds.

The key distinction is:

- Interior hole repair can be useful.
- External boundary alpha raising is dangerous and causes halos.

The pipeline needs to distinguish those two topologically, not with only a
global keyer threshold.

## Observations From Sample 12

- Strategy: `white_bg`
- Background estimate: `[254, 254, 254]`
- Keyer mode: `luminance`
- Despill: `unmix`
- Original output had a visible upper-right panel hole.
- The broad component-fill attempt improved recomposition on the observed white
  background, but made edge quality worse on black background.
- The bad repair lifted low-alpha pixels both inside the panel and along the
  subject's outside antialiased contour. These two regions share the same large
  keyer connected component, so plain connected-component merging is too coarse.

## What Not To Do

- Do not set `alpha = max(matting_alpha, key_alpha)` for the whole represented
  component.
- Do not use luminance keyer output as a direct alpha replacement on white or
  black backgrounds.
- Do not tune only `bg_max` / `fg_min` thresholds and expect this class of issue
  to be solved; the failure is spatial/topological.
- Do not treat a lower recomposition error on the original white background as
  sufficient proof of quality. Dark-background QA can expose foreground
  contamination that white-background recomposition hides.

## Candidate Fix Directions

### 1. Interior-Hole-Only Repair

Create a dedicated repair pass for pixels where:

- keyer alpha is high,
- matting alpha is low,
- the candidate region is enclosed by confident foreground,
- the candidate region does not touch the external background,
- the candidate region is not part of the subject's outer antialiasing band.

Possible implementation shape:

1. Build a confident foreground mask from matting alpha, for example
   `alpha >= 0.85`.
2. Build a keyer foreground candidate mask, for example `key_alpha >= 0.75`.
3. Find candidate holes where keyer says foreground but matting alpha is low.
4. Reject candidates connected to image exterior or to the exterior background
   after flood-fill.
5. Reject candidates close to the external contour using distance transform or
   contour ownership.
6. Fill only accepted interior holes, with feathered alpha at the boundary.

This keeps the repair away from external edges.

### 2. Component Topology With Exterior Flood Fill

Instead of looking only at keyer connected components, explicitly label the
outside background:

- Threshold matting/keyer into a rough foreground support.
- Flood-fill background from the image border.
- Any low-alpha candidate connected to that exterior region is boundary fringe,
  not an interior hole.
- Only enclosed regions are eligible for repair.

This should handle large subjects where the panel and frame are one connected
component.

### 3. Separate Alpha Repair From Foreground Recovery

If alpha is raised for an accepted interior hole, foreground color recovery must
also be reconsidered. For interior panel holes, using `unmix` with raised alpha
may be acceptable because the region is real foreground. For soft external
edges, it is not.

A future patch should make the repair mask available to despill/recovery, so
different color handling can be applied to:

- repaired interior holes,
- original model soft edges,
- external keyer-gated halo cleanup.

### 4. Prefer Saturated Probe Backgrounds For Pale Subjects

When generation can be controlled, white is a poor probe background for pale
panels, white objects, translucent glass, or lightly colored UI/card assets.
Saturated green/cyan/magenta backgrounds give the keyer a chromatic signal and
avoid the information loss that happens when foreground and background share
lightness.

For this class of asset, the generation contract should prefer the existing
green-screen convention over white-background output.

### 5. Add A Regression Case

Before implementing a production repair, add a regression asset or synthetic
case that includes:

- pale interior panel on white background,
- dark/green decorative foreground over the panel,
- external soft antialiased contour,
- QA assertion on black background to catch white residue.

The test should fail if a repair fills the panel but worsens the outer edge.

## Proposed Acceptance Criteria

A future fix for sample 12 should satisfy all of these:

- Upper-right panel is not cut out.
- Black-background QA does not show new white residue on the outer edge.
- Edge halo score on black does not regress materially compared with the
  pre-repair output.
- Recomposition error on the observed background improves or stays comparable.
- The repair mask is spatially limited to interior holes, not the whole keyer
  component.

