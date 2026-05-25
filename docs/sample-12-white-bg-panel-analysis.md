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

## 2026-05-25 Design Update: Subject Ownership First

After further discussion, the problem was reframed more narrowly:

```text
The core question is not "is this a hole?"
The core question is "does this region belong to the subject?"
```

Pure topology is not enough. A flood-fill / enclosed-region rule can identify
regions that look like holes, but it cannot distinguish:

- a true matting-model recall hole inside a pale panel,
- an intentional transparent opening in a frame, logo, wreath, or decorative
  structure,
- background visible through a real gap,
- outer antialiasing pixels that still contain white-background mixture.

For sample 12, the desired repair is the first case: a pale green panel region
that should be subject-owned, but received low alpha from the matting model.
For other assets, an enclosed low-alpha region may be intentional transparency
and must not be filled.

The practical decision is therefore:

- Do not let keyer output decide subject ownership by itself.
- Do not treat topology-only hole detection as production proof.
- Require an independent `subject_mask` / `subject_support` ownership signal
  before repairing low-alpha interior regions.
- Use keyer and topology as constraints after ownership is established, not as
  replacements for subject recognition.

The ownership-gated repair rule is:

```text
repair_region =
  subject_support
  AND keyer_foreground_confident
  AND matting_alpha_low
  AND not_exterior_fringe
  AND anchored_to_confident_matting_foreground
```

This keeps the existing ERMBG path conservative. Without a `subject_mask`, the
sample 12 output is not automatically changed; the system refuses to invent
semantic ownership from luminance keying alone.

## Current Implementation Notes

The first conservative implementation added an optional subject-ownership repair
path:

- `ermbg.keyer.repair_alpha_with_subject_support(...)`
- `ermbg.matting.matte(..., subject_support=...)`
- `ermbg.api.matte_image(..., subject_mask=...)`
- ComfyUI optional `subject_mask` input on `ERMBG AutoMatte`

The repair function only raises alpha inside the supplied support mask. It also
rejects candidates near the exterior background using a distance margin, and
requires each accepted component to touch confident existing foreground after a
small dilation. This avoids using the keyer as a whole-component alpha
replacement.

The intended upstream source of `subject_mask` is a stronger or more targeted
subject recognizer, for example:

- a second segmentation model,
- a SAM / grounding segmentation workflow,
- a ComfyUI segmentation branch,
- a prompt-aware object mask when generation context is available.

Current tests cover three safety properties:

- a subject-owned interior low-alpha region can be repaired,
- a true transparent hole excluded from `subject_mask` is not repaired,
- outer contour fringe is not lifted even if keyer confidence is high.

This is a deliberately conservative interface-level fix, not a full automatic
sample 12 solution. The remaining work is to produce or route a reliable
subject-ownership mask for sample 12, then run QA against black / saturated
backgrounds to confirm that the upper-right panel is restored without adding
white edge residue.

## 2026-05-25 Experiment: Ownership Mask Sources

Two candidate `subject_mask` sources were tested against `samples/inputs/12.png`.
Both used the same ERMBG white-background path; only the optional ownership mask
changed.

### RMBG / IS-Net as ownership mask

Remote ComfyUI `Image Rembg (isnet-general-use)` was tried first because it is
already available as the RMBG baseline.

Result:

- It detected the vine frame and leaves well.
- It did not treat most of the pale green panel as subject-owned.
- It authorized only a very small repair:
  - `accepted_components = 2`
  - `accepted_pixels = 350`
- It hit the visible upper-right triangular miss, but did not provide a
  complete semantic ownership mask for the panel.

QA comparison:

| Variant | Recomp err | Black halo | Halo mean |
|---|---:|---:|---:|
| no `subject_mask` | 0.0308 | 9.9987 | 3.3779 |
| RMBG alpha as `subject_mask` | 0.0298 | 9.8343 | 3.3768 |

Conclusion: RMBG alpha is useful evidence for the leaf/frame foreground, but it
is not enough for sample 12 because the panel itself is the ambiguous subject
part.

### CLIPSeg prompt-aware ownership mask

ComfyUI `CLIPSeg Masking` was then tested with prompts such as:

```text
pale green panel with vine leaves frame
the entire framed green panel
```

These masks covered the intended object much better: the pale panel and the
decorative vine frame were both included as one subject-owned region.

Using `clipseg_0` (`pale green panel with vine leaves frame`) as
`subject_mask`:

- `accepted_components = 35`
- `accepted_pixels = 6522`
- `rejected_components = 444`

Using `clipseg_3` (`the entire framed green panel`) as `subject_mask`:

- `accepted_components = 29`
- `accepted_pixels = 6355`
- `rejected_components = 416`

QA comparison:

| Variant | Recomp err | Black halo | Halo mean |
|---|---:|---:|---:|
| no `subject_mask` | 0.0308 | 9.9987 | 3.3779 |
| CLIPSeg prompt 0 | 0.0120 | 6.0433 | 3.1480 |
| CLIPSeg prompt 3 | 0.0125 | 5.9681 | 3.1298 |

Visual crop inspection confirmed that the upper-right panel cut-out is repaired
without the broad white-edge contamination caused by the earlier whole-component
fill attempt.

Conclusion: for this class of asset, prompt-aware subject ownership is the right
second signal. The mask does not need to be the final matte. It only needs to
answer the ownership question: "is this region part of the object that should
survive transparency?"

## Current Recommendation For Sample 12

For production handling of sample 12-like assets:

1. Generate or provide a prompt-aware `subject_mask` for the complete object,
   including pale panels and decorative elements.
2. Run ERMBG with `subject_mask` as an ownership constraint.
3. Keep the existing known-background keyer and exterior-fringe rejection as
   safety checks.
4. Validate on black/checker/cyan/magenta QA backgrounds, not only on the
   original white background.

The preferred near-term automation path is a ComfyUI branch:

```text
LoadImage
  -> CLIPSeg / Florence / SAM prompt-aware object mask
  -> ERMBG AutoMatte(subject_mask=...)
  -> Save RGBA + QA
```

The plain ERMBG API path remains conservative by default: without a
`subject_mask`, no semantic hole repair is attempted.

## 2026-05-25 General Architecture: Semantic Matting Planner

The sample 12 issue should be treated as evidence for a broader architecture,
not as a reason to tune one image. The general problem is:

```text
matting = object ownership segmentation
        + alpha estimation
        + foreground color recovery
```

Classic matting and keying mainly solve the last two parts. They can estimate
soft edges and remove known background color, but they do not understand which
regions semantically belong to the object. Sample 12 fails because the pale
panel is subject-owned but visually close to the white background.

The proposed next-stage ERMBG architecture has three roles:

```text
Vision model       -> describe the image and mark semantic regions
Language planner   -> choose region-specific matting operations
Local algorithms   -> execute deterministic alpha/color recovery and QA
```

The vision / language models should not directly replace the final matte. Their
job is to provide semantic structure and a plan. Pixel-level execution remains
local, testable, and inspectable.

### 1. Vision Model: Annotation And Region Evidence

The vision model should answer questions such as:

- What is the intended subject?
- Which regions are subject-owned?
- Which regions are background?
- Which openings are intentional holes?
- Which parts are opaque, soft, translucent, reflective, or low contrast?
- Which areas are ambiguous and need conservative treatment?

For sample 12, a good annotation would say:

```json
{
  "subject": {
    "description": "a pale green rectangular panel with a vine leaf frame",
    "support_mask": "mask_id_subject",
    "parts": [
      {
        "name": "pale green panel",
        "mask": "mask_id_panel",
        "material": "opaque flat graphic",
        "should_keep": true
      },
      {
        "name": "vine leaves and stems",
        "mask": "mask_id_vines",
        "material": "opaque illustrated foliage",
        "should_keep": true
      },
      {
        "name": "outside white background",
        "mask": "mask_id_background",
        "should_keep": false
      }
    ],
    "intentional_holes": []
  }
}
```

For a real cut-out frame, wreath, jewelry item, logo, or window, the same layer
must also be able to mark intentional holes:

```json
{
  "name": "inner opening of the frame",
  "mask": "mask_id_hole",
  "should_keep": false,
  "reason": "intentional transparent opening"
}
```

This distinction is exactly what topology-only hole filling cannot provide.

### 2. Language Model: Region-Specific Matting Plan

The language model should not write arbitrary image-processing code. It should
produce a constrained plan over known ERMBG operations.

Example plan for sample 12:

```json
{
  "strategy": "known_white_background_semantic_support",
  "regions": [
    {
      "target": "pale green panel",
      "operation": "repair_low_alpha_inside_subject_support",
      "constraints": {
        "use_keyer_evidence": true,
        "avoid_exterior_fringe": true,
        "max_alpha_raise_on_outer_edge": 0
      }
    },
    {
      "target": "vine leaves and stems",
      "operation": "preserve_matting_edges_with_unmix",
      "constraints": {
        "allow_soft_edges": true,
        "despill": "unmix"
      }
    },
    {
      "target": "outside white background",
      "operation": "force_transparent"
    }
  ],
  "qa": ["black", "checker", "cyan", "magenta"]
}
```

The allowed operation set should be finite and local:

```text
repair_low_alpha_inside_subject_support
preserve_soft_edges
force_background_transparent
unmix_known_background
chroma_cap
local_borrow
edge_gate
hole_preserve
qa_composite
```

This keeps the system explainable. The planner decides *which* operation applies
to each region; ERMBG decides exactly *how* it is executed.

### 3. Local Algorithms: Deterministic Execution

The local pipeline should consume the semantic plan and execute pixel operations:

- background diagnosis,
- keyer evidence,
- matting-model alpha,
- ownership-constrained alpha repair,
- exterior fringe protection,
- foreground color recovery,
- multi-background QA.

This preserves the current ERMBG strengths. Known-background `unmix`, despill,
lightwrap, QA, and edge gating remain deterministic code paths with tests.

### Required Intermediate Representations

The architecture needs richer state than a single mask:

```text
OwnershipMap   which pixels belong to the intended subject
PartMap        subject parts and material/role labels
HoleMap        intentional transparent openings
RiskMap        regions where matting/keyer/semantic evidence disagree
Plan           local operations selected per region
```

`OwnershipMap` should be probabilistic and source-aware:

```text
support_prob: HxW float32
sources:
  matting_alpha
  keyer_alpha
  prompt_segmentation
  detector_or_sam
  upstream_user_mask
risk_flags:
  low_contrast_with_background
  possible_intentional_hole
  exterior_soft_edge
  foreground_background_disagreement
```

Typical fusion cases:

| Evidence pattern | Interpretation | Action |
|---|---|---|
| matting high + keyer high | confident foreground | preserve |
| matting low + keyer high + semantic support high | likely recall hole | repair conservatively |
| matting low + keyer high + semantic support low | keyer false positive or background | reject |
| matting high + keyer low | soft edge / same-color subject / translucent material | protect matting |
| semantic support high + keyer low | possible low-contrast subject | keep but mark risky |

### Trigger Policy

Vision-model annotation should be used when cheap local evidence is insufficient,
not necessarily on every image.

Escalation triggers:

- low contrast between subject candidate and background,
- large matting/keyer disagreement,
- enclosed low-alpha regions inside likely subject support,
- possible intentional holes,
- transparent or translucent materials,
- QA failure on black/checker/saturated backgrounds,
- user intent selects a specific object among multiple objects.

Simple cases should stay local:

- clean RGBA pass-through,
- saturated green/cyan/magenta background with strong keyer agreement,
- high-confidence BiRefNet/RMBG alpha with clean QA,
- no disagreement or semantic ambiguity.

### Guiding Principle

The long-term ERMBG direction should be:

```text
Vision model understands.
Language model plans.
Local code executes and verifies.
```

This avoids both weak pixel-only heuristics and opaque end-to-end black-box
matting. It lets ERMBG stay engineering-first while adding semantic awareness
only where the local evidence cannot answer the ownership question.

## 2026-05-25 Current Conclusion

The current conclusion is:

```text
The core ERMBG problem is not "which hole should be filled?"
The core problem is "which regions belong to the intended subject?"
```

Sample 12 demonstrates this clearly. The pale green panel is visually close to
the white background, so alpha matting and luminance keying alone can treat part
of the panel as background. But semantically that region is part of the subject:
the intended object is the complete pale panel plus the vine frame.

Therefore, the reliable general solution is not to keep tuning thresholds for a
single image. ERMBG needs a semantic-aware matting pipeline:

1. A vision model provides accurate image understanding and region annotation:
   subject support, parts, background, intentional holes, ambiguous areas, and
   materials.
2. A language planner reasons over those annotations and selects constrained
   region-level operations from ERMBG's known algorithm set.
3. Local deterministic code executes the plan: alpha repair, keying, unmix,
   despill, edge protection, foreground recovery, and QA composites.

In this architecture, traditional CV algorithms are execution tools. They should
not be asked to decide semantic ownership by themselves. Vision models provide
the missing ownership evidence; the language model decides how that evidence
should route local algorithms; ERMBG keeps pixel execution testable and
inspectable.

The sample 12 experiments support this direction:

- RMBG / IS-Net alpha was not sufficient as an ownership mask because it found
  the leaves and frame but missed much of the pale panel.
- CLIPSeg prompt-aware support covered the intended complete object and allowed
  the conservative subject-mask repair to fix the panel miss without recreating
  the earlier broad white-edge contamination.

The next engineering direction should therefore be a general `semantic matting
planner`, not a sample-specific parameter patch. The system should escalate to
vision-model annotation when local evidence cannot answer ownership safely,
especially for low-contrast objects, intentional openings, translucent parts,
and large matting/keyer disagreements.

## 2026-05-26 Product Reframe: Interactive Intent Matting

After discussing the tool as something people actually use, the second-stage
goal was reframed again. The desired product behavior is not:

```text
Ask the user to draw a perfect mask.
```

It is:

```text
Default to automatic.
When automatic evidence is ambiguous, let the user add a tiny amount of intent.
Let ERMBG do the pixel-level refinement.
```

This preserves the earlier semantic-planner direction, but makes the user
interaction lighter and more explicit. The user should not need to trace edges,
paint alpha, or understand matting internals. A rough stroke or a short sentence
is enough if it tells the system what it cannot infer safely:

- "This is a wreath; the center hole should be transparent."
- "Keep the whole pale green panel."
- "Only keep the badge on the left."
- A loose keep stroke over the region that should survive.
- A loose remove stroke over a region that should disappear.
- A loose hole stroke over an intentional opening.

The important distinction is:

```text
User input is intent evidence, not the final matte.
```

In implementation terms, `subject_mask` becomes one member of a broader family
of lightweight intent inputs:

```text
instruction   natural-language intent
keep_mask     rough region that should belong to the subject
remove_mask   rough region that should become background
hole_mask     rough intentional transparent opening
subject_mask  prompt-aware ownership support from CLIPSeg / Florence / SAM
```

All of them should be converted into internal maps:

```text
OwnershipMap
BackgroundMap
HoleMap
RiskMap
MattingPlan
```

Then local ERMBG operations execute the plan:

- repair low-alpha pixels only inside owned support,
- preserve / force transparent user-marked holes,
- reject repair near exterior antialiasing,
- use known-background keyer evidence as a constraint,
- recover foreground color with unmix / despill,
- verify with black / checker / cyan / magenta QA.

### Interaction Policy

The ideal interaction policy is:

1. Try the automatic path first.
2. If QA or evidence disagreement indicates ambiguity, expose one small
   correction action.
3. The correction should express intent, not require precision.
4. The system should show debug overlays so the user can see what was accepted
   or rejected.

Examples:

```text
Detected low-contrast foreground on white. Roughly mark the area to keep.
Detected an uncertain internal opening. Mark it as "hole" if it should be transparent.
Detected multiple possible subjects. Roughly mark or describe the one to keep.
```

### New Phase-2 Definition

The practical Phase-2 target is therefore better named:

```text
Interactive Intent Matting v0
```

Scope:

- Keep current no-parameter automatic behavior for easy cases.
- Accept rough `keep/remove/hole` masks and short instructions.
- Treat those inputs as constraints, not as final alpha.
- Produce an explainable plan and report.
- Keep pixel execution deterministic and locally testable.

Out of scope for Phase 2:

- Fully automatic remote vision calls on every image.
- LLM-generated image-processing code.
- Directly using prompt masks as finished alpha.
- Solving hard translucent materials perfectly.

This reframing keeps the full project evolution intact:

```text
known-background matting
  -> router/keyer/despill/QA
  -> RGBA hygiene
  -> subject_mask ownership repair
  -> semantic ownership planning
  -> lightweight human intent correction
```
