# Known-B Candidate Matting

Date: 2026-05-26

This document holds the longer design notes that used to live in
`clean_transparent_matting_engineering_plan.md`. The main engineering document is
now a short handoff entry point; this file keeps the reasoning for candidate
generation, evidence-to-policy mapping, and VLM planner boundaries.

## Current Contract

The main line remains:

```text
Known background color B -> robust alpha + foreground recovery
```

Priorities:

1. Saturated / white / black / grey backgrounds should all use known-B evidence
   as much as possible.
2. Keyer output is evidence and constraint, not a direct replacement for final
   alpha.
3. Interior low-alpha repair must have topology guards; external soft contours
   must not be lifted blindly.
4. Multi-background QA is the acceptance layer, especially black / checker /
   saturated backgrounds.
5. Only foreground/B same-color ambiguity, object selection, true-hole
   ambiguity, or local material-policy conflict should escalate to candidates,
   light interaction, or VLM semantic hints.

Phase 2 name:

```text
Known-B Candidate Matting
```

It has two parts:

1. **Robust Alpha Fusion**: when pixel color is clearly not B but matting alpha
   is low, use known-B/keyer evidence plus topology guards to conservatively
   raise alpha.
2. **Ambiguity Candidate Generation**: when pixel color equals B and could be a
   transparent hole or subject marking, generate a small set of candidates for
   user selection.

## Historical Notes

The sample-12 white-background panel analysis is archived at
[docs/archive/sample-12-white-bg-panel-analysis.md](archive/sample-12-white-bg-panel-analysis.md).
It records temporary iterations and reversals, but is no longer a current design
source.

G02-G soft shadow work is tracked separately at
[docs/g02-soft-shadow-analysis.md](g02-soft-shadow-analysis.md). Keep G02 same
color material issues out of shadow acceptance; those should be avoided upstream
by choosing a non-colliding generation background.

## Evidence-To-Policy Map

Some images contain different local matting policies inside one asset:

- black outlines should behave like hard edges;
- hair/fur/smoke should preserve continuous soft alpha;
- glass/fabric/translucent material needs partial alpha plus foreground
  recovery;
- same-background interior regions may be transparent holes or subject markings.

The architecture separates two layers:

1. Local CV outputs **EvidenceRegion**: measured pixel/topology evidence only.
2. VLM/LLM or user intent interprets EvidenceRegion as **RegionPolicy** /
   `CandidatePlan`: hard edge, soft edge, transparent hole, same-color marking,
   translucent material, owned shadow, etc.

Current code still uses `RiskRegion` as an implementation name, but its product
meaning is EvidenceRegion. It should not be read as "local code already decided
the semantic policy".

The intended middle representation:

```text
Image / known-B evidence / alpha-keyer disagreement / topology
  -> EvidenceRegion[]
  -> optional VLM/user interpretation
  -> RegionPolicyMap / CandidatePlan[]
  -> finite local MattingPlan
  -> deterministic executor
  -> QA / debug overlays
```

Evidence names should stay measurement-oriented:

| Evidence kind | Meaning | Question for VLM/User |
|---|---|---|
| `same_bg_enclosed_region` | color close to known B, low alpha, enclosed by foreground | transparent hole or same-color subject marking? |
| `alpha_keyer_disagreement` | keyer sees foreground, matting alpha is low | subject miss, outside contour pollution, material, or shadow? |
| `hard_edge_candidate` | high-contrast small-area keyer foreground with low matting alpha | hard outline/text edge or soft edge to preserve? |

For compatibility, code keeps `kind` for executor keys and exposes
`evidence_kind` in planner/VLM JSON:

- `same_bg_enclosed_region` -> `same_bg_low_alpha_enclosed`
- `alpha_keyer_disagreement` -> `keyer_fg_matting_low`
- `hard_edge_candidate` -> `high_contrast_keyer_fg_matting_low`

`ToolCatalog` exposes both `allowed_region_kinds` and
`allowed_evidence_kinds`: the former for local validators/executors, the latter
for VLM understanding.

VLM/LLM output is the semantic layer:

| Policy | Typical region | Execution strategy |
|---|---|---|
| `hard_edge` | logo outline, icon contour, text edge | known-B keyer / edge snapping can lift alpha |
| `soft_hair` | hair, fur, smoke-like edge | preserve BiRefNet soft alpha; prevent hard gate/fill |
| `opaque_interior` | solid subject interior | known-B repair may fix low-alpha holes |
| `translucent` | glass, thin fabric | preserve partial alpha; use unmix / foreground recovery |
| `intentional_hole` | wreath, frame, cutout logo, window | hole constraint wins; do not repair as subject |
| `shadow_or_contact` | cast/contact shadow | handle via shadow-prior path, not foreground material |
| `unknown` | conflicting evidence | generate candidates or request light intent input |

The visual model does not re-measure pixels. It interprets local evidence and
image semantics. Final pixel operations stay in local keyer / matting / unmix /
topology / QA code.

The first landed local rule is `hard_edge`: in white/black graphic paths, only
small high-confidence keyer components with large background contrast and
foreground adjacency can lift alpha. This protects 1px outlines without
binarizing all external antialiasing.

## Candidate And Light Interaction Boundary

When local evidence cannot prove a unique answer, do not immediately ask the
user to paint a mask. First generate a small set of reasonable candidates:

```text
base known-B matte
  -> detect ambiguous regions
  -> propose finite interpretations
  -> execute local plans
  -> render candidate composites
  -> user selects one
```

Typical candidates:

| Ambiguity | Candidate A | Candidate B |
|---|---|---|
| interior region color equals B | `transparent_hole` | `same_color_marking` |
| multiple objects | `selected_object` | `all_objects` |
| hard/soft edge conflict | `hard_edge_snap` | `preserve_soft_alpha` |

Candidate count is variable. Local rules may return 0-2 candidates. A VLM
planner may choose more when evidence supports it, but more than 4 usually means
the ambiguity is not structured well enough.

Constraints:

1. Candidates must be composed of registered local tools.
2. Candidates should act on local EvidenceRegion (`RiskRegion` implementation)
   or user intent maps; the model can split/name/order/explain but cannot
   bypass evidence.
3. If all candidates are wrong, then escalate to keep/remove/hole brush or text
   intent. That input expresses intent, not final alpha.

Intent correction path:

```text
instruction / keep_mask / remove_mask / hole_mask / subject_mask
  -> OwnershipMap / BackgroundMap / HoleMap / RegionPolicyMap / RiskMap
  -> finite MattingPlan
  -> deterministic executor
```

Phase boundaries:

- **Phase 2**: known-B robust alpha fusion + ambiguity candidate generation.
- **Phase 3**: local RegionPolicyMap rules for hard edge / soft hair /
  intentional hole.
- **Phase 4**: router / QA can optionally call a visual model to interpret
  EvidenceRegion and provide region-policy priors.
- **Phase 5**: VLM/LLM emits only a finite plan schema and still never touches
  pixels directly.

## VLM Tool Planner

The architecture is not "large-model matting"; it is constrained planning:

```text
Local Analyzer
  -> EvidenceRegion[] / local measurements / ToolCatalog
VLM/LLM Planner
  -> CandidatePlan[] with variable length
Local Executor
  -> deterministic ERMBG tools
Validator / QA
  -> reject illegal plans, score candidates, produce debug overlays
```

The model can reason about:

- whether a region is subject, background, transparent hole, hard edge, soft
  edge, translucent material, or owned shadow;
- which registered local tools should run on each region;
- how many candidates are needed and how to rank/explain them.

The model must not:

- output final alpha;
- output final RGBA;
- write or select arbitrary unregistered image-processing code;
- bypass topology guards, QA, or parameter ranges.

The local analyzer measures:

- whether color is close to known B;
- whether alpha/keyer disagree;
- connected components, bbox, area, and external-background contact;
- evidence coalescing into fewer planner-friendly groups.

Coalescing is evidence compression, not semantic grouping. For example, 40
high-contrast edge fragments can become one evidence group while still meaning
only "high-contrast keyer/matting conflict", not "definitely hard edge".

### VLM Region Budget

Image complexity should control how many local evidence regions are generated,
but it should not directly control how many regions are sent to one VLM request.
The local analyzer may produce a large candidate pool for complex images; the
planner request still needs an explicit inference budget.

The intended policy is:

```text
image complexity -> candidate pool size
inference budget -> regions per VLM round
uncertainty      -> follow-up rounds or local splitting
```

Do not pass every detected region to the VLM just because the image is complex.
Large or low-information regions can slow the model down and make the semantic
answer worse. In particular, broad `alpha_keyer_disagreement` components should
be split locally, demoted, or handled by deterministic rules before they are
eligible for VLM review.

Default planning guidance:

- keep the full local `EvidenceRegion[]` pool for debugging and deterministic
  fallback;
- rank regions by expected impact on final alpha, semantic ambiguity, local
  confidence conflict, and area;
- send only the top high-value regions per VLM round, with a typical budget of
  6-8 and a hard cap near 12 unless an explicit batch mode is being tested;
- split oversized regions before VLM review when their bbox covers unrelated
  subject/background structure;
- use follow-up rounds for genuinely complex images instead of one very large
  prompt;
- keep `max_new_tokens` proportional to the expected JSON plan size, not to the
  source image size.

This keeps the VLM as a semantic planner rather than an expensive reviewer of
all local CV noise. Complex images should get better prioritization and, when
needed, more rounds; they should not automatically get unbounded prompt size.

## ToolCatalog v0

First version exposes only deterministic tools:

| Tool | Purpose | Base implementation |
|---|---|---|
| `preserve_hole` | keep an interior hole transparent | keep specified region low alpha |
| `fill_same_color_region` | interpret same-B interior as subject marking | local alpha lift, preserve input RGB |
| `repair_opaque_interior` | repair low-alpha holes in owned subject support | `repair_alpha_with_known_bg_key` / `repair_alpha_with_subject_support` |
| `snap_hard_edge` | repair logo/text/outline hard edges | `repair_hard_edge_alpha` |
| `preserve_soft_alpha` | protect hair/fur/smoke soft edge | prevent hard gate/fill on that region |
| `mark_translucent` | mark glass/thin fabric regions | preserve partial alpha, use unmix/foreground recovery |

Each tool needs a machine-readable contract:

```text
name / purpose / input parameters / parameter ranges
allowed_when / rejects_when / risks
```

## Plan Schema v0

Planner output is strict JSON with variable candidates:

```json
{
  "candidates": [
    {
      "id": "transparent_center",
      "label": "中心透明",
      "confidence": 0.78,
      "operations": [
        {"tool": "preserve_hole", "region_id": "r3"},
        {"tool": "snap_hard_edge", "region_id": "r1", "alpha_floor": 0.95}
      ],
      "reason": "主体像环形徽章,中心白色区域更可能是镂空"
    }
  ]
}
```

Before execution:

- `tool` must exist in `ToolCatalog`.
- `region_id` must come from local EvidenceRegion (`RiskRegion`) or a user
  intent map.
- parameters must be inside tool contract ranges.
- tool `allowed_when` must be supported by local evidence or user intent.

The first implementation target was local closed loop, not real VLM:

```text
EvidenceRegion[] + ToolCatalog
  -> rule/mock CandidatePlan[]
  -> PlanExecutor
  -> MatteCandidate[]
  -> QA/report/UI
```

After schema, validator, and executor are stable, real VLM plugs in only at the
`CandidatePlan[]` generation point.

## Landed Local Loop

Current landed pieces:

- [ermbg/planner.py](../ermbg/planner.py): `RiskRegion`, `evidence_kind`
  aliases, `ToolCatalog`, `PlannerPromptBundle`, `CandidatePlan`, plan
  validator, and rule planner.
- [ermbg/vlm_planner.py](../ermbg/vlm_planner.py): `PlannerClient` protocol,
  `RulePlannerClient`, `FixturePlannerClient`, and model JSON parser into
  `CandidatePlan[]`.
- [ermbg/vlm_payload.py](../ermbg/vlm_payload.py): VLM request packaging with
  original thumbnail, base matte preview, evidence overlay, region crops, and
  strict JSON schema. It does not call remote models.
- [ermbg/vlm_openai.py](../ermbg/vlm_openai.py): optional OpenAI Responses API
  client for explicit debug/provider paths. It still returns
  `CandidatePlan[]`.
- [ermbg/risk.py](../ermbg/risk.py): local evidence region extraction for
  same-B enclosed regions, alpha/keyer disagreement, and hard-edge candidates.
- [ermbg/executor.py](../ermbg/executor.py): independent `PlanExecutor` with
  operation metadata.
- [ermbg/candidates.py](../ermbg/candidates.py): same-color hole candidates now
  consume EvidenceRegion -> planner bundle -> planner client ->
  `CandidatePlan[]`.
- `/api/matte-candidates`: candidate payload exposes `plan`, `regions`, and
  `operation_results`.
- [scripts/06_risk_overlay.py](../scripts/06_risk_overlay.py): exports evidence
  overlay PNG, regions JSON, and planner bundle JSON.
- [scripts/07_vlm_planner_debug.py](../scripts/07_vlm_planner_debug.py):
  exports VLM request JSON, attachment PNGs, manifest, optional fixture/OpenAI
  response parsing, and executed candidate PNGs.

Sample-10 evidence snapshot:

```text
raw evidence counts:
  same_bg_enclosed_region = 1
  alpha_keyer_disagreement = 33
  hard_edge_candidate = 40

coalesced shown counts:
  same_bg_enclosed_region = 1
  alpha_keyer_disagreement = 18
  hard_edge_candidate = 1

outputs:
  out/risk_overlays/sample_10_risk_overlay_coalesced.png
  out/risk_overlays/sample_10_risk_overlay_coalesced.regions.json
  out/risk_overlays/sample_10_risk_overlay_coalesced.planner_bundle.json
```

Before real VLM use becomes default:

- define timeout, retry, empty-response, and invalid-JSON fallback behavior;
- keep all model output behind `parse_candidate_plans` +
  `validate_candidate_plans` + local executor;
- preserve fixture regression as the default test path so core tests do not
  depend on network or model drift.

## Summary

The same-color interior candidate work is now a replaceable planner loop:

```text
local CV evidence
  -> EvidenceRegion / RiskRegion
  -> PlannerPromptBundle + ToolCatalog
  -> RulePlannerClient / future VLM planner
  -> CandidatePlan[]
  -> PlanExecutor
  -> MatteCandidate[] + API/UI/debug overlay
```

Boundaries:

- Local CV measures pixels, topology, connected components, bbox, area,
  keyer/matting disagreement, and evidence coalescing; it does not interpret
  semantics.
- VLM/LLM only performs evidence-to-policy reasoning and candidate planning; it
  does not output alpha/RGBA/code.
- `kind` stays as the internal compatibility key; `evidence_kind` is exposed to
  planner/VLM.
- Candidate count is planner-decided, but every candidate must pass validator
  and the finite local tool catalog.
