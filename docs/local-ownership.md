# ERMBG Current Direction: Local Ownership

Date: 2026-05-27

This is the current engineering source of truth for the known-background
matting direction. Older model-planning notes are archived under `docs/archive/`
and should be treated as historical context only.

## Current Decision

ERMBG should default to a local, deterministic ownership planner for
known-background assets. The active engineering shape is:

```text
known-background image
  -> local matte
  -> local evidence regions
  -> local multi-hypothesis ownership scoring
  -> global execution-mask arbitration
  -> protected matte only when a soft subject layer needs protection
```

The key reframing is that shadows and translucent material are both mixtures
with the known background:

```text
C = alpha * F + (1 - alpha) * B
```

The important question is not only "what alpha is this pixel?" It is "which
operation owns this region?"

## Ownership Roles

The current local planner ranks each evidence region into these roles:

| Role | Meaning | Execution intent |
|---|---|---|
| `hole` | region is background / transparent opening | keep alpha low |
| `opaque_subject` | hard subject support was missed | allow guarded alpha repair |
| `subject_soft_layer` | glass, glow, smoke, soft antialiasing, translucent material | preserve soft alpha and protect from destructive keyer/repair |
| `shadow_like_layer` | scalar darkening of known background near subject support | preserve measured shadow path |
| `conservative_unknown` | local evidence is ambiguous | keep current alpha; avoid hard repair |

Implementation:

- [ermbg/ownership.py](../ermbg/ownership.py) measures local signals and ranks
  roles.
- [scripts/10_local_ownership_batch.py](../scripts/10_local_ownership_batch.py)
  runs the local batch flow and writes browsable summaries.
- [ermbg/matting.py](../ermbg/matting.py) accepts `subject_material_mask` as an
  execution constraint and restores protected soft-layer alpha after keyer
  repair/gating.
- [ermbg/web.py](../ermbg/web.py) can browse `local_ownership_*` eval
  batches in the game eval UI.

## Local Signals

The planner uses measurable signals, not semantic labels:

- alpha distribution: low/mid/high alpha fractions;
- color distance from known background in OKLab;
- local saturation / chroma shift;
- scalar darkening fit: `C_linear ~= scale * B_linear`;
- scalar darkening strength: `1 - scale`;
- topology: near subject support, exterior fraction, border touch;
- evidence kind from existing risk/debug region extractors.

These are intentionally general. They do not encode sample IDs, filenames, or
one-off coordinates.

## Execution Arbitration

Local scoring is permissive because it should expose weak evidence. Execution
masks need stricter global arbitration:

- tiny `subject_soft_layer` speckles are dropped before they can trigger
  material protection;
- when a coherent `subject_soft_layer` dominates, small scalar-looking
  fragments are suppressed as shadow candidates;
- shadow-only samples keep the base matte path and do not run the protected
  material rerender;
- protected rerender is used only when `subject_soft_layer` exists.

This prevents the two important cross-failures:

- G02-style soft shadow being misread as translucent material;
- G04/G06-style glass/glow being reopened as shadow.

The current empirical gates live near `resolve_execution_masks()` in
[ermbg/ownership.py](../ermbg/ownership.py), with comments describing the signal
and failure mode they protect against.

## Current Eval Target

The active sample target is the full confirmed CorridorKey semantic set:

```text
samples/corridorkey_semantic/manifest.json
```

The previous G02/G04/G06 loop has been superseded by the B/I/C sample IDs in the
new manifest. Use targeted subsets for fast local ownership checks only when a
mechanism needs focused debugging:

```text
B004/B005/B019/B020: shadow-like button layers.
B011/B026/B041/B046: translucent/glass button layers.
I010/I019/C004/C009: soft-alpha icon/effect and composite character layers.
```

The next full step is not more sample generation; it is Phase 2 recognition and
route audit over the approved 83 samples.

## How To Reproduce

```bash
.venv/bin/pytest -q tests/test_ownership.py tests/test_shadow.py tests/test_risk.py

.venv/bin/python scripts/10_local_ownership_batch.py \
  --out-dir out/local_ownership_corridorkey_semantic_probe \
  --sample-id B004,B011,I019,C004
```

Run the web UI:

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860
```

## What Is Still Open

The local ownership direction is good enough for the current ownership decision
target. The next real gap is not semantic classification.

Open work:

- improve W-group foreground/color recovery so protected soft layers do not
  become pale or over-white;
- add ambiguity candidates for large/high-impact soft subject regions where
  local evidence supports more than one plausible execution policy. G06-G shows
  the current failure mode: the center purple subject content is protected as a
  `subject_soft_layer`, but its alpha remains too low, so color is lost on
  recomposition. When local signals cannot safely distinguish "preserve
  translucent layer" from "repair underestimated subject material", generate
  both deterministic candidates:
  - `preserve_translucent_layer`: keep the current soft alpha and protect it
    from destructive keyer/repair changes;
  - `repair_underestimated_subject_material`: raise alpha only in coherent
    subject-owned interior material using local known-background evidence,
    topology, and a feathered distance-field transition.
  The base/protected matte should remain fast and local; candidate generation
  should be optional, browsable in eval/UI, and testable without network calls.
- turn the debug batch flow into a cleaner first-class eval command if this
  route becomes the default workflow;
- run and inspect the full confirmed sample set once Phase 2 route recognition
  instrumentation is in place.

Archived docs remain useful for historical reasoning, but they are no longer
the active plan.
