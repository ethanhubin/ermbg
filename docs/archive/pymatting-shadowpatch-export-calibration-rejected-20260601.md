# PyMatting ShadowPatch Export Calibration Rejected Experiment - 2026-06-01

> Archived rejected experiment. This document records a failed exploration
> around PyMatting Known-B hard-button edge cleanup and ShadowPatch export.
> It is historical context only; do not treat it as the active implementation
> plan.

## Context

The investigation started from hard-button PyMatting failures:

- B002-B005 showed small dark-green edge residue.
- B038 showed endpoint shadow specks and jagged subject/shadow edges.
- B055 showed discrete edge dots and a rough shadow/subject boundary.
- B056 showed a different failure class: hard metal/blue-background subject
  edges were partially treated as transparent.

The working quality standard was split into two views:

1. Composite the exported RGBA back over the measured original background.
   If the decomposition is correct, this reprojection should be close to the
   source image.
2. Composite over a complementary background and inspect continuity against
   the source, especially edge smoothness and detached alpha/RGB specks.

Fixed-scale diagnostic maps were used for comparison. Auto-normalized maps are
useful for finding pixels, but misleading for judging severity.

## What Was Tried

Several related ideas were explored:

- Dynamic `fg_threshold` was kept as the broad direction, but it was not enough
  to resolve the conflict between removing near-background edge dots and
  preserving low-contrast hard subject edges.
- A hard-button visible screen-edge residue pass was prototyped to borrow
  nearby non-screen foreground for tiny export-visible green dots.
- ShadowPatch was changed to fold Objective ShadowPatch through an sRGB
  display-space equation instead of a linear-light black layer. This improved
  source reprojection for some shadow cases.
- A source-fidelity guard was added around near-edge shadow regularization.
  It rejected most proposed B055 smoothing because those pixels helped match
  the original green-screen source.
- A final Known-B export calibration pass was prototyped. It tried to solve
  final straight RGBA so that `RGBA over measured B ~= source`.
- Additional protections were added after regressions:
  - preserve high-confidence colored subject antialias RGB;
  - do not convert shadow-only black-alpha pixels into screen-colored RGB;
  - propagate `execution_profile=pymatting-hard-button` into PyMatting execution.

The experimental patch was saved at:

```text
out/rejected_experiment_20260601/rejected_shadow_export_experiment.patch
```

## Results

The experiment produced mixed numeric results. A local same-command comparison
against `HEAD` showed:

| Sample | Official source p99 / gt5 / continuity p99 | Experiment source p99 / gt5 / continuity p99 | Result |
|---|---:|---:|---|
| B002 | 1.137 / 156 / 0.0166 | 0.854 / 8 / 0.0118 | Better numerically |
| B005 | 6.872 / 565 / 0.0268 | 0.570 / 137 / 0.0132 | Better numerically |
| B038 | 1.381 / 154 / 0.0048 | 1.129 / 6 / 0.0049 | Mostly better |
| B055 | 1.142 / 1057 / 0.0281 | 1.141 / 1260 / 0.0393 | Worse visually/continuity |
| B056 | 1.310 / 1423 / 0.0555 | 1.388 / 1485 / 0.0550 | Worse source fidelity |

Ethan compared the output visually and judged the overall result worse than the
last official version. The experiment was therefore rejected and reverted.

## Why It Failed

The key mistake was treating final source reprojection as a license to change
the exported RGBA representation. That can improve fixed original-background
error while making alternate-background appearance worse.

Two specific failure modes appeared:

- Shadow-only pixels should transfer as black alpha. If an export calibration
  pass solves their RGB directly from the original green/blue background, it can
  create screen-colored or grey-green specks on complementary backgrounds.
- Colored subject antialias pixels can legitimately have saturated foreground
  RGB even when their original-background mixed source pixel is less saturated.
  Pulling those RGB values toward the source composite weakens subject material.

The broader lesson: original-background reprojection is a necessary invariant,
not a sufficient objective. It must be paired with representation constraints:

- subject RGB is owned by subject reconstruction;
- shadow is represented as black/neutral alpha, not colored RGB;
- edge cleanup should not overwrite material evidence;
- ShadowPatch continuity should be fixed in shadow support/field construction,
  not by final RGB calibration.

## Final State

The experiment was reverted from:

- `ermbg/api.py`
- `tests/test_api.py`
- `comfy_nodes/ermbg_nodes.py`

The remote ComfyUI source and custom node wrapper were synchronized back to the
official code and restarted. Verification after revert:

```text
tests/test_api.py + tests/test_comfy_nodes.py: 67 passed
out/auto_b055_restored_official_20260601
```

The restored B055 smoke no longer contained the experimental export calibration
metadata, confirming the production path was back on the official version.

## Follow-Up Direction

Future work should not resume the final export calibration approach as-is.
The next promising path is narrower:

- keep the official RGBA representation;
- inspect B055/B038 ShadowPatch support continuity directly;
- repair detached or jagged shadow support as shadow support, not as RGB;
- keep B056 as a separate hard metal/blue-background subject recall problem.

