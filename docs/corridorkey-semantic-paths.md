# CorridorKey Semantic Paths

## Current Status

Phase 1 sample construction is complete. The canonical full test set is:

```text
samples/corridorkey_semantic/manifest.json
samples/corridorkey_semantic/sheets/full_samples_v1_sheet.jpg
```

The approved set contains 83 samples:

| Class | Count | Focus |
|---|---:|---|
| Button | 54 | outlined/unoutlined/translucent button boundaries, hard/soft shadow strengths, white-outline buttons, real glass buttons |
| Icon / effect | 20 | hard boundary, soft boundary, translucent icons, particle effects, smooth glow |
| Character | 9 | 1024x1024 composite cases combining hair/fur, hard opaque edges, translucent material, and glow |

Screen convention:

- green: RGB(0, 200, 0)
- blue: RGB(0, 0, 200)

Each case has one approved screen input. Evaluation code should count selected
samples from the manifest instead of multiplying every case by every possible
background.

## Baseline: 2026-05-31

The active B016-B030 blue-screen button block now uses green subject buttons on
blue, not yellow buttons on blue. This keeps the sample set aligned with the
intended screen-role split: yellow/orange buttons are already solvable on green
screen, while blue-screen coverage should focus on green subject material that
green screen cannot separate cleanly.

Changed sample block:

| Samples | Screen | Subject | Focus |
|---|---|---|---|
| B016-B020 | blue | green outlined buttons | opaque hard UI with outline/no-shadow/hard/soft shadow |
| B021-B025 | blue | green unoutlined buttons | opaque hard UI without outline/no-shadow/hard/soft shadow |
| B026-B030 | blue | translucent green buttons | translucent UI material on blue screen |

The previous `button_blue_yellow_*` records were removed from the active sample
tree and manifest. Do not use them as regression targets; they measured a
CorridorKey decomposition weakness for yellow-on-blue rather than the intended
green-subject blue-screen pressure.

Latest full baseline:

```text
out/corridorkey_full_blue_green_baseline_20260531/summary.json
```

Result: 83/83 completed successfully with `backend=auto`.

Profile distribution in that run:

| Profile | Count |
|---|---:|
| edge_cleanup | 23 |
| opaque_hard_ui_soft_shadow | 16 |
| opaque_hard_ui_hard_shadow | 13 |
| screen_tinted_translucency | 9 |
| translucent_button | 7 |
| key_color_material | 7 |
| opaque_hard_ui_no_shadow | 5 |
| balanced | 3 |

B016-B030 all ran successfully. The current low-coverage list is dominated by
known hard cases: green translucent buttons B012-B015, real glass buttons,
white-outline soft shadow B053, weak-contrast icon I003, same-screen
translucency I010, and soft/additive glow icons I012/I013/I016/I018-I020. The
coverage metric uses `alpha > 128`, so treat it as a triage signal rather than
ground truth for translucent/glow samples.

## Phase Plan

### Phase 1: Sample Coverage

Status: complete.

The goal was to build a realistic game-asset sample set before tuning route
recognition. The final set intentionally covers boundary and alpha mechanisms
through real-looking assets rather than abstract placeholders.

Accepted scope:

- Buttons are controlled UI geometry: borders/no borders, shadow strength,
  translucent material, and model-generated real glass.
- Icons and effects are one class because their key matting pressure is complex
  boundary plus internal color/soft alpha.
- Characters are composite samples because hair, semi-transparent material, and
  hard edges naturally appear together.

### Phase 2: Recognition And Route Audit

Status: in progress.

The next step is to run the full confirmed sample set through the Web/Game Eval
path and inspect failures by family. This phase should answer: which route or
candidate set should each sample use?

Inputs:

- `samples/corridorkey_semantic/manifest.json`
- Web eval page: `http://127.0.0.1:7860/eval/game`
- CLI eval:

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend comfy-corridorkey
```

Audit outputs should be written under `out/` and include a summary JSON. Do not
use parameter tuning to hide a wrong route decision. Missing or low-confidence
routes should be recorded as recognizer gaps.

### Phase 3: Path-Specific Parameter Tuning

Status: deferred until Phase 2 is usable.

Only after route selection is accurate enough should CorridorKey parameters be
tuned per route: color protection, despill, refiner strength, despeckle,
foreground recovery, and shadow/soft-layer handling.

## Route Pressures To Audit

| Pressure | Representative sample families |
|---|---|
| hard opaque UI boundary | button A/B no-shadow and hard-shadow rows |
| owned contact shadow | button hard/soft shadow rows |
| translucent UI glass | button C rows and real glass buttons |
| hard icon boundary | icon A group |
| soft/fragmented icon boundary | icon B group |
| translucent icon material | icon C group |
| additive or soft-alpha effect | icon D group |
| mixed character boundary | character composite group |

## Inactive Historical Sets

The old `samples/vlm_eval` and `samples/vlm_eval_game` directories have been
removed. Historical analysis in `docs/archive/` may still mention G-style sample
IDs, but active development and Web/Game Eval should use the B/I/C sample IDs in
`samples/corridorkey_semantic/manifest.json`.
