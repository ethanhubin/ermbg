# CorridorKey Semantic Paths

## Current Status

Phase 1 sample construction is complete. The canonical full test set is:

```text
samples/corridorkey_semantic/manifest.json
samples/corridorkey_semantic/sheets/full_samples_v1_sheet.jpg
```

The approved set contains 85 samples:

| Class | Count | Focus |
|---|---:|---|
| Button | 56 | outlined/unoutlined/translucent button boundaries, hard/soft shadow strengths, white-outline buttons, real glass buttons, known-B hole regressions |
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

Only the green-subject blue-screen block is active for B016-B030. Do not add
alternate blue-screen color studies to the manifest unless they represent a new
current failure class.

Latest full RouteMatte baseline:

```text
out/auto_routematte_routefix_20260531/summary.json
out/auto_routematte_routefix_20260531/timing_report.md
```

Result: 85/85 completed successfully with Web/API `backend=auto`, which submits
the remote `ErmbgRouteMatte` node. Auto no longer invokes RMBG fallback; unknown
or unstable backgrounds route to `comfy-pymatting-known-b` as
`pymatting_fallback`.

Route distribution in the latest full B/I/C run:

| Route | Backend | Count |
|---|---|---:|
| `pymatting_known_b` | `comfy-pymatting-known-b` | 37 |
| `corridorkey` | `comfy-corridorkey` | 48 |

Category/backend split:

| Category | `comfy-pymatting-known-b` | `comfy-corridorkey` |
|---|---:|---:|
| Button | 37 | 19 |
| Icon / effect | 0 | 20 |
| Character | 0 | 9 |

Latest execution-profile verification:

```text
out/verify_route_profiles_character_glass_icon_20260531/summary.json
```

This targeted run verifies that the router now selects a final
`execution_profile` before execution, so CorridorKey parameter choices do not
bleed between glass buttons, characters, and icons.

| Sample family | Execution profile | Backend |
|---|---|---|
| B046-B049 real glass buttons | `corridorkey-transparent-button` | `comfy-corridorkey` |
| I011-I012 soft/effect icons | `corridorkey-effect-icon` | `comfy-corridorkey` |
| I019-I020 shaped glow icons | `corridorkey-shaped-icon` | `comfy-corridorkey` |
| C001-C009 characters | `corridorkey-character` | `comfy-corridorkey` |

Profile rules:

- `corridorkey-transparent-button`: full-frame glass control, color protection
  forced off, profile-specific mask prior.
- `corridorkey-character`: full-frame character control, color protection
  forced off, character-specific mask prior.
- `corridorkey-effect-icon`: full-frame effect control for additive/soft-alpha
  icon material.
- `corridorkey-shaped-icon`: shaped foreground hint, including key-color
  material icons where protection may remain enabled.
- `pymatting-hard-button`: deterministic known-B hard UI/button route.

Timing on the 2026-05-31 full run:

| Scope | Count | Avg | Median | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Overall client elapsed | 85 | 1.073s | 0.936s | 3.961s | 0.190s | 5.172s |
| `comfy-pymatting-known-b` client elapsed | 37 | 0.355s | 0.266s | 0.846s | 0.190s | 1.085s |
| `comfy-corridorkey` client elapsed | 48 | 1.626s | 1.025s | 4.569s | 0.915s | 5.172s |
| Button client elapsed | 56 | 0.585s | 0.433s | 1.196s | 0.190s | 1.355s |
| Icon client elapsed | 20 | 1.077s | 1.021s | 1.405s | 0.989s | 1.432s |
| Character client elapsed | 9 | 4.097s | 3.985s | 4.949s | 3.173s | 5.172s |

The route-fix run moves hard soft-shadow buttons B020/B025, white-outline
contact-shadow buttons B053/B054, and known-B hole buttons B055/B056 back to
PyMatting. Slowest cases remain character CorridorKey runs, led by C005 at
5.172s client / 3.048s node. Lowest `alpha > 128` coverage is I010, then B056,
I020, I019, and several heavy-shadow/soft-shadow button cases. B056 is now
correctly routed by policy but needs PyMatting-path quality tuning. The coverage
metric is still a triage signal rather than ground truth for translucent/glow
samples.

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

Status: route profile contract established; continue auditing sample families
for recognizer gaps and path-specific quality.

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
tuned per execution profile: color protection, despill, refiner strength,
despeckle, foreground recovery, and shadow/soft-layer handling. Do not tune
generic CorridorKey settings in a way that makes `corridorkey-character`,
`corridorkey-transparent-button`, `corridorkey-effect-icon`, or
`corridorkey-shaped-icon` affect each other.

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

## Active Set Boundary

Historical generated sets are not active inputs. Historical analysis in
`docs/archive/` may still mention retired sample IDs, but active development and
Web/Game Eval should use the B/I/C sample IDs in
`samples/corridorkey_semantic/manifest.json`.
