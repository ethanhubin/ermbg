# ERMBG Route Strategy

ERMBG is now the route strategy layer, not a local Mac matting pipeline.
`backend="auto"` submits the image to the remote ComfyUI `ErmbgRouteMatte`
node. That node runs `ermbg.router.classify_route()` inside the Comfy process
and dispatches to one of the concrete maintained paths:

- clean RGBA input: passthrough
- deterministic hard UI/button on stable known background: `comfy-pymatting-known-b`
- green/blue icon, character, translucent button, or glass/complex button:
  `comfy-corridorkey`
- unknown or unstable background: PyMatting fallback through
  `comfy-pymatting-known-b` with the configured fallback background color

Route selection must produce the final execution profile and parameters before
any matting code runs. Execution code should consume `params.execution_profile`
and `params.corridorkey_execution_profile`; it must not re-infer "character",
"glass button", or "effect icon" from CorridorKey semantic analysis. This keeps
profile-specific tuning isolated.

The removed legacy path included local BiRefNet/GrabCut full matting,
`ErmbgAutoMatte`, `comfy-ermbg`, VLM-protected reruns, and the old CLIPSeg ->
AutoMatte subject-mask workflow. Those paths are no longer public API or Web
options.

## Execution Profiles

`RouteDecision.params.execution_profile` is the production contract between the
router and the selected backend.

| Execution profile | Backend | Asset kind | Intent |
|---|---|---|---|
| `corridorkey-character` | `comfy-corridorkey` | `character` | Composite 1024 character assets with hair, fur, translucent material, glow, and hard edges. Uses full-frame character control and disables color protection. |
| `corridorkey-transparent-button` | `comfy-corridorkey` | `button` | Glass or translucent buttons on known green/blue screen. Uses the glass full-frame control profile and disables color protection. |
| `corridorkey-effect-icon` | `comfy-corridorkey` | `icon` | Additive/soft-alpha effect icons where the whole effect layer should be solved by CorridorKey. |
| `corridorkey-shaped-icon` | `comfy-corridorkey` | `icon` | Icons with a shaped known-background hint, including key-color material cases that still need protection. |
| `pymatting-hard-button` | `comfy-pymatting-known-b` | `button` | Deterministic hard UI/buttons, including hard edges, stable known-B holes, and hard/soft shadow families routed away from CorridorKey. |
| `pymatting-known-bg` | `comfy-pymatting-known-b` | `known_bg_graphic` | Stable non-character/non-icon known-background graphic. |
| `pymatting-fallback` | `comfy-pymatting-known-b` | `unknown_fallback` | Unknown or unstable background fallback. Auto does not invoke RMBG. |

The CorridorKey semantic profile (`parameter_profile`, for example
`composite_character_corridor_only` or `translucent_button`) is analysis
metadata. It may help the router choose an execution profile, but after routing
the execution profile is the source of truth for hint mode, mask prior, color
protection, refiner, despeckle, and downstream metadata.

## ComfyUI Nodes

Available ERMBG nodes:

- `ErmbgRouteStrategy`: server-side route decision, returns backend, route,
  asset kind, and JSON metadata.
- `ErmbgRouteMatte`: production auto node. Runs route selection plus the
  selected PyMatting Known-B, CorridorKey, or passthrough path in the same Comfy
  process and returns foreground, alpha, RGB-for-RGBA, and JSON metadata. Auto
  no longer invokes RMBG fallback.
- `ErmbgPyMattingKnownB`: known-background PyMatting node used by the hard
  button path.
- `ErmbgClassify (preview)`: legacy lightweight classifier preview.
- `Convert Masks to Images`: utility conversion node.

`ErmbgAutoMatte` is not coming back; `ErmbgRouteMatte` is the replacement
contract for Web/API auto mode. `ErmbgRouteStrategy` remains useful for debug
and custom graph branching.

## PyMatting Known-B Ownership Contract

The `pymatting-hard-button`, `pymatting-known-bg`, and `pymatting-fallback`
profiles use a three-pass known-background contract. These responsibilities
must stay separate:

1. `fg_threshold` recalls hard subject anchors. It is dynamic, but its target is
   the background/subject separation valley plus a coherent seed guard. It must
   not be raised globally to remove edge residue, because B056-like silver,
   dark metal, or hard UI borders can have OKLab distance in the same range as
   weak screen-colored edge artifacts.
2. Edge repair is a local ownership-arbitration pass over a dynamic edge band,
   not a fixed 1px strip. It uses same-background reprojection to choose among
   smooth subject AA, tiny/thin screen-residue cleanup, disconnected alpha
   speck removal, and shadow support. This pass owns B002-B005/B055-like
   pinpricks, B033/B035/B038/B040-like bottom contact fringes, and B038-like
   curved-edge stair steps; `fg_threshold` does not.
3. ShadowPatch is a reconstruction patch. It may grow support to fill 1px
   contact seams, but grown support is accepted only when compositing the
   subject plus the patch back onto the same known background reprojects close
   to the source image. Shadow support is evidence, not a blanket foreground
   veto.

The practical invariant is:

```text
fg_threshold preserves hard subject structure
edge repair arbitrates subject AA, residue, specks, and shadow support
ShadowPatch fills source-proved shadow/contact gaps
```

Changing one pass to compensate for another usually reintroduces the known
failure pair: B002-B005/B055 edge dots on one side, or B056 metal edges turning
semi-transparent/yellow on the other.

### Failure Mechanism

Known-B PyMatting receives a trimap derived from measured distance to the known
background. That distance is a good way to find high-confidence subject seeds,
but it is not a complete ownership decision for every edge pixel. Several
different physical roles can have overlapping distance ranges:

- hard subject metal or dark outline near a blue/green screen;
- true subject antialiasing where the source pixel is a mixture of subject
  material and the known background;
- tiny screen-colored foreground fragments produced by the matting solve;
- single-pixel contact seams between subject and shadow;
- broad scalar known-background darkening that should become ShadowPatch.

The bug class came from asking one global threshold to solve all of those roles.
Raising `fg_threshold` can remove some green residue dots, but it also risks
dropping B056-like metal borders into semi-transparent/yellow foreground. Lowering
it protects hard borders, but leaves B002-B005/B055-like edge dots and
B038-like broken lower contact edges. The fix is to keep thresholding as a
subject-anchor recall step, then perform local edge ownership before shadow
repair.

### Pass 1: Dynamic Subject Anchors

`build_known_background_trimap()` computes an adaptive foreground threshold.
The threshold is allowed to move with the image, but it is guarded by coherent
seed evidence:

- candidate distances come from the observed background/subject separation;
- the selected threshold must leave enough foreground seed pixels;
- the largest seed component must be large enough to anchor the subject;
- fallback values remain bounded by the requested threshold and a raise cap.

This pass must answer only: "where are reliable subject anchors?" It must not
answer: "which individual edge specks are residue?" That distinction is why the
threshold debug reports `fg_threshold_source`, `fg_threshold_effective`,
`fg_threshold_seed_pixels`, and `fg_threshold_largest_seed_component`.

### Pass 2: Dynamic Edge Ownership

`_repair_known_b_edge_ownership()` owns the local edge arbitration. It estimates
a dynamic edge band from the matte transition and image scale. The band is
usually a few pixels on hard UI, but can expand for softer generated edges. It
is still capped so broad interiors are not re-solved as edge.

Inside that band, the pass runs the edge roles in one place so they cannot fight
each other:

- `_reconstruct_known_b_subject_edge_alpha()` repairs source-proved subject
  antialiasing. It borrows a nearby stable subject color, solves
  `source ~= alpha * foreground + (1 - alpha) * known_background`, and accepts
  only low-error pixels. Scalar known-B darkening is rejected here so it remains
  ShadowPatch ownership.
- `_stabilize_pymatting_subject_foreground_for_export()` extends stable subject
  RGB into soft colored UI edges where straight foreground division created
  saturated export colors. It does not create new shadow evidence.
- `_repair_screen_dominant_edge_residue_foreground()` removes tiny or thin
  screen-dominant edge fragments after source-proved subject AA has had a chance
  to win. Coherent green/blue subject material is protected by component shape
  and local support, not by a hard "green means background" rule.

The execution order matters. Subject AA reconstruction comes first because a
valid mixed edge pixel should become smooth subject ownership. Screen residue
cleanup comes after that because only the remaining screen-dominant fragments
are suspicious. ShadowPatch runs after both so it sees a cleaned subject layer
and can fill source-proved shadow/contact support.

Useful debug fields live under `debug.shadow.edge_ownership`:

- `subject_edge_reconstruction.edge_band_px`
- `subject_edge_reconstruction.candidate_pixels`
- `subject_edge_reconstruction.reconstructed_pixels`
- `subject_edge_reconstruction.raised_alpha_pixels`
- `subject_edge_reconstruction.lowered_alpha_pixels`
- `screen_residue_repair.repaired_pixels`
- `screen_residue_repair.alpha_lowered_pixels`
- `screen_residue_repair.thin_edge_component_count`

For a healthy hard-button batch, not every sample needs every counter to be
non-zero. A no-shadow, no-residue button may have only subject reconstruction.
A soft-shadow sample may have no screen-residue repair. The important invariant
is that counters describe a local, source-proved role, not a sample-id-specific
rule.

### Pass 3: Source-Proved ShadowPatch

`_pymatting_known_b_objective_shadow_from_source()` treats ShadowPatch as a
repair/reconstruction layer, not as another foreground classifier. It composites
the current subject over the known background, compares that prediction to the
source image, and solves only the residual explainable as scalar known-B
darkening.

ShadowPatch may grow support around a coherent shadow anchor. Growth is needed
because a one-pixel miss next to the subject can become a visible seam. However,
growth only creates support; written opacity still comes from source residuals.
The same-background reprojection rule is:

```text
subject + shadow composited on original known background ~= original source
```

If that equality fails, the pixel is not a valid patch. This prevents endpoint
specks and plain background-colored holes from being invented as shadow while
still letting source-proved contact gaps close.

Useful debug fields live under `debug.shadow.objective_shadow`:

- `seed_pixels`
- `kept_seed_pixels`
- `support_pixels`
- `raw_source_pixels`
- `completed_support_pixels`
- `completed_fill_pixels`
- `border_shadow_floor`
- `support_alpha_min`
- `components`
- `support_components`

### Tuning Rules

When changing this path, tune by mechanism:

- To recover a hard border, inspect adaptive foreground seed recall and subject
  edge reconstruction fit. Do not lower residue cleanup gates first.
- To remove colored dots, inspect `screen_residue_repair` inside the dynamic
  edge band. Do not raise `fg_threshold` globally unless seed recall itself is
  wrong.
- To close a seam, inspect ShadowPatch source residuals and completed support.
  Do not classify the seam as subject just because it is adjacent to subject.
- To protect real green/blue material, require component coherence and
  source-reprojection fit. Do not use channel color alone as ownership evidence.
- To add a threshold, document the observable signal it keys on: reconstruction
  error, seed component size, source residual floor, edge-band distance, or
  connected-component geometry.

Regression tests should be synthetic when the mechanism is simple, plus real
sample batch runs when the failure came from user-facing output. Do not encode
sample IDs, fixed coordinates, or one observed color as an execution rule.

### Validation Record

The current edge-ownership implementation was validated on June 1, 2026 with:

```bash
.venv/bin/pytest -q

.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend auto \
  --sample-id B001,B002,B003,B004,B005,B006,B007,B008,B009,B010,B016,B017,B018,B019,B020,B021,B022,B023,B024,B025,B031,B032,B033,B034,B035,B036,B037,B038,B039,B040,B050,B051,B052,B053,B054,B055,B056 \
  --out-dir out/remote_auto_pymatting_full_edge_ownership_20260601
```

Local deterministic batch:

```text
out/local_pymatting_full_edge_ownership_20260601/summary.json
37/37 ok
```

Remote production auto batch:

```text
out/remote_auto_pymatting_full_edge_ownership_20260601/summary.json
37/37 ok
```

Focused production Web smoke:

```text
POST /api/matte-candidates backend=auto B038
backend=comfy-pymatting-known-b
route=pymatting_known_b
execution_profile=pymatting-hard-button
edge_reconstructed=359
screen_residue_repaired=140
```

## Direct Worker Backend

`backend="direct-worker"` is an experimental Web/API and Game Eval backend for
testing the same route strategy without ComfyUI prompt execution or Comfy's
single queue. It runs on the remote GPU host through
`ermbg.direct_worker_server` and exposes HTTP endpoints on
`ERMBG_DIRECT_URL` or the default `http://192.168.0.8:7871`.

Direct Worker must preserve the same route/profile contract as
`ErmbgRouteMatte`:

- route selection still uses `ermbg.router.classify_route()`;
- `selected_backend` remains the logical production backend, for example
  `comfy-corridorkey` or `comfy-pymatting-known-b`;
- actual execution is reported separately as `direct-corridorkey` or
  `direct-pymatting-known-b`;
- `parameter_profile` and `execution_profile` must match the Comfy auto path
  for the same input.

The direct path is not a second implementation of CorridorKey behavior.
`ermbg.corridorkey_runner.LocalCorridorKeyClient` is the single maintained
in-process CorridorKey adapter. Both the Comfy custom node wrapper and the
Direct Worker call this runner so hint conversion, model invocation, color
protection, and debug metadata cannot drift. PyMatting Known-B already calls
the shared `_matte_image_pymatting_known_b()` implementation.

When changing CorridorKey execution details, update the shared runner instead
of patching `comfy_nodes/ermbg_nodes.py` or `ermbg/direct_worker.py`
separately. After the change, compare an `auto` batch and a `direct-worker`
batch on the same samples. Expected residual differences should be only
floating-point/8-bit rounding level; profile or hint-source differences are a
bug.

Useful commands:

```bash
# Start/restart the remote Direct Worker on the GPU host.
ssh ermbg-comfy 'cd /d C:\Users\darkv\ermbg_src && E:/ComfyUI/.venv/Scripts/python.exe -m ermbg.direct_worker_server --host 0.0.0.0 --port 7871 --cpu-workers 4'

# Focused parity check.
.venv/bin/python scripts/run_corridorkey_game_eval.py --backend auto --sample-id I003,I019,I008,B010 --out-dir out/auto_parity_<date>
.venv/bin/python scripts/run_corridorkey_game_eval.py --backend direct-worker --sample-id I003,I019,I008,B010 --out-dir out/direct_parity_<date>

# HTTP worker smoke.
.venv/bin/python scripts/smoke_direct_worker_http.py --base-url http://192.168.0.8:7871 --sample-id B001,I011
```

## Web Verification

After Web-facing route changes:

1. Restart the local Web server on `127.0.0.1:7860`.
2. Verify the index contains `Auto RouteMatte`, `direct-worker`,
   `comfy-pymatting-known-b`, and `comfy-corridorkey`.
3. Post real samples to `/api/matte-candidates` with `backend=auto` and confirm
   `requested_backend`, `backend`, `debug.auto_route.selected_backend`,
   `debug.auto_route.route`, `execution_profile`, and `server_elapsed_sec`.
4. For Direct Worker changes, also post a known CorridorKey sample with
   `backend=direct-worker` and confirm `requested_backend="direct-worker"`,
   `debug.direct_worker.execution_backend`, and `server_elapsed_sec`.

The standard smoke set is:

- hard button -> `comfy-pymatting-known-b` / `pymatting-hard-button`
- blue/green glass button -> `comfy-corridorkey` / `corridorkey-transparent-button`
- effect icon -> `comfy-corridorkey` / `corridorkey-effect-icon`
- shaped icon -> `comfy-corridorkey` / `corridorkey-shaped-icon`
- character -> `comfy-corridorkey` / `corridorkey-character`
- random/unknown background -> `comfy-pymatting-known-b` /
  `pymatting-fallback`
