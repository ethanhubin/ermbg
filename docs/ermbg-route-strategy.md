# ERMBG Route Strategy

This document defines the route/profile/backend contract used by Web, API,
Direct Worker, Game Eval, and optional adapters.

Web `backend="auto"` submits images to the configured Direct Worker service.
The worker runs `ermbg.router.classify_route()`, selects an execution profile,
and dispatches to the maintained matting path. Optional ComfyUI nodes use the
same route contract for custom graphs.

Route selection must finish before matting starts. Execution code consumes
`RouteDecision.params.execution_profile` and related parameters directly.
Profile-specific tuning belongs in the shared route/execution code, so Web,
Direct Worker, and optional adapters stay aligned.

## Execution Profiles

`execution_profile` is the public contract between image analysis and matting.
`parameter_profile` is analysis metadata that explains why the route was chosen.

| Execution profile | Execution path | Asset kind | Intent |
|---|---|---|---|
| `corridorkey-character` | CorridorKey | `character` | Character assets with hair, fur, glow, translucent material, and hard edges. |
| `corridorkey-transparent-button` | CorridorKey | `button` | Glass or translucent buttons on green/blue screen. |
| `corridorkey-effect-icon` | CorridorKey | `icon` | Additive or soft-alpha effect icons solved as one effect layer. |
| `corridorkey-shaped-icon` | CorridorKey | `icon` | Icons with shaped hints and key-color material protection. |
| `pymatting-hard-button` | PyMatting Known-B | `button` | Hard UI/buttons with stable solid-color background evidence. |
| `pymatting-known-bg` | PyMatting Known-B | `known_bg_graphic` | Stable known-background graphic outside the button/icon/character classes. |
| `pymatting-fallback` | PyMatting fallback | `unknown_fallback` | Unknown or unstable background fallback. |

Clean RGBA inputs may use passthrough when alpha quality is already usable.

## Route Responsibilities

The router identifies:

- existing alpha quality and passthrough eligibility;
- background color and stability;
- solid green/blue screen evidence;
- hard UI, icon, effect, character, glass, and translucent-material signals;
- final execution profile and backend parameters;
- fallback background color for unstable or unknown backgrounds.

Execution code must preserve these decisions. It should not re-classify an asset
kind from local semantic hints after the router has selected the execution
profile.

## Known-B Path

The PyMatting Known-B path targets game UI and stable solid-background graphics.
Its job is pixel-level repair on top of known background evidence:

- dynamic subject anchors from measured background/subject separation;
- edge ownership repair for hard edges, antialiasing, pinholes, thin residue,
  holes, and contact regions;
- ShadowPatch reconstruction for source-proved scalar darkening on the known
  background;
- foreground RGB stabilization for export.

Ownership decisions use measurable evidence such as color distance, connected
components, local support, and same-background reprojection error. See
`docs/local-ownership.md` for the detailed evidence model.

## CorridorKey Path

CorridorKey handles complex green/blue-screen assets that benefit from
film-style keying practice:

- characters with hair, fur, glow, and soft alpha;
- translucent/glass buttons;
- shaped icons with key-color material protection;
- effect icons with additive or smoke-like soft edges.

`ermbg.corridorkey_runner.LocalCorridorKeyClient` is the shared in-process
adapter. Direct Worker and optional Comfy nodes call the same runner so hint
conversion, color protection, model invocation, and debug metadata stay aligned.

## Direct Worker Backend

`backend="direct-worker"` is the Web/API and Game Eval service backend. The
service URL comes from `services.direct_worker_url` in `ermbg.config.json`, with
`ERMBG_DIRECT_URL` available as an environment override.

Direct Worker reports two layers of backend metadata:

- `selected_backend`: logical route backend selected by the router;
- `debug.direct_worker.execution_backend`: concrete direct execution path, such
  as `direct-corridorkey` or `direct-pymatting-known-b`.

For the same input, Web auto and Direct Worker runs should keep
`parameter_profile` and `execution_profile` stable. Expected output differences
between adapters should stay at floating-point or 8-bit rounding level.

## Optional Comfy Nodes

ComfyUI support lives in `comfy_nodes/` for custom Comfy graphs. It uses the
same route/profile contract and is configured through `services.comfy_url` or
`COMFY_URL`. Web default configuration uses Direct Worker.

Node details and install steps are documented in `comfy_nodes/README.md` and
`DEPLOY.md`.

## Verification

Use focused samples that cover each profile:

- hard button -> `pymatting-hard-button`
- blue/green glass button -> `corridorkey-transparent-button`
- effect icon -> `corridorkey-effect-icon`
- shaped icon -> `corridorkey-shaped-icon`
- character -> `corridorkey-character`
- random/unknown background -> `pymatting-fallback`

Useful commands:

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py --backend auto --sample-id I003,I019,I008,B010 --out-dir out/auto_parity_<date>
.venv/bin/python scripts/run_corridorkey_game_eval.py --backend direct-worker --sample-id I003,I019,I008,B010 --out-dir out/direct_parity_<date>
.venv/bin/python scripts/smoke_direct_worker_http.py --base-url <services.direct_worker_url> --sample-id B001,I011
```

After Web-facing route changes:

1. Restart the local Web service.
2. Verify the index contains `Auto Direct Worker` and `direct-worker`.
3. Post real samples to `/api/matte-candidates` with `backend=auto`.
4. Confirm `requested_backend`, `backend`, `debug.auto_route.selected_backend`,
   `debug.auto_route.route`, `execution_profile`, and `server_elapsed_sec`.
5. For Direct Worker changes, also post a CorridorKey sample with
   `backend=direct-worker` and confirm
   `debug.direct_worker.execution_backend`.

Algorithm changes should be mechanism-driven. Tests should capture the failure
class with synthetic coverage when possible, plus real sample batch coverage for
user-facing regressions.
