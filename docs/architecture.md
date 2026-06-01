# ERMBG Architecture

ERMBG should be treated as a shared matting core with multiple adapters and
runtime backends. The production default is still local Web/API orchestration
calling the remote ComfyUI `ErmbgRouteMatte` node, but the system is no longer
just a Web app plus one Comfy workflow.

## Current Shape

```text
Entry points
  Web UI / Web API
  CLI / Python API
  ComfyUI custom nodes
        |
        v
ERMBG core contract
  input normalization
  route strategy
  parameter_profile
  execution_profile
  shared matting / shadow / metadata code
        |
        v
Runtime backends
  remote ComfyUI ErmbgRouteMatte
  remote Direct Worker
  local lightweight Python/CV paths
```

The important boundary is that entry points express user intent and pass image
data. They should not independently decide whether an asset is a character,
button, icon, transparent material, or fallback case. That decision belongs to
the shared route/profile contract.

## Layers

### Core

The core is the code under `ermbg/` that defines the production behavior:

- `router.py`: route decision, asset kind, `parameter_profile`, and
  `execution_profile`.
- `api.py`: high-level `matte_image()` contract and maintained local execution
  helpers, including PyMatting Known-B.
- `corridorkey_runner.py`: shared in-process CorridorKey adapter used by the
  Comfy node wrapper and Direct Worker.
- `shadow.py`, `ownership.py`, `known_bg_hard_ui.py`, `pymatting_refine.py`,
  and related modules: reusable matting mechanisms.

The core owns output semantics: foreground RGB, alpha, RGBA RGB layer, route
metadata, debug metadata, and timing metadata.

### Adapters

Adapters translate an external caller into the core contract:

- `ermbg.web`: local FastAPI Web UI, Web API, and Game Eval launcher.
- CLI/Python API: direct local integration for scripts and tests.
- `comfy_nodes/ermbg_nodes.py`: ComfyUI custom node wrapper.
- `integrations/openclaw`: optional independent OpenClaw `ermbg-matte` skill
  integration. This is not part of the main production loop.

Adapters should stay thin. They may expose UI controls and choose a requested
backend, but they should not fork route logic or duplicate execution-profile
tuning.

### Runtimes

Runtime backends decide where execution happens:

- **Comfy runtime**: production default for `backend=auto`. Local code uploads
  the image to remote ComfyUI, submits the single-node `ErmbgRouteMatte`
  workflow, polls history, and downloads foreground/alpha/metadata.
- **Direct Worker runtime**: remote HTTP worker that bypasses the Comfy prompt
  queue. It is the speed/parity validation path and must share router/profile
  and execution code with the Comfy path.
- **Local runtime**: lightweight deterministic work such as PyMatting,
  OpenCV/numpy utilities, route debugging, diagnostics, and tests. Mac local
  execution must not load SDXL/FLUX/Qwen-scale generation models.

## Current Production Contract

`backend=auto` means:

```text
local Web/API/CLI
  -> remote ComfyUI workflow
  -> ErmbgRouteMatte
  -> classify_route()
  -> passthrough / PyMatting Known-B / CorridorKey / PyMatting fallback
  -> foreground + alpha + rgba_rgb + metadata
```

The Comfy node executes route analysis, parameter selection, selected matting
path, ShadowPatch, and metadata generation inside the Comfy process. The local
Mac side is orchestration and display.

`backend=direct-worker` means:

```text
local Web/API/eval client
  -> remote ermbg.direct_worker_server
  -> shared route/profile contract
  -> direct-corridorkey or direct-pymatting-known-b execution
```

Direct Worker is not a separate algorithm family. When it differs from
`backend=auto`, first compare route metadata, execution profile, shared runner
debug fields, and alpha/RGBA diffs before tuning thresholds.

## ComfyUI Node Contract

The maintained Comfy node surface is:

- `ERMBG Route Matte`: production auto route and matte node.
- `ERMBG Route Strategy`: route-only debug/branching node.
- `ERMBG PyMatting Known-B`: deterministic known-background node for hard UI
  and stable known-background graphics.
- `ERMBG Classify (preview)`: legacy lightweight diagnostic node.
- `Convert Masks to Images`: utility node.

Custom Comfy graphs may use `ERMBG Route Strategy` for branching, but Web/API
production should submit `ERMBG Route Matte` for `backend=auto`.

## Optional OpenClaw Adapter

OpenClaw integration is kept as an optional independent `ermbg-matte` skill,
not a mainline runtime:

```bash
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
  --image /path/to/input.png
```

This path should submit the same remote `ErmbgRouteMatte` workflow used by
Web/API auto mode and archive `output.png`, `workflow.json`, `manifest.json`,
and Comfy history metadata. It should not duplicate ERMBG route logic inside
the skill. It should not be described or routed as a RMBG/rembg sub-mode.

Do not expand OpenClaw-specific behavior ahead of the main Web/API/Comfy and
Direct Worker paths. If OpenClaw needs more features later, add them as a thin
adapter over the same route/matte contract.

## Recommended Direction

1. Keep one source of truth for route/profile decisions.
   `router.py` should remain the only place that decides asset family and final
   `execution_profile`.

2. Keep adapters thin.
   Web, Comfy nodes, CLI, and optional adapters should pass intent and data
   into the shared contract instead of branching on asset classes themselves.

3. Promote Direct Worker from experiment to first-class runtime only after
   parity is proven.
   The useful distinction is not algorithmic behavior; it is scheduling,
   throughput, and service shape.

4. Add runtime capability checks.
   Local Web startup and smoke tests should be able to report remote ERMBG
   version, Comfy node availability, Direct Worker availability, and active
   capability flags.

   Current Web API entry:

   ```bash
   curl -sS "http://127.0.0.1:7860/api/runtime-capabilities?include_object_info=false"
   ```

5. Standardize artifacts.
   Every adapter should produce a browsable batch or run directory with a
   predictable manifest, output image, alpha, foreground, route metadata, and
   runtime timing.

   Current schema: `ermbg.run.v1`. Python API/CLI `output_dir` runs, Web
   `/api/matte-candidates` runs, and Game Eval case directories write
   `manifest.json` without replacing their existing `summary.json` /
   `*.report.json` compatibility files.
   Web exposes these through `GET /api/artifacts` and
   `GET /api/artifacts/<artifact_id>`.

6. Treat ComfyUI as the model and graph host, not as the only service boundary.
   Comfy remains best for custom graphs and GPU model ecosystem work. Direct
   Worker is the better long-term shape for high-throughput API service once
   its parity with `ErmbgRouteMatte` is maintained.

## Anti-Patterns

- Duplicating route heuristics in Web JavaScript, optional adapter code, Comfy
  wrapper code, or Direct Worker glue.
- Adding a new backend that implies a new profile contract.
- Fixing a sample by branching on sample IDs, filenames, one-off dimensions, or
  fixed coordinates.
- Running heavy generation or VLM models locally on the Mac.
- Treating a local source change as deployed before the remote Comfy node or
  Direct Worker has been synced and smoke-tested.
