# ERMBG Architecture

ERMBG is a shared matting core with multiple adapters and runtime backends. The
production default is local Web/API orchestration calling the ERMBG Direct
Worker service. ComfyUI provides optional custom nodes for Comfy graphs.

## Shape

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
  Direct Worker HTTP service
  local lightweight Python/CV paths
  optional ComfyUI custom nodes
```

Entry points express user intent and pass image data. Asset classification
belongs to the shared route/profile contract.

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
- `comfy_nodes/ermbg_nodes.py`: optional ComfyUI custom node wrapper.
- `integrations/openclaw`: optional independent OpenClaw `ermbg-matte` skill
  integration.

Adapters stay thin. They expose UI controls, choose a requested backend, and
pass the request into shared route logic.

### Runtimes

Runtime backends decide where execution happens:

- **Direct Worker runtime**: default Web/API runtime for `backend=auto`. It is
  an HTTP worker around the shared router/profile and execution code.
- **Local runtime**: lightweight deterministic work such as PyMatting,
  OpenCV/numpy utilities, route debugging, diagnostics, and tests.
- **Comfy runtime**: optional graph/node adapter over the shared route/profile
  and execution code.

## Production Contract

Web `backend=auto` means:

```text
local Web/API/CLI
  -> ermbg.direct_worker_server
  -> classify_route()
  -> passthrough / PyMatting Known-B / CorridorKey / PyMatting fallback
  -> foreground + alpha + rgba_rgb + metadata
```

`backend=direct-worker` means the same service is requested explicitly:

```text
local Web/API/eval client
  -> ermbg.direct_worker_server
  -> shared route/profile contract
  -> direct-corridorkey or direct-pymatting-known-b execution
```

Direct Worker consumes the shared route metadata and execution profile.

## ComfyUI Node Contract

The maintained Comfy node surface is:

- `ERMBG Route Matte`: optional Comfy graph auto route and matte node.
- `ERMBG Route Strategy`: route-only debug/branching node.
- `ERMBG PyMatting Known-B`: deterministic known-background node for hard UI
  and stable known-background graphics.
- `ERMBG Classify (preview)`: lightweight diagnostic node.
- `Convert Masks to Images`: utility node.

Custom Comfy graphs may use `ERMBG Route Strategy` for branching. Web/API
production uses Direct Worker for `backend=auto`; explicit Comfy paths are
debug/extension paths.

## Optional OpenClaw Adapter

OpenClaw integration is an optional independent `ermbg-matte` skill:

```bash
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
  --image /path/to/input.png
```

This path should call the maintained ERMBG service/API and archive `output.png`,
`manifest.json`, and runtime metadata. Keep ERMBG route logic in the shared
core.

OpenClaw-specific features should remain a thin adapter over the same
route/matte contract.

## Operating Rules

1. `router.py` is the source of truth for asset family,
   `parameter_profile`, and `execution_profile`.
2. Adapters stay thin. Web, CLI, Direct Worker, Comfy nodes, and optional
   integrations pass data into the shared contract.
3. Direct Worker is the Web/API service boundary.
4. ComfyUI is an optional graph host for custom Comfy workflows.
5. Every adapter should write browsable artifacts with output PNGs, route
   metadata, timing metadata, and an `ermbg.run.v1` manifest where applicable.

## Anti-Patterns

- Duplicating route heuristics in Web JavaScript, optional adapter code, Comfy
  wrapper code, or Direct Worker glue.
- Adding a new backend that implies a new profile contract.
- Fixing a sample by branching on sample IDs, filenames, one-off dimensions, or
  fixed coordinates.
- Running heavy generation or VLM models in the local Web process.
- Making normal Web startup fail because ComfyUI is unavailable.
- Treating a local source change as deployed before the relevant Direct Worker
  or optional Comfy adapter has been restarted and smoke-tested.
