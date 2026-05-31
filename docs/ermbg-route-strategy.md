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

## Web Verification

After Web-facing route changes:

1. Restart the local Web server on `127.0.0.1:7860`.
2. Verify the index contains `Auto RouteMatte`, `comfy-pymatting-known-b`, and
   `comfy-corridorkey`.
3. Post real samples to `/api/matte-candidates` with `backend=auto` and confirm
   `requested_backend`, `backend`, `debug.auto_route.selected_backend`,
   `debug.auto_route.route`, `execution_profile`, and `server_elapsed_sec`.

The standard smoke set is:

- hard button -> `comfy-pymatting-known-b` / `pymatting-hard-button`
- blue/green glass button -> `comfy-corridorkey` / `corridorkey-transparent-button`
- effect icon -> `comfy-corridorkey` / `corridorkey-effect-icon`
- shaped icon -> `comfy-corridorkey` / `corridorkey-shaped-icon`
- character -> `comfy-corridorkey` / `corridorkey-character`
- random/unknown background -> `comfy-pymatting-known-b` /
  `pymatting-fallback`
