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

The removed legacy path included local BiRefNet/GrabCut full matting,
`ErmbgAutoMatte`, `comfy-ermbg`, VLM-protected reruns, and the old CLIPSeg ->
AutoMatte subject-mask workflow. Those paths are no longer public API or Web
options.

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
   `debug.auto_route.route`, and `server_elapsed_sec`.

The standard smoke set is:

- hard button -> `comfy-pymatting-known-b`
- blue/green glass button -> `comfy-corridorkey`
- icon -> `comfy-corridorkey`
- character -> `comfy-corridorkey`
- random/unknown background -> `comfy-pymatting-known-b` /
  `pymatting_fallback`
