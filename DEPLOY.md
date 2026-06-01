# Optional ComfyUI Node Deployment

ERMBG Web/API uses Direct Worker by default. This document covers installation
of the optional ERMBG nodes into a ComfyUI environment.

Service endpoints are configured in `ermbg.config.json`:

```json
{
  "services": {
    "comfy_url": "..."
  }
}
```

Environment variable `COMFY_URL` can override `services.comfy_url` for one shell
session.

## Nodes

The optional Comfy package exposes:

- `ERMBG Route Matte`
- `ERMBG Route Strategy`
- `ERMBG PyMatting Known-B`
- `ERMBG Classify`

## Install

1. Install ERMBG into the Python environment used by ComfyUI:

   ```powershell
   <comfy-python> -m pip install -e <ermbg-root>
   ```

2. Copy or link the node package into ComfyUI `custom_nodes`:

   ```powershell
   Copy-Item -Recurse -Force <ermbg-root>\comfy_nodes <comfy-custom-nodes>\ermbg-comfy
   ```

   The custom node directory must not be named `ermbg`, because that conflicts
   with the Python package import.

3. Restart ComfyUI. Custom nodes are loaded at process startup.

4. Verify node registration from the configured `services.comfy_url`:

   ```bash
   curl -s "<services.comfy_url>/object_info" | python -c "import json,sys; d=json.load(sys.stdin); print([k for k in d if k.startswith('Ermbg')])"
   ```

   Expected keys include `ErmbgRouteMatte`, `ErmbgRouteStrategy`,
   `ErmbgPyMattingKnownB`, and `ErmbgClassify`.

## Update

- Changes under `ermbg/` require the Comfy Python environment to import the
  updated ERMBG package. Editable installs make this simple.
- Changes under `comfy_nodes/` require copying/linking the node package again
  and restarting ComfyUI.
- Keep Comfy wrappers thin. Route/profile decisions and CorridorKey execution
  belong in shared ERMBG code, not in Comfy-only branches.
