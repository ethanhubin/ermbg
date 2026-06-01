# ERMBG install and startup

The default service flow uses Direct Worker. Direct Worker can run on the same
machine as Web or on a remote server; Web reads its HTTP URL from configuration.

## Runtime shape

```text
Browser
  -> local ERMBG Web UI / API :7860
  -> configured ERMBG Direct Worker URL
  -> shared router + execution profiles
  -> PyMatting Known-B / CorridorKey runner / passthrough
```

ComfyUI is an optional adapter. Install `comfy_nodes/` when a Comfy graph needs
ERMBG nodes.

## Install

```powershell
cd <ermbg-root>
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"
```

The `torch` extra supports the Direct Worker CorridorKey route. PyMatting-only
tests can run without it. Game-asset service installs should include `torch`.

## Configuration

Service endpoints live in `ermbg.config.json`:

```json
{
  "services": {
    "direct_worker_url": "...",
    "comfy_url": "..."
  },
  "web": {
    "auto_backend": "direct-worker",
    "auto_fallback_backend": "pymatting-known-b",
    "enable_comfy": false
  }
}
```

Environment variables `ERMBG_DIRECT_URL`, `COMFY_URL`, `ERMBG_WEB_AUTO_BACKEND`,
`ERMBG_WEB_AUTO_FALLBACK_BACKEND`, and `ERMBG_ENABLE_COMFY` can override the
config for one shell session.

## Start Web with local Direct Worker

Start both services:

```powershell
.\scripts\start_local.ps1
```

Manual commands:

```powershell
Start-Process -WindowStyle Hidden -WorkingDirectory . `
  -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList "-m ermbg.direct_worker_server --host 127.0.0.1 --port 7871"

Start-Process -WindowStyle Hidden -WorkingDirectory . `
  -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList "-m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860"
```

Open:

```text
<web-url>
```

## Start Web with remote Direct Worker

Start the worker on the remote server:

```powershell
cd C:\path\to\ermbg
.\.venv\Scripts\python.exe -m ermbg.direct_worker_server --host 0.0.0.0 --port 7871 --cpu-workers 4
```

Start the local Web service and point it at the remote worker:

```powershell
.\scripts\start_local.ps1 -SkipDirectWorker -DirectUrl <services.direct_worker_url>
```

Environment form:

```powershell
$env:ERMBG_DIRECT_URL = "<services.direct_worker_url>"
$env:ERMBG_WEB_AUTO_BACKEND = "direct-worker"
$env:ERMBG_ENABLE_COMFY = "0"
.\.venv\Scripts\python.exe -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860
```

## Verify

```powershell
curl.exe -sS "<services.direct_worker_url>/health"
curl.exe -sS "<web-url>/api/runtime-capabilities?include_comfy=false&include_object_info=false"
```

Capability response:

- `local.status = ok`
- `direct_worker.status = ok`
- `web.auto_backend = direct-worker`
- `web.enable_comfy = false`
- `comfy.status = disabled`

## Optional Comfy adapter

Comfy graph support:

1. Install ERMBG into the Comfy Python environment.
2. Copy `comfy_nodes/` to Comfy's `custom_nodes/ermbg-comfy`.
3. Restart Comfy because custom nodes are scanned at process start.
4. Verify `/object_info` contains `ErmbgRouteMatte`, `ErmbgRouteStrategy`,
   `ErmbgPyMattingKnownB`, and `ErmbgClassify`.

Web-side debugging of explicit Comfy backends:

```text
ERMBG_ENABLE_COMFY=1
COMFY_URL=<services.comfy_url>
```

Normal Web/API usage leaves `ERMBG_ENABLE_COMFY=0`.
