# ERMBG · Agent Contract

This file is the bootstrap contract. Load deeper docs only when the task touches
that area.

## Runtime Defaults

- Web/API `backend=auto` uses Direct Worker.
- Service endpoints and Web defaults live in `ermbg.config.json`.
- Environment overrides are allowed for one shell session:
  `ERMBG_DIRECT_URL`, `COMFY_URL`, `ERMBG_WEB_AUTO_BACKEND`,
  `ERMBG_WEB_AUTO_FALLBACK_BACKEND`, `ERMBG_ENABLE_COMFY`.
- Normal Web startup and runtime capability checks use Direct Worker.
- ComfyUI is optional extension support for custom Comfy graphs or explicit
  `comfy-*` backend debugging.

## Where To Look

- Product overview and install/startup: `README.md`, `docs/install-startup.md`.
- Architecture and profile contract: `docs/architecture.md`,
  `docs/ermbg-route-strategy.md`.
- Local ownership details: `docs/local-ownership.md`.
- Optional Comfy node work: `comfy_nodes/README.md`, `DEPLOY.md`.
- Historical material: `docs/archive/`; do not treat archived plans as active.

## Development Basics

- Python venv is `.venv/` with Python 3.12.
- On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\pytest.exe`.
- Keep tests passing. For Web/runtime changes, include `tests/test_web.py` and
  `tests/test_runtime_capabilities.py`.
- Generated eval/debug artifacts must go under a self-contained batch directory
  in `out/`, with a machine-readable summary.

## Algorithm Rules

- Design target: pixel-perfect transparent matting on solid-color backgrounds.
- Game UI assets use PyMatting Known-B plus pixel-level repair from measurable
  known-background evidence.
- Complex green/blue-screen assets use CorridorKey.
- Image feature classification chooses algorithm and parameters before
  execution. Execution consumes `execution_profile`; it must not re-infer asset
  classes.
- Algorithm fixes must be mechanism-driven. Do not tune around sample IDs,
  filenames, coordinates, icon sizes, or one observed color unless Ethan asks
  for a one-off workaround.
- Heuristic thresholds, confidence gates, falloff widths, area ratios, and
  remapping constants need nearby comments explaining intent, observable signal,
  and protected failure mode.
- Web matting must preserve shadow handling unless Ethan explicitly asks for a
  preview-only speed mode.

## Web Verification

After changes to `ermbg/web.py`, Web UI/API behavior, or a Web-facing backend:

1. Restart local services with `scripts\start_local.ps1` or equivalent.
2. Confirm port `7860` is owned by the expected `uvicorn ermbg.web:app` process.
3. Confirm the index contains the changed marker, such as `Auto Direct Worker`
   or the relevant UI text.
4. Run a real HTTP smoke through `/api/matte-candidates` with `backend=auto`;
   confirm HTTP 200, route/profile metadata, and `server_elapsed_sec`.
5. If Web reports a Direct Worker connection error, check
   `<services.direct_worker_url>/health` before changing algorithm code.

Final Web status should state the `7860` PID, how Web was started, Direct Worker
health, and the real HTTP smoke result.

## Optional Comfy Work

- Use Comfy only for custom graph support or explicit `comfy-*` backend work.
- If `comfy_nodes/` changes, reinstall/sync the custom node and restart ComfyUI.
- Keep Comfy wrappers thin over the shared ERMBG API and CorridorKey runner.
