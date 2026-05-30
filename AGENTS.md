# ERMBG · Engineering Contract

## Available Infrastructure

### ComfyUI Server (REMOTE — preferred over local model loads)

ComfyUI is reached through `COMFY_URL` from the environment or local `.env`.
If it is not configured, code falls back to **`http://192.168.0.8:8000`**.

- **Hardware**: Windows + RTX 4090 (24 GB VRAM), 64 GB RAM. Verify ComfyUI version with `/system_stats` before version-sensitive work.
- **Always running** — do **not** install or run SDXL / FLUX / Qwen / RMBG generation models locally on the Mac. The Mac is for orchestration, CLI, lightweight CV/numpy, and allowed local BiRefNet-matting only. Heavy generation/inference goes to ComfyUI.
- **Mac local model budget**: BiRefNet-matting (≈1 GB MPS) is allowed; SDXL/FLUX/Qwen-scale models are not allowed locally.

**Installed nodes / models** (cache the full list with `curl -s "${COMFY_URL:-http://192.168.0.8:8000}/object_info" > /tmp/comfy_object_info.json` when needed; do not call `/object_info` on latency-sensitive hot paths):

- Generation backends: Qwen-Image-Edit 2511 fp8, Flux Dev fp8, Flux 2 Klein 9b, FLUX schnell, Z-Image-Turbo
- Background removal: `Image Rembg` (isnet-general-use / u2net / u2netp / silueta / isnet-anime), `BriaRemoveImageBackground`, `RemoveBackground`, `LayerMask: RmBgUltra V2`
- IPAdapter / ControlNet / many LoRAs available
- VAE: qwen_image_vae, flux2-vae, flux ultra vae

**HTTP API** the Python client uses ([ermbg/probe/comfyui.py](ermbg/probe/comfyui.py) is the reference implementation):

```
POST /upload/image          (multipart with name + overwrite=true)
POST /prompt                (workflow JSON, returns prompt_id)
GET  /history/<prompt_id>   (poll until status.completed)
GET  /view?filename=...&subfolder=...&type=output
```

Workflow templates live in `ermbg/probe/comfyui_*.json`, with `${variable}` placeholders filled by `string.Template`.

### When to use ComfyUI vs local

| Task | Where |
|---|---|
| Full ERMBG AutoMatte / Web default matting | ComfyUI (`ErmbgAutoMatte`, backend `comfy-ermbg`) |
| BiRefNet-matting (1 GB) | Local MPS (`ermbg.segmenter.BiRefNetSegmenter`) |
| BRIA RMBG-2.0 (gated) | ComfyUI (`BriaRemoveImageBackground` node) |
| RMBG-1.4 / IS-Net family | ComfyUI (`Image Rembg` with `isnet-general-use`) |
| SDXL / FLUX / Qwen-Edit generation | ComfyUI (always) |
| Pure numpy / OpenCV / scipy ops | Local |

### OpenAI API

`OPENAI_API_KEY` lives in `.env` (gitignored). Use it only when a task explicitly needs `gpt-image-1` cloud editing. Prefer ComfyUI for image generation.

---

## Project structure

```
ermbg/
  segmenter.py        BiRefNet (default: ZhengPeng7/BiRefNet-matting)
  matting.py          end-to-end pipeline
  diagnose.py         background diagnoser (B, purity, edge_q10)
  trimap.py           trimap construction (legacy path only)
  alpha.py            projection / per-channel / guided filter (legacy path)
  foreground.py       KNN F_ref (used by despill.local_borrow)
  recover.py          legacy decontamination (deprecated)
  despill.py          chroma_cap | local_borrow | closed_form | none
  lightwrap.py        edge halo suppression (Brinkmann light wrap)
  qa.py               composite to 6 backgrounds, score halos
  cli.py              segment / diagnose / matte / phase1 / probe
  probe/
    generator.py      backend protocol
    synthetic.py      mask-and-paste baseline
    sdxl_inpaint.py   archived local SDXL inpainting path; do not run on Mac
    comfyui.py        Qwen-Edit via remote ComfyUI
    comfyui_ermbg_matte.py  full ERMBG AutoMatte via remote ComfyUI
    openai_image.py   gpt-image-1 via OpenAI API
    comfyui_*.json    workflow templates
    prompts.py        GREEN_SCREEN_RGB / GREEN_SCREEN_PROMPT
samples/corridorkey_semantic/  current B/I/C Web/Game Eval sample set
tests/                pytest suite; keep `.venv/bin/pytest -q` passing
```

## Conventions

- All code in **linear RGB** internally; convert at I/O boundary via `ermbg.io.{srgb_to_linear, linear_to_srgb_u8}`.
- Color distance work in **OKLab** via `ermbg.colorspace.oklab_distance`.
- Soft mask / alpha is **float32 [0, 1]**, H×W. RGBA outputs are **uint8 sRGB** with alpha in last channel.
- Tests must keep passing (`pytest -q`). When adding modules, add a smoke test.
- The python venv is **`.venv/`** (managed by uv, Python 3.12). All `.venv/bin/ermbg`, `.venv/bin/pytest`, `.venv/bin/python` commands.

### AI / algorithm tuning contract

- Algorithm fixes must be mechanism-driven, not sample-driven. Real failure
  samples are for discovering and validating a general failure class; do not
  tune around one file, one coordinate range, one icon size, or one observed
  color unless Ethan explicitly asks for a one-off workaround.
- Each visual/matting fix must name the underlying failure mechanism in code
  comments or tests: for example, "whole-edge bg sampling contaminated but
  corners are stable", "known-B key evidence disagrees with local matting
  recall inside an anchored subject component", or "shadow-like darkening is
  separable from opaque subject ownership".
- Any algorithm-detail adjustment made by an AI agent, especially heuristic thresholds, confidence gates, falloff widths, area ratios, or display/export remapping constants, must include a nearby code comment explaining the intent and the failure mode it protects against.
- Comments must distinguish broad invariants from empirical values. If a value is experience-driven, say what observable signal it keys on and which class of samples motivated it; do not leave a naked magic number.
- Do not encode sample IDs, file names, or one-off coordinates as fixes unless explicitly requested. Prefer feature-based rules and document why the rule should generalize.
- When changing visual/matting behavior, write or update a focused test that captures the intended class of failure, not just the current sample. Prefer synthetic tests for the mechanism plus a real-sample production regression when the failure came from user-facing output.

### Test / eval batch convention

- All sample tests, visual regressions, local-ownership/game eval reruns, probe comparisons, and one-off debugging runs that produce artifacts must write into a **batch directory** under `out/`.
- Do not write new eval artifacts directly into loose ad-hoc paths. Use a stable batch id such as `out/local_ownership_<purpose>_<YYYYMMDD>/` or an explicit user-provided batch name.
- Each batch should be self-contained and browsable: keep per-case outputs under the batch root and write a machine-readable summary (`summary.json`, `summary_*.json`, or `eval_report.json`) at a predictable location.
- Web/debug tooling should discover and browse batches rather than hard-code a single result directory. New test flows should automatically register their outputs through the batch summary.
- When re-running a specific sample such as `B001`, still run it through the same batch flow and record the selected sample id in the summary; do not create orphan outputs.

### Web server update / verification contract

After any change to [ermbg/web.py](ermbg/web.py), Web UI behavior, Web API behavior, or a Web-facing backend path, an AI agent must verify that port **7860** is running the updated server before asking Ethan to test in the browser.

- Do **not** assume `pkill -f 'uvicorn ermbg.web:app'` is enough. First inspect the listener:

  ```bash
  lsof -nP -iTCP:7860 -sTCP:LISTEN
  ps -p <PID> -o pid,ppid,pgid,command
  launchctl print gui/$(id -u)/com.ethanhu.ermbg-web 2>&1 | head -n 80
  ```

- If `com.ethanhu.ermbg-web` is loaded, stop it before starting a manual test server. Do not test against a LaunchAgent-managed `7860` process unless the task explicitly concerns that LaunchAgent.

  ```bash
  launchctl bootout gui/$(id -u)/com.ethanhu.ermbg-web || true
  ```

- Start the Web server in a detached `screen` session for local browser testing.

  ```bash
  screen -S ermbg-web -X quit >/dev/null 2>&1 || true
  screen -dmS ermbg-web bash -lc 'cd /Users/ethanhu/Desktop/Git/ERMBG && PYTHONPATH=/Users/ethanhu/Desktop/Git/ERMBG .venv/bin/python -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860 > /tmp/ermbg-web.log 2>&1'
  ```

- After starting, verify all three layers before reporting success:
  1. `lsof -nP -iTCP:7860 -sTCP:LISTEN` shows the expected `.venv/bin/python -m uvicorn ermbg.web:app` process.
  2. `curl -sS http://127.0.0.1:7860/` contains a marker from the change under test, not just any HTML response. For backend changes, verify expected option/text such as `comfy-ermbg`, `server_elapsed_sec`, or the specific UI string under test.
  3. Run a real HTTP smoke test through `127.0.0.1:7860`, not only `TestClient`. For Comfy-backed matting, post a small PNG to `/api/matte-candidates` with `backend=comfy-ermbg` and confirm status 200 plus JSON `backend == "comfy-ermbg"` and `server_elapsed_sec` is present.

- If Web reports a ComfyUI connection error, compare from the same shell before changing algorithm code:

  ```bash
  curl -sS --connect-timeout 3 "${COMFY_URL:-http://192.168.0.8:8000}/queue"
  .venv/bin/python - <<'PY'
  import os
  import requests
  print(requests.get(os.environ.get("COMFY_URL", "http://192.168.0.8:8000") + "/system_stats", timeout=3).status_code)
  PY
  ```

- Final status updates about Web testing must state which PID owns `7860`, how the server was started, and the result of the real HTTP smoke test.

### Comfy ERMBG development contract

`comfy-ermbg` is the production matting path. The Web UI and slice-to-matte flow
use the remote ComfyUI `ErmbgAutoMatte` node because local full matting is too
slow for normal interactive use.

- Local Python changes under `ermbg/` are source changes, not sufficient
  deployment proof. After Web-facing or algorithmic changes, verify the remote
  Comfy node and a real Web HTTP request before declaring the production path
  fixed.
- Do **not** use commit/pull as the normal ComfyUI iteration path. Sync the
  working tree directly over SSH with [scripts/sync_comfy_ssh.sh](scripts/sync_comfy_ssh.sh).
  The default SSH alias is `ermbg-comfy` (configured in `~/.ssh/config` and
  currently using `~/.ssh/id_nas`), the target root is
  `C:/Users/darkv/ermbg_src`, and the
  remote ComfyUI Python is `E:/ComfyUI/.venv/Scripts/python.exe`.
- The remote venv is expected to import ERMBG from the editable source tree:
  `C:\Users\darkv\ermbg_src\ermbg\__init__.py`. If `python -c "import ermbg;
  print(ermbg.__file__)"` points at `site-packages`, run the one-time setup:

  ```bash
  scripts/sync_comfy_ssh.sh --clean --install-editable --smoke
  ```

- For ordinary algorithm changes under `ermbg/`, run:

  ```bash
  scripts/sync_comfy_ssh.sh --smoke
  ```

  Do not write passwords into files or docs. Prefer the configured SSH key.
  Only fall back to `ERMBG_SSH_PASSWORD` for emergency repair when key auth is
  unavailable.
- `--smoke` proves the remote source tree and remote Python import path are
  correct; it does **not** prove the already-running ComfyUI process has
  reloaded changed modules. For faster iteration, start ComfyUI with
  `ERMBG_DEV_RELOAD=1` (see [scripts/restart_comfy_ssh.sh](scripts/restart_comfy_ssh.sh)
  `--dev-reload`), then ordinary `ermbg/` source syncs are picked up on the
  next `ErmbgAutoMatte` prompt. If dev reload is off, a behavior change may
  still require a ComfyUI restart even after `--smoke` passes.
- Only use `--nodes` when `comfy_nodes/` changed. That syncs the Comfy custom
  node wrapper and requires a ComfyUI restart because Comfy loads node classes
  at startup:

  ```bash
  scripts/sync_comfy_ssh.sh --nodes
  ```

- Pure source syncs do not reinstall packages. However, the current running
  ComfyUI process may keep already-imported Python modules in memory. If a
  synced algorithm change is not reflected in `comfy-ermbg`, restart ComfyUI
  once before debugging the algorithm further.
- When restarting ComfyUI for ERMBG work, prefer the offline restart helper:

  ```bash
  scripts/restart_comfy_ssh.sh --restart --dev-reload
  ```

  The helper sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so the
  cached BiRefNet model is used without blocking on Hugging Face HEAD requests.
- Keep `comfy_nodes/` compatible with local API changes and update/restart the
  remote custom-node install when the node wrapper or called API surface changes.
- Use [docs/comfy-ermbg-development.md](docs/comfy-ermbg-development.md) as the
  checklist for required pytest groups, direct `backend="comfy-ermbg"` smoke,
  and `/api/matte-candidates` Web smoke.
- Generated game-eval samples do not replace real user regressions. Add real
  failure classes under `samples/regression/<case_id>/input.png` with a
  `case.json`, and add at least one focused pytest guard that can run without
  network access.

## Default decisions

- **Matting model**: `ZhengPeng7/BiRefNet-matting` (MIT, matting-trained)
- **Background convention**: green-screen RGB (0, 200, 0) — see `ermbg.probe.prompts.GREEN_SCREEN_PROMPT`
- **Despill default**: `chroma_cap` (auto-degrades to `local_borrow` when B has no dominant channel)
- **QA backgrounds**: black / white / grey / cyan / magenta / checker, plus a `_lightwrap` variant for each

### Algorithm / deployment contract

- Local ownership decision semantics are documented in `docs/local-ownership.md`.
- Use local, deterministic ownership scoring from measurable known-background evidence for ownership decisions.
- The Web UI default backend is `comfy-ermbg`; slice-to-matte transfer also selects `comfy-ermbg`. This runs the full ERMBG pipeline in the remote ComfyUI `ErmbgAutoMatte` node and returns foreground/alpha outputs to the Mac-side Web server.
- Do not add Web quick paths that skip shadow removal. Web matting must preserve `WEB_SHADOW_MODE = "on"` unless Ethan explicitly asks for a preview-only speed mode.
- Local ownership flow:
  1. local matte and known-background diagnosis compute `B`, alpha, and foreground/debug outputs;
  2. local evidence extractors produce risk/debug regions;
  3. `ermbg.ownership.rank_regions_ownership()` ranks each region as `hole`, `opaque_subject`, `subject_soft_layer`, `shadow_like_layer`, or `conservative_unknown`;
  4. `ermbg.ownership.resolve_execution_masks()` performs global arbitration before any role becomes an execution mask;
  5. `ermbg.matting.matte()` uses `subject_material_mask` only as a protective constraint, restoring soft-layer alpha after destructive keyer/repair changes;
  6. shadow opacity is still measured locally from `C_linear ~= scale * B_linear`.
- Do not route ownership ambiguity to model planning by default. First inspect local signals, role scores, execution masks, and whether foreground/color recovery is the real failure.
- Use focused B/I/C subsets from `samples/corridorkey_semantic/manifest.json` for fast loops; full route/eval work should run through the manifest-backed batch flow.
- Documents under `docs/archive/` are historical reference only; do not treat archived plans as active instructions.
