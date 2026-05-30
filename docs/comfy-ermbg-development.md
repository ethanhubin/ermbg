# Comfy ERMBG Development Workflow

`comfy-ermbg` is currently a diagnostic/comparison backend, not the automatic
production route. The Web/API automatic path routes green/blue screen assets to
`comfy-corridorkey` plus local ShadowPatch, and unknown backgrounds directly to
`comfy-rmbg` fallback. Treat the local Python package as the source
implementation and the Comfy custom nodes as deployed runtime wrappers.

## What Must Move Together

When changing matting behavior, update and verify the whole chain:

1. Local implementation under `ermbg/`.
2. Comfy wrapper under `comfy_nodes/`.
3. Remote ComfyUI custom-node install on `http://192.168.0.8:8000`.
4. Web/API path using `backend=auto`, plus explicit backend smoke where needed.
5. Regression samples and batch summaries under `samples/` and `out/`.

Do not accept a local-only pass as proof that Web is fixed. A change is not
production-ready until the remote node and Web endpoint exercise the updated
code.

## Next Algorithm Priority

The current game UI mainline is documented in
[`docs/corridorkey-game-ui-plan.md`](corridorkey-game-ui-plan.md). The short
version: use remote CorridorKey as the detail matting engine for game UI assets,
while ERMBG provides screen/color analysis, parameter adaptation, mask hints,
ShadowPatch, QA, Web controls, and fallback routing. The current automatic route
is green/blue screen -> `comfy-corridorkey` + ShadowPatch, unknown background ->
`comfy-rmbg` fallback. `comfy-ermbg` is temporarily outside the automatic path
and remains an explicit diagnostic/comparison backend.

For `comfy-corridorkey`, local ERMBG code now owns the post-CorridorKey
ShadowPatch step. The remote node still produces the subject layer; the Mac-side
API measures known-background scalar darkening and composites a separate shadow
layer underneath only when the patch gate confirms both high-confidence shadow
evidence and missing CorridorKey shadow alpha. This route is part of the
CorridorKey mainline, not a `comfy-ermbg` fallback.

The earlier solid-background/local-ownership plan remains available in
[`docs/solid-bg-graphic-plan.md`](solid-bg-graphic-plan.md) as historical
reference and fallback design material. It is no longer the primary detail
matting roadmap for game UI assets.

## Generality Contract

Use real regressions to reveal a failure class, but do not solve only the
sample in front of you. Every algorithm change should be stated as a general
mechanism with measurable signals, then guarded by tests that exercise that
mechanism.

Good examples:

- "Corner patches agree even though the whole edge is contaminated by subject
  pixels, so solid-background routing should trust the stable corners."
- "Known-background color evidence says a low-alpha region is not background,
  and topology says it is anchored to high-confidence subject, so repair local
  matting recall inside the subject component."
- "A dark region matches scalar known-background darkening and is outside
  subject ownership, so preserve it as shadow rather than opaque material."
- "CorridorKey preserved the subject but missed a measured cast shadow, so add a
  ShadowPatch layer underneath only when CorridorKey alpha is low over the
  shadow support."

Bad examples:

- Encoding a sample id, filename, exact icon size, hard-coded crop, or one-off
  coordinate range.
- Moving a threshold until one contact sheet looks right without documenting
  the observable signal and the class of failures it protects.
- Treating generated eval cases as replacements for real user regressions.

When a fix needs empirical gates, write a nearby comment that distinguishes the
invariant from the experience-driven value, and add a focused synthetic test
for the mechanism. For CorridorKey ShadowPatch changes, test both the positive
case (missing shadow gets patched below the subject) and the negative case
(CorridorKey already preserved shadow alpha, so the patch is skipped). For
production regressions, also run the real sample through Web smoke.

## Fast SSH Sync

For iteration, do not use commit/pull. Sync the working tree directly to the
Windows ComfyUI server:

```bash
scripts/sync_comfy_ssh.sh --smoke
```

The script streams a tarball over SSH to `C:/Users/darkv/ermbg_src`, excluding
`.git`, `.venv`, `out`, caches, and bytecode. It uses the `ermbg-comfy` SSH
alias by default; that alias should use key auth from `~/.ssh/config`.

One-time setup on a fresh remote environment:

```bash
scripts/sync_comfy_ssh.sh \
  --clean \
  --install-editable \
  --smoke
```

After that, ordinary algorithm iteration is:

```bash
scripts/sync_comfy_ssh.sh --smoke
```

Important: `--smoke` runs a fresh remote Python process and proves that the
remote source tree imports correctly. It does not prove that the long-running
ComfyUI process has reloaded already-imported modules.

Only use `--nodes` when `comfy_nodes/` changed:

```bash
scripts/sync_comfy_ssh.sh --nodes
```

Changing the Comfy node wrapper still requires a ComfyUI restart because Comfy
loads custom node classes at process startup. Pure `ermbg/` source syncs do not
need reinstall, but the current running ComfyUI process may keep already-imported
Python modules in memory. If behavior does not change after sync, restart once
or move the runtime behind a worker/subprocess boundary before continuing a
tight iteration loop.

## Dev Reload And Offline Restart

For fast algorithm iteration, start the remote ComfyUI with ERMBG dev reload:

```bash
scripts/restart_comfy_ssh.sh --restart --dev-reload
```

With `ERMBG_DEV_RELOAD=1`, the `ErmbgAutoMatte` node reloads ERMBG's pure
Python routing/matting modules at the start of each prompt and rebinds the
`ermbg.api` function references that are normally captured at import time. The
API module itself stays loaded, so its segmenter/model cache survives. This is
the preferred loop for changes under `ermbg/`:

```bash
scripts/sync_comfy_ssh.sh --smoke
.venv/bin/python scripts/run_comfy_regression.py \
  samples/regression/small_ui_icon_green/input.png \
  --batch out/comfy_ermbg_debug_<YYYYMMDD> \
  --phase after_change
```

Dev reload is intentionally opt-in. Still restart ComfyUI when:

- `comfy_nodes/` changed, because ComfyUI loads custom node classes at startup.
- `ermbg.api` public signatures changed in a way the wrapper calls directly.
- third-party package versions, model paths, or custom node dependencies changed.
- you need to prove behavior under production-like no-reload conditions.

The restart helper sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before
launching ComfyUI. This avoids slow or flaky Hugging Face HEAD requests after a
restart when the model is already cached on the Windows server.

## Required Verification Ladder

Run these in order after algorithm or Web-facing backend changes:

```bash
.venv/bin/pytest -q tests/test_router.py tests/test_matting.py tests/test_shadow.py tests/test_ownership.py
```

For Comfy wrapper/API changes:

```bash
.venv/bin/pytest -q tests/test_api.py tests/test_comfy_ermbg_matte.py tests/test_comfy_nodes.py tests/test_web.py
```

Then verify the remote server directly with at least one representative input:

```bash
.venv/bin/python scripts/run_comfy_regression.py \
  samples/regression/small_ui_icon_green/input.png \
  --batch out/comfy_ermbg_smoke_<YYYYMMDD> \
  --phase remote
```

For Web-facing work, start a fresh server exactly as specified in `AGENTS.md`,
then post through the real HTTP endpoint:

```bash
curl -sS -F 'file=@samples/regression/small_ui_icon_green/input.png' \
  -F 'backend=comfy-ermbg' \
  http://127.0.0.1:7860/api/matte-candidates \
  > out/comfy_ermbg_web_smoke_<YYYYMMDD>/response.json
```

The response must include `backend == "comfy-ermbg"` and `server_elapsed_sec`.
For quality-sensitive changes, also save the returned candidate PNG and inspect
it on checker, white, and black backgrounds.

The regression runner can perform the Web smoke and save a JSON-safe summary in
the same case directory:

```bash
.venv/bin/python scripts/run_comfy_regression.py \
  samples/regression/small_ui_icon_green/input.png \
  --batch out/comfy_ermbg_smoke_<YYYYMMDD> \
  --phase web \
  --web-smoke
```

The runner writes `input.png`, `rgba.png`, `alpha.png`, `foreground.png`,
`contact_sheet.png`, per-run `summary.json`, and an aggregate batch
`summary.json`. It records color-distance coverage metrics for known-background
regressions; use those metrics as objective guards alongside visual inspection.

## Regression Sample Rules

Add samples when a real user-facing failure exposes a new class of risk. Keep
the sample small, named by failure class, and documented with a nearby
`case.json`.

Current real-world regression:

- `samples/regression/small_ui_icon_green/input.png`
- Failure class: small UI/game icon, green corners visible, subject or rim
  touches the border, whole-edge background sampling becomes contaminated.
- Required local guard: router must classify it as `saturated`, not `noisy`.
- Required production guard: `comfy-ermbg` Web result must preserve the full icon
  silhouette, not only the most salient interior symbol.

Generated game-eval samples are useful for ownership and shadow reasoning, but
they do not replace real small-asset regressions. Keep both sets active.
