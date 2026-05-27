# ERMBG Handoff

The active engineering contract lives in [AGENTS.md](AGENTS.md). Read that file
before making changes.

Important current rule: `comfy-ermbg` is the production matting path. Do not use
git commit/pull for normal ComfyUI iteration. Sync the working tree directly to
the Windows ComfyUI server with:

```bash
ERMBG_SSH_PASSWORD='...' scripts/sync_comfy_ssh.sh --smoke
```

One-time editable install for a fresh remote environment:

```bash
ERMBG_SSH_PASSWORD='...' scripts/sync_comfy_ssh.sh --clean --install-editable --smoke
```

Only use `--nodes` and restart ComfyUI when `comfy_nodes/` changed. Never write
the SSH password into files; pass it via `ERMBG_SSH_PASSWORD` or use SSH keys.
