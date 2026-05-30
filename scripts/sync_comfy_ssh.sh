#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync ERMBG source to the ComfyUI server over SSH without git.

Environment defaults:
  ERMBG_COMFY_SSH=ermbg-comfy
  ERMBG_REMOTE_ROOT=C:/Users/darkv/ermbg_src
  ERMBG_REMOTE_COMFY=E:/ComfyUI
  ERMBG_REMOTE_PYTHON=E:/ComfyUI/.venv/Scripts/python.exe
  ERMBG_SSH_PASSWORD=...  optional; uses sshpass when set

Usage:
  scripts/sync_comfy_ssh.sh [--clean] [--nodes] [--install-editable] [--smoke]

Options:
  --clean             Remove synced source subdirectories before extraction.
  --nodes             Copy comfy_nodes into ComfyUI/custom_nodes/ermbg-comfy.
                      This changes Comfy node code and needs a ComfyUI restart.
  --install-editable  Run pip install -e on the remote source tree.
                      Usually needed once after first sync, not every iteration.
  --smoke             Run a quick remote Python import/router smoke.
EOF
}

clean=0
sync_nodes=0
install_editable=0
smoke=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) clean=1 ;;
    --nodes) sync_nodes=1 ;;
    --install-editable) install_editable=1 ;;
    --smoke) smoke=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

host="${ERMBG_COMFY_SSH:-ermbg-comfy}"
remote_root="${ERMBG_REMOTE_ROOT:-C:/Users/darkv/ermbg_src}"
remote_comfy="${ERMBG_REMOTE_COMFY:-E:/ComfyUI}"
remote_python="${ERMBG_REMOTE_PYTHON:-E:/ComfyUI/.venv/Scripts/python.exe}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${ERMBG_SSH_PASSWORD:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "ERMBG_SSH_PASSWORD is set but sshpass is not installed." >&2
    exit 2
  fi
  export SSHPASS="$ERMBG_SSH_PASSWORD"
  ssh_base=(sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$host")
else
  ssh_base=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$host")
fi

echo "==> checking SSH: $host"
"${ssh_base[@]}" "echo ERMBG_SSH_OK" >/dev/null

if [[ "$clean" -eq 1 ]]; then
  echo "==> cleaning remote source subsets under $remote_root"
  "${ssh_base[@]}" "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '$remote_root' | Out-Null; foreach (\$p in @('ermbg','comfy_nodes','docs','samples/corridorkey_semantic')) { \$full = Join-Path '$remote_root' \$p; if (Test-Path \$full) { Remove-Item -Recurse -Force \$full } }\""
else
  "${ssh_base[@]}" "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path '$remote_root' | Out-Null\""
fi

echo "==> streaming source tar to $host:$remote_root"
tar \
  --exclude='.DS_Store' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.venv' \
  --exclude='.git' \
  --exclude='out' \
  --exclude='ermbg.egg-info' \
  -czf - \
  ermbg comfy_nodes docs samples/corridorkey_semantic pyproject.toml README.md DEPLOY.md AGENTS.md \
  | "${ssh_base[@]}" "tar -xzf - -C \"$remote_root\""

if [[ "$install_editable" -eq 1 ]]; then
  echo "==> installing remote source editable into ComfyUI Python"
  "${ssh_base[@]}" "powershell -NoProfile -Command \"& '$remote_python' -m pip install -e '$remote_root'\""
fi

if [[ "$sync_nodes" -eq 1 ]]; then
  echo "==> syncing Comfy custom node wrapper (ComfyUI restart required)"
  "${ssh_base[@]}" "powershell -NoProfile -Command \"\$dst = '$remote_comfy/custom_nodes/ermbg-comfy'; if (Test-Path \$dst) { Remove-Item -Recurse -Force \$dst }; New-Item -ItemType Directory -Force -Path \$dst | Out-Null; Copy-Item -Recurse -Force '$remote_root/comfy_nodes/*' \$dst\""
fi

if [[ "$smoke" -eq 1 ]]; then
  echo "==> remote import/router smoke"
  "${ssh_base[@]}" "powershell -NoProfile -Command \"cd '$remote_root'; & '$remote_python' -c \\\"from ermbg import classify_image; s=classify_image(r'samples/corridorkey_semantic/button/button_green_yellow_a_outlined_hard_lite_shadow/green.png'); print(s.name, s.bg_type, s.keyer_mode)\\\"\""
fi

echo "==> sync complete"
