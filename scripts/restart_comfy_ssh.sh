#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Restart or inspect the remote ComfyUI server over SSH.

Environment defaults:
  ERMBG_COMFY_SSH=ermbg-comfy
  ERMBG_REMOTE_ROOT=C:/Users/darkv/ermbg_src
  ERMBG_REMOTE_COMFY_SOURCE=C:/Users/darkv/AppData/Local/Programs/ComfyUI/resources/ComfyUI
  ERMBG_REMOTE_COMFY_BASE=E:/ComfyUI
  ERMBG_REMOTE_PYTHON=E:/ComfyUI/.venv/Scripts/python.exe
  ERMBG_SSH_PASSWORD=...  optional; uses sshpass when set
  ERMBG_DEV_RELOAD=1      optional; enables ERMBG module hot reload in the custom node

Usage:
  scripts/restart_comfy_ssh.sh [--restart|--start|--stop|--status] [--dev-reload]

The launched process sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 so a
restart uses the already-cached matting model instead of blocking on network
HEAD requests to Hugging Face.
EOF
}

mode="restart"
dev_reload="${ERMBG_DEV_RELOAD:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) mode="restart" ;;
    --start) mode="start" ;;
    --stop) mode="stop" ;;
    --status) mode="status" ;;
    --dev-reload) dev_reload="1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

host="${ERMBG_COMFY_SSH:-ermbg-comfy}"
remote_root="${ERMBG_REMOTE_ROOT:-C:/Users/darkv/ermbg_src}"
remote_comfy_source="${ERMBG_REMOTE_COMFY_SOURCE:-C:/Users/darkv/AppData/Local/Programs/ComfyUI/resources/ComfyUI}"
remote_comfy_base="${ERMBG_REMOTE_COMFY_BASE:-E:/ComfyUI}"
remote_python="${ERMBG_REMOTE_PYTHON:-E:/ComfyUI/.venv/Scripts/python.exe}"
task_name="${ERMBG_COMFY_TASK:-ERMBGComfyOffline}"
port="${ERMBG_COMFY_PORT:-8000}"
listen="${ERMBG_COMFY_LISTEN:-192.168.0.8}"

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

remote_status='
$listeners = Get-NetTCPConnection -LocalPort __PORT__ -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
  $listeners | Select-Object LocalAddress,LocalPort,OwningProcess | Format-Table
  foreach ($ownerPid in ($listeners.OwningProcess | Sort-Object -Unique)) {
    Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" | Select-Object ProcessId,ParentProcessId,CommandLine | Format-List
  }
} else {
  Write-Output "No listener on port __PORT__"
}
'

remote_stop='
$ids = @()
$listeners = Get-NetTCPConnection -LocalPort __PORT__ -State Listen -ErrorAction SilentlyContinue
if ($listeners) { $ids += $listeners.OwningProcess }
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*--port*__PORT__*" }
if ($procs) { $ids += $procs.ProcessId }
foreach ($ownerPid in ($ids | Sort-Object -Unique)) {
  if ($ownerPid -and $ownerPid -ne $PID) {
    Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
  }
}
'

remote_start='
New-Item -ItemType Directory -Force -Path "__REMOTE_ROOT__" | Out-Null
$cmd = @(
  "@echo off",
  "set HF_HUB_OFFLINE=1",
  "set TRANSFORMERS_OFFLINE=1",
  "set ERMBG_DEV_RELOAD=__DEV_RELOAD__",
  "cd /d __COMFY_SOURCE__",
  "__PYTHON__ __COMFY_SOURCE__/main.py --user-directory __COMFY_BASE__/user --input-directory __COMFY_BASE__/input --output-directory __COMFY_BASE__/output --front-end-root __COMFY_SOURCE__/web_custom_versions/desktop_app --base-directory __COMFY_BASE__ --database-url sqlite:///__COMFY_BASE__/user/comfyui.db --extra-model-paths-config C:/Users/darkv/AppData/Roaming/ComfyUI/extra_models_config.yaml --log-stdout --listen __LISTEN__ --port __PORT__ --enable-manager"
)
$cmdPath = Join-Path "__REMOTE_ROOT__" "start_comfy_offline.cmd"
Set-Content -Encoding ASCII -Path $cmdPath -Value $cmd
schtasks /Create /TN "__TASK_NAME__" /TR $cmdPath /SC ONCE /ST 23:59 /F /RL HIGHEST /IT | Out-Host
schtasks /Run /TN "__TASK_NAME__" | Out-Host
'

subst_script() {
  sed \
    -e "s#__PORT__#${port}#g" \
    -e "s#__REMOTE_ROOT__#${remote_root}#g" \
    -e "s#__COMFY_SOURCE__#${remote_comfy_source}#g" \
    -e "s#__COMFY_BASE__#${remote_comfy_base}#g" \
    -e "s#__PYTHON__#${remote_python}#g" \
    -e "s#__LISTEN__#${listen}#g" \
    -e "s#__TASK_NAME__#${task_name}#g" \
    -e "s#__DEV_RELOAD__#${dev_reload}#g"
}

run_ps() {
  local script
  local encoded
  script="$(cat | subst_script)"
  script="\$ProgressPreference='SilentlyContinue'; $script"
  encoded="$(printf "%s" "$script" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n')"
  "${ssh_base[@]}" "powershell -NoProfile -NonInteractive -EncodedCommand $encoded"
}

case "$mode" in
  status)
    printf "%s" "$remote_status" | run_ps
    ;;
  stop)
    printf "%s" "$remote_stop" | run_ps
    sleep 2
    printf "%s" "$remote_status" | run_ps
    ;;
  start)
    printf "%s" "$remote_start" | run_ps
    sleep 12
    printf "%s" "$remote_status" | run_ps
    ;;
  restart)
    printf "%s" "$remote_stop" | run_ps
    sleep 3
    printf "%s" "$remote_start" | run_ps
    sleep 12
    printf "%s" "$remote_status" | run_ps
    ;;
esac

if [[ "$mode" == "start" || "$mode" == "restart" ]]; then
  for _ in {1..20}; do
    if curl -sS --connect-timeout 3 "http://${listen}:${port}/system_stats" >/tmp/ermbg_comfy_system_stats.json 2>/tmp/ermbg_comfy_system_stats.err; then
      echo "ComfyUI /system_stats OK: http://${listen}:${port}"
      exit 0
    fi
    sleep 2
  done
  cat /tmp/ermbg_comfy_system_stats.err >&2 || true
  exit 1
fi
