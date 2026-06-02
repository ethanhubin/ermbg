#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Restart or inspect the remote ERMBG Direct Worker over SSH.

Environment defaults:
  ERMBG_COMFY_SSH=ermbg-comfy
  ERMBG_REMOTE_ROOT=C:/Users/darkv/ermbg_src
  ERMBG_REMOTE_PYTHON=E:/ComfyUI/.venv/Scripts/python.exe
  ERMBG_DIRECT_PORT=7871
  ERMBG_DIRECT_LISTEN=0.0.0.0
  ERMBG_DIRECT_CPU_WORKERS=4
  ERMBG_DIRECT_TASK=ERMBGDirectWorker
  ERMBG_BUILD_GIT_SHA=<local git sha, plus -dirty when worktree is dirty>
  ERMBG_SSH_PASSWORD=...  optional; uses sshpass when set

Usage:
  scripts/restart_direct_worker_ssh.sh [--restart|--start|--stop|--status]
EOF
}

mode="restart"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) mode="restart" ;;
    --start) mode="start" ;;
    --stop) mode="stop" ;;
    --status) mode="status" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

host="${ERMBG_COMFY_SSH:-ermbg-comfy}"
remote_root="${ERMBG_REMOTE_ROOT:-C:/Users/darkv/ermbg_src}"
remote_python="${ERMBG_REMOTE_PYTHON:-E:/ComfyUI/.venv/Scripts/python.exe}"
port="${ERMBG_DIRECT_PORT:-7871}"
listen="${ERMBG_DIRECT_LISTEN:-0.0.0.0}"
cpu_workers="${ERMBG_DIRECT_CPU_WORKERS:-4}"
task_name="${ERMBG_DIRECT_TASK:-ERMBGDirectWorker}"

default_build_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git diff --quiet --ignore-submodules -- 2>/dev/null; then
  default_build_sha="${default_build_sha}-dirty"
fi
build_sha="${ERMBG_BUILD_GIT_SHA:-$default_build_sha}"

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
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*ermbg.direct_worker_server*" -and $_.CommandLine -like "*--port*__PORT__*" }
if ($procs) { $ids += $procs.ProcessId }
foreach ($ownerPid in ($ids | Sort-Object -Unique)) {
  if ($ownerPid -and $ownerPid -ne $PID) {
    Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
  }
}
'

remote_start='
New-Item -ItemType Directory -Force -Path "__REMOTE_ROOT__" | Out-Null
$taskScript = @"
`$env:ERMBG_BUILD_GIT_SHA = "__BUILD_SHA__"
`$env:PYTHONPATH = "__REMOTE_ROOT_WIN__"
Set-Location "__REMOTE_ROOT_WIN__"
& "__PYTHON_WIN__" -m ermbg.direct_worker_server --host __LISTEN__ --port __PORT__ --cpu-workers __CPU_WORKERS__
"@
$taskPath = Join-Path "__REMOTE_ROOT__" "start_direct_worker_task.ps1"
Set-Content -Encoding UTF8 -Path $taskPath -Value $taskScript
schtasks /Delete /TN "__TASK_NAME__" /F 2>$null | Out-Null
$time = (Get-Date).AddMinutes(1).ToString("HH:mm")
schtasks /Create /TN "__TASK_NAME__" /SC ONCE /ST $time /TR "powershell -NoProfile -ExecutionPolicy Bypass -File $taskPath" /F | Out-Host
schtasks /Run /TN "__TASK_NAME__" | Out-Host
'

subst_script() {
  # Keep forward slashes here. PowerShell accepts them for local paths, and
  # using backslashes in sed replacements would require another escaping layer.
  local remote_root_win="${remote_root}"
  local remote_python_win="${remote_python}"
  sed \
    -e "s#__PORT__#${port}#g" \
    -e "s#__LISTEN__#${listen}#g" \
    -e "s#__CPU_WORKERS__#${cpu_workers}#g" \
    -e "s#__TASK_NAME__#${task_name}#g" \
    -e "s#__BUILD_SHA__#${build_sha}#g" \
    -e "s#__REMOTE_ROOT__#${remote_root}#g" \
    -e "s#__REMOTE_ROOT_WIN__#${remote_root_win}#g" \
    -e "s#__PYTHON_WIN__#${remote_python_win}#g"
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
    sleep 8
    printf "%s" "$remote_status" | run_ps
    ;;
  restart)
    printf "%s" "$remote_stop" | run_ps
    sleep 2
    printf "%s" "$remote_start" | run_ps
    sleep 8
    printf "%s" "$remote_status" | run_ps
    ;;
esac
