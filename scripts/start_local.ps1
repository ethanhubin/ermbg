param(
    [string]$HostAddress = "127.0.0.1",
    [int]$WebPort = 7860,
    [int]$DirectPort = 7871,
    [int]$CpuWorkers = 4,
    [string]$DirectUrl = "",
    [switch]$SkipDirectWorker
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "out\local_services"

if (-not (Test-Path $Python)) {
    throw "Missing venv Python: $Python. Run: uv venv .venv --python 3.12; uv pip install --python .\.venv\Scripts\python.exe -e `".[web,dev,torch]`""
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:PYTHONPATH = "$Root"
$ConfiguredDirectUrl = & $Python -c "from ermbg.settings import get_direct_worker_url; print(get_direct_worker_url())"
$EffectiveDirectUrl = if ($DirectUrl) { $DirectUrl.TrimEnd("/") } elseif ($ConfiguredDirectUrl) { $ConfiguredDirectUrl.TrimEnd("/") } else { "http://${HostAddress}:${DirectPort}" }
$env:ERMBG_DIRECT_URL = $EffectiveDirectUrl
$env:ERMBG_WEB_AUTO_BACKEND = "direct-worker"
$env:ERMBG_WEB_AUTO_FALLBACK_BACKEND = "pymatting-known-b"
$env:ERMBG_ENABLE_COMFY = "0"

function Stop-Listener {
    param([int]$Port)
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Stop-Listener -Port $WebPort
if (-not $SkipDirectWorker) {
    Stop-Listener -Port $DirectPort
}
Start-Sleep -Seconds 1

$directArgs = "-m ermbg.direct_worker_server --host $HostAddress --port $DirectPort --cpu-workers $CpuWorkers"
$webArgs = "-m uvicorn ermbg.web:app --host $HostAddress --port $WebPort"

if ($SkipDirectWorker) {
    $direct = $null
} else {
    $direct = Start-Process -FilePath $Python -ArgumentList $directArgs -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "direct-worker.out.log") -RedirectStandardError (Join-Path $LogDir "direct-worker.err.log") -PassThru
}
$web = Start-Process -FilePath $Python -ArgumentList $webArgs -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "web.out.log") -RedirectStandardError (Join-Path $LogDir "web.err.log") -PassThru

Start-Sleep -Seconds 3

$directHealth = try { Invoke-RestMethod -Uri "$EffectiveDirectUrl/health" -TimeoutSec 5 } catch { $null }
$webHealth = try { Invoke-WebRequest -Uri "http://${HostAddress}:${WebPort}/health" -TimeoutSec 5 } catch { $null }

[pscustomobject]@{
    WebUrl = "http://${HostAddress}:${WebPort}"
    WebPid = $web.Id
    WebHealth = if ($webHealth) { $webHealth.StatusCode } else { "failed" }
    DirectUrl = $EffectiveDirectUrl
    DirectPid = if ($direct) { $direct.Id } else { "remote/configured" }
    DirectHealth = if ($directHealth) { $directHealth.status } else { "failed" }
    LogDir = $LogDir
}
