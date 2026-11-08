# scripts/start_all.ps1
#
# Start the three resident shiro processes.
#
# Order is by dependency (CONTRACTS/00_overview.md section 9):
#   P1 first  - it is the only phase source
#   P3 second
#   P2 third  - it consumes P1 and P3
#
# No readiness wait: ZeroMQ PUB/SUB reconnects on its own.
# No heartbeat check: liveness is judged by events, not timers.
#
# 2026-11-08: refuse to start a second set. Two sets fight over 5560/5561/5562;
# the loser then crash-loops through its supervisor (5 identical failures, then
# exit), which looks like "it started but nothing is being recorded".

param(
    [switch]$NoP2,
    [switch]$NoP3,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[start] missing .venv\Scripts\python.exe" -ForegroundColor Red
    exit 1
}

if (-not $Force) {
    $running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "src\.runtime\.process_[123]_" }
    if ($running) {
        Write-Host ("[start] already running ({0} processes). Use scripts\stop_all.ps1 first," -f $running.Count) -ForegroundColor Yellow
        Write-Host "        or pass -Force to start anyway." -ForegroundColor Yellow
        exit 2
    }
}

Write-Host "[start] root: $root"

$procs = @()

function Start-Shiro {
    param([string]$Tag, [string]$Module)
    Write-Host "[start] launching $Tag ($Module)" -ForegroundColor Cyan
    Start-Process -FilePath $py -ArgumentList "-m", $Module -WorkingDirectory $root -PassThru
}

$procs += Start-Shiro -Tag "P1" -Module "src.runtime.process_1_fast"
Start-Sleep -Milliseconds 800

if (-not $NoP3) {
    $procs += Start-Shiro -Tag "P3" -Module "src.runtime.process_3_life"
    Start-Sleep -Milliseconds 500
}

if (-not $NoP2) {
    $procs += Start-Shiro -Tag "P2" -Module "src.runtime.process_2_mind"
}

Write-Host ""
Write-Host "[start] launched $($procs.Count) launchers (each spawns the real interpreter):" -ForegroundColor Green
foreach ($p in $procs) { Write-Host ("  PID {0}" -f $p.Id) }
Write-Host ""
Write-Host "[start] watch:  .\.venv\Scripts\python.exe tools\watch.py --window 60 --duration 3600"
Write-Host "[start] stop:   scripts\stop_all.ps1"
Write-Host "[start] observations: data\logs\observations.jsonl"