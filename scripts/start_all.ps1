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

param(
    [switch]$NoP2,
    [switch]$NoP3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[start] missing .venv\Scripts\python.exe" -ForegroundColor Red
    exit 1
}

Write-Host "[start] root: $root"

$procs = @()

function Start-Shiro {
    param([string]$Tag, [string]$Module)
    Write-Host "[start] launching $Tag ($Module)" -ForegroundColor Cyan
    Start-Process -FilePath $py -ArgumentList "-m", $Module -WorkingDirectory $root -PassThru
}

# 1. P1: L0 + L1 + L2
$procs += Start-Shiro -Tag "P1" -Module "src.runtime.process_1_fast"
Start-Sleep -Milliseconds 800

# 2. P3: L3 + L7 + L8 (currently only L7)
if (-not $NoP3) {
    $procs += Start-Shiro -Tag "P3" -Module "src.runtime.process_3_life"
    Start-Sleep -Milliseconds 500
}

# 3. P2: L2 + L4 + L5 (currently only L5)
if (-not $NoP2) {
    $procs += Start-Shiro -Tag "P2" -Module "src.runtime.process_2_mind"
}

Write-Host ""
Write-Host "[start] launched $($procs.Count) processes:" -ForegroundColor Green
foreach ($p in $procs) {
    Write-Host ("  PID {0}" -f $p.Id)
}
Write-Host ""
Write-Host "[start] stop with scripts\stop_all.ps1"
Write-Host "[start] observations: data\logs\observations.jsonl"
Write-Host "[start] shape:        data\state\*.pkl"
