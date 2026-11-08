# scripts/verify.ps1
#
# One command to run a timed observation session.
#
#   scripts\verify.ps1                 # stop, start, watch 60 min, stop
#   scripts\verify.ps1 -Minutes 15     # shorter session
#   scripts\verify.ps1 -KeepRunning    # leave the processes running afterwards
#
# It only chains what is already there: stop_all.ps1 -> start_all.ps1 ->
# tools\watch.py -> stop_all.ps1. Nothing new is decided here.

param(
    [int]$Minutes = 60,
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$ps = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $ps)) {
    Write-Host "[verify] missing .venv\Scripts\python.exe" -ForegroundColor Red
    exit 1
}

$stopScript  = Join-Path $PSScriptRoot "stop_all.ps1"
$startScript = Join-Path $PSScriptRoot "start_all.ps1"

Write-Host "=== 1/4 stop ===" -ForegroundColor Cyan
& $stopScript | Out-Host

Write-Host "=== 2/4 start ===" -ForegroundColor Cyan
& $startScript | Out-Host
Start-Sleep -Seconds 8

Write-Host "=== 3/4 watch ($Minutes min) ===" -ForegroundColor Cyan
Write-Host "    Ctrl+C gives the summary early." -ForegroundColor DarkGray
$duration = $Minutes * 60
& $ps (Join-Path $root "tools\watch.py") --window 180 --duration $duration | Out-Host

Write-Host "=== 4/4 result ===" -ForegroundColor Cyan
$latest = Get-ChildItem (Join-Path $root "data\logs") -Filter "watch_*.md" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) {
    Write-Host ("report: " + $latest.FullName) -ForegroundColor Green
    Get-Content $latest.FullName | Out-Host
} else {
    Write-Host "[verify] no report file was produced" -ForegroundColor Yellow
}

if ($KeepRunning) {
    Write-Host "[verify] processes left running (use scripts\stop_all.ps1 to stop)" -ForegroundColor Yellow
} else {
    Write-Host "=== stop ===" -ForegroundColor Cyan
    & $stopScript | Out-Host
}