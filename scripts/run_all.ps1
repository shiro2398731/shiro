"""
scripts/run_all.ps1

Start the four processes. Run this yourself.

Usage:
    $env:DEEPSEEK_API_KEY = "<your key>"
    .\\scripts\\run_all.ps1

Each process opens its own window so you can watch them separately.
P4 is the one you type into.
"""

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "[warn] DEEPSEEK_API_KEY is not set." -ForegroundColor Yellow
    Write-Host "       The encoder will fall back to hashing, and output will" -ForegroundColor Yellow
    Write-Host "       fall back to templates. The pipeline still runs." -ForegroundColor Yellow
}

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[error] venv not found at $py" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $root "data\input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "data\logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "data\output") | Out-Null

$mods = @(
    @{ name = "P1 fast (L0+L1+L2)"; mod = "src.runtime.process_1_fast" },
    @{ name = "P2 mind (L5)";      mod = "src.runtime.process_2_mind" },
    @{ name = "P3 life (L3+L4)";   mod = "src.runtime.process_3_life" },
    @{ name = "P4 voice (L6)";     mod = "src.runtime.process_4_voice" }
)

foreach ($m in $mods) {
    Write-Host ("[start] " + $m.name)
    Start-Process -FilePath $py `
        -ArgumentList @("-m", $m.mod) `
        -WorkingDirectory $root
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "All started. P4 is the window to type into." -ForegroundColor Green
Write-Host "Or append a line to: data\input\inbox.txt" -ForegroundColor Green
