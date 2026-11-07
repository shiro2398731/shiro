# scripts/stop_all.ps1
#
# Stop the shiro processes.
#
# Note: Stop-Process is forceful. The processes save their shape on a timer
# (every 5 minutes) and on clean exit; a forced kill may lose the last few
# minutes of shape changes, but never corrupts what is already on disk.

param(
    [switch]$Force
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[stop] looking for shiro processes..."

$targets = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match "src\.runtime\.process_[123]_" }

if (-not $targets) {
    Write-Host "[stop] nothing running"
    exit 0
}

foreach ($t in $targets) {
    $m = [regex]::Match($t.CommandLine, "process_([123])_")
    $tag = if ($m.Success) { "P" + $m.Groups[1].Value } else { "?" }
    Write-Host ("  {0}  PID {1}" -f $tag, $t.ProcessId)
}

Write-Host "[stop] stopping..."
foreach ($t in $targets) {
    try {
        Stop-Process -Id $t.ProcessId -ErrorAction SilentlyContinue
    } catch {}
}

Start-Sleep -Seconds 2
Write-Host "[stop] done. last saved shape time:"
Get-ChildItem "data\state" -Filter *.pkl -ErrorAction SilentlyContinue |
    Select-Object Name, LastWriteTime |
    Format-Table -AutoSize
