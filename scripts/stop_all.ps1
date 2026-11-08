# scripts/stop_all.ps1
#
# Stop the shiro processes.
#
# 2026-11-08 fix: .venv\Scripts\python.exe is a launcher shim on this machine.
# It spawns the real interpreter as a CHILD process, and both carry the module
# name in their command line. Killing the launcher alone left the child alive,
# still holding ports 5560/5561/5562, which then made the next start_all
# crash-loop on bind. We now kill the whole tree with taskkill /F /T.
#
# Stop-Process is forceful; shape is saved every 5 minutes and on clean exit,
# so a forced kill may lose the last few minutes of shape changes but never
# corrupts what is already on disk.

param([switch]$Force)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[stop] looking for shiro processes..."

function Get-Shiro {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "src\.runtime\.process_[123]_" }
}

$targets = Get-Shiro
if (-not $targets) {
    Write-Host "[stop] nothing running"
    exit 0
}

# Only kill tree roots: a target whose parent is not itself a target.
$pids = $targets.ProcessId
$roots = $targets | Where-Object { $pids -notcontains $_.ParentProcessId }

foreach ($t in $targets) {
    $m = [regex]::Match($t.CommandLine, "process_([123])_")
    $tag = if ($m.Success) { "P" + $m.Groups[1].Value } else { "?" }
    Write-Host ("  {0}  PID {1}  (parent {2})" -f $tag, $t.ProcessId, $t.ParentProcessId)
}

Write-Host "[stop] killing trees..."
foreach ($t in $roots) {
    & taskkill /F /T /PID $t.ProcessId 2>&1 | Out-Null
}

Start-Sleep -Seconds 3

# Verify, and retry once on whatever survived.
$survivors = Get-Shiro
if ($survivors) {
    Write-Host ("[stop] {0} survived, retrying" -f $survivors.Count) -ForegroundColor Yellow
    foreach ($s in $survivors) { & taskkill /F /T /PID $s.ProcessId 2>&1 | Out-Null }
    Start-Sleep -Seconds 2
    $survivors = Get-Shiro
}

if ($survivors) {
    Write-Host ("[stop] FAILED: {0} still alive: {1}" -f $survivors.Count,
        (($survivors | ForEach-Object { $_.ProcessId }) -join ", ")) -ForegroundColor Red
    exit 1
}

Write-Host "[stop] all processes confirmed gone" -ForegroundColor Green
Write-Host "[stop] last saved shape:"
Get-ChildItem "data\state" -Filter *.pkl -ErrorAction SilentlyContinue |
    Select-Object Name, LastWriteTime | Format-Table -AutoSize