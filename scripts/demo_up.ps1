<#
.SYNOPSIS
  Start the whole system for a demo with one command: production frontend build,
  API server, and (optionally) a public HTTPS link.

.EXAMPLE
  .\scripts\demo_up.ps1            # http://localhost:4173 on this machine
  .\scripts\demo_up.ps1 -Public    # also a temporary https://*.trycloudflare.com link
  .\scripts\demo_up.ps1 -Stop      # stop everything this script started

.NOTES
  - MySQL (service MySQL80) must already be running.
  - The frontend is a production build (vite build) served by vite preview,
    which proxies /api to the API server on port 8000 (see frontend/vite.config.js).
  - -Public uses a Cloudflare quick tunnel: no account, a new URL every run, and
    the link dies when the tunnel stops. Google sign-in will not work through it
    (its origin is not registered); email login and the demo personas do.
#>
param(
    [switch]$Public,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root "logs"
$pidFile = Join-Path $logs "demo_pids.json"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Stop-Demo {
    if (-not (Test-Path $pidFile)) { Write-Host "Nothing to stop."; return }
    $pids = Get-Content $pidFile | ConvertFrom-Json
    foreach ($name in $pids.PSObject.Properties.Name) {
        $id = $pids.$name
        # /T kills the child processes too (npm.cmd -> node, python -> workers).
        & taskkill /PID $id /T /F 2>$null | Out-Null
        Write-Host "stopped $name (pid $id)"
    }
    Remove-Item $pidFile
}

if ($Stop) { Stop-Demo; exit 0 }
if (Test-Path $pidFile) { Write-Host "A previous demo is still recorded; stopping it first."; Stop-Demo }

# 1. MySQL
$mysql = Get-Service -Name "MySQL80" -ErrorAction SilentlyContinue
if ($null -eq $mysql -or $mysql.Status -ne "Running") {
    throw "MySQL80 is not running. Start it (services.msc, or 'net start MySQL80' as admin) and rerun."
}
Write-Host "[1/4] MySQL80 is running"

# 2. Frontend production build
Write-Host "[2/4] Building the frontend..."
Push-Location (Join-Path $root "frontend")
try {
    & npm.cmd run build | Out-File (Join-Path $logs "demo_build.log")
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed; see logs\demo_build.log" }
} finally { Pop-Location }

# 3. API server (no --reload: one process, so embedded Qdrant's single-process lock holds)
$python = Join-Path $root ".venv\Scripts\python.exe"
$api = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000" `
    -RedirectStandardOutput (Join-Path $logs "demo_api.log") -RedirectStandardError (Join-Path $logs "demo_api.err.log")
$pids = [ordered]@{ api = $api.Id }
$pids | ConvertTo-Json | Set-Content $pidFile

Write-Host "[3/4] Starting the API (loads SBERT, the catalogue and Qdrant; about 40 s)..."
$deadline = (Get-Date).AddSeconds(180)
$ready = $false
while ((Get-Date) -lt $deadline) {
    if ($api.HasExited) { throw "The API exited during startup; see logs\demo_api.err.log" }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3
        $ready = $true; break
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { Stop-Demo; throw "The API did not become healthy within 180 s; see logs\demo_api.err.log" }
Write-Host "      API healthy: $($health | ConvertTo-Json -Compress)"

# 4. Frontend server
$web = Start-Process -FilePath "npm.cmd" -WorkingDirectory (Join-Path $root "frontend") -WindowStyle Hidden -PassThru `
    -ArgumentList "run", "preview", "--", "--port", "4173", "--strictPort" `
    -RedirectStandardOutput (Join-Path $logs "demo_web.log") -RedirectStandardError (Join-Path $logs "demo_web.err.log")
$pids.web = $web.Id
$pids | ConvertTo-Json | Set-Content $pidFile

$deadline = (Get-Date).AddSeconds(60)
$webReady = $false
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-WebRequest -Uri "http://localhost:4173/api/health" -UseBasicParsing -TimeoutSec 3 | Out-Null
        $webReady = $true; break
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $webReady) { Stop-Demo; throw "The frontend did not answer within 60 s; see logs\demo_web.err.log" }
Write-Host "[4/4] App: http://localhost:4173   (API docs: http://127.0.0.1:8000/docs)"

if ($Public) {
    $tunnelLog = Join-Path $logs "demo_tunnel.log"
    $tunnel = Start-Process -FilePath "cloudflared" -WindowStyle Hidden -PassThru `
        -ArgumentList "tunnel", "--url", "http://localhost:4173", "--no-autoupdate" `
        -RedirectStandardOutput (Join-Path $logs "demo_tunnel.out.log") -RedirectStandardError $tunnelLog
    $pids.tunnel = $tunnel.Id
    $pids | ConvertTo-Json | Set-Content $pidFile

    $url = $null
    $deadline = (Get-Date).AddSeconds(45)
    while (-not $url -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        if (Test-Path $tunnelLog) {
            $match = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($match) { $url = $match.Matches[0].Value }
        }
    }
    if ($url) { Write-Host "      Public link: $url" } else { Write-Host "      Tunnel started but no URL yet; see logs\demo_tunnel.log" }
}

Write-Host ""
Write-Host "Running. Stop everything with: .\scripts\demo_up.ps1 -Stop"
