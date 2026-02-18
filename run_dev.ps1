$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $repoRoot "frontend"

Write-Host "Starting fullstack dev environment..."
Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"

$backend = Start-Process -FilePath "python" -ArgumentList "-m uvicorn silo_blend.api.app:app --host 127.0.0.1 --port 8000 --reload" -WorkingDirectory $repoRoot -PassThru
$frontend = Start-Process -FilePath "npm" -ArgumentList "run dev -- --host 127.0.0.1 --port 5173" -WorkingDirectory $frontendDir -PassThru

Write-Host "Backend PID: $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Press Ctrl+C to stop both processes."

try {
  while (-not $backend.HasExited -and -not $frontend.HasExited) {
    Start-Sleep -Seconds 1
    $backend.Refresh()
    $frontend.Refresh()
  }
}
finally {
  if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
  if (-not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force }
}
