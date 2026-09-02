# One-command demo for Windows PowerShell.
Set-Location "$PSScriptRoot\backend"
python -m entitygrid.sim.generate
if (-not $?) { exit 1 }
python -m entitygrid.pipeline
if (-not $?) { exit 1 }
Write-Host ""
Write-Host "  landing page  ->  http://127.0.0.1:8000/"
Write-Host "  live console  ->  http://127.0.0.1:8000/dashboard"
Write-Host ""
python -m uvicorn entitygrid.api.main:app --port 8000
