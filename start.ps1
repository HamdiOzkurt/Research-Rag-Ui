# AI Research Assistant - PowerShell Starter
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Research Assistant" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Paths
$projectRoot = $PSScriptRoot
$venvPath = Join-Path (Split-Path $projectRoot -Parent) "venv\Scripts\Activate.ps1"
$frontendPath = Join-Path $projectRoot "copilotkit-ui"

Write-Host "Backend baslatiliyor - Port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$venvPath'; cd '$projectRoot'; python -m uvicorn src.simple_copilot_backend:app --reload --port 8000"

Start-Sleep -Seconds 3

Write-Host "Frontend baslatiliyor - Port 3000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; npm run dev"

Write-Host ""
Write-Host "Servisler baslatildi!" -ForegroundColor Green
Write-Host ""
Write-Host "Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Read-Host "Enter'a basin"
