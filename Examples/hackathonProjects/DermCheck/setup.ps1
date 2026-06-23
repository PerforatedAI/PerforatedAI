# GuardianEdge Windows Setup Script (PowerShell)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "DermCheck Windows Setup Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PerforatedAI from parent directory
Write-Host "Installing PerforatedAI..." -ForegroundColor Yellow
pip install -e ../../..

# Install project requirements
Write-Host "Installing project dependencies (including MONAI)..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "To start training:" -ForegroundColor Cyan
Write-Host "  python train_derm.py --task both --epochs 20" -ForegroundColor Gray
Write-Host ""
Write-Host "To run demo:" -ForegroundColor Cyan
Write-Host "  python demo_integrated.py --source path/to/images/" -ForegroundColor Gray
Write-Host ""
