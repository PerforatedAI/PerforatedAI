# GuardianEdge Windows Setup Script (PowerShell)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "GuardianEdge Windows Setup Script" -ForegroundColor Cyan
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
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
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
Write-Host "  python train_guardian.py --data coco128.yaml --model yolov8n.pt --epochs 10" -ForegroundColor Gray
Write-Host ""
Write-Host "To run inference:" -ForegroundColor Cyan
Write-Host "  python inference.py --model models/best_model_pai.pt --source 0" -ForegroundColor Gray
Write-Host ""
