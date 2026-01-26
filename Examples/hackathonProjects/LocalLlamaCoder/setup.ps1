# LocalLlama Coder Setup Script for Windows
Write-Host "🦙 Initializing LocalLlama Coder Setup..." -ForegroundColor Cyan

# 1. Create virtual environment
if (-Not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "✅ Virtual environment already exists." -ForegroundColor Green
}

# 2. Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 4. Install PerforatedAI from root directory
Write-Host "🚀 Installing PerforatedAI..." -ForegroundColor Yellow
pip install -e ..\..\..

# 5. Install project dependencies
Write-Host "📥 Installing Transformers and dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "------------------------------------------------" -ForegroundColor Green
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "" 
Write-Host "To start working:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python finetune_llama.py --demo  # Quick demo"
Write-Host "  python inference_code.py --prompt 'def hello():' # Test inference"
Write-Host "------------------------------------------------" -ForegroundColor Green
