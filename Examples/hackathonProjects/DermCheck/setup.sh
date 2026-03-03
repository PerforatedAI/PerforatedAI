#!/bin/bash

# DermCheck Setup Script for Linux/Mac
# Automated setup for PerforatedAI + MONAI hackathon project

echo "🏥 Initializing DermCheck Setup..."

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PerforatedAI from root directory
echo "🚀 Installing PerforatedAI..."
pip install -e ../../..

# 5. Install project dependencies
echo "📥 Installing MONAI and other requirements..."
pip install -r requirements.txt

echo "------------------------------------------------"
echo "✅ Setup Complete!"
echo ""
echo "To start working:"
echo "  source venv/bin/activate"
echo "  python train_derm.py --demo # Run a quick demo training"
echo "------------------------------------------------"
