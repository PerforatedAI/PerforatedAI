#!/bin/bash

echo "========================================="
echo "GuardianEdge Setup Script"
echo "========================================="
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PerforatedAI from parent directory
echo "Installing PerforatedAI..."
pip install -e ../../..

# Install project requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment:"
echo "  Linux/Mac: source venv/bin/activate"
echo "  Windows:   venv\\Scripts\\activate"
echo ""
echo "To start training:"
echo "  python train_guardian.py --data coco128.yaml --model yolov8n.pt --epochs 10"
echo ""
echo "To run inference:"
echo "  python inference.py --model models/best_model_pai.pt --source 0"
echo ""
