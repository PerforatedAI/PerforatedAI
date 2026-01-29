#!/bin/bash

# Experiment runner script for dendritic vs vanilla comparison
# Usage: ./run_experiment.sh [quick|full|final] [--no-wandb]
#
# Modes:
#   quick - 3 epochs, 2 layers, small models (128/64 dim)
#   full  - 10 epochs, 2 layers, full models (256/128 dim)
#   final - 30 epochs, 3 layers, full models (256/128 dim) - COMPREHENSIVE

set -e

MODE=${1:-quick}
WANDB_FLAG=""
if [[ "$*" == *"--no-wandb"* ]]; then
    WANDB_FLAG="--no_wandb"
    echo "Running without W&B logging"
fi

echo "========================================================================"
echo "  DENDRITIC VS VANILLA TRANSFORMER EXPERIMENT"
echo "========================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Install PerforatedAI if needed
if ! python -c "import perforatedai" &> /dev/null; then
    echo "Installing PerforatedAI..."
    cd ../../..  # Go from Examples/baseExamples/transformer to project root
    pip install -e .
    cd Examples/baseExamples/transformer  # Return to example directory
    echo "✓ PerforatedAI installed"
fi

# Run setup test
echo ""
echo "========================================================================"
echo "  RUNNING SETUP TEST"
echo "========================================================================"
echo ""
python test_setup.py

# Set experiment parameters based on mode
if [ "$MODE" = "quick" ]; then
    EPOCHS=3
    BATCH_SIZE=32
    EMBED_DIM_VANILLA=128
    EMBED_DIM_DENDRITIC=64
    NUM_LAYERS=2
    echo ""
    echo "Running QUICK experiment (3 epochs, 2 layers, smaller models)..."
elif [ "$MODE" = "full" ]; then
    EPOCHS=10
    BATCH_SIZE=32
    EMBED_DIM_VANILLA=256
    EMBED_DIM_DENDRITIC=128
    NUM_LAYERS=2
    echo ""
    echo "Running FULL experiment (10 epochs, 2 layers, full-size models)..."
elif [ "$MODE" = "final" ]; then
    EPOCHS=30
    BATCH_SIZE=32
    EMBED_DIM_VANILLA=128
    EMBED_DIM_DENDRITIC=64
    NUM_LAYERS=2
    echo ""
    echo "Running FINAL experiment (30 epochs, 3 layers, full comparison)..."
else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run_experiment.sh [quick|full|final] [--no-wandb]"
    exit 1
fi

# Create results directory
mkdir -p results

echo ""
echo "========================================================================"
echo "  TRAINING VANILLA MODEL"
echo "========================================================================"
echo ""
python train.py \
    --model_type vanilla \
    --epochs $EPOCHS \
    --embed_dim $EMBED_DIM_VANILLA \
    --batch_size $BATCH_SIZE \
    --num_layers $NUM_LAYERS \
    --learning_rate 0.001 \
    $WANDB_FLAG

echo ""
echo "========================================================================"
echo "  TRAINING DENDRITIC MODEL"
echo "========================================================================"
echo ""
python train.py \
    --model_type dendritic \
    --epochs $EPOCHS \
    --embed_dim $EMBED_DIM_DENDRITIC \
    --batch_size $BATCH_SIZE \
    --num_layers $NUM_LAYERS \
    --learning_rate 0.001 \
    $WANDB_FLAG

echo ""
echo "========================================================================"
echo "  EXPERIMENT COMPLETE"
echo "========================================================================"
echo ""
echo "✓ Both models trained successfully!"
echo ""
echo "To analyze results, run:"
echo "  python analyze_results.py --project dendritic-transformer-comparison"
echo ""
echo "Or view results in Weights & Biases:"
echo "  https://wandb.ai"
echo ""

