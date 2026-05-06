#!/bin/bash
# Launch PAI hyperparameter sweep with W&B

set -e

# Configuration
SWEEP_CONFIG=${1:-"ml/sweeps/sweep_pai_core.yaml"}
PROJECT=${WANDB_PROJECT:-"v0-pai-experiments"}
ENTITY=${WANDB_ENTITY:-""}
NUM_AGENTS=${2:-1}

echo "=========================================="
echo "Launching PAI Sweep"
echo "=========================================="
echo "Config: $SWEEP_CONFIG"
echo "Project: $PROJECT"
echo "Agents: $NUM_AGENTS"
echo ""

# Check W&B login
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set"
    echo "Run: wandb login"
fi

# Create sweep
echo "Creating sweep..."
if [ -n "$ENTITY" ]; then
    SWEEP_ID=$(wandb sweep --project "$PROJECT" --entity "$ENTITY" "$SWEEP_CONFIG" 2>&1 | grep "wandb agent" | awk '{print $NF}')
else
    SWEEP_ID=$(wandb sweep --project "$PROJECT" "$SWEEP_CONFIG" 2>&1 | grep "wandb agent" | awk '{print $NF}')
fi

echo "Sweep ID: $SWEEP_ID"

# Launch agents
echo "Launching $NUM_AGENTS agent(s)..."
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i..."
    if [ "$NUM_AGENTS" -gt 1 ]; then
        # Background for multiple agents
        wandb agent "$SWEEP_ID" &
    else
        # Foreground for single agent
        wandb agent "$SWEEP_ID"
    fi
done

if [ "$NUM_AGENTS" -gt 1 ]; then
    echo "Agents launched in background"
    echo "Monitor at: https://wandb.ai/$ENTITY/$PROJECT/sweeps"
    wait
fi

echo "Sweep complete!"
