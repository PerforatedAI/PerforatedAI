#!/usr/bin/env bash
#
# Launch W&B sweep with multiple agents
#
# Usage:
#   ./ml/scripts/launch_sweep.sh <SWEEP_ID> [NUM_AGENTS] [PROJECT] [ENTITY]
#
# Example:
#   ./ml/scripts/launch_sweep.sh abc123 4 v0-ai-tldr-highlights lowlucy
#
# SECURITY: Set WANDB_API_KEY in environment before running:
#   export WANDB_API_KEY="<YOUR_KEY>"

set -e

SWEEP_ID=$1
AGENTS=${2:-4}
PROJECT=${3:-"v0-ai-tldr-highlights"}
ENTITY=${4:-"lowlucy"}

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 <SWEEP_ID> [NUM_AGENTS] [PROJECT] [ENTITY]"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not set"
    echo "Set it with: export WANDB_API_KEY=\"<YOUR_KEY>\""
    exit 1
fi

echo "Starting $AGENTS agents for sweep $SWEEP_ID"
echo "Project: $PROJECT, Entity: $ENTITY"

# Start agents in background
for i in $(seq 1 $AGENTS); do
    echo "Starting agent $i..."
    wandb agent --count 10 "$ENTITY/$PROJECT/$SWEEP_ID" &
    sleep 2
done

echo "All agents started. Use 'jobs' to see running agents."
echo "Use 'pkill -f wandb' to stop all agents."

# Wait for all background jobs
wait

echo "All agents completed."
