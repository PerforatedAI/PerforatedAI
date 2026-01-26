"""
WandB Sweep Launcher for ResNet18 + PAI Experiments
Run this script to create and execute a WandB sweep.

Usage:
    python run_sweep.py                  # Create new sweep and run 30 agents
    python run_sweep.py --count 10       # Run 10 sweep agents
    python run_sweep.py --sweep_id <id>  # Resume existing sweep
"""

import argparse
import os
import yaml
import wandb

# Project settings
PROJECT_NAME = "cifar100-dendritic-demo"
SWEEP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sweep_config.yaml")


def load_sweep_config():
    """Load sweep configuration from YAML file."""
    with open(SWEEP_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def create_sweep():
    """Create a new WandB sweep and return the sweep ID."""
    config = load_sweep_config()
    sweep_id = wandb.sweep(config, project=PROJECT_NAME)
    print(f"\n{'='*60}")
    print(f"Created new sweep: {sweep_id}")
    print(f"Project: {PROJECT_NAME}")
    print(f"View at: https://wandb.ai/{wandb.api.default_entity}/{PROJECT_NAME}/sweeps/{sweep_id}")
    print(f"{'='*60}\n")
    return sweep_id


def run_agent(sweep_id, count):
    """Run wandb sweep agent."""
    print(f"\nStarting {count} sweep agent(s) for sweep: {sweep_id}")
    wandb.agent(sweep_id, project=PROJECT_NAME, count=count)


def main():
    parser = argparse.ArgumentParser(description="Run WandB sweep for ResNet18+PAI")
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Existing sweep ID to resume (creates new if not provided)")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of sweep runs to execute (default: 10)")
    parser.add_argument("--create_only", action="store_true",
                        help="Only create the sweep, don't run agents")
    
    args = parser.parse_args()
    
    # Login check
    if not wandb.api.api_key:
        print("Please login to WandB first: wandb login")
        return
    
    # Create or use existing sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        sweep_id = create_sweep()
    
    # Save sweep ID for reference
    sweep_id_file = os.path.join(os.path.dirname(__file__), ".last_sweep_id")
    with open(sweep_id_file, 'w') as f:
        f.write(sweep_id)
    print(f"Sweep ID saved to: {sweep_id_file}")
    
    # Run agents
    if not args.create_only:
        run_agent(sweep_id, args.count)
        
        # Print final summary
        print("\n" + "="*60)
        print("SWEEP COMPLETE")
        print("="*60)
        print(f"View results: https://wandb.ai/{wandb.api.default_entity}/{PROJECT_NAME}/sweeps/{sweep_id}")
        print("\nTo find best run:")
        print("  1. Go to the sweep page above")
        print("  2. Sort by 'efficiency_score' (ascending)")
        print("  3. The top run has the best parameter/accuracy tradeoff")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
