#!/usr/bin/env python
"""
Compare Baseline vs Dendritic Model Results

This script helps analyze and compare the results from baseline
and dendritic training runs for the hackathon submission.

Usage:
    python -m src.training.compare_results
"""

import os
import json
import argparse
from datetime import datetime


def calculate_remaining_error_reduction(baseline_loss: float, dendritic_loss: float) -> float:
    """
    Calculate the Remaining Error Reduction (RER) percentage.
    
    RER = (baseline_error - dendritic_error) / baseline_error * 100
    
    For loss-based metrics (lower is better):
    If baseline_loss = 0.40 and dendritic_loss = 0.35
    RER = (0.40 - 0.35) / 0.40 * 100 = 12.5%
    
    This means dendrites eliminated 12.5% of the baseline error.
    """
    if baseline_loss <= 0:
        return 0.0
    
    reduction = (baseline_loss - dendritic_loss) / baseline_loss * 100
    return max(0, reduction)  # Can't have negative reduction


def calculate_accuracy_rer(baseline_acc: float, dendritic_acc: float) -> float:
    """
    Calculate Remaining Error Reduction for accuracy metrics.
    
    For accuracy (higher is better):
    If baseline_acc = 90% and dendritic_acc = 92%
    Baseline error = 10%, Dendritic error = 8%
    RER = (10 - 8) / 10 * 100 = 20%
    
    This means dendrites eliminated 20% of the remaining errors.
    """
    baseline_error = 100 - baseline_acc
    dendritic_error = 100 - dendritic_acc
    
    if baseline_error <= 0:
        return 0.0
    
    reduction = (baseline_error - dendritic_error) / baseline_error * 100
    return max(0, reduction)


def format_results_table(results: dict) -> str:
    """Format results as a markdown table."""
    table = """
| Metric | Baseline | Dendritic | Improvement |
|--------|----------|-----------|-------------|
"""
    
    for metric, values in results.items():
        baseline = values.get('baseline', 'N/A')
        dendritic = values.get('dendritic', 'N/A')
        improvement = values.get('improvement', 'N/A')
        
        if isinstance(baseline, float):
            baseline = f"{baseline:.4f}"
        if isinstance(dendritic, float):
            dendritic = f"{dendritic:.4f}"
        if isinstance(improvement, float):
            improvement = f"{improvement:.2f}%"
            
        table += f"| {metric} | {baseline} | {dendritic} | {improvement} |\n"
    
    return table


def generate_hackathon_readme(
    baseline_loss: float,
    dendritic_loss: float,
    baseline_params: float,
    dendritic_params: float,
    dendrite_count: int,
    output_path: str = "RESULTS.md"
):
    """Generate a hackathon submission README with results."""
    
    rer = calculate_remaining_error_reduction(baseline_loss, dendritic_loss)
    param_overhead = ((dendritic_params - baseline_params) / baseline_params) * 100
    
    readme_content = f"""# Qwen2.5 + Dendritic Optimization - Hackathon Submission

## Project Overview

This project applies Perforated AI's Dendritic Optimization to Qwen2.5-1.5B-Instruct
for mathematical reasoning on the GSM8K dataset.

**Team:** [Your Team Name]
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## What is Dendritic Optimization?

Dendritic optimization adds artificial dendrites to neural networks, inspired by 
biological neuroscience research showing that dendrites perform additional computation
beyond what traditional artificial neurons model.

## Results

### Loss Comparison

| Model | Final Eval Loss | Parameters (M) | Notes |
|-------|-----------------|----------------|-------|
| Baseline (Qwen2.5-1.5B) | {baseline_loss:.4f} | {baseline_params:.2f}M | Standard fine-tuning |
| Dendritic (Qwen2.5-1.5B + PAI) | {dendritic_loss:.4f} | {dendritic_params:.2f}M | {dendrite_count} dendrite sets |

### Key Metrics

- **Remaining Error Reduction (RER):** {rer:.2f}%
- **Parameter Overhead:** {param_overhead:.2f}%
- **Dendrite Sets Added:** {dendrite_count}

### Interpretation

{"✅ Dendritic optimization improved performance!" if rer > 0 else "⚠️ No improvement observed - consider tuning hyperparameters"}

The Remaining Error Reduction of {rer:.2f}% means that dendritic optimization 
eliminated {rer:.2f}% of the error that remained after baseline training.

## Reproduction

### Requirements

```bash
pip install -e .
# or
uv sync
```

### Training Commands

**Baseline:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.training.train_baseline \\
    --learning_rate 5e-5 \\
    --num_train_epochs 3 \\
    --train_samples 500 \\
    --eval_samples 100
```

**Dendritic:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.training.train_dendritic \\
    --learning_rate 5e-5 \\
    --num_train_epochs 3 \\
    --num_dendrites 3 \\
    --train_samples 500 \\
    --eval_samples 100
```

## Files

- `src/training/train_baseline.py` - Baseline training script
- `src/training/train_dendritic.py` - Dendritic training script
- `src/models/baseline_model.py` - Baseline model wrapper
- `src/models/dendritic_model.py` - Dendritic model wrapper
- `config/sweep_baseline.yaml` - W&B sweep config for baseline
- `config/sweep_dendritic.yaml` - W&B sweep config for dendritic

## W&B Project

View training runs at: https://wandb.ai/[your-username]/qwen-dendritic-hackathon

## Acknowledgments

- [Perforated AI](https://www.perforatedai.com/) for the dendritic optimization library
- [Qwen Team](https://github.com/QwenLM/Qwen2.5) for the base model
- [Weights & Biases](https://wandb.ai/) for experiment tracking
"""
    
    with open(output_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Results README saved to {output_path}")
    return readme_content


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs dendritic results")
    parser.add_argument("--baseline_loss", type=float, default=0.3905, help="Baseline eval loss")
    parser.add_argument("--dendritic_loss", type=float, default=None, help="Dendritic eval loss")
    parser.add_argument("--baseline_params", type=float, default=1543.71, help="Baseline params (M)")
    parser.add_argument("--dendritic_params", type=float, default=None, help="Dendritic params (M)")
    parser.add_argument("--dendrite_count", type=int, default=0, help="Number of dendrite sets added")
    parser.add_argument("--output", type=str, default="RESULTS.md", help="Output file path")
    
    args = parser.parse_args()
    
    # If dendritic results not provided, show placeholder
    if args.dendritic_loss is None:
        print("⚠️ Dendritic loss not provided - using placeholder values")
        print("Run the dendritic training first, then update with actual values:")
        print("")
        print("  python -m src.training.compare_results \\")
        print("      --baseline_loss 0.3905 \\")
        print("      --dendritic_loss <YOUR_VALUE> \\")
        print("      --dendritic_params <YOUR_VALUE> \\")
        print("      --dendrite_count <YOUR_VALUE>")
        print("")
        args.dendritic_loss = args.baseline_loss  # Placeholder
        args.dendritic_params = args.baseline_params
    
    # Calculate metrics
    rer = calculate_remaining_error_reduction(args.baseline_loss, args.dendritic_loss)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Baseline Eval Loss:  {args.baseline_loss:.4f}")
    print(f"Dendritic Eval Loss: {args.dendritic_loss:.4f}")
    print(f"Remaining Error Reduction: {rer:.2f}%")
    print("="*60 + "\n")
    
    # Generate README
    generate_hackathon_readme(
        baseline_loss=args.baseline_loss,
        dendritic_loss=args.dendritic_loss,
        baseline_params=args.baseline_params,
        dendritic_params=args.dendritic_params or args.baseline_params,
        dendrite_count=args.dendrite_count,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
