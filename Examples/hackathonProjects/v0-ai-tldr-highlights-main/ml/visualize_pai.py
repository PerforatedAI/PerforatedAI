"""
PAI Visualization Scripts

Converts PAI tracker outputs into readable visualizations:
- Dendrite activation heatmaps
- Restructuring event timelines
- Parameter efficiency curves

Usage:
    python -m ml.visualize_pai --artifact-dir ./artifacts/compressed_pai/...
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Training Curves",
):
    """
    Plot training loss and evaluation score curves.
    
    Args:
        history: List of {step, loss, eval_score} dicts
        output_path: Where to save the plot
        title: Plot title
    """
    if not history:
        logger.warning("No history data to plot")
        return
    
    steps = [h["step"] for h in history]
    losses = [h.get("loss", 0) for h in history]
    scores = [h.get("eval_score", 0) for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Loss curve
    ax1.plot(steps, losses, 'b-', linewidth=2, label='Loss')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score curve
    ax2.plot(steps, scores, 'g-', linewidth=2, label='Eval Score')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {output_path}")


def plot_restructure_events(
    events: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Dendritic Restructuring Events",
):
    """
    Visualize when restructuring events occurred during training.
    
    Args:
        events: List of {step, type, details} dicts
        output_path: Where to save the plot
        title: Plot title
    """
    if not events:
        logger.warning("No restructure events to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    steps = [e["step"] for e in events]
    types = [e.get("type", "unknown") for e in events]
    
    # Color by event type
    colors = {
        "restructure": "red",
        "prune": "orange",
        "grow": "green",
        "unknown": "gray",
    }
    
    for step, etype in zip(steps, types):
        color = colors.get(etype, "gray")
        ax.axvline(x=step, color=color, alpha=0.7, linewidth=2)
    
    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in colors.items() if t in types]
    ax.legend(handles=patches, loc='upper right')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved restructure events to {output_path}")


def plot_parameter_efficiency(
    baseline_params: int,
    compressed_params: int,
    pai_active_params: int,
    metrics: Dict[str, float],
    output_path: Path,
):
    """
    Plot parameter counts vs. performance metrics.
    
    Args:
        baseline_params: Parameter count for baseline model
        compressed_params: Parameter count for compressed model
        pai_active_params: Active parameters after PAI optimization
        metrics: Dict with rouge_l, bertscore, etc. for each config
        output_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of parameter counts
    configs = ['Baseline', 'Compressed', 'PAI Active']
    params = [baseline_params / 1e6, compressed_params / 1e6, pai_active_params / 1e6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(configs, params, color=colors)
    ax1.set_ylabel('Parameters (M)', fontsize=12)
    ax1.set_title('Parameter Counts', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}M', ha='center', va='bottom', fontsize=10)
    
    # Scatter plot of params vs. performance
    if metrics:
        ax2.scatter(params[0], metrics.get('baseline_rouge', 0), 
                   s=200, c='#1f77b4', label='Baseline', marker='o')
        ax2.scatter(params[1], metrics.get('compressed_rouge', 0),
                   s=200, c='#ff7f0e', label='Compressed', marker='s')
        ax2.scatter(params[2], metrics.get('pai_rouge', 0),
                   s=200, c='#2ca02c', label='PAI', marker='^')
        
        ax2.set_xlabel('Parameters (M)', fontsize=12)
        ax2.set_ylabel('ROUGE-L', fontsize=12)
        ax2.set_title('Efficiency Frontier', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved parameter efficiency plot to {output_path}")


def plot_dendrite_heatmap(
    activations: np.ndarray,
    layer_names: List[str],
    output_path: Path,
    title: str = "Dendrite Activation Heatmap",
):
    """
    Plot heatmap of dendrite activations across layers.
    
    Args:
        activations: 2D array of shape (layers, dendrites)
        layer_names: Names for each layer
        output_path: Where to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(activations, aspect='auto', cmap='viridis')
    
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel('Dendrite Index', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Activation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved dendrite heatmap to {output_path}")


def generate_comparison_table(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """
    Generate markdown comparison table from experiment results.
    
    Args:
        results: Dict mapping experiment name to metrics
        output_path: Where to save the markdown table
    """
    headers = ["Experiment", "Params (M)", "ROUGE-L", "BERTScore", "HP@3", "Latency (ms)"]
    
    rows = []
    for exp_name, metrics in results.items():
        row = [
            exp_name,
            f"{metrics.get('total_parameters', 0) / 1e6:.1f}",
            f"{metrics.get('rougeL', 0):.4f}",
            f"{metrics.get('bertscore_f1', 0):.4f}",
            f"{metrics.get('highlight_precision_at_3', 0):.4f}",
            f"{metrics.get('latency_ms', 0):.1f}",
        ]
        rows.append(row)
    
    # Generate markdown
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(row) + " |\n"
    
    with open(output_path, "w") as f:
        f.write(md)
    
    logger.info(f"Saved comparison table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PAI Visualization")
    parser.add_argument("--artifact-dir", type=str, required=True,
                       help="Directory containing PAI artifacts")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots")
    args = parser.parse_args()
    
    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir) if args.output_dir else artifact_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training history if available
    history_path = artifact_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, output_dir / "training_curves.png")
    
    # Load restructure events if available
    events_path = artifact_dir / "restructure_events.json"
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)
        plot_restructure_events(events, output_dir / "restructure_events.png")
    
    # Load eval results if available
    eval_path = artifact_dir / "eval_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results = json.load(f)
        generate_comparison_table({"Current": results}, output_dir / "comparison.md")
    
    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
