# Entry point for hackathon demo

import argparse
import json
import os
import sys

# Ensure src is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_loader import DatasetLoader
from baseline_model import BaselineModel
from dendritic_wrapper import DendriticWrapper
from metrics_viewer import MetricsViewer


def load_static_metrics(results_dir):
    with open(os.path.join(results_dir, "baseline_metrics.json")) as f:
        baseline = json.load(f)

    with open(os.path.join(results_dir, "dendritic_metrics.json")) as f:
        dendritic = json.load(f)

    return baseline, dendritic


def main():
    parser = argparse.ArgumentParser(description="Perforated Dendritic Routing Demo")
    parser.add_argument(
        "--mode",
        choices=["live", "safe", "static"],
        default="safe",
        help="Execution mode",
    )
    args = parser.parse_args()

    print("\n=== Perforated Dendritic Routing Demo ===")
    print(f"Mode: {args.mode}\n")

    results_dir = os.path.join(CURRENT_DIR, "results")

    if args.mode == "static":
        baseline_metrics, dendritic_metrics = load_static_metrics(results_dir)
        viewer = MetricsViewer()
        viewer.show(baseline_metrics, dendritic_metrics)
        return

    loader = DatasetLoader(max_samples=16)
    data = loader.load()
    batches = data["batches"]

    baseline_model = BaselineModel()
    baseline_metrics = baseline_model.evaluate(batches)

    dendritic = DendriticWrapper(
        enabled=True,
        num_branches=4,
        safety_fallback=True,
    )

    dendritic_result = dendritic.run(
        batches=batches,
        baseline_model=baseline_model,
        mode="eval",
    )

    dendritic_metrics = dendritic_result["metrics"]

    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    with open(os.path.join(results_dir, "dendritic_metrics.json"), "w") as f:
        json.dump(dendritic_metrics, f, indent=2)

    viewer = MetricsViewer()
    viewer.show(baseline_metrics, dendritic_metrics)

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
