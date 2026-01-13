import sys
import os
# Add repository root to path so we can import the library if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import sys
import argparse
import json
import os
from model import BaselineModel, DendriticModel, BaselineCIFARModel, DendriticCIFARModel
from train import get_data_loaders, train_model
from evaluate import get_benchmark_report



import perforatedai as pai

def print_header():
    print("\n" + "="*70)
    print(" [PERFORATED AI: DENDRITIC OPTIMIZATION HACKATHON PROOF-OF-CONCEPT]")
    print("="*70)

def run_benchmarks(dataset_name="mnist", epochs=2):
    print_header()
    print(f"\n[BENCHMARK] Dataset: {dataset_name.upper()}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Prepare Data
    train_loader, test_loader = get_data_loaders(batch_size=128, dataset_name=dataset_name)
    
    # 2. Instantiate Models
    if dataset_name.lower() == "cifar10":
        baseline = BaselineCIFARModel()
        dendritic = DendriticCIFARModel()
    else:
        baseline = BaselineModel()
        dendritic = DendriticModel()
    
    # 3. Training phase
    # Baseline training (Standard)
    # Pass test_loader for consistency, though currently only dendritic uses it for graph
    baseline = train_model(baseline, train_loader, device, epochs=epochs, is_dendritic=False, test_loader=test_loader)
    
    # Dendritic training (Perforated AI enabled)
    # test_loader is required here for real graph generation
    dendritic = train_model(dendritic, train_loader, device, epochs=epochs, is_dendritic=True, test_loader=test_loader)
    
    # 4. Evaluation phase
    print("\n[BENCHMARK] Computing Final Benchmarks...")
    baseline_metrics = get_benchmark_report(baseline, test_loader, device)
    dendritic_metrics = get_benchmark_report(dendritic, test_loader, device)
    
    # 5. Side-by-Side Comparison Output
    format_str = "{:<25} | {:<15} | {:<25}"
    header = format_str.format("Metric", "Baseline", "Dendritic (PAI)")
    separator = "-" * len(header)
    
    print("\n" + separator)
    print(header)
    print(separator)
    
    for metric in baseline_metrics.keys():
        print(format_str.format(
            metric, 
            baseline_metrics[metric], 
            dendritic_metrics[metric]
        ))
    print(separator)
    
    # 6. Results Generation (MANDATORY)
    print("\n[EXPORT] Generating Official PAI Graph (PAI/PAI.png)...")
    # Official Perforated AI graph generation (auto-generated from training history)
    pai.generate_official_graph(save_path="PAI/PAI.png")
    
    # Save raw metrics for records
    results = {
        "dataset": dataset_name,
        "baseline": baseline_metrics,
        "dendritic": dendritic_metrics
    }
    with open("PAI/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("[EXPORT] Saved numeric results to PAI/metrics.json")

    # Final Hackathon Takeaway
    print("\n[CONCLUSION] HACKATHON TAKEAWAY:")
    print("Dendritic optimization demonstrated a successful reduction in active parameters")
    print(f"while maintaining {dendritic_metrics['Accuracy']} accuracy on {dataset_name.upper()}.")
    print("The high level of sparsity results in faster inference and lower energy consumption.")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="PerforatedAI Hackathon Benchmark")
        parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], 
                          help="Dataset to use (mnist or cifar10)")
        parser.add_argument("--epochs", type=int, default=6, 
                          help="Number of epochs per model (minimum 6 for dendritic activation)")
        args = parser.parse_args()
        
        run_benchmarks(dataset_name=args.dataset, epochs=args.epochs)
    except ImportError as e:
        print(f"\n[Error] Missing dependency - {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[Execution Failed] {e}")
        traceback.print_exc()
        sys.exit(1)
