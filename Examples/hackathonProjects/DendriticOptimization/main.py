import sys
import os
import argparse
import json
import torch

# Add repository root to path (only if running locally without install)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from model import BaselineModel, BaselineCIFARModel
from train import get_data_loaders, train_model
from evaluate import get_benchmark_report

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


def print_header():
    print("\n" + "=" * 70)
    print(" [PERFORATED AI: DENDRITIC OPTIMIZATION HACKATHON PROOF-OF-CONCEPT]")
    print("=" * 70)


def run_benchmarks(dataset_name="mnist", epochs=6):
    print_header()
    print(f"\n[BENCHMARK] Dataset: {dataset_name.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_data_loaders(
        batch_size=128, dataset_name=dataset_name
    )

    # ---- MODEL INIT (BASELINE ONLY) ----
    if dataset_name.lower() == "cifar10":
        model = BaselineCIFARModel()
    else:
        model = BaselineModel()

    # ---- PAI CONVERSION (MANDATORY) ----
    model = UPA.initialize_pai(model, maximizing_score=True)
    model.to(device)

    # ---- PAI CONFIGURATION ----
    # Set switching logic based on epochs: switch mode after ~1/3 of total epochs
    GPA.pc.set_n_epochs_to_switch(max(1, epochs // 3))
    GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    GPA.pc.set_improvement_threshold(0.001) # Small threshold for POC
    GPA.pc.set_testing_dendrite_capacity(False) # Avoid pdb breakpoints

    # ---- OPTIMIZER SETUP VIA TRACKER ----
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    optimArgs = {"params": model.parameters(), "lr": 1e-3}
    schedArgs = {"mode": "max", "patience": 5}

    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optimArgs, schedArgs
    )

    # ---- TRAINING LOOP ----
    while True:
        train_acc = train_model(
            model, train_loader, device, optimizer=optimizer
        )

        val_metrics = get_benchmark_report(model, test_loader, device)
        # Convert string accuracy "98.5%" back to float 98.5 for the tracker
        val_score = float(val_metrics["Accuracy"].replace("%", ""))

        # Track test score for the benchmark record
        GPA.pai_tracker.add_test_score(val_score, 'Test Accuracy')

        model, restructured, training_complete = (
            GPA.pai_tracker.add_validation_score(val_score, model)
        )

        model.to(device)

        if training_complete:
            break

        if restructured:
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, optimArgs, schedArgs
            )

    # ---- FINAL EVALUATION ----
    print("\n[BENCHMARK] Final Evaluation")
    final_metrics = get_benchmark_report(model, test_loader, device)

    print("\nMetric Summary")
    print("-" * 40)
    for k, v in final_metrics.items():
        print(f"{k:<20}: {v}")
    print("-" * 40)

    # ---- SAVE METRICS ONLY (NO IMAGE GENERATION) ----
    os.makedirs("PAI", exist_ok=True)
    with open("PAI/metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("\n[INFO] Metrics saved to PAI/metrics.json")
    print("[INFO] PAI graphs will be generated automatically by the system.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="PerforatedAI Hackathon Benchmark"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="mnist",
            choices=["mnist", "cifar10"],
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=6,
        )
        args = parser.parse_args()

        run_benchmarks(
            dataset_name=args.dataset,
            epochs=args.epochs,
        )

    except Exception as e:
        import traceback

        print(f"\n[Execution Failed] {e}")
        traceback.print_exc()
        sys.exit(1)
