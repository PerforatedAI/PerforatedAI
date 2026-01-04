#!/usr/bin/env python3
"""
Dendritic YOLOv8n Training Script
PerforatedAI Dendritic Optimization Hackathon

This script trains a YOLOv8n model with PerforatedAI's dendritic
optimization on COCO128 dataset.
"""

import argparse
import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from ultralytics import YOLO

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


# PyTorch 2.6+ checkpoint loading patch
_orig_load = torch.load
def torch_load_unsafe(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = torch_load_unsafe


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_speed(model, device, img_size=640, num_runs=100):
    """Measure average inference time in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def apply_dendritic_optimization(model, save_name="DendriticYOLOv8"):
    """Apply PerforatedAI dendritic optimization to the model.
    
    CRITICAL: Skips model.0 (input stem) to avoid weight loading issues.
    """
    # Save the input stem before optimization
    input_stem = model.model[0]
    
    # Configure PerforatedAI settings
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_verbose(True)
    GPA.pc.set_dendrite_update_mode(True)
    
    # Apply PerforatedAI initialization
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name=save_name,
        maximizing_score=True
    )
    
    # Restore input stem to avoid weight loading issues
    model.model[0] = input_stem
    
    return model


def setup_pai_optimizer(model, lr=1e-3):
    """Setup optimizer through PerforatedAI tracker."""
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(ReduceLROnPlateau)
    
    optimArgs = {'params': model.parameters(), 'lr': lr}
    schedArgs = {'mode': 'max', 'patience': 3, 'factor': 0.5}
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    return optimizer, scheduler, optimArgs, schedArgs


def main():
    parser = argparse.ArgumentParser(description="Train dendritic YOLOv8n on COCO128")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Dataset config")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--project", type=str, default="runs/dendritic", help="Project directory")
    parser.add_argument("--name", type=str, default="yolov8n_dendritic", help="Run name")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--save-json", type=str, default="dendritic_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Setup device
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Initialize W&B
    if args.wandb:
        wandb.init(
            project="Dendritic-YOLOv8-Hackathon",
            name="dendritic-yolov8n",
            tags=["dendritic", "perforatedai", "yolov8n", "coco128"],
            config={
                "model": "yolov8n",
                "dataset": args.data,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimization": "perforatedai_dendritic"
            }
        )

    # Load model
    print("Loading YOLOv8n model...")
    yolo_model = YOLO("yolov8n.pt")
    
    # Get baseline parameter count before optimization
    baseline_params, _ = count_parameters(yolo_model.model)
    print(f"Parameters before optimization: {baseline_params / 1e6:.2f}M")

    # Apply dendritic optimization
    print("\nApplying PerforatedAI dendritic optimization...")
    dendritic_model = apply_dendritic_optimization(yolo_model.model)
    dendritic_model = dendritic_model.to(device)
    
    # Get parameter count after optimization
    total_params, trainable_params = count_parameters(dendritic_model)
    print(f"Parameters after optimization: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")

    if args.wandb:
        wandb.log({
            "dendritic_params_M": total_params / 1e6,
            "param_reduction_pct": (1 - total_params / baseline_params) * 100
        })

    # Re-assign the modified model back to the YOLO wrapper
    yolo_model.model = dendritic_model

    # Train model using Ultralytics infrastructure
    print(f"\nStarting dendritic training for {args.epochs} epochs...")
    results = yolo_model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=0 if torch.cuda.is_available() and args.device != "cpu" else "cpu",
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        optimizer="Adam",
        lr0=args.learning_rate
    )

    # Validate model
    print("\nValidating model...")
    val_results = yolo_model.val(
        data=args.data,
        device=0 if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )

    # Measure inference speed
    inference_ms = measure_inference_speed(yolo_model.model, device)

    # Compile metrics
    metrics = {
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
        "precision": float(val_results.box.mp),
        "recall": float(val_results.box.mr),
        "params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
        "inference_ms": inference_ms,
        "baseline_params_M": baseline_params / 1e6,
        "param_reduction_pct": (1 - total_params / baseline_params) * 100
    }

    # Print results
    print("\n" + "="*50)
    print("DENDRITIC RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("="*50)

    # Log to W&B
    if args.wandb:
        wandb.log({f"dendritic_{k}": v for k, v in metrics.items()})
        wandb.finish()

    # Save results to JSON
    results_data = {
        "model": "YOLOv8n",
        "type": "dendritic",
        "optimization": "PerforatedAI",
        "dataset": args.data,
        "epochs": args.epochs,
        "metrics": metrics
    }

    with open(args.save_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {args.save_json}")

    return metrics


if __name__ == "__main__":
    main()
