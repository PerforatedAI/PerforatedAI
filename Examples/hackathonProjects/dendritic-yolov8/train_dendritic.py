#!/usr/bin/env python3
"""
Dendritic YOLOv8n Training Script
PerforatedAI Dendritic Optimization Hackathon

Applies dendritic optimization to YOLOv8n using PerforatedAI.
Uses add_validation_score() for proper dendrite restructuring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

import wandb
import argparse


# PyTorch 2.6+ checkpoint loading patch
_orig_load = torch.load
def torch_load_unsafe(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = torch_load_unsafe


def count_parameters(model):
    """Count parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        imgs = batch["img"].to(device).float() / 255.0
        
        optimizer.zero_grad()
        preds = model(imgs)
        
        # Compute detection loss (simplified)
        loss = torch.zeros(1, device=device)
        for pred in preds if isinstance(preds, tuple) else [preds]:
            if isinstance(pred, torch.Tensor):
                loss += pred.mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / max(len(dataloader), 1)


def validate(model, yolo_wrapper, device):
    """Validate model and return mAP score."""
    model.eval()
    
    # Use YOLO's built-in validation
    yolo_wrapper.model = model
    results = yolo_wrapper.val(data="coco128.yaml", verbose=False)
    
    return float(results.box.map50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize W&B
    if args.wandb:
        wandb.init(project="Dendritic-YOLOv8-Hackathon", name="dendritic-yolov8n")
    
    # Load YOLOv8n
    yolo = YOLO("yolov8n.pt")
    model = yolo.model
    
    baseline_params = count_parameters(model)
    print(f"Baseline params: {baseline_params / 1e6:.2f}M")
    
    # Save input stem (skip model.0 to avoid weight issues)
    input_stem = model.model[0]
    
    # Apply PerforatedAI dendritic optimization
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_verbose(True)
    GPA.pc.set_dendrite_update_mode(True)
    
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name="DendriticYOLOv8",
        maximizing_score=True
    )
    
    # Restore input stem
    model.model[0] = input_stem
    model = model.to(device)
    
    dendritic_params = count_parameters(model)
    print(f"Dendritic params: {dendritic_params / 1e6:.2f}M")
    
    # Setup optimizer through PerforatedAI tracker
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(ReduceLROnPlateau)
    
    optimArgs = {'params': model.parameters(), 'lr': args.lr}
    schedArgs = {'mode': 'max', 'patience': 3, 'factor': 0.5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Training loop with add_validation_score()
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Validate to get score
        score = validate(model, yolo, device)
        print(f"Epoch {epoch}: mAP50 = {score:.4f}")
        
        if args.wandb:
            wandb.log({"mAP50": score, "epoch": epoch})
        
        # Add validation score to PerforatedAI tracker
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            score, model
        )
        
        if restructured:
            print("Model restructured! Re-initializing optimizer...")
            optimArgs['params'] = model.parameters()
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, optimArgs, schedArgs
            )
            model = model.to(device)
        
        if training_complete:
            print("Training complete - dendrite optimization finished!")
            break
        
        # Train one epoch using YOLO's built-in training
        yolo.model = model
        yolo.train(
            data="coco128.yaml",
            epochs=1,
            imgsz=640,
            batch=16,
            device=device,
            exist_ok=True,
            verbose=False
        )
        model = yolo.model
    
    # Final validation
    final_score = validate(model, yolo, device)
    final_params = count_parameters(model)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Baseline params: {baseline_params / 1e6:.2f}M")
    print(f"Final params: {final_params / 1e6:.2f}M")
    print(f"Reduction: {(1 - final_params/baseline_params)*100:.1f}%")
    print(f"Final mAP50: {final_score:.4f}")
    print("="*50)
    
    if args.wandb:
        wandb.log({
            "final_mAP50": final_score,
            "final_params_M": final_params / 1e6,
            "param_reduction_pct": (1 - final_params/baseline_params)*100
        })
        wandb.finish()
    
    print("\nPAI.png should be generated in the PAI/ folder by PerforatedAI.")


if __name__ == "__main__":
    main()
