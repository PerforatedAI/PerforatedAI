#!/usr/bin/env python3
"""
Dendritic YOLOv8n Training Script
PerforatedAI Dendritic Optimization Hackathon

Applies dendritic optimization to YOLOv8n using PerforatedAI.
Uses add_validation_score() for proper dendrite restructuring.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ultralytics import YOLO

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


def auto_exclude_yolov8_duplicates(model):
    """
    Pre-configure PerforatedAI to skip YOLOv8's shared module pointers.

    Based on the reference implementation in Examples/hackathonProject/yolo-dendritic/
    This is the PROPER way to avoid debugger prompts - no monkey patching needed.

    YOLOv8 shares activation function instances across layers for memory efficiency.
    This function tells PerforatedAI which duplicate references to skip BEFORE
    initialize_pai() encounters them.

    Args:
        model: YOLOv8 model instance

    Returns:
        int: Number of duplicate modules excluded
    """
    print("\n" + "="*70)
    print("Pre-configuring PerforatedAI for YOLOv8 shared modules...")
    print("="*70)

    # Step 1: Automatically find ALL duplicate module pointers
    seen = {}
    excluded = []

    for name, mod in model.named_modules():
        mid = id(mod)
        if mid in seen:
            # This module shares a pointer with another - exclude it
            GPA.pc.append_module_names_to_not_save([name])
            excluded.append(name)
        else:
            seen[mid] = name

    # Step 2: Additionally exclude known YOLOv8 shared activation patterns
    # (Following the pattern from yolo-dendritic reference implementation)

    # Backbone layers (model.1 through model.22)
    for i in range(1, 23):
        activation_suffixes = [
            '.act',
            '.default_act',
            '.cv1.act',
            '.cv1.default_act',
            '.cv2.act',
            '.cv2.default_act',
            '.conv.act',
            '.conv.default_act',
        ]
        for suffix in activation_suffixes:
            GPA.pc.append_module_names_to_not_save([f".model.{i}{suffix}"])

    # C2f module activations (nested m.0, m.1, etc.)
    for i in range(1, 23):
        for m_idx in range(10):  # Cover up to 10 nested modules
            for cv in ['cv1', 'cv2']:
                GPA.pc.append_module_names_to_not_save([
                    f".model.{i}.m.{m_idx}.{cv}.act"
                ])
                GPA.pc.append_module_names_to_not_save([
                    f".model.{i}.m.{m_idx}.{cv}.default_act"
                ])

    # Detection head activations (model.22)
    for cv in ["cv2", "cv3"]:
        for i in range(3):
            for j in range(3):
                GPA.pc.append_module_names_to_not_save([
                    f".model.22.{cv}.{i}.{j}.act"
                ])
                GPA.pc.append_module_names_to_not_save([
                    f".model.22.{cv}.{i}.{j}.default_act"
                ])

    print(f"✓ Pre-configured {len(excluded)} duplicate exclusions")
    print("✓ Model ready for non-interactive PAI initialization")
    print("="*70 + "\n")

    return len(excluded)


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

    # CRITICAL: Pre-configure duplicate exclusions BEFORE initialize_pai()
    # This is the proper solution from the yolo-dendritic reference implementation
    # Uses official PerforatedAI API - no monkey patching needed!
    auto_exclude_yolov8_duplicates(model)

    # Apply PerforatedAI dendritic optimization
    print("Configuring PerforatedAI...")
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_verbose(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    print("[OK] PerforatedAI configured\n")

    print("Initializing PerforatedAI (should run non-interactively)...")
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name="DendriticYOLOv8",
        maximizing_score=True
    )
    print("[OK] PerforatedAI initialized without debugger prompts!")
    
    # Restore input stem
    model.model[0] = input_stem
    model = model.to(device)
    
    dendritic_params = count_parameters(model)
    print(f"Dendritic params: {dendritic_params / 1e6:.2f}M")
    
    # Setup optimizer through PerforatedAI tracker
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(ReduceLROnPlateau)

    # CRITICAL FIX: Pass parameters as list, not iterator
    # After initialize_pai(), model.parameters() needs to be converted to list
    optimArgs = {'params': list(model.parameters()), 'lr': args.lr}
    schedArgs = {'mode': 'max', 'patience': 3, 'factor': 0.5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Training loop with proper PAI integration
    # KEY: Use add_extra_score() for training and add_validation_score() for validation
    # This produces the correct graph format (see Dendrite Recommendations.pdf):
    # - Green line: Training scores
    # - Orange line: Validation scores
    # - Vertical bars: Dendrite addition epochs
    print(f"\nTraining for up to {args.epochs} epochs (will stop when PAI signals completion)...")

    import os
    os.makedirs('PAI', exist_ok=True)

    epoch = 0
    training_complete = False

    while not training_complete and epoch < args.epochs:
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        # Train one epoch using YOLO's built-in training
        yolo.model = model
        results = yolo.train(
            data="coco128.yaml",
            epochs=1,
            imgsz=640,
            batch=16,
            device=device,
            exist_ok=True,
            verbose=False,
            project='runs/train',
            name='dendritic'
        )
        model = yolo.model

        # Get training score from results
        train_score = float(results.results_dict.get('metrics/mAP50(B)', 0))

        # IMPORTANT: Add TRAINING score to PAI tracker (creates green line in graph)
        GPA.pai_tracker.add_extra_score(train_score * 100, 'train')

        # Validate to get validation score
        val_score = validate(model, yolo, device)
        print(f"  Train mAP50: {train_score:.4f}")
        print(f"  Val mAP50:   {val_score:.4f}")

        if args.wandb:
            wandb.log({
                "train_mAP50": train_score,
                "val_mAP50": val_score,
                "epoch": epoch,
                "params": count_parameters(model)
            })

        # IMPORTANT: Add VALIDATION score to PAI tracker (creates orange line, triggers dendrites)
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_score * 100, model  # Convert to percentage
        )
        model = model.to(device)
        yolo.model = model

        if restructured:
            current_params = count_parameters(model)
            print(f"\n>>> DENDRITES ADDED! <<<")
            print(f"    Parameters: {baseline_params/1e6:.2f}M -> {current_params/1e6:.2f}M")
            # CRITICAL FIX: Pass parameters as list when resetting optimizer
            optimArgs['params'] = list(model.parameters())
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, optimArgs, schedArgs
            )

        if training_complete:
            print("\n" + "="*60)
            print("TRAINING COMPLETE - PerforatedAI optimization finished!")
            print("="*60)

        epoch += 1

    # Save the PAI graphs
    print("\nSaving PAI graphs...")
    try:
        GPA.pai_tracker.save_graphs()
        print("Graphs saved to PAI/PAI.png")
    except Exception as e:
        print(f"Note: {e}")

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
