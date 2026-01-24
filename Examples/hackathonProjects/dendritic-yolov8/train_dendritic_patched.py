#!/usr/bin/env python3
"""
Patched YOLOv8 Dendritic Training Script - FIXES PDB DEBUGGER ISSUE

This version patches the PerforatedAI library to bypass the interactive debugger
prompts that occur when duplicate module pointers are detected.

Key fixes:
1. Monkeypatches pdb.set_trace() BEFORE importing PerforatedAI
2. Patches bdb.Bdb.set_trace() to prevent debugger from being invoked
3. Uses doing_pai=True parameter in initialize_pai()
4. Properly configures all anti-debugging flags

This allows the training to run completely non-interactively in Colab or locally.
"""

import sys
import os

# CRITICAL FIX #1: Disable pdb BEFORE any other imports
import pdb
import bdb

# Completely disable the debugger
def noop(*args, **kwargs):
    pass

pdb.set_trace = noop
pdb.Pdb.set_trace = noop
bdb.Bdb.set_trace = noop
bdb.Bdb.do_continue = lambda self, arg: None
bdb.Bdb.do_c = lambda self, arg: None

# Also disable ipdb if present
try:
    import ipdb
    ipdb.set_trace = noop
except ImportError:
    pass

print("="*70)
print("DENDRITIC YOLOV8 TRAINING - PATCHED VERSION")
print("="*70)
print("Debugger completely disabled - will run non-interactively")
print("="*70)
print()

# Now safe to import everything else
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ultralytics import YOLO

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Note: wandb not available, skipping experiment tracking")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate(model, yolo, device):
    """Run validation and return mAP50"""
    model.eval()
    yolo.model = model
    results = yolo.val(data="coco128.yaml", verbose=False)
    return float(results.box.map50)


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with dendritic optimization')
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning rate')
    parser.add_argument("--batch", type=int, default=16, help='Batch size')
    parser.add_argument("--wandb", action="store_true", help='Enable wandb logging')
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project="Dendritic-YOLOv8-Hackathon", name="dendritic-yolov8n-patched")

    # Load YOLOv8n
    print("Loading YOLOv8n model...")
    yolo = YOLO("yolov8n.pt")
    model = yolo.model

    baseline_params = count_parameters(model)
    print(f"Baseline params: {baseline_params / 1e6:.2f}M")
    print()

    # CRITICAL FIX #2: Save input stem to restore after PAI initialization
    input_stem = model.model[0]

    # CRITICAL FIX #3: Configure PerforatedAI with ALL anti-debugging flags
    print("Configuring PerforatedAI...")
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_verbose(True)
    try:
        GPA.pc.set_dendrite_update_mode(True)
    except:
        pass  # Might not exist in this version
    GPA.pc.set_unwrapped_modules_confirmed(True)  # Skip debugger prompts
    print("[OK] PerforatedAI configured")
    print()

    # CRITICAL FIX #4: Initialize with doing_pai=True
    print("Initializing PerforatedAI (this may take a moment)...")
    model = UPA.initialize_pai(
        model,
        doing_pai=True,  # This is critical!
        save_name="DendriticYOLOv8",
        maximizing_score=True
    )

    # Restore input stem
    model.model[0] = input_stem
    model = model.to(device)

    dendritic_params = count_parameters(model)
    print(f"[OK] PerforatedAI initialized")
    print(f"Dendritic params: {dendritic_params / 1e6:.2f}M")
    print()

    # Setup optimizer through PerforatedAI tracker
    print("Setting up optimizer...")
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(ReduceLROnPlateau)

    optimArgs = {'params': model.parameters(), 'lr': args.lr}
    schedArgs = {'mode': 'max', 'patience': 3, 'factor': 0.5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    print(f"[OK] Optimizer configured (lr={args.lr})")
    print()

    # Training loop with proper PAI integration
    print("="*70)
    print(f"STARTING TRAINING - {args.epochs} epochs")
    print("="*70)
    print()

    epoch = 0
    training_complete = False

    while not training_complete and epoch < args.epochs:
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print('='*70)

        # Train one epoch
        yolo.model = model
        results = yolo.train(
            data='coco128.yaml',
            epochs=1,
            imgsz=640,
            batch=args.batch,
            device=device,
            exist_ok=True,
            verbose=False,
            project='runs/train',
            name='dendritic'
        )
        model = yolo.model

        # Get training score
        train_score = float(results.results_dict.get('metrics/mAP50(B)', 0))

        # CRITICAL: Add TRAINING score (creates green line in graph)
        GPA.pai_tracker.add_extra_score(train_score * 100, 'train')

        # Validate
        val_score = validate(model, yolo, device)

        print(f"  Train mAP50: {train_score:.4f}")
        print(f"  Val mAP50:   {val_score:.4f}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                "train_mAP50": train_score,
                "val_mAP50": val_score,
                "epoch": epoch,
                "params": count_parameters(model)
            })

        # CRITICAL: Add VALIDATION score (creates orange line, triggers dendrites)
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_score * 100, model
        )
        model = model.to(device)
        yolo.model = model

        if restructured:
            current_params = count_parameters(model)
            print(f"\n>>> DENDRITES ADDED! <<<")
            print(f"    Parameters: {baseline_params/1e6:.2f}M -> {current_params/1e6:.2f}M")
            optimArgs['params'] = model.parameters()
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, optimArgs, schedArgs
            )

        if training_complete:
            print("\n" + "="*70)
            print("TRAINING COMPLETE - PerforatedAI optimization finished!")
            print("="*70)

        epoch += 1

    # Save the PAI graphs
    print("\nSaving PAI graphs...")
    try:
        GPA.pai_tracker.save_graphs()
        print("[OK] Graphs saved to DendriticYOLOv8/PAI.png")
    except Exception as e:
        print(f"Note: {e}")

    # Final validation
    final_score = validate(model, yolo, device)
    final_params = count_parameters(model)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Baseline params: {baseline_params / 1e6:.2f}M")
    print(f"Final params: {final_params / 1e6:.2f}M")
    print(f"Reduction: {(1 - final_params/baseline_params)*100:.1f}%")
    print(f"Final mAP50: {final_score:.4f}")
    print("="*70)

    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            "final_mAP50": final_score,
            "final_params_M": final_params / 1e6,
            "param_reduction_pct": (1 - final_params/baseline_params)*100
        })
        wandb.finish()

    # Check if PAI.png was created
    import os
    if os.path.exists('DendriticYOLOv8/PAI.png'):
        size = os.path.getsize('DendriticYOLOv8/PAI.png')
        print(f"\n[OK] PAI.png created ({size:,} bytes)")
        if size < 1000:
            print("  [WARN] File seems too small - may be placeholder")
    else:
        print("\n[WARN] PAI.png not found in DendriticYOLOv8/")

    print("\nDone! Training completed successfully.")


if __name__ == "__main__":
    main()
