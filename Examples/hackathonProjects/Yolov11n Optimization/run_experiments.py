#!/usr/bin/env python3
"""
PAI-YOLO Data Efficiency Experiments Runner
============================================
Runs 8 experiments testing dendrite optimization at different data percentages.

Experiments:
  1. baseline_100 - 100% data, no PAI
  2. dendrite_100 - 100% data, with PAI
  3. baseline_50  - 50% data, no PAI
  4. dendrite_50  - 50% data, with PAI
  5. baseline_25  - 25% data, no PAI
  6. dendrite_25  - 25% data, with PAI
  7. baseline_15  - 15% data, no PAI
  8. dendrite_15  - 15% data, with PAI

Usage:
    python run_experiments.py --data-dir /path/to/VOC --output-dir ./runs
    
    # Run specific experiment only
    python run_experiments.py --experiment dendrite_50
    
    # Use TPU (Kaggle)
    python run_experiments.py --use-tpu
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENTS = [
    {"name": "baseline_100", "data_pct": 100, "use_pai": False},
    {"name": "dendrite_100", "data_pct": 100, "use_pai": True},
    {"name": "baseline_50",  "data_pct": 50,  "use_pai": False},
    {"name": "dendrite_50",  "data_pct": 50,  "use_pai": True},
    {"name": "baseline_25",  "data_pct": 25,  "use_pai": False},
    {"name": "dendrite_25",  "data_pct": 25,  "use_pai": True},
    {"name": "baseline_15",  "data_pct": 15,  "use_pai": False},
    {"name": "dendrite_15",  "data_pct": 15,  "use_pai": True},
]

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

SEED = 42
BATCH_SIZE = 32  # Balanced for VRAM
IMGSZ = 640
LR = 0.005  # Base learning rate
WARMUP_EPOCHS = 3  # Warmup period for stability

# ========== DOING_HISTORY MODE (Adaptive Plateau Detection) ==========
# Best for: Maximizing performance, adding dendrites when truly needed
# OPTIMIZED: Updated from WandB sweep results (best config)
HISTORY_MAX_EPOCHS = 500
HISTORY_EARLY_STOP_PATIENCE_BASELINE = 12
HISTORY_EARLY_STOP_PATIENCE_DENDRITIC = 50  # From sweep: early_stop_patience=50
HISTORY_MAX_DENDRITES = 4  # From sweep: max_dendrites=4
HISTORY_N_EPOCHS_TO_SWITCH = 15  # From sweep: n_epochs_to_switch=15
HISTORY_P_EPOCHS_TO_SWITCH = 15  # IGNORED for Open Source GD (only used in PerforatedBP)
HISTORY_HISTORY_LOOKBACK = 3  # From sweep: history_lookback=3 ✓
HISTORY_IMPROVEMENT_THRESHOLD = [0.001, 0.0001, 0]  # From sweep: [0.001, 0.0001, 0]
HISTORY_SCHEDULER_PATIENCE = 5  # Scheduler patience for HISTORY mode (< n_epochs_to_switch)

# ========== DOING_FIXED MODE (Fixed Epoch Intervals) ==========
# Best for: Systematic experiments, predictable dendrite addition
FIXED_MAX_EPOCHS = 500  # 3 dendrites at epochs 50, 100, 150 + 100 more for fine-tuning
FIXED_EARLY_STOP_PATIENCE_BASELINE = 15
FIXED_EARLY_STOP_PATIENCE_DENDRITIC = 50  # More patience for 3 dendrite recovery
FIXED_MAX_DENDRITES = 3  # 3 dendrites at epochs 50, 100, 150
FIXED_N_EPOCHS_TO_SWITCH = 50  # Add dendrite every 50 epochs
FIXED_P_EPOCHS_TO_SWITCH = 50  # Not used in open source, but kept for consistency
FIXED_HISTORY_LOOKBACK = 5  # Secondary role in FIXED mode
FIXED_IMPROVEMENT_THRESHOLD = [0.002, 0.001, 0]  # Very lenient - FIXED ignores this mostly
FIXED_SCHEDULER_PATIENCE = 15  # Scheduler patience for FIXED mode (shorter than n_epochs)



# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enabled for speed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[SEED] All random seeds set to {seed}")


# ============================================================================
# DEVICE SETUP (TPU/GPU/CPU)
# ============================================================================

def setup_device() -> Tuple[torch.device, str]:
    """
    Setup best available device (GPU or CPU).
    
    Returns:
        Tuple of (device, device_type: 'gpu'|'cpu')
    """
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n{'='*50}")
        print(f"✅ GPU INITIALIZED")
        print(f"  Device: {device_name}")
        print(f"  VRAM: {vram:.1f} GB")
        print(f"{'='*50}\n")
        return device, "gpu"
    
    # CUDA REQUIRED - Exit if not available
    print("\n❌ ERROR: CUDA GPU not available!")
    print("This script requires a CUDA-capable GPU for training.")
    print("Please ensure:")
    print("  1. You have a CUDA-capable GPU")
    print("  2. CUDA drivers are installed")
    print("  3. PyTorch is installed with CUDA support")
    import sys
    sys.exit(1)


# ============================================================================
# DATA SUBSET CREATION
# ============================================================================

def create_train_subset(
    full_train_images: List[str],
    percentage: int,
    seed: int = 42
) -> List[str]:
    """
    Create reproducible subset of training images.
    
    Same seed + same percentage = EXACT same images selected.
    """
    random.seed(seed)
    n_samples = int(len(full_train_images) * percentage / 100)
    subset = sorted(random.sample(full_train_images, n_samples))
    print(f"[DATA] Selected {n_samples}/{len(full_train_images)} images ({percentage}%)")
    return subset


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopper:
    """Early stopping based on validation mAP."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} epochs. "
                      f"Best: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            return False


# ============================================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    exp_name: str,
    data_pct: int,
    use_pai: bool,
    device: torch.device,
    data_dir: Path,
    output_path: Path,
    use_wandb: bool = True,
    n_epochs_to_switch: int = None,  # Will be set based on switch_mode
    p_epochs_to_switch: int = None,  # Will be set based on switch_mode
    max_dendrites: int = None,  # Will be set based on switch_mode
    use_opensource_gd: bool = False,  # True=Open Source GD, False=PerforatedBP
    baseline_patience: int = None,  # Will be set based on switch_mode
    dendritic_patience: int = None,  # Will be set based on switch_mode
    switch_mode: str = 'DOING_HISTORY'  # 'DOING_HISTORY' or 'DOING_FIXED'
) -> Dict:
    """
    Run a single experiment (baseline or dendrite).
    
    Returns dict with results.
    """
    from pai_yolo_training import train_pai_yolo
    
    # ========== SELECT CONFIGURATION BASED ON SWITCH MODE ==========
    if switch_mode.upper() == 'DOING_FIXED':
        # Use DOING_FIXED configuration
        if n_epochs_to_switch is None:
            n_epochs_to_switch = FIXED_N_EPOCHS_TO_SWITCH
        if p_epochs_to_switch is None:
            p_epochs_to_switch = FIXED_P_EPOCHS_TO_SWITCH
        if max_dendrites is None:
            max_dendrites = FIXED_MAX_DENDRITES
        if baseline_patience is None:
            baseline_patience = FIXED_EARLY_STOP_PATIENCE_BASELINE
        if dendritic_patience is None:
            dendritic_patience = FIXED_EARLY_STOP_PATIENCE_DENDRITIC
        max_epochs = FIXED_MAX_EPOCHS
        improvement_threshold = FIXED_IMPROVEMENT_THRESHOLD
        history_lookback = FIXED_HISTORY_LOOKBACK
        scheduler_patience = FIXED_SCHEDULER_PATIENCE  # Fixed mode: 15 epochs
    else:
        # Use DOING_HISTORY configuration (default)
        if n_epochs_to_switch is None:
            n_epochs_to_switch = HISTORY_N_EPOCHS_TO_SWITCH
        if p_epochs_to_switch is None:
            p_epochs_to_switch = HISTORY_P_EPOCHS_TO_SWITCH
        if max_dendrites is None:
            max_dendrites = HISTORY_MAX_DENDRITES
        if baseline_patience is None:
            baseline_patience = HISTORY_EARLY_STOP_PATIENCE_BASELINE
        if dendritic_patience is None:
            dendritic_patience = HISTORY_EARLY_STOP_PATIENCE_DENDRITIC
        max_epochs = HISTORY_MAX_EPOCHS
        improvement_threshold = HISTORY_IMPROVEMENT_THRESHOLD
        history_lookback = HISTORY_HISTORY_LOOKBACK
        scheduler_patience = HISTORY_SCHEDULER_PATIENCE  # History mode: 5 epochs
    
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Data: {data_pct}% | PAI: {'Yes' if use_pai else 'No'} | Mode: {switch_mode.upper()}")
    print("=" * 70)
    
    # Set seeds for reproducibility
    set_all_seeds(SEED)
    
    # Create experiment output directory
    exp_dir = output_path / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Import YOLO for test evaluation later
    from ultralytics import YOLO
    
    # NOTE: WandB initialization moved to after training check
    # This allows separate runs for test-only evaluation
    
    # NOTE: Model loading and PAI setup is done inside train_pai_yolo
    # Don't duplicate it here - train_pai_yolo is self-contained
    
    # Load dataset & Create Subset
    # ----------------------------
    print(f"[DATA] Creating {data_pct}% subset list...")
    # Assume data_dir structure: data_dir/images/train2007/*.jpg
    # The VOC directory structure usually has:
    #   images/train/, images/val/, etc. or images/train2007, images/val2007
    
    # We need to find the actual train images first
    train_img_dir = data_dir / "images" / "train2007"
    if not train_img_dir.exists():
        # Try checking VOC2007_portable.yaml path relative to data_dir
        # But let's assume standard structure: data_dir points to root containing images/
        train_img_dir = data_dir / "images" / "train2007"
        
    full_train_images = sorted(list(train_img_dir.glob("*.jpg")))
    if not full_train_images:
        print(f"⚠️ Warning: No images found in {train_img_dir}. Checking relative paths...")
        # Fallback: check based on current directory
        train_img_dir = Path("datasets") / "images" / "train2007"
        full_train_images = sorted(list(train_img_dir.glob("*.jpg")))
        
    if not full_train_images:
        raise FileNotFoundError(f"Could not find training images in {train_img_dir}")
        
    # Select subset
    subset_images = create_train_subset([str(p) for p in full_train_images], data_pct, seed=SEED)
    
    # Write subset .txt file
    subset_txt_path = exp_dir / "train_subset.txt"
    with open(subset_txt_path, 'w') as f:
        f.write('\n'.join(subset_images))
        
    # Create valid subset YAML
    # We copy the base YAML structure but point 'train' to our .txt file
    # We assume standard devkit structure or portable structure
    base_yaml_path = Path("VOC2007.yaml")
    if not base_yaml_path.exists():
         # Create a dummy base YAML if missing (portable mode)
         base_yaml_content = {
             'path': str(data_dir.absolute()), # Absolute path to root
             'train': str(subset_txt_path.absolute()), # Point to our subset list
             'val': 'images/val2007',
             'test': 'images/test2007',
             'nc': 20,
             'names': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
         }
         # For portable relative paths, we need to be careful.
         # Ultralytics handles absolute paths well.
    else:
        # Read existing YAML
        import yaml
        with open(base_yaml_path, 'r') as f:
            base_yaml_content = yaml.safe_load(f)
            
    
    # Update train path
    base_yaml_content['train'] = str(subset_txt_path.absolute())
    # CRITICAL: Val and Test paths are NOT modified - they remain constant!
    # Only the training set varies by percentage
    
    subset_yaml_path = exp_dir / "data_subset.yaml"
    with open(subset_yaml_path, 'w') as f:
        import yaml
        yaml.dump(base_yaml_content, f)
        
    print(f"[DATA] Subset YAML created: {subset_yaml_path}")
    print(f"[DATA] Training on {len(subset_images)} images ({data_pct}% of full train set)")
    print(f"[DATA] ✅ Val/Test sets: CONSTANT across all experiments (not subset)")


    # Save configuration before training
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "experiment": exp_name,
            "data_pct": data_pct,
            "use_pai": use_pai,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_dendrites": max_dendrites if use_pai else 0,
        }, f, indent=2)
    
    # TRAINING
    # --------
    best_map50 = 0.0
    trained_model = None
    wandb_initialized = False

    print(f"\n[TRAINING] Starting {exp_name}...")
    
    # Initialize WandB for full training
    if use_wandb:
        try:
            import wandb
            
            # Add suffix for open source GD
            run_name = exp_name
            if use_pai and use_opensource_gd:
                run_name += "_os"
                
            wandb.init(
                project="PAI-YOLO-DataEfficiency",
                name=run_name,
                config={
                    "experiment": exp_name,
                    "data_percentage": data_pct,
                    "use_pai": use_pai,
                    "use_opensource_gd": use_opensource_gd if use_pai else False,
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "max_dendrites": max_dendrites if use_pai else 0,
                    "early_stop_patience": dendritic_patience if use_pai else baseline_patience,
                    "n_epochs_to_switch": n_epochs_to_switch,
                    "p_epochs_to_switch": p_epochs_to_switch,
                }
            )
            wandb_initialized = True
        except ImportError:
            print("⚠️ WandB not installed, skipping logging")
            use_wandb = False
    
    # Import the UNIFIED training function
    from pai_yolo_training import train_pai_yolo
    
    # Call the UNIFIED training loop
    # OPTIMIZED: All parameters now match WandB sweep best config
    trained_model, best_map50 = train_pai_yolo(
        data_yaml=str(subset_yaml_path),
        epochs=max_epochs,  # Use mode-specific max_epochs
        batch_size=BATCH_SIZE,
        imgsz=IMGSZ,
        device=device.type, # Pass device.type ('cuda' or 'cpu')
        lr=LR,
        warmup_epochs=WARMUP_EPOCHS,
        save_name=str(exp_dir),
        pretrained='yolo11n.pt',
        seed=SEED,
        n_epochs_to_switch=n_epochs_to_switch,
        p_epochs_to_switch=p_epochs_to_switch,
        max_dendrites=max_dendrites,
        improvement_threshold=improvement_threshold,  # Mode-specific
        history_lookback=history_lookback,  # Mode-specific
        early_stop_patience=dendritic_patience if use_pai else baseline_patience,
        use_pai=use_pai,
        use_perforated_bp=not use_opensource_gd,  # opensource_gd=True means perforated_bp=False
        switch_mode=switch_mode,  # Pass switch_mode to training function
        scheduler_patience=scheduler_patience,  # Mode-specific scheduler patience
        # From sweep best config:
        candidate_weight_init=0.005 if use_pai else None,  # From sweep: 0.005
        pai_forward_function="sigmoid" if use_pai else None,  # From sweep: "sigmoid"
        find_best_lr=True if use_pai else False  # From sweep: true (PAI auto LR search)
    )
        
    # TEST EVALUATION
    # ---------------
    # NOTE: train_pai_yolo() now loads best_model_state before returning,
    # so trained_model already has the best weights. We just evaluate directly.
    print(f"\n[TESTING] Evaluating best model on Test set...")
    
    # Initialize results dictionary
    results = {
        'experiment': exp_name,
        'data_pct': data_pct,
        'use_pai': use_pai,
        'best_val_map50': best_map50,
        'best_epoch': 0,
        'total_epochs': 0
    }
    
    try:
        # The trained_model returned from train_pai_yolo already has best weights loaded
        # (best_model_state is restored before returning)
        
        val_yolo = YOLO('yolo11n.pt')  # Load base YOLO wrapper
        val_yolo.model = trained_model  # Inject our trained model with best weights
        
        # For PAI models: Prevent fuse() call by pretending model is already fused
        # This allows full yolo.val() output while preventing PAI module errors
        if use_pai:
            trained_model.is_fused = lambda: True
        
        # FULL TEST EVALUATION with detailed output (same as baseline!)
        # This shows class breakdown, model summary, speed etc.
        print(f"[Test] Final evaluation of best model...")
        test_results = val_yolo.val(
            data=str(subset_yaml_path),
            imgsz=IMGSZ,
            split='test',
            plots=False,
            save=False,
            verbose=True  # Show full detailed output
        )
        test_map50 = float(test_results.box.map50)
        print(f"  Test mAP50: {test_map50:.4f}")
        
        # Restore is_fused for PAI models
        if use_pai and hasattr(trained_model, 'is_fused'):
            delattr(trained_model, 'is_fused')
        
        results["test_map50"] = test_map50
        results["best_val_map50"] = best_map50
        
        # Try to get best_epoch from saved checkpoint
        best_pt = exp_dir / "best_model.pt"
        if best_pt.exists():
            checkpoint = torch.load(str(best_pt), map_location=device)
            results["best_epoch"] = checkpoint.get('epoch', 0)
        
        # Log to WandB if initialized
        if wandb_initialized:
            try:
                import wandb
                wandb.log({
                    "test/mAP50": test_map50,
                    "val/best_mAP50": best_map50,
                    "best_epoch": results.get("best_epoch", 0)
                })
            except:
                pass
    except Exception as e:
        print(f"⚠️ Warning: Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        results["test_map50"] = 0.0

    # Finish WandB (only if it was initialized)
    if wandb_initialized:
        try:
            import wandb
            # Force finish to ensure runs are separated
            wandb.finish()
        except:
            pass
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PAI-YOLO Data Efficiency Experiments")
    parser.add_argument("--data-dir", type=str, default="datasets/VOC",
                        help="Path to VOC dataset")
    parser.add_argument("--output-dir", type=str, default="runs/data_efficiency",
                        help="Output directory for results")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run specific experiment only (e.g., 'dendrite_50')")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only baseline experiments (baseline_100, baseline_50, baseline_25, baseline_15)")
    parser.add_argument("--baseline-subset", action="store_true",
                        help="Run baseline experiments EXCLUDING baseline_100 (only baseline_50, baseline_25, baseline_15)")
    parser.add_argument("--dendritic-only", action="store_true",
                        help="Run only dendritic experiments (dendrite_100, dendrite_50, dendrite_25, dendrite_15)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    
    # PAI configuration arguments
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Epochs in N-phase before checking plateau (default: 7)")
    parser.add_argument("--p-epochs", type=int, default=None,
                        help="Epochs in P-phase (PerforatedBP only, ignored in open source GD, default: 9)")
    parser.add_argument("--max-dendrites", type=int, default=None,
                        help="Maximum dendrites per neuron (default: 2)")
    parser.add_argument("--opensource-gd", action="store_true",
                        help="Use open source gradient descent (instant P-phase) instead of PerforatedBP")
    parser.add_argument("--mode", type=str, default="DOING_HISTORY",
                        choices=["DOING_HISTORY", "DOING_FIXED"],
                        help="PAI switch mode: DOING_HISTORY (adaptive) or DOING_FIXED (fixed intervals)")
    
    # Early stopping patience - SEPARATE for baseline and dendritic
    parser.add_argument("--baseline-patience", type=int, default=None,
                        help="Early stopping patience for BASELINE runs in epochs (default: 10)")
    parser.add_argument("--dendritic-patience", type=int, default=None,
                        help="Early stopping patience for DENDRITIC runs in epochs (default: 35)")
    
    args = parser.parse_args()
    
    # Set defaults based on selected mode
    if args.mode.upper() == 'DOING_FIXED':
        default_n_epochs = FIXED_N_EPOCHS_TO_SWITCH
        default_p_epochs = FIXED_P_EPOCHS_TO_SWITCH
        default_max_dendrites = FIXED_MAX_DENDRITES
        default_baseline_patience = FIXED_EARLY_STOP_PATIENCE_BASELINE
        default_dendritic_patience = FIXED_EARLY_STOP_PATIENCE_DENDRITIC
    else:  # DOING_HISTORY
        default_n_epochs = HISTORY_N_EPOCHS_TO_SWITCH
        default_p_epochs = HISTORY_P_EPOCHS_TO_SWITCH
        default_max_dendrites = HISTORY_MAX_DENDRITES
        default_baseline_patience = HISTORY_EARLY_STOP_PATIENCE_BASELINE
        default_dendritic_patience = HISTORY_EARLY_STOP_PATIENCE_DENDRITIC
    
    # Override defaults with command-line args if provided
    n_epochs_to_switch = args.n_epochs if args.n_epochs is not None else default_n_epochs
    p_epochs_to_switch = args.p_epochs if args.p_epochs is not None else default_p_epochs
    max_dendrites = args.max_dendrites if args.max_dendrites is not None else default_max_dendrites
    baseline_patience = args.baseline_patience if args.baseline_patience is not None else default_baseline_patience
    dendritic_patience = args.dendritic_patience if args.dendritic_patience is not None else default_dendritic_patience
    
    # Setup device (GPU or CPU)
    device, device_type = setup_device()
    
    # Filter experiments if specific one requested
    experiments_to_run = EXPERIMENTS
    if args.experiment:
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] == args.experiment]
        if not experiments_to_run:
            print(f"ERROR: Unknown experiment '{args.experiment}'")
            print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)
    elif args.baseline_only:
        experiments_to_run = [e for e in EXPERIMENTS if not e["use_pai"]]
        print(f"\n[MODE] Running BASELINE experiments only: {[e['name'] for e in experiments_to_run]}\n")
    elif args.baseline_subset:
        experiments_to_run = [e for e in EXPERIMENTS if not e["use_pai"] and e["data_pct"] != 100]
        print(f"\n[MODE] Running BASELINE experiments (excluding 100%): {[e['name'] for e in experiments_to_run]}\n")
    elif args.dendritic_only:
        experiments_to_run = [e for e in EXPERIMENTS if e["use_pai"]]
        print(f"\n[MODE] Running DENDRITIC experiments only: {[e['name'] for e in experiments_to_run]}\n")
    
    # Print PAI config ONLY if running dendritic experiments
    has_dendritic = any(e["use_pai"] for e in experiments_to_run)
    if has_dendritic:
        mode_str = "Open Source GD" if args.opensource_gd else "PerforatedBP"
        if args.opensource_gd:
            print(f"[PAI Config] Mode: {mode_str}, n_epochs={n_epochs_to_switch}, max_dendrites={max_dendrites} (p_epochs ignored)")
        else:
            print(f"[PAI Config] Mode: {mode_str}, n_epochs={n_epochs_to_switch}, p_epochs={p_epochs_to_switch}, max_dendrites={max_dendrites}")
        print()  # Extra newline for spacing
    
    # Run experiments
    all_results = []
    failed_experiments = []
    
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\n{'='*80}")
        print(f"  RUNNING EXPERIMENT {i}/{len(experiments_to_run)}: {exp['name']}")
        print(f"{'='*80}\n")
        
        try:
            result = run_single_experiment(
                exp_name=exp["name"],
                data_pct=exp["data_pct"],
                use_pai=exp["use_pai"],
                device=device,
                data_dir=Path(args.data_dir),
                output_path=Path(args.output_dir),
                use_wandb=not args.no_wandb,
                n_epochs_to_switch=n_epochs_to_switch,
                p_epochs_to_switch=p_epochs_to_switch,
                max_dendrites=max_dendrites,
                use_opensource_gd=args.opensource_gd,
                baseline_patience=baseline_patience,
                dendritic_patience=dendritic_patience,
                switch_mode=args.mode  # Pass user-selected mode
            )
            all_results.append(result)
            print(f"\n✅ {exp['name']} COMPLETED SUCCESSFULLY\n")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  USER INTERRUPTED - Stopping all experiments\n")
            break
            
        except Exception as e:
            error_msg = f"❌ EXPERIMENT FAILED: {exp['name']}"
            print(f"\n{'='*80}")
            print(error_msg)
            print(f"{'='*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            import traceback
            traceback.print_exc()
            print(f"{'='*80}\n")
            
            # Log failed experiment
            failed_experiments.append({
                'name': exp['name'],
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            
            # Save partial result with error info
            all_results.append({
                'experiment': exp['name'],
                'data_pct': exp['data_pct'],
                'use_pai': exp['use_pai'],
                'status': 'FAILED',
                'error': str(e),
                'best_val_map50': 0.0,
                'best_epoch': 0,
                'total_epochs': 0
            })
            
            print(f"⏭️  Continuing to next experiment...\n")
            continue
    
    
    
    # Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    successful_runs = [r for r in all_results if r.get('status') != 'FAILED']
    failed_runs = [r for r in all_results if r.get('status') == 'FAILED']
    
    if successful_runs:
        print(f"\n✅ SUCCESSFUL EXPERIMENTS ({len(successful_runs)}/{len(all_results)}):")
        print("-" * 70)
        for r in successful_runs:
            print(f"  {r['experiment']:20s}: mAP50={r['best_val_map50']:.4f} "
                  f"(epoch {r.get('best_epoch', 0)}/{r.get('total_epochs', 0)})")
    
    if failed_runs:
        print(f"\n❌ FAILED EXPERIMENTS ({len(failed_runs)}/{len(all_results)}):")
        print("-" * 70)
        for r in failed_runs:
            print(f"  {r['experiment']:20s}: {r.get('error', 'Unknown error')[:40]}...")
    
    # ========== GENERATE FINAL OUTPUTS ==========
    output_path = Path(args.output_dir)
    
    # 1. Save summary JSON
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SUMMARY] Saved to {summary_path}")
    
    # 2. Save summary CSV
    csv_path = output_path / "all_results.csv"
    if all_results:
        import csv
        fieldnames = ['experiment', 'data_pct', 'use_pai', 'best_val_map50', 'best_epoch', 'total_epochs']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"[SUMMARY] CSV saved to {csv_path}")
    
    # 3. Generate comparison bar chart
    if len(all_results) >= 2:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Separate PAI and Baseline results
            baseline_results = [r for r in all_results if not r.get('use_pai', False)]
            pai_results = [r for r in all_results if r.get('use_pai', False)]
            
            if baseline_results and pai_results:
                # Get unique data percentages
                data_pcts = sorted(set(r['data_pct'] for r in all_results))
                
                baseline_scores = []
                pai_scores = []
                
                for pct in data_pcts:
                    bl = next((r['best_val_map50'] for r in baseline_results if r['data_pct'] == pct), 0)
                    pa = next((r['best_val_map50'] for r in pai_results if r['data_pct'] == pct), 0)
                    baseline_scores.append(bl)
                    pai_scores.append(pa)
                
                # Create comparison bar chart
                x = range(len(data_pcts))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars1 = ax.bar([i - width/2 for i in x], baseline_scores, width, label='Baseline', color='#3498db')
                bars2 = ax.bar([i + width/2 for i in x], pai_scores, width, label='PAI', color='#e74c3c')
                
                ax.set_xlabel('Data Percentage (%)', fontsize=12)
                ax.set_ylabel('Validation mAP50', fontsize=12)
                ax.set_title('PAI vs Baseline: Data Efficiency Comparison', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([f'{p}%' for p in data_pcts])
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                for bar in bars2:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                chart_path = output_path / "comparison_chart.png"
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[SUMMARY] Comparison chart saved to {chart_path}")
                
                # Create data efficiency line chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data_pcts, baseline_scores, 'b-o', linewidth=2, markersize=8, label='Baseline')
                ax.plot(data_pcts, pai_scores, 'r-s', linewidth=2, markersize=8, label='PAI')
                
                ax.set_xlabel('Training Data Percentage (%)', fontsize=12)
                ax.set_ylabel('Validation mAP50', fontsize=12)
                ax.set_title('Data Efficiency: Performance vs Data Percentage', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                efficiency_path = output_path / "data_efficiency.png"
                plt.savefig(efficiency_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[SUMMARY] Data efficiency chart saved to {efficiency_path}")
                
        except Exception as e:
            print(f"[SUMMARY] Warning: Could not generate charts: {e}")


if __name__ == "__main__":
    main()
