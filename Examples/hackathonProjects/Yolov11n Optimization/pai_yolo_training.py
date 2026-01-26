"""
PAI-YOLOv11n Custom Training Script
====================================
Author: PerforatedAI Hackathon Implementation
Goal: Demonstrate DATA EFFICIENCY - same accuracy with way less training data

This script implements a custom training loop for YOLOv11n with PerforatedAI (PAI)
dendritic optimization. The default Ultralytics YOLO.train() CANNOT be used with PAI
because add_validation_score() must be called EVERY EPOCH.

Usage (VOC2007 - Automatic Download):
    # VOC2007 will be automatically downloaded by Ultralytics
    python pai_yolo_training.py --experiment all --use-voc --epochs 100 --save-name runs/voc_exp
    
Usage (Custom Dataset):
    python pai_yolo_training.py --data custom.yaml --epochs 100 --experiment pai
    python pai_yolo_training.py --experiment all --source-dir ./MyDataset --epochs 100

Requirements:
    - ultralytics
    - perforatedai (proprietary, requires account)
    - torch, torchvision
    - PyYAML

Datasets:
    VOC2007 (Recommended - Auto-download): 
    - Automatically downloaded by Ultralytics when using --use-voc flag
    - 20 object classes, ~5,000 train images, ~5,000 test images
    
    Alternative - SODA-A (Aerial Small Object Detection): 
    - Kaggle: https://www.kaggle.com/datasets/sovitrath/soda-a
    - Official: https://shaunyuan22.github.io/SODA/

Last Updated: January 2026
"""

import os

# ============================================================================
# PERFORATED AI LICENSE CREDENTIALS (Dendrites 2.0)
# Must be set BEFORE importing perforatedai
# ============================================================================
os.environ["PAIEMAIL"] = "hacker@perforatedai.com"
os.environ["PAITOKEN"] = "InJ9BjZSB+B+l30bmSzhqOwsXxOx0NRKAe8dtdAqdQcT/pKjmme1fqB1zrnCd5CWNrhJm40PVjaDbIrjR5xU+q2uhcUWX8gk2Kb2lHjafkUnizPXyP+yckbv+UxlU25ZlrvC3XlLu/AZdVKJE7Eov9+4c76sKe2hbRnH1fny2xIPYmy2/m/sY1gxXbhPtTa1mtxk2EgLeo5pRu/eL/7pSXWmEoRmvVorgQEJzt1VYOZyp0vP4bLxF72tOgSjXGBO8SHHcN16CbOVJuIEm3jmEc/AfPyyB+G4TEqhH7UZ0W2R/bnXtNberKqF2bQTuyT26etQw6NEMoXwuugDcrBXEw=="

# ============================================================================
# DISABLE PDB DEBUGGER COMPLETELY
# PAI sometimes triggers pdb breakpoints for warnings/errors. This disables them.
# ============================================================================
os.environ["PYTHONBREAKPOINT"] = "0"  # Disable breakpoint() calls
import bdb
bdb.Bdb.set_trace = lambda self, frame=None: None  # Disable set_trace

import sys
sys.breakpointhook = lambda *args, **kwargs: None  # Disable breakpointhook

import pdb
pdb.set_trace = lambda: None  # Disable pdb.set_trace
import io  # For stdout suppression
import random
import math
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import csv
import json
import matplotlib.pyplot as plt

# WandB import (optional - fails gracefully if not installed)
try:
    import wandb
except ImportError:
    wandb = None

# Ultralytics imports
try:
    from ultralytics import YOLO
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import LOGGER, colorstr
except ImportError as e:
    print(f"[ERROR] Ultralytics not installed: {e}")
    print("Install with: pip install ultralytics")
    sys.exit(1)

# PerforatedAI imports
# CRITICAL: Must use globals_perforatedai (NOT globalbp) as per official examples
try:
    # Suppress PAI's startup messages ("Building dendrites with Perforated Backpropagation")
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    
    # Restore stdout
    sys.stdout = old_stdout
    PAI_AVAILABLE = True
except ImportError as e:
    # Restore stdout in case of error
    sys.stdout = old_stdout
    print(f"[WARNING] PerforatedAI not installed: {e}")
    print("PAI experiments will not be available.")
    print("Install with: pip install perforatedai")
    PAI_AVAILABLE = False



# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================
# Standard augmentation settings for YOLO training
# These apply to both PAI and Baseline for fair comparison

AUGMENTATION_CONFIG = {
    'mosaic': 1.0,        # Mosaic augmentation probability
    'mixup': 0.0,         # Mixup augmentation probability  
    'copy_paste': 0.0,    # Copy-paste augmentation probability
    'degrees': 0.0,       # Rotation degrees
    'translate': 0.1,     # Translation fraction
    'scale': 0.5,         # Scale +/- gain
    'shear': 0.0,         # Shear +/- degrees
    'perspective': 0.0,   # Perspective +/- fraction
    'flipud': 0.0,        # Vertical flip probability
    'fliplr': 0.5,        # Horizontal flip probability
    'hsv_h': 0.015,       # HSV-Hue augmentation
    'hsv_s': 0.7,         # HSV-Saturation augmentation
    'hsv_v': 0.4,         # HSV-Value augmentation
    'erasing': 0.0,       # Random erasing probability
}


# =============================================================================
# EMA (Exponential Moving Average) - Matching Ultralytics
# =============================================================================
class ModelEMA:
    """
    Exponential Moving Average of model weights.
    
    Maintains a smoothed version of model weights which typically
    provides better generalization than the raw trained weights.
    
    This matches Ultralytics' implementation in ultralytics/utils/torch_utils.py
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
            tau: Time constant for decay ramp-up
        """
        from copy import deepcopy
        
        # Create EMA model
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        
        self.decay = decay
        self.tau = tau
        self.updates = 0
    
    def update(self, model: nn.Module):
        """
        Update EMA weights with current model weights.
        
        Args:
            model: Current training model
        """
        self.updates += 1
        # Ramp up decay from 0 to self.decay over tau updates
        decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        
        with torch.no_grad():
            model_state = model.state_dict()
            ema_state = self.ema.state_dict()
            
            for k, v in ema_state.items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1 - decay) * model_state[k].detach()
    
    def update_attr(self, model: nn.Module, include: list = None):
        """Copy attributes from model to EMA."""
        include = include or []
        for k, v in model.__dict__.items():
            if k.startswith('_') or k in include:
                setattr(self.ema, k, v)


# =============================================================================
# SECTION 1: REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    CRITICAL: Call this BEFORE anything else to ensure both baseline
    and PAI experiments use identical random initializations.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[Reproducibility] All seeds set to {seed}")


# =============================================================================
# SECTION 2: DATASET UTILITIES
# =============================================================================

def create_fixed_split(
    source_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Create a FIXED val/test split from the END of the dataset.
    The remaining images form the FULL train pool.
    
    CRITICAL: Val and Test are ALWAYS the same images across all experiments.
    Only Train varies (subsets taken from the train pool).
    
    Args:
        source_dir: Directory containing images and labels
        output_dir: Directory to save the split
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for test (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_pool_dir, val_dir, test_dir) paths
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(source_path.glob(f'images/{ext}'))
    
    if not all_images:
        for ext in image_extensions:
            all_images.extend(source_path.glob(ext))
    
    if not all_images:
        raise ValueError(f"No images found in {source_dir}")
    
    all_images = sorted(all_images)
    random.shuffle(all_images)
    
    n = len(all_images)
    
    # CRITICAL: Fix val and test from the END of the list
    # This ensures they NEVER change regardless of train percentage
    test_count = int(test_ratio * n)
    val_count = int(val_ratio * n)
    
    test_images = all_images[-test_count:]  # Last test_count images
    val_images = all_images[-(test_count + val_count):-test_count]  # Before test
    train_pool = all_images[:-(test_count + val_count)]  # Everything else = train pool
    
    def copy_images_with_labels(images, dest_dir, source_path):
        img_dir = Path(dest_dir) / 'images'
        label_dir = Path(dest_dir) / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for img in images:
            shutil.copy(img, img_dir / img.name)
            
            label_name = img.stem + '.txt'
            label_candidates = [
                img.parent.parent / 'labels' / label_name,
                img.parent / label_name,
                source_path / 'labels' / label_name
            ]
            
            for label_path in label_candidates:
                if label_path.exists():
                    shutil.copy(label_path, label_dir / label_name)
                    break
    
    # Create the three directories
    train_pool_dir = output_path / 'train_pool'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    copy_images_with_labels(train_pool, train_pool_dir, source_path)
    copy_images_with_labels(val_images, val_dir, source_path)
    copy_images_with_labels(test_images, test_dir, source_path)
    
    print(f"[Dataset Split] Created FIXED val/test split:")
    print(f"  Train Pool: {len(train_pool)} images (100% available for training)")
    print(f"  Val (FIXED): {len(val_images)} images")
    print(f"  Test (FIXED): {len(test_images)} images")
    
    return str(train_pool_dir), str(val_dir), str(test_dir)


def create_data_efficiency_splits(
    train_pool_dir: str,
    output_dir: str,
    ratios: list = [1.0, 0.5, 0.2, 0.1, 0.05],
    seed: int = 42
) -> Dict[str, str]:
    """
    Create training subsets from the FIXED train pool.
    
    IMPORTANT: This takes subsets from an ALREADY FIXED train pool,
    ensuring val/test remain constant across all experiments.
    
    Args:
        train_pool_dir: Path to the full train pool (from create_fixed_split)
        output_dir: Where to save the train splits
        ratios: List of data percentages to create (relative to train pool)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping ratio to split directory path
    """
    random.seed(seed)
    
    source_path = Path(train_pool_dir)
    output_path = Path(output_dir)
    
    # Find all images in train pool
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(source_path.glob(f'images/{ext}'))
    
    if not all_images:
        for ext in image_extensions:
            all_images.extend(source_path.glob(ext))
    
    all_images = sorted(all_images)
    random.shuffle(all_images)
    
    full_train_size = len(all_images)
    split_paths = {}
    
    print(f"\n[Data Efficiency] Creating train splits from pool of {full_train_size} images:")
    
    for ratio in ratios:
        subset_size = int(full_train_size * ratio)
        subset_images = all_images[:subset_size]
        
        split_name = f"train_{int(ratio * 100)}pct"
        split_dir = output_path / split_name
        img_dir = split_dir / 'images'
        label_dir = split_dir / 'labels'
        
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for img in subset_images:
            shutil.copy(img, img_dir / img.name)
            
            label_name = img.stem + '.txt'
            label_candidates = [
                img.parent.parent / 'labels' / label_name,
                img.parent / label_name,
                source_path / 'labels' / label_name
            ]
            
            for label_path in label_candidates:
                if label_path.exists():
                    shutil.copy(label_path, label_dir / label_name)
                    break
        
        split_paths[f"{int(ratio * 100)}pct"] = str(split_dir)
        print(f"  {int(ratio * 100)}% split: {subset_size} images")
    
    return split_paths


def create_dataset_yaml(
    output_path: str,
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    nc: int = 9,
    names: Optional[list] = None
) -> str:
    """
    Create a YOLO dataset YAML configuration file.
    
    Args:
        output_path: Path to save the YAML file
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images (optional)
        nc: Number of classes
        names: List of class names
        
    Returns:
        Path to created YAML file
    """
    if names is None:
        # Default SODA-A classes
        names = ['pedestrian', 'cyclist', 'car', 'truck', 'tram', 
                 'tricycle', 'bus', 'moped', 'stroller'][:nc]
    
    data = {
        'path': str(Path(train_path).parent.parent),
        'train': train_path,
        'val': val_path,
        'nc': nc,
        'names': names
    }
    
    if test_path:
        data['test'] = test_path
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"[Dataset] Created YAML config: {output_path}")
    return output_path


# =============================================================================
# SECTION 3: PAI CONFIGURATION
# =============================================================================

def setup_pai_config(
    n_epochs_to_switch: int = 15,  # From sweep: 15
    p_epochs_to_switch: int = 15,  # IGNORED for Open Source GD (only for PerforatedBP)
    max_dendrites: int = 4,  # From sweep: 4
    improvement_threshold: list = None,  # Will default to [0.001, 0.0001, 0] from sweep
    history_lookback: int = 3,  # From sweep: 3 ✓
    verbose: bool = False,  # Default FALSE to reduce PAI output clutter
    use_perforated_bp: bool = True,  # True=PerforatedBP, False=Open Source GD
    switch_mode: str = 'DOING_HISTORY',  # 'DOING_HISTORY' or 'DOING_FIXED'
    find_best_lr: bool = True  # From sweep: true (PAI's automatic LR search)
) -> None:
    """
    Configure PAI for YOLOv11n object detection.
    
    OPTIMIZED: Defaults updated from WandB sweep results (best config).
    
    This sets up the Probability Controller (pc) with optimal settings
    for object detection tasks with data efficiency focus.
    
    Args:
        n_epochs_to_switch: Epochs in N-phase before checking plateau (or fixed interval for DOING_FIXED)
        p_epochs_to_switch: Epochs in P-phase (dendrite training) - only for PerforatedBP
        max_dendrites: Maximum dendrites per neuron
        improvement_threshold: Minimum improvement to continue
        history_lookback: Epochs to look back for improvement
        verbose: Print PAI decisions
        use_perforated_bp: If True, use PerforatedBP; if False, use open source GD
        switch_mode: 'DOING_HISTORY' for adaptive or 'DOING_FIXED' for fixed epoch intervals
    """
    if not PAI_AVAILABLE:
        raise RuntimeError("PerforatedAI not installed. Cannot configure PAI.")
    
    print("\n" + "=" * 60)
    print("  PAI CONFIGURATION FOR YOLO")
    print("=" * 60)
    
    # CRITICAL: Reset PAI state from previous experiments
    # This prevents "bias already exists" and other state-related errors
    try:
        # Reset module tracking lists to avoid accumulation
        if hasattr(GPA, 'pai_tracker'):
            # Clear previous tracker if it exists
            if hasattr(GPA.pai_tracker, 'reset'):
                GPA.pai_tracker.reset()
                print("  PAI Tracker: RESET")
        
        # Clear module names to not save (accumulated from previous runs)
        if hasattr(GPA.pc, 'set_module_names_to_not_save'):
            GPA.pc.set_module_names_to_not_save([])
            print("  Module names to not save: CLEARED")
            
        # Clear module names to track
        if hasattr(GPA.pc, 'set_module_names_to_track'):
            GPA.pc.set_module_names_to_track([])
            print("  Module names to track: CLEARED")
    except Exception as e:
        print(f"  Warning: Could not reset PAI state: {e}")
    
    # FIRST: Disable dimension debugging to suppress verbose output
    # Set to 0 to hide "setting d shape for" messages
    if hasattr(GPA.pc, 'set_debugging_output_dimensions'):
        GPA.pc.set_debugging_output_dimensions(0)  # 0 = silent, 1 = verbose
        print("  Dimension debugging: DISABLED (clean output)")
    
    # Disable testing mode
    if hasattr(GPA.pc, 'set_testing_dendrite_capacity'):
        GPA.pc.set_testing_dendrite_capacity(False)
    
    # Set PerforatedBP mode based on parameter
    # True = PerforatedBP (proprietary), False = Open Source Gradient Descent
    if hasattr(GPA.pc, 'set_perforated_backpropagation'):
        GPA.pc.set_perforated_backpropagation(use_perforated_bp)
        if use_perforated_bp:
            print("  Perforated Backpropagation: ENABLED")
        else:
            print("  Perforated Backpropagation: DISABLED (using Open Source GD)")
    # Enable verbose output
    if hasattr(GPA.pc, 'set_verbose'):
        GPA.pc.set_verbose(verbose)
    
    # Enable dendrite update mode (if available)
    if hasattr(GPA.pc, 'set_dendrite_update_mode'):
        GPA.pc.set_dendrite_update_mode(True)
        print("  Dendrite update mode: ENABLED")
    
    # ========== SWITCH MODE CONFIGURATION ==========
    # DOING_HISTORY: Adaptive plateau detection - adds dendrite when improvement slows
    # DOING_FIXED: Fixed epoch intervals - adds dendrite at regular intervals (e.g., every 50 epochs)
    
    if switch_mode.upper() == 'DOING_FIXED':
        # DOING_FIXED: Predictable dendrite addition at fixed epoch intervals
        # Best for: Systematic experimentation, comparing dendrite effects at specific stages
        if hasattr(GPA.pc, 'DOING_FIXED'):
            GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED)
            print(f"  Switch Mode: DOING_FIXED (dendrites at epochs {n_epochs_to_switch}, {n_epochs_to_switch*2}, ...)")
        else:
            print("  Warning: DOING_FIXED not available, falling back to DOING_HISTORY")
            if hasattr(GPA.pc, 'DOING_HISTORY'):
                GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    else:
        # DOING_HISTORY: Adaptive plateau detection
        # Best for: Maximizing performance, adding dendrites when model truly plateaus
        if hasattr(GPA.pc, 'DOING_HISTORY'):
            GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
            print(f"  Switch Mode: DOING_HISTORY (adaptive plateau detection)")
        else:
            print("  Warning: DOING_HISTORY not available")
    
    # Epoch configuration
    if hasattr(GPA.pc, 'set_n_epochs_to_switch'):
        GPA.pc.set_n_epochs_to_switch(n_epochs_to_switch)
        print(f"  N-Epochs to Switch: {n_epochs_to_switch}")
    
    if hasattr(GPA.pc, 'set_p_epochs_to_switch'):
        GPA.pc.set_p_epochs_to_switch(p_epochs_to_switch)
        print(f"  P-Epochs to Switch: {p_epochs_to_switch}")
    
    # History lookback for plateau detection
    if hasattr(GPA.pc, 'set_history_lookback'):
        GPA.pc.set_history_lookback(history_lookback)
        print(f"  History Lookback: {history_lookback}")
    
    # Dendrite limits
    if hasattr(GPA.pc, 'set_max_dendrites'):
        GPA.pc.set_max_dendrites(max_dendrites)
        print(f"  Max Dendrites: {max_dendrites}")
    
    # Find Best LR - PAI's automatic LR search after dendrite addition
    # When True: PAI tests multiple LRs and picks the best (RECOMMENDED by PAI)
    # When False: Use manual progressive decay (LR/5, LR/10, LR/20)
    # NOTE: When True, PAI may print testing messages - actual dendrite addition
    # happens when restructured=True in add_validation_score
    if hasattr(GPA.pc, 'set_find_best_lr'):
        GPA.pc.set_find_best_lr(find_best_lr)
        if find_best_lr:
            print(f"  Find Best LR: ENABLED (PAI will auto-test LRs after dendrite addition)")
        else:
            print(f"  Find Best LR: DISABLED (using manual LR decay: /5, /10, /20)")
    
    # Improvement threshold - PROGRESSIVE pattern from WandB sweep (best config)
    # 1st dendrite: STRICT (0.001 = 0.1%) - Only add if truly plateaued
    # 2nd dendrite: VERY STRICT (0.0001 = 0.01%) - Very high bar
    # 3rd+ dendrite: PERMISSIVE (0 = any) - Add even with tiny improvements
    if improvement_threshold is None:
        improvement_threshold = [0.001, 0.0001, 0]  # From sweep: [0.001, 0.0001, 0]
    if hasattr(GPA.pc, 'set_improvement_threshold'):
        GPA.pc.set_improvement_threshold(improvement_threshold)
        print(f"  Improvement Threshold: {improvement_threshold} (from sweep optimization)")
    
    # Dendrite weight initialization - from WandB sweep (best config)
    # Lower value = gentler dendrite integration, reduces disruption to trained weights
    if hasattr(GPA.pc, 'set_candidate_weight_initialization_multiplier'):
        GPA.pc.set_candidate_weight_initialization_multiplier(0.005)  # From sweep: 0.005
        print(f"  Candidate Weight Init: 0.005 (from sweep optimization)")
    
    # CRITICAL: Silence BatchNorm warnings
    if hasattr(GPA.pc, 'set_unwrapped_modules_confirmed'):
        GPA.pc.set_unwrapped_modules_confirmed(True)
        print("  Unwrapped Modules: CONFIRMED (BatchNorm warnings silenced)")
    
    # Silence weight_decay recommendation warning (we use weight_decay=0 anyway)
    if hasattr(GPA.pc, 'set_weight_decay_accepted'):
        GPA.pc.set_weight_decay_accepted(True)
        print("  Weight Decay: ACCEPTED (warning silenced)")
    
    # CRITICAL FIX: Enable SafeTensors mode to fix .item() error in PB!
    # This makes PAI handle type conversions internally so perforatedbp works correctly
    if hasattr(GPA.pc, 'set_using_safe_tensors'):
        GPA.pc.set_using_safe_tensors(True)
        print("  SafeTensors Mode: ENABLED (fixes PB .item() error)")
    
    # ========== MODULE CONFIGURATION (Matching PAI Official Examples) ==========
    # Like EfficientNet example uses: append_module_names_to_convert(['MBConv', 'Conv2dNormActivation'])
    # We explicitly tell PAI which YOLO module types should get dendrites
    #
    # YOLOv11n Module Types:
    # - CONVERT (add dendrites): C3k2, C3k, C2PSA, Conv, Bottleneck, PSABlock, DWConv
    # - TRACK (no dendrites): BatchNorm2d, SiLU, Identity, Upsample, MaxPool2d
    # - EXCLUDE: Detect, DFL, SPPF (detection-specific)
    
    # Main feature extraction blocks - ADD DENDRITES TO THESE
    main_blocks_to_convert = ['C3k2', 'C3k', 'C2PSA', 'Bottleneck', 'PSABlock']
    if hasattr(GPA.pc, 'append_module_names_to_convert'):
        GPA.pc.append_module_names_to_convert(main_blocks_to_convert)
        print(f"  Module Types to CONVERT: {main_blocks_to_convert}")
    
    # Normalization/activation layers - TRACK but don't add dendrites
    layers_to_track = ['BatchNorm2d', 'SiLU', 'Identity', 'Upsample', 'MaxPool2d', 'Concat']
    if hasattr(GPA.pc, 'append_module_names_to_track'):
        GPA.pc.append_module_names_to_track(layers_to_track)
        print(f"  Module Types to TRACK: {layers_to_track}")
    
    # ========== PAI SAVE CONFIGURATIONS ==========
    # Enable test saves (best_model.pt + latest.pt)
    if hasattr(GPA.pc, 'set_test_saves'):
        GPA.pc.set_test_saves(True)
        print("  Test Saves: ENABLED (best_model.pt + latest.pt)")
    
    # Enable PAI saves (clean _pai.pt versions without optimizer state)
    if hasattr(GPA.pc, 'set_pai_saves'):
        GPA.pc.set_pai_saves(True)
        print("  PAI Saves: ENABLED (clean _pai.pt versions)")
    
    # Enable graph generation (4-panel training visualization)
    if hasattr(GPA.pc, 'set_making_graphs'):
        GPA.pc.set_making_graphs(True)
        print("  Making Graphs: ENABLED (4-panel visualization)")
    
    # Enable extra graphs (learning rate + time plots)
    if hasattr(GPA.pc, 'set_drawing_extra_graphs'):
        GPA.pc.set_drawing_extra_graphs(True)
        print("  Extra Graphs: ENABLED (LR + time plots)")
    
    # CRITICAL: Set candidate gradient clipping to prevent NaN/inf in candidate outputs
    # PAI recommends this when seeing "Got a NaN or inf in candidate outputs" warnings
    if hasattr(GPA.pc, 'set_candidate_grad_clipping'):
        GPA.pc.set_candidate_grad_clipping(1.0)
        print("  Candidate Grad Clipping: 1.0 (prevents NaN in dendrite outputs)")
    
    # CRITICAL FIX FOR HUGE DROPS: Reduce dendrite weight initialization
    # Default is 0.01, but this causes massive disruption when dendrites are added
    # PAI docs recommend trying 0.1 or 0.01 - we go even smaller for YOLO stability
    if hasattr(GPA.pc, 'set_candidate_weight_initialization_multiplier'):
        GPA.pc.set_candidate_weight_initialization_multiplier(0.001)
        print("  Dendrite Weight Init: 0.001 (10x smaller than default for stability)")
    
    # CRITICAL FIX FOR HUGE DROPS: Set PAI forward function to match YOLO's activations
    # Default is sigmoid, but YOLO uses SiLU throughout - ReLU is closest match
    # This prevents activation mismatch that causes output corruption
    if hasattr(GPA.pc, 'set_pai_forward_function'):
        import torch
        GPA.pc.set_pai_forward_function(torch.relu)
        print("  PAI Forward Function: ReLU (matches YOLO's SiLU activation pattern)")
    
    # Suppress covariance NaN warnings (informational, not critical)
    if hasattr(GPA.pc, 'set_covariance_warning_silenced'):
        GPA.pc.set_covariance_warning_silenced(True)
        print("  Covariance Warnings: SILENCED")
    
    # Suppress candidate output warnings (we already set clipping)
    if hasattr(GPA.pc, 'set_candidate_warning_silenced'):
        GPA.pc.set_candidate_warning_silenced(True)
        print("  Candidate Output Warnings: SILENCED")
    
    # Add BatchNorm modules to tracking (they have params but shouldn't get dendrites)
    # This silences "Parameter does not have parameter_type attribute" warnings
    if hasattr(GPA.pc, 'append_module_names_to_track'):
        GPA.pc.append_module_names_to_track(["BatchNorm2d", "BatchNorm1d", "SyncBatchNorm"])
        print("  BatchNorm Tracking: ENABLED (suppresses param warnings)")
    
    # CRITICAL FOR YOLO: Handle duplicate module pointers
    # YOLOv11n often shares activation modules between layers (e.g., SiLU)
    # We tell PAI to NOT save these to prevent the "duplicate pointer" error
    if hasattr(GPA.pc, 'set_duplicate_pointer_confirmed'):
        GPA.pc.set_duplicate_pointer_confirmed(True)
        print("  Duplicate Pointer: CONFIRMED")
    
    
    # NOTE: We can't configure specific module names here because we don't have the model object yet.
    # Dynamic detection of shared modules is done in initialize_pai_model instead.
    print(f"  YOLO Shared Module Handling: Done dynamically in initialize_pai_model")
    
    print("=" * 60 + "\n")


def normalize_pai_tracker_state():
    """
    Normalize PAI tracker internal state to prevent type errors.
    
    PAI's internal code sometimes expects tensors with .item() but receives ints,
    or vice versa. This comprehensively converts all numeric values to proper
    Python primitives to prevent AttributeError.
    
    This is called BEFORE each add_validation_score() call and after restructuring.
    """
    if not PAI_AVAILABLE:
        return
        
    try:
        tracker = GPA.pai_tracker
        if not hasattr(tracker, 'member_vars'):
            return
        
        normalized_count = 0
        
        # Normalize all member_vars values
        for key, val in tracker.member_vars.items():
            # Skip non-numeric types and special objects
            if val is None or isinstance(val, (bool, str, dict, list, type)):
                continue
                
            # Convert tensors to Python primitives
            if isinstance(val, torch.Tensor):
                try:
                    if val.numel() == 1:
                        tracker.member_vars[key] = float(val.item())
                        normalized_count += 1
                except:
                    pass
            # Convert numpy arrays
            elif hasattr(val, 'item') and callable(getattr(val, 'item')):
                try:
                    tracker.member_vars[key] = float(val.item())
                    normalized_count += 1
                except:
                    pass
        
        # Additionally, check for tensor values in accuracy-related lists
        list_fields = ['accuracies', 'n_accuracies', 'p_accuracies', 'running_accuracies', 
                       'test_accuracies', 'test_scores', 'training_loss', 'training_learning_rates']
        for field in list_fields:
            if field in tracker.member_vars and isinstance(tracker.member_vars[field], list):
                for i, item in enumerate(tracker.member_vars[field]):
                    if isinstance(item, torch.Tensor):
                        try:
                            tracker.member_vars[field][i] = float(item.item())
                            normalized_count += 1
                        except:
                            pass
        
        if normalized_count > 0:
            print(f"  [PAI] Normalized {normalized_count} tracker state values")
                    
    except Exception as e:
        print(f"  [PAI] Could not normalize tracker state: {e}")


# =============================================================================
# SECTION 4: MODEL HANDLING
# =============================================================================

def extract_yolo_model(yolo_wrapper: YOLO, train: bool = True) -> nn.Module:
    """
    Extract the raw PyTorch model from Ultralytics YOLO wrapper.
    
    This is needed because PAI needs direct access to nn.Module,
    not the YOLO wrapper class.
    
    CRITICAL FIX: We use deepcopy because yolo_wrapper.model may be a
    property that returns different objects, causing our modifications
    to be lost.
    
    Args:
        yolo_wrapper: Ultralytics YOLO model wrapper
        train: If True, enable gradients on all parameters (default: True)
        
    Returns:
        Raw PyTorch nn.Module with trainable parameters
    """
    import copy
    
    # CRITICAL: Use deepcopy to ensure we OWN this model
    # This fixes the issue where modifications to requires_grad are lost
    original_model = yolo_wrapper.model
    model = copy.deepcopy(original_model)
    
    print(f"[YOLO] Extracted model via deepcopy")
    print(f"[YOLO] Original model id: {id(original_model)}, Copy id: {id(model)}")
    
    # CRITICAL FIX: Copy args from YOLO wrapper which has correct initialization
    # The YOLO wrapper stores all hyperparameters we need for loss computation
    # Don't try to recreate - just use what Ultralytics already set up
    if hasattr(yolo_wrapper, 'overrides') and yolo_wrapper.overrides:
        # Ultralytics stores training config in .overrides
        if not hasattr(model, 'args') or model.args is None:
            model.args = SimpleNamespace(**yolo_wrapper.overrides)
            print("[YOLO] Copied args from wrapper.overrides")
    
    # Ensure minimum required hyperparameters exist
    required_hyp = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'fl_gamma': 0.0,  # Focal loss gamma
        'mosaic': 1.0,
        'copy_paste': 0.0,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'mixup': 0.0,
    }
    
    # CRITICAL FIX: Ensure model.args is ALWAYS a SimpleNamespace, not a dict
    if hasattr(model, 'args') and model.args is not None:
        if isinstance(model.args, dict):
            # Convert dict to SimpleNamespace, merging with required_hyp
            merged = {**required_hyp, **model.args}  # model.args overwrites defaults
            model.args = SimpleNamespace(**merged)
            print("[YOLO] Converted model.args from dict to SimpleNamespace")
        else:
            # Already an object (SimpleNamespace or IterableSimpleNamespace)
            # Just ensure required fields exist
            for k, v in required_hyp.items():
                if not hasattr(model.args, k):
                    setattr(model.args, k, v)
    else:
        model.args = SimpleNamespace(**required_hyp)
        print("[YOLO] Created model.args with required hyperparameters")
    
    # VERIFY: Ensure model.args.box is accessible (critical for loss function)
    try:
        _ = model.args.box
        print(f"[YOLO] ✓ model.args verified: box={model.args.box}, cls={model.args.cls}, dfl={model.args.dfl}")
    except AttributeError as e:
        print(f"[YOLO] ✗ model.args.box FAILED! Type: {type(model.args)}, Error: {e}")
        # Force create as SimpleNamespace
        model.args = SimpleNamespace(**required_hyp)
        print(f"[YOLO] Force-created model.args as SimpleNamespace")

    if train:
        # CRITICAL: Enable gradients on ALL parameters
        # Ultralytics freezes pretrained weights by default
        model.train()  # Set to training mode
        for param in model.parameters():
            param.requires_grad = True
        
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[YOLO] Enabled requires_grad on {trainable} parameters")
    
    return model




def check_gradients(model: nn.Module):
    """Debug utility to verify model gradients are enabled."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    
    print(f"[GradCheck] Trainable params: {len(trainable)}")
    print(f"[GradCheck] Frozen params: {len(frozen)}")
    
    if len(trainable) == 0:
        print("CRITICAL WARNING: No trainable parameters found! Enabling now...")
        for param in model.parameters():
            param.requires_grad = True
        print("[GradCheck] Force-enabled gradients on all parameters.")

def initialize_pai_model(
    model: nn.Module,
    save_name: str = "PAI_YOLO",
    maximizing_score: bool = True
) -> nn.Module:
    """
    Initialize PAI on a PyTorch model.
    """
    if not PAI_AVAILABLE:
        raise RuntimeError("PerforatedAI not installed.")
    
    # Check gradients before PAI init
    check_gradients(model)
    
    # CRITICAL FIX: PAI has a bug in tracker_perforatedai.py line 2970:
    #   save_folder = "./" + self.save_name + "/"
    # This breaks when save_name is an absolute path.
    # 
    # SOLUTION: Pass ONLY the folder basename to PAI, create full dir ourselves.
    import os
    
    # Get the full absolute path for directory creation
    full_save_path = os.path.abspath(save_name)
    
    # Extract just the folder name (basename) for PAI
    save_name_basename = os.path.basename(full_save_path)
    
    print(f"[PAI] Full save path: '{full_save_path}'")
    print(f"[PAI] Using basename for PAI: '{save_name_basename}'")
    print(f"[PAI] maximizing_score={maximizing_score}")
    
    # Pre-create the directory with full path
    save_dir = Path(full_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create the relative directory that PAI will use
    os.makedirs(save_name_basename, exist_ok=True)
    
    # CRITICAL: Automatically handle shared modules using robust ID detection
    # Ultralytics models reuse activation modules (SiLU) extensively in deeply nested blocks
    # We use a comprehensive graph traversal to find ALL modules that point to the same object
    # and tell PAI to ignore the duplicates.
    #
    # IMPORTANT: We need TWO passes:
    # 1. Instance-level submodules (via named_modules with remove_duplicate=False)
    # 2. Class-level attributes (like Conv.default_act = nn.SiLU())
    
    from collections import defaultdict
    id_to_names = defaultdict(list)
    
    print("[PAI] Scanning model for shared modules (comprehensive detection)...")
    
    # PASS 1: Find all registered submodules (including duplicates)
    for name, module in model.named_modules(remove_duplicate=False):
        id_to_names[id(module)].append(name)
        
        # PASS 2: Also check CLASS-LEVEL attributes that are nn.Module instances
        # This catches things like Conv.default_act = nn.SiLU() which is shared across ALL Conv instances
        module_class = type(module)
        for attr_name in dir(module_class):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(module_class, attr_name)
                if isinstance(attr_value, nn.Module):
                    full_name = f"{name}.{attr_name}" if name else attr_name
                    id_to_names[id(attr_value)].append(full_name)
            except Exception:
                pass  # Skip attributes that can't be accessed
        
    shared_module_names = []
    duplicate_count = 0
    
    for mod_id, names in id_to_names.items():
        if len(names) > 1:
            # Sort to ensure deterministic behavior (shortest name usually best to keep)
            names.sort(key=len)
            
            # The first one is the "canonical" one we keep.
            # All others are duplicates that PAI should ignore.
            duplicates = names[1:]
            
            # CRITICAL FIX: PAI expects module names to start with a dot "."
            # named_modules() returns "model.0.act", but PAI knows it as ".model.0.act"
            # We must prepend the dot for the configuration to work.
            formatted_duplicates = ["." + name if not name.startswith(".") else name for name in duplicates]
            
            shared_module_names.extend(formatted_duplicates)
            duplicate_count += len(duplicates)

    if shared_module_names:
        print(f"[PAI] Auto-detected {duplicate_count} unique duplicate pointers (including class-level).")
        
        # Configure PAI to ignore these patterns
        if hasattr(GPA.pc, 'append_module_names_to_not_save'):
            GPA.pc.append_module_names_to_not_save(shared_module_names)
            print(f"[PAI] Successfully configured PAI to ignore {len(shared_module_names)} duplicates.")
    
    # CRITICAL FIX: Tell PAI to SKIP detection head layers (DFL, Detect)
    # These layers have special behavior that doesn't work with PAI dendrites.
    # The PAI function is append_module_ids_to_track, NOT set_layers_to_skip!
    # 
    # YOLO Detection Head Structure (.model.23 = Detect module):
    # - .model.23.dfl - Distribution Focal Loss
    #   - .model.23.dfl.conv - DFL conv (causes "dendrite_values missing" error)
    # - .model.23.cv2 - Box regression heads (one per scale: 0, 1, 2)
    #   - .model.23.cv2.0.0, .model.23.cv2.0.1, .model.23.cv2.0.2 - nested convs
    # - .model.23.cv3 - Classification heads (one per scale: 0, 1, 2)
    #   - Similar nested structure
    #
    # ADDITIONAL EXCLUSIONS FOR STABILITY:
    # - .model.22 - Final neck block (feeds directly to detection head)
    # - .model.21 - Second-to-last neck block  
    # - .model.9 - SPPF (Spatial Pyramid Pooling Fast) - critical pooling
    #
    # We need to exclude ALL of these to prevent state dict mismatches!
    detection_head_modules = [
        # Top-level detection module
        '.model.23',
        
        # DFL module and its conv
        '.model.23.dfl',
        '.model.23.dfl.conv',
        
        # Box regression heads (cv2) - all scales and all nested convs
        '.model.23.cv2',
        '.model.23.cv2.0', '.model.23.cv2.1', '.model.23.cv2.2',
        '.model.23.cv2.0.0', '.model.23.cv2.0.1', '.model.23.cv2.0.2',
        '.model.23.cv2.1.0', '.model.23.cv2.1.1', '.model.23.cv2.1.2',
        '.model.23.cv2.2.0', '.model.23.cv2.2.1', '.model.23.cv2.2.2',
        
        # Classification heads (cv3) - all scales and all nested convs
        '.model.23.cv3',
        '.model.23.cv3.0', '.model.23.cv3.1', '.model.23.cv3.2',
        '.model.23.cv3.0.0', '.model.23.cv3.0.1',
        '.model.23.cv3.1.0', '.model.23.cv3.1.1',
        '.model.23.cv3.2.0', '.model.23.cv3.2.1',
        '.model.23.cv3.0.0.0', '.model.23.cv3.0.0.1',
        '.model.23.cv3.0.1.0', '.model.23.cv3.0.1.1',
        '.model.23.cv3.1.0.0', '.model.23.cv3.1.0.1',
        '.model.23.cv3.1.1.0', '.model.23.cv3.1.1.1',
        '.model.23.cv3.2.0.0', '.model.23.cv3.2.0.1',
        '.model.23.cv3.2.1.0', '.model.23.cv3.2.1.1',
        
        # NOTE: .model.21 and .model.22 (neck blocks) NOW ALLOWED for dendrites
        # Previously excluded as "too conservative" - now enabling for better dendrite impact
        
        # CRITICAL: Exclude SPPF - critical pooling module
        '.model.9',   # SPPF block
    ]
    
    if hasattr(GPA.pc, 'append_module_ids_to_track'):
        GPA.pc.append_module_ids_to_track(detection_head_modules)
        print(f"[PAI] Excluding {len(detection_head_modules)} critical YOLO modules from conversion (head + neck + SPPF)")
    
    # Initialize PAI - use ONLY the basename to avoid path concatenation bugs
    # making_graphs=True saves all PAI visualization artifacts
    try:
        model = UPA.initialize_pai(
            model,
            save_name=save_name_basename,  # CRITICAL: basename only!
            making_graphs=True,
            maximizing_score=maximizing_score
        )
        print(f"[PAI] Model initialized successfully")
    except (AttributeError, KeyError) as e:
        error_str = str(e)
        if "bias" in error_str or "already exists" in error_str:
            print(f"[PAI] WARNING: 'bias already exists' error detected.")
            print(f"[PAI] Error details: {e}")
            print(f"[PAI] This is a known PAI-YOLO compatibility issue.")
            print(f"[PAI] Attempting workaround: reinitialize with fresh model...")
            
            # Workaround: Re-load a fresh YOLO model and try again
            try:
                # Reload fresh model
                from ultralytics import YOLO
                fresh_yolo = YOLO('yolo11n.pt')
                fresh_model = fresh_yolo.model
                fresh_model.train()
                for param in fresh_model.parameters():
                    param.requires_grad = True
                
                # Retry PAI initialization
                model = UPA.initialize_pai(
                    fresh_model,
                    save_name=save_name_basename,
                    making_graphs=False,  # Skip graphs to reduce complexity
                    maximizing_score=maximizing_score
                )
                print(f"[PAI] Model initialized successfully on fresh reload")
            except Exception as e2:
                print(f"[PAI] CRITICAL ERROR: PAI initialization failed even with fresh model: {e2}")
                print(f"[PAI] Falling back to training WITHOUT dendrites (baseline mode)...")
                # Return unmodified model - will train as baseline
                return model
        else:
            raise
    
    return model


def setup_pai_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    scheduler_patience: int = 10
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    Setup optimizer and scheduler through PAI tracker.
    
    CRITICAL: 
    - weight_decay should be 0 for PAI (helps dendrite stability)
    - scheduler patience MUST be < n_epochs_to_switch
    
    Args:
        model: PAI-initialized model
        lr: Learning rate
        weight_decay: Weight decay (default 0 for PAI)
        scheduler_patience: Patience before LR reduction
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    if not PAI_AVAILABLE:
        raise RuntimeError("PerforatedAI not installed.")
    
    # Register optimizer and scheduler types with PAI
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    # Optimizer arguments - NOTE: DO NOT pass 'params' here!
    # PAI's setup_optimizer will handle getting params based on mode (n vs p)
    # Passing model.parameters() directly can result in empty list if filtered
    optim_args = {
        'lr': lr,
        'weight_decay': weight_decay  # Should be 0 for PAI!
    }
    
    # Scheduler arguments
    # CRITICAL: patience MUST be < n_epochs_to_switch!
    sched_args = {
        'mode': 'max',  # We're maximizing score (negative loss or mAP)
        'patience': scheduler_patience,
        'factor': 0.1,  # Standard PyTorch default, used by most PAI examples
        'min_lr': 1e-6
    }
    
    print(f"[PAI Optimizer] lr={lr}, weight_decay={weight_decay}")
    print(f"[PAI Scheduler] patience={scheduler_patience}, factor=0.1")
    
    # Setup through PAI tracker
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optim_args, sched_args
    )
    
    return optimizer, scheduler


# =============================================================================
# SECTION 5: TRAINING FUNCTIONS
# =============================================================================

def preprocess_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Preprocess batch by moving tensors to device and normalizing images.
    
    This replicates what Ultralytics' BaseTrainer.preprocess_batch() does.
    CRITICAL: Images MUST be normalized to [0, 1] for YOLO models!
    
    Args:
        batch: Batch dictionary from YOLO dataloader
        device: Target device
        
    Returns:
        Preprocessed batch dictionary
    """
    # Move image to device and normalize to [0, 1]
    batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255.0
    
    # Move other tensors to device if they exist
    if 'bboxes' in batch:
        batch['bboxes'] = batch['bboxes'].to(device)
    if 'cls' in batch:
        batch['cls'] = batch['cls'].to(device)
    if 'batch_idx' in batch:
        batch['batch_idx'] = batch['batch_idx'].to(device)
    
    return batch

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    log_interval: int = 10,
    suppress_pai_output: bool = True,
    ema: Optional['ModelEMA'] = None
) -> float:
    """
    Train for one epoch and return average loss.
    
    IMPORTANT: Ultralytics YOLO model.forward() when passed a batch dict
    returns (loss, loss_items) tuple, NOT just loss.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        scaler: Optional GradScaler for mixed precision (GPU only)
        epoch: Current epoch number (for logging)
        log_interval: How often to log batch progress
        suppress_pai_output: If True, suppress stdout during first batch (hides PAI d_shape msgs)
        ema: Optional ModelEMA for exponential moving average
        
    Returns:
        Average loss for the epoch
    """
    import sys
    import io
    from tqdm import tqdm
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(train_loader)
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        # Preprocess batch - move tensors to device
        # Ultralytics batch dict contains: 'img', 'cls', 'bboxes', 'batch_idx', etc.
        batch = preprocess_batch(batch, device)
        
        # set_to_none=True is slightly faster than zeroing
        optimizer.zero_grad(set_to_none=True)
        
        # PAI prints verbose warnings during forward pass (NaN covariance, candidate outputs)
        # Suppress stdout during forward/backward pass to hide these
        # ALWAYS suppress for all batches to keep output clean
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # Forward pass with optional mixed precision
            if scaler is not None:
                # GPU with mixed precision
                with torch.cuda.amp.autocast():
                    result = model(batch)
                    if isinstance(result, tuple):
                        loss = result[0]
                        if loss.dim() > 0:
                            loss = loss.sum()
                    elif isinstance(result, dict):
                        loss = sum(result.values())
                    else:
                        loss = result
                
                # DEBUG: Check if loss has gradients
                if batch_idx == 0 and epoch == 1:
                    print(f"\n  [DEBUG] Loss type: {type(loss)}")
                    print(f"  [DEBUG] Loss value: {loss.item():.4f}")
                    print(f"  [DEBUG] Loss requires_grad: {loss.requires_grad}")
                    print(f"  [DEBUG] Loss grad_fn: {loss.grad_fn}")
                    
                    if not loss.requires_grad:
                        print("  [DEBUG] ⚠️ Loss has no gradients! Investigating...")
                        # Check model parameters
                        trainable = sum(1 for p in model.parameters() if p.requires_grad)
                        print(f"  [DEBUG] Model trainable params: {trainable}")
                
                # If loss doesn't have grad, this will fail
                if not loss.requires_grad:
                    raise RuntimeError(
                        f"Loss tensor has no gradient (requires_grad={loss.requires_grad}, "
                        f"grad_fn={loss.grad_fn}). Model params with grad: "
                        f"{sum(1 for p in model.parameters() if p.requires_grad)}"
                    )
                
                scaler.scale(loss).backward()
                
                # Check if optimizer params have gradients BEFORE any scaler operations
                # In PAI P-mode, optimizer only has dendrite params (subset of model params)
                # If those params don't have gradients, skip all scaler ops to avoid state issues
                has_grads = any(
                    p.grad is not None for group in optimizer.param_groups for p in group['params']
                )
                
                if has_grads:
                    # Normal AMP path: unscale, clip, step, update
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Fallback when optimizer params have no gradients (PAI P-mode edge case)
                    # Skip ALL scaler operations to avoid "unscale already called" error
                    # Still clip gradients for model stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                    if batch_idx == 0:
                        print(f"  [AMP] Manual step (optimizer params have no gradients)")
                
                # EMA update disabled for now - testing if this causes high loss
                # if ema is not None:
                #     ema.update(model)
            else:
                # CPU or GPU without mixed precision
                result = model(batch)
                if isinstance(result, tuple):
                    loss = result[0]
                    if loss.dim() > 0:
                        loss = loss.sum()
                elif isinstance(result, dict):
                    loss = sum(result.values())
                else:
                    loss = result
                
                # DEBUG for first batch
                if batch_idx == 0 and epoch == 1:
                    print(f"\n  [DEBUG] Loss requires_grad: {loss.requires_grad}")
                    if not loss.requires_grad:
                        trainable = sum(1 for p in model.parameters() if p.requires_grad)
                        print(f"  [DEBUG] Model trainable params: {trainable}")
                
                if not loss.requires_grad:
                    raise RuntimeError(
                        f"Loss tensor has no gradient. This usually means the model's "
                        f"parameters don't have requires_grad=True or the loss computation "
                        f"is detached."
                    )
                
                loss.backward()
                
                # CRITICAL: Gradient clipping matching official Ultralytics
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                
                # EMA update disabled for now - testing if this causes high loss
                # if ema is not None:
                #     ema.update(model)
        finally:
            # Always restore stdout (we suppress for all batches now)
            sys.stdout = old_stdout
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update tqdm with current loss
        avg_loss = total_loss / num_batches
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    pbar.close()
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss




@torch.no_grad()
def validate(
    yolo_wrapper: YOLO,
    data_yaml: str,
    imgsz: int = 640,
    split: str = 'val'
) -> dict:
    """
    Run validation using Ultralytics built-in validator to get comprehensive metrics.
    
    IMPORTANT: Ultralytics val() internally disables requires_grad on model parameters.
    This function restores gradients after validation to allow training to continue.
    
    Args:
        yolo_wrapper: YOLO model wrapper (with current weights)
        data_yaml: Path to dataset YAML
        imgsz: Image size for validation
        split: Data split to validate on ('val' or 'test')
        
    Returns:
        Dict with: mAP50, mAP50-95, precision, recall
    """
    import contextlib
    import os
    
    print("  Validating...", end='', flush=True)
    
    model = yolo_wrapper.model
    
    # CRITICAL: Save requires_grad state BEFORE validation
    # Ultralytics val() internally modifies the model (fuse, half, eval mode)
    # which can permanently disable requires_grad on parameters
    param_grad_state = {name: p.requires_grad for name, p in model.named_parameters()}
    
    # Prevent model.fuse() call during validation
    original_is_fused = getattr(model, 'is_fused', None)
    model.is_fused = lambda: True
    
    try:
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            results = yolo_wrapper.val(
                data=data_yaml,
                imgsz=imgsz,
                split=split,
                plots=False,
                save=False,
                verbose=False
            )
    finally:
        # Restore original is_fused method
        if original_is_fused is not None:
            model.is_fused = original_is_fused
        elif hasattr(model, 'is_fused'):
            delattr(model, 'is_fused')
        
        # CRITICAL: Restore requires_grad state after validation
        # This undoes any gradient disabling that Ultralytics val() performed
        for name, p in model.named_parameters():
            if name in param_grad_state:
                p.requires_grad = param_grad_state[name]
        
        # Also ensure model is back in training mode
        model.train()
    
    print("\r", end='')  # Clear "Validating..." line
    
    # Return comprehensive metrics
    return {
        'map50': float(results.box.map50),      # mAP@0.5
        'map': float(results.box.map),          # mAP@0.5:0.95
        'precision': float(results.box.mp),     # Mean precision
        'recall': float(results.box.mr)         # Mean recall
    }


# =============================================================================
# SECTION 4.5: EARLY STOPPING UTILITY
# =============================================================================
class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('-inf')
        self.best_epoch = 0
        self.early_stop = False
        self.score_history = []  # Track recent scores

    def __call__(self, score: float, epoch: int):
        self.score_history.append((epoch, score))
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def get_summary(self):
        """Get summary of recent performance for early stop visualization"""
        recent = self.score_history[-min(10, len(self.score_history)):]
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'recent_scores': recent
        }


def train_pai_yolo(
    data_yaml: str,
    epochs: int = 300,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = 'cuda',
    lr: float = 0.005,
    warmup_epochs: int = 3,  # NEW: Warmup period
    save_name: str = 'PAI_YOLO',
    pretrained: str = 'yolo11n.pt',
    seed: int = 42,
    n_epochs_to_switch: int = 10,
    p_epochs_to_switch: int = 10,
    max_dendrites: int = 10,
    early_stop_patience: int = 10,
    use_pai: bool = True,
    use_perforated_bp: bool = True,  # True=PerforatedBP, False=Open Source GD
    # NEW: Sweep-specific PAI parameters
    history_lookback: int = 4,
    improvement_threshold: list = None,  # [0.01, 0.001, 0.0001, 0] or [0.001, 0.0001, 0]
    candidate_weight_init: float = None,  # 0.1 or 0.01
    pai_forward_function: str = None,  # 'sigmoid', 'relu', 'tanh'
    data_fraction: float = 1.0,  # Fraction of training data to use (0.5 = 50%)
    switch_mode: str = 'DOING_HISTORY',  # 'DOING_HISTORY' or 'DOING_FIXED'
    scheduler_patience: int = None,  # Scheduler patience (if None, auto-calculated)
    find_best_lr: bool = False  # PAI's automatic LR search (False uses manual decay)
) -> Tuple[nn.Module, float]:
    """
    Complete Unified Training Function (PAI & Baseline).
    
    Guarantees IDENTICAL data loading, processing, and evaluation
    for both Baseline and PAI experiments.
    """
    if use_pai and not PAI_AVAILABLE:
        raise RuntimeError("PerforatedAI not installed but use_pai=True requested.")
    
    print("\n" + "=" * 60)
    print(f"  {'PAI' if use_pai else 'BASELINE'} YOLO TRAINING")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup device - GPU or CPU
    if isinstance(device, str):
        # Accept both 'cuda' and 'gpu' as GPU device strings
        if device in ('cuda', 'gpu') and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print(f"Device: CPU")
    else:
        # Already a torch.device object
        print(f"Device: {device}")
    
    # ========== STEP 1: Setup PAI Configuration (If Enabled) ==========
    if use_pai:
        setup_pai_config(
            n_epochs_to_switch=n_epochs_to_switch,
            p_epochs_to_switch=p_epochs_to_switch,
            max_dendrites=max_dendrites,
            use_perforated_bp=use_perforated_bp,
            history_lookback=history_lookback,
            improvement_threshold=improvement_threshold,
            switch_mode=switch_mode,  # Pass switch_mode to setup
            find_best_lr=find_best_lr  # Pass find_best_lr to setup
        )
        
        # Apply sweep-specific overrides after setup_pai_config
        if candidate_weight_init is not None:
            GPA.pc.set_candidate_weight_initialization_multiplier(candidate_weight_init)
            print(f"  Candidate Weight Init: {candidate_weight_init}")
        
        if pai_forward_function is not None:
            if pai_forward_function == 'sigmoid':
                GPA.pc.set_pai_forward_function(torch.sigmoid)
            elif pai_forward_function == 'relu':
                GPA.pc.set_pai_forward_function(torch.relu)
            elif pai_forward_function == 'tanh':
                GPA.pc.set_pai_forward_function(torch.tanh)
            print(f"  PAI Forward Function: {pai_forward_function}")
    else:
        print("[Config] Running in standard Baseline mode (No PAI)")
    
    # ========== STEP 2: Load YOLO Model ==========
    print(f"\n[Model] Loading pretrained: {pretrained}")
    yolo = YOLO(pretrained)
    model = extract_yolo_model(yolo)
    model = model.to(device)
    
    # CRITICAL: Verify gradients for baseline AND PAI
    check_gradients(model)
    
    # ========== STEP 3: Initialize PAI (If Enabled) ==========
    if use_pai:
        model = initialize_pai_model(
            model,
            save_name=save_name,
            maximizing_score=True
        )
    model = model.to(device)
    
    # ========== STEP 4: Setup Data Loaders (Shared logic) ==========
    print(f"\n[Data] Loading dataset from: {data_yaml}")
    
    data_dict = check_det_dataset(data_yaml)
    
    # Merge basic config with augmentations
    # CRITICAL: Ultralytics expects a config object with attributes, not a dictionary
    # We use SimpleNamespace to convert our dict to an object with .attribute access
    from types import SimpleNamespace
    
    # Defaults required by build_yolo_dataset
    default_cfg = {
        'cache': None,
        'single_cls': False,
        'classes': None,
        'fraction': data_fraction,  # Use data_fraction parameter for 50% data sweep
        'task': 'detect',
        'rect': False,
        'imgsz': imgsz,
        'project': 'runs',
        'name': 'exp',
        'close_mosaic': 0,
        # Default keys for robustness (Augmentations defaulting to 0/None if not in AUGMENTATION_CONFIG)
        'cutmix': 0.0,
        'crop_fraction': 1.0,
        'auto_augment': None,
        'mask_ratio': 4,
        'overlap_mask': True,
        'bgr': 0.0, # Image channel flip probability
        'copy_paste_mode': 'flip', # [CRITICAL] Must be 'flip' or 'mixup', accessed in v8_transforms
    }
    
    # Train Config
    train_cfg_dict = default_cfg.copy()
    train_cfg_dict.update(AUGMENTATION_CONFIG) # Apply augmentations
    train_cfg_dict['rect'] = False # Mosaic requires rect=False
    train_cfg = SimpleNamespace(**train_cfg_dict)
    
    if data_fraction < 1.0:
        print(f"[Data] Using {int(data_fraction * 100)}% of training data")
    print(f"[Augmentation] Mosaic: {train_cfg.mosaic}, FlipLR: {train_cfg.fliplr}, Scale: {train_cfg.scale}")
    
    train_dataset = build_yolo_dataset(
        cfg=train_cfg,
        img_path=data_dict.get('train', ''),
        batch=batch_size,
        data=data_dict,
        mode='train'
    )
    
    train_loader = build_dataloader(
        train_dataset,
        batch=batch_size,
        workers=8,  # Increased for faster GPU performance
        shuffle=True,
        pin_memory=True  # Enabled for speed
    )
    
    # Validation Config
    val_cfg_dict = default_cfg.copy()
    val_cfg_dict['rect'] = True
    val_cfg_dict['mosaic'] = 0.0 # No mosaic for validation
    val_cfg = SimpleNamespace(**val_cfg_dict)
    
    val_dataset = build_yolo_dataset(
        cfg=val_cfg,
        img_path=data_dict.get('val', ''),
        batch=batch_size,
        data=data_dict,
        mode='val'
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch=batch_size,
        workers=8,  # Increased for faster validation
        shuffle=False,
        pin_memory=True  # Enabled for speed
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ========== STEP 5: Setup Optimizer ==========
    # Use provided scheduler_patience or auto-calculate
    if scheduler_patience is None:
        # Auto-calculate: for HISTORY mode, scheduler should be < n_epochs_to_switch
        # For FIXED mode with large n_epochs, use a fixed smaller value
        if n_epochs_to_switch <= 20:
            scheduler_patience = max(1, n_epochs_to_switch - 3) if use_pai else 5
        else:
            scheduler_patience = 15  # For FIXED mode with n_epochs=50
    print(f"  Scheduler patience: {scheduler_patience}")
    
    if use_pai:
        optimizer, scheduler = setup_pai_optimizer(
            model,
            lr=lr,
            weight_decay=0,
            scheduler_patience=scheduler_patience
        )
    else:
        # Standard Optimizer with ReduceLROnPlateau (MATCHING DENDRITIC FOR FAIR COMPARISON)
        # Previously used CosineAnnealing which forced LR decay regardless of improvement
        print("[Optimizer] Using AdamW + ReduceLROnPlateau (matching dendritic setup)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        # Use same scheduler as dendritic for fair comparison
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize mAP50
            factor=0.1,   # Standard PyTorch default (same as PAI examples)
            patience=scheduler_patience
        )
        print(f"[Scheduler] ReduceLROnPlateau: patience={scheduler_patience}, factor=0.1")
    
    # Warmup scheduler - gradual LR increase for first few epochs
    def get_warmup_factor(epoch):
        """Returns LR multiplier for warmup period."""
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    print(f"[Warmup] Using {warmup_epochs} warmup epochs")
    
    # Mixed precision scaler - only for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("[AMP] Using CUDA mixed precision")
    
    # Create save directory
    save_dir = Path(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 6: Setup Early Stopping (Standard logic) ==========
    # Use patience passed from arguments (different for PAI vs Baseline)
    es_patience = early_stop_patience
    early_stopper = EarlyStopper(patience=es_patience)
    print(f"[EarlyStopping] Patience: {es_patience} epochs")
    
    # EMA is disabled (commented out in training loop at line ~1199)
    
    # ========== STEP 7: Training Loop ==========
    best_score = float('-inf')
    best_model_state = None  # Store copy of best model's state_dict for proper evaluation
    training_history = []
    
    print(f"\n{'=' * 60}")
    print(f"  STARTING TRAINING: {epochs} epochs")
    print(f"{'=' * 60}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'─' * 70}")
        
        # ========== APPLY WARMUP LR SCALING ==========
        warmup_factor = get_warmup_factor(epoch - 1)  # 0-indexed internally
        if warmup_factor < 1.0:
            # Safety check: ensure optimizer has param_groups (not a list from PAI error)
            if hasattr(optimizer, 'param_groups'):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * warmup_factor
                print(f"  [Warmup] LR scaled to: {lr * warmup_factor:.6f}")
            else:
                print(f"  [Warmup] Skipped - optimizer not properly initialized")
        
        # NOTE: Gradient restoration is handled inside validate() function.
        # Ultralytics val() disables requires_grad, so validate() saves and restores it.
        
        # ========== TRAIN ONE EPOCH ==========
        # Critical errors will crash the run - this is correct behavior
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch
        )
        
        # ========== VALIDATE ==========
        # NOTE: Using training model for validation (not EMA, disabled)
        # EMA needs many updates to warm up - at tau=2000, only ~50% after 1400 updates
        # For short training runs, training model gives more accurate metrics
        yolo.model = model
        val_metrics = validate(yolo, data_yaml, imgsz, split='val')
        val_score = val_metrics['map50']  # Primary metric for PAI
        
        # ========== DISPLAY METRICS TABLE ==========
        print(f"\n  📊 TRAINING METRICS (Epoch {epoch})")
        print(f"  {'─' * 66}")
        print(f"  {'Metric':<20} {'Value':>12} {'Status':>15}")
        print(f"  {'─' * 66}")
        print(f"  {'Train Loss':<20} {train_loss:>12.4f} {'':>15}")
        print(f"  {'Val mAP@0.5':<20} {val_metrics['map50']:>12.4f} {'(PRIMARY)':>15}")
        print(f"  {'Val mAP@0.5:0.95':<20} {val_metrics['map']:>12.4f} {'':>15}")
        print(f"  {'Val Precision':<20} {val_metrics['precision']:>12.4f} {'':>15}")
        print(f"  {'Val Recall':<20} {val_metrics['recall']:>12.4f} {'':>15}")
        
        restructured = False
        training_complete = False
        pai_mode = "N"  # Initialize for PAI status display
        pai_dendrites = 0
        
        # ========== Report to PAI (Only if use_pai) ==========
        if use_pai:
            # SAFETY: Ensure val_score is a valid Python float for PAI
            # PAI's internal code expects a float, not tensor or invalid values
            import math
            
            # Normalize to Python float regardless of input type
            if isinstance(val_score, torch.Tensor):
                val_score = float(val_score.detach().cpu().item())
            elif hasattr(val_score, 'item'):
                val_score = float(val_score.item())
            else:
                val_score = float(val_score)
            
            # Handle NaN/inf - replace with tiny positive value
            if math.isnan(val_score) or math.isinf(val_score):
                val_score = 1e-8
                print(f"  [PAI] Warning: val_score was NaN/inf, using {val_score}")
            
            # Ensure non-negative (PAI expects positive scores when maximizing)
            if val_score < 0:
                val_score = abs(val_score)
            
            # Very small scores can cause issues - use minimum threshold
            if val_score < 1e-10:
                val_score = 1e-8
            
            try:
                # Call PAI's add_validation_score directly
                # SafeTensors mode (enabled in setup_pai_config) handles type conversions internally
                # Don't suppress stdout - PAI needs to print important state changes
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                    val_score, model
                )
                
                model = model.to(device)
                
                # Update PAI mode and dendrite info for display
                try:
                    if hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'member_vars'):
                        tracker = GPA.pai_tracker
                        pai_mode = tracker.member_vars.get('mode', 'N').upper()
                        pai_dendrites = tracker.member_vars.get('num_dendrites_added', 0)
                except:
                    pass
                
            except Exception as e:
                # Just print the error and continue training
                print(f"  [PAI] Error on dendrite addition: {type(e).__name__}: {e}")
                restructured = False
                training_complete = False
            
            # NOTE: Do NOT force requires_grad=True here.
            # PAI manages parameter training via its optimizer setup.
            # Forcing re-enable would interfere with PAI's phased training.
            
            if restructured:
                print(f"\n  {'[PAI]':<20} {'⚡ DENDRITES ADDED! Resetting optimizer...':<46}")
                
                # Determine post-dendrite LR based on find_best_lr setting
                # Following PAI's approach:
                # - When find_best_lr=True: PAI auto-tests LRs, use base LR here
                # - When find_best_lr=False: Manually reduce LR (scheduler handles rest)
                
                if not find_best_lr:
                    # Manual LR decay (only when PAI's auto-search is disabled)
                    # Get current dendrite count for progressive LR decay
                    current_dendrites = 1  # Default to 1st dendrite
                    try:
                        if hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'member_vars'):
                            current_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 1)
                    except:
                        pass
                    
                    # PROGRESSIVE LR DECAY: Each dendrite gets lower LR for stability
                    # 1st dendrite: LR/5  (aggressive, model still has room to improve)
                    # 2nd dendrite: LR/10 (moderate, fine-tuning on top of 1st dendrite)
                    # 3rd+ dendrite: LR/20 (conservative, avoid disrupting learned patterns)
                    if current_dendrites == 1:
                        lr_divisor = 5.0
                    elif current_dendrites == 2:
                        lr_divisor = 10.0
                    else:
                        lr_divisor = 20.0
                    
                    post_dendrite_lr = lr / lr_divisor
                    print(f"  {'[PAI]':<20} Manual LR decay: Dendrite #{current_dendrites}, LR/{lr_divisor:.0f} = {post_dendrite_lr:.6f}")
                else:
                    # When find_best_lr=True, use base LR (PAI tests LRs internally)
                    # Following PAI EfficientNet example: just use base 'learning_rate'
                    post_dendrite_lr = lr
                    print(f"  {'[PAI]':<20} Using base LR: {post_dendrite_lr:.6f} (PAI will auto-test LRs)")
                
                optimizer, scheduler = setup_pai_optimizer(
                    model,
                    lr=post_dendrite_lr,
                    weight_decay=0,
                    scheduler_patience=scheduler_patience
                )
                
                # DIAGNOSTIC: Check trainable params after restructuring
                trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                total_params = sum(1 for p in model.parameters())
                opt_params = sum(len(g['params']) for g in optimizer.param_groups)
                print(f"  {'[DIAG]':<20} Trainable params: {trainable_count}/{total_params}, Optimizer params: {opt_params}")
                
                # FIX: If no trainable params, PAI might have disabled gradients
                # This causes "No inf checks" error because backward() produces no grads
                if trainable_count == 0:
                    print(f"  {'[FIX]':<20} Re-enabling requires_grad on all params")
                    for param in model.parameters():
                        param.requires_grad = True
                    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                    print(f"  {'[FIX]':<20} Now trainable: {trainable_count}")
                
                # CRITICAL: Reset GradScaler when optimizer changes
                # The scaler holds state for the old optimizer
                if scaler is not None:
                    scaler = torch.cuda.amp.GradScaler()
                    print(f"  {'[AMP]':<20} {'GradScaler reset for new optimizer':<46}")
                
                # EMA disabled - no re-initialization needed
                
                # Re-read dendrite count AFTER restructuring (it updates during add_validation_score)
                try:
                    if hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'member_vars'):
                        pai_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
                except:
                    pass
            
            # Display PAI status (after any restructuring)
            print(f"  [PAI] Mode: {pai_mode} | Dendrites: {pai_dendrites}")
        
        # Initialize current_lr before scheduler block (for history tracking)
        current_lr = lr  # Default to base LR
        
        # Update scheduler (CosineAnnealing doesn't need val_score)
        # CRITICAL: PAI steps the scheduler internally in add_validation_score!
        # From PAI README: "the scheduler will get stepped inside our code so get rid of your scheduler.step()"
        # We must skip scheduler.step() when use_pai=True to avoid DOUBLE STEPPING
        if scheduler is not None and not use_pai:
            try:
                if hasattr(scheduler, 'step'):
                    # CosineAnnealing: no args, ReduceLROnPlateau: needs score
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_score)
                    else:
                        scheduler.step()
            except Exception as e:
                print(f"  ⚠️ Scheduler error: {e}")
        
        # Safe LR getting (works for both PAI and baseline)
        if scheduler is not None:
            try:
                current_lr = optimizer.param_groups[0]['lr']
            except:
                current_lr = 0.0
            print(f"  {'Learning Rate':<20} {current_lr:>12.2e} {'':>15}")
        
        print(f"  {'─' * 66}\n")
        
        # Track history with all metrics for CSV and plots
        n_dendrites = 0
        if use_pai and hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'member_vars'):
            n_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_map50': val_metrics['map50'],
            'val_map': val_metrics['map'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'learning_rate': current_lr,
            'restructured': restructured,
            'n_dendrites': n_dendrites
        })
        
        # Log to WandB
        try:
            log_data = {
                "train/loss": train_loss,
                "val/mAP50": val_metrics['map50'],
                "val/mAP50-95": val_metrics['map'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "lr": current_lr,
                "epoch": epoch
            }
            if use_pai:
                log_data["pai/restructured"] = int(restructured)
                if hasattr(GPA.pai_tracker, 'member_vars'):
                     log_data["pai/n_dendrites"] = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
            
            if wandb is not None:
                wandb.log(log_data)
        except:
            pass
        
        # ========== SAVE BEST MODEL (with error handling) ==========
        if val_score > best_score:
            best_score = val_score
            
            # CRITICAL: Save a COPY of the state_dict for later restoration
            # This ensures evaluation uses the exact model that achieved the best score
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            
            # Save training model weights (EMA needs longer training to be beneficial)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,  # Use the copy
                'optimizer_state_dict': optimizer.state_dict(),
                'score': val_score,
                'train_loss': train_loss,
                'metrics': val_metrics
            }
            
            # Try primary save path, fallback to backup if needed
            save_path = save_dir / 'best_model.pt'
            try:
                torch.save(checkpoint, save_path)
                print(f"  ✅ NEW BEST mAP@0.5: {val_score:.4f} (saved)\n")
            except OSError as e:
                # Fallback: save in current directory
                fallback_path = Path(f"best_model_backup_{epoch}.pt")
                try:
                    torch.save(checkpoint, fallback_path)
                    print(f"  ⚠️ Saved to fallback: {fallback_path}")
                except Exception as e2:
                    print(f"  ❌ Could not save model: {e2}")
        
        # PAI early stop signal
        if use_pai and training_complete:
            print(f"\n{'='*70}")
            print(f"  [PAI] Training complete signal received at epoch {epoch}")
            print(f"  Best mAP@0.5: {best_score:.4f}")
            print(f"{'='*70}\n")
            break
            
        # Manual early stop for baseline
        if early_stopper(val_score, epoch):
            summary = early_stopper.get_summary()
            print(f"\n{'='*70}")
            print(f"  [EARLY STOP] No improvement for {early_stopper.patience} epochs")
            print(f"  Best mAP@0.5: {summary['best_score']:.4f} at epoch {summary['best_epoch']}")
            print(f"\n  Recent performance (last {len(summary['recent_scores'])} epochs):")
            print(f"  {'─' * 66}")
            for ep, sc in summary['recent_scores']:
                marker = " ⭐" if ep == summary['best_epoch'] else ""
                print(f"    Epoch {ep:3d}: mAP@0.5 = {sc:.4f}{marker}")
            print(f"  {'─' * 66}")
            print(f"{'='*70}\n")
            break
        
    # ========== SAVE ALL OUTPUTS ==========
    print(f"\n[SAVING] Generating all outputs...")
    
    # 1. Save training history as YAML
    history_path = save_dir / 'training_history.yaml'
    with open(history_path, 'w') as f:
        yaml.dump(training_history, f)
    print(f"[SAVE] Training history YAML: {history_path}")
    
    
    # 2. Save training metrics as CSV
    csv_path = save_dir / 'training_metrics.csv'
    if training_history:
        fieldnames = ['epoch', 'train_loss', 'val_map50', 'val_map', 'val_precision', 'val_recall', 
                      'learning_rate', 'restructured', 'n_dendrites']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in training_history:
                writer.writerow(row)
        print(f"[SAVE] Training metrics CSV: {csv_path}")
    
    # 3. Generate training curves plot
    if training_history and len(training_history) > 1:
        epochs_list = [h['epoch'] for h in training_history]
        train_losses = [h.get('train_loss', 0) for h in training_history]
        val_scores = [h.get('val_map50', 0) for h in training_history]  # Fixed: use val_map50
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs_list, train_losses, 'b-', linewidth=2, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs_list, val_scores, 'g-', linewidth=2, label='Val mAP50')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP50')
        ax2.set_title('Validation mAP50')
        ax2.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_idx = val_scores.index(max(val_scores))
        ax2.axvline(x=epochs_list[best_idx], color='r', linestyle='--', alpha=0.7)
        ax2.scatter([epochs_list[best_idx]], [val_scores[best_idx]], color='r', s=100, zorder=5)
        ax2.legend()
        
        exp_type = 'PAI' if use_pai else 'Baseline'
        plt.suptitle(f'{exp_type} Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVE] Training curves plot: {plot_path}")
    
    # 4. PAI-SPECIFIC: Call PAI's built-in save_graphs() for comprehensive outputs
    if use_pai and hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'save_graphs'):
        try:
            print("[SAVE] Generating PAI graphs and CSVs...")
            GPA.pai_tracker.save_graphs("_final")
            print(f"[SAVE] PAI graphs saved to: {save_dir}")
        except Exception as e:
            print(f"[SAVE] Warning: PAI save_graphs failed: {e}")
    # NOTE: Test evaluation is handled in run_experiments.py using validate() function
    # yolo.val() fails on PAI-wrapped models due to internal fuse() call
    
    # 5. Save final summary JSON (test_mAP50 will be updated by run_experiments.py)
    summary = {
        'experiment_type': 'PAI' if use_pai else 'Baseline',
        'best_val_mAP50': round(best_score, 4),
        'test_mAP50': 0.0,  # Will be updated by run_experiments.py
        'total_epochs': len(training_history),
        'pai_enabled': use_pai,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'imgsz': imgsz,
            'n_epochs_to_switch': n_epochs_to_switch,
            'p_epochs_to_switch': p_epochs_to_switch,
            'max_dendrites': max_dendrites
        },
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = save_dir / 'final_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Final summary: {summary_path}")
    
    print(f"\n{'=' * 60}")
    print(f"  TRAINING FINISHED")
    print(f"  Best Val mAP50: {best_score:.4f}")
    print(f"  Results saved to: {save_dir}")
    print(f"{'=' * 60}\n")
    
    # Restore best model weights before returning (for test evaluation in run_experiments.py)
    if best_model_state is not None:
        # CRITICAL: Handle PAI tracker_string size mismatches
        # tracker_string is a PAI internal buffer that changes size when dendrites are added
        # PyTorch strict=False doesn't handle SIZE mismatches, only missing/unexpected keys
        # So we must filter out size-mismatched keys manually
        current_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        
        for key, value in best_model_state.items():
            if key in current_state:
                if current_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    skipped_keys.append(key)
            else:
                # Key doesn't exist in current model (dendrite was added after save)
                skipped_keys.append(key)
        
        # Load the compatible weights
        model.load_state_dict(filtered_state, strict=False)
        
        if skipped_keys:
            print(f"  [Loaded] Best model (skipped {len(skipped_keys)} PAI buffers with size mismatch)")
    
    return model, best_score


# =============================================================================
# SECTION 5.5: SIMPLIFIED PAI TRAINING (Alternative - More Reliable)
# =============================================================================

def train_pai_yolo_simple(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = 'cuda',
    lr: float = 0.01,
    save_name: str = 'PAI_YOLO_Simple',
    pretrained: str = 'yolo11n.pt',
    seed: int = 42,
    n_epochs_to_switch: int = 10,
    p_epochs_to_switch: int = 10,
    max_dendrites: int = 10
) -> Tuple[YOLO, float]:
    """
    SIMPLIFIED PAI-YOLO training using YOLO's train() for 1 epoch at a time.
    
    This approach is MORE RELIABLE than the custom loop because:
    1. Uses Ultralytics' internal data loading (no API compatibility issues)
    2. Uses proper YOLO augmentations and loss computation
    3. Simply wraps the training with PAI's add_validation_score() calls
    
    The key insight is: train for 1 epoch, validate, report to PAI, repeat!
    
    Args:
        data_yaml: Path to dataset YAML
        epochs: Maximum training epochs
        batch_size: Batch size
        imgsz: Image size
        device: 'cuda' or 'cpu'
        lr: Learning rate
        save_name: Name for saving outputs
        pretrained: Path to pretrained weights
        seed: Random seed
        n_epochs_to_switch: PAI N-phase epochs
        p_epochs_to_switch: PAI P-phase epochs
        max_dendrites: Maximum dendrites per neuron
        
    Returns:
        Tuple of (YOLO wrapper, best score)
    """
    if not PAI_AVAILABLE:
        raise RuntimeError("PerforatedAI not installed. Use train_baseline_yolo() instead.")
    
    print("\n" + "=" * 60)
    print("  PAI-YOLO SIMPLE TRAINING (Epoch-by-Epoch)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup PAI configuration
    setup_pai_config(
        n_epochs_to_switch=n_epochs_to_switch,
        p_epochs_to_switch=p_epochs_to_switch,
        max_dendrites=max_dendrites
    )
    
    # Load YOLO model
    print(f"\n[Model] Loading pretrained: {pretrained}")
    yolo = YOLO(pretrained)
    
    # Extract and initialize PAI on the model
    model = extract_yolo_model(yolo)
    model = initialize_pai_model(
        model,
        save_name=save_name,
        maximizing_score=True
    )
    yolo.model = model  # Put PAI model back into wrapper
    
    # Setup optimizer through PAI
    scheduler_patience = max(1, n_epochs_to_switch - 3)
    optimizer, scheduler = setup_pai_optimizer(
        model,
        lr=lr,
        weight_decay=0,
        scheduler_patience=scheduler_patience
    )
    
    # Create save directory
    save_dir = Path(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop - train 1 epoch at a time using YOLO's trainer
    best_score = float('-inf')
    training_history = []
    
    print(f"\n{'=' * 60}")
    print(f"  STARTING SIMPLE TRAINING: {epochs} epochs")
    print(f"{'=' * 60}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'─' * 40}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'─' * 40}")
        
        # Train for 1 epoch using Ultralytics trainer
        # resume=True continues from last checkpoint, epochs=epoch trains TO this epoch
        try:
            results = yolo.train(
                data=data_yaml,
                epochs=1,  # Train 1 epoch at a time
                batch=batch_size,
                imgsz=imgsz,
                device=device,
                lr0=lr,
                seed=seed,
                project=str(save_dir),
                name=f'epoch_{epoch}',
                exist_ok=True,
                verbose=False,
                plots=False,
                save=False  # We'll save manually
            )
            train_loss = float(results.results_dict.get('train/box_loss', 0))
        except Exception as e:
            print(f"  [Warning] Train error: {e}")
            train_loss = 0.0
        
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate using YOLO's built-in validator
        val_results = yolo.val(data=data_yaml, imgsz=imgsz, verbose=False)
        val_score = val_results.box.map50
        print(f"  Val mAP50: {val_score:.4f}")
        
        # ========== CRITICAL: Report to PAI Every Epoch ==========
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_score, model
        )
        
        # Update yolo wrapper with potentially restructured model
        yolo.model = model
        
        # If restructured, reset optimizer
        if restructured:
            print(f"\n  [PAI] ⚡ Dendrites added! Resetting optimizer...")
            optimizer, scheduler = setup_pai_optimizer(
                model,
                lr=lr,
                weight_decay=0,
                scheduler_patience=scheduler_patience
            )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_score)
        
        # Track history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_score': val_score,
            'restructured': restructured
        })
        
        # Save best model
        if val_score > best_score:
            best_score = val_score
            yolo.save(save_dir / 'best.pt')
            print(f"  ✓ New Best Score: {val_score:.4f}")
        
        # Check if PAI says training is complete
        if training_complete:
            print(f"\n{'=' * 60}")
            print(f"  [PAI] Training complete at epoch {epoch}!")
            print(f"{'=' * 60}")
            break
    
    # Save history
    with open(save_dir / 'training_history.yaml', 'w') as f:
        yaml.dump(training_history, f)
    
    print(f"\n{'=' * 60}")
    print(f"  SIMPLE TRAINING FINISHED")
    print(f"  Best Val mAP50: {best_score:.4f}")
    print(f"  Results saved to: {save_dir}")
    print(f"{'=' * 60}\n")
    
    return yolo, best_score


# =============================================================================
# SECTION 6: BASELINE TRAINING (For Comparison)
# =============================================================================

def train_baseline_yolo(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = 'cuda',
    lr: float = 0.001,
    save_name: str = 'baseline_YOLO',
    pretrained: str = 'yolo11n.pt',
    seed: int = 42
) -> Tuple[nn.Module, float]:
    """
    Standard YOLO training WITHOUT PAI for baseline comparison.
    
    This uses the same custom loop structure as PAI training
    for fair comparison (same augmentations, same epochs, etc.)
    
    Args:
        data_yaml: Path to dataset YAML
        epochs: Maximum training epochs
        batch_size: Batch size
        imgsz: Image size
        device: 'cuda' or 'cpu'
        lr: Learning rate
        save_name: Name for saving outputs
        pretrained: Path to pretrained weights
        seed: Random seed
        
    Returns:
        Tuple of (trained model, best score)
    """
    print("\n" + "=" * 60)
    print("  BASELINE YOLO TRAINING (No PAI)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Setup device - GPU or CPU
    if isinstance(device, str):
        if device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print(f"Device: CPU")
    else:
        print(f"Device: {device}")
    
    # Load YOLO Model
    print(f"\n[Model] Loading pretrained: {pretrained}")
    yolo = YOLO(pretrained)
    model = extract_yolo_model(yolo)
    model = model.to(device)
    
    # Setup Data Loaders
    print(f"\n[Data] Loading dataset from: {data_yaml}")
    data_dict = check_det_dataset(data_yaml)
    
    train_dataset = build_yolo_dataset(
        cfg={'imgsz': imgsz},
        img_path=data_dict.get('train', ''),
        batch=batch_size,
        data=data_dict,
        mode='train'
    )
    
    train_loader = build_dataloader(
        train_dataset,
        batch=batch_size,
        workers=4,
        shuffle=True
    )
    
    val_dataset = build_yolo_dataset(
        cfg={'imgsz': imgsz},
        img_path=data_dict.get('val', ''),
        batch=batch_size,
        data=data_dict,
        mode='val'
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch=batch_size,
        workers=4,
        shuffle=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Setup Optimizer (standard, no PAI)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=7, factor=0.5
    )
    
    # Mixed precision scaler - only for CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Create save directory
    save_dir = Path(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training Loop
    best_score = float('-inf')
    training_history = []
    
    print(f"\n{'=' * 60}")
    print(f"  STARTING BASELINE TRAINING: {epochs} epochs")
    print(f"{'=' * 60}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'─' * 40}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'─' * 40}")
        
        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch
        )
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        yolo.model = model
        val_score = validate(yolo, data_yaml, imgsz)
        print(f"  Val mAP50: {val_score:.4f}")
        
        # Update scheduler
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current LR: {current_lr:.2e}")
        
        # Track history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_score': val_score
        })
        
        # Save best model
        if val_score > best_score:
            best_score = val_score
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': val_score
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"  ✓ New Best Score: {val_score:.4f}")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': val_score,
            'history': training_history
        }, save_dir / 'last_model.pt')
    
    # Save history
    with open(save_dir / 'training_history.yaml', 'w') as f:
        yaml.dump(training_history, f)
    
    print(f"\n{'=' * 60}")
    print(f"  BASELINE TRAINING FINISHED")
    print(f"  Best Score: {best_score:.4f}")
    print(f"  Results saved to: {save_dir}")
    print(f"{'=' * 60}\n")
    
    return model, best_score


# =============================================================================
# SECTION 7: DATA EFFICIENCY EXPERIMENT
# =============================================================================

def run_data_efficiency_experiment(
    source_dir: str,
    output_dir: str = 'runs/efficiency_experiments',
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = 'cuda',
    seed: int = 42,
    nc: int = 9,
    class_names: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run the complete suite of 10 experiments:
    Baseline and PAI for 5 splits (100%, 50%, 20%, 10%, 5%).
    
    CRITICAL: Val and Test sets are FIXED across all experiments.
    Only the training set size varies.
    
    Args:
        source_dir: Raw dataset directory with all images/labels
        output_dir: Where to save all experiment outputs
        epochs: Training epochs per run
        batch_size: Batch size
        imgsz: Image size
        device: Device (cuda/cpu)
        seed: Random seed
        nc: Number of classes
        class_names: List of class names (optional)
    """
    print("\n" + "=" * 70)
    print("  FULL DATA EFFICIENCY SUITE (10 RUNS)")
    print("  Val/Test: FIXED | Train: 100%, 50%, 20%, 10%, 5%")
    print("=" * 70 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default class names for SODA-A
    if class_names is None:
        class_names = ['pedestrian', 'cyclist', 'car', 'truck', 'tram', 
                       'tricycle', 'bus', 'moped', 'stroller'][:nc]
    
    # ========== STEP 1: Create FIXED val/test split ==========
    print("[Step 1/3] Creating FIXED val/test split...")
    train_pool_dir, val_dir, test_dir = create_fixed_split(
        source_dir=source_dir,
        output_dir=str(output_path / 'data'),
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    
    # ========== STEP 2: Create train subsets from the pool ==========
    print("\n[Step 2/3] Creating train subsets...")
    split_paths = create_data_efficiency_splits(
        train_pool_dir=train_pool_dir,
        output_dir=str(output_path / 'data' / 'train_splits'),
        ratios=[1.0, 0.5, 0.2, 0.1, 0.05],
        seed=seed
    )
    
    # ========== STEP 3: Create YAML files for each experiment ==========
    print("\n[Step 3/3] Creating dataset YAML configs...")
    split_yamls = {}
    for pct, train_path in split_paths.items():
        yaml_name = output_path / f"data_{pct}.yaml"
        split_yamls[pct] = create_dataset_yaml(
            output_path=str(yaml_name),
            train_path=str(Path(train_path) / 'images'),
            val_path=str(Path(val_dir) / 'images'),  # FIXED val
            test_path=str(Path(test_dir) / 'images'),  # FIXED test
            nc=nc,
            names=class_names
        )

    results = {}
    ratios = ['100pct', '50pct', '20pct', '10pct', '5pct']
    
    # 3. Run all experiments
    for pct in ratios:
        yaml_file = split_yamls[pct]
        print(f"\n\n{'#' * 80}")
        print(f"RUNNING EXPERIMENTS FOR {pct} DATA")
        print(f"{'#' * 80}")
        
        # --- Baseline ---
        print(f"\n[RUN] Baseline - {pct} data")
        baseline_save = output_path / f'baseline_{pct}'
        _, b_score = train_baseline_yolo(
            data_yaml=yaml_file,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            device=device,
            save_name=str(baseline_save),
            seed=seed
        )
        
        # --- PAI ---
        if PAI_AVAILABLE:
            print(f"\n[RUN] PAI (Dendrites) - {pct} data")
            pai_save = output_path / f'pai_{pct}'
            _, p_score = train_pai_yolo(
                data_yaml=yaml_file,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
                device=device,
                save_name=str(pai_save),
                seed=seed
            )
        else:
            p_score = 0.0
            
        results[pct] = {
            'baseline_mAP50': b_score,
            'pai_mAP50': p_score,
            'improvement': p_score - b_score
        }
    
    # Final Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Split':<10} | {'Baseline':<10} | {'PAI':<10} | {'Gain':<10}")
    print("-" * 45)
    for pct in ratios:
        r = results[pct]
        print(f"{pct:<10} | {r['baseline_mAP50']:<10.4f} | {r['pai_mAP50']:<10.4f} | {r['improvement']:<10.4f}")
    print("=" * 70)
    
    with open(output_path / 'final_results.yaml', 'w') as f:
        yaml.dump(results, f)
        
    return results


# =============================================================================
# SECTION 8: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PAI-YOLOv11n Custom Training for Data Efficiency',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        '--data', type=str, required=False,
        help='Path to dataset YAML file (not needed if --use-voc is set)'
    )
    parser.add_argument(
        '--use-voc', action='store_true',
        help='Use VOC2007 dataset (automatically downloaded by Ultralytics)'
    )
    
    # Experiment type
    parser.add_argument(
        '--experiment', type=str, default='pai',
        choices=['pai', 'baseline', 'both', 'all'],
        help='Experiment type:\n'
             '  pai      - Train with PAI dendritic optimization\n'
             '  baseline - Train standard YOLO (no PAI)\n'
             '  both     - Run both on current --data\n'
             '  all      - AUTOMATED: Run 10 runs (Baseline + PAI for all splits)'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda (GPU) or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # PAI parameters
    parser.add_argument('--n-epochs-switch', type=int, default=10,
                        help='PAI: epochs in N-phase before checking plateau')
    parser.add_argument('--p-epochs-switch', type=int, default=10,
                        help='PAI: epochs in P-phase (dendrite training)')
    parser.add_argument('--max-dendrites', type=int, default=10,
                        help='PAI: maximum dendrites per neuron')
    
    # Model
    parser.add_argument('--pretrained', type=str, default='yolo11n.pt',
                        help='Path to pretrained weights')
    parser.add_argument('--save-name', type=str, default='runs/pai_yolo',
                        help='Directory to save results')
    
    # Utility commands
    parser.add_argument('--create-splits', action='store_true',
                        help='Create data efficiency splits from source data')
    parser.add_argument('--source-dir', type=str,
                        help='Source directory for creating splits (not needed for VOC)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not args.use_voc and not args.data:
        print("Error: Either --use-voc or --data must be specified")
        sys.exit(1)
    
    # Set dataset path
    if args.use_voc:
        data_yaml = 'VOC.yaml'  # Ultralytics built-in, auto-downloads VOC2007
        print("\n[Dataset] Using VOC2007 (will be auto-downloaded if needed)")
        print("  This may take a few minutes on first run...")
    else:
        data_yaml = args.data
    
    # Handle utility commands
    if args.create_splits:
        if args.use_voc:
            print("Note: --create-splits not needed for VOC (uses built-in splits)")
            return
        if not args.source_dir:
            print("Error: --source-dir required when using --create-splits")
            sys.exit(1)
        
        # Step 1: Create fixed val/test split
        train_pool_dir, val_dir, test_dir = create_fixed_split(
            source_dir=args.source_dir,
            output_dir='data_splits',
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed
        )
        
        # Step 2: Create train subsets from the pool
        create_data_efficiency_splits(
            train_pool_dir=train_pool_dir,
            output_dir='data_splits/train_subsets',
            ratios=[1.0, 0.5, 0.2, 0.1, 0.05],
            seed=args.seed
        )
        return

    
    if args.experiment == 'all':
        if args.use_voc:
            # For VOC, we can't create custom splits easily since it's auto-downloaded
            # Instead, we'll run experiments on the full dataset with different training strategies
            print("\n[VOC Mode] Running simplified experiments:")
            print("  - Baseline on full VOC train set")
            print("  - PAI on full VOC train set")
            print("  Note: Custom data splits not supported for auto-downloaded VOC")
            print("  For full data efficiency experiments, use a custom dataset with --source-dir\n")
            
            # Run baseline
            print("\n" + "="*60)
            print("EXPERIMENT 1: Baseline YOLO on VOC2007")
            print("="*60)
            _, baseline_score = train_baseline_yolo(
                data_yaml='VOC.yaml',
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=args.device,
                lr=args.lr,
                save_name=str(Path(args.save_name) / 'baseline_voc'),
                seed=args.seed
            )
            
            # Run PAI
            if PAI_AVAILABLE:
                print("\n" + "="*60)
                print("EXPERIMENT 2: PAI-YOLO on VOC2007")
                print("="*60)
                _, pai_score = train_pai_yolo(
                    data_yaml='VOC.yaml',
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    imgsz=args.imgsz,
                    device=args.device,
                    lr=args.lr,
                    save_name=str(Path(args.save_name) / 'pai_voc'),
                    seed=args.seed,
                    n_epochs_to_switch=args.n_epochs_switch,
                    p_epochs_to_switch=args.p_epochs_switch,
                    max_dendrites=args.max_dendrites
                )
                
                print("\n" + "="*60)
                print("VOC2007 RESULTS SUMMARY")
                print("="*60)
                print(f"Baseline mAP50: {baseline_score:.4f}")
                print(f"PAI mAP50:      {pai_score:.4f}")
                print(f"Improvement:    {pai_score - baseline_score:.4f}")
                print("="*60)
            else:
                print("\nWarning: PAI not available, skipping PAI experiment")
            
            return
        else:
            # Custom dataset - full data efficiency experiment
            if not args.source_dir:
                print("Error: --source-dir required for --experiment all with custom dataset")
                sys.exit(1)
            run_data_efficiency_experiment(
                source_dir=args.source_dir,
                output_dir=args.save_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=args.device,
                seed=args.seed
            )
            return


    # Run individual experiments
    if args.experiment in ['pai', 'both']:
        if not PAI_AVAILABLE:
            print("Error: PerforatedAI not installed. Cannot run PAI experiment.")
            if args.experiment == 'both':
                print("Falling back to baseline only...")
            else:
                sys.exit(1)
        else:
            train_pai_yolo(
                data_yaml=data_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=args.device,
                lr=args.lr,
                save_name=args.save_name + '_pai' if args.experiment == 'both' else args.save_name,
                pretrained=args.pretrained,
                seed=args.seed,
                n_epochs_to_switch=args.n_epochs_switch,
                p_epochs_to_switch=args.p_epochs_switch,
                max_dendrites=args.max_dendrites
            )
    
    if args.experiment in ['baseline', 'both']:
        train_baseline_yolo(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            lr=args.lr,
            save_name=args.save_name + '_baseline' if args.experiment == 'both' else args.save_name,
            pretrained=args.pretrained,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
