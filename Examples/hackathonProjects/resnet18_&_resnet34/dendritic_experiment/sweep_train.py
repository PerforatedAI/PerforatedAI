"""
Sweep Training Script for ResNet18 + PAI (Dendrites)
Implements efficiency scoring and parameter safety guards for WandB sweeps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import wandb
import time
from tqdm import tqdm

import config
import models
import utils

# Setup PAI path and environment variables BEFORE importing PAI
sys.path.append(config.PAI_REPO_PATH)
os.environ['PAIEMAIL'] = config.PAI_EMAIL
os.environ['PAITOKEN'] = config.PAI_TOKEN

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
except ImportError:
    print("ERROR: PerforatedAI not found. Cannot run sweep without PAI.")
    PAI_AVAILABLE = False
    sys.exit(1)


def calculate_efficiency_score(val_acc, current_params):
    """
    Calculate efficiency score (lower is better).
    efficiency_score = (baseline34_acc - val_acc) + 0.25 * (current_params / baseline34_params)
    """
    acc_gap = config.BASELINE_RESNET34_ACC - val_acc
    param_ratio = current_params / config.BASELINE_RESNET34_PARAMS
    return acc_gap + 0.25 * param_ratio


def get_target_layer_names(targeted_layers_str):
    """
    Parse targeted layers string into list of layer target names.
    E.g., "layer3,layer4" -> ["layer3.0.conv2", "layer3.1.conv2", "layer4.0.conv2", "layer4.1.conv2"]
    """
    layers = targeted_layers_str.split(",")
    targets = []
    
    for layer in layers:
        layer = layer.strip()
        if layer == "layer3":
            targets.extend(["layer3.0.conv2", "layer3.1.conv2"])
        elif layer == "layer4":
            targets.extend(["layer4.0.conv2", "layer4.1.conv2"])
    
    return targets


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pb_available = GPA.pc.get_perforated_backpropagation()

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # PAI: Apply PB gradients if available
        if pb_available:
            try:
                GPA.pai_tracker.apply_pb_grads()
            except Exception:
                pass

        optimizer.step()

        # PAI: Zero PB gradients
        if pb_available:
            try:
                GPA.pai_tracker.apply_pb_zero()
            except Exception:
                pass

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def run_sweep_experiment():
    """Main sweep training function."""
    # Initialize wandb run (sweep agent handles config)
    run = wandb.init()
    sweep_config = wandb.config
    
    print(f"\n{'='*60}")
    print(f"Starting Sweep Run: {run.name}")
    print(f"Config: {dict(sweep_config)}")
    print(f"{'='*60}\n")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    print("Loading Data...")
    trainloader, testloader = utils.get_dataloaders(
        batch_size=sweep_config.get("batch_size", 128)
    )
    
    # Model initialization
    print("Initializing ResNet18...")
    model = models.get_resnet18(pretrained=True).to(device)
    
    # --- PAI Initialization with Sweep Parameters ---
    print("Initializing PerforatedAI with sweep parameters...")
    
    # Targeted layer setup
    class TargetedConv2d(nn.Conv2d):
        """Wrapper class to target specific layers for PAI."""
        pass

    def replace_layer(module, target_name, new_class):
        """Replaces a specific layer with a new class, preserving weights."""
        parts = target_name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        target_attr = parts[-1]
        old_layer = getattr(parent, target_attr)
        
        new_layer = new_class(
            old_layer.in_channels, old_layer.out_channels,
            old_layer.kernel_size, old_layer.stride, old_layer.padding,
            old_layer.dilation, old_layer.groups,
            old_layer.bias is not None, old_layer.padding_mode
        )
        
        new_layer.weight.data = old_layer.weight.data.clone()
        if old_layer.bias is not None:
            new_layer.bias.data = old_layer.bias.data.clone()
            
        setattr(parent, target_attr, new_layer)
        print(f"  -> Targeted: {target_name}")

    # Apply targeting based on sweep config
    targeted_layers_str = sweep_config.get("targeted_layers", "layer3,layer4")
    targets = get_target_layer_names(targeted_layers_str)
    
    print(f"Applying dendrite targeting to: {targets}")
    for target in targets:
        try:
            replace_layer(model, target, TargetedConv2d)
        except AttributeError:
            print(f"  WARNING: Could not find {target}, skipping.")

    # Configure PAI with sweep parameters
    print("Configuring PAI settings from sweep...")
    GPA.pc.set_modules_to_convert([TargetedConv2d])
    GPA.pc.append_modules_to_track([nn.Conv2d, nn.Linear, nn.BatchNorm2d])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_switch_mode(GPA.pc.DOING_SWITCH_EVERY_TIME)
    
    # Apply sweep parameters
    GPA.pc.set_improvement_threshold(sweep_config.get("improvement_threshold", 0.002))
    GPA.pc.set_max_dendrites(sweep_config.get("max_dendrites", 2))
    GPA.pc.set_n_epochs_to_switch(sweep_config.get("patience_dendrite", 6))
    GPA.pc.set_p_epochs_to_switch(sweep_config.get("patience_dendrite", 6))
    GPA.pc.set_testing_dendrite_capacity(False)  # CRITICAL: Disable test mode
    
    # Initialize PAI
    model = UPA.initialize_pai(model, maximizing_score=True)
    GPA.pc.set_save_name(f"PAI_sweep_{run.name}")
    
    if GPA.pc.get_perforated_backpropagation():
        print("Perforated Backpropagation ENABLED")
    else:
        print("Using Gradient Descent Dendrites")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=sweep_config.get("learning_rate", 0.001),
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )
    num_epochs = sweep_config.get("num_epochs", 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    GPA.pai_tracker.set_optimizer_instance(optimizer)

    # Tracking variables
    best_acc = 0.0
    best_efficiency = float('inf')
    dendrites_added = 0
    param_guard_triggered = False
    param_safety_limit = config.PARAM_SAFETY_RATIO * config.BASELINE_RESNET34_PARAMS
    
    print(f"\nParameter Safety Limit: {param_safety_limit/1e6:.2f}M")
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        scheduler.step()
        
        # Current parameters
        current_params = utils.count_parameters(model)
        
        # --- PARAMETER SAFETY GUARD ---
        if current_params >= param_safety_limit and not param_guard_triggered:
            print(f"\n⚠️  PARAMETER GUARD TRIGGERED at {current_params/1e6:.2f}M params!")
            print(f"    Stopping dendrite growth (limit: {param_safety_limit/1e6:.2f}M)")
            GPA.pc.set_max_dendrites(0)  # Stop further dendrite growth
            param_guard_triggered = True
        
        # PAI validation score handling
        restructured = False
        training_complete = False
        active_mode = "N"  # Normal phase by default
        
        if not param_guard_triggered:
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
            
            if restructured:
                dendrites_added += 1
                print(f"Epoch {epoch+1}: Dendrite added! Total: {dendrites_added}")
                
                # Reset optimizer after restructuring
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=sweep_config.get("learning_rate", 0.001),
                    momentum=config.MOMENTUM,
                    weight_decay=config.WEIGHT_DECAY
                )
                GPA.pai_tracker.set_optimizer_instance(optimizer)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Determine active PAI mode
        try:
            active_mode = "P" if GPA.pai_tracker.in_p_phase() else "N"
        except:
            active_mode = "N"
        
        # Calculate efficiency score
        efficiency_score = calculate_efficiency_score(val_acc, current_params)
        
        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "current_params": current_params,
            "current_params_M": current_params / 1e6,
            "dendrites_added": dendrites_added,
            "efficiency_score": efficiency_score,
            "active_PAI_mode": active_mode,
            "param_guard_triggered": int(param_guard_triggered),
            "lr": optimizer.param_groups[0]['lr'],
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Params: {current_params/1e6:.2f}M | "
              f"Efficiency: {efficiency_score:.4f} | "
              f"Dendrites: {dendrites_added}")

        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
        if efficiency_score < best_efficiency:
            best_efficiency = efficiency_score

        # Check for training completion from PAI
        if training_complete:
            print("PAI signaled training completion.")
            break

    # Final summary
    final_params = utils.count_parameters(model)
    final_efficiency = calculate_efficiency_score(best_acc, final_params)
    
    wandb.log({
        "final_val_acc": best_acc,
        "final_params": final_params,
        "final_params_M": final_params / 1e6,
        "final_dendrites": dendrites_added,
        "final_efficiency_score": final_efficiency,
    })
    
    print(f"\n{'='*60}")
    print(f"Sweep Run Complete: {run.name}")
    print(f"Best Val Acc: {best_acc:.2f}%")
    print(f"Final Params: {final_params/1e6:.2f}M")
    print(f"Dendrites Added: {dendrites_added}")
    print(f"Efficiency Score: {final_efficiency:.4f}")
    print(f"{'='*60}\n")
    
    # Comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<25} | {'Params (M)':<12} | {'Accuracy':<10}")
    print("-"*60)
    print(f"{'ResNet18 (baseline)':<25} | {config.BASELINE_RESNET18_PARAMS/1e6:<12.2f} | {config.BASELINE_RESNET18_ACC:<10.2f}%")
    print(f"{'ResNet34 (baseline)':<25} | {config.BASELINE_RESNET34_PARAMS/1e6:<12.2f} | {config.BASELINE_RESNET34_ACC:<10.2f}%")
    print(f"{'ResNet18 + PAI (this run)':<25} | {final_params/1e6:<12.2f} | {best_acc:<10.2f}%")
    print("="*60 + "\n")
    
    wandb.finish()


if __name__ == "__main__":
    if not PAI_AVAILABLE:
        print("ERROR: PerforatedAI is required for sweep training.")
        sys.exit(1)
    
    run_sweep_experiment()
