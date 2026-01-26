import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import wandb
import time
from tqdm import tqdm

import config
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
    print("WARNING: PerforatedAI not found. PAI features will be disabled.")
    PAI_AVAILABLE = False

def train_one_epoch(model, loader, optimizer, criterion, device, use_pai=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pb_available = False
    if use_pai and PAI_AVAILABLE:
        pb_available = GPA.pc.get_perforated_backpropagation()

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        try:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

        # PAI: Apply PB gradients if available
            if use_pai and PAI_AVAILABLE and pb_available:
                try:
                    GPA.pai_tracker.apply_pb_grads()
                except Exception:
                    pass

            optimizer.step()

            # PAI: Zero PB gradients
            if use_pai and PAI_AVAILABLE and pb_available:
                try:
                    GPA.pai_tracker.apply_pb_zero()
                except Exception:
                    pass

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        except Exception as e:
            print(f"\n[ERROR] Exception during training batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
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

def run_experiment(model, model_name, trainloader, testloader, device, use_pai=False):
    print(f"\nStarting experiment: {model_name}")
    
    # Initialize WandB
    wandb.init(project="cifar100-dendritic-demo", name=model_name, reinit="finish_previous")
    
    # PAI Initialization
    if use_pai and PAI_AVAILABLE:
        print("Initializing PerforatedAI...")
        
        # --- TARGETED DENDRITE LOGIC ---
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
            
            # Create new layer
            new_layer = new_class(
                old_layer.in_channels, old_layer.out_channels,
                old_layer.kernel_size, old_layer.stride, old_layer.padding,
                old_layer.dilation, old_layer.groups,
                old_layer.bias is not None, old_layer.padding_mode
            )
            
            # Copy weights and bias
            new_layer.weight.data = old_layer.weight.data.clone()
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data.clone()
                
            # Replace
            setattr(parent, target_attr, new_layer)
            print(f"  -> Targeted: {target_name}")

        print("Applying strict dendrite targeting...")
        # Target specific layers as requested:
        # layer3.block1.conv2 -> layer3.0.conv2
        # layer3.block2.conv2 -> layer3.1.conv2
        # layer4.block1.conv2 -> layer4.0.conv2
        targets = [
            "layer3.0.conv2",
            "layer3.1.conv2"
        ]
        
        for target in targets:
            try:
                replace_layer(model, target, TargetedConv2d)
            except AttributeError:
                print(f"  WARNING: Could not find {target}, skipping.")

        # Configure PAI *BEFORE* initialization (Critical for conversion flags)
        print("Configuring PAI settings...")
        GPA.pc.set_modules_to_convert([TargetedConv2d])
        
        # Fix: Track other layers so optimizer doesn't crash
        GPA.pc.append_modules_to_track([nn.Conv2d, nn.Linear, nn.BatchNorm2d])
        
        GPA.pc.set_unwrapped_modules_confirmed(True) # Prevent PAI from asking about skipped layers
        GPA.pc.set_weight_decay_accepted(True) # Suppress weight decay warning
        
        if config.PAI_SWITCH_MODE == "DOING_HISTORY":
            GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
        elif config.PAI_SWITCH_MODE == "DOING_TRIALS":
            GPA.pc.set_switch_mode(GPA.pc.DOING_TRIALS)
        elif config.PAI_SWITCH_MODE == "DOING_SWITCH_EVERY_TIME":
            GPA.pc.set_switch_mode(GPA.pc.DOING_SWITCH_EVERY_TIME)
            
        GPA.pc.set_improvement_threshold(config.PAI_IMPROVEMENT_THRESHOLD)
        GPA.pc.set_max_dendrites(config.PAI_MAX_DENDRITES)
        GPA.pc.set_n_epochs_to_switch(config.PAI_N_EPOCHS)
        GPA.pc.set_p_epochs_to_switch(config.PAI_P_EPOCHS)
        GPA.pc.set_testing_dendrite_capacity(False)  # Disable testing mode to prevent debugger
        
        # Initialize PAI on the modified model
        model = UPA.initialize_pai(model, maximizing_score=True)
        GPA.pc.set_save_name(f"PAI_{model_name}")
        
        # Check PB status
        if GPA.pc.get_perforated_backpropagation():
            print("Perforated Backpropagation ENABLED (Paid License Active)")
        else:
            print("Using Gradient Descent Dendrites (Open Source Mode)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, 
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    if use_pai and PAI_AVAILABLE:
        GPA.pai_tracker.set_optimizer_instance(optimizer)

    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device, use_pai)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        
        scheduler.step()
        
        # PAI Logic
        restructured = False
        training_complete = False
        num_dendrites = 0
        
        # Parameter Safety Guard
        current_params = utils.count_parameters(model)
        param_limit = config.PARAM_SAFETY_RATIO * config.BASELINE_RESNET34_PARAMS
        if use_pai and current_params >= param_limit:
            print(f"⚠️ PARAMETER GUARD: Stopping growth at {current_params/1e6:.2f}M (Limit: {param_limit/1e6:.2f}M)")
            GPA.pc.set_max_dendrites(0)

        if use_pai and PAI_AVAILABLE:
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
            
            # Get dendrite count
            # This is a bit hacky, usually we track it via restructuring events or internal vars
            # But for now we can just check if restructured is true to increment a counter in main loop if we wanted
            # Or inspect model structure. 
            # Let's rely on PAI internal logging or just log 'restructured' event.
            
            if restructured:
                print(f"Epoch {epoch+1}: Model Restructured! Adding Dendrite.")
                # Reset optimizer
                optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, 
                                      momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
                # Note: We might want to adjust LR for P phase, but keeping simple for now or letting scheduler handle it (though scheduler reset might be needed)
                # For simplicity, we restart optimizer but keep scheduler state? 
                # Actually, standard PAI practice is to just reset optimizer.
                GPA.pai_tracker.set_optimizer_instance(optimizer)
                # Re-attach scheduler?
                # Don't pass last_epoch when creating a fresh scheduler for a fresh optimizer
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

        # Logging
        current_params = utils.count_parameters(model)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "params": current_params,
            "lr": optimizer.param_groups[0]['lr'],
            "pai_restructured": int(restructured) if use_pai else 0
        })
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Params: {current_params/1e6:.2f}M")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, f"{model_name}_best.pth"))
        else:
            patience_counter += 1

        if use_pai and training_complete:
            print("PAI signaled training completion limit, but continuing until NUM_EPOCHS for convergence.")
            # We don't break anymore to ensure full training
            pass
            
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    wandb.finish()
    return best_acc, utils.count_parameters(model)
