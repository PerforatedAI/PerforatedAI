"""
Project Med-Edge: Perforated AI for Portable Dermatology
Training Script (Baseline + Dendritic)
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb
import os
from tqdm import tqdm

# FIX: Disable pdb breakpoints (PAI library has debug code that breaks on Windows)
import pdb
pdb.set_trace = lambda: None

from src.model import DermoNet_Edge
from src.dataset import get_loaders

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Credentials
os.environ["PAIEMAIL"] = "hacker_token@perforatedai.com"
os.environ["PAITOKEN"] = "MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+r4VqudaOlJuH10MIQWm9Uqg=="


def train_epoch(model, device, train_loader, optimizer, criterion):
    """Single training epoch."""
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    
    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device).squeeze().long()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        total_loss += loss.item()
    
    train_acc = 100.0 * correct / total
    return train_acc, total_loss / len(train_loader)


def validate(model, device, val_loader, criterion):
    """Validation pass."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).squeeze().long()
            out = model(x)
            loss = criterion(out, y)
            
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
    
    val_acc = 100.0 * correct / total
    return val_acc, total_loss / len(val_loader)


def train(cfg, use_dendrites=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | Dendrites: {use_dendrites}")
    
    # Ensure directories exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "PAI"), exist_ok=True)
    
    # Data
    data_dir = os.path.join(base_dir, "data")
    train_loader, val_loader, _ = get_loaders(
        cfg['dataset'], 
        cfg['training']['batch_size'], 
        root=data_dir
    )
    
    # Merge W&B Sweep config if active
    if wandb.run and wandb.config:
        for k, v in wandb.config.items():
            if k in cfg['training']:
                cfg['training'][k] = v
                print(f"🧹 Sweep Override: cfg['training']['{k}'] = {v}")
            # Map top-level sweep params to nested config if needed
            elif k == 'lr':
                cfg['training']['lr'] = v
            elif k == 'weight_decay':
                cfg['training']['weight_decay'] = v
            elif k == 'batch_size':
                cfg['training']['batch_size'] = v
            elif k == 'epochs':
                cfg['training']['epochs'] = v
                
    # Model (with dropout support for sweep)
    # Default dropouts if not in config
    d1 = cfg['training'].get('dropout1', 0.25)
    d2 = cfg['training'].get('dropout2', 0.5)
    
    # If sweep passes 'dropout', assume it sets dropout1
    if 'dropout' in wandb.config:
        d1 = wandb.config.dropout
        print(f"🧹 Sweep Override: dropout1 = {d1}")

    model = DermoNet_Edge(dropout1=d1, dropout2=d2).to(device)
    
    # PAI Setup (PROPER Configuration matching MNIST example)
    if use_dendrites:
        print("⚡ Configuring Perforated AI...")
        
        # CRITICAL: Set improvement thresholds (aggressive)
        GPA.pc.set_improvement_threshold([0.01, 0.001, 0.0001, 0])
        
        # CRITICAL: Weight initialization for new dendrites
        GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
        
        # CRITICAL: Activation function for dendrites
        GPA.pc.set_pai_forward_function(torch.sigmoid)
        
        # Global PAI settings
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_n_epochs_to_switch(cfg['dendrites']['epochs_to_switch'])
        GPA.pc.set_max_dendrites(cfg['dendrites']['max_dendrites'])
        GPA.pc.set_weight_decay_accepted(True)  # Allow weight decay
        GPA.pc.set_verbose(True)
        
        # Initialize PAI on the model
        model = UPA.initialize_pai(model, save_name="DermoNet_PAI")
        model.to(device)
        
        # PROPER SETUP: Set optimizer and scheduler CLASSES (like MNIST example)
        GPA.pai_tracker.set_optimizer(optim.Adam)
        GPA.pai_tracker.set_scheduler(StepLR)
        
        # Define optimizer and scheduler arguments
        optimArgs = {
            "params": model.parameters(),
            "lr": cfg['training']['lr'],
            "weight_decay": cfg['training']['weight_decay'],
        }
        schedArgs = {"step_size": 1, "gamma": 0.98}  # Decay LR by 2% each epoch
        
        # Let PAI create both optimizer and scheduler
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        print(f"✅ PAI initialized with Adam optimizer and StepLR scheduler")
    else:
        optimizer = optim.Adam(
            model.parameters(), 
            lr=cfg['training']['lr'],
            weight_decay=cfg['training']['weight_decay']
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epochs = cfg['training']['epochs']
    
    # Training Loop
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        val_acc, val_loss = validate(model, device, val_loader, criterion)
        
        print(f"Epoch {epoch}/{epochs}: Train={train_acc:.2f}% | Val={val_acc:.2f}%")
        
        # Log to W&B
        wandb.log({
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch
        })
        
        # PAI Logic (Edge Voice Pattern)
        if use_dendrites:
            # Add training score for PAI tracking
            GPA.pai_tracker.add_extra_score(train_acc, "Train")
            
            # Add validation score - this may trigger dendrite growth
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                val_acc, model
            )
            model.to(device)
            
            if restructured:
                print(">>> ⚡ Architecture Restructured (Dendrite Added)! <<<")
                # Recreate optimizer and scheduler after restructuring (PROPER WAY)
                optimArgs = {
                    "params": model.parameters(),
                    "lr": cfg['training']['lr'],
                    "weight_decay": cfg['training']['weight_decay'],
                }
                schedArgs = {"step_size": 1, "gamma": 0.98}
                optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
                
                wandb.log({
                    "dendrite_count": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
                    "param_count": UPA.count_params(model)
                })
            
            if training_complete:
                print("🏆 PAI Training Complete - Optimal Architecture Found!")
                break
        else:
            # Standard scheduler step
            scheduler.step()

        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(base_dir, "checkpoints", f"best_{'pai' if use_dendrites else 'base'}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 New best: {val_acc:.2f}% saved to {save_path}")
    
    print(f"\n🏁 Training Complete! Best Val Accuracy: {best_acc:.2f}%")
    
    # Log final stats
    if use_dendrites:
        final_params = UPA.count_params(model)
        final_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
        print(f"📊 Final Model: {final_params} params, {final_dendrites} dendrites")
        wandb.log({
            "final_val_acc": best_acc,
            "final_param_count": final_params,
            "final_dendrite_count": final_dendrites
        })
        
        # CRITICAL: Save PAI graph (required for submission)
        try:
            print("💾 Saving PAI graph...")
            GPA.pai_tracker.save_graphs()
            print("✅ PAI graph saved to PAI/PAI.png")
        except Exception as e:
            print(f"⚠️ Warning: Could not save PAI graph: {e}")
    else:
        wandb.log({"final_val_acc": best_acc})
    
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Med-Edge Training")
    parser.add_argument("--dendrites", action="store_true", help="Enable Perforated AI dendrites")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ Warning: Ignoring unknown arguments: {unknown} (likely from W&B Sweep)")
    
    # Load config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "default.yaml")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize W&B
    run_name = "dendritic" if args.dendrites else "baseline"
    wandb.init(
        project=cfg['project_name'],
        name=run_name,
        config=cfg
    )
    
    # Train
    train(cfg, use_dendrites=args.dendrites)
    
    wandb.finish()
