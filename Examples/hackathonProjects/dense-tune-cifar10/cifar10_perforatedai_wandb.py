"""
Dense-Tune CIFAR-10 with Weights & Biases Integration

This script enables hyperparameter sweeping and experiment tracking via W&B.
Run a single experiment: python cifar10_perforatedai_wandb.py --count 1
Run a sweep: python cifar10_perforatedai_wandb.py --count 25
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from types import SimpleNamespace
import os
import sys

# Import PAI from parent directories
sys.path.append(os.path.abspath("../../../"))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

import wandb


# --- MODEL DEFINITION ---
class DendriticCIFARNet(nn.Module):
    """Deep CNN for CIFAR-10 with PAI compatibility."""
    
    def __init__(self, dropout=0.25, width_mult=1.0):
        super(DendriticCIFARNet, self).__init__()
        
        # Scale channels by width multiplier
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        fc_size = int(512 * width_mult)
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        
        # Conv Block 2
        self.conv3 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv4 = nn.Conv2d(c2, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        
        # Conv Block 3
        self.conv5 = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv6 = nn.Conv2d(c3, c3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        
        # Fully connected
        self.fc1 = nn.Linear(c3 * 4 * 4, fc_size)
        self.fc2 = nn.Linear(fc_size, 10)
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # Block 1
        x = F.silu(self.conv1(x))
        x = F.silu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2
        x = F.silu(self.conv3(x))
        x = F.silu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        
        # Block 3
        x = F.silu(self.conv5(x))
        x = F.silu(self.bn3(self.conv6(x)))
        x = self.pool(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- SWEEP PARAMETERS ---
def get_parameters_dict():
    """Return the parameters dictionary for W&B sweep."""
    parameters_dict = {
        # Network Architecture
        "dropout": {"values": [0.1, 0.25, 0.5]},
        "width_mult": {"values": [0.5, 1.0, 1.5]},
        
        # Training Hyperparameters
        "learning_rate": {"values": [0.01, 0.05, 0.1]},
        "weight_decay": {"values": [0, 0.0001, 0.001]},
        "batch_size": {"values": [64, 128]},
        
        # PAI Dendritic Parameters
        "improvement_threshold": {"values": [0, 1, 2]},
        "dendrite_mode": {"values": [0, 1]},  # 0=Traditional, 1=GD Dendrites
    }
    return parameters_dict


# --- TRAINING LOOP ---
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_acc = 100. * correct / len(train_loader.dataset)
    
    # PAI: Track training score
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    model.to(device)
    
    return train_acc


# --- TESTING LOOP ---
def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({val_acc:.2f}%)\n')
    
    # PAI: Add validation score and check for restructuring
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
    model.to(device)
    
    # Reset optimizer if dendrites were added
    if restructured and not training_complete:
        print("  ✨ Dendrites Added! Resetting optimizer.")
        optimArgs = {
            'params': model.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
        schedArgs = {'step_size': 10, 'gamma': args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    return model, optimizer, scheduler, training_complete, val_acc, restructured


# --- MAIN FUNCTION ---
def main(run=None):
    parser = argparse.ArgumentParser(description='Dense-Tune CIFAR-10 with W&B')
    parser.add_argument('--save-name', type=str, default='PAI_CIFAR10_WB')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-mps', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--sweep-id', type=str, default='main')
    
    args, _ = parser.parse_known_args()
    
    # Apply W&B config if available
    if run is not None:
        config = run.config
        if hasattr(config, 'learning_rate'): args.lr = config.learning_rate
        if hasattr(config, 'batch_size'): args.batch_size = config.batch_size
        if hasattr(config, 'weight_decay'): args.weight_decay = config.weight_decay
        if hasattr(config, 'dropout'): args.dropout = config.dropout
        if hasattr(config, 'width_mult'): args.width_mult = config.width_mult
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"⚙️ Device: {device}")
    
    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    # PAI Configuration
    try:
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_output_dimensions([-1, 0, -1, -1])  # For Conv layers
        GPA.pc.set_improvement_threshold([0.01, 0.001, 0.0001, 0])
        GPA.pc.set_switch_mode(GPA.PASwitchMode.DOING_HISTORY)
    except Exception as e:
        print(f"PAI config warning: {e}")
    
    # Initialize model with PAI
    model = DendriticCIFARNet(dropout=args.dropout, width_mult=args.width_mult).to(device)
    model = UPA.initialize_pai(model, save_name=args.save_name)
    
    # Optimizer setup
    GPA.pai_tracker.set_optimizer(optim.SGD)
    GPA.pai_tracker.set_scheduler(StepLR)
    
    optimArgs = {
        'params': model.parameters(),
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': args.weight_decay
    }
    schedArgs = {'step_size': 10, 'gamma': args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Tracking variables
    dendrite_count = 0
    max_val = 0
    global_max_val = 0
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete, val_acc, restructured = test(
            model, device, test_loader, optimizer, scheduler, args
        )
        
        scheduler.step()
        
        # Track metrics
        current_params = UPA.count_params(model)
        current_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
        
        if val_acc > max_val:
            max_val = val_acc
        if val_acc > global_max_val:
            global_max_val = val_acc
        
        # Log to W&B
        if run is not None:
            metrics = {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "param_count": current_params,
                "dendrite_count": current_dendrites,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # Log architecture snapshot on dendrite growth
            if restructured and current_dendrites != dendrite_count:
                dendrite_count = current_dendrites
                metrics.update({
                    "arch_max_val": max_val,
                    "arch_param_count": current_params,
                    "arch_dendrites": current_dendrites - 1
                })
                max_val = 0  # Reset for new architecture
            
            run.log(metrics)
        
        print(f"Epoch {epoch:03d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
              f"Dendrites: {current_dendrites} | Params: {current_params:,}")
        
        if training_complete:
            print("🏁 Training Complete (PAI stopped)")
            if run is not None:
                run.log({
                    "final_val_acc": global_max_val,
                    "final_param_count": current_params,
                    "final_dendrites": current_dendrites
                })
            break
    
    print(f"\n✅ Best Validation Accuracy: {global_max_val:.2f}%")


def run_wrapper():
    """Wrapper for W&B sweep agent."""
    try:
        with wandb.init() as wandb_run:
            main(wandb_run)
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep-id", type=str, default="main")
    parser.add_argument("--count", type=int, default=1)
    args, _ = parser.parse_known_args()
    
    wandb.login()
    
    # Sweep configuration
    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': get_parameters_dict()
    }
    
    if args.sweep_id == "main":
        sweep_id = wandb.sweep(sweep_config, project="Dense-Tune-CIFAR10")
        print(f"🚀 Initialized sweep: {sweep_id}")
        wandb.agent(sweep_id, run_wrapper, count=args.count)
    else:
        wandb.agent(args.sweep_id, run_wrapper, count=args.count, project="Dense-Tune-CIFAR10")
