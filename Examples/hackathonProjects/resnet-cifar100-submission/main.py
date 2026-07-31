"""
ResNet-CIFAR100 Training with PerforatedAI Dendritic Optimization
Adapted from Emotion Recognition Example and User Reference
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from types import SimpleNamespace

# Add repository root to sys.path to import perforatedai
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(repo_root)

# Weights & Biases
import wandb

# PerforatedAI imports
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
except ImportError:
    # Fallback/Debug
    print(f"Failed to import perforatedai from {repo_root}")
    # Try adding one more level up if running from deep inside
    repo_root_2 = os.path.abspath(os.path.join(current_dir, "../../"))
    sys.path.append(repo_root_2)
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA


from model import get_model
from utils import get_dataloaders


def train(args, model, device, train_loader, optimizer, epoch, use_dendrites):
    """Train for one epoch."""
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        pbar.set_postfix({'loss': f'{running_loss / (batch_idx + 1):.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    train_acc = 100.0 * correct / total
    
    # Add training score to PAI tracker
    if use_dendrites:
        GPA.pai_tracker.add_extra_score(train_acc, "train")
    
    return train_acc


def test(model, device, test_loader, optimizer, scheduler, args, use_dendrites):
    """Evaluate model on validation/test data."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    val_acc = 100.0 * correct / total

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({val_acc:.2f}%)\n')

    # Add validation score to PAI tracker - this may restructure the model with dendrites
    restructured = False
    training_complete = False
    
    if use_dendrites:
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_acc, model
        )
        model.to(device)

        # If restructured (dendrite added), reset optimizer and scheduler
        if restructured and not training_complete:
            print("Model restructured (dendrites added). Re-initializing optimizer...")
            optimArgs = {
                'params': model.parameters(),
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'momentum': 0.9
            }
            schedArgs = {'step_size': args.step_size, 'gamma': args.gamma}
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, optimArgs, schedArgs
            )

    return model, optimizer, scheduler, training_complete, val_acc, restructured


def main(run=None):
    """Main training function."""
    parser = argparse.ArgumentParser(description='ResNet-CIFAR100 with PerforatedAI + W&B')

    # Core Arguments
    parser.add_argument('--mode', type=str, default='dendritic', choices=['dendritic', 'standard'], 
                        help="Training mode")
    parser.add_argument('--split', type=float, default=1.0, 
                        help="Dataset split ratio (e.g. 0.5 for 50%)")
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay')
    parser.add_argument('--step-size', type=int, default=30, help='LR Step Size')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    
    # Specific Dendrite Settings
    parser.add_argument('--dendrites', type=int, default=5, help="Max Dendrite Count")
    
    parser.add_argument('--dry-run', action='store_true', default=False, help='Fast run for debugging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # W&B arguments
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='resnet-cifar100-pai', help='W&B project name')

    args = parser.parse_args()
    
    use_dendrites = (args.mode == 'dendritic')

    # W&B Config
    if run is not None:
        config = run.config
    else:
        # Default config mimicking simple args
        config = SimpleNamespace(
            mode=args.mode,
            split=args.split,
            dendrites=args.dendrites,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )

    # Set Device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    print(f"Wrapper Mode: {args.mode}, Split: {args.split*100}%")

    torch.manual_seed(args.seed)

    # PAI Configuration
    if use_dendrites:
        GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
        GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
        GPA.pc.set_pai_forward_function(torch.sigmoid)
        
        GPA.pc.set_max_dendrites(args.dendrites)
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_verbose(True)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        # Assuming ~5 dendrites over 60 epochs => one check every 10 epochs
        GPA.pc.set_n_epochs_to_switch(max(1, args.epochs // (args.dendrites + 1))) 
        # GPA.pc.set_dendrite_update_mode(True)
        # GPA.pc.set_initial_history_after_switches(0)

    # Weights & Biases Run Name
    name_str = f"{args.mode}_split-{int(args.split*100)}"
    if use_dendrites:
        name_str = f"dendritic_d{args.dendrites}_{name_str}"
    
    if run is None and not args.no_wandb:
         wandb.init(project=args.project, name=name_str, config=config)
         run = wandb.run
    elif run is not None:
         run.name = name_str

    # Load Data
    train_loader, test_loader = get_dataloaders(args.batch_size, split_ratio=args.split, seed=args.seed)
    
    if args.dry_run:
        print("Dry run: limiting epochs to 1")
        args.epochs = 1

    # Create model
    model = get_model(num_classes=100)
    model = model.to(device)

    # Initialize PAI
    if use_dendrites:
        print("Initializing PerforatedAI dendrites...")
        # Custom tracking logic if needed, but 'get_model' is standard ResNet so auto-tracking should work
        # or we can force it like in the reference if ResNet structure is tricky
        # GPA.pc.append_module_ids_to_track(['.layer1', '.layer2', ...]) 
        
        model = UPA.initialize_pai(model, save_name=name_str)
        model = model.to(device)
        
        # Warmup / Shape Initialization
        print("Running dummy forward/backward pass to initialize PAI shapes...")
        model.train()
        dummy_input = torch.randn(2, 3, 32, 32).to(device) # CIFAR size
        dummy_label = torch.randint(0, 100, (2,)).to(device)
        
        temp_optim = optim.SGD(model.parameters(), lr=0.001)
        temp_optim.zero_grad()
        dummy_out = model(dummy_input)
        temp_loss = nn.CrossEntropyLoss()(dummy_out, dummy_label)
        temp_loss.backward()
        temp_optim.step()
        print("Warmup complete.")

    # Optimizer Setup
    # Note: PAI tracker requires ownership of the optimizer in dendritic mode
    if use_dendrites:
        GPA.pai_tracker.set_optimizer(optim.SGD)
        GPA.pai_tracker.set_scheduler(StepLR)
    
    weight_decay = args.weight_decay
    optimArgs = {
        'params': model.parameters(),
        'lr': args.lr,
        'weight_decay': weight_decay,
        'momentum': 0.9
    }
    schedArgs = {'step_size': args.step_size, 'gamma': args.gamma}
    
    if use_dendrites:
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    else:
        optimizer = optim.SGD(**optimArgs)
        scheduler = StepLR(optimizer, **schedArgs)

    # Param Count Logging
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params}")
    if run is not None:
        run.log({"total_parameters": params})
        
    # Training Loop
    print(f"\nStarting Training: Mode={args.mode}")
    print("="*60)
    
    epoch = -1
    dendrite_count = 0
    max_val = 0
    
    while True:
        epoch += 1
        
        if epoch >= args.epochs:
             print(f"Reached maximum epochs ({args.epochs}). Stopping.")
             break

        # Train
        train_acc = train(args, model, device, train_loader, optimizer, epoch, use_dendrites)

        # Validate
        model, optimizer, scheduler, training_complete, val_acc, restructured = test(
            model, device, test_loader, optimizer, scheduler, args, use_dendrites
        )
        
        # Scheduler Step
        if not use_dendrites:
            scheduler.step()
        else:
            # In dendritic mode, check if we should step manually or if PAI handles it. 
            # Reference Emotion sample didn't explicit step but ResNet needs it. 
            # Safest is to step scheduler if it exists.
             if scheduler:
                 # Check if PAI tracker wraps it in a way that needs caution
                 # Standard PAI usage: user steps scheduler.
                 scheduler.step()

        # Logging / Tracking
        if val_acc > max_val:
            max_val = val_acc
            
        current_dendrites = 0
        if use_dendrites:
            current_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Dendrites: {current_dendrites}")

        if run is not None:
             run.log({
                "epoch": epoch + 1,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "dendrite_count": current_dendrites,
                "lr": scheduler.get_last_lr()[0] if scheduler else args.lr
            })

        # Update PAI graphs every epoch
        if use_dendrites:
             GPA.pai_tracker.save_graphs()

        if training_complete and use_dendrites:
            print("\nPerforatedAI training complete (Tracker Signal).")
            break

    return max_val

if __name__ == '__main__':
    main()
