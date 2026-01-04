"""
Emotion Recognition Training with PerforatedAI Dendritic Optimization
With Weights & Biases Integration for Experiment Tracking

Usage:
    # Standard training with dendrites and W&B logging
    python main.py --data_dir ./data/ravdess --epochs 50
    
    # Run W&B hyperparameter sweep
    python main.py --data_dir ./data/ravdess --sweep --count 10
    
    # Quick test with synthetic data (no dataset required)
    python main.py --synthetic --epochs 10
    
    # Disable W&B logging
    python main.py --data_dir ./data/ravdess --no-wandb
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Weights & Biases
import wandb

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from model import get_model
from dataset import get_data_loaders, create_synthetic_dataset, IDX_TO_EMOTION


def train(args, model, device, train_loader, optimizer, epoch, use_wandb=True):
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
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    train_acc = 100.0 * correct / len(train_loader.dataset)
    avg_loss = running_loss / len(train_loader)
    
    # Add training score to PAI tracker
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    model.to(device)
    
    return train_acc, avg_loss


def test(model, device, test_loader, optimizer, scheduler, args, use_wandb=True):
    """Evaluate model on validation/test data."""
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
    val_acc = 100.0 * correct / len(test_loader.dataset)
    
    print(f'\nValidation: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({val_acc:.2f}%)\n')
    
    # Add validation score to PAI tracker - this may restructure the model with new dendrites
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    model.to(device)
    
    # If restructured (dendrite added), reset optimizer and scheduler
    if restructured and not training_complete:
        optimArgs = {
            'params': model.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }
        schedArgs = {'step_size': 1, 'gamma': args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, schedArgs
        )
    
    return model, optimizer, scheduler, training_complete, val_acc, test_loss, restructured


def main(config=None):
    parser = argparse.ArgumentParser(description='Emotion Recognition with PerforatedAI + W&B')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/ravdess',
                        help='Path to RAVDESS dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # PAI arguments
    parser.add_argument('--save-name', type=str, default='PAI',
                        help='Save name for PAI outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # W&B arguments
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--sweep', action='store_true',
                        help='Run W&B hyperparameter sweep')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of sweep runs')
    parser.add_argument('--project', type=str, default='emotion-recognition-pai',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Override with sweep config if provided
    if config is not None:
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    use_wandb = not args.no_wandb
    
    # Initialize W&B
    if use_wandb and config is None:  # Don't re-init if already in sweep
        wandb.init(
            project=args.project,
            config={
                'model': args.model,
                'dropout': args.dropout,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'weight_decay': args.weight_decay,
                'epochs': args.epochs,
                'optimizer': 'Adam',
                'scheduler': 'StepLR',
                'gamma': args.gamma,
                'dendrite_type': 'GD',
            },
            name=f"emotion-{args.model}-lr{args.lr}-bs{args.batch_size}"
        )
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    if args.synthetic:
        print("Using synthetic dataset for testing...")
        train_loader, val_loader, test_loader = create_synthetic_dataset(
            num_samples=500, num_classes=8
        )
    else:
        print(f"Loading RAVDESS dataset from {args.data_dir}...")
        train_loader, val_loader, test_loader = get_data_loaders(
            args.data_dir, batch_size=args.batch_size, num_workers=0
        )
    
    # Create model
    num_classes = 8  # 8 emotions in RAVDESS
    model = get_model(
        model_type=args.model,
        num_classes=num_classes,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    # Configure PAI settings
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)
    GPA.pc.set_max_dendrites(5)
    GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
    GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
    GPA.pc.set_pai_forward_function(torch.sigmoid)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # Initialize PAI - this wraps the model with dendritic capabilities
    print("Initializing PerforatedAI dendrites...")
    model = UPA.initialize_pai(model, save_name=args.save_name)
    model = model.to(device)
    
    # Watch model with W&B
    if use_wandb:
        wandb.watch(model, log='all', log_freq=100)
    
    # Setup optimizer and scheduler using PAI tracker
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(StepLR)
    
    optimArgs = {
        'params': model.parameters(),
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }
    schedArgs = {'step_size': 1, 'gamma': args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Training loop
    best_val_acc = 0.0
    dendrites_added = 0
    
    print("\n" + "="*60)
    print("Starting Training with Dendritic Optimization + W&B Logging")
    print("="*60 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_acc, train_loss = train(args, model, device, train_loader, optimizer, epoch, use_wandb)
        
        # Validate and potentially restructure model with new dendrites
        model, optimizer, scheduler, training_complete, val_acc, val_loss, restructured = test(
            model, device, val_loader, optimizer, scheduler, args, use_wandb
        )
        
        # Get current dendrite count and param count
        current_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
        param_count = UPA.count_params(model)
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Dendrites: {current_dendrites}")
        
        # Log to W&B
        if use_wandb:
            log_data = {
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'param_count': param_count,
                'dendrites_added': current_dendrites,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }
            wandb.log(log_data)
        
        if restructured:
            dendrites_added = current_dendrites
            print(f"  -> 🌳 DENDRITE ADDED! Total dendrites: {current_dendrites}, Params: {param_count:,}")
            if use_wandb:
                wandb.log({
                    'dendrite_added_epoch': epoch,
                    'dendrite_count': current_dendrites,
                    'params_after_dendrite': param_count,
                })
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> 💾 New best model saved! (Val Acc: {val_acc:.2f}%)")
            if use_wandb:
                wandb.run.summary['best_val_acc'] = val_acc
                wandb.run.summary['best_epoch'] = epoch
        
        # Check if PAI training is complete
        if training_complete:
            print("\n" + "="*60)
            print("PerforatedAI training complete!")
            print("="*60)
            break
    
    # Final test evaluation
    print("\nFinal Test Evaluation:")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_acc = 100.0 * correct / len(test_loader.dataset)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Total Dendrites Added: {current_dendrites}")
    print(f"Total Parameters: {param_count:,}")
    print(f"{'='*60}")
    print(f"\nResults graph saved to: {args.save_name}/{args.save_name}.png")
    
    # Log final results to W&B
    if use_wandb:
        wandb.run.summary['test_acc'] = test_acc
        wandb.run.summary['total_dendrites'] = current_dendrites
        wandb.run.summary['final_params'] = param_count
        wandb.finish()
    
    return test_acc


def run_sweep():
    """Run W&B hyperparameter sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='emotion-recognition-pai')
    parser.add_argument('--count', type=int, default=10)
    args, _ = parser.parse_known_args()
    
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'dropout': {
                'values': [0.2, 0.3, 0.4, 0.5]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            'model': {
                'values': ['cnn', 'resnet']
            }
        }
    }
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"\n✨ Created sweep: {sweep_id}")
    print(f"View at: https://wandb.ai/{args.project}/{sweep_id}\n")
    
    # Run sweep agent
    def sweep_train():
        with wandb.init() as run:
            config = dict(wandb.config)
            main(config=config)
    
    wandb.agent(sweep_id, sweep_train, count=args.count)


if __name__ == '__main__':
    import sys
    
    # Check if running sweep
    if '--sweep' in sys.argv:
        run_sweep()
    else:
        main()
