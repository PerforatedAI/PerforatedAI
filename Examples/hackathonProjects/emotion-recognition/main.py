"""
Emotion Recognition Training with PerforatedAI Dendritic Optimization
With Weights & Biases Integration - Following Official Example Pattern

Usage:
    # Standard training with dendrites and W&B logging
    python main.py --data_dir ./data/ravdess --epochs 100
    
    # Run W&B hyperparameter sweep
    python main.py --use-wandb --sweep-id main --count 10
    
    # Disable W&B logging
    python main.py --data_dir ./data/ravdess --no-wandb
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from types import SimpleNamespace

# Weights & Biases
import wandb

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from model import get_model
from dataset import get_data_loaders, create_synthetic_dataset


def train(args, model, device, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    correct = 0
    total = 0
    
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    train_acc = 100.0 * correct / len(train_loader.dataset)
    
    # Add training score to PAI tracker
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    model.to(device)
    
    return train_acc


def test(model, device, test_loader, optimizer, scheduler, args):
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
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({val_acc:.0f}%)\n')
    
    # Add validation score to PAI tracker - this may restructure the model with dendrites
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
    
    return model, optimizer, scheduler, training_complete, val_acc, restructured


def main(run=None):
    """Main training function - can be called standalone or from W&B sweep."""
    parser = argparse.ArgumentParser(description='Emotion Recognition with PerforatedAI + W&B')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/ravdess',
                        help='Path to RAVDESS dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet'], help='Model architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs (PAI will auto-stop)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.98,
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
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable W&B sweep mode')
    parser.add_argument('--sweep-id', type=str, default='main',
                        help='Sweep ID to join, or "main" to create new sweep')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of sweep runs to perform')
    parser.add_argument('--project', type=str, default='emotion-recognition-pai',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Get config from wandb run or use defaults
    if run is not None:
        config = run.config
    else:
        config = SimpleNamespace(
            dropout=0.3,
            weight_decay=args.weight_decay,
            improvement_threshold=1,
            candidate_weight_initialization_multiplier=0.01,
            pai_forward_function=0,
            dendrite_mode=1,
        )
    
    # Set PAI configuration
    # Decode improvement_threshold
    if hasattr(config, 'improvement_threshold'):
        if config.improvement_threshold == 0:
            thresh = [0.01, 0.001, 0.0001, 0]
        elif config.improvement_threshold == 1:
            thresh = [0.001, 0.0001, 0]
        elif config.improvement_threshold == 2:
            thresh = [0]
        else:
            thresh = [0.001, 0.0001, 0]
    else:
        thresh = [0.001, 0.0001, 0]
    GPA.pc.set_improvement_threshold(thresh)
    
    # Set candidate weight initialization
    cwim = getattr(config, 'candidate_weight_initialization_multiplier', 0.01)
    GPA.pc.set_candidate_weight_initialization_multiplier(cwim)
    
    # Decode pai_forward_function
    pff = getattr(config, 'pai_forward_function', 0)
    if pff == 0:
        pai_forward_function = torch.sigmoid
    elif pff == 1:
        pai_forward_function = torch.relu
    elif pff == 2:
        pai_forward_function = torch.tanh
    else:
        pai_forward_function = torch.sigmoid
    GPA.pc.set_pai_forward_function(pai_forward_function)
    
    # Set dendrite mode
    dendrite_mode = getattr(config, 'dendrite_mode', 1)
    if dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    else:
        GPA.pc.set_max_dendrites(5)
    
    # Other PAI settings
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_n_epochs_to_switch(25)

    # Set wandb run name
    if run is not None:
        dropout_val = getattr(config, 'dropout', 0.3)
        wd_val = getattr(config, 'weight_decay', args.weight_decay)
        name_str = f"Dendrites-{dendrite_mode}_dropout{dropout_val}_wd{wd_val}"
        run.name = name_str
    
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    
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
    dropout = getattr(config, 'dropout', 0.3)
    model = get_model(
        model_type=args.model,
        num_classes=num_classes,
        dropout_rate=dropout
    )
    model = model.to(device)
    
    # Initialize PAI - this wraps the model with dendritic capabilities
    print("Initializing PerforatedAI dendrites...")
    model = UPA.initialize_pai(model, save_name=args.save_name)
    model = model.to(device)
    
    # Setup optimizer and scheduler using PAI tracker
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(StepLR)
    
    weight_decay = getattr(config, 'weight_decay', args.weight_decay)
    optimArgs = {
        'params': model.parameters(),
        'lr': args.lr,
        'weight_decay': weight_decay,
    }
    schedArgs = {'step_size': 1, 'gamma': args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Initialize tracking variables for architecture logging (per official example)
    dendrite_count = 0
    max_val = 0
    max_train = 0
    max_params = 0
    
    global_max_val = 0
    global_max_train = 0
    global_max_params = 0
    
    print("\n" + "="*60)
    print("Starting Training with Dendritic Optimization")
    print("="*60 + "\n")
    
    epoch = -1
    #for epoch in range(1, args.epochs + 1):
    while True:
        epoch += 1
        # Train
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        
        # Validate and potentially restructure model with new dendrites
        model, optimizer, scheduler, training_complete, val_acc, restructured = test(
            model, device, val_loader, optimizer, scheduler, args
        )
        
        # Update max values for current architecture
        if val_acc > max_val:
            max_val = val_acc
            max_train = train_acc
            max_params = UPA.count_params(model)
        
        # Update global max values
        if val_acc > global_max_val:
            global_max_val = val_acc
            global_max_train = train_acc
            global_max_params = UPA.count_params(model)
        
        # Log to wandb
        if run is not None:
            run.log({
                'ValAcc': val_acc,
                'TrainAcc': train_acc,
                'Param Count': UPA.count_params(model),
                'Dendrite Count': GPA.pai_tracker.member_vars['num_dendrites_added'],
            })
            
            # Log architecture maximums when dendrites are added
            if restructured:
                if GPA.pai_tracker.member_vars['mode'] == 'n' and (
                    dendrite_count != GPA.pai_tracker.member_vars['num_dendrites_added']
                ):
                    # Reset architecture-level tracking
                    run.log({
                        'Arch Max Val': max_val,
                        'Arch Max Train': max_train,
                        'Arch Param Count': max_params,
                        'Arch Dendrite Count': GPA.pai_tracker.member_vars['num_dendrites_added'] - 1,
                    })
                    dendrite_count = GPA.pai_tracker.member_vars['num_dendrites_added']
                    # Reset max values for new architecture
                    max_val = 0
                    max_train = 0
                    max_params = 0
                    print(f"  -> 🌳 DENDRITE ADDED! Total: {dendrite_count}")
        else:
            # Console logging when not using wandb
            current_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Dendrites: {current_dendrites}")
            
            if restructured and current_dendrites != dendrite_count:
                dendrite_count = current_dendrites
                print(f"  -> 🌳 DENDRITE ADDED! Total: {dendrite_count}, Params: {UPA.count_params(model):,}")
        
        # Check if PAI training is complete
        if training_complete:
            # Log final architecture max
            if run is not None:
                run.log({
                    'Arch Max Val': max_val,
                    'Arch Max Train': max_train,
                    'Arch Param Count': max_params,
                    'Arch Dendrite Count': GPA.pai_tracker.member_vars['num_dendrites_added'],
                })
                # Log final global max
                run.log({
                    'Final Max Val': global_max_val,
                    'Final Max Train': global_max_train,
                    'Final Param Count': global_max_params,
                    'Final Dendrite Count': GPA.pai_tracker.member_vars['num_dendrites_added'],
                })
            
            print("\n" + "="*60)
            print("PerforatedAI training complete!")
            print(f"Final Max Val: {global_max_val:.2f}%")
            print(f"Final Max Train: {global_max_train:.2f}%")
            print(f"Final Dendrite Count: {GPA.pai_tracker.member_vars['num_dendrites_added']}")
            print(f"Final Param Count: {global_max_params:,}")
            print("="*60)
            break
    
    print(f"\nResults graph saved to: {args.save_name}/{args.save_name}.png")
    
    return global_max_val


def get_parameters_dict():
    """Return the parameters dictionary for the sweep."""
    parameters_dict = {
        'dropout': {'values': [0.2, 0.3, 0.4]},
        'weight_decay': {'values': [0, 0.0001, 0.001]},
        'improvement_threshold': {'values': [0, 1, 2]},
        'candidate_weight_initialization_multiplier': {'values': [0.1, 0.01]},
        'pai_forward_function': {'values': [0, 1, 2]},
        'dendrite_mode': {'values': [0, 1]},  # 0 = no dendrites, 1 = GD dendrites
    }
    return parameters_dict


def run_sweep():
    """Wrapper function for wandb sweep."""
    try:
        with wandb.init() as wandb_run:
            main(wandb_run)
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # Parse minimal args for sweep control
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--sweep-id', type=str, default='main')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--project', type=str, default='emotion-recognition-pai')
    parser.add_argument('--no-wandb', action='store_true')
    args, _ = parser.parse_known_args()
    
    if args.use_wandb and not args.no_wandb:
        # W&B sweep mode
        wandb.login()
        project = args.project
        
        sweep_config = {'method': 'random'}
        metric = {'name': 'Final Max Val', 'goal': 'maximize'}
        sweep_config['metric'] = metric
        parameters_dict = get_parameters_dict()
        sweep_config['parameters'] = parameters_dict
        
        if args.sweep_id == 'main':
            sweep_id = wandb.sweep(sweep_config, project=project)
            print(f"\nInitialized sweep: {sweep_id}")
            print(f"Use --sweep-id {sweep_id} to join on other machines.\n")
            wandb.agent(sweep_id, run_sweep, count=args.count)
        else:
            wandb.agent(args.sweep_id, run_sweep, count=args.count, project=project)
    else:
        # Standard training without sweep
        if not args.no_wandb:
            wandb.init(project=args.project, name='emotion-recognition-training')
            main(wandb.run)
            wandb.finish()
        else:
            main()
