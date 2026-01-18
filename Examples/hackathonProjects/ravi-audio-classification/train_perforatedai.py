"""
Train CNN14 model WITH dendrites using PerforatedAI on ESC-50.
"""
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from utils.model import CNN14ESC50
from utils.data_utils import load_preprocessed_data, create_dataloaders
from utils.metrics import evaluate_model, plot_confusion_matrix, calculate_error_reduction
import config


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS backend doesn't support deterministic operations yet
        pass
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return accuracy for PAI tracking"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, optimizer, args):
    """
    Validate model and handle PAI restructuring.
    Returns updated model, optimizer, scheduler, training_complete flag, and metrics.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    
    # Add validation score to PAI tracker - this may trigger restructuring
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    model.to(device)
    
    # If restructured, reset optimizer and scheduler
    if restructured and not training_complete:
        optimArgs = {
            'params': model.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
        schedArgs = {
            'mode': config.SCHEDULER['mode'],
            'patience': config.SCHEDULER['patience'],
            'factor': config.SCHEDULER['factor']
        }
        optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    return model, optimizer, training_complete, restructured, val_loss, val_acc


def train_perforatedai(args):
    """Main training function with PerforatedAI dendrites"""
    
    # Set random seed for reproducibility
    set_seed(config.PREPROCESSING['random_state'])
    
    # Create directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Setup device
    if config.DEVICE['prefer_mps'] and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal GPU)")
    elif config.DEVICE['prefer_cuda'] and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data_dict = load_preprocessed_data(args.data_dir)
    
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = args.batch_size if args.batch_size is not None else config.TRAINING['batch_size']
    loaders = create_dataloaders(
        data_dict, 
        batch_size=batch_size,
        num_workers=2
    )
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # ========================================================================
    # PerforatedAI Configuration
    # ========================================================================
    print("\n" + "="*60)
    print("Configuring PerforatedAI...")
    print("="*60)
    
    # Get PAI config from config file or args
    max_dendrites = args.max_dendrites if args.max_dendrites != 5 else config.PAI.get('max_dendrites', 5)
    
    # Set PAI global parameters
    GPA.pc.set_testing_dendrite_capacity(False)  # Real training mode
    GPA.pc.set_unwrapped_modules_confirmed(True)  # Skip BatchNorm warnings
    GPA.pc.set_perforated_backpropagation(False)  # Use GD dendrites (open source)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)  # Reduce output noise
    
    # Set improvement threshold (when to stop adding dendrites)
    # Lower thresholds mean more dendrites can be added
    GPA.pc.set_improvement_threshold(config.PAI.get('improvement_threshold', [0.001, 0.0001, 0]))
    
    # Set max dendrites (0 = unlimited, but usually 3-5 is enough)
    GPA.pc.set_max_dendrites(max_dendrites)
    
    # Dendrite forward function
    forward_func_name = config.PAI.get('forward_function', 'sigmoid')
    forward_funcs = {'sigmoid': torch.sigmoid, 'relu': torch.relu, 'tanh': torch.tanh}
    GPA.pc.set_pai_forward_function(forward_funcs.get(forward_func_name, torch.sigmoid))
    
    # Weight initialization for new dendrites
    GPA.pc.set_candidate_weight_initialization_multiplier(
        config.PAI.get('weight_init_multiplier', 0.01)
    )
    
    print(f"Max dendrites: {max_dendrites}")
    
    # ========================================================================
    # Initialize Model with PAI
    # ========================================================================
    print("\nInitializing CNN14 model with PerforatedAI...")
    model = CNN14ESC50(num_classes=config.MODEL['num_classes'])
    
    # Convert model to PAI model (adds dendrite capability to layers)
    model = UPA.initialize_pai(model, save_name="PAI_CNN14")
    
    model = model.to(device)
    print(f"Initial parameters: {UPA.count_params(model):,}")
    
    # ========================================================================
    # Setup Optimizer via PAI Tracker
    # ========================================================================
    criterion = nn.CrossEntropyLoss()
    args.lr = args.lr if args.lr is not None else config.TRAINING['learning_rate']
    args.weight_decay = args.weight_decay if args.weight_decay is not None else config.TRAINING['weight_decay']
    
    # Set optimizer and scheduler types in PAI tracker
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(optim.lr_scheduler.ReduceLROnPlateau)
    
    # Setup optimizer through PAI tracker
    optimArgs = {
        'params': model.parameters(),
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    schedArgs = {
        'mode': config.SCHEDULER['mode'],
        'patience': config.SCHEDULER['patience'],
        'factor': config.SCHEDULER['factor']
    }
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Training configuration
    max_epochs = args.epochs if args.epochs is not None else config.TRAINING['max_epochs']
    best_model_path = os.path.join(config.MODELS_DIR, 'pai_best.pt')
    
    # ====================================================================
    # Training Loop (PAI controlled)
    # ====================================================================
    print(f"\nTraining with PerforatedAI (max {max_epochs} epochs)...")
    print("Note: Training will continue until PAI determines convergence.")
    
    best_val_acc = 0.0
    dendrite_count = 0
    
    for epoch in range(max_epochs):
            current_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
            print(f"\nEpoch {epoch + 1}/{max_epochs} (Dendrites: {current_dendrites})")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, loaders['train'], criterion, optimizer, device
            )
            
            # Track training accuracy with PAI
            GPA.pai_tracker.add_extra_score(train_acc, 'Train')
            model.to(device)
            
            # Evaluate on test set (for tracking only, not for training decisions)
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, labels in loaders['test']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            test_acc = 100.0 * test_correct / test_total
            
            # Track test score BEFORE validation (per API recommendation)
            GPA.pai_tracker.add_test_score(test_acc, 'Test Accuracy')
            model.to(device)
            
            # Validate (this handles PAI restructuring)
            model, optimizer, training_complete, restructured, val_loss, val_acc = validate(
                model, loaders['val'], criterion, device, optimizer, args
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Test Acc: {test_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Parameters: {UPA.count_params(model):,}")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
                torch.save(model.state_dict(), best_model_path)
            
            # Check if dendrites were added
            if restructured:
                new_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
                if new_dendrites > dendrite_count:
                    dendrite_count = new_dendrites
                    print(f"\n*** DENDRITES ADDED! Now have {dendrite_count} dendrite(s) ***")
                    print(f"New parameter count: {UPA.count_params(model):,}")
            
            # Check if PAI says training is complete
            if training_complete:
                print(f"\nPAI training complete at epoch {epoch + 1}")
                print("PAI determined that additional dendrites won't improve performance.")
                break
    
    # ====================================================================
    # Final Evaluation
    # ====================================================================
    print("\n" + "="*60)
    print("Training Complete - Final Evaluation")
    print("="*60)
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    try:
        model.load_state_dict(torch.load(best_model_path), strict=False)
    except Exception as e:
        print(f"Warning: Could not load best model ({e})")
        print("Using current model state instead...")
    model.to(device)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(model, loaders['test'], criterion, device)
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Final Parameter Count: {UPA.count_params(model):,}")
    print(f"Total Dendrites Added: {GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(config.MODELS_DIR, 'pai_confusion_matrix.png')
    cm = plot_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        label_names=None,
        save_path=cm_path
    )
    
    # ====================================================================
    # Compare with Baseline
    # ====================================================================
    baseline_results_path = os.path.join(config.MODELS_DIR, 'baseline_results.json')
    if os.path.exists(baseline_results_path):
        print("\n" + "="*60)
        print("Comparison with Baseline")
        print("="*60)
        
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        
        baseline_acc = baseline_results['test_accuracy']
        pai_acc = test_results['accuracy']
        
        improvement = pai_acc - baseline_acc
        error_reduction = calculate_error_reduction(baseline_acc, pai_acc)
        
        print(f"\nBaseline Test Accuracy: {baseline_acc:.2f}%")
        print(f"PAI Test Accuracy:      {pai_acc:.2f}%")
        print(f"Improvement:            {improvement:+.2f}%")
        print(f"Error Reduction:        {error_reduction:.2f}%")
    
    # Save results
    results = {
        'model': 'PAI_CNN14',
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'best_val_accuracy': float(best_val_acc),
        'num_parameters': UPA.count_params(model),
        'epochs_trained': epoch + 1,
        'dendrites_added': GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
    }
    
    results_path = os.path.join(config.MODELS_DIR, 'pai_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Best model saved to: {best_model_path}")
    print("PAI output graphs saved to: PAI_CNN14/PAI_CNN14.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN14 with PerforatedAI dendrites on ESC-50')
    parser.add_argument('--data_dir', type=str, default=None,
                        help=f'Directory with preprocessed data (default: {config.OUTPUT_DIR})')
    parser.add_argument('--batch_size', type=int, default=None,
                        help=f'Batch size for training (default: {config.TRAINING["batch_size"]})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Learning rate (default: {config.TRAINING["learning_rate"]})')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help=f'Weight decay (default: {config.TRAINING["weight_decay"]})')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Maximum number of epochs (default: {config.TRAINING["max_epochs"]})')
    parser.add_argument('--max_dendrites', type=int, default=5,
                        help='Maximum number of dendrites to add (default: 5)')
    
    args = parser.parse_args()
    
    # Use config defaults if args are None
    if args.data_dir is None:
        args.data_dir = config.OUTPUT_DIR
    
    train_perforatedai(args)
