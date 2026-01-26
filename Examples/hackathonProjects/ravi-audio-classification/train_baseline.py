"""
Train baseline CNN14 model without dendrites on ESC-50.
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

from utils.model import CNN14ESC50
from utils.data_utils import load_preprocessed_data, create_dataloaders
from utils.metrics import evaluate_model, plot_confusion_matrix
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
    """Train for one epoch"""
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


def validate(model, dataloader, criterion, device):
    """Validate model"""
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
    
    return val_loss, val_acc


def train_baseline(args):
    """Main training function"""
    
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
    
    # Initialize model
    print("\nInitializing CNN14 model...")
    model = CNN14ESC50(num_classes=config.MODEL['num_classes'])
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    lr = args.lr if args.lr is not None else config.TRAINING['learning_rate']
    weight_decay = args.weight_decay if args.weight_decay is not None else config.TRAINING['weight_decay']
    
    if config.OPTIMIZER['type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER['type']}")
    
    # Scheduler
    if config.SCHEDULER['type'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.SCHEDULER['mode'],
            patience=config.SCHEDULER['patience'],
            factor=config.SCHEDULER['factor']
        )
    else:
        scheduler = None
    
    # Training configuration
    max_epochs = args.epochs if args.epochs is not None else config.TRAINING['max_epochs']
    patience = args.patience if args.patience is not None else config.TRAINING['patience']
    best_model_path = os.path.join(config.MODELS_DIR, 'baseline_best.pt')
    
    # Training loop
    print(f"\nTraining for up to {max_epochs} epochs...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, loaders['val'], criterion, device
        )
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Final test evaluation
    print("Evaluating on test set...")
    test_results = evaluate_model(model, loaders['test'], criterion, device)
    
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    # Plot and save confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(config.MODELS_DIR, 'baseline_confusion_matrix.png')
    cm = plot_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        label_names=None,  # Too many classes for readable labels
        save_path=cm_path
    )
    
    # Save results to JSON
    results = {
        'model': 'Baseline CNN14',
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'best_val_accuracy': float(best_val_acc),
        'num_parameters': model.count_parameters(),
        'epochs_trained': epoch + 1
    }
    
    results_path = os.path.join(config.MODELS_DIR, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline CNN on ESC-50')
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
    parser.add_argument('--patience', type=int, default=None,
                        help=f'Early stopping patience (default: {config.TRAINING["patience"]})')
    
    args = parser.parse_args()
    
    # Use config defaults if args are None
    if args.data_dir is None:
        args.data_dir = config.OUTPUT_DIR
    
    train_baseline(args)
