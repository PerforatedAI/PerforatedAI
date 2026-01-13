"""
Train baseline CNN model without dendrites.
Tracks experiments with MLflow.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import pickle

from utils.model import AudioCNN
from utils.data_utils import load_preprocessed_data, create_dataloaders
from utils.metrics import evaluate_model, plot_confusion_matrix


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
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Setup device (M4 Mac uses MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data_dict = load_preprocessed_data(args.data_dir)
    
    # Load label mapping
    with open(os.path.join(args.data_dir, 'label_mapping.pkl'), 'rb') as f:
        label_mapping = pickle.load(f)
    
    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(data_dict, batch_size=args.batch_size)
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Initialize model
    print("\nInitializing model...")
    model = AudioCNN(num_classes=50).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    
    # MLflow tracking
    mlflow.set_experiment("ESC-50-Baseline")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model': 'AudioCNN',
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'max_epochs': args.epochs,
            'patience': args.patience,
            'device': str(device),
            'num_parameters': model.count_parameters()
        })
        
        # Training loop
        print(f"\nTraining for up to {args.epochs} epochs...")
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, loaders['train'], criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc = validate(
                model, loaders['val'], criterion, device
            )
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            }, step=epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
                torch.save(model.state_dict(), 'models/baseline_best.pt')
                mlflow.log_metric('best_val_acc', best_val_acc)
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        
        # Load best model for final evaluation
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load('models/baseline_best.pt'))
        
        # Final test evaluation
        print("Evaluating on test set...")
        test_results = evaluate_model(model, loaders['test'], criterion, device)
        
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_results['loss']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
        
        # Log final metrics
        mlflow.log_metrics({
            'final_test_loss': test_results['loss'],
            'final_test_acc': test_results['accuracy']
        })
        
        # Plot and save confusion matrix
        print("\nGenerating confusion matrix...")
        cm = plot_confusion_matrix(
            test_results['labels'],
            test_results['predictions'],
            label_names=None,  # Too many classes for readable labels
            save_path='models/baseline_confusion_matrix.png'
        )
        mlflow.log_artifact('models/baseline_confusion_matrix.png')
        
        # Save results to JSON
        results = {
            'model': 'Baseline CNN',
            'test_accuracy': float(test_results['accuracy']),
            'test_loss': float(test_results['loss']),
            'best_val_accuracy': float(best_val_acc),
            'num_parameters': model.count_parameters(),
            'epochs_trained': epoch + 1
        }
        
        with open('models/baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        mlflow.log_artifact('models/baseline_results.json')
        mlflow.pytorch.log_model(model, "model")
        
        print("\nTraining complete!")
        print(f"Best model saved to: models/baseline_best.pt")
        print(f"Results saved to: models/baseline_results.json")
        print(f"\nTo view MLflow results, run: mlflow ui")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline CNN on ESC-50')
    parser.add_argument('--data_dir', type=str, default='preprocessed',
                        help='Directory with preprocessed data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_baseline(args)
