"""
Train baseline MobileNetV2 (pretrained on ImageNet) fine-tuned on UrbanSound8K.
Uses mel-spectrograms as input.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.mobilenet_audio_model import MobileNetAudio
from utils.urbansound_data_utils import load_urbansound_data, create_urbansound_dataloaders
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


def train_baseline_mobilenet(args):
    """Main training function"""
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load preprocessed data (mel-spectrograms)
    print("\nLoading preprocessed UrbanSound8K data...")
    data_dict = load_urbansound_data(args.data_dir)
    
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = args.batch_size if args.batch_size is not None else 32
    loaders = create_urbansound_dataloaders(
        data_dict, 
        batch_size=batch_size,
        num_workers=2
    )
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Pretrained MobileNetV2...")
    print("="*60)
    num_classes = data_dict.get('num_classes', 10)
    model = MobileNetAudio(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    print("Transfer Learning: ImageNet pretrained -> UrbanSound8K fine-tuning")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    lr = args.lr if args.lr is not None else 0.001  # Higher LR than transformer
    weight_decay = args.weight_decay if args.weight_decay is not None else 1e-5
    
    print(f"\nUsing LR={lr:.6f}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        factor=0.5
    )
    
    # Training configuration
    max_epochs = args.epochs if args.epochs is not None else 50
    patience = args.patience if args.patience is not None else 15
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'baseline_mobilenet_best.pt')
    
    # Training loop
    print(f"\nFine-tuning for up to {max_epochs} epochs...")
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
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    class_names = data_dict.get('class_names', None)
    cm_path = os.path.join(models_dir, 'baseline_mobilenet_confusion_matrix.png')
    cm = plot_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        label_names=class_names,
        save_path=cm_path
    )
    
    # Save results to JSON
    results = {
        'model': 'Baseline MobileNetV2 (Pretrained)',
        'dataset': 'UrbanSound8K',
        'num_classes': num_classes,
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'best_val_accuracy': float(best_val_acc),
        'num_parameters': model.count_parameters(),
        'epochs_trained': epoch + 1,
        'pretrained': True,
        'pretrained_source': 'ImageNet'
    }
    
    results_path = os.path.join(models_dir, 'baseline_mobilenet_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline MobileNetV2 on UrbanSound8K')
    parser.add_argument('--data_dir', type=str, default='data/urbansound_processed',
                        help='Directory with preprocessed data')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Maximum epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (default: 15)')
    
    args = parser.parse_args()
    train_baseline_mobilenet(args)
