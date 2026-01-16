"""
Train baseline AST (Audio Spectrogram Transformer) model without dendrites on UrbanSound8K.
Uses pretrained AST from Huggingface.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from utils.urbansound_ast_utils import load_urbansound_ast_data, create_urbansound_ast_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        inputs = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs).logits
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
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * sum([p == l for p, l in zip(all_predictions, all_labels)]) / len(all_labels)
    avg_loss = running_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels
    }


def train_baseline_ast(args):
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
    
    # Load preprocessed metadata
    print("\nLoading UrbanSound8K metadata...")
    metadata_dict = load_urbansound_ast_data(args.data_dir)
    num_classes = metadata_dict['num_classes']
    
    # Initialize AST feature extractor
    print("\nInitializing AST Feature Extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    # Create dataloaders
    print("Creating dataloaders...")
    batch_size = args.batch_size if args.batch_size is not None else 8
    loaders = create_urbansound_ast_dataloaders(
        metadata_dict,
        feature_extractor,
        batch_size=batch_size,
        num_workers=args.num_workers,
        max_length=10.0  # 10 second max audio
    )
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Initialize pretrained AST model
    print("\n" + "="*60)
    print("Initializing Pretrained AST Model...")
    print("="*60)
    print("Loading AST pretrained on AudioSet...")
    
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print("Transfer Learning: Fine-tuning pretrained AST on UrbanSound8K")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    lr = args.lr if args.lr is not None else 5e-5  # Lower LR for fine-tuning
    weight_decay = args.weight_decay if args.weight_decay is not None else 1e-5
    
    print(f"\nUsing LR={lr:.6f} (fine-tuning rate)")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=3,
        factor=0.5
    )
    
    # Training configuration
    max_epochs = args.epochs if args.epochs is not None else 25
    patience = args.patience if args.patience is not None else 10
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'baseline_ast_best.pt')
    
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
    
    # Save results to JSON
    results = {
        'model': 'Baseline AST (Pretrained)',
        'dataset': 'UrbanSound8K',
        'num_classes': num_classes,
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'best_val_accuracy': float(best_val_acc),
        'num_parameters': total_params,
        'epochs_trained': epoch + 1,
        'pretrained': True,
        'pretrained_source': 'AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593)'
    }
    
    results_path = os.path.join(models_dir, 'baseline_ast_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline AST on UrbanSound8K')
    parser.add_argument('--data_dir', type=str, default='data/urbansound_ast_processed',
                        help='Directory with preprocessed metadata')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 5e-5 for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Maximum number of epochs (default: 25)')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    
    args = parser.parse_args()
    train_baseline_ast(args)
