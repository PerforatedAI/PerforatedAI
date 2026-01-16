"""
Train AST (Audio Spectrogram Transformer) WITH dendrites using PerforatedAI on UrbanSound8K.
Uses pretrained AST from Huggingface + dendritic optimization.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from utils.urbansound_ast_utils import load_urbansound_ast_data, create_urbansound_ast_dataloaders
from utils.metrics import calculate_error_reduction


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return accuracy for PAI tracking"""
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


def validate(model, dataloader, criterion, device, optimizer, args):
    """
    Validate model and handle PAI restructuring.
    Returns updated model, optimizer, training_complete flag, and metrics.
    """
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
            'mode': 'max',
            'patience': 3,
            'factor': 0.5
        }
        optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    return model, optimizer, training_complete, restructured, val_loss, val_acc


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


def train_perforatedai_ast(args):
    """Main training function with PerforatedAI dendrites"""
    
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
        max_length=10.0
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
    
    max_dendrites = args.max_dendrites
    
    # Set PAI global parameters
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)
    
    GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
    GPA.pc.set_max_dendrites(max_dendrites)
    GPA.pc.set_pai_forward_function(torch.sigmoid)
    GPA.pc.set_candidate_weight_initialization_multiplier(0.01)
    
    print(f"Max dendrites: {max_dendrites}")
    
    # ========================================================================
    # Initialize Model with PAI
    # ========================================================================
    print("\n" + "="*60)
    print("Initializing Pretrained AST with PerforatedAI...")
    print("="*60)
    
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Convert model to PAI model
    print("Converting AST to PAI model...")
    model = UPA.initialize_pai(model, save_name="PAI_AST_UrbanSound")
    
    model = model.to(device)
    print(f"Initial parameters: {UPA.count_params(model):,}")
    print("Strategy: Transfer Learning + Dendritic Optimization")
    
    # ========================================================================
    # Setup Optimizer via PAI Tracker
    # ========================================================================
    criterion = nn.CrossEntropyLoss()
    args.lr = args.lr if args.lr is not None else 5e-5
    args.weight_decay = args.weight_decay if args.weight_decay is not None else 1e-5
    
    print(f"\nUsing LR={args.lr:.6f} (fine-tuning rate)")
    
    # Set optimizer and scheduler types in PAI tracker
    GPA.pai_tracker.set_optimizer(optim.AdamW)
    GPA.pai_tracker.set_scheduler(optim.lr_scheduler.ReduceLROnPlateau)
    
    # Setup optimizer through PAI tracker
    optimArgs = {
        'params': model.parameters(),
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    schedArgs = {
        'mode': 'max',
        'patience': 3,
        'factor': 0.5
    }
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Training configuration
    max_epochs = args.epochs if args.epochs is not None else 25
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'pai_ast_best.pt')
    
    # ====================================================================
    # Training Loop (PAI controlled)
    # ====================================================================
    print(f"\nTraining with PerforatedAI (max {max_epochs} epochs)...")
    
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
        
        # Evaluate on test set (for tracking only)
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in loaders['test']:
                inputs = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(inputs).logits
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * test_correct / test_total
        
        # Track test score BEFORE validation
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
    model.to(device)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = evaluate_model(model, loaders['test'], criterion, device)
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Final Parameter Count: {UPA.count_params(model):,}")
    print(f"Total Dendrites Added: {GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)}")
    
    # ====================================================================
    # Compare with Baseline
    # ====================================================================
    baseline_results_path = os.path.join(models_dir, 'baseline_ast_results.json')
    if os.path.exists(baseline_results_path):
        print("\n" + "="*60)
        print("Comparison with Baseline AST")
        print("="*60)
        
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
        
        baseline_acc = baseline_results['test_accuracy']
        pai_acc = test_results['accuracy']
        
        improvement = pai_acc - baseline_acc
        error_reduction = calculate_error_reduction(baseline_acc, pai_acc)
        
        print(f"\nBaseline AST Accuracy: {baseline_acc:.2f}%")
        print(f"AST + PAI Accuracy:    {pai_acc:.2f}%")
        print(f"Improvement:           {improvement:+.2f}%")
        print(f"Error Reduction:       {error_reduction:.2f}%")
    
    # Save results
    results = {
        'model': 'PAI_AST_UrbanSound',
        'dataset': 'UrbanSound8K',
        'num_classes': num_classes,
        'test_accuracy': float(test_results['accuracy']),
        'test_loss': float(test_results['loss']),
        'best_val_accuracy': float(best_val_acc),
        'num_parameters': UPA.count_params(model),
        'epochs_trained': epoch + 1,
        'dendrites_added': GPA.pai_tracker.member_vars.get('num_dendrites_added', 0),
        'pretrained': True,
        'pretrained_source': 'AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593)'
    }
    
    results_path = os.path.join(models_dir, 'pai_ast_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Best model saved to: {best_model_path}")
    print("PAI output graphs saved to: PAI_AST_UrbanSound/PAI_AST_UrbanSound.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AST with PerforatedAI on UrbanSound8K')
    parser.add_argument('--data_dir', type=str, default='data/urbansound_ast_processed',
                        help='Directory with preprocessed metadata')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Maximum epochs (default: 25)')
    parser.add_argument('--max_dendrites', type=int, default=5,
                        help='Max dendrites (default: 5)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Data loading workers (default: 2)')
    
    args = parser.parse_args()
    train_perforatedai_ast(args)
