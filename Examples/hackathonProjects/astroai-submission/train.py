"""
AstroAI Training Script with Perforated AI Dendritic Optimization
This script trains the TransitDetector model with and without dendritic optimization
to demonstrate the benefits of Perforated AI's technology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Perforated AI imports - conditional to allow baseline training
try:
    import perforatedai as pai
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False
    pai = None

from model import TransitDetector, TransitDetectorCNN
from simulator import simulate_light_curve, generate_dataset


def train_baseline(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train baseline model without Perforated AI."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accuracies, best_val_acc


def train_with_pai(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', save_name='PAI_AstroAI'):
    """Train model with Perforated AI dendritic optimization."""
    if not PAI_AVAILABLE:
        print("Perforated AI not available. Please install with: pip install perforatedai")
        return None, [], [], [], 0
    
    model = model.to(device)
    
    # Initialize Perforated AI
    model = pai.initialize_pai(
        model,
        doing_pai=True,
        save_name=save_name,
        making_graphs=True,
        maximizing_score=True,  # We want to maximize accuracy
    )
    
    # Setup optimizer with PAI tracker
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    pai.pai_tracker.set_optimizer(optimizer, scheduler)
    
    criterion = nn.BCELoss()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    epoch = 0
    training_complete = False
    
    while not training_complete and epoch < epochs:
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Optional: track training score
        pai.pai_tracker.add_train_score(1 - train_loss)  # Convert loss to score
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # PAI validation tracking - this is required!
        model, optimizer, training_complete = pai.pai_tracker.add_validation_score(
            val_acc, model, optimizer
        )
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        epoch += 1
    
    print(f"Training completed after {epoch} epochs")
    return model, train_losses, val_losses, val_accuracies, best_val_acc


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = (outputs > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, predictions, targets


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_comparison(baseline_results, pai_results, save_path='results'):
    """Plot comparison between baseline and PAI results."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training Loss
    axes[0].plot(baseline_results['train_losses'], label='Baseline', alpha=0.7)
    axes[0].plot(pai_results['train_losses'], label='PAI', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[1].plot(baseline_results['val_losses'], label='Baseline', alpha=0.7)
    axes[1].plot(pai_results['val_losses'], label='PAI', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Validation Accuracy
    axes[2].plot(baseline_results['val_accuracies'], label='Baseline', alpha=0.7)
    axes[2].plot(pai_results['val_accuracies'], label='PAI', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Validation Accuracy')
    axes[2].set_title('Validation Accuracy Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_comparison.png'), dpi=150)
    plt.close()
    
    # Results summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = ['Baseline', 'PAI (Dendritic)']
    accuracies = [baseline_results['best_acc'], pai_results['best_acc']]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(x, accuracies, color=colors)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('AstroAI: Baseline vs Perforated AI Dendritic Optimization')
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.2%}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_comparison.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train AstroAI with Perforated AI')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--baseline_only', action='store_true', help='Train only baseline model')
    parser.add_argument('--pai_only', action='store_true', help='Train only PAI model')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AstroAI: Exoplanet Transit Detection with Perforated AI")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: {args.model.upper()}")
    print(f"Samples: {args.samples}")
    print()
    
    # Generate dataset
    print("Generating synthetic light curve dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = generate_dataset(
        n_samples=args.samples,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()
    
    results = {}
    
    # Train baseline model
    if not args.pai_only:
        print("-" * 40)
        print("Training Baseline Model...")
        print("-" * 40)
        
        if args.model == 'cnn':
            baseline_model = TransitDetectorCNN()
        else:
            baseline_model = TransitDetector()
        
        baseline_params = count_parameters(baseline_model)
        print(f"Baseline parameters: {baseline_params:,}")
        
        baseline_model, train_losses, val_losses, val_accs, best_val_acc = train_baseline(
            baseline_model, train_loader, val_loader, 
            epochs=args.epochs, lr=args.lr, device=args.device
        )
        
        test_acc, _, _ = evaluate_model(baseline_model, test_loader, args.device)
        print(f"Baseline Test Accuracy: {test_acc:.4f}")
        
        results['baseline'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'best_acc': test_acc,
            'params': baseline_params
        }
        
        # Save baseline model
        torch.save(baseline_model.state_dict(), 'results/baseline_model.pth')
    
    # Train with Perforated AI
    if not args.baseline_only:
        print()
        print("-" * 40)
        print("Training with Perforated AI Dendritic Optimization...")
        print("-" * 40)
        
        if not PAI_AVAILABLE:
            print("⚠ Perforated AI not available. Skipping PAI training.")
            print("  Install with: pip install perforatedai")
        else:
            if args.model == 'cnn':
                pai_model = TransitDetectorCNN()
            else:
                pai_model = TransitDetector()
            
            pai_params_before = count_parameters(pai_model)
            print(f"PAI parameters (before): {pai_params_before:,}")
            
            pai_model, train_losses, val_losses, val_accs, best_val_acc = train_with_pai(
                pai_model, train_loader, val_loader,
                epochs=args.epochs * 2,  # PAI may need more epochs
                lr=args.lr, device=args.device
            )
            
            if pai_model is not None:
                pai_params_after = count_parameters(pai_model)
                print(f"PAI parameters (after): {pai_params_after:,}")
                
                test_acc, _, _ = evaluate_model(pai_model, test_loader, args.device)
                print(f"PAI Test Accuracy: {test_acc:.4f}")
                
                results['pai'] = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accs,
                    'best_acc': test_acc,
                    'params_before': pai_params_before,
                    'params_after': pai_params_after
                }
                
                # Save PAI model
                torch.save(pai_model.state_dict(), 'results/pai_model.pth')
    
    # Generate comparison plots
    if 'baseline' in results and 'pai' in results:
        print()
        print("-" * 40)
        print("Results Summary")
        print("-" * 40)
        print(f"Baseline Accuracy: {results['baseline']['best_acc']:.4f}")
        print(f"PAI Accuracy:      {results['pai']['best_acc']:.4f}")
        improvement = results['pai']['best_acc'] - results['baseline']['best_acc']
        print(f"Improvement:       {improvement:+.4f} ({improvement/results['baseline']['best_acc']*100:+.2f}%)")
        print()
        print(f"Baseline Parameters: {results['baseline']['params']:,}")
        print(f"PAI Parameters:      {results['pai']['params_after']:,}")
        
        plot_comparison(results['baseline'], results['pai'])
        print()
        print("Results saved to 'results/' directory")
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
