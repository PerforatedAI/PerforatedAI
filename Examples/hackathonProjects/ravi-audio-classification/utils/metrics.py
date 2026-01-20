"""
Metrics and evaluation utilities
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy"""
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def calculate_error_reduction(baseline_acc, dendrite_acc):
    """
    Calculate remaining error reduction percentage.
    
    Formula: (baseline_error - dendrite_error) / baseline_error * 100
    
    Args:
        baseline_acc: Baseline accuracy (e.g., 70.0 for 70%)
        dendrite_acc: Dendrite accuracy (e.g., 78.0 for 78%)
        
    Returns:
        Error reduction percentage
    """
    baseline_error = 100.0 - baseline_acc
    dendrite_error = 100.0 - dendrite_acc
    
    if baseline_error == 0:
        return 0.0
    
    error_reduction = (baseline_error - dendrite_error) / baseline_error * 100
    return error_reduction


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with loss, accuracy, predictions, and labels
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(y_true, y_pred, label_names=None, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: List of class names
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=label_names if label_names else 'auto',
                yticklabels=label_names if label_names else 'auto')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return cm


def get_classification_report(y_true, y_pred, label_names=None):
    """
    Generate classification report with per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: List of class names
        
    Returns:
        Classification report as string
    """
    return classification_report(
        y_true, y_pred, 
        target_names=label_names if label_names else None,
        digits=3
    )
