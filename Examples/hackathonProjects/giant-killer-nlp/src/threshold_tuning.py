"""
Threshold Tuning for Toxicity Classification

This script finds the optimal classification threshold to balance
precision and recall for toxic comment detection.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_jigsaw_dataset, create_dataloaders, get_tokenizer
from models import create_bert_tiny_model, wrap_with_dendrites


def get_predictions_and_labels(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions and true labels.
    
    Args:
        model: Trained model
        dataloader: DataLoader to evaluate
        device: Device to run on
        
    Returns:
        Tuple of (probabilities, predictions, labels)
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Get probabilities for toxic class (class 1)
            probs = F.softmax(logits, dim=1)[:, 1]
            
            # Get predictions with default threshold 0.5
            preds = (probs >= 0.5).long()
            
            # Collect results
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def evaluate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate metrics at a specific threshold.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    preds = (probs >= threshold).astype(int)
    
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Count predictions
    num_positive = np.sum(preds)
    num_negative = len(preds) - num_positive
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_positive': int(num_positive),
        'num_negative': int(num_negative)
    }


def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = None
) -> Tuple[float, Dict]:
    """
    Find optimal threshold by sweeping and evaluating metrics.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        thresholds: List of thresholds to evaluate
        
    Returns:
        Tuple of (best_threshold, all_results)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05).tolist()
    
    print(f"\nEvaluating {len(thresholds)} thresholds...")
    results = []
    best_f1 = 0
    best_threshold = 0.5
    best_result = None
    
    for threshold in thresholds:
        result = evaluate_threshold(probs, labels, threshold)
        results.append(result)
        
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_threshold = threshold
            best_result = result
    
    return best_threshold, results, best_result


def plot_threshold_analysis(
    results: List[Dict],
    output_path: str = "threshold_analysis.png"
):
    """
    Plot precision, recall, and F1 across thresholds.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save plot
    """
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, 'g-', label='F1 Score', linewidth=2)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds), max(thresholds))
    plt.ylim(0, 1)
    
    # Find best F1 threshold
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    plt.axvline(x=best_threshold, color='purple', linestyle='--', 
                label=f'Best F1: {best_threshold:.2f}')
    
    # Plot prediction counts
    plt.subplot(1, 2, 2)
    positive_counts = [r['num_positive'] for r in results]
    negative_counts = [r['num_negative'] for r in results]
    
    plt.plot(thresholds, positive_counts, 'b-', label='Predicted Toxic', linewidth=2)
    plt.plot(thresholds, negative_counts, 'r-', label='Predicted Non-Toxic', linewidth=2)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Counts vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds), max(thresholds))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nThreshold analysis plot saved to: {output_path}")
    plt.close()


def plot_precision_recall_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str = "precision_recall_curve.png"
):
    """
    Plot precision-recall curve.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        output_path: Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add F1 contours
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.3, linestyle='--')
        plt.annotate(f'F1={f_score:.1f}', xy=(0.9, y[45] + 0.02), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Tune classification threshold")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset size for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="threshold_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("THRESHOLD TUNING")
    print("Finding optimal classification threshold")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Model: {args.model_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    # Load dataset
    print("\n2. Loading dataset...")
    _, _, val_texts, val_labels, test_texts, test_labels = \
        load_jigsaw_dataset(sample_size=args.sample_size)
    
    # Create dataloaders (use validation set for threshold tuning)
    print("\n3. Creating dataloaders...")
    _, val_loader, test_loader, _ = create_dataloaders(
        train_texts=val_texts,  # Dummy train data
        train_labels=val_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        tokenizer=tokenizer,
        batch_size=32,
        max_length=128,
        num_workers=0
    )
    
    # Load model
    print("\n4. Loading model...")
    
    # Check if this is a dendritic model
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Check for dendritic layers
    has_dendrites = any('dendrite_module' in key or 'main_module' in key for key in state_dict.keys())
    
    model = create_bert_tiny_model(num_labels=2)
    
    if has_dendrites:
        print("   Detected dendritic model, wrapping with dendrites...")
        model = wrap_with_dendrites(model)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print("   Model loaded successfully")
    
    # Get predictions on validation set
    print("\n5. Getting predictions on validation set...")
    val_probs, val_preds, val_labels_array = get_predictions_and_labels(
        model, val_loader, device
    )
    
    # Calculate baseline metrics (threshold = 0.5)
    print("\n6. Baseline metrics (threshold = 0.5):")
    baseline_metrics = evaluate_threshold(val_probs, val_labels_array, 0.5)
    print(f"   Precision: {baseline_metrics['precision']:.4f}")
    print(f"   Recall:    {baseline_metrics['recall']:.4f}")
    print(f"   F1 Score:  {baseline_metrics['f1']:.4f}")
    print(f"   Predictions: {baseline_metrics['num_positive']} toxic, "
          f"{baseline_metrics['num_negative']} non-toxic")
    
    # Find optimal threshold
    print("\n7. Finding optimal threshold...")
    best_threshold, results, best_result = find_optimal_threshold(
        val_probs, val_labels_array
    )
    
    print(f"\n   Best threshold: {best_threshold:.3f}")
    print(f"   Precision: {best_result['precision']:.4f} "
          f"({(best_result['precision'] - baseline_metrics['precision']) / baseline_metrics['precision'] * 100:+.1f}%)")
    print(f"   Recall:    {best_result['recall']:.4f} "
          f"({(best_result['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100:+.1f}%)")
    print(f"   F1 Score:  {best_result['f1']:.4f} "
          f"({(best_result['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100:+.1f}%)")
    
    # Evaluate on test set with best threshold
    print("\n8. Evaluating on test set with optimal threshold...")
    test_probs, test_preds, test_labels_array = get_predictions_and_labels(
        model, test_loader, device
    )
    
    test_baseline = evaluate_threshold(test_probs, test_labels_array, 0.5)
    test_optimal = evaluate_threshold(test_probs, test_labels_array, best_threshold)
    
    print(f"\n   Test Set - Baseline (0.5):")
    print(f"     Precision: {test_baseline['precision']:.4f}")
    print(f"     Recall:    {test_baseline['recall']:.4f}")
    print(f"     F1 Score:  {test_baseline['f1']:.4f}")
    
    print(f"\n   Test Set - Optimal ({best_threshold:.3f}):")
    print(f"     Precision: {test_optimal['precision']:.4f} "
          f"({(test_optimal['precision'] - test_baseline['precision']) / test_baseline['precision'] * 100:+.1f}%)")
    print(f"     Recall:    {test_optimal['recall']:.4f} "
          f"({(test_optimal['recall'] - test_baseline['recall']) / test_baseline['recall'] * 100:+.1f}%)")
    print(f"     F1 Score:  {test_optimal['f1']:.4f} "
          f"({(test_optimal['f1'] - test_baseline['f1']) / test_baseline['f1'] * 100:+.1f}%)")
    
    # Save results
    print("\n9. Saving results...")
    results_dict = {
        'best_threshold': float(best_threshold),
        'validation': {
            'baseline': baseline_metrics,
            'optimal': best_result,
            'all_thresholds': results
        },
        'test': {
            'baseline': test_baseline,
            'optimal': test_optimal
        }
    }
    
    results_path = output_dir / "threshold_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"   Results saved to: {results_path}")
    
    # Plot analysis
    print("\n10. Creating visualizations...")
    plot_threshold_analysis(results, str(output_dir / "threshold_analysis.png"))
    plot_precision_recall_curve(val_probs, val_labels_array, 
                                str(output_dir / "precision_recall_curve.png"))
    
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 60)
    print(f"\nRecommendation: Use threshold = {best_threshold:.3f}")
    print(f"Expected F1 improvement: {(best_result['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100:.1f}%")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
