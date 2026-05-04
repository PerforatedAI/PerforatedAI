"""
Model evaluation script
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.baseline_model import BaselineModel
from models.dendritic_model import DendriticModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(model, test_loader):
    """Evaluate a model on test data"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            predictions.extend(outputs.numpy().flatten())
            actuals.extend(batch_labels.numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    
    # Convert logits to probabilities and classes
    probs = 1 / (1 + np.exp(-predictions))
    preds_class = (probs > 0.5).astype(int)
    
    accuracy = (preds_class == actuals).mean()
    
    return {
        'MSE': mse,
        'MAE': mae,
        'Accuracy': accuracy
    }


def main():
    """Main evaluation function"""
    
    print("Loading data...")
    # Load test data 
    try:
        features = torch.load("../data/processed/features.pt")
        labels = torch.load("../data/processed/labels.pt")
        # Use last 20% as test set (same split logic as training)
        n_split = int(len(features) * 0.8)
        test_features = features[n_split:]
        test_labels = labels[n_split:]
        
    except FileNotFoundError:
        print("Data not found. Generating random test data...")
        test_features = torch.randn(20, 5)
        test_labels = torch.randint(0, 2, (20, 1)).float()
    
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate baseline model
    print("Evaluating Baseline Model...")
    baseline_model = BaselineModel(input_dim=5)
    try:
        baseline_model.load_state_dict(torch.load('../models/baseline_checkpoint.pt'))
        baseline_metrics = evaluate_model(baseline_model, test_loader)
        
        print("\nBaseline Model Metrics:")
        for metric, value in baseline_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except:
        print("Could not load baseline checkpoint")
        baseline_metrics = {}
    
    # Evaluate dendritic model
    print("\nEvaluating Dendritic Model...")
    dendritic_model = DendriticModel(input_dim=5)
    try:
        dendritic_model.load_state_dict(torch.load('../models/dendritic_checkpoint.pt'))
        dendritic_metrics = evaluate_model(dendritic_model, test_loader)
        
        print("\nDendritic Model Metrics:")
        for metric, value in dendritic_metrics.items():
            print(f"  {metric}: {value:.4f}")
    except:
        print("Could not load dendritic checkpoint")
        dendritic_metrics = {}
    
    # Compare models
    print("\n" + "="*50)
    print("Model Comparison:")
    print("="*50)
    for metric in baseline_metrics.keys():
        improvement = ((baseline_metrics[metric] - dendritic_metrics[metric]) / 
                      baseline_metrics[metric] * 100)
        print(f"{metric}: {improvement:+.2f}% improvement")


if __name__ == "__main__":
    main()
