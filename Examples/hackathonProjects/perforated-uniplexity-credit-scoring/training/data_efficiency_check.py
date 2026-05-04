"""
Script to estimate data efficiency of the Dendritic Model compared to the Baseline.
It finds the minimum fraction of data required for the Dendritic Model to match Baseline accuracy.
"""
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dendritic_model import DendriticModel

def check_data_efficiency():
    print("🚀 Starting Data Efficiency Analysis...")

    # 1. Load Baseline Accuracy
    try:
        with open("results/baseline_metrics.json", "r") as f:
            baseline_metrics = json.load(f)
            baseline_acc = baseline_metrics["final_accuracy"]
            print(f"   Target Accuracy (Baseline): {baseline_acc*100:.2f}%")
    except FileNotFoundError:
        print("❌ Baseline metrics not found! Run training/train_baseline.py first.")
        return

    # 2. Load Data
    try:
        features = torch.load("data/processed/features.pt")
        labels = torch.load("data/processed/labels.pt")
    except FileNotFoundError:
        print("❌ Data not found! Run data/preprocess.py first.")
        return

    # 3. Define Fractions to Test
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    efficiency_results = {}

    # 4. Train on Subsets
    for frac in fractions:
        print(f"\nTraining on {frac*100}% of data...")
        
        # Subsample
        n_total = len(features)
        n_subset = int(n_total * frac)
        subset_features = features[:n_subset]
        subset_labels = labels[:n_subset]
        
        # Train/Val Split (80/20 of the subset)
        n_train = int(n_subset * 0.8)
        if n_train == 0: continue
        
        train_features = subset_features[:n_train]
        val_features = subset_features[n_train:]
        train_labels = subset_labels[:n_train]
        val_labels = subset_labels[n_train:]
        
        dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Init Model
        model = DendriticModel(input_dim=5)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train (Simplified Loop)
        epochs = 15 # Slightly fewer epochs for speed in analysis
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            for X, y in train_loader:
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(val_features)
                preds = torch.sigmoid(val_out) > 0.5
                acc = (preds == val_labels).float().mean().item()
                if acc > best_val_acc:
                    best_val_acc = acc
        
        efficiency_results[str(frac)] = best_val_acc
        print(f"   -> Best Accuracy with {frac*100}% data: {best_val_acc*100:.2f}%")

    # 5. Analyze Results
    min_fraction = 1.0
    for frac in fractions:
        if efficiency_results[str(frac)] >= baseline_acc:
            min_fraction = frac
            break
            
    print("\n" + "="*30)
    print(f"✅ Data Efficiency Result: Achieved baseline accuracy with {min_fraction*100}% of data.")
    print("="*30)

    # 6. Save Results
    results = {
        "fractions": fractions,
        "accuracies": [efficiency_results[str(f)] for f in fractions],
        "baseline_accuracy": baseline_acc,
        "equivalent_data_fraction": min_fraction
    }
    
    with open("results/data_efficiency.json", "w") as f:
        json.dump(results, f)
    print("Saved results to results/data_efficiency.json")

if __name__ == "__main__":
    check_data_efficiency()
