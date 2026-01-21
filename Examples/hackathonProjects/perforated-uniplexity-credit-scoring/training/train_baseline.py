"""
Training script for baseline model
"""
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.baseline_model import BaselineModel

def train_baseline():
    """Train the baseline model"""
    
    print("🚀 Starting Baseline Model Training...")
    
    # Load data
    try:
        features = torch.load("data/processed/features.pt")
        labels = torch.load("data/processed/labels.pt")
    except FileNotFoundError:
        print("❌ Data not found! Run data/preprocess.py first.")
        return

    # Simple split (80/20) - in real world use fixed split
    n_train = int(len(features) * 0.8)
    train_features, val_features = features[:n_train], features[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    
    dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = BaselineModel(input_dim=5) # Ensure input_dim is 5
    
    # Use BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 20 # Reduced for demo speed
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_labels)
            
            # Calculate accuracy
            preds = torch.sigmoid(val_outputs) > 0.5
            acc = (preds == val_labels).float().mean()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
            
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_checkpoint.pt")
    
    # Save Metrics for Comparison
    import json
    os.makedirs("results", exist_ok=True)
    
    # Calculate number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        "final_accuracy": float(acc),
        "final_loss": float(val_loss.item()),
        "num_parameters": num_params
    }
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(metrics, f)
        
    print(f"✅ Training completed! Final Accuracy: {acc:.4f}")
    print("Files saved:")
    print(" - Model: models/baseline_checkpoint.pt")
    print(" - Metrics: results/baseline_metrics.json")


if __name__ == "__main__":
    train_baseline()
