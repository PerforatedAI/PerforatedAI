"""
Training script for Hybrid model (Baseline + Dendritic)
"""
import sys
# Add the project root directory to sys.path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.hybrid_model import HybridModel

def train_hybrid():
    """Train the hybrid model"""
    
    print("🚀 Starting Hybrid Model Training...")
    
    # Load data
    try:
        features = torch.load("data/processed/features.pt")
        labels = torch.load("data/processed/labels.pt")
    except FileNotFoundError:
        print("❌ Data not found! Run data/preprocess.py first.")
        return
        
    # Split
    n_train = int(len(features) * 0.8)
    train_features, val_features = features[:n_train], features[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    
    dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = HybridModel(input_dim=5)
    
    # Use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    # Use AdamW for better regularization, same as improved Dendritic
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    epochs = 40
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # History tracking
    train_losses = []
    val_accuracies = []
    
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
        
        # Step the scheduler
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_labels)
            
            preds = torch.sigmoid(val_outputs) > 0.5
            acc = (preds == val_labels).float().mean()
            val_accuracies.append(float(acc))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/hybrid_checkpoint.pt")
    
    # Save Metrics
    import json
    os.makedirs("results", exist_ok=True)
    
    # Calculate number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        "final_accuracy": float(acc),
        "final_loss": float(val_loss.item()),
        "train_loss_history": train_losses,
        "val_acc_history": val_accuracies,
        "num_parameters": num_params
    }
    with open("results/hybrid_metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"✅ Hybrid Training completed! Final Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_hybrid()
