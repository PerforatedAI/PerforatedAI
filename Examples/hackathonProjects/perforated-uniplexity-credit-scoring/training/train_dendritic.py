"""
Training script for dendritic model
"""
import sys
# Add the project root directory to sys.path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.dendritic_model import DendriticModel

def train_dendritic():
    """Train the dendritic model"""
    
    print("🚀 Starting Dendritic Model Training...")
    
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
    model = DendriticModel(input_dim=5)
    
    # Use BCEWithLogitsLoss
    # Use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    # Use AdamW for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    epochs = 40
    
    # Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # History tracking
    train_losses = []
    val_accuracies = []
    epochs_range = range(1, epochs + 1)
    
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
    torch.save(model.state_dict(), "models/dendritic_checkpoint.pt")
    
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
    with open("results/dendritic_metrics.json", "w") as f:
        json.dump(metrics, f)

    # --- Generate PAI.png (Raw Results Graph) ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.plot(epochs_range, train_losses, 'g-', label='Train Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        # Plot Accuracy (Right Axis)
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, [v * 100 for v in val_accuracies], 'b-', label='Validation Accuracy')
        ax2.set_ylabel('Accuracy (%)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title("Perforated AI Training Dynamics (Actual Run)")
        plt.savefig('PAI.png') # Submit to root for hackathon
        plt.close()
        print("✅ Generated mandatory result graph: PAI.png")
    except ImportError:
        print("⚠️ Could not generate PAI.png (matplotlib missing)")

    print(f"✅ Training completed! Final Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_dendritic()
