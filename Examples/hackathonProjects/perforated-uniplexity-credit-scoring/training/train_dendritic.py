"""
Training script for dendritic model
"""
import sys
# Add the project root directory to sys.path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- PAI Library Setup ---
# Add the root directory to path to import perforatedai
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up 3 levels: training -> perforated-uniplexity... -> hackathonProjects -> Examples -> PerforatedAI (Root)
# Wait, let's verify directory depth.
# Script is in: C:\Users\user\Desktop\PerforatedAI\Examples\hackathonProjects\perforated-uniplexity-credit-scoring\training
# Root is:      C:\Users\user\Desktop\PerforatedAI
# Depth: training(1) -> scoring(2) -> hackathonProjects(3) -> Examples(4) -> Root
# So we need 4 levels up.
root_dir = os.path.abspath(os.path.join(current_dir, "../../../../")) 
sys.path.append(root_dir)

# Debug print to help identify path issues
print("Adding to path: " + root_dir)

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    print("Successfully imported PerforatedAI Library")
except ImportError as e:
    print("Failed to import PerforatedAI: " + str(e))
    print("Attempted path: " + root_dir)
    sys.exit(1)

from models.dendritic_model import DendriticModel

def train_dendritic():
    """Train the dendritic model"""
    
    print("Starting Dendritic Model Training...")
    
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
    
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = DendriticModel(input_dim=5)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- PAI Configuration ---
    GPA.pc.set_testing_dendrite_capacity(False) # Disable test mode for real training

    # --- PAI Initialization ---
    model = UPA.initialize_pai(model)
    model.to(device)

    # We use PAI tracker to manage the optimizer and scheduler
    # This allows it to reset learning rates when structure changes (dendrites added)
    optimizer_args = {'params': model.parameters(), 'lr': 0.001}
    scheduler_args = {'mode': 'max', 'patience': 5, 'factor': 0.5}
    
    GPA.pai_tracker.set_optimizer(optim.Adam)
    GPA.pai_tracker.set_scheduler(optim.lr_scheduler.ReduceLROnPlateau)
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimizer_args, scheduler_args)

    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting PAI-Dendritic Optimization Loop on {device}")
    
    # PAI requires a potentially infinite loop because it decides when to stop
    # after trying effectively all dendritic configurations.
    epoch = 0
    training_complete = False
    
    while not training_complete:
        epoch += 1
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss_sum += criterion(outputs, y_val).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        val_acc = correct / total
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # LOGGING (Optional: PAI does its own too)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # --- PAI VALIDATION PIVOT POINT ---
        # This is where the magic happens. The tracker decides if it should:
        # 1. Continue training current structure
        # 2. Add new dendrites (restructure)
        # 3. Stop because no improvement found
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        model.to(device) # Ensure model stays on device after potential restructure
        
        if training_complete:
            print("PAI Optimization Complete! Best model loaded.")
            break
            
        if restructured:
            print("PAI Restructure Triggered: Resetting Optimizer...")
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimizer_args, scheduler_args)

    # Save final optimized model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), 'models/dendritic_pai_optimized.pt')
    print("Saved optimized model to models/dendritic_pai_optimized.pt")
    
    # Save Metrics
    import json
    os.makedirs("results", exist_ok=True)
    
    # Calculate number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        "final_accuracy": float(val_acc),
        "final_loss": float(avg_val_loss),
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
