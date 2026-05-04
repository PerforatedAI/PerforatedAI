"""
W&B Sweep Runner for Hyperparameter Tuning
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

# Robust Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.dendritic_model import DendriticModel


def train_sweep():
    """Training function for W&B sweep"""
    
    # Initialize wandb
    wandb.init()
    config = wandb.config
    
    # Load Real Data
    try:
        data_path_features = os.path.join(project_root, "data", "processed", "features.pt")
        data_path_labels = os.path.join(project_root, "data", "processed", "labels.pt")
        
        features = torch.load(data_path_features)
        labels = torch.load(data_path_labels)
    except FileNotFoundError:
        print("❌ Error: Processed data not found. Please run 'python data/preprocess.py' first.")
        return

    # Split Data (80/20)
    train_size = int(0.8 * len(features))
    train_dataset = TensorDataset(features[:train_size], labels[:train_size])
    val_dataset = TensorDataset(features[train_size:], labels[train_size:])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Initialize model with sweep parameters
    model = DendriticModel(
        input_dim=5, # corrected from 6
        hidden_dim=config.hidden_dim,
        output_dim=1,
        num_dendrites=config.num_dendrites
    )
    
    criterion = nn.BCEWithLogitsLoss() # Better for binary classification
    
    # Select optimizer based on config
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    epochs = 20 # fixed for sweep
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_acc = correct / total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_acc
        })


def main():
    """Main function to start sweep"""
    
    # Initialize sweep
    # Note: 'sweep_config.yaml' must be in the same directory or passed correctly
    sweep_id = wandb.sweep(
        sweep=torch.load(os.path.join(current_dir, 'sweep_config.yaml')), # This is tricky with PyYAML, assume CLI usage usually
        project='credit-scoring-sweep'
    )
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train_sweep, count=20)


if __name__ == "__main__":
    train_sweep()  # For individual debug runs if called directly
