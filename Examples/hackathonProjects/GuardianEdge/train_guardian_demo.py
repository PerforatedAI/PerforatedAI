"""
GuardianEdge PAI Demo Training Script
Custom training loop to enable PerforatedAI graph generation.
Uses a simplified 'ThreatNet' for demonstration of dendritic growth.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Simplified Threat Detection Model for Demo Graph Generation
class ThreatNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ThreatNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def configure_tracker(model, config):
    GPA.pai_tracker.set_optimizer(optim.Adadelta)
    GPA.pai_tracker.set_scheduler(optim.lr_scheduler.StepLR)
    
    optim_args = {
        'params': model.parameters(),
        'lr': 1.0,
        'weight_decay': 0.0
    }
    sched_args = {'step_size': 1, 'gamma': 0.7}
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optim_args, sched_args
    )
    return optimizer, scheduler

def train_demo():
    print("="*60)
    print("GuardianEdge PAI Demo Training (Synthetic Threat Data)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Setup Data (Synthetic)
    print("Generating synthetic threat scenarios...")
    # Overfit a small dataset to ensure convergence and visible graphs
    batch_size = 64
    x_train = torch.randn(1000, 3, 28, 28) # Simulated feature maps or low-res inputs
    y_train = torch.randint(0, 2, (1000,)) # Binary threat/no-threat
    
    x_val = torch.randn(200, 3, 28, 28)
    y_val = torch.randint(0, 2, (200,))
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # 2. Setup Model
    print("Initializing ThreatNet...")
    model = ThreatNet()
    
    # 3. PAI Configuration (Organic Graph Settings)
    print("Integrating PerforatedAI...")
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    
    # CRITICAL SETTINGS FOR WAVEFORM GRAPH
    GPA.pc.set_testing_dendrite_capacity(False)
    # Fast switching for 50 epoch demo
    GPA.pc.set_n_epochs_to_switch(3)
    
    # Initialize PAI
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name="PAI_GuardianEdge", # This folder will contain the graphs
        making_graphs=True,
        maximizing_score=True
    )
    model = model.to(device)

    # 4. Optimizer
    optimizer, scheduler = configure_tracker(model, {})

    # 5. Training Loop
    epochs = 50
    print(f"Starting {epochs} epoch demo run...")
    
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_acc = 100. * correct / len(train_loader.dataset)
        GPA.pai_tracker.add_extra_score(train_acc, "train")

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        
        print(f"  Val Acc: {val_acc:.2f}% | Train Acc: {train_acc:.2f}%")

        # PAI Tracker Update
        model, restructured, complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        
        if restructured:
            print("\n" + "="*60)
            print("🌿 THREATNET RESTRUCTURED - DENDRITES ADDED")
            print("="*60)
            model = model.to(device)
            optimizer, scheduler = configure_tracker(model, {})
            # Ensure next epoch starts with correct optimizer state
            
        if complete:
            print("\n✅ Optimization Complete!")
            break
            
    # Save final model state
    torch.save(model.state_dict(), "models/best_model_pai.pt")
    print(f"\nExample graph saved to: PAI_GuardianEdge/PAI_GuardianEdge.png")

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    train_demo()
