"""
Guardian-Pulse Ghana: Training Pipeline (OFFICIAL INTEGRATION)
==============================================================
Training workflow using the Official Perforated AI Library.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import time
from datetime import datetime
import sys
import os

# --- OFFICIAL PERFORATED AI IMPORTS ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'PerforatedAI'))

try:
    from perforatedai import utils_perforatedai as UPA
    from perforatedai import globals_perforatedai as GPA
    print("✅ Perforated AI Library Loaded Successfully")
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    sys.exit(1)

from models import create_signal_geo_model, create_cyber_sec_model
from data_loader import get_signal_geo_loader, get_cyber_sec_loader

class GuardianTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        wandb.init(
            project=self.config['project_name'],
            name=self.config['run_name'],
            config=self.config,
            tags=['edge-ai', 'ghana', 'real-dendritic-optimization']
        )
        print(f"\n{'='*70}\nGuardian-Pulse System Initialized\n{'='*70}")

    def setup_model(self):
        print(f"Model Setup: {self.config['model_type']}")
        
        if self.config['model_type'] == 'signal-geo':
            self.model = create_signal_geo_model(num_classes=2)
        else:
            self.model = create_cyber_sec_model(num_classes=4)

        # --- CRITICAL FLAGS: Disable Interactive Breakpoints ---
        print("   - Configuring PAI Flags to skip interactive checks...")
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        
        print("\n⚡ Applying Perforated AI 'initialize_pai' transformation...")
        self.model = UPA.initialize_pai(
            self.model, 
            doing_pai=True,          
            save_name="Guardian_PAI", 
            making_graphs=True,     
            maximizing_score=True    
        )
        self.model.to(self.device)

        # --- OPTIMIZER SETUP ---
        print("   - Registering Optimizer Class (AdamW)...")
        GPA.pai_tracker.set_optimizer(optim.AdamW)

        print("   - Configuring Optimizer Arguments...")
        optim_args = {
            'params': self.model.parameters(), 
            'lr': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay']
        }
        
        self.optimizer = GPA.pai_tracker.setup_optimizer(
            self.model, 
            optim_args
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.setup_model()
        
        if self.config['model_type'] == 'signal-geo':
            train_loader = get_signal_geo_loader(batch_size=self.config['batch_size'])
        else:
            train_loader = get_cyber_sec_loader(batch_size=self.config['batch_size'])
            
        print(f"\n{'='*70}\nStarting Real Dendritic Training Loop\n{'='*70}")
        
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                     print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | Scanning...")

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            # --- CRITICAL FIX: Re-attach optimizer before validation call ---
            # This prevents the 'NoneType' crash if the library dropped the reference
            GPA.pai_tracker.member_vars['optimizer_instance'] = self.optimizer
            # --------------------------------------------------------------

            self.model, restructured, training_complete = GPA.pai_tracker.add_validation_score(accuracy, self.model)
            
            print(f"-"*50)
            print(f"Epoch {epoch+1} Complete | Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")
            print(f"-"*50)
            
            wandb.log({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})

        print("\n✓ Training Complete. Ready for Pull Request.")
        wandb.finish()

if __name__ == "__main__":
    config = {
        'project_name': 'guardian-pulse-ghana',
        'run_name': f'REAL-dendritic-{datetime.now().strftime("%H%M")}',
        'mission': 'Detecting IMSI catchers & Cyber Threats',
        'deployment_region': 'Ghana (Accra)',
        'model_type': 'signal-geo', 
        'batch_size': 32,
        'num_epochs': 2, 
        'learning_rate': 0.001,
        'weight_decay': 0.0001
    }
    GuardianTrainer(config).train()