"""
Training script using PyTorch Lightning
🔥 Streamlined training with Lightning
"""
"""
Training script using PyTorch Lightning
🔥 Streamlined training with Lightning
"""
import sys
import os
import json
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.lightning_module import CreditScoringLightningModule


def train_with_lightning():
    """Train model using PyTorch Lightning"""
    print("⚡ Starting Lightning Training (Hybrid Model)...")
    
    # Load data
    try:
        features = torch.load("data/processed/features.pt")
        labels = torch.load("data/processed/labels.pt")
    except FileNotFoundError:
        # Try relative path if run from root
        try:
            features = torch.load("data/processed/features.pt")
            labels = torch.load("data/processed/labels.pt")
        except FileNotFoundError:
            print("❌ Data not found! Run data/preprocess.py first.")
            return

    # Split
    n_train = int(len(features) * 0.8)
    train_dataset = TensorDataset(features[:n_train], labels[:n_train])
    val_dataset = TensorDataset(features[n_train:], labels[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize Lightning module with Hybrid model
    model = CreditScoringLightningModule(model_type='hybrid', learning_rate=1e-3)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='hybrid-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    
    # Setup logger (Offline for simplicity unless user has wandb)
    # wandb_logger = WandbLogger(project='credit-scoring', name='lightning-hybrid', offline=True)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=3, # Reduced for speed
        callbacks=[checkpoint_callback],
        logger=False, # Disable wandb for now to avoid login prompt issues
        accelerator='auto',
        devices=1,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Get final validation accuracy
    val_results = trainer.validate(model, val_loader, verbose=False)
    val_acc = val_results[0]['val_acc']
    print(f"✅ Lightning Training completed! Final Accuracy: {val_acc:.4f}")
    
    # Save Metrics for Report
    metrics = {
        "final_accuracy": float(val_acc),
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/lightning_hybrid_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    print("Saved metrics to results/lightning_hybrid_metrics.json")



if __name__ == "__main__":
    train_with_lightning()
