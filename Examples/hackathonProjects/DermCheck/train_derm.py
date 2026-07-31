"""
DermCheck Training Script with MONAI and PerforatedAI
"""

import argparse
import yaml
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121, UNet
from monai.losses import DiceLoss
from tqdm import tqdm
from pathlib import Path

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from utils.pai_monai import setup_pai_monai, configure_tracker
from utils.data_loader import get_loaders

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(config, args):
    print("="*60)
    print("DermCheck Training: MONAI + PerforatedAI")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Models
    if args.task == 'classification' or args.task == 'both':
        print("Initializing Classification Model (DenseNet121)...")
        model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=config['data']['num_classes'])
    elif args.task == 'segmentation':
        print("Initializing Segmentation Model (UNet)...")
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
    
    # 2. Integrate PerforatedAI
    if config['pai']['enabled']:
        print("Integrating PerforatedAI Dendritic Optimization...")
        model = setup_pai_monai(model, config)
        model = model.to(device)
        optimizer, scheduler = configure_tracker(model, config)
    else:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        scheduler = None

    # 3. Data Loaders
    train_loader, val_loader = get_loaders(config)
    
    # 4. Loss functions
    class_loss_fn = nn.CrossEntropyLoss()
    seg_loss_fn = DiceLoss(sigmoid=True)
    
    # 5. Training Loop
    epochs = args.epochs if args.epochs else config['training']['epochs']
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            if args.task == 'classification':
                labels = batch['label'].to(device)
                loss = class_loss_fn(outputs, labels)
            elif args.task == 'segmentation':
                masks = batch['mask'].to(device)
                loss = seg_loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        val_score = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                outputs = model(images)
                # Simple score for PAI: accuracy or 1-Dice
                val_score += 1.0 # Placeholder: actual metric should be calculated
        
        val_score /= len(val_loader)
        print(f"Validation Score: {val_score:.4f}")
        
        # PAI Tracker Update
        if config['pai']['enabled']:
            model, restructured, complete = GPA.pai_tracker.add_validation_score(val_score, model)
            if restructured:
                print("Model Restructured: Dendrites added!")
                model = model.to(device)
                optimizer, scheduler = configure_tracker(model, config)
            if complete:
                print("PAI Training Complete!")
                break
                
    # Save Model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / f"{args.task}_best.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'segmentation', 'both'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()
    
    config = load_config()
    train(config, args)
