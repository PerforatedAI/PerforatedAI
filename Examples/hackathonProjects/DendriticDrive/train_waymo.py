"""
DendriticDrive - Main Training Script
3D Object Detection with PerforatedAI Dendritic Optimization
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from utils.pai_pcdet import get_3d_model, setup_pai_pcdet, configure_tracker
from utils.data_loader_waymo import get_loaders


def load_config(path="config.yaml"):
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_detection_loss(predictions, labels, config):
    """
    Compute detection loss (simplified for demo)
    
    Args:
        predictions: Model predictions [B, num_classes]
        labels: Ground truth labels (list of label tensors)
        config: Configuration dict
    
    Returns:
        loss: Total loss
    """
    # For demo: use simple classification loss
    # In production, this would be Focal Loss + Box Regression Loss + Direction Loss
    
    # Extract first label from each sample (simplified)
    batch_labels = torch.stack([label[0] if len(label) > 0 else torch.tensor(0) 
                                for label in labels]).to(predictions.device)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, batch_labels)
    
    return loss


def compute_map(predictions, labels):
    """
    Compute mean Average Precision (simplified for demo)
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
    
    Returns:
        mAP: Mean Average Precision score
    """
    # Simplified: use accuracy as a proxy for mAP
    batch_labels = torch.stack([label[0] if len(label) > 0 else torch.tensor(0) 
                                for label in labels]).to(predictions.device)
    
    pred_classes = torch.argmax(predictions, dim=1)
    accuracy = (pred_classes == batch_labels).float().mean()
    
    return accuracy.item()


def train(config, args):
    """Main training function"""
    print("=" * 80)
    print("🧠 DendriticDrive - 3D Object Detection with Dendritic Optimization")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'DEMO (Synthetic Data)' if args.demo else 'REAL (Waymo Dataset)'}")
    print()
    
    # Override config with demo mode if specified
    if args.demo:
        config['demo']['enabled'] = True
    
    # 1. Data loaders (need this for total_steps)
    train_loader, val_loader = get_loaders(config, demo=args.demo)
    epochs = args.epochs if args.epochs else config['training']['epochs']
    
    # 2. Configure OneCycleLR params if used
    if config['optimizer']['scheduler'] == 'OneCycleLR':
        steps_per_epoch = len(train_loader)
        config['optimizer']['scheduler_kwargs']['total_steps'] = epochs * steps_per_epoch
        config['optimizer']['scheduler_kwargs']['max_lr'] = config['training']['learning_rate']
    
    # 3. Create model
    model = get_3d_model(config)
    
    # 4. Integrate PerforatedAI
    if config['pai']['enabled']:
        model = setup_pai_pcdet(model, config)
        model = model.to(device)
        optimizer, scheduler = configure_tracker(model, config)
    else:
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = None
    
    # 4. Training loop
    epochs = args.epochs if args.epochs else config['training']['epochs']
    best_map = 0.0
    
    print()
    print("=" * 80)
    print("🚀 Starting Training")
    print("=" * 80)
    
    for epoch in range(epochs):
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in pbar:
            points = batch['points']
            labels = batch['labels']
            
            # Move points to device
            points = [p.to(device) for p in points]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(points)
            
            # Compute loss
            loss = compute_detection_loss(predictions, labels, config)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        
        # ========== Validation Phase ==========
        model.eval()
        val_map = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False):
                points = batch['points']
                labels = batch['labels']
                
                points = [p.to(device) for p in points]
                predictions = model(points)
                
                # Compute mAP
                batch_map = compute_map(predictions, labels)
                val_map += batch_map
                num_val_batches += 1
        
        avg_val_map = val_map / num_val_batches if num_val_batches > 0 else 0.0
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val mAP:    {avg_val_map:.4f}")
        
        # Save best model
        if avg_val_map > best_map:
            best_map = avg_val_map
            save_path = Path("models") / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved (mAP: {best_map:.4f})")
        
        # ========== PAI Tracker Update ==========
        if config['pai']['enabled']:
            model, restructured, complete = GPA.pai_tracker.add_validation_score(avg_val_map, model)
            
            if restructured:
                print("\n" + "=" * 80)
                print("🌿 DENDRITE ADDITION DETECTED!")
                print("=" * 80)
                print("Model has been restructured with new dendritic connections.")
                print("Reinitializing optimizer...")
                
                # Move model back to device and reinitialize optimizer
                model = model.to(device)
                optimizer, scheduler = configure_tracker(model, config)
            
            if complete:
                print("\n" + "=" * 80)
                print("✅ PAI OPTIMIZATION COMPLETE!")
                print("=" * 80)
                print(f"Maximum restructures reached ({config['pai']['max_restructures']}).")
                print("Training will continue with the optimized dendritic architecture.")
                
                # Save PAI visualization
                if config['pai']['save_graphs']:
                    print(f"Saving PAI graph to {config['pai']['output_dir']}/")
                    # The actual graph generation happens inside pai_tracker
                
                break
        
        print("-" * 80)
    
    # ========== Final Save ==========
    final_save_path = Path("models") / "final_model.pt"
    torch.save(model.state_dict(), final_save_path)
    print(f"\n✓ Final model saved to {final_save_path}")
    print(f"✓ Best validation mAP: {best_map:.4f}")
    
    print("\n" + "=" * 80)
    print("🎉 Training Complete!")
    print("=" * 80)
    
    if config['pai']['enabled']:
        print("\n📊 Check the PAI optimization graph:")
        print(f"   {config['pai']['output_dir']}/PAI_DendriticDrive.png")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DendriticDrive Training")
    parser.add_argument('--demo', action='store_true', 
                        help='Run in demo mode with synthetic data')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command-line args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Run training
    train(config, args)
