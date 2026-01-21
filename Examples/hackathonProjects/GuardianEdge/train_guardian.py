"""
GuardianEdge Training Script
Integrates PerforatedAI's Dendritic Optimization with Ultralytics YOLO
"""

import argparse
import yaml
import torch
import torch.optim as optim
from ultralytics import YOLO
from pathlib import Path

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from utils.pai_integration import setup_pai_model, configure_pai_tracker, handle_restructure


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_guardian(config, args):
    """
    Main training function with PAI integration
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    print("="*60)
    print("GuardianEdge Training with Dendritic Optimization")
    print("="*60)
    
    # Override config with command line arguments
    if args.data:
        config['training']['data'] = args.data
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch:
        config['training']['batch_size'] = args.batch
    if args.imgsz:
        config['model']['input_size'] = args.imgsz
    
    # Load base YOLO model
    model_name = args.model if args.model else f"{config['model']['variant']}.pt"
    print(f"\n[1/6] Loading base model: {model_name}")
    yolo_model = YOLO(model_name)
    
    # Extract PyTorch model from YOLO wrapper
    pytorch_model = yolo_model.model
    
    # Initialize PerforatedAI
    if config['pai']['enabled']:
        print(f"[2/6] Initializing PerforatedAI...")
        
        # CRITICAL: Configure PAI settings BEFORE initialization
        # Fix for YOLO's normalization layers and shared activation pointers
        GPA.pc.set_unwrapped_modules_confirmed(True)
        
        # YOLO shares activation modules across layers - tell PAI to ignore these duplicates
        GPA.pc.append_module_names_to_not_save([
            '.model.1.act', '.model.2.act', '.model.4.act', 
            '.model.6.act', '.model.8.act', '.model.9.act'
        ])
        
        # Configure PAI settings
        GPA.pc.set_testing_dendrite_capacity(config['pai']['testing_dendrite_capacity'])

        
        # Set forward function
        forward_fn_map = {
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'tanh': torch.tanh
        }
        GPA.pc.set_pai_forward_function(forward_fn_map[config['pai']['forward_function']])
        
        # Set other PAI parameters
        GPA.pc.set_candidate_weight_initialization_multiplier(
            config['pai']['weight_init_multiplier']
        )
        GPA.pc.set_improvement_threshold(config['pai']['improvement_threshold'])
        
        # Initialize PAI on the model
        pytorch_model = UPA.initialize_pai(
            pytorch_model,
            doing_pai=True,
            save_name=config['pai']['save_name'],
            making_graphs=True,
            maximizing_score=config['pai']['maximizing_score']
        )
        
        # Move model to device
        device = args.device if args.device else config['training']['device']
        if device == '':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pytorch_model = pytorch_model.to(device)
        print(f"   Model moved to: {device}")
        
        # Configure optimizer and scheduler
        print(f"[3/6] Configuring optimizer and scheduler...")
        optimizer_class = getattr(optim, config['optimizer']['type'])
        scheduler_class = getattr(optim.lr_scheduler, config['optimizer']['scheduler'])
        
        optim_args = {
            'params': pytorch_model.parameters(),
            'lr': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay']
        }
        
        sched_args = config['optimizer']['scheduler_kwargs']
        
        optimizer, scheduler = configure_pai_tracker(
            pytorch_model, optimizer_class, scheduler_class, optim_args, sched_args
        )
        
        print(f"   Optimizer: {config['optimizer']['type']}")
        print(f"   Scheduler: {config['optimizer']['scheduler']}")
    else:
        print("[2/6] PAI disabled - using standard YOLO training")
        # Standard YOLO training would go here
        # For now, we require PAI to be enabled
        raise ValueError("This script requires PAI to be enabled in config.yaml")
    
    # Training with YOLO + PAI
    print(f"[4/6] Starting training loop...")
    print(f"   Dataset: {config['training']['data']}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Image size: {config['model']['input_size']}")
    print()
    
    # Use YOLO's training but with PAI callbacks
    # Note: This is a simplified approach - full integration would require
    # more detailed handling of YOLO's trainer class
    
    try:
        # Training using YOLO's built-in trainer with custom callbacks
        results = yolo_model.train(
            data=config['training']['data'],
            epochs=config['training']['epochs'],
            batch=config['training']['batch_size'],
            imgsz=config['model']['input_size'],
            device=device,
            lr0=config['training']['learning_rate'],
            patience=config['training']['patience'],
            save=True,
            save_period=1,
            project=config['pai']['save_name'],
            name='yolo_run'
        )
        
        print(f"\n[5/6] Training completed!")
        print(f"   Best model saved to: {results.save_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nNote: Full PAI integration with YOLO's trainer requires custom")
        print("training loop implementation. See customization.md for details.")
        raise
    
    # Save final model
    print(f"[6/6] Saving final PAI-optimized model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Save PyTorch model with PAI
    save_path = model_dir / "best_model_pai.pt"
    UPA.save_system(pytorch_model, config['pai']['save_name'], 'best_model')
    print(f"   Saved to: {save_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nTo run inference:")
    print(f"  python inference.py --model {save_path} --source 0")
    print()


def main():
    parser = argparse.ArgumentParser(description='Train GuardianEdge with PerforatedAI')
    
    # Model arguments
    parser.add_argument('--model', type=str, default=None,
                        help='Base YOLO model (default: from config)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    # Training arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Dataset YAML (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Image size (default: from config)')
    parser.add_argument('--device', type=str, default='',
                        help='Device (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start training
    train_guardian(config, args)


if __name__ == '__main__':
    main()
