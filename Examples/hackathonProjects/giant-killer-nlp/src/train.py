"""
Giant-Killer NLP: Main Training Script

This script trains a BERT-Tiny model with Perforated Backpropagation
for toxicity classification. The goal is to achieve BERT-Base level
performance with 15-40x speed improvement.

Usage:
    python src/train.py
    python src/train.py --config configs/config.yaml
    python src/train.py --sample-size 1000  # For quick testing
"""

import argparse
import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_jigsaw_dataset, create_dataloaders, get_tokenizer
from models import create_bert_tiny_model, wrap_with_dendrites
from models.bert_tiny import setup_perforated_optimizer
from training import train_model


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Giant-Killer NLP Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset size for testing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--no-dendrites",
        action="store_true",
        help="Train without dendritic optimization (baseline)",
    )
    parser.add_argument(
        "--augment-toxic",
        action="store_true",
        help="Augment toxic samples to address class imbalance",
    )
    parser.add_argument(
        "--target-toxic-count",
        type=int,
        default=600,
        help="Target number of toxic samples after augmentation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("GIANT-KILLER NLP TRAINING")
    print("Dendritic Optimization for Toxicity Classification")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Configuration: {args.config}")
    print(f"Dendritic Optimization: {'Disabled' if args.no_dendrites else 'Enabled'}")
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    print(f"Random seed: {config.get('seed', 42)}")
    
    # Create output directories
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["logging"]["save_dir"], exist_ok=True)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer(config["model"]["name"])
    print(f"   Tokenizer: {config['model']['name']}")
    
    # Load dataset
    print("\n2. Loading dataset...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_jigsaw_dataset(
            sample_size=args.sample_size,
            augment_toxic=args.augment_toxic,
            target_toxic_count=args.target_toxic_count
        )
    
    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        num_workers=0,  # Set to 0 for Windows compatibility
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    # Create model
    print("\n4. Creating model...")
    model = create_bert_tiny_model(
        num_labels=config["model"]["num_labels"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
    )
    
    # Wrap with dendritic optimization if enabled
    if not args.no_dendrites and config["perforated_ai"]["enabled"]:
        print("\n5. Wrapping model with dendritic optimization...")
        model = wrap_with_dendrites(model)
    else:
        print("\n5. Skipping dendritic wrapping (baseline mode)")
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and scheduler
    print("\n6. Setting up optimizer...")
    optimizer, scheduler = setup_perforated_optimizer(
        model=model,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        scheduler_step_size=config["training"]["scheduler"]["step_size"],
        scheduler_gamma=config["training"]["scheduler"]["gamma"],
        use_pai=(not args.no_dendrites and config["perforated_ai"]["enabled"]),
    )
    
    # Train the model
    print("\n7. Starting training...")
    print("-" * 60)
    
    save_path = os.path.join(config["logging"]["save_dir"], "best_model.pt")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config["training"]["epochs"],
        patience=config["training"]["early_stopping"]["patience"],
        save_path=save_path,
        class_weights=class_weights,
    )
    
    # Save final model
    print("\n8. Saving final model...")
    final_path = os.path.join(config["logging"]["save_dir"], "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "history": history,
    }, final_path)
    print(f"   Final model saved to: {final_path}")
    
    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"\nModel saved to: {save_path}")
    print(f"\nNext step: Run 'python src/evaluate.py' to benchmark the model")


if __name__ == "__main__":
    main()
