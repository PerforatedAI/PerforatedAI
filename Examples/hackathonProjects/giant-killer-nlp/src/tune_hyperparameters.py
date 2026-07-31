"""
Hyperparameter Tuning Script

Performs grid search over learning rate, class weights, and batch size
to find optimal configuration for toxicity classification.

Usage:
    python src/tune_hyperparameters.py --sample-size 3000 --epochs 5
"""

import argparse
import os
import sys
import yaml
import torch
import json
import itertools
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_jigsaw_dataset, create_dataloaders, get_tokenizer
from models import create_bert_tiny_model, wrap_with_dendrites
from models.bert_tiny import setup_perforated_optimizer
from training import train_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def tune_hyperparameters(
    sample_size: int,
    epochs: int,
    device: torch.device,
    config: dict,
    output_dir: str = "tuning_results",
):
    """
    Perform grid search over hyperparameters.
    
    Args:
        sample_size: Number of samples to use for tuning
        epochs: Number of epochs per trial
        device: Training device
        config: Base configuration
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"tuning_results_{timestamp}.json")
    
    # Define hyperparameter grid
    param_grid = {
        "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
        "batch_size": [16, 32, 64],
        "class_weight_multiplier": [1.0, 1.5, 2.0],  # Multiplier for toxic class weight
    }
    
    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(config["model"]["name"])
    
    # Load dataset once
    print(f"Loading dataset (sample_size={sample_size})...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_jigsaw_dataset(sample_size=sample_size)
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid["learning_rate"],
        param_grid["batch_size"],
        param_grid["class_weight_multiplier"]
    ))
    
    print(f"\nGrid Search: {len(param_combinations)} combinations")
    print(f"  Learning rates: {param_grid['learning_rate']}")
    print(f"  Batch sizes: {param_grid['batch_size']}")
    print(f"  Class weight multipliers: {param_grid['class_weight_multiplier']}")
    print(f"  Epochs per trial: {epochs}")
    print(f"  Total trials: {len(param_combinations)}")
    print("="*60)
    
    results = []
    best_val_acc = 0.0
    best_params = None
    
    for trial_idx, (lr, batch_size, weight_mult) in enumerate(param_combinations, 1):
        print(f"\n[Trial {trial_idx}/{len(param_combinations)}]")
        print(f"  Learning Rate: {lr}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Class Weight Multiplier: {weight_mult}")
        print("-"*60)
        
        try:
            # Create dataloaders with current batch size
            train_loader, val_loader, test_loader, class_weights = create_dataloaders(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=config["data"]["max_length"],
                num_workers=0,
            )
            
            # Apply class weight multiplier
            if weight_mult != 1.0:
                class_weights[1] *= weight_mult  # Multiply toxic class weight
                print(f"  Adjusted class weights: Non-toxic={class_weights[0]:.4f}, Toxic={class_weights[1]:.4f}")
            
            class_weights = class_weights.to(device)
            
            # Create model (fresh for each trial)
            model = create_bert_tiny_model(
                num_labels=config["model"]["num_labels"],
                hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
            )
            model = model.to(device)
            
            # Setup optimizer with current learning rate
            optimizer, scheduler = setup_perforated_optimizer(
                model=model,
                learning_rate=lr,
                weight_decay=config["training"]["weight_decay"],
                scheduler_step_size=config["training"]["scheduler"]["step_size"],
                scheduler_gamma=config["training"]["scheduler"]["gamma"],
                use_pai=False,  # Use standard optimizer for faster tuning
            )
            
            # Train
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=epochs,
                patience=epochs,  # No early stopping during tuning
                save_path=None,  # Don't save models during tuning
                class_weights=class_weights,
            )
            
            # Get best validation accuracy
            best_trial_val_acc = max(history["val_acc"])
            best_trial_val_loss = min(history["val_loss"])
            final_train_acc = history["train_acc"][-1]
            final_train_loss = history["train_loss"][-1]
            
            # Store results
            trial_result = {
                "trial": trial_idx,
                "learning_rate": lr,
                "batch_size": batch_size,
                "class_weight_multiplier": weight_mult,
                "best_val_acc": best_trial_val_acc,
                "best_val_loss": best_trial_val_loss,
                "final_train_acc": final_train_acc,
                "final_train_loss": final_train_loss,
                "history": history,
            }
            results.append(trial_result)
            
            print(f"\n  Results:")
            print(f"    Best Val Acc: {best_trial_val_acc:.4f}")
            print(f"    Best Val Loss: {best_trial_val_loss:.4f}")
            print(f"    Final Train Acc: {final_train_acc:.4f}")
            
            # Update best
            if best_trial_val_acc > best_val_acc:
                best_val_acc = best_trial_val_acc
                best_params = {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "class_weight_multiplier": weight_mult,
                }
                print(f"    [NEW BEST] Val Acc: {best_val_acc:.4f}")
            
            # Save intermediate results
            with open(results_file, "w") as f:
                json.dump({
                    "best_params": best_params,
                    "best_val_acc": best_val_acc,
                    "results": results,
                }, f, indent=2)
                
        except Exception as e:
            print(f"  [ERROR] Trial failed: {e}")
            trial_result = {
                "trial": trial_idx,
                "learning_rate": lr,
                "batch_size": batch_size,
                "class_weight_multiplier": weight_mult,
                "error": str(e),
            }
            results.append(trial_result)
    
    # Print final summary
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    print(f"\nBest Parameters:")
    print(f"  Learning Rate: {best_params['learning_rate']}")
    print(f"  Batch Size: {best_params['batch_size']}")
    print(f"  Class Weight Multiplier: {best_params['class_weight_multiplier']}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Print top 5 results
    print(f"\nTop 5 Configurations:")
    sorted_results = sorted(
        [r for r in results if "best_val_acc" in r],
        key=lambda x: x["best_val_acc"],
        reverse=True
    )[:5]
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"\n{rank}. Val Acc: {result['best_val_acc']:.4f}")
        print(f"   LR={result['learning_rate']}, BS={result['batch_size']}, "
              f"WM={result['class_weight_multiplier']}")
    
    print(f"\nResults saved to: {results_file}")
    
    return best_params, results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Giant-Killer NLP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Number of samples for tuning (use smaller for faster search)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per trial",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_results",
        help="Directory to save results",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("Giant-Killer NLP - Grid Search Optimization")
    print("="*60)
    print(f"\nDevice: {device}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Epochs per trial: {args.epochs}")
    
    # Run grid search
    best_params, results = tune_hyperparameters(
        sample_size=args.sample_size,
        epochs=args.epochs,
        device=device,
        config=config,
        output_dir=args.output_dir,
    )
    
    print("\nUse these parameters in your config.yaml:")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  batch_size: {best_params['batch_size']}")
    print(f"  # Apply weight multiplier {best_params['class_weight_multiplier']} "
          f"to toxic class weights")


if __name__ == "__main__":
    main()
