"""
Unified training script for vanilla and dendritic Transformer language models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import time
import math
from tqdm import tqdm
import os

from data_preparation import load_wikitext2
from model import create_model


def get_device():
    """Get the best available device (MPS for Mac, CUDA for GPU, CPU otherwise)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_perplexity(model, data_loader, criterion, device):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Reshape for cross entropy: (batch * seq_len, vocab_size) and (batch * seq_len)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def train_vanilla(args, model, train_loader, val_loader, device, vocab):
    """Train vanilla transformer model."""
    print(f"\n{'='*60}")
    print("TRAINING VANILLA MODEL")
    print(f"{'='*60}\n")

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"vanilla_{args.embed_dim}d_{args.num_layers}l",
        config={
            "model_type": "vanilla",
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "vocab_size": len(vocab),
            "total_params": model.count_parameters(),
        },
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    best_val_perplexity = float("inf")
    training_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / num_batches

        # Validation
        val_loss, val_perplexity = compute_perplexity(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start_time

        # Logging
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.2f}s")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
                "total_params": model.count_parameters(),
            }
        )

        # Save best model
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            torch.save(model.state_dict(), "best_vanilla_model.pt")
            print(f"  ✓ New best model saved!")

    total_training_time = time.time() - training_start_time

    print(f"\n{'='*60}")
    print("VANILLA MODEL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(
        f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)"
    )
    print(f"Best validation perplexity: {best_val_perplexity:.2f}")
    print(f"Final model parameters: {model.count_parameters():,}")
    print(f"{'='*60}\n")

    wandb.log(
        {
            "total_training_time": total_training_time,
            "best_val_perplexity": best_val_perplexity,
            "final_params": model.count_parameters(),
        }
    )

    wandb.finish()

    return model, best_val_perplexity


def train_dendritic(args, model, train_loader, val_loader, device, vocab):
    """Train dendritic transformer model with PerforatedAI."""
    print(f"\n{'='*60}")
    print("TRAINING DENDRITIC MODEL")
    print(f"{'='*60}\n")

    try:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA

        # Store base parameters before adding dendrites
        base_params = model.count_parameters()

        # Save the original output projection to restore it after PerforatedAI initialization
        # (We don't want dendrites on the final projection layer)
        import copy

        original_output_proj = copy.deepcopy(model.output_projection)

        # Configure PerforatedAI for Transformers (3D tensors: [batch, sequence, features])
        GPA.pc.set_testing_dendrite_capacity(False)  # False for real experiments, True for testing
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_input_dimensions([-1, -1, 0])  # Critical for 3D tensor support
        GPA.pc.set_module_names_to_convert(["Linear"])
        GPA.pc.set_improvement_threshold([0.01, 0.001, 0])  # 1% improvement threshold
        GPA.pc.set_candidate_weight_initialization_multiplier(0.1)
        # Final projection doesn't need dendrites
        GPA.pc.append_module_ids_to_track([".output_projection"])
        # Initialize model with dendrites
        model = UPA.initialize_pai(
            model,
            doing_pai=True,
            save_name=f"dendritic_{args.embed_dim}d_{args.num_layers}l",
            making_graphs=True,
            maximizing_score=False,
        )

        model = model.to(device)

        # Set up optimizer tracking
        GPA.pai_tracker.set_optimizer(torch.optim.Adam)
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

        optim_args = {"params": model.parameters(), "lr": args.learning_rate}
        sched_args = {"mode": "min", "patience": 2, "factor": 0.5}

        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optim_args, sched_args
        )

    except ImportError as e:
        print(f"\n⚠ WARNING: PerforatedAI library not available: {e}")
        print("Falling back to vanilla training mode...\n")
        return train_vanilla(args, model, train_loader, val_loader, device, vocab)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"dendritic_{args.embed_dim}d_{args.num_layers}l",
        config={
            "model_type": "dendritic",
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "vocab_size": len(vocab),
            "initial_params": model.count_parameters(),
        },
    )

    criterion = nn.CrossEntropyLoss()

    best_val_perplexity = float("inf")
    training_start_time = time.time()
    initial_params = model.count_parameters()  # Params after initialize_pai
    initial_dendrites = (
        initial_params - base_params
    )  # Dendrites added at initialization
    dynamic_dendrites_added = 0  # Counter for dendrites added DURING training

    print(f"\n📊 Dendrite Initialization:")
    print(f"  Base model parameters: {base_params:,}")
    print(f"  After initialize_pai: {initial_params:,}")
    print(
        f"  Initial dendrites added: +{initial_dendrites:,} (+{initial_dendrites/base_params*100:.1f}%)\n"
    )
    epoch = -1
    while True:
        epoch += 1
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / num_batches
        # Validation
        val_loss, val_perplexity = compute_perplexity(
            model, val_loader, criterion, device
        )

        current_params = model.count_parameters()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start_time

        # Logging
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Current Params: {current_params:,}")
        print(f"  Time: {epoch_time:.2f}s")

        GPA.pai_tracker.add_extra_score(avg_train_loss, "train loss")
        GPA.pai_tracker.add_extra_score_without_graphing(
            val_perplexity / 100.0, "val perplexity/100"
        )

        # Add validation score to pai_tracker
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_loss, model
        )

        if restructured:
            dynamic_dendrites_added += 1
            new_param_count = model.count_parameters()
            param_increase = new_param_count - current_params

            print(f"  🌳 DYNAMIC DENDRITE ADDED! #{dynamic_dendrites_added}")
            print(f"     Previous params: {current_params:,}")
            print(f"     New params: {new_param_count:,}")
            print(
                f"     Increase: +{param_increase:,} ({param_increase/current_params*100:.2f}%)"
            )

            # Re-initialize model on device and optimizer
            model = model.to(device)
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, optim_args, sched_args
            )

            wandb.log(
                {
                    "dynamic_dendrite_added": dynamic_dendrites_added,
                    "params_after_dendrite": new_param_count,
                }
            )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
                "total_params": model.count_parameters(),
                "dynamic_dendrites_count": dynamic_dendrites_added,
            }
        )

        # Save best model
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            torch.save(model.state_dict(), "best_dendritic_model.pt")
            print(f"  ✓ New best model saved!")

        # Check if training is complete
        if training_complete:
            print("\n  ⚠ pbTracker indicates training is complete. Stopping early.")
            break
        
        # Check if we've reached the epoch limit
        if epoch + 1 >= args.epochs:
            print(f"\n  ⚠ Reached epoch limit ({args.epochs}). Stopping.")
            break

    total_training_time = time.time() - training_start_time
    final_params = model.count_parameters()
    dynamic_param_increase = final_params - initial_params
    total_param_increase = final_params - base_params

    print(f"\n{'='*60}")
    print("DENDRITIC MODEL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(
        f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)"
    )
    print(f"Best validation perplexity: {best_val_perplexity:.2f}")
    print(f"\nParameter Summary:")
    print(f"  Base model: {base_params:,}")
    print(
        f"  Initial dendrites (at init): +{initial_dendrites:,} (+{initial_dendrites/base_params*100:.1f}%)"
    )
    print(f"  Dynamic dendrites (during training): +{dynamic_param_increase:,}")
    print(f"  Final parameters: {final_params:,}")
    print(
        f"  Total increase: +{total_param_increase:,} (+{total_param_increase/base_params*100:.1f}%)"
    )
    print(f"\nDynamic dendrite additions: {dynamic_dendrites_added}")
    print(f"{'='*60}\n")

    wandb.log(
        {
            "total_training_time": total_training_time,
            "best_val_perplexity": best_val_perplexity,
            "base_params": base_params,
            "initial_dendrites": initial_dendrites,
            "initial_params_with_dendrites": initial_params,
            "final_params": final_params,
            "total_param_increase": total_param_increase,
            "dynamic_param_increase": dynamic_param_increase,
            "dynamic_dendrite_additions": dynamic_dendrites_added,
        }
    )

    wandb.finish()

    return model, best_val_perplexity


def main():
    parser = argparse.ArgumentParser(description="Train Transformer Language Models")

    # Model type
    parser.add_argument(
        "--model_type",
        type=str,
        default="vanilla",
        choices=["vanilla", "dendritic"],
        help="Type of model to train",
    )

    # Model hyperparameters
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="Embedding dimension (default: 256 for vanilla, 128 for dendritic)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate for model layers"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=50, help="Sequence length")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--max_vocab_size", type=int, default=10000, help="Maximum vocabulary size"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dendritic-transformer-comparison",
        help="W&B project name",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Set default embed_dim based on model type if not specified
    if args.embed_dim is None:
        args.embed_dim = 256 if args.model_type == "vanilla" else 128

    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, vocab = load_wikitext2(
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
    )

    # Create model
    model = create_model(
        vocab_size=len(vocab),
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Train
    if args.model_type == "vanilla":
        model, best_ppl = train_vanilla(
            args, model, train_loader, val_loader, device, vocab
        )
    else:
        model, best_ppl = train_dendritic(
            args, model, train_loader, val_loader, device, vocab
        )

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
