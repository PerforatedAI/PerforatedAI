#!/usr/bin/env python
"""
Baseline Qwen Training Script

Train Qwen2.5-1.8B-Instruct on GSM8K dataset WITHOUT dendritic optimization.
This serves as a control for comparing against the dendritic version.

Supports W&B sweeps for hyperparameter optimization.

Usage:
    # Direct run
    python -m src.training.train_baseline --learning_rate 5e-5 --num_train_epochs 3
    
    # W&B sweep (args passed automatically)
    wandb agent <sweep_id>
"""

import os
import sys
import argparse
from datetime import datetime

import torch
import wandb
from transformers import Trainer, TrainingArguments

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.baseline_model import QwenBaseline
from src.data.dataset_loader import load_gsm8k_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline Qwen model on GSM8K")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path"
    )
    
    # Training hyperparameters (these can be overridden by sweep)
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    
    # Dataset arguments
    parser.add_argument("--train_samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of eval samples")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results/baseline", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging frequency")
    
    return parser.parse_args()


def train_baseline():
    """Train baseline Qwen model without dendrites."""
    args = parse_args()
    
    # Initialize W&B - let sweep control config if running as agent
    wandb.init(
        project="qwen-dendritic-hackathon",
        name=f"baseline-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": args.model_name,
            "approach": "baseline",
            "num_dendrites": 0,
        }
    )
    
    # Get config from wandb (sweep overrides args)
    config = wandb.config
    learning_rate = getattr(config, "learning_rate", args.learning_rate)
    batch_size = getattr(config, "per_device_train_batch_size", args.per_device_train_batch_size)
    num_epochs = getattr(config, "num_train_epochs", args.num_train_epochs)
    warmup_steps = getattr(config, "warmup_steps", args.warmup_steps)
    
    print(f"Training with: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
    
    # Load model
    print(f"Loading baseline model: {args.model_name}")
    model_wrapper = QwenBaseline(args.model_name)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    
    # Log model size
    model_size = model_wrapper.get_model_size()
    print(f"Model parameters: {model_size['total_millions']:.2f}M total, {model_size['trainable_millions']:.2f}M trainable")
    wandb.log({
        "model_params_millions": model_size["total_millions"],
        "trainable_params_millions": model_size["trainable_millions"],
    })
    
    # Load dataset with train/eval split
    print("Loading GSM8K dataset...")
    train_dataset, eval_dataset = load_gsm8k_dataset(
        tokenizer,
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",  # Evaluate every epoch for sweep metric
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    train_result = trainer.train()
    
    # Final evaluation
    print("Running final evaluation...")
    eval_result = trainer.evaluate()
    
    # Log final metrics
    wandb.log({
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_result["eval_loss"],
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    })
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    wandb.finish()
    print("✅ Baseline training complete!")


if __name__ == "__main__":
    train_baseline()

