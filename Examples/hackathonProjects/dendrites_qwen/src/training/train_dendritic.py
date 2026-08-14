#!/usr/bin/env python
"""
Dendritic Qwen Training Script

Train Qwen2.5-1.5B-Instruct on GSM8K dataset WITH Perforated AI dendritic optimization.
This uses artificial dendrites to improve model performance.

Based on Perforated AI API: https://www.perforatedai.com/docs/api
Example reference: https://github.com/PerforatedAI/PerforatedAI/tree/main/Examples/hackathonProjects/efficientnet-example

Supports W&B sweeps for hyperparameter optimization.

Usage:
    # Direct run
    CUDA_VISIBLE_DEVICES=0 python -m src.training.train_dendritic --learning_rate 5e-5 --num_train_epochs 3
    
    # W&B sweep (args passed automatically)
    wandb agent <sweep_id>
"""

import os
import sys
import argparse
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# Perforated AI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset_loader import load_gsm8k_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train dendritic Qwen model on GSM8K")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path"
    )
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (0 recommended for PAI)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--cleanup_checkpoints", action="store_true", help="Clean up old checkpoints before training")
    
    # Dataset arguments
    parser.add_argument("--train_samples", type=int, default=500, help="Number of training samples")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of eval samples")
    
    # Dendritic optimization arguments
    parser.add_argument("--num_dendrites", type=int, default=3, help="Maximum number of dendrite sets to add")
    parser.add_argument("--n_epochs_to_switch", type=int, default=5, help="Epochs before adding dendrites")
    parser.add_argument("--improvement_threshold", type=float, default=0.01, help="Min improvement to continue")
    parser.add_argument("--testing_dendrite_capacity", action="store_true", help="Test dendrite capacity only")
    parser.add_argument("--dendrite_forward_function", type=str, default="tanh", 
                        choices=["sigmoid", "relu", "tanh"], help="Dendrite activation function")
    parser.add_argument("--candidate_weight_init_multiplier", type=float, default=0.01,
                        help="Multiplier for dendrite weight initialization")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./results/dendritic", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_name", type=str, default="PAI_Qwen", help="Save name for PAI outputs")
    
    return parser.parse_args()


def configure_perforated_ai(args):
    """Configure Perforated AI global settings before model initialization."""
    
    # Testing mode - set to False for real training
    GPA.pc.set_testing_dendrite_capacity(args.testing_dendrite_capacity)
    
    # Maximum number of dendrite sets to add
    GPA.pc.set_max_dendrites(args.num_dendrites)
    
    # How many epochs before switching to dendrite training
    GPA.pc.set_n_epochs_to_switch(args.n_epochs_to_switch)
    
    # Improvement threshold - stop if improvement is less than this
    # Using a list allows progressive thresholds
    if args.improvement_threshold == 0.01:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif args.improvement_threshold == 0.001:
        thresh = [0.001, 0.0001, 0]
    else:
        thresh = [args.improvement_threshold]
    GPA.pc.set_improvement_threshold(thresh)
    
    # Dendrite weight initialization
    GPA.pc.set_candidate_weight_initialization_multiplier(args.candidate_weight_init_multiplier)
    
    # Set dendrite forward function
    if args.dendrite_forward_function == "sigmoid":
        GPA.pc.set_pai_forward_function(torch.sigmoid)
    elif args.dendrite_forward_function == "relu":
        GPA.pc.set_pai_forward_function(torch.relu)
    else:  # tanh
        GPA.pc.set_pai_forward_function(torch.tanh)
    
    # Input dimensions for transformer models: [batch, sequence, hidden]
    GPA.pc.set_output_dimensions([-1, -1, 0])
    
    # Confirm unwrapped modules to skip interactive prompt
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # Accept weight decay to skip the interactive prompt
    # Note: PAI recommends weight_decay=0, but we accept it to avoid prompts
    GPA.pc.set_weight_decay_accepted(True)
    
    # For LLMs, we typically only convert the Linear layers
    # This is similar to what LoRA does - targeting specific projection layers
    GPA.pc.set_modules_to_convert([nn.Linear])
    
    # Track but don't convert these (normalization layers, embeddings)
    GPA.pc.append_module_names_to_track(['RMSNorm', 'Embedding', 'RotaryEmbedding'])
    
    # Enable verbose output for debugging
    GPA.pc.set_verbose(True)
    
    # History lookback for determining when to switch
    GPA.pc.set_history_lookback(1)


def cleanup_old_checkpoints(output_dir: str, save_name: str):
    """Remove old checkpoints to free up space."""
    # Clean PAI save directory
    pai_dir = save_name
    if os.path.exists(pai_dir):
        print(f"🧹 Removing old PAI directory: {pai_dir}")
        shutil.rmtree(pai_dir)
    
    # Clean output directory checkpoints
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint"):
                print(f"🧹 Removing old checkpoint: {item_path}")
                shutil.rmtree(item_path)


def untie_weights(model):
    """
    Untie shared weights in the model to avoid safetensors save issues.
    
    Qwen and many LLMs share weights between embedding and lm_head.
    This causes issues with safetensors saving. We clone the weights
    to break the sharing.
    """
    if hasattr(model, 'get_input_embeddings') and hasattr(model, 'get_output_embeddings'):
        input_emb = model.get_input_embeddings()
        output_emb = model.get_output_embeddings()
        
        if input_emb is not None and output_emb is not None:
            if hasattr(input_emb, 'weight') and hasattr(output_emb, 'weight'):
                if input_emb.weight.data_ptr() == output_emb.weight.data_ptr():
                    print("📌 Untying shared weights between embedding and lm_head...")
                    output_emb.weight = nn.Parameter(input_emb.weight.clone())
    
    return model


def save_model_safe(model, save_path: str):
    """
    Save model safely, handling shared tensors.
    Uses torch.save with cloned tensors to avoid sharing issues.
    """
    state_dict = {}
    for key, value in model.state_dict().items():
        state_dict[key] = value.clone()
    torch.save(state_dict, save_path)


def train_dendritic():
    """Train Qwen model with Perforated AI dendrites."""
    args = parse_args()
    
    # Clean up old checkpoints if requested
    if args.cleanup_checkpoints:
        cleanup_old_checkpoints(args.output_dir, args.save_name)
    
    # Configure Perforated AI BEFORE anything else
    configure_perforated_ai(args)
    
    # Initialize W&B
    run_name = f"dendritic-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="qwen-dendritic-hackathon",
        name=run_name,
        config={
            "model": args.model_name,
            "approach": "dendritic",
            "num_dendrites": args.num_dendrites,
            "n_epochs_to_switch": args.n_epochs_to_switch,
            "improvement_threshold": args.improvement_threshold,
            "dendrite_forward_function": args.dendrite_forward_function,
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "num_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
        }
    )
    
    # Get config from wandb (sweep overrides args)
    config = wandb.config
    learning_rate = getattr(config, "learning_rate", args.learning_rate)
    batch_size = getattr(config, "per_device_train_batch_size", args.per_device_train_batch_size)
    num_epochs = getattr(config, "num_train_epochs", args.num_train_epochs)
    
    print(f"Training with: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
    print(f"Dendritic config: num_dendrites={args.num_dendrites}, switch_epochs={args.n_epochs_to_switch}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # CRITICAL: Untie weights BEFORE PAI initialization to avoid safetensors issues
    model = untie_weights(model)
    
    # Get original parameter count
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params / 1e6:.2f}M")
    
    # Initialize Perforated AI - this adds dendrite scaffolding
    print("Initializing Perforated AI dendrites...")
    model = UPA.initialize_pai(
        model, 
        save_name=args.save_name,
        maximizing_score=False,  # We're minimizing loss
        making_graphs=True,
    )
    
    # Move to device after PAI initialization
    model = model.to(device)
    
    # Get new parameter count (after dendrite scaffolding)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dendrite_overhead = total_params - original_params
    
    print(f"Total parameters after PAI: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Dendrite overhead: {dendrite_overhead / 1e6:.2f}M ({100 * dendrite_overhead / original_params:.2f}%)")
    
    wandb.log({
        "original_params_millions": original_params / 1e6,
        "total_params_millions": total_params / 1e6,
        "trainable_params_millions": trainable_params / 1e6,
        "dendrite_overhead_percent": 100 * dendrite_overhead / original_params,
    })
    
    # Load dataset
    print("Loading GSM8K dataset...")
    train_dataset, eval_dataset = load_gsm8k_dataset(
        tokenizer,
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
    )
    
    # Setup optimizer and scheduler with PAI tracker
    # This is the recommended way per PAI API docs
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    optim_args = {
        'params': model.parameters(),
        'lr': learning_rate,
        'weight_decay': args.weight_decay,
    }
    sched_args = {
        'mode': 'min',  # Minimize loss
        'patience': 3,
        'factor': 0.5,
    }
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
    
    # Training loop
    print("Starting dendritic training...")
    global_step = 0
    best_eval_loss = float('inf')
    training_complete = False
    
    # Track metrics for each architecture (dendrite count)
    arch_metrics = {
        'max_train_loss': float('inf'),
        'max_eval_loss': float('inf'),
        'dendrite_count': 0,
    }
    
    epoch = 0
    while not training_complete:
        epoch += 1
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
            global_step += 1
            
            # Log training metrics
            if global_step % args.logging_steps == 0:
                avg_loss = total_train_loss / num_train_batches
                print(f"Step {global_step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                })
        
        # Calculate epoch training loss
        avg_train_loss = total_train_loss / num_train_batches
        
        # Evaluation
        model.eval()
        total_eval_loss = 0
        num_eval_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_eval_loss += outputs.loss.item()
                num_eval_batches += 1
        
        avg_eval_loss = total_eval_loss / num_eval_batches
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_eval_loss:.4f}")
        
        # Log to W&B
        wandb.log({
            "eval/loss": avg_eval_loss,
            "eval/epoch": epoch,
            "train/epoch_loss": avg_train_loss,
        })
        
        # Track extra score for PAI graphs
        GPA.pai_tracker.add_extra_score(avg_train_loss, 'Train Loss')
        
        # This is the key PAI call - it handles dendrite addition and training switching
        try:
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                avg_eval_loss, 
                model
            )
        except RuntimeError as e:
            if "share memory" in str(e):
                # Handle the shared tensor issue - retry without PAI's auto-save
                print("⚠️ Shared tensor issue detected, continuing without auto-save...")
                restructured = False
                training_complete = epoch >= num_epochs
            else:
                raise e
        
        # Move model back to device AND dtype after potential restructuring
        # PAI may load model in float32, so we need to convert back to bfloat16
        model = model.to(device=device, dtype=torch.bfloat16)
        
        if restructured:
            # Model was restructured (dendrites added or incorporated)
            # Need to reinitialize optimizer
            print("🌿 Model restructured - reinitializing optimizer...")
            
            # CRITICAL: Ensure model is in correct dtype after PAI restructuring
            # PAI saves/loads in float32, we need bfloat16 for Qwen
            model = model.to(dtype=torch.bfloat16)
            
            # Get current dendrite count
            dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
            print(f"   Dendrite count: {dendrite_count}")
            
            # Log architecture metrics
            wandb.log({
                "arch/dendrite_count": dendrite_count,
                "arch/eval_loss": avg_eval_loss,
                "arch/train_loss": avg_train_loss,
            })
            
            # Reinitialize optimizer with PAI tracker
            optim_args = {
                'params': model.parameters(),
                'lr': learning_rate,
                'weight_decay': args.weight_decay,
            }
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
        
        # Track best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"✨ New best eval loss: {best_eval_loss:.4f}")
        
        # Safety check - don't train forever
        if epoch >= num_epochs * 10:  # Allow up to 10x epochs for dendrite training
            print("⚠️ Max epochs reached, stopping training")
            break
    
    # Final evaluation
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    
    # Get final stats
    final_params = sum(p.numel() for p in model.parameters())
    final_dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
    
    print(f"Final dendrite count: {final_dendrite_count}")
    print(f"Final parameter count: {final_params / 1e6:.2f}M")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    
    # Log final metrics
    wandb.log({
        "final/eval_loss": best_eval_loss,
        "final/dendrite_count": final_dendrite_count,
        "final/params_millions": final_params / 1e6,
        "final/overhead_percent": 100 * (final_params - original_params) / original_params,
    })
    
    # Save model using our safe method
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "model_dendritic.pt")
    print(f"💾 Saving model to {model_save_path}...")
    save_model_safe(model, model_save_path)
    print(f"✅ Model saved!")
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    wandb.finish()
    print("✅ Dendritic training complete!")


if __name__ == "__main__":
    train_dendritic()
