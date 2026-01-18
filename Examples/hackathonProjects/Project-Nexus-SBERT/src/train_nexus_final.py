"""
PROJECT NEXUS: Dendritic SBERT Training Engine (Final Version)
Combines: Compression + Regularization + Early Stopping + Best Checkpointing

Features:
- Compression: Reduce dense layer capacity (--compression 0.10 = 10% of original)
- Dropout: Randomly zero activations (--dropout 0.3)
- Noise: Add Gaussian noise during training (--noise_std 0.01)
- Gradient Clipping: Prevent exploding gradients (max_norm=1.0)
- Early Stopping: Stop when overfit gap exceeds threshold
- Best Checkpoint: Save and restore best validation model
"""

import argparse
import json
import os
import random
import uuid
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Set PAI License
os.environ["PAIEMAIL"] = "hacker_token@perforatedai.com"
os.environ["PAITOKEN"] = "MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+r4VqudaOlJuH10MIQWm9Uqg=="

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception as e:
    print(f"W&B not available: {e}")
    WANDB_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RegularizationWrapper(nn.Module):
    """Wraps a module with dropout and noise regularization."""
    
    def __init__(self, module, dropout_rate=0.3, noise_std=0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout_rate)
        self.noise_std = noise_std
    
    def forward(self, features):
        output = self.module(features)
        if 'sentence_embedding' in output:
            # Apply dropout
            output['sentence_embedding'] = self.dropout(output['sentence_embedding'])
            # Add Gaussian noise during training only
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(output['sentence_embedding']) * self.noise_std
                output['sentence_embedding'] = output['sentence_embedding'] + noise
        return output


def load_data():
    print("Loading STS Benchmark...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    train_examples = []
    for row in dataset['train']:
        train_examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']], 
            label=float(row['score']) / 5.0
        ))
        
    val_evaluator = EmbeddingSimilarityEvaluator(
        [x['sentence1'] for x in dataset['validation']],
        [x['sentence2'] for x in dataset['validation']],
        [float(x['score'])/5.0 for x in dataset['validation']],
        name='sts-dev'
    )
    
    train_evaluator = EmbeddingSimilarityEvaluator(
        [x['sentence1'] for x in dataset['train']],
        [x['sentence2'] for x in dataset['train']],
        [float(x['score'])/5.0 for x in dataset['train']],
        name='sts-train'
    )
    
    return train_examples, val_evaluator, train_evaluator


def train(config):
    global WANDB_AVAILABLE
    
    # Initialize W&B
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(config=config, project=config.get('wandb_project', 'NEXUS'))
            config = wandb.config
            print("W&B initialized successfully")
        except Exception as e:
            print(f"W&B init failed: {e}")
            WANDB_AVAILABLE = False
    
    # Convert config to namespace if dict
    if isinstance(config, dict):
        config = argparse.Namespace(**config)
    
    set_seed(42)
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # --- Model Setup ---
    mode_str = "DENDRITIC" if config.use_dendrites else "BASELINE"
    compress_str = f"[COMPRESSED {config.compression}x]" if config.compression < 1.0 else ""
    reg_str = f"[DROPOUT:{config.dropout}]" if config.dropout > 0 else ""
    print(f"Initializing NEXUS {mode_str} {compress_str} {reg_str} on {DEVICE}...")
    
    # Build model
    word_embedding_model = models.Transformer(BASE_MODEL)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
    # Dense layer with optional compression
    in_features = word_embedding_model.get_word_embedding_dimension()  # 384
    out_features = int(in_features * config.compression)  # Apply compression
    
    if out_features != in_features:
        print(f"Compressing dense layer: {in_features} -> {out_features} (compression factor: {config.compression})")
    
    dense_model = models.Dense(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        activation_function=nn.Identity()
    )
    
    # Wrap with regularization if dropout or noise is enabled
    if config.dropout > 0 or config.noise_std > 0:
        print(f"Adding regularization: dropout={config.dropout}, noise_std={config.noise_std}")
        dense_model = RegularizationWrapper(dense_model, config.dropout, config.noise_std)
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # --- Dendritic Setup ---
    if config.use_dendrites:
        print("Injecting Dendrites...")
        
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_verbose(True)
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_n_epochs_to_switch(config.warmup_epochs)
        
        pai_save_name = f"PAI_{uuid.uuid4().hex[:8]}"
        
        # Initialize PAI on the dense layer (inside wrapper if wrapped)
        if isinstance(model[2], RegularizationWrapper):
            model[2].module = UPA.initialize_pai(model[2].module, save_name=pai_save_name)
        else:
            model[2] = UPA.initialize_pai(model[2], save_name=pai_save_name)
        
        print(f"Running Dendritic Experiment with warmup={config.warmup_epochs} epochs")
    else:
        print("Running in BASELINE mode.")
    
    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    if config.use_dendrites:
        GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
    
    # --- Training Loop ---
    train_examples, val_evaluator, train_evaluator = load_data()
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=config.batch_size, 
        collate_fn=model.smart_batching_collate
    )
    train_loss = losses.CosineSimilarityLoss(model)
    
    print("Training Started...")
    
    metrics = []
    best_val_score = 0
    best_epoch = 0
    epochs_without_improvement = 0
    
    epoch = -1
    while True:
        epoch += 1
        
        # Stopping conditions
        if not config.use_dendrites and epoch >= config.epochs:
            break
        if config.use_dendrites and epoch >= 100:
            print("Reached max epochs (100). Stopping.")
            break
        
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            features, labels = batch
            
            features = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                        for k, v in f.items()} for f in features]
            labels = labels.to(DEVICE)
            
            loss_value = train_loss(features, labels)
            loss_value.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss_value.item()
        
        avg_loss = total_loss / len(train_dataloader)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            model.to('cpu')
            raw_val_score = val_evaluator(model)
            raw_train_score = train_evaluator(model)
            model.to(DEVICE)
        
        # Extract scores
        val_score = raw_val_score.get("sts-dev_spearman_cosine", next(iter(raw_val_score.values()))) if isinstance(raw_val_score, dict) else raw_val_score
        train_score = raw_train_score.get("sts-train_spearman_cosine", next(iter(raw_train_score.values()))) if isinstance(raw_train_score, dict) else raw_train_score
        
        overfit_gap = train_score - val_score
        
        # Track training score for PAI visualization
        if config.use_dendrites:
            GPA.pai_tracker.add_extra_score(train_score * 100, 'Train Score')
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Train: {train_score:.4f} | Val: {val_score:.4f} | Gap: {overfit_gap:.4f}")
        
        # Best checkpoint tracking
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model (handle RegularizationWrapper)
            best_path = f"{config.save_dir}/best_model"
            if isinstance(model[2], RegularizationWrapper):
                # Temporarily swap in the inner module for saving
                original_module = model[2]
                model._modules['2'] = model[2].module
                try:
                    model.save(best_path)
                finally:
                    model._modules['2'] = original_module
            else:
                model.save(best_path)
            print(f"  -> New best! Saved to {best_path}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping on overfit gap
        if overfit_gap > config.max_overfit_gap and epoch > config.warmup_epochs:
            print(f"Overfit gap ({overfit_gap:.4f}) exceeds threshold ({config.max_overfit_gap}). Early stopping.")
            break
        
        # Early stopping on patience
        if epochs_without_improvement >= config.patience:
            print(f"No improvement for {config.patience} epochs. Early stopping.")
            break
        
        # W&B Logging
        if WANDB_AVAILABLE:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_spearman": train_score,
                    "val_spearman": val_score,
                    "overfit_gap": overfit_gap,
                    "best_val_spearman": best_val_score
                })
            except:
                pass
        
        # Save metrics
        metrics.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_spearman": train_score,
            "val_spearman": val_score,
            "overfit_gap": overfit_gap
        })
        
        with open(f"{config.save_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Dendritic evolution
        if config.use_dendrites:
            if isinstance(model[2], RegularizationWrapper):
                model[2].module, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score * 100, model[2].module)
            else:
                model[2], restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score * 100, model[2])
            
            if restructured:
                print(">>> DENDRITES ACTIVATED! Architecture Evolved. <<<")
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
            
            if training_complete:
                print("Training Complete per PAI.")
                break
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Validation Spearman: {best_val_score:.4f} (epoch {best_epoch})")
    print(f"Final Epoch: {epoch}")
    print(f"Best model saved to: {config.save_dir}/best_model")
    print("="*60)
    
    # Final W&B logging
    if WANDB_AVAILABLE:
        try:
            wandb.log({
                "final_best_val_spearman": best_val_score,
                "final_best_epoch": best_epoch
            })
            wandb.finish()
        except:
            pass
    
    return best_val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project NEXUS Final Training")
    
    # Mode
    parser.add_argument("--use_dendrites", action="store_true", help="Enable Dendritic Optimization")
    
    # Training
    parser.add_argument("--epochs", type=int, default=15, help="Max epochs for baseline mode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs before dendrites can activate")
    
    # Regularization
    parser.add_argument("--compression", type=float, default=1.0, help="Dense layer compression (0.1 = 10% of original)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Gaussian noise std")
    
    # Early stopping
    parser.add_argument("--max_overfit_gap", type=float, default=0.15, help="Max allowed overfit gap before early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Epochs without improvement before early stopping")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="experiments/final", help="Save directory")
    parser.add_argument("--wandb_project", type=str, default="NEXUS-Final", help="W&B project")
    
    args = parser.parse_args()
    
    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    train(vars(args))
