"""
PROJECT NEXUS V2: Dendritic SBERT Training Engine
Updated based on reviewer feedback - addressing overfitting with compression experiments

Changes from V1:
1. Added training score tracking (instead of loss) as suggested by Rorry
2. Added --compression flag for testing smaller dense layers
3. Added --dropout flag to combat overfitting
4. Added option to wrap transformer encoder layers
"""

import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Set PAI License (required for Dendrites 2.0)
os.environ["PAIEMAIL"] = "hacker_token@perforatedai.com"
os.environ["PAITOKEN"] = "MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+r4VqudaOlJuH10MIQWm9Uqg=="

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception as e:
    print(f"⚠️ W&B not available: {e}")
    WANDB_AVAILABLE = False

# --- CONFIG & SEEDING ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    print("📊 Loading STS Benchmark...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    train_examples = []
    for row in dataset['train']:
        train_examples.append(InputExample(
            texts=[row['sentence1'], row['sentence2']], 
            label=float(row['score']) / 5.0
        ))
    
    # Create evaluators for both train and validation
    train_evaluator = EmbeddingSimilarityEvaluator(
        [x['sentence1'] for x in dataset['train']],
        [x['sentence2'] for x in dataset['train']],
        [float(x['score'])/5.0 for x in dataset['train']],
        name='sts-train'
    )
        
    val_evaluator = EmbeddingSimilarityEvaluator(
        [x['sentence1'] for x in dataset['validation']],
        [x['sentence2'] for x in dataset['validation']],
        [float(x['score'])/5.0 for x in dataset['validation']],
        name='sts-dev'
    )
    return train_examples, train_evaluator, val_evaluator


class DenseWithDropout(nn.Module):
    """Dense layer with dropout for regularization - helps combat overfitting."""
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, features):
        x = features['sentence_embedding']
        x = self.dropout(x)
        x = self.linear(x)
        features['sentence_embedding'] = x
        return features


def train(config=None):
    # Initialize W&B if available
    global WANDB_AVAILABLE
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(config=config)
            config = wandb.config
            print("✅ W&B initialized successfully")
        except Exception as e:
            print(f"⚠️ W&B init failed: {e}. Continuing without W&B.")
            WANDB_AVAILABLE = False
            # Convert config to args object if it's a dict
            if isinstance(config, dict):
                import argparse
                config = argparse.Namespace(**config)
    else:
        # Convert config to args object if it's a dict
        if isinstance(config, dict):
            import argparse
            config = argparse.Namespace(**config)
    
    set_seed(42)
    
    mode = "DENDRITIC" if config.use_dendrites else "BASELINE"
    compression_info = f" [COMPRESSED {config.compression}x]" if config.compression < 1.0 else ""
    print(f"🚀 Initializing Project NEXUS V2 [{mode}]{compression_info} on {DEVICE}...")
    
    # 1. Architecture Setup
    word_embedding_model = models.Transformer(BASE_MODEL)
    embedding_dim = word_embedding_model.get_word_embedding_dimension()  # 384 for MiniLM
    
    pooling_model = models.Pooling(embedding_dim)
    
    # COMPRESSION: Reduce the dense layer size
    if config.compression < 1.0:
        compressed_dim = int(embedding_dim * config.compression)
        print(f"📐 Compressing dense layer: {embedding_dim} → {compressed_dim} (compression factor: {config.compression})")
        
        if config.dropout > 0:
            # Custom dense with dropout
            dense_model = DenseWithDropout(
                in_features=embedding_dim,
                out_features=compressed_dim,
                dropout_rate=config.dropout
            )
        else:
            dense_model = models.Dense(
                in_features=embedding_dim,
                out_features=compressed_dim,
                bias=True,
                activation_function=nn.Identity()
            )
    else:
        # Original size dense layer
        if config.dropout > 0:
            dense_model = DenseWithDropout(
                in_features=embedding_dim,
                out_features=embedding_dim,
                dropout_rate=config.dropout
            )
        else:
            dense_model = models.Dense(
                in_features=embedding_dim,
                out_features=embedding_dim,
                bias=True,
                activation_function=nn.Identity()
            )
    
    # Combine modules
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 2. Dendritic Injection
    if config.use_dendrites:
        print("⚡ Injecting Dendrites into Adapter Layer...")
        
        # PAI Configuration
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_verbose(True)
        
        # Disable Capacity Test mode
        GPA.pc.set_testing_dendrite_capacity(False)
        
        # Configure Switch Behavior
        print(f"⏳ Forcing {config.warmup_epochs}-epoch warmup before Dendritic Switch...")
        GPA.pc.set_n_epochs_to_switch(config.warmup_epochs)
        
        # Generate unique save name
        if WANDB_AVAILABLE and wandb.run and wandb.run.id:
            pai_save_name = f"PAI_{wandb.run.id}"
        else:
            import uuid
            pai_save_name = f"PAI_{str(uuid.uuid4())[:8]}"
            
        # Initialize PAI on the adapter layer (model[2])
        model[2] = UPA.initialize_pai(
            model[2], 
            save_name=pai_save_name 
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
        
    else:
        print("⚠️ Running in BASELINE mode.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 3. Training Loop
    train_examples, train_evaluator, val_evaluator = load_data()
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=config.batch_size, 
        collate_fn=model.smart_batching_collate
    )
    train_loss = losses.CosineSimilarityLoss(model)
    
    print(f"🎯 Training Started...")
    epoch = -1
    metrics = []
    
    while True:
        epoch += 1
        
        # Stopping conditions
        if not config.use_dendrites and epoch >= config.epochs:
            break
        if config.use_dendrites and epoch >= 100:
            print("⚠️ Reached maximum safety epoch limit (100). Stopping.")
            break
            
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            features, labels = batch
            loss_value = train_loss(features, labels)
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.item()

        # Validation & Training Evaluation
        model.eval()
        with torch.no_grad():
            val_score_raw = val_evaluator(model)
            train_score_raw = train_evaluator(model)
        
        # Extract scores
        if isinstance(val_score_raw, dict):
            val_score = val_score_raw.get("sts-dev_spearman_cosine")
            if val_score is None:
                val_score = next(iter(val_score_raw.values()))
        else:
            val_score = val_score_raw
            
        if isinstance(train_score_raw, dict):
            train_score = train_score_raw.get("sts-train_spearman_cosine")
            if train_score is None:
                train_score = next(iter(train_score_raw.values()))
        else:
            train_score = train_score_raw

        avg_loss = total_loss / len(train_dataloader)
        
        # Track TRAINING SCORE (not loss) for PAI visualization - as per Rorry's suggestion
        if config.use_dendrites:
            GPA.pai_tracker.add_extra_score(train_score * 100, 'Train Score')
        
        # Calculate overfitting gap
        overfit_gap = train_score - val_score
        
        print(f"📊 Epoch {epoch} | Loss: {avg_loss:.4f} | Train Spearman: {train_score:.4f} | Val Spearman: {val_score:.4f} | Overfit Gap: {overfit_gap:.4f}")
        
        # W&B Logging
        if WANDB_AVAILABLE:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_spearman": train_score,
                    "val_spearman": val_score,
                    "overfit_gap": overfit_gap
                })
            except:
                pass

        # Store metrics
        metrics.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_spearman": train_score,
            "val_spearman": val_score,
            "overfit_gap": overfit_gap
        })
        
        # Save metrics immediately
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        metrics_path = f"{config.save_dir}/metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        except:
            pass

        # Dendritic Evolution Logic
        if config.use_dendrites:
            model[2], restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score * 100, model[2])
            
            if restructured:
                print(">>> ⚡ DENDRITES ACTIVATED! Architecture Evolved. ⚡ <<<")
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
                
                if WANDB_AVAILABLE:
                    try:
                        wandb.log({"dendrite_restructure": epoch})
                    except:
                        pass
                
            if training_complete:
                print("🏆 Training Complete per PAI.")
                break
        
        # Save Checkpoints
        if epoch > 0:
            save_path = f"{config.save_dir}/checkpoint_epoch_{epoch}"
            model.save(save_path)
    
    # Final save
    final_path = f"{config.save_dir}/final_model"
    model.save(final_path)
    print(f"✅ Model saved to {final_path}")
    
    # Final metrics summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"Mode: {mode}{compression_info}")
    print(f"Final Train Spearman: {metrics[-1]['train_spearman']:.4f}")
    print(f"Final Val Spearman: {metrics[-1]['val_spearman']:.4f}")
    print(f"Final Overfit Gap: {metrics[-1]['overfit_gap']:.4f}")
    print(f"Best Val Spearman: {max(m['val_spearman'] for m in metrics):.4f}")
    print("="*60)
    
    if WANDB_AVAILABLE:
        try:
            wandb.finish()
        except:
            pass

    # Save final metrics
    metrics_path = f"{config.save_dir}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project NEXUS V2 Training - Compression Experiments")
    parser.add_argument("--use_dendrites", action="store_true", help="Enable Dendritic Optimization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=6, help="Epochs before allowing dendrites to spawn")
    parser.add_argument("--save_dir", type=str, default="experiments/default_run", help="Directory to save output")
    parser.add_argument("--wandb_project", type=str, default="PerforatedAI-Examples_hackathonProjects_Project-Nexus-SBERT_src", help="W&B project name")
    
    # NEW: Compression and Regularization arguments
    parser.add_argument("--compression", type=float, default=1.0, 
                        help="Compression factor for dense layer (0.25 = 25% of original size, 0.5 = 50%, 1.0 = no compression)")
    parser.add_argument("--dropout", type=float, default=0.0, 
                        help="Dropout rate for regularization (0.0 = no dropout, 0.1-0.3 recommended)")
    
    args = parser.parse_args()
    
    # Configure W&B
    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Convert args to dict for W&B
    config = vars(args)
    train(config)
