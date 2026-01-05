"""
PROJECT NEXUS: Dendritic SBERT Training Engine
Optimized for Perforated AI Hackathon 2025
"""

import argparse
import os
import random
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
        
    evaluator = EmbeddingSimilarityEvaluator(
        [x['sentence1'] for x in dataset['validation']],
        [x['sentence2'] for x in dataset['validation']],
        [float(x['score'])/5.0 for x in dataset['validation']],
        name='sts-dev'
    )
    return train_examples, evaluator

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
    print(f"🚀 Initializing Project NEXUS [{mode}] on {DEVICE}...")
    
    # 1. Architecture Setup
    word_embedding_model = models.Transformer(BASE_MODEL)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=word_embedding_model.get_word_embedding_dimension(),
        out_features=word_embedding_model.get_word_embedding_dimension(),
        bias=True,
        activation_function=nn.Identity() 
    )
    
    # Combine modules
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    model.to(DEVICE)

    # 2. Dendritic Injection
    if config.use_dendrites:
        print("⚡ Injecting Dendrites into Adapter Layer ONLY...")
        
        # PAI Configuration (Must be done BEFORE initialization in some cases)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_verbose(True) # Enable verbose logging to debug switch
        
        # CRITICAL: Disable 'Capacity Test' mode (which stops after 3 dendrites)
        # This allows the model to train for the full duration
        GPA.pc.set_testing_dendrite_capacity(False)
        
        # Configure Switch Behavior
        # Ideally, we want to ensure it doesn't switch too aggressively
        print(f"⏳ Forcing {config.warmup_epochs}-epoch warmup before Dendritic Switch...")
        GPA.pc.set_n_epochs_to_switch(config.warmup_epochs) 
        
        # Initialize PAI on ONLY the adapter layer (model[2])
        model[2] = UPA.initialize_pai(
            model[2], 
            save_name="PAI" 
        )
        
        # Optimizer - use model.parameters() to train all layers
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # Register optimizer with PAI tracker
        GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
        
    else:
        print("⚠️ Running in BASELINE mode.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 3. Training Loop
    train_examples, evaluator = load_data()
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
        
        # In Dendritic mode, we strictly follow PAI's training_complete signal
        # In Baseline mode, we follow the fixed epochs
        if not config.use_dendrites and epoch >= config.epochs:
            break
        # Safety break for dendritic to prevent infinite loops if something goes wrong
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

        # Validation
        model.eval()
        with torch.no_grad():
            raw_score = evaluator(model)
        
        if isinstance(raw_score, dict):
            score = raw_score.get("sts-dev_spearman_cosine")
            if score is None:
                score = next(iter(raw_score.values()))
        else:
            score = raw_score

        avg_loss = total_loss / len(train_dataloader)
        
        # Track training loss for PAI visualization
        if config.use_dendrites:
            GPA.pai_tracker.add_extra_score(avg_loss, 'Train Loss')
        
        print(f"📊 Epoch {epoch} | Loss: {avg_loss:.4f} | Spearman: {score:.4f}")
        
        # W&B Logging
        if WANDB_AVAILABLE:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_spearman": score
                })
            except:
                pass

        # Store metrics
        metrics.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_spearman": score
        })
        
        # Save metrics immediately (so we can plot progress)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        metrics_path = f"{config.save_dir}/metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        except:
            pass # Don't crash training on file IO error

        # Dendritic Evolution Logic
        if config.use_dendrites:
            model[2], restructured, training_complete = GPA.pai_tracker.add_validation_score(score * 100, model[2])
            
            if restructured:
                print(">>> ⚡ DENDRITES ACTIVATED! Architecture Evolved. ⚡ <<<")
                # Recreate optimizer with updated parameters
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                # Re-register optimizer with PAI tracker
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
    
    if WANDB_AVAILABLE:
        try:
            wandb.finish()
        except:
            pass

    # Save metrics locally
    import json
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    
    metrics_path = f"{config.save_dir}/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project NEXUS Training")
    parser.add_argument("--use_dendrites", action="store_true", help="Enable Dendritic Optimization")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=6, help="Epochs before allowing dendrites to spawn")
    parser.add_argument("--save_dir", type=str, default="experiments/default_run", help="Directory to save output")
    parser.add_argument("--wandb_project", type=str, default="project-nexus-sbert", help="W&B project name")
    args = parser.parse_args()
    
    # Configure W&B
    os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Convert args to dict for W&B
    config = vars(args)
    train(config)
