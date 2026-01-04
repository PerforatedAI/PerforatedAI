"""
PROJECT NEXUS: Dendritic SBERT Training Engine (No W&B version)
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

def train(args):
    set_seed(42)
    
    mode = "DENDRITIC" if args.use_dendrites else "BASELINE"
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
    if args.use_dendrites:
        print("⚡ Injecting Dendrites into Adapter Layer ONLY...")
        
        # Safety Flags
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_testing_dendrite_capacity(False)  # Disable test mode for production
        GPA.pc.set_n_epochs_to_switch(10)  # 🔥 Extended tracking: wait 10 epochs before switching
        
        # Initialize PAI on ONLY the adapter layer (model[2])
        model[2] = UPA.initialize_pai(
            model[2], 
            save_name="PAI" 
        )
        
        print("✅ PAI Settings: n_epochs_to_switch=10 (extended tracking for fuller graph)")
        
        # Optimizer - use model.parameters() to train all layers
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Register optimizer with PAI tracker
        GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
        
    else:
        print("⚠️ Running in BASELINE mode.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Training Loop
    train_examples, evaluator = load_data()
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=args.batch_size, 
        collate_fn=model.smart_batching_collate
    )
    train_loss = losses.CosineSimilarityLoss(model)
    
    print(f"🎯 Training Started...")
    epoch = -1
    
    # Track metrics
    metrics_log = []
    
    while True:
        epoch += 1
        if epoch >= args.epochs:
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
        if args.use_dendrites:
            GPA.pai_tracker.add_extra_score(avg_loss, 'Train Loss')
        
        print(f"📊 Epoch {epoch} | Loss: {avg_loss:.4f} | Spearman: {score:.4f}")
        
        # Log metrics
        metrics_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_spearman": score
        })

        # Dendritic Evolution Logic
        if args.use_dendrites:
            model[2], restructured, training_complete = GPA.pai_tracker.add_validation_score(score * 100, model[2])
            
            if restructured:
                print(">>> ⚡ DENDRITES ACTIVATED! Architecture Evolved. ⚡ <<<")
                # Recreate optimizer with updated parameters
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # Re-register optimizer with PAI tracker
                GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
                
            if training_complete:
                print("🏆 Training Complete per PAI.")
                break
        
        # Save Checkpoints
        if epoch > 0:
            save_path = f"{args.save_dir}/checkpoint_epoch_{epoch}"
            model.save(save_path)
    
    # Final save
    final_path = f"{args.save_dir}/final_model"
    model.save(final_path)
    print(f"✅ Model saved to {final_path}")
    
    # Save metrics
    import json
    with open(f"{args.save_dir}/metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"✅ Metrics saved to {args.save_dir}/metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project NEXUS Training")
    parser.add_argument("--use_dendrites", action="store_true", help="Enable Dendritic Optimization")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="../experiments/default", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
