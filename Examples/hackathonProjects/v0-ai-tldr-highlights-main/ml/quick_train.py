#!/usr/bin/env python3
"""
Quick Training Script - Minimal working example

This script runs a minimal training loop to verify W&B, HF, and PAI integration.
It trains for just a few steps and logs to W&B.

Usage:
    python -m ml.quick_train
    
Environment Variables Required:
    WANDB_API_KEY: Your Weights & Biases API key
    HF_TOKEN: (Optional) HuggingFace token for gated models
"""

import os
import sys
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Quick Training - W&B + HuggingFace + PAI Integration Test")
    logger.info("=" * 60)
    
    # Check environment
    wandb_key = os.environ.get('WANDB_API_KEY')
    if not wandb_key:
        logger.error("WANDB_API_KEY not set!")
        logger.info("Set it with: export WANDB_API_KEY=your_key")
        return 1
    
    # Import dependencies
    import wandb
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from datasets import load_dataset
    
    # Try importing PAI (optional)
    pai_available = False
    try:
        import perforatedai as PA
        from perforatedai import pb_globals as PBG
        pai_available = True
        logger.info("[OK] PerforatedAI available")
    except ImportError:
        logger.warning("[WARN] PerforatedAI not installed - running without dendrites")
    
    # Initialize W&B
    logger.info("Initializing Weights & Biases...")
    wandb.login(key=wandb_key)
    
    run = wandb.init(
        project="tldr-highlights",
        name=f"quick-train-{int(time.time())}",
        config={
            "model": "t5-small",
            "dataset": "samsum",
            "batch_size": 4,
            "learning_rate": 3e-4,
            "max_steps": 50,
            "pai_enabled": pai_available,
        },
        tags=["quick-train", "verification"],
    )
    logger.info(f"[OK] W&B run: {run.url}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    wandb.config.update({"device": str(device)})
    
    # Load model and tokenizer
    logger.info("Loading T5-small model...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # Convert for PAI if available
    if pai_available:
        logger.info("Converting model for PerforatedAI...")
        try:
            # Configure PAI globals
            PBG.switchMode = PBG.SwitchMode.EPOCH_LAYERWISE
            PBG.nEpochsToSwitch = 5
            PBG.testingDendriteCapacity = 8
            
            # Convert model
            PA.convert_network(model, module_names_to_convert=["encoder.block", "decoder.block"])
            logger.info("[OK] Model converted for PAI")
            wandb.config.update({"pai_converted": True})
            
            # Initialize tracker
            tracker = PA.PerforatedBackPropagationTracker(
                do_pb=True,
                save_name="quick_train",
                maximizing_score=True,
            )
            logger.info("[OK] PAI tracker initialized")
            
        except Exception as e:
            logger.warning(f"PAI conversion failed: {e}")
            pai_available = False
            tracker = None
    else:
        tracker = None
    
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
    })
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup optimizer
    if tracker:
        try:
            tracker.setup_optimizer(
                torch.optim.AdamW,
                torch.optim.lr_scheduler.CosineAnnealingLR,
                lr=3e-4,
                weight_decay=1e-4,
                T_max=50,
            )
            optimizer = tracker.optimizer
            logger.info("[OK] PAI optimizer configured")
        except Exception as e:
            logger.warning(f"PAI optimizer setup failed: {e}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Load dataset
    logger.info("Loading SAMSum dataset...")
    ds = load_dataset("samsum", split="train[:100]", trust_remote_code=True)
    logger.info(f"[OK] Loaded {len(ds)} examples")
    
    # Create simple dataset
    class SimpleDataset(Dataset):
        def __init__(self, examples, tokenizer):
            self.examples = examples
            self.tokenizer = tokenizer
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            ex = self.examples[idx]
            
            source = f"summarize: {ex['dialogue']}"
            target = ex['summary']
            
            source_enc = self.tokenizer(
                source, max_length=256, truncation=True, padding="max_length", return_tensors="pt"
            )
            target_enc = self.tokenizer(
                target, max_length=64, truncation=True, padding="max_length", return_tensors="pt"
            )
            
            return {
                "input_ids": source_enc["input_ids"].squeeze(),
                "attention_mask": source_enc["attention_mask"].squeeze(),
                "labels": target_enc["input_ids"].squeeze(),
            }
    
    train_ds = SimpleDataset(ds, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    global_step = 0
    max_steps = 50
    eval_every = 10
    
    for batch in train_loader:
        if global_step >= max_steps:
            break
        
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        global_step += 1
        
        # Log to W&B
        wandb.log({
            "train/loss": loss.item(),
            "train/step": global_step,
            "train/lr": optimizer.param_groups[0]['lr'],
        }, step=global_step)
        
        if global_step % 10 == 0:
            logger.info(f"Step {global_step}/{max_steps}, Loss: {loss.item():.4f}")
        
        # Evaluation
        if global_step % eval_every == 0:
            model.eval()
            
            # Generate sample
            with torch.no_grad():
                sample_input = tokenizer(
                    "summarize: Hello, how are you today? I am doing well, thank you for asking.",
                    return_tensors="pt",
                    max_length=64,
                    truncation=True,
                ).to(device)
                
                generated = model.generate(
                    **sample_input,
                    max_length=32,
                    num_beams=2,
                )
                decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            logger.info(f"Sample generation: '{decoded}'")
            
            # Simple eval score (inverse loss)
            eval_score = 1.0 / (1.0 + loss.item())
            
            wandb.log({
                "eval/score": eval_score,
                "eval/sample": decoded,
            }, step=global_step)
            
            # PAI feedback
            if tracker:
                try:
                    result = tracker.add_validation_score(model, eval_score)
                    if isinstance(result, tuple) and len(result) >= 4:
                        model, improved, restructured, training_complete = result[:4]
                        
                        if restructured:
                            logger.info("[PAI] Model restructured!")
                            wandb.log({"pai/restructured": 1}, step=global_step)
                            
                            # Reinit optimizer
                            tracker.setup_optimizer(
                                torch.optim.AdamW,
                                torch.optim.lr_scheduler.CosineAnnealingLR,
                                lr=3e-4,
                                weight_decay=1e-4,
                                T_max=max_steps - global_step,
                            )
                            optimizer = tracker.optimizer
                            model.to(device)
                        
                        if training_complete:
                            logger.info("[PAI] Training complete signal received!")
                            break
                except Exception as e:
                    logger.warning(f"PAI feedback error: {e}")
            
            model.train()
    
    # Final logging
    logger.info("Training complete!")
    
    # Log final metrics
    wandb.log({
        "final/steps": global_step,
        "final/loss": loss.item(),
    })
    
    # Save model artifact
    artifact = wandb.Artifact(
        name="quick-train-model",
        type="model",
        metadata={
            "steps": global_step,
            "pai_enabled": pai_available,
        }
    )
    
    # Save checkpoint
    checkpoint_path = "quick_train_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": global_step,
    }, checkpoint_path)
    
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)
    logger.info(f"[OK] Model artifact logged to W&B")
    
    # Cleanup
    os.remove(checkpoint_path)
    
    # Finish W&B run
    run.finish()
    
    logger.info("=" * 60)
    logger.info("Quick training completed successfully!")
    logger.info(f"View results at: {run.url}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
