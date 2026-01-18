"""
Central Training Script for PerforatedAI Experiments

This is the main entry point for running PAI experiments. It supports:
- All three experiment types (baseline, compressed_pai, compressed_control)
- CLI and YAML configuration
- W&B logging with comprehensive metrics
- Checkpoint management
- PAI-controlled training termination

Usage:
    python -m ml.train --experiment-type baseline --dataset samsum --wandb enabled
    python -m ml.train --config-file ml/configs/t5_pai.yaml
"""

import logging
import sys
import time
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ml.config import ExperimentConfig, configure_experiment, ExperimentType
from ml.data import load_dataset, preprocess_for_summarization
from ml.models import build_summarizer, convert_model_for_pai
from ml.pai_utils import (
    init_pai_tracker,
    setup_pai_optimizer,
    add_validation_score,
    save_run_metadata,
    dump_pai_graphs,
    apply_safetensors_workaround,
)
from ml.eval import evaluate_checkpoint

from utils.wandb_utils import (
    wandb_init,
    log_metrics,
    log_artifact_checkpoint,
    log_pai_metrics,
    log_highlight_eval_table,
    log_training_curve,
    save_eval_artifacts,
    finish_wandb,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """PyTorch Dataset for summarization."""
    
    def __init__(self, examples, tokenizer, max_source_length=512, max_target_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize source
        source_encoding = self.tokenizer(
            ex.source_text,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            ex.target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


def main():
    """Main training function."""
    
    # Parse config from CLI
    config = ExperimentConfig.from_cli()
    
    logger.info("=" * 60)
    logger.info("PerforatedAI Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_type.value}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"PAI Enabled: {config.use_pai}")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create artifact directory
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.save()
    
    wandb_run = None
    if config.wandb_enabled:
        wandb_run = wandb_init(
            project=config.wandb_project,
            config=config.to_dict(),
            run_name=config.save_name,
            tags=[
                config.experiment_type.value,
                config.dataset,
                config.model_family,
                f"pai_{config.use_pai}",
            ],
            entity=config.wandb_entity,
        )
        if wandb_run:
            logger.info(f"W&B initialized: {wandb_run.url}")
    
    # Step 1: Configure PAI globals (BEFORE model creation)
    pai_configured = configure_experiment(config)
    
    # Step 2: Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    
    # Step 3: Apply compression if needed (experiments B and C)
    if config.use_compression and config.compression_ratio < 1.0:
        logger.info(f"Applying compression ratio: {config.compression_ratio}")
        # In practice, you'd reduce layers/dimensions here
    
    # Step 4: Convert model for PAI (experiment B only)
    converted_modules = []
    if config.use_pai:
        model, converted_modules = convert_model_for_pai(model, model_type=config.model_family)
        apply_safetensors_workaround(enabled=True)
    
    model.to(device)
    
    if wandb_run:
        initial_params = sum(p.numel() for p in model.parameters())
        log_metrics(0, {"model/initial_params": initial_params}, wandb_run)
    
    # Step 5: Initialize PAI tracker (experiment B only)
    tracker = None
    if config.use_pai:
        tracker = init_pai_tracker(
            save_name=config.save_name,
            maximizing_score=True,
            make_graphs=True,
            do_pb=True,
        )
    
    # Step 6: Setup optimizer
    if tracker:
        optimizer = setup_pai_optimizer(
            tracker, model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            T_max=config.max_steps,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    # Step 7: Load datasets
    logger.info("Loading datasets...")
    train_examples = list(preprocess_for_summarization(
        load_dataset(config.dataset, "train", max_samples=config.max_train_samples)
    ))
    eval_examples = list(preprocess_for_summarization(
        load_dataset(config.dataset, "validation", max_samples=config.max_eval_samples)
    ))
    
    train_dataset = SummarizationDataset(train_examples, tokenizer)
    eval_dataset = SummarizationDataset(eval_examples, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size)
    
    logger.info(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
    
    # Step 8: Training loop
    logger.info("Starting training...")
    global_step = 0
    best_score = 0.0
    patience_counter = 0
    train_start = time.time()
    training_complete = False
    
    training_history = []
    
    while global_step < config.max_steps:
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            if wandb_run and global_step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log_metrics(global_step, {
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                }, wandb_run)
            
            # Evaluation every N steps
            if global_step % config.eval_steps == 0:
                eval_score = evaluate_model(model, eval_loader, tokenizer, device)
                avg_loss = epoch_loss / max(batch_count, 1)
                
                logger.info(f"Step {global_step}: eval_score={eval_score:.4f}, avg_loss={avg_loss:.4f}")
                
                if wandb_run:
                    eval_metrics = {
                        "eval/score": eval_score,
                        "eval/avg_loss": avg_loss,
                        "train/epoch_loss": epoch_loss,
                    }
                    log_metrics(global_step, eval_metrics, wandb_run)
                    
                    # Log PAI-specific metrics
                    log_pai_metrics(wandb_run, model, tracker, global_step)
                
                # Track history for curves
                training_history.append({
                    "step": global_step,
                    "eval_score": eval_score,
                    "loss": avg_loss,
                })
                
                # PAI feedback
                if tracker:
                    model, improved, restructured, training_complete = add_validation_score(
                        tracker, model, eval_score
                    )
                    
                    if wandb_run and restructured:
                        log_metrics(global_step, {"pai/restructured": 1}, wandb_run)
                    
                    if restructured:
                        logger.info("Reinitializing optimizer after restructuring")
                        optimizer = setup_pai_optimizer(
                            tracker, model,
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                        )
                        model.to(device)
                    
                    if training_complete:
                        logger.info("PAI signaled training complete")
                        break
                
                # Early stopping
                if eval_score > best_score:
                    best_score = eval_score
                    patience_counter = 0
                    ckpt_path = save_checkpoint(model, optimizer, global_step, config)
                    
                    if wandb_run:
                        log_artifact_checkpoint(
                            wandb_run,
                            ckpt_path,
                            f"best-{config.save_name}",
                            metadata={"step": global_step, "score": eval_score},
                        )
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    logger.info("Early stopping triggered")
                    break
                
                epoch_loss = 0.0
                batch_count = 0
            
            if global_step >= config.max_steps:
                break
        
        if patience_counter >= 10 or training_complete:
            break
    
    # Step 9: Final evaluation and cleanup
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.1f}s")
    logger.info(f"Best score: {best_score:.4f}")
    
    # Save final checkpoint
    final_ckpt_path = save_checkpoint(model, optimizer, global_step, config, name="final")
    
    if tracker:
        dump_pai_graphs(tracker, config.artifact_dir / "pai_graphs")
    
    if wandb_run:
        # Log training curve
        log_training_curve(wandb_run, training_history)
        
        # Log final checkpoint
        log_artifact_checkpoint(
            wandb_run,
            final_ckpt_path,
            f"final-{config.save_name}",
            metadata={"total_steps": global_step, "best_score": best_score},
        )
        
        # Log evaluation summary
        save_eval_artifacts(wandb_run, {
            "best_score": best_score,
            "total_steps": global_step,
            "train_time_seconds": train_time,
            "experiment_type": config.experiment_type.value,
            "dataset": config.dataset,
            "model_name": config.model_name,
            "use_pai": config.use_pai,
        })
        
        # Log final summary metrics
        log_metrics(global_step, {
            "final/best_score": best_score,
            "final/train_time_seconds": train_time,
            "final/total_steps": global_step,
        }, wandb_run)
        
        finish_wandb(wandb_run)
    
    save_run_metadata(
        config, config.artifact_dir,
        converted_modules=converted_modules,
        wandb_run_id=wandb_run.id if wandb_run else None,
    )
    
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Artifacts saved to: {config.artifact_dir}")
    logger.info("=" * 60)


def evaluate_model(model, eval_loader, tokenizer, device) -> float:
    """Quick evaluation returning a single score."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / max(len(eval_loader), 1)
    return 1.0 / (1.0 + avg_loss)


def save_checkpoint(model, optimizer, step, config, name="checkpoint") -> str:
    """Save model checkpoint and return path."""
    path = config.artifact_dir / f"{name}_step{step}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
    }, path)
    logger.info(f"Saved: {path}")
    return str(path)


if __name__ == "__main__":
    main()
