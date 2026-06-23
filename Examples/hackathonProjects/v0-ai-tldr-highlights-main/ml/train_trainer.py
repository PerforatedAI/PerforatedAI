"""
HuggingFace Trainer Wrapper with PerforatedAI Integration

This module provides a Trainer-compatible PAI integration where:
- num_train_epochs is set very high (letting PAI control termination)
- Evaluation strategy is step-based for proper PAI feedback
- Optimizer/scheduler are managed through PAI tracker

Usage:
    python -m ml.train_trainer --experiment-type compressed_pai --dataset samsum
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainerCallback,
)
from datasets import load_dataset as hf_load_dataset

from ml.config import ExperimentConfig, configure_experiment, ExperimentType
from ml.models import convert_model_for_pai
from ml.pai_utils import (
    init_pai_tracker,
    add_validation_score,
    dump_pai_graphs,
    save_run_metadata,
    apply_safetensors_workaround,
)
from utils.wandb_utils import wandb_init, finish_wandb

logger = logging.getLogger(__name__)


class PAICallback(TrainerCallback):
    """
    Callback for PerforatedAI integration with HuggingFace Trainer.
    
    Handles:
    - Passing validation scores to PAI tracker
    - Detecting restructuring events
    - Triggering early stopping when PAI signals completion
    """
    
    def __init__(self, tracker, config: ExperimentConfig):
        self.tracker = tracker
        self.config = config
        self.training_complete = False
        self.restructured_count = 0
    
    def on_evaluate(self, args, state, control, metrics, model, **kwargs):
        """Called after evaluation - feed score to PAI tracker."""
        if self.tracker is None:
            return
        
        # Get validation score (use eval_loss as proxy, inverted)
        eval_loss = metrics.get("eval_loss", 1.0)
        score = 1.0 / (1.0 + eval_loss)  # Higher is better
        
        logger.info(f"PAI Callback: eval_loss={eval_loss:.4f}, score={score:.4f}")
        
        # Feed to tracker
        model, improved, restructured, training_complete = add_validation_score(
            self.tracker, model, score
        )
        
        if restructured:
            self.restructured_count += 1
            logger.info(f"Model restructured (count: {self.restructured_count})")
        
        if training_complete:
            logger.info("PAI signaled training complete")
            self.training_complete = True
            control.should_training_stop = True
        
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Check if PAI wants to stop training."""
        if self.training_complete:
            control.should_training_stop = True
        return control


def run_pai_trainer(config: ExperimentConfig):
    """
    Run training using HuggingFace Trainer with PAI integration.
    
    Args:
        config: ExperimentConfig with experiment settings
    """
    logger.info("=" * 60)
    logger.info("PAI HuggingFace Trainer")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_type.value}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"PAI: {config.use_pai}")
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Configure PAI globals BEFORE model creation
    pai_configured = configure_experiment(config)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    
    # Convert for PAI if enabled
    tracker = None
    converted_modules = []
    
    if config.use_pai:
        model, converted_modules = convert_model_for_pai(model, config.model_family)
        apply_safetensors_workaround(enabled=True)
        
        tracker = init_pai_tracker(
            save_name=config.save_name,
            maximizing_score=True,
            make_graphs=True,
        )
    
    # Load dataset
    if config.dataset == "samsum":
        dataset = hf_load_dataset("samsum")
        
        def preprocess(examples):
            inputs = ["summarize: " + d for d in examples["dialogue"]]
            model_inputs = tokenizer(
                inputs, max_length=512, truncation=True, padding="max_length"
            )
            labels = tokenizer(
                examples["summary"], max_length=128, truncation=True, padding="max_length"
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
        train_dataset = tokenized["train"]
        eval_dataset = tokenized["validation"]
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    
    # Create artifact directory
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments - PAI controls stopping, so set high epochs
    training_args = TrainingArguments(
        output_dir=str(config.artifact_dir),
        num_train_epochs=1000000.0,  # PAI controls actual termination
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps",  # Step-based for PAI feedback
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config.wandb_enabled else "none",
        run_name=config.save_name,
        seed=config.seed,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Create trainer with PAI callback
    callbacks = []
    if tracker:
        callbacks.append(PAICallback(tracker, config))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Initialize W&B
    wandb_run = None
    if config.wandb_enabled:
        wandb_run = wandb_init(
            project=config.wandb_project,
            config=config.to_dict(),
            run_name=config.save_name,
            entity=config.wandb_entity,
        )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(str(config.artifact_dir / "final_model"))
    
    # Dump PAI graphs
    if tracker:
        dump_pai_graphs(tracker, config.artifact_dir / "pai_graphs")
    
    # Save metadata
    save_run_metadata(
        config, config.artifact_dir,
        converted_modules=converted_modules,
        wandb_run_id=wandb_run.id if wandb_run else None,
    )
    
    if wandb_run:
        finish_wandb(wandb_run)
    
    logger.info("Training complete!")
    logger.info(f"Artifacts saved to: {config.artifact_dir}")


def main():
    config = ExperimentConfig.from_cli()
    run_pai_trainer(config)


if __name__ == "__main__":
    main()
