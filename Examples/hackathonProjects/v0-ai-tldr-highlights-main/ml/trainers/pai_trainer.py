"""
PerforatedAI Trainer with HuggingFace Integration

This module provides a training loop that integrates dendritic optimization
with HuggingFace's Trainer paradigm while following PerforatedAI best practices.

Key features:
- Step-based evaluation (not epoch-based) so PAI controls stopping
- Integrated dendritic feedback after each validation
- Logging of both NLP metrics and dendrite-specific metrics
- Proper handling of model restructuring events

Usage:
    from ml.trainers.pai_trainer import PAIDendriticTrainer
    
    trainer = PAIDendriticTrainer(
        model=model,
        config=pai_config,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    results = trainer.train()
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate

from ml.config.pai_config import PAIConfig, ExperimentType, configure_pai_globals

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    # Standard NLP metrics
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_f1: float = 0.0
    accuracy: float = 0.0
    
    # Dendritic-specific metrics
    total_parameters: int = 0
    active_parameters: int = 0
    dendrite_count: int = 0
    neurite_saturation: float = 0.0
    parameter_reduction_pct: float = 0.0
    
    # Efficiency metrics
    inference_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    effective_flops: int = 0
    
    # Training state
    epoch: int = 0
    step: int = 0
    phase: str = "neurite"  # or "dendrite"
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class PAIDendriticTrainer:
    """
    Trainer for dendritic optimization with HuggingFace-style interface.
    
    This trainer implements the PerforatedAI training paradigm:
    1. Configure PAI globals BEFORE training starts
    2. Initialize tracker with explicit arguments
    3. Use step-based evaluation so PAI controls stopping
    4. Handle model restructuring events properly
    5. Log comprehensive metrics for all experiment types
    
    Args:
        model: The model to train (should already be converted for PAI)
        config: PAIConfig with dendritic optimization settings
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        tokenizer: Tokenizer for decoding predictions
        compute_metrics: Optional custom metrics function
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PAIConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        tokenizer: Any,
        compute_metrics: Optional[Callable] = None,
        output_dir: str = "./outputs",
        wandb_project: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics or self._default_compute_metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.tracker = None
        self.optimizer = None
        self.scheduler = None
        self.best_score = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0
        
        # Metrics
        self.rouge = evaluate.load("rouge")
        self.metrics_history: list = []
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # W&B integration
        self.wandb_project = wandb_project
        self._setup_wandb()
        
        # Initialize PAI
        self._setup_pai()
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    config=self.config.to_dict(),
                    name=self.config.save_name,
                )
                self.wandb = wandb
                logger.info(f"W&B initialized: {self.wandb_project}")
            except ImportError:
                logger.warning("wandb not installed - metrics will not be logged to W&B")
                self.wandb = None
        else:
            self.wandb = None
    
    def _setup_pai(self):
        """Initialize PerforatedAI tracker and optimizer."""
        
        # Step 1: Configure globals
        configure_pai_globals(self.config)
        
        # Step 2: Initialize tracker (only for experiment B)
        if not self.config.do_pb:
            logger.info(f"Experiment {self.config.experiment_type.value}: Standard training (no PAI)")
            self._setup_standard_optimizer()
            return
        
        try:
            import perforatedai as PA
            
            # Initialize tracker with explicit arguments
            # Save name is deterministic from experiment metadata
            self.tracker = PA.PerforatedBackPropagationTracker(
                do_pb=True,
                save_name=self.config.save_name,
                maximizing_score=self.config.maximizing_score,
                make_graphs=self.config.make_graphs,
            )
            
            logger.info(f"PAI tracker initialized: {self.config.save_name}")
            
            # Setup optimizer via tracker
            self.tracker.setup_optimizer(
                torch.optim.AdamW,
                torch.optim.lr_scheduler.CosineAnnealingLR,
                lr=3e-4,
                weight_decay=1e-4,
                T_max=10000,  # Will be controlled by PAI
            )
            
            self.optimizer = self.tracker.optimizer
            logger.info("PAI optimizer configured")
            
        except ImportError:
            logger.warning("perforatedai not installed - falling back to standard training")
            self._setup_standard_optimizer()
    
    def _setup_standard_optimizer(self):
        """Setup standard optimizer for baseline/control experiments."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
        )
    
    def _default_compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        """Default metrics computation using ROUGE."""
        scores = self.rouge.compute(predictions=predictions, references=references)
        return {
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
        }
    
    def train(self, max_steps: int = 100000) -> Dict[str, Any]:
        """
        Main training loop with dendritic optimization.
        
        NOTE: max_steps is set intentionally high because PAI controls stopping,
        not the trainer. Training will exit when:
        1. PAI signals training_complete, or
        2. Early stopping triggers, or
        3. max_steps is reached (safety limit)
        
        Args:
            max_steps: Maximum training steps (safety limit)
            
        Returns:
            Dictionary with final metrics and training statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info(f"  Experiment: {self.config.experiment_type.value}")
        logger.info(f"  Max steps: {max_steps} (PAI controls actual stopping)")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 60)
        
        self.model.to(self.device)
        self.model.train()
        
        train_start = time.time()
        
        # Training loop - PAI controls stopping, not epochs
        while self.global_step < max_steps:
            self.current_epoch += 1
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                loss = self._training_step(batch)
                epoch_loss += loss
                self.global_step += 1
                
                # Evaluation every N steps (step-based, not epoch-based)
                if self.global_step % 500 == 0:
                    should_stop = self._evaluation_step()
                    if should_stop:
                        logger.info("Training complete (PAI or early stopping)")
                        break
                
                if self.global_step >= max_steps:
                    break
            
            # Log epoch summary
            avg_loss = epoch_loss / max(len(self.train_dataloader), 1)
            logger.info(f"Epoch {self.current_epoch}: avg_loss={avg_loss:.4f}, step={self.global_step}")
        
        # Final evaluation
        final_metrics = self._final_evaluation()
        
        train_time = time.time() - train_start
        
        results = {
            "experiment_type": self.config.experiment_type.value,
            "final_metrics": final_metrics.to_dict(),
            "training_time_seconds": train_time,
            "total_steps": self.global_step,
            "total_epochs": self.current_epoch,
            "best_score": self.best_score,
            "config": self.config.to_dict(),
        }
        
        # Save results
        self._save_results(results)
        
        if self.wandb:
            self.wandb.log(results)
            self.wandb.finish()
        
        return results
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        if self.scheduler and not self.tracker:
            self.scheduler.step()
        
        return loss.item()
    
    def _evaluation_step(self) -> bool:
        """
        Evaluation step with dendritic feedback.
        
        Returns:
            True if training should stop, False otherwise
        """
        self.model.eval()
        
        # Compute metrics
        metrics = self._compute_eval_metrics()
        
        # Log metrics
        self._log_metrics(metrics)
        
        # Get primary score for PAI feedback
        primary_score = metrics.rougeL if metrics.rougeL > 0 else metrics.accuracy
        
        # Dendritic feedback (only for experiment B with PAI)
        if self.tracker is not None:
            returned = self.tracker.add_validation_score(self.model, primary_score)
            
            if isinstance(returned, tuple) and len(returned) >= 4:
                self.model, improved, restructured, training_complete = returned[:4]
                
                if restructured:
                    logger.info(f"Step {self.global_step}: Model RESTRUCTURED by dendritic algorithm")
                    # Reinitialize optimizer after restructuring
                    self.tracker.setup_optimizer(
                        torch.optim.AdamW,
                        torch.optim.lr_scheduler.CosineAnnealingLR,
                        lr=3e-4,
                        weight_decay=1e-4,
                        T_max=10000,
                    )
                    self.optimizer = self.tracker.optimizer
                    self.model.to(self.device)
                
                if training_complete:
                    logger.info(f"Step {self.global_step}: PAI signaled TRAINING COMPLETE")
                    return True
        
        # Early stopping check
        if primary_score > self.best_score:
            self.best_score = primary_score
            self.patience_counter = 0
            self._save_checkpoint("best")
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.early_stopping_patience:
            logger.info(f"Early stopping triggered at step {self.global_step}")
            return True
        
        self.model.train()
        return False
    
    def _compute_eval_metrics(self) -> TrainingMetrics:
        """Compute comprehensive evaluation metrics."""
        metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
        )
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Generate predictions
                if hasattr(self.model, "generate"):
                    outputs = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=128,
                        num_beams=4,
                    )
                    decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    predictions.extend(decoded)
                    references.extend(batch.get("labels_text", batch.get("summary", [])))
                else:
                    # Classification model
                    outputs = self.model(**batch)
                    preds = outputs["predictions"] if isinstance(outputs, dict) else outputs.logits.argmax(-1)
                    labels = batch["labels"]
                    metrics.accuracy = (preds == labels).float().mean().item()
        
        # Compute ROUGE if we have predictions
        if predictions and references:
            rouge_scores = self.compute_metrics(predictions, references)
            metrics.rouge1 = rouge_scores.get("rouge1", 0.0)
            metrics.rouge2 = rouge_scores.get("rouge2", 0.0)
            metrics.rougeL = rouge_scores.get("rougeL", 0.0)
        
        # Compute dendritic metrics
        metrics.total_parameters = sum(p.numel() for p in self.model.parameters())
        metrics.active_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Compute inference latency
        metrics.inference_latency_ms = self._measure_inference_latency()
        
        # Memory usage
        if torch.cuda.is_available():
            metrics.memory_usage_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _measure_inference_latency(self, num_samples: int = 10) -> float:
        """Measure average inference latency in milliseconds."""
        self.model.eval()
        latencies = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.eval_dataloader):
                if i >= num_samples:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                start = time.time()
                _ = self.model(**batch)
                latencies.append((time.time() - start) * 1000)
        
        return sum(latencies) / max(len(latencies), 1)
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to console and W&B."""
        logger.info(
            f"Step {metrics.step}: "
            f"ROUGE-L={metrics.rougeL:.4f}, "
            f"Acc={metrics.accuracy:.4f}, "
            f"Params={metrics.total_parameters:,}, "
            f"Latency={metrics.inference_latency_ms:.1f}ms"
        )
        
        if self.wandb:
            self.wandb.log(metrics.to_dict())
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint_{name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_score": self.best_score,
            "config": self.config.to_dict(),
        }, checkpoint_dir / "checkpoint.pt")
        
        logger.info(f"Saved checkpoint: {checkpoint_dir}")
    
    def _final_evaluation(self) -> TrainingMetrics:
        """Run final comprehensive evaluation."""
        logger.info("Running final evaluation...")
        return self._compute_eval_metrics()
    
    def _save_results(self, results: Dict[str, Any]):
        """Save final results to disk."""
        import json
        
        results_path = self.output_dir / f"results_{self.config.save_name}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
