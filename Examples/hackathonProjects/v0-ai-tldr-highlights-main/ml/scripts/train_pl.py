#!/usr/bin/env python
"""
PyTorch Lightning Training Script for TL;DR Summarization

Supports all three experiment types with PAI integration:
- baseline: Full model without compression or dendrites
- compressed_pai: Compressed model WITH dendritic optimization
- compressed_control: Compressed model WITHOUT dendrites (ablation)

Usage:
    python -m ml.scripts.train_pl --config ml/configs/base_config.yaml
    python -m ml.scripts.train_pl --experiment-type compressed_pai --dataset samsum
"""

import logging
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.config import ExperimentConfig, configure_experiment
from ml.data.datamodule import TLDRDataModule
from ml.models.lightning_module import TLDRLightningModule, PAITrackerCallback
from ml.models import build_t5_summarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    
    # Parse configuration
    config = ExperimentConfig.from_cli()
    
    logger.info("=" * 60)
    logger.info("TL;DR Summarization Training (PyTorch Lightning)")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_type.value}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"PAI Enabled: {config.use_pai}")
    logger.info("=" * 60)
    
    # Set seed
    pl.seed_everything(config.seed, workers=True)
    
    # Configure PAI globals (BEFORE model creation)
    configure_experiment(config)
    
    # Create data module
    dm = TLDRDataModule(
        model_name=config.model_name,
        dataset_name=config.dataset,
        batch_size=config.batch_size,
        max_train_samples=config.max_train_samples,
        max_eval_samples=config.max_eval_samples,
    )
    
    # Build model
    model, tokenizer = build_t5_summarizer(
        model_name=config.model_name,
        dropout_rate=0.1,
        compression_ratio=config.compression_ratio,
    )
    
    # Create Lightning module
    lightning_module = TLDRLightningModule(
        cfg=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.artifact_dir / 'checkpoints',
            filename='best-{epoch}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=config.save_total_limit,
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=10,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Add PAI callback if enabled
    if config.use_pai:
        callbacks.append(PAITrackerCallback(
            use_pai=True,
            save_graphs=True,
        ))
    
    # Setup logger
    wandb_logger = None
    if config.wandb_enabled:
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.save_name,
            save_dir=str(config.artifact_dir),
            tags=[
                config.experiment_type.value,
                config.dataset,
                config.model_family,
                f"pai_{config.use_pai}",
            ],
        )
        wandb_logger.log_hyperparams(config.to_dict())
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=config.max_steps,
        val_check_interval=config.eval_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=config.max_grad_norm,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=str(config.artifact_dir),
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(lightning_module, dm)
    
    # Test
    if dm.test_dataset:
        logger.info("Running test evaluation...")
        trainer.test(lightning_module, dm)
    
    # Save final config
    config.save()
    
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Artifacts saved to: {config.artifact_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
