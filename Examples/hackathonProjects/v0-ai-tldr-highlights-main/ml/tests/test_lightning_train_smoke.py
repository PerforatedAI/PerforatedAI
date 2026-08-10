"""
Smoke test for Lightning training

Run with: WANDB_MODE=disabled pytest ml/tests/test_lightning_train_smoke.py -v
"""

import os
import pytest
import torch


# Disable W&B for tests
os.environ['WANDB_MODE'] = 'disabled'


def test_lightning_module_forward():
    """Test that Lightning module forward pass works."""
    from ml.config import ExperimentConfig, ExperimentType
    from ml.models.lightning_module import TLDRLightningModule
    from ml.models import build_t5_summarizer
    
    config = ExperimentConfig(
        experiment_type=ExperimentType.BASELINE,
        max_steps=10,
    )
    
    model, tokenizer = build_t5_summarizer('t5-small')
    
    lightning_module = TLDRLightningModule(
        cfg=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, 1000, (2, 64)),
        'attention_mask': torch.ones(2, 64, dtype=torch.long),
        'labels': torch.randint(0, 1000, (2, 32)),
    }
    
    # Forward pass
    loss = lightning_module.training_step(batch, 0)
    
    assert loss is not None
    assert loss.item() > 0


@pytest.mark.slow
def test_lightning_training_smoke():
    """Smoke test: train for 1 step."""
    import pytorch_lightning as pl
    from ml.config import ExperimentConfig, ExperimentType
    from ml.data.datamodule import TLDRDataModule
    from ml.models.lightning_module import TLDRLightningModule
    from ml.models import build_t5_summarizer
    
    config = ExperimentConfig(
        experiment_type=ExperimentType.BASELINE,
        max_steps=2,
        eval_steps=1,
    )
    
    dm = TLDRDataModule.small_synthetic_dataset(size=20)
    model, tokenizer = build_t5_summarizer('t5-small')
    
    lightning_module = TLDRLightningModule(
        cfg=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    trainer = pl.Trainer(
        max_steps=2,
        accelerator='cpu',
        enable_progress_bar=False,
        logger=False,
    )
    
    # Should complete without error
    trainer.fit(lightning_module, dm)
