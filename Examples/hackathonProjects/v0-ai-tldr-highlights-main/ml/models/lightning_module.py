"""
PyTorch Lightning Module for TL;DR Summarization with PAI

Wraps HuggingFace T5/BART models with Lightning training lifecycle
and PerforatedAI dendritic optimization hooks.

Usage:
    from ml.models.lightning_module import TLDRLightningModule
    
    model = TLDRLightningModule(cfg, hf_model, tokenizer)
    trainer.fit(model, datamodule)
"""

import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TLDRLightningModule(pl.LightningModule):
    """
    Lightning Module for summarization with PAI integration.
    
    Handles:
    - Training/validation/test steps
    - Optimizer and scheduler configuration
    - PAI tracker lifecycle (save/load checkpoint hooks)
    - W&B metric logging via self.log()
    """
    
    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        pai_tracker: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'tokenizer', 'pai_tracker'])
        
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.pai_tracker = pai_tracker
        
        # For tracking restructuring events
        self.restructure_count = 0
        self.best_val_score = 0.0
    
    def forward(self, **batch):
        return self.model(**batch)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        val_loss = outputs.loss
        
        self.log('val/loss', val_loss, on_epoch=True, prog_bar=True)
        
        # Compute simple accuracy-like score for PAI feedback
        # (In production, use ROUGE or BERTScore)
        val_score = 1.0 / (1.0 + val_loss.item())
        self.log('val/score', val_score, on_epoch=True)
        
        return {'val_loss': val_loss, 'val_score': val_score}
    
    def on_validation_epoch_end(self):
        """Called at end of validation - feed score to PAI tracker."""
        if not self.pai_tracker:
            return
        
        # Get aggregated validation score
        val_score = self.trainer.callback_metrics.get('val/score', 0.0)
        if isinstance(val_score, torch.Tensor):
            val_score = val_score.item()
        
        try:
            result = self.pai_tracker.add_validation_score(self.model, val_score)
            
            if isinstance(result, tuple) and len(result) >= 4:
                model, improved, restructured, training_complete = result[:4]
                
                if restructured:
                    self.restructure_count += 1
                    self.log('pai/restructure_count', float(self.restructure_count))
                    logger.info(f"PAI restructured model (count: {self.restructure_count})")
                    
                    # Reinitialize optimizer after restructuring
                    self._reinit_optimizer_after_restructure()
                
                if training_complete:
                    logger.info("PAI signaled training complete")
                    self.trainer.should_stop = True
                
                if improved:
                    self.best_val_score = val_score
                    self.log('pai/best_score', self.best_val_score)
        
        except Exception as e:
            logger.warning(f"PAI validation feedback failed: {e}")
    
    def _reinit_optimizer_after_restructure(self):
        """Reinitialize optimizer after PAI restructuring."""
        try:
            if hasattr(self.pai_tracker, 'setup_optimizer'):
                self.pai_tracker.setup_optimizer(
                    torch.optim.AdamW,
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    lr=self.cfg.learning_rate,
                    weight_decay=self.cfg.weight_decay,
                    T_max=self.cfg.max_steps,
                )
                logger.info("Optimizer reinitialized after PAI restructure")
        except Exception as e:
            logger.warning(f"Failed to reinit optimizer: {e}")
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        self.log('test/loss', outputs.loss, on_epoch=True)
        return {'test_loss': outputs.loss}
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.max_steps,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save PAI tracker state with checkpoint."""
        if self.pai_tracker:
            try:
                checkpoint['pai_tracker_state'] = {
                    'restructure_count': self.restructure_count,
                    'best_val_score': self.best_val_score,
                }
                logger.info("PAI state saved to checkpoint")
            except Exception as e:
                logger.warning(f"Failed to save PAI state: {e}")
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore PAI tracker state from checkpoint."""
        if 'pai_tracker_state' in checkpoint:
            state = checkpoint['pai_tracker_state']
            self.restructure_count = state.get('restructure_count', 0)
            self.best_val_score = state.get('best_val_score', 0.0)
            logger.info(f"PAI state restored: restructures={self.restructure_count}")


class PAITrackerCallback(pl.Callback):
    """
    Lightning Callback for PerforatedAI tracker management.
    
    Handles:
    - Model conversion before training
    - Validation score feedback
    - Graph generation at end of training
    """
    
    def __init__(
        self,
        use_pai: bool = True,
        module_names: Optional[List[str]] = None,
        save_graphs: bool = True,
    ):
        self.use_pai = use_pai
        self.module_names = module_names or ['encoder.block', 'decoder.block']
        self.save_graphs = save_graphs
        self.tracker = None
    
    def on_fit_start(self, trainer: pl.Trainer, pl_module: TLDRLightningModule):
        """Initialize PAI tracker and convert model."""
        if not self.use_pai:
            return
        
        try:
            import perforatedai as PA
            from ml.pai_utils import init_pai_tracker
            from ml.models import convert_model_for_pai
            
            # Convert model
            pl_module.model, converted = convert_model_for_pai(
                pl_module.model,
                model_type=pl_module.cfg.model_family,
                modules_to_convert=self.module_names,
            )
            
            # Initialize tracker
            self.tracker = init_pai_tracker(
                save_name=pl_module.cfg.save_name,
                maximizing_score=True,
                make_graphs=self.save_graphs,
            )
            
            pl_module.pai_tracker = self.tracker
            logger.info(f"PAI initialized, converted modules: {converted}")
            
        except ImportError:
            logger.warning("PerforatedAI not installed - running without PAI")
            self.use_pai = False
        except Exception as e:
            logger.error(f"PAI initialization failed: {e}")
            self.use_pai = False
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: TLDRLightningModule):
        """Save PAI graphs at end of training."""
        if self.tracker and self.save_graphs:
            try:
                from ml.pai_utils import dump_pai_graphs
                dump_pai_graphs(self.tracker, pl_module.cfg.artifact_dir / 'pai_graphs')
            except Exception as e:
                logger.warning(f"Failed to save PAI graphs: {e}")
