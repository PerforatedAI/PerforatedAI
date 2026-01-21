"""
ML Training Module for AI TL;DR + Smart Highlights

This module provides research-grade training infrastructure for
PerforatedAI dendritic optimization of NLP models.

Quick Start:
    from ml.config.pai_config import PAIConfig, configure_pai_globals
    from ml.models.pai_converter import convert_model_for_pai
    from ml.trainers.pai_trainer import PAIDendriticTrainer
    
    config = PAIConfig(experiment_type="B")
    configure_pai_globals(config)
    
    model = convert_model_for_pai(model, "t5")
    trainer = PAIDendriticTrainer(model, config, ...)
    results = trainer.train()
"""

from ml.config.pai_config import PAIConfig, ExperimentType, configure_pai_globals
from ml.models.pai_converter import convert_model_for_pai

__all__ = [
    "PAIConfig",
    "ExperimentType", 
    "configure_pai_globals",
    "convert_model_for_pai",
]
