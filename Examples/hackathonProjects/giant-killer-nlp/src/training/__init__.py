"""
Training module for Giant-Killer NLP project.
Implements the Perforated training loop with PAI tracker.
"""

from .trainer import (
    PerforatedTrainer,
    train_epoch,
    validate,
    train_model,
)

__all__ = [
    "PerforatedTrainer",
    "train_epoch",
    "validate",
    "train_model",
]
