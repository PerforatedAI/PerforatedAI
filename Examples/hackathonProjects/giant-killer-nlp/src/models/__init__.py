"""
Models module for Giant-Killer NLP project.
Contains BERT-Tiny model with dendritic optimization wrapper.
"""

from .bert_tiny import (
    create_bert_tiny_model,
    create_bert_base_model,
    wrap_with_dendrites,
    ToxicityClassifier,
)

__all__ = [
    "create_bert_tiny_model",
    "create_bert_base_model",
    "wrap_with_dendrites",
    "ToxicityClassifier",
]
