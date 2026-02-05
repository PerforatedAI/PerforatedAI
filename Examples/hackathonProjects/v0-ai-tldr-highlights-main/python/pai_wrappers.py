"""
PerforatedAI Module Wrappers for T5 and SBERT Models

This module provides helper functions to wrap transformer layers so they're
compatible with PerforatedAI dendritic optimization. The wrappers ensure
LayerNorms remain internal to converted modules.
"""

import logging
import torch.nn as nn
from typing import Any

logger = logging.getLogger(__name__)


class _ModuleWrapper(nn.Module):
    """
    Generic wrapper that holds a reference to the original module and forwards calls.
    Wrapping the original block as a single module preserves LayerNorms being internal
    to the module for PerforatedAI conversion.
    """
    def __init__(self, orig_module: nn.Module):
        super().__init__()
        self.orig = orig_module

    def forward(self, *args, **kwargs):
        return self.orig(*args, **kwargs)


def wrap_t5_layers_for_pai(model: nn.Module) -> nn.Module:
    """
    Wrap each encoder and decoder block of a T5ForConditionalGeneration-like model
    so LayerNorms and submodules remain inside a single module for conversion.
    
    Args:
        model: T5ForConditionalGeneration model from transformers
        
    Returns:
        Modified model with wrapped layers
    """
    wrapped = False
    
    # Encoder blocks
    try:
        if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
            for i, blk in enumerate(model.encoder.block):
                if isinstance(blk, _ModuleWrapper):
                    continue
                model.encoder.block[i] = _ModuleWrapper(blk)
            wrapped = True
            logger.info("Wrapped T5 encoder.block modules for PAI.")
    except Exception as e:
        logger.warning("Could not wrap model.encoder.block: %s", e)

    # Decoder blocks
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            for i, blk in enumerate(model.decoder.block):
                if isinstance(blk, _ModuleWrapper):
                    continue
                model.decoder.block[i] = _ModuleWrapper(blk)
            wrapped = True
            logger.info("Wrapped T5 decoder.block modules for PAI.")
    except Exception as e:
        logger.warning("Could not wrap model.decoder.block: %s", e)

    if not wrapped:
        logger.warning("wrap_t5_layers_for_pai: no encoder.block/decoder.block found. Model unchanged.")
    
    return model


def wrap_sbert_layers_for_pai(model: nn.Module) -> nn.Module:
    """
    Wrap Transformer encoder layers for SBERT-like models (BERT backbone) so layernorms
    live inside the converted module.
    
    Args:
        model: BERT-based model (BertModel, SentenceTransformer, etc.)
        
    Returns:
        Modified model with wrapped layers
    """
    wrapped_any = False

    # Try common attribute paths
    paths_to_try = [
        ("bert.encoder.layer", lambda m: m.bert.encoder.layer if hasattr(m, "bert") else None),
        ("base_model.encoder.layer", lambda m: m.base_model.encoder.layer if hasattr(m, "base_model") and hasattr(m.base_model, "encoder") else None),
        ("auto_model.encoder.layer", lambda m: m.auto_model.encoder.layer if hasattr(m, "auto_model") and hasattr(m.auto_model, "encoder") else None),
        ("encoder.layer", lambda m: m.encoder.layer if hasattr(m, "encoder") and hasattr(m.encoder, "layer") else None),
    ]

    for path_name, accessor in paths_to_try:
        try:
            layers = accessor(model)
            if layers is None:
                continue
                
            # Wrap each layer
            for i, blk in enumerate(layers):
                if isinstance(blk, _ModuleWrapper):
                    continue
                layers[i] = _ModuleWrapper(blk)
            
            wrapped_any = True
            logger.info(f"Wrapped layers at {path_name} for PAI.")
            break
            
        except Exception as e:
            logger.debug(f"Could not wrap {path_name}: {e}")

    if not wrapped_any:
        logger.warning("wrap_sbert_layers_for_pai: no encoder layers found. Model unchanged.")
    
    return model


def wrap_bart_layers_for_pai(model: nn.Module) -> nn.Module:
    """
    Wrap BART encoder/decoder layers for dendritic optimization.
    
    Args:
        model: BartForConditionalGeneration model
        
    Returns:
        Modified model with wrapped layers
    """
    wrapped = False
    
    # Encoder layers
    try:
        if hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model.encoder, "layers"):
            for i, layer in enumerate(model.model.encoder.layers):
                if isinstance(layer, _ModuleWrapper):
                    continue
                model.model.encoder.layers[i] = _ModuleWrapper(layer)
            wrapped = True
            logger.info("Wrapped BART encoder layers for PAI.")
    except Exception as e:
        logger.warning(f"Could not wrap BART encoder layers: {e}")
    
    # Decoder layers
    try:
        if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            for i, layer in enumerate(model.model.decoder.layers):
                if isinstance(layer, _ModuleWrapper):
                    continue
                model.model.decoder.layers[i] = _ModuleWrapper(layer)
            wrapped = True
            logger.info("Wrapped BART decoder layers for PAI.")
    except Exception as e:
        logger.warning(f"Could not wrap BART decoder layers: {e}")
    
    if not wrapped:
        logger.warning("wrap_bart_layers_for_pai: no layers found. Model unchanged.")
    
    return model
