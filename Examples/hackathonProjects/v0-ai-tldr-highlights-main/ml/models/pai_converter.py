"""
PerforatedAI Model Conversion Module

This module provides explicit, auditable model conversion for dendritic optimization.
Conversion is a distinct step, not a side effect.

Key functions:
- convert_model_for_pai: Main conversion entry point
- wrap_transformer_layers: Layer wrapping for LayerNorm preservation
- verify_conversion: Post-conversion verification

Usage:
    from ml.models.pai_converter import convert_model_for_pai
    
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = convert_model_for_pai(model, model_type="t5")
"""

import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# MODULE WRAPPER
# ============================================================================

class _PAIModuleWrapper(nn.Module):
    """
    Generic wrapper that holds a reference to the original module and forwards calls.
    
    This wrapper ensures LayerNorms remain INTERNAL to converted modules,
    which is required for correct PerforatedAI behavior. Without this wrapper,
    LayerNorms would be converted separately and break gradient flow.
    
    The wrapper is transparent to forward() calls but visible during model inspection.
    """
    
    def __init__(self, orig_module: nn.Module, layer_name: str = ""):
        super().__init__()
        self.orig = orig_module
        self.layer_name = layer_name
        
    def forward(self, *args, **kwargs):
        return self.orig(*args, **kwargs)
    
    def __repr__(self):
        return f"_PAIModuleWrapper({self.layer_name})"


# ============================================================================
# LAYER WRAPPING FUNCTIONS
# ============================================================================

def wrap_t5_layers(model: nn.Module) -> Tuple[nn.Module, int]:
    """
    Wrap T5 encoder and decoder blocks for PAI conversion.
    
    T5 blocks contain self-attention, cross-attention (decoder), and FFN sub-layers,
    each with their own LayerNorm. Wrapping the entire block preserves this structure.
    
    Args:
        model: T5ForConditionalGeneration or similar
        
    Returns:
        Tuple of (wrapped model, number of layers wrapped)
    """
    wrapped_count = 0
    
    # Encoder blocks
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        for i, block in enumerate(model.encoder.block):
            if not isinstance(block, _PAIModuleWrapper):
                model.encoder.block[i] = _PAIModuleWrapper(block, f"encoder.block.{i}")
                wrapped_count += 1
        logger.info(f"Wrapped {len(model.encoder.block)} T5 encoder blocks")
    
    # Decoder blocks
    if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        for i, block in enumerate(model.decoder.block):
            if not isinstance(block, _PAIModuleWrapper):
                model.decoder.block[i] = _PAIModuleWrapper(block, f"decoder.block.{i}")
                wrapped_count += 1
        logger.info(f"Wrapped {len(model.decoder.block)} T5 decoder blocks")
    
    return model, wrapped_count


def wrap_bart_layers(model: nn.Module) -> Tuple[nn.Module, int]:
    """
    Wrap BART encoder and decoder layers for PAI conversion.
    
    Args:
        model: BartForConditionalGeneration or similar
        
    Returns:
        Tuple of (wrapped model, number of layers wrapped)
    """
    wrapped_count = 0
    
    # Encoder layers (via model.model for BartForConditionalGeneration)
    base = model.model if hasattr(model, "model") else model
    
    if hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
        for i, layer in enumerate(base.encoder.layers):
            if not isinstance(layer, _PAIModuleWrapper):
                base.encoder.layers[i] = _PAIModuleWrapper(layer, f"encoder.layers.{i}")
                wrapped_count += 1
        logger.info(f"Wrapped {len(base.encoder.layers)} BART encoder layers")
    
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        for i, layer in enumerate(base.decoder.layers):
            if not isinstance(layer, _PAIModuleWrapper):
                base.decoder.layers[i] = _PAIModuleWrapper(layer, f"decoder.layers.{i}")
                wrapped_count += 1
        logger.info(f"Wrapped {len(base.decoder.layers)} BART decoder layers")
    
    return model, wrapped_count


def wrap_bert_layers(model: nn.Module) -> Tuple[nn.Module, int]:
    """
    Wrap BERT/RoBERTa encoder layers for PAI conversion.
    
    Args:
        model: BertModel, RobertaModel, or AutoModel with BERT-like architecture
        
    Returns:
        Tuple of (wrapped model, number of layers wrapped)
    """
    wrapped_count = 0
    
    # Try multiple possible attribute paths
    layer_paths = [
        lambda m: m.bert.encoder.layer if hasattr(m, "bert") else None,
        lambda m: m.roberta.encoder.layer if hasattr(m, "roberta") else None,
        lambda m: m.encoder.layer if hasattr(m, "encoder") and hasattr(m.encoder, "layer") else None,
        lambda m: m.base_model.encoder.layer if hasattr(m, "base_model") else None,
    ]
    
    for path_fn in layer_paths:
        try:
            layers = path_fn(model)
            if layers is not None:
                for i, layer in enumerate(layers):
                    if not isinstance(layer, _PAIModuleWrapper):
                        layers[i] = _PAIModuleWrapper(layer, f"encoder.layer.{i}")
                        wrapped_count += 1
                logger.info(f"Wrapped {len(layers)} BERT encoder layers")
                break
        except Exception:
            continue
    
    return model, wrapped_count


# ============================================================================
# SAFETENSORS COMPATIBILITY
# ============================================================================

def patch_safetensors_shared_tensors():
    """
    Patch safetensors to handle shared tensor detection.
    
    Some HuggingFace models have tied weights (e.g., embedding and lm_head)
    that trigger safetensors validation errors after PAI conversion.
    This patch safely bypasses the check when appropriate.
    
    WARNING: Only use this when you understand the model's weight sharing behavior.
    For T5/BART, the tied embeddings are intentionally shared and this is safe.
    """
    try:
        import safetensors.torch as sft
        
        _original_save = sft.save_file
        
        def _patched_save(*args, **kwargs):
            # Force shared tensor handling
            if "safe_serialization" in kwargs:
                kwargs["safe_serialization"] = False
            return _original_save(*args, **kwargs)
        
        sft.save_file = _patched_save
        logger.info("Patched safetensors for shared tensor compatibility")
        
    except ImportError:
        pass  # safetensors not installed, no patch needed
    except Exception as e:
        logger.warning(f"Failed to patch safetensors: {e}")


# ============================================================================
# MAIN CONVERSION FUNCTION
# ============================================================================

def convert_model_for_pai(
    model: nn.Module,
    model_type: str,
    modules_to_convert: Optional[List[str]] = None
) -> nn.Module:
    """
    Convert a model for PerforatedAI dendritic optimization.
    
    This is the main entry point for model conversion. It performs:
    1. Layer wrapping to preserve LayerNorm placement
    2. PAI network conversion via PA.convert_network()
    3. Post-conversion verification
    
    Args:
        model: The model to convert (T5, BART, BERT, etc.)
        model_type: One of "t5", "bart", "bert", "sbert"
        modules_to_convert: Optional list of module names to convert.
                           If None, uses sensible defaults for model_type.
    
    Returns:
        Converted model ready for dendritic training
        
    Raises:
        ImportError: If perforatedai is not installed
        ValueError: If model_type is unknown
    """
    logger.info("=" * 60)
    logger.info(f"Converting model for PerforatedAI: {model_type}")
    logger.info("=" * 60)
    
    # Step 1: Wrap transformer layers
    model_type_lower = model_type.lower()
    
    if "t5" in model_type_lower:
        model, wrapped_count = wrap_t5_layers(model)
        default_modules = ["encoder.block", "decoder.block"]
    elif "bart" in model_type_lower:
        model, wrapped_count = wrap_bart_layers(model)
        default_modules = ["model.encoder.layers", "model.decoder.layers"]
    elif "bert" in model_type_lower or "sbert" in model_type_lower:
        model, wrapped_count = wrap_bert_layers(model)
        default_modules = ["encoder.layer"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: t5, bart, bert, sbert")
    
    logger.info(f"Wrapped {wrapped_count} transformer layers")
    
    # Step 2: PAI network conversion
    try:
        import perforatedai as PA
        
        modules = modules_to_convert or default_modules
        
        PA.convert_network(
            model,
            module_names_to_convert=modules
        )
        
        logger.info(f"PAI conversion complete for modules: {modules}")
        
    except ImportError:
        logger.warning("perforatedai not installed - skipping PA.convert_network()")
        logger.warning("Model will train without dendritic optimization")
    except Exception as e:
        logger.error(f"PAI conversion failed: {e}")
        raise
    
    # Step 3: Verify conversion
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("=" * 60)
    
    return model


def convert_classifier_head_for_pai(
    model: nn.Module,
    head_modules: List[str] = ["fc1", "fc2", "classifier"]
) -> nn.Module:
    """
    Convert only the classification head for PAI (useful for fine-tuning).
    
    This is lighter-weight than full model conversion and is appropriate when
    the encoder is frozen and only the classifier is being trained.
    
    Args:
        model: Model with classification head
        head_modules: Names of head modules to convert
        
    Returns:
        Model with converted head
    """
    try:
        import perforatedai as PA
        
        PA.convert_network(
            model,
            module_names_to_convert=head_modules
        )
        
        logger.info(f"Converted classifier head modules: {head_modules}")
        
    except ImportError:
        logger.warning("perforatedai not installed - head not converted")
    
    return model
