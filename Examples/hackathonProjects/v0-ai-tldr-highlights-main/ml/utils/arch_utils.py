"""
Architecture Utilities for Model Compression

Provides functions to:
- Reduce encoder/decoder layers
- Reduce model hidden size (d_model)
- Reduce attention heads
- Copy weights between different-sized models

Usage:
    from ml.utils.arch_utils import instantiate_smaller_t5, copy_weights_partial
    
    small_model, info = instantiate_smaller_t5("t5-small", keep_encoder_layers=4)
    matched = copy_weights_partial(full_model, small_model)
"""

import logging
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def instantiate_smaller_t5(
    base_model_name: str,
    keep_encoder_layers: Optional[int] = None,
    keep_decoder_layers: Optional[int] = None,
    d_model: Optional[int] = None,
    num_heads: Optional[int] = None,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Create a smaller T5 model config and instantiate it.
    
    Args:
        base_model_name: Base model to derive config from
        keep_encoder_layers: Number of encoder layers to keep
        keep_decoder_layers: Number of decoder layers to keep
        d_model: New hidden dimension (scales d_ff proportionally)
        num_heads: New number of attention heads
        device: Target device
        
    Returns:
        Tuple of (model, info_dict)
    """
    from transformers import T5Config, T5ForConditionalGeneration
    
    base_cfg = T5Config.from_pretrained(base_model_name)
    new_cfg = base_cfg.to_dict()
    
    info = {
        'original_config': base_cfg.to_dict(),
        'modifications': {}
    }
    
    # Adjust hidden dimension
    if d_model is not None:
        original_d_model = base_cfg.d_model
        new_cfg['d_model'] = d_model
        # Scale feed-forward proportionally
        if 'd_ff' in new_cfg:
            new_cfg['d_ff'] = max(64, int(new_cfg['d_ff'] * (d_model / original_d_model)))
        info['modifications']['d_model'] = {'from': original_d_model, 'to': d_model}
    
    # Adjust attention heads
    if num_heads is not None:
        original_heads = base_cfg.num_heads
        new_cfg['num_heads'] = num_heads
        info['modifications']['num_heads'] = {'from': original_heads, 'to': num_heads}
    
    # Adjust encoder layers
    if keep_encoder_layers is not None:
        original_layers = base_cfg.num_layers
        new_cfg['num_layers'] = keep_encoder_layers
        info['modifications']['num_layers'] = {'from': original_layers, 'to': keep_encoder_layers}
    
    # Adjust decoder layers
    if keep_decoder_layers is not None:
        original_decoder_layers = base_cfg.num_decoder_layers
        new_cfg['num_decoder_layers'] = keep_decoder_layers
        info['modifications']['num_decoder_layers'] = {'from': original_decoder_layers, 'to': keep_decoder_layers}
    
    # Create new config and model
    cfg = T5Config(**new_cfg)
    model = T5ForConditionalGeneration(cfg)
    model.to(device)
    
    info['new_config'] = cfg.to_dict()
    info['parameter_count'] = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Created smaller T5: {info['parameter_count']:,} parameters")
    for key, val in info['modifications'].items():
        logger.info(f"  {key}: {val['from']} -> {val['to']}")
    
    return model, info


def copy_weights_partial(
    src_model: nn.Module,
    tgt_model: nn.Module,
    max_encoder_layers_to_copy: Optional[int] = None,
    max_decoder_layers_to_copy: Optional[int] = None,
) -> int:
    """
    Copy weights from src_model to tgt_model for matching layers.
    
    Handles dimension mismatches by slicing larger tensors.
    
    Args:
        src_model: Source model with pretrained weights
        tgt_model: Target model (potentially smaller)
        max_encoder_layers_to_copy: Limit encoder layer copying
        max_decoder_layers_to_copy: Limit decoder layer copying
        
    Returns:
        Number of successfully matched parameters
    """
    src_sd = src_model.state_dict()
    tgt_sd = tgt_model.state_dict()
    
    matched = 0
    skipped = 0
    sliced = 0
    
    for key in list(tgt_sd.keys()):
        if key not in src_sd:
            skipped += 1
            continue
        
        src_shape = src_sd[key].shape
        tgt_shape = tgt_sd[key].shape
        
        if src_shape == tgt_shape:
            # Direct copy
            tgt_sd[key] = src_sd[key].clone()
            matched += 1
        elif len(src_shape) == len(tgt_shape) and all(s >= t for s, t in zip(src_shape, tgt_shape)):
            # Slice larger tensor to fit smaller target
            slices = tuple(slice(0, t) for t in tgt_shape)
            tgt_sd[key] = src_sd[key][slices].clone()
            sliced += 1
            matched += 1
        else:
            logger.debug(f"Skipping {key}: shape mismatch src={src_shape} tgt={tgt_shape}")
            skipped += 1
    
    tgt_model.load_state_dict(tgt_sd)
    
    logger.info(f"Weight copy: {matched} matched, {sliced} sliced, {skipped} skipped")
    return matched


def prune_encoder_layers_inplace(model: nn.Module, keep: int):
    """
    Remove encoder blocks beyond 'keep' layers in-place.
    
    Args:
        model: T5 or similar model with encoder.block
        keep: Number of layers to keep
    """
    if keep <= 0:
        raise ValueError("keep must be > 0")
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        original = len(model.encoder.block)
        model.encoder.block = nn.ModuleList(list(model.encoder.block)[:keep])
        model.config.num_layers = keep
        logger.info(f"Pruned encoder: {original} -> {keep} layers")
    else:
        logger.warning("Model does not have encoder.block attribute")


def prune_decoder_layers_inplace(model: nn.Module, keep: int):
    """
    Remove decoder blocks beyond 'keep' layers in-place.
    
    Args:
        model: T5 or similar model with decoder.block
        keep: Number of layers to keep
    """
    if keep <= 0:
        raise ValueError("keep must be > 0")
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        original = len(model.decoder.block)
        model.decoder.block = nn.ModuleList(list(model.decoder.block)[:keep])
        model.config.num_decoder_layers = keep
        logger.info(f"Pruned decoder: {original} -> {keep} layers")
    else:
        logger.warning("Model does not have decoder.block attribute")


def apply_compression_ratio(
    model_name: str,
    compression_ratio: float,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply compression ratio to create a smaller model.
    
    Args:
        model_name: Base model name
        compression_ratio: Ratio of layers to keep (0.0-1.0)
        device: Target device
        
    Returns:
        Tuple of (compressed_model, compression_info)
    """
    from transformers import T5Config, T5ForConditionalGeneration
    
    if compression_ratio >= 1.0:
        # No compression
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        return model, {'compression_ratio': 1.0, 'method': 'none'}
    
    base_cfg = T5Config.from_pretrained(model_name)
    
    # Calculate layers to keep
    keep_encoder = max(1, int(base_cfg.num_layers * compression_ratio))
    keep_decoder = max(1, int(base_cfg.num_decoder_layers * compression_ratio))
    
    # Create smaller model
    small_model, info = instantiate_smaller_t5(
        model_name,
        keep_encoder_layers=keep_encoder,
        keep_decoder_layers=keep_decoder,
        device=device
    )
    
    # Load and copy weights from pretrained
    pretrained = T5ForConditionalGeneration.from_pretrained(model_name)
    matched = copy_weights_partial(pretrained, small_model)
    
    info['compression_ratio'] = compression_ratio
    info['method'] = 'layer_reduction'
    info['weights_matched'] = matched
    
    return small_model, info


def get_model_size_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_mb': param_bytes / (1024 * 1024),
        'buffer_memory_mb': buffer_bytes / (1024 * 1024),
        'total_memory_mb': (param_bytes + buffer_bytes) / (1024 * 1024),
    }
