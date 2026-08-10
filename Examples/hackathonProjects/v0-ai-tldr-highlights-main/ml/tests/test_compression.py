"""
Tests for Model Compression Utilities

Ensures compression and PAI conversion pipelines work correctly.
"""

import pytest
import torch


def test_instantiate_smaller_t5():
    """Test creating a smaller T5 model."""
    from ml.utils.arch_utils import instantiate_smaller_t5
    
    small, info = instantiate_smaller_t5(
        't5-small',
        keep_encoder_layers=2,
        keep_decoder_layers=2,
        device='cpu'
    )
    
    assert small is not None
    assert info['modifications']['num_layers']['to'] == 2
    assert len(small.encoder.block) == 2
    assert len(small.decoder.block) == 2


def test_copy_weights_partial():
    """Test copying weights between different-sized models."""
    from transformers import T5ForConditionalGeneration
    from ml.utils.arch_utils import instantiate_smaller_t5, copy_weights_partial
    
    # Create source and target
    full = T5ForConditionalGeneration.from_pretrained('t5-small')
    small, _ = instantiate_smaller_t5('t5-small', keep_encoder_layers=2, device='cpu')
    
    # Copy weights
    matched = copy_weights_partial(full, small)
    
    assert matched > 0
    # Embedding weights should match
    assert torch.allclose(
        full.shared.weight[:small.shared.weight.size(0), :small.shared.weight.size(1)],
        small.shared.weight
    )


def test_apply_compression_ratio():
    """Test applying compression ratio."""
    from ml.utils.arch_utils import apply_compression_ratio
    
    model, info = apply_compression_ratio('t5-small', 0.5, device='cpu')
    
    assert model is not None
    assert info['compression_ratio'] == 0.5
    assert info['method'] == 'layer_reduction'


def test_build_t5_with_dropout():
    """Test building T5 with custom dropout."""
    from ml.models import build_t5_summarizer
    
    model, tokenizer = build_t5_summarizer(
        't5-small',
        dropout_rate=0.2,
        compression_ratio=1.0,
        device='cpu'
    )
    
    assert model is not None
    assert tokenizer is not None
    # Check dropout was applied
    assert model.config.dropout_rate == 0.2


def test_compression_and_pai_conversion():
    """Test full pipeline: compression then PAI conversion."""
    from ml.utils.arch_utils import apply_compression_ratio
    from ml.models import convert_model_for_pai
    
    # Compress
    model, info = apply_compression_ratio('t5-small', 0.75, device='cpu')
    
    # Convert for PAI (may fail if PAI not installed)
    try:
        model, modules = convert_model_for_pai(model, 't5')
        assert len(modules) > 0
    except ImportError:
        pytest.skip("perforatedai not installed")


def test_distillation_loss():
    """Test distillation loss computation."""
    from ml.distill import distillation_step
    from transformers import T5ForConditionalGeneration
    
    teacher = T5ForConditionalGeneration.from_pretrained('t5-small')
    student = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Dummy inputs
    inputs = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones_like(inputs)
    targets = torch.randint(0, 1000, (2, 16))
    
    loss, metrics = distillation_step(
        student, teacher,
        inputs, attention_mask, targets,
        temperature=2.0, alpha=0.5
    )
    
    assert loss.item() > 0
    assert 'ce_loss' in metrics
    assert 'kl_loss' in metrics


def test_model_size_info():
    """Test getting model size information."""
    from transformers import T5ForConditionalGeneration
    from ml.utils.arch_utils import get_model_size_info
    
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    info = get_model_size_info(model)
    
    assert info['total_parameters'] > 0
    assert info['total_memory_mb'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
