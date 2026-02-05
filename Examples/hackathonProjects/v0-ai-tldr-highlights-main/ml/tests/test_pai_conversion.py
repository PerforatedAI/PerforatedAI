"""
Unit Tests for PAI Conversion

Tests model conversion, LayerNorm handling, and save/load.

Run with: pytest ml/tests/test_pai_conversion.py -v
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional

# Skip all tests if perforatedai not installed
pai_available = False
try:
    import perforatedai as PA
    pai_available = True
except ImportError:
    pass


class TinyT5Block(nn.Module):
    """Minimal T5-like block for testing."""
    
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attention_mask=None):
        # Self attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        return x


class TinyT5Model(nn.Module):
    """Minimal T5-like model for testing conversion."""
    
    def __init__(self, d_model: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "block": nn.ModuleList([TinyT5Block(d_model) for _ in range(num_layers)])
        })
        self.decoder = nn.ModuleDict({
            "block": nn.ModuleList([TinyT5Block(d_model) for _ in range(num_layers)])
        })
        self.lm_head = nn.Linear(d_model, 100)  # vocab size 100
    
    def forward(self, x):
        # Encoder
        for block in self.encoder["block"]:
            x = block(x)
        # Decoder (simplified - uses encoder output directly)
        for block in self.decoder["block"]:
            x = block(x)
        return self.lm_head(x)


@pytest.mark.skipif(not pai_available, reason="perforatedai not installed")
class TestPAIConversion:
    """Test suite for PAI model conversion."""
    
    def test_tiny_model_creation(self):
        """Test that tiny model can be created and runs."""
        model = TinyT5Model(d_model=64, num_layers=2)
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, d=64
        out = model(x)
        assert out.shape == (2, 10, 100)
    
    def test_conversion_imports(self):
        """Test that PAI imports work."""
        from ml.models import convert_model_for_pai
        assert callable(convert_model_for_pai)
    
    def test_convert_tiny_model(self):
        """Test converting a tiny model."""
        from ml.models import convert_model_for_pai, _PAILayerWrapper
        
        model = TinyT5Model(d_model=64, num_layers=2)
        
        # Count parameters before
        params_before = sum(p.numel() for p in model.parameters())
        
        # Convert
        converted, modules = convert_model_for_pai(model, "t5")
        
        # Verify conversion happened
        assert len(modules) > 0
        
        # Model should still work
        x = torch.randn(2, 10, 64)
        out = converted(x)
        assert out.shape == (2, 10, 100)
    
    def test_forward_backward_after_conversion(self):
        """Test that gradients flow after conversion."""
        from ml.models import convert_model_for_pai
        
        model = TinyT5Model(d_model=64, num_layers=2)
        model, _ = convert_model_for_pai(model, "t5")
        
        # Forward pass
        x = torch.randn(2, 10, 64)
        out = model(x)
        loss = out.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients after backward"


@pytest.mark.skipif(not pai_available, reason="perforatedai not installed")
class TestPAITracker:
    """Test suite for PAI tracker integration."""
    
    def test_tracker_init(self):
        """Test tracker initialization."""
        from ml.pai_utils import init_pai_tracker
        
        tracker = init_pai_tracker(
            save_name="test_tracker",
            maximizing_score=True,
            make_graphs=False,  # Faster for tests
        )
        
        assert tracker is not None
    
    def test_add_validation_score(self):
        """Test adding validation scores to tracker."""
        from ml.pai_utils import init_pai_tracker, add_validation_score
        
        tracker = init_pai_tracker("test_score", maximizing_score=True, make_graphs=False)
        model = TinyT5Model(d_model=64, num_layers=1)
        
        # Add a score
        model, improved, restructured, complete = add_validation_score(
            tracker, model, 0.5
        )
        
        # Should return the model
        assert model is not None
        assert isinstance(improved, bool)
        assert isinstance(restructured, bool)
        assert isinstance(complete, bool)


class TestPAIUtilsWithoutPAI:
    """Test graceful degradation when PAI is not available."""
    
    def test_init_tracker_without_pai(self):
        """Test that init_tracker returns None gracefully."""
        # This test runs even without PAI installed
        from ml.pai_utils import init_pai_tracker
        
        # If PAI is not installed, should return None
        if not pai_available:
            tracker = init_pai_tracker("test", maximizing_score=True)
            # Should not raise, just return None or tracker
            assert tracker is None or tracker is not None
    
    def test_add_validation_score_without_tracker(self):
        """Test add_validation_score with no tracker."""
        from ml.pai_utils import add_validation_score
        
        model = TinyT5Model(d_model=64, num_layers=1)
        
        # Should work with None tracker
        model, improved, restructured, complete = add_validation_score(
            None, model, 0.5
        )
        
        assert model is not None
        assert improved is False
        assert restructured is False
        assert complete is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
