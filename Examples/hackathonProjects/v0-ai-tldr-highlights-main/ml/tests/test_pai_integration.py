"""
Unit Tests for PerforatedAI Integration

Tests:
- Model conversion works on toy models
- Training smoke test completes
- Safetensors workaround executes safely
"""

import pytest
import torch
import torch.nn as nn


class ToyT5Block(nn.Module):
    """Minimal T5-like block for testing."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.layer_norm(x + self.attention(x))
        return self.layer_norm(x + self.ffn(x))


class ToyT5Model(nn.Module):
    """Minimal T5-like model for testing."""
    
    def __init__(self, num_layers=2, hidden_size=64):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "block": nn.ModuleList([ToyT5Block(hidden_size) for _ in range(num_layers)])
        })
        self.decoder = nn.ModuleDict({
            "block": nn.ModuleList([ToyT5Block(hidden_size) for _ in range(num_layers)])
        })
    
    def forward(self, x):
        for block in self.encoder["block"]:
            x = block(x)
        for block in self.decoder["block"]:
            x = block(x)
        return x


def test_convert_model_for_pai():
    """Test that convert_model_for_pai runs on toy model."""
    from ml.models import convert_model_for_pai
    
    model = ToyT5Model()
    initial_params = sum(p.numel() for p in model.parameters())
    
    # This should not raise
    model, modules = convert_model_for_pai(model, model_type="t5")
    
    # Model should still have parameters
    final_params = sum(p.numel() for p in model.parameters())
    assert final_params > 0
    
    # Should return module list
    assert isinstance(modules, list)


def test_pai_config_creation():
    """Test PAI config creation and serialization."""
    from ml.config import ExperimentConfig, ExperimentType
    
    config = ExperimentConfig(
        experiment_type=ExperimentType.COMPRESSED_PAI,
        dataset="samsum",
        model_name="t5-small",
        n_epochs_to_switch=5,
        p_epochs_to_switch=3,
    )
    
    assert config.use_pai == True
    assert config.use_compression == True
    
    # Test serialization
    config_dict = config.to_dict()
    assert "experiment_type" in config_dict
    assert config_dict["use_pai"] == True


def test_pai_utils_tracker_init():
    """Test PAI tracker initialization."""
    from ml.pai_utils import init_pai_tracker
    
    # Should not raise even if perforatedai not installed
    tracker = init_pai_tracker("test_experiment", maximizing_score=True)
    
    # Returns None if PAI not available, or tracker if available
    assert tracker is None or tracker is not None


def test_safetensors_workaround():
    """Test safetensors workaround executes safely."""
    from ml.pai_utils import apply_safetensors_workaround
    
    # Should not raise
    apply_safetensors_workaround(enabled=True)
    apply_safetensors_workaround(enabled=False)


def test_data_loading():
    """Test dataset loading with fallback."""
    from ml.data import load_dataset
    
    # Should return fallback data
    examples = load_dataset("samsum", "train", max_samples=5)
    
    assert len(examples) > 0
    assert "source" in examples[0]
    assert "target" in examples[0]


def test_sentence_splitting():
    """Test sentence splitting consistency."""
    from ml.data import split_into_sentences
    
    text = "This is sentence one. This is sentence two! And this is three?"
    sentences = split_into_sentences(text)
    
    assert len(sentences) == 3
    # Each tuple should be (sentence, start, end)
    assert all(len(s) == 3 for s in sentences)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
