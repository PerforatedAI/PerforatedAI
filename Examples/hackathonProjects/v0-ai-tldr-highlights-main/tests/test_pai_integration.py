"""
Unit tests for PerforatedAI integration.

Run with: pytest tests/test_pai_integration.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleClassifier(nn.Module):
    """Simple classifier for testing PAI conversion."""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.classifier(x)


class TestPAIConfig:
    """Tests for PAI configuration module."""
    
    def test_config_creation(self):
        """Test PAIConfig can be created with defaults."""
        from ml.config.pai_config import PAIConfig
        
        config = PAIConfig()
        assert config.n_epochs_to_switch == 5
        assert config.p_epochs_to_switch == 3
        assert config.dendrite_capacity == 8
    
    def test_config_do_pb_property(self):
        """Test do_pb property based on experiment type."""
        from ml.config.pai_config import PAIConfig, ExperimentType
        
        config_b = PAIConfig(experiment_type=ExperimentType.COMPRESSED_DENDRITES)
        assert config_b.do_pb is True
        
        config_a = PAIConfig(experiment_type=ExperimentType.BASELINE)
        assert config_a.do_pb is False
        
        config_c = PAIConfig(experiment_type=ExperimentType.COMPRESSED_CONTROL)
        assert config_c.do_pb is False
    
    def test_config_save_name(self):
        """Test deterministic save name generation."""
        from ml.config.pai_config import PAIConfig, ExperimentType
        
        config = PAIConfig(
            experiment_type=ExperimentType.COMPRESSED_DENDRITES,
            experiment_name="test_model",
            dendrite_capacity=16,
            n_epochs_to_switch=10,
            p_epochs_to_switch=5,
        )
        
        assert "test_model" in config.save_name
        assert "expB" in config.save_name
        assert "cap16" in config.save_name
        assert "n10" in config.save_name
        assert "p5" in config.save_name


class TestPAIConverter:
    """Tests for model conversion module."""
    
    def test_wrapper_forward_pass(self):
        """Test that wrapped modules maintain forward pass."""
        from ml.models.pai_converter import _PAIModuleWrapper
        
        original = nn.Linear(10, 5)
        wrapped = _PAIModuleWrapper(original, "test_layer")
        
        x = torch.randn(2, 10)
        
        # Should produce identical outputs
        with torch.no_grad():
            orig_out = original(x)
            wrap_out = wrapped(x)
        
        assert torch.allclose(orig_out, wrap_out)
    
    def test_simple_model_conversion(self):
        """Test conversion of simple classifier."""
        model = SimpleClassifier()
        
        # Verify forward pass works before conversion
        x = torch.randn(4, 768)
        out_before = model(x)
        assert out_before.shape == (4, 6)
        
        # Try conversion (will skip if PAI not installed)
        try:
            from ml.models.pai_converter import convert_classifier_head_for_pai
            model = convert_classifier_head_for_pai(model, ["fc1", "fc2", "classifier"])
        except ImportError:
            pytest.skip("perforatedai not installed")
        
        # Verify forward pass still works after conversion
        out_after = model(x)
        assert out_after.shape == (4, 6)
    
    def test_backward_pass_after_conversion(self):
        """Test that gradients flow correctly after conversion."""
        model = SimpleClassifier()
        
        try:
            from ml.models.pai_converter import convert_classifier_head_for_pai
            model = convert_classifier_head_for_pai(model, ["fc1", "fc2"])
        except ImportError:
            pytest.skip("perforatedai not installed")
        
        # Forward pass
        x = torch.randn(4, 768, requires_grad=True)
        out = model(x)
        loss = out.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are not NaN
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestDataLoader:
    """Tests for data loading module."""
    
    def test_sentence_splitter(self):
        """Test sentence splitting consistency."""
        from ml.data.data_loader import SentenceSplitter
        
        splitter = SentenceSplitter()
        
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = splitter.split(text)
        
        assert len(sentences) == 3
        assert sentences[0][0] == "This is sentence one."
        assert sentences[1][0] == "This is sentence two!"
        assert sentences[2][0] == "Is this sentence three?"
    
    def test_sentence_offsets(self):
        """Test that sentence offsets are correct."""
        from ml.data.data_loader import SentenceSplitter
        
        splitter = SentenceSplitter()
        
        text = "First sentence here. Second sentence follows."
        sentences = splitter.split(text)
        
        for sentence, start, end in sentences:
            # Verify offset extraction matches sentence
            extracted = text[start:end].strip()
            assert sentence in extracted or extracted in sentence


class TestHighlightEvaluator:
    """Tests for highlight evaluation module."""
    
    def test_sentence_match_exact(self):
        """Test exact sentence matching."""
        from ml.evaluation.highlight_evaluator import HighlightEvaluator
        
        # Create minimal evaluator (no model needed for this test)
        class MockModel(nn.Module):
            def forward(self, **kwargs):
                return type('obj', (object,), {'pooler_output': torch.randn(1, 768)})()
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        evaluator = HighlightEvaluator(MockModel(), tokenizer, device="cpu")
        
        # Test matching
        reference_set = {"This is a test sentence.", "Another reference."}
        
        assert evaluator._sentence_match("This is a test sentence.", reference_set)
        assert evaluator._sentence_match("Another reference.", reference_set)
        assert not evaluator._sentence_match("Completely different text.", reference_set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
