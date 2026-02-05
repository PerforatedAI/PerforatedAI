"""
Smoke Tests for PAI Training

Quick tests to verify training loop works with and without PAI.

Run with: pytest ml/tests/test_pai_smoke_train.py -v --timeout=60
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Check PAI availability
pai_available = False
try:
    import perforatedai as PA
    pai_available = True
except ImportError:
    pass


class MinimalModel(nn.Module):
    """Minimal model for smoke testing."""
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "layer": nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for i in range(2)
            ])
        })
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        for layer in self.encoder["layer"]:
            x = layer(x)
        return self.classifier(x)


def create_dummy_dataloader(batch_size=4, num_samples=20, input_dim=32, output_dim=10):
    """Create a dummy dataloader for testing."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestSmokeTrainWithoutPAI:
    """Smoke tests that work without PAI installed."""
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = MinimalModel()
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 10)
    
    def test_one_step_training(self):
        """Test one training step."""
        model = MinimalModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        loader = create_dummy_dataloader()
        
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            break  # One step only
        
        assert loss.item() > 0
    
    def test_full_epoch(self):
        """Test full training epoch."""
        model = MinimalModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        loader = create_dummy_dataloader(num_samples=20)
        
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        assert total_loss > 0


@pytest.mark.skipif(not pai_available, reason="perforatedai not installed")
class TestSmokeTrainWithPAI:
    """Smoke tests requiring PAI."""
    
    def test_converted_model_forward(self):
        """Test forward pass after PAI conversion."""
        from ml.models import convert_model_for_pai
        
        model = MinimalModel()
        model, modules = convert_model_for_pai(model, "bert")
        
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 10)
    
    def test_one_step_with_pai(self):
        """Test one training step with PAI conversion."""
        from ml.models import convert_model_for_pai
        from ml.pai_utils import init_pai_tracker
        
        model = MinimalModel()
        model, _ = convert_model_for_pai(model, "bert")
        
        tracker = init_pai_tracker("smoke_test", maximizing_score=True, make_graphs=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        loader = create_dummy_dataloader()
        
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            break
        
        assert loss.item() > 0
    
    def test_eval_with_tracker(self):
        """Test evaluation with PAI tracker feedback."""
        from ml.models import convert_model_for_pai
        from ml.pai_utils import init_pai_tracker, add_validation_score
        
        model = MinimalModel()
        model, _ = convert_model_for_pai(model, "bert")
        
        tracker = init_pai_tracker("eval_test", maximizing_score=True, make_graphs=False)
        
        # Simulate evaluation
        model, improved, restructured, complete = add_validation_score(
            tracker, model, 0.7
        )
        
        assert model is not None
        # One score shouldn't trigger completion
        assert not complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
