"""
Baseline Neural Network Model for Credit Scoring
"""
import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """Simple feedforward neural network for credit scoring"""
    
    def __init__(self, input_dim=5, hidden_dims=[64, 32], output_dim=1):
        super(BaselineModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        # No sigmoid here, we will use BCEWithLogitsLoss
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    # Test the model
    model = BaselineModel()
    sample_input = torch.randn(1, 5)
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
