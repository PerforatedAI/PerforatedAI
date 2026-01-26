"""
Dendritic Neural Network Model for Credit Scoring
Implements dendritic computation for enhanced feature processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: This model is intentionally defined as a Standard MLP.
# The PerforatedAI library (UPA.initialize_pai) will automatically 
# convert these Linear layers into Dendritic Layers at runtime.

class DendriticModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=24, output_dim=1):
        super(DendriticModel, self).__init__()
        
        # Standard Linear Layers (to be converted by PAI)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # We explicitly init weights (optional, but good practice)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = F.gelu(x) # PAI supports GELU
        
        # Layer 2 (Output)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = DendriticModel()
    sample_input = torch.randn(1, 5)
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
