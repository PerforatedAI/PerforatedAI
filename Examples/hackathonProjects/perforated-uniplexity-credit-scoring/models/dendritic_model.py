"""
Dendritic Neural Network Model for Credit Scoring
Implements dendritic computation for enhanced feature processing
"""
import torch
import torch.nn as nn


class DendriticLayer(nn.Module):
    """Dendritic layer with compartmentalized processing"""
    
    def __init__(self, input_dim, output_dim, num_dendrites=4):
        super(DendriticLayer, self).__init__()
        self.num_dendrites = num_dendrites
        
        # Each dendrite processes a subset of features
        self.dendrites = nn.ModuleList([
            nn.Linear(input_dim, output_dim // num_dendrites)
            for _ in range(num_dendrites)
        ])
        
        # Batch Normalization for stability
        self.bn = nn.BatchNorm1d(output_dim)
        
        # Somatic integration
        self.soma = nn.Linear(output_dim, output_dim)
        # Enhanced non-linearity
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Process through each dendrite
        dendrite_outputs = [dendrite(x) for dendrite in self.dendrites]
        
        # Concatenate dendrite outputs
        combined = torch.cat(dendrite_outputs, dim=-1)
        
        # Apply BatchNorm
        combined = self.bn(combined)
        
        # Somatic integration
        output = self.soma(combined)
        return self.activation(output)


class DendriticModel(nn.Module):
    """Dendritic neural network for credit scoring"""
    
    def __init__(self, input_dim=5, hidden_dim=24, output_dim=1, num_dendrites=4):
        super(DendriticModel, self).__init__()
        
        self.dendritic_layer1 = DendriticLayer(input_dim, hidden_dim, num_dendrites)
        self.dendritic_layer2 = DendriticLayer(hidden_dim, hidden_dim // 2, num_dendrites)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dendritic_layer1(x)
        x = self.dropout(x)
        x = self.dendritic_layer2(x)
        x = self.dropout(x)
        # No sigmoid here, use BCEWithLogitsLoss
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = DendriticModel()
    sample_input = torch.randn(1, 5)
    output = model(sample_input)
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
