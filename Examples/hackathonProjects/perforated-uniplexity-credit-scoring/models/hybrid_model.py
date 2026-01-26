"""
Hybrid Neural Network Model for Credit Scoring
Combines Baseline MLP and Dendritic Model predictions
"""
import torch
import torch.nn as nn
from models.baseline_model import BaselineModel
from models.dendritic_model import DendriticModel

class HybridModel(nn.Module):
    """Hybrid model comparing and fusing Baseline and Dendritic streams"""
    
    def __init__(self, input_dim=5):
        super(HybridModel, self).__init__()
        
        # Initialize sub-models
        self.baseline = BaselineModel(input_dim=input_dim)
        self.dendritic = DendriticModel(input_dim=input_dim)
        
        # Fusion layer: takes logits from both models and produces final prediction
        # Input: 2 (one logit from baseline, one from dendritic)
        self.fusion = nn.Linear(2, 1)
        
    def forward(self, x):
        # Get logits from both streams
        # Note: Both sub-models return logits (pre-sigmoid)
        baseline_out = self.baseline(x)
        dendritic_out = self.dendritic(x)
        
        # Concatenate logits
        combined = torch.cat([baseline_out, dendritic_out], dim=1)
        
        # Final fusion
        final_out = self.fusion(combined)
        return final_out

if __name__ == "__main__":
    model = HybridModel()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
