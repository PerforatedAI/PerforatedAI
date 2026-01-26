"""
HuggingFace-Compatible Tabular Model for Credit Scoring
🤗 Enables integration with HuggingFace ecosystem
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig


class CreditScoringConfig(PretrainedConfig):
    """Configuration class for credit scoring model"""
    model_type = "credit_scoring"
    
    def __init__(
        self,
        input_dim=5,
        hidden_dims=[64, 32],
        output_dim=1,
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout


class HFTabularModel(PreTrainedModel):
    """HuggingFace-compatible tabular model"""
    config_class = CreditScoringConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, input_features, labels=None):
        logits = self.network(input_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


if __name__ == "__main__":
    # Test the HuggingFace model
    config = CreditScoringConfig()
    model = HFTabularModel(config)
    sample_input = torch.randn(1, 6)
    output = model(sample_input)
    print(f"Model output: {output}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
