import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DendriticLayer(nn.Module):
    """
    Dendritic optimization adds learned branch-like sub-computations 
    to increase representational power without many extra parameters.
    """
    def __init__(self, input_dim, hidden_dim):
        super(DendriticLayer, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(3) # 3 dendritic branches
        ])
        self.gate = nn.Linear(input_dim, 3)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        gate_weights = torch.softmax(self.gate(x), dim=-1)
        branch_outputs = torch.stack([branch(x) for branch in self.branches], dim=-1)
        
        # Weighted sum of branches
        combined = torch.sum(branch_outputs * gate_weights.unsqueeze(-2), dim=-1)
        return self.norm(x + combined)

class SarcasmVibeModel(nn.Module):
    def __init__(self, model_name="huawei-noah/TinyBERT_General_4L_312D", num_labels=3):
        super(SarcasmVibeModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Add Dendritic Layer for Hinglish nuance
        self.dendrites = DendriticLayer(self.config.hidden_size, self.config.hidden_size // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply dendritic optimization on the sequence output
        optimized_output = self.dendrites(sequence_output)
        
        # Use [CLS] token for classification
        cls_output = optimized_output[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits
