
"""
Improved Transformer language model with explicit Linear layers.
This version exposes all linear layers for dendritic augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with explicit Linear layers."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with explicit Linear layers."""
    
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class TransformerLM(nn.Module):
    """Transformer language model for causal language modeling."""
    
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        ff_dim=None,
        dropout=0.1,
        max_seq_len=512
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        if ff_dim is None:
            ff_dim = 4 * embed_dim
        self.ff_dim = ff_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask for autoregressive modeling."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src):
        """Forward pass through the model."""
        seq_len = src.size(1)
        device = src.device
        mask = self.generate_square_subsequent_mask(seq_len, device)
        
        x = self.token_embedding(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.output_projection(x)
        return logits
    
    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self):
        """Get detailed parameter count breakdown."""
        breakdown = {}
        
        # Token embedding
        breakdown['token_embedding'] = sum(p.numel() for p in self.token_embedding.parameters())
        
        # Positional encoding (no parameters)
        breakdown['pos_encoder'] = 0
        
        # Transformer layers
        total_transformer = 0
        for i, layer in enumerate(self.layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            total_transformer += layer_params
            
            # Detailed breakdown per layer
            attn_params = sum(p.numel() for p in layer.self_attn.parameters())
            ffn_params = sum(p.numel() for p in layer.ffn.parameters())
            norm_params = sum(p.numel() for p in layer.norm1.parameters()) + sum(p.numel() for p in layer.norm2.parameters())
            
            breakdown[f'layer_{i}_attention'] = attn_params
            breakdown[f'layer_{i}_ffn'] = ffn_params
            breakdown[f'layer_{i}_norm'] = norm_params
        
        breakdown['transformer_total'] = total_transformer
        
        # Output projection
        breakdown['output_projection'] = sum(p.numel() for p in self.output_projection.parameters())
        
        breakdown['total'] = self.count_parameters()
        
        return breakdown
    
    def count_linear_layers(self):
        """Count the number of Linear layers (for dendrite estimation)."""
        count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                count += 1
        return count


def create_model(vocab_size, model_type='vanilla', **kwargs):
    """Factory function to create Transformer models."""
    if model_type == 'vanilla':
        defaults = {
            'embed_dim': 256,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    elif model_type == 'dendritic':
        defaults = {
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    defaults.update(kwargs)
    
    model = TransformerLM(vocab_size=vocab_size, **defaults)
    
    print(f"\n{'='*50}")
    print(f"Created {model_type.upper()} model")
    print(f"{'='*50}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {defaults['embed_dim']}")
    print(f"Number of heads: {defaults['num_heads']}")
    print(f"Number of layers: {defaults['num_layers']}")
    print(f"Feed-forward dimension: {defaults.get('ff_dim', 4 * defaults['embed_dim'])}")
    
    num_linear = model.count_linear_layers()
    print(f"Total Linear layers: {num_linear}")
    
    breakdown = model.get_parameter_breakdown()
    print(f"\nParameter breakdown:")
    for name, count in breakdown.items():
        if 'total' not in name.lower() and 'transformer_total' not in name:
            print(f"  {name}: {count:,}")
    print(f"  {'='*40}")
    print(f"  TOTAL: {breakdown['total']:,}")
    print(f"{'='*50}\n")
    
    return model


if __name__ == "__main__":
    # Test model creation
    vocab_size = 10000
    
    print("Testing Vanilla Model:")
    vanilla_model = create_model(vocab_size, 'vanilla')
    
    print("\nTesting Dendritic Model (base):")
    dendritic_model = create_model(vocab_size, 'dendritic')
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    
    output = vanilla_model(dummy_input)
    print(f"Vanilla output shape: {output.shape}")
    
    output = dendritic_model(dummy_input)
    print(f"Dendritic output shape: {output.shape}")

