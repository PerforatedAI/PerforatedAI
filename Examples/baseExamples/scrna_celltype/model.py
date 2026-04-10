import torch
import torch.nn as nn

from config import N_CHUNKS, D_MODEL, N_HEAD, N_LAYERS, D_FF, DROPOUT


class GeneTransformer(nn.Module):
    """
    Transformer encoder for scRNA-seq cell type annotation.

    Treats a gene expression profile as a sequence of tokens:
    HVGs are split into fixed-size chunks, each chunk is linearly projected,
    and self-attention is applied across chunks before mean-pooling to a
    cell-level representation.

    All nn.Linear layers are exposed for PAI dendritic augmentation:
    input_proj (1) + per-layer out_proj/linear1/linear2 (3 × N_LAYERS) + classifier (1).
    """

    def __init__(self, n_genes: int, n_cell_types: int,
                 n_chunks: int = N_CHUNKS, d_model: int = D_MODEL,
                 nhead: int = N_HEAD, num_layers: int = N_LAYERS,
                 d_ff: int = D_FF, dropout: float = DROPOUT):
        super().__init__()
        assert n_genes % n_chunks == 0, (
            f'n_genes ({n_genes}) must be divisible by n_chunks ({n_chunks})'
        )
        self.n_chunks   = n_chunks
        self.chunk_size = n_genes // n_chunks

        self.input_proj = nn.Linear(self.chunk_size, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, n_chunks, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.classifier  = nn.Linear(d_model, n_cell_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.view(b, self.n_chunks, self.chunk_size)
        x = self.input_proj(x) + self.pos_emb
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)
