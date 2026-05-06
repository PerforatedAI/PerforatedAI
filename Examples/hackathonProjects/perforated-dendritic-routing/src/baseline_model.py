import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """
    Simple feed-forward baseline model used to evaluate
    dendritic routing as an inference-time optimization.

    Design goals:
    - Deterministic
    - CPU-friendly
    - Interpretable compute cost
    - Clear output dimensionality
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 16):
        super().__init__()

        self.out_features = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        No conditional logic here — this is the reference model.
        """
        return self.net(x)
