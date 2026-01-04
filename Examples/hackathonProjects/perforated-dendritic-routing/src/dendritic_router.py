import torch
import torch.nn as nn


class DendriticRouter(nn.Module):
    """
    Dendritic routing wrapper for inference-time conditional computation.

    Design goals:
    - Drop-in wrapper (no model changes)
    - Per-sample routing
    - Deterministic and interpretable behavior
    - Safe for CPU-only demo
    """

    def __init__(self, base_model: nn.Module, threshold: float):
        super().__init__()
        self.base_model = base_model
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routing rule:
        - Compute normalized per-sample energy
        - If energy >= threshold → run full model
        - Else → cheap passthrough (identity projection)
        """

        # Compute routing signal
        
        # Per-sample normalized energy (0–1 range)
        energy = x.abs().mean(dim=1)
        energy = energy / (energy.max() + 1e-8)

        # Boolean routing mask
        active_mask = energy >= self.threshold

        # Allocate output tensor
        out = torch.zeros(
            x.size(0),
            self.base_model.out_features,
            device=x.device,
            dtype=x.dtype,
        )

         
        # Active path (full model)
       
        if active_mask.any():
            out[active_mask] = self.base_model(x[active_mask])

        # Inactive path (cheap passthrough)
        if (~active_mask).any():
            # Linear projection using mean-preserving identity
            inactive_x = x[~active_mask]
            out[~active_mask] = inactive_x.mean(dim=1, keepdim=True).repeat(
                1, self.base_model.out_features
            )

        return out
