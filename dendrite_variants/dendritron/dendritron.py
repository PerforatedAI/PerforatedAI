"""A sparse, routed Dendritron candidate for PerforatedAI.

This variant changes only the candidate architecture. PerforatedAI therefore
continues to choose the learning rule: standard gradient descent in the open
source package, or the configured Perforated Backpropagation rule when that
separate package is enabled.
"""

from __future__ import print_function

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from perforatedai import globals_perforatedai as GPA


class DendritronLinear(nn.Module):
    """Shape-preserving replacement for an ``nn.Linear`` dendrite candidate.

    A learned router selects ``top_k`` specialist branches for each input. The
    selected outputs are mixed, projected, combined with a residual path, and
    passed through GELU. Leading dimensions are preserved, so both ``[N, F]``
    and higher-rank inputs such as ``[N, T, F]`` are supported.
    """

    def __init__(
        self,
        in_features,
        out_features,
        branches=4,
        top_k=2,
        hidden_features=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if branches < 1:
            raise ValueError("branches must be at least 1")
        if not 1 <= top_k <= branches:
            raise ValueError("top_k must be between 1 and branches")

        self.in_features = in_features
        self.out_features = out_features
        self.branches = branches
        self.top_k = top_k
        if hidden_features is None:
            hidden_features = max(in_features, out_features)
        elif hidden_features < 1:
            raise ValueError("hidden_features must be at least 1")
        self.hidden_features = hidden_features
        self.use_bias = bias

        factory_kwargs = {"device": device, "dtype": dtype}
        self.router = nn.Linear(in_features, branches, **factory_kwargs)
        self.specialists = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features, self.hidden_features, bias=bias, **factory_kwargs
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.hidden_features, out_features, bias=bias, **factory_kwargs
                    ),
                )
                for _ in range(branches)
            ]
        )
        self.post_projection = nn.Linear(
            out_features, out_features, bias=bias, **factory_kwargs
        )
        if in_features == out_features:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Linear(
                in_features, out_features, bias=False, **factory_kwargs
            )
        self.activation = nn.GELU()
        self.last_router_probabilities = None  # type: Optional[torch.Tensor]
        self.last_sparse_weights = None  # type: Optional[torch.Tensor]

    def forward(self, inputs):
        probabilities = self.router(inputs).softmax(dim=-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        sparse_weights = torch.zeros_like(probabilities).scatter(
            -1, top_indices, top_values
        )
        sparse_weights = sparse_weights / sparse_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        specialist_outputs = torch.stack(
            [specialist(inputs) for specialist in self.specialists], dim=-2
        )
        mixed = torch.sum(specialist_outputs * sparse_weights.unsqueeze(-1), dim=-2)
        self.last_router_probabilities = probabilities
        self.last_sparse_weights = sparse_weights
        residual = self.residual_projection(inputs)
        return self.activation(self.post_projection(mixed) + residual)

    def balance_loss(self):
        """Return a differentiable penalty for unequal soft branch usage."""
        if self.last_router_probabilities is None:
            return self.router.weight.new_zeros(())
        reduce_dimensions = tuple(range(self.last_router_probabilities.ndim - 1))
        utilization = self.last_router_probabilities.mean(dim=reduce_dimensions)
        target = torch.full_like(utilization, 1.0 / self.branches)
        return self.branches * (utilization - target).square().sum()

    def routing_metrics(self):
        """Return detached routing diagnostics from the most recent forward pass."""
        if self.last_router_probabilities is None:
            return {}
        probabilities = self.last_router_probabilities.detach().clamp_min(1e-8)
        reduce_dimensions = tuple(range(probabilities.ndim - 1))
        utilization = probabilities.mean(dim=reduce_dimensions)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
        return {
            "router_entropy": float(entropy),
            "min_branch_utilization": float(utilization.min()),
            "max_branch_utilization": float(utilization.max()),
        }


def create_dendritron_dendrite(
    original_module, branches=4, top_k=2, hidden_features=None
):
    """Create a Dendritron candidate compatible with a linear parent module."""
    if isinstance(original_module, DendritronLinear):
        return DendritronLinear(
            original_module.in_features,
            original_module.out_features,
            branches=original_module.branches,
            top_k=original_module.top_k,
            hidden_features=original_module.hidden_features,
            bias=original_module.use_bias,
            device=original_module.router.weight.device,
            dtype=original_module.router.weight.dtype,
        )
    if not isinstance(original_module, nn.Linear):
        raise TypeError(
            "create_dendritron_dendrite expected nn.Linear or "
            "DendritronLinear, got %s" % type(original_module).__name__
        )
    return DendritronLinear(
        original_module.in_features,
        original_module.out_features,
        branches=branches,
        top_k=top_k,
        hidden_features=hidden_features,
        bias=original_module.bias is not None,
        device=original_module.weight.device,
        dtype=original_module.weight.dtype,
    )


def initialize_variant_dendrite(branches=4, top_k=2, hidden_features=None):
    """Register Dendritron creation for every currently perforated module.

    Call this after ``UPA.perforate_model``. Restrict perforation to ``Linear``
    modules before wrapping the model because this variant intentionally does
    not create candidates for convolutional or other module types.
    """
    create_fn = partial(
        create_dendritron_dendrite,
        branches=branches,
        top_k=top_k,
        hidden_features=hidden_features,
    )
    GPA.pai_tracker.set_create_dendrite_global(create_fn)
