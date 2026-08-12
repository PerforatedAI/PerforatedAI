"""
Parameter counting helpers for the Adult Income dendritic example
"""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn


def count_params(module: nn.Module) -> int:
    """Return the number of trainable parameters for a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def layerwise_param_breakdown(module: nn.Module) -> List[Tuple[str, int]]:
    """helper for debugging individual layer parameter counts."""
    breakdown: List[Tuple[str, int]] = []
    for name, sub_module in module.named_modules():
        if name == "":
            continue
        param_total = count_params(sub_module)
        if param_total > 0:
            breakdown.append((name, param_total))
    return breakdown
