"""
Tiny smoke test to ensure both vanilla and dendritic models run a forward pass.
"""

from __future__ import annotations

import types
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent

# Sibling module (train); perforatedai comes from the installed package.
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train import AdultMLP, prepare_dendritic_model


def _make_args(use_dendrites: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        width=64,
        dropout=0.1,
        use_dendrites=use_dendrites,
        exclude_output_proj=True,
        seed=42,
    )


def run_smoke_test() -> None:
    torch.manual_seed(0)
    batch_size, feature_dim = 8, 32
    dummy_input = torch.randn(batch_size, feature_dim)

    vanilla_args = _make_args(use_dendrites=False)
    vanilla_model = AdultMLP(
        feature_dim, hidden_width=vanilla_args.width, dropout=vanilla_args.dropout
    )
    vanilla_output = vanilla_model(dummy_input)
    assert vanilla_output.shape in {(batch_size,), (batch_size, 1)}

    dendrite_args = _make_args(use_dendrites=True)
    dendritic_model = AdultMLP(
        feature_dim, hidden_width=dendrite_args.width, dropout=dendrite_args.dropout
    )
    dendritic_model = prepare_dendritic_model(
        dendritic_model,
        exclude_output_proj=dendrite_args.exclude_output_proj,
        save_name="adult_test_smoke",
    )
    dendritic_output = dendritic_model(dummy_input)
    assert dendritic_output.shape in {(batch_size,), (batch_size, 1)}

    print("OK")


if __name__ == "__main__":
    run_smoke_test()
