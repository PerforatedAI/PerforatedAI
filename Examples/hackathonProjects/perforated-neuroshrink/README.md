# Perforated NeuroShrink

## Summary
This project integrates the PerforatedAI framework to evaluate dendritic optimization in a reproducible MNIST classification setting.
A standard CNN baseline is compared against a dendritic CNN implemented using PerforatedAI.

## Method
We replace a standard convolution layer with `PerforatedConv2d` from the PerforatedAI framework, enabling dendritic-style selective computation.

## Results
Run `benchmark.py` to compare parameter counts between baseline and dendritic models.

## Adaptive Computation (Analysis)
The original NeuroShrink project explored adaptive dendritic computation, where dendritic branch utilization dynamically responds to activation strength during inference.
This submission focuses on a minimal PerforatedAI-compliant implementation for reproducibility, while adaptive computation is presented as an extension and future direction.


