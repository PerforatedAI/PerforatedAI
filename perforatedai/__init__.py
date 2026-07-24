"""Perforated AI: Better accuracy, smaller models, less data - enabled by perforated learning.

Perforated AI adds neuron-specific learning signals (artificial dendrites)
during training, helping models achieve higher accuracy with fewer
parameters, less data, and lower deployment costs. It integrates directly
into existing PyTorch workflows with minimal code changes.

Modules
-------
- `perforatedai.globals_perforatedai` — Configuration classes and utilities
  (`PAIConfig`): device settings, dendrite management, module conversion
  options, and training parameters.
- `perforatedai.utils_perforatedai` — Entry point for converting a model
  (`perforate_model`) plus helpers for saving, loading, and inspecting
  PAI networks.
- `perforatedai.modules_perforatedai` — The core module wrappers that add
  dendritic copies to layers and manage dendrite state during training.
- `perforatedai.network_perforatedai` — Network-level conversion and
  checkpoint loading for PAI models.
- `perforatedai.tracker_perforatedai` — Training tracker: validation-score
  history, learning-rate management, and deciding when to add dendrites.
- `perforatedai.library_perforatedai` — Processors and reference
  architectures (LSTM processors, ResNet variants) for modules that need
  custom input/output handling.
- `perforatedai.blockwise_perforatedai` — Blockwise optimization of
  converted networks.
- `perforatedai.clean_perforatedai` — Utilities for exporting a trained
  PAI model to a cleaned-up, scaffold-free form.

See the [README](https://github.com/PerforatedAI/PerforatedAI) for
installation, examples, and key results.
"""
