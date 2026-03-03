# Model Compression & Dendritic Optimization Guide

This guide explains how to run compression + PAI experiments for the AI TL;DR + Highlights project.

## Experimental Design

We run three experiment types to isolate the effect of dendritic optimization:

| Experiment | Compression | PAI | Purpose |
|------------|-------------|-----|---------|
| A (Baseline) | No | No | Full model reference |
| B (Compressed+PAI) | Yes | Yes | Test dendritic benefits |
| C (Compressed Control) | Yes | No | Isolate PAI effect |

## Quick Start

### Run Bayesian Sweep

```bash
# Initialize sweep
wandb sweep ml/sweeps/sweep_dendrite_dropout_bayes.yaml

# Start agents
wandb agent YOUR_SWEEP_ID
```

### Run Grid Sweep with Hyperband

```bash
wandb sweep ml/sweeps/sweep_dendrite_grid_hb.yaml
wandb agent YOUR_SWEEP_ID
```

### Single Experiment

```bash
# Baseline (A)
python -m ml.train --experiment-type baseline --compression-ratio 1.0

# Compressed + PAI (B)
python -m ml.train --experiment-type compressed_pai --compression-ratio 0.75 --use-pai --dropout-rate 0.1

# Compressed Control (C)
python -m ml.train --experiment-type compressed_control --compression-ratio 0.75 --dropout-rate 0.1
```

## Compression Options

### Layer Reduction

Reduce encoder/decoder layers:

```python
from ml.utils.arch_utils import apply_compression_ratio

model, info = apply_compression_ratio('t5-small', 0.75, device='cuda')
# Keeps ~75% of layers
```

### Dropout Regularization

Apply dropout to prevent overfitting in compressed models:

```python
from ml.models import build_t5_summarizer

model, tokenizer = build_t5_summarizer(
    't5-small',
    dropout_rate=0.15,  # Recommended: 0.1-0.2
    compression_ratio=0.75
)
```

## Distillation (Optional)

Use teacher-student distillation to recover performance:

```python
from ml.distill import DistillationTrainer

trainer = DistillationTrainer(
    teacher=full_model,    # Pretrained T5
    student=small_model,   # Compressed model
    temperature=2.0,
    alpha=0.5
)

loss, metrics = trainer.step(batch)
```

## PAI Knobs

Key PerforatedAI hyperparameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `n_epochs_to_switch` | 2-8 | Neuron epochs before switching |
| `p_epochs_to_switch` | 1-3 | Dendrite epochs per cycle |
| `switch_mode` | history/reinforcement | Switching algorithm |

## Expected Results

Based on preliminary experiments:

| Model | Params | ROUGE-L | HP@3 | Latency |
|-------|--------|---------|------|---------|
| Baseline | 60M | 32.0 | 82% | 420ms |
| Compressed | 45M | 31.1 | 78% | 355ms |
| Compressed+PAI | 45M | 32.3 | 85% | 360ms |

PAI typically recovers 1-2 ROUGE points and 3-7% highlight precision over naive compression.

## Running Tests

```bash
pytest ml/tests/test_compression.py -v
