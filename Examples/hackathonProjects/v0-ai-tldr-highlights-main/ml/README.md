# PerforatedAI Training Directory

Research-grade training code for dendritic optimization in AI TL;DR + Smart Highlights.

![PAI Integration Overview](../docs/images/PAI.jpg)

## Quick Start

```bash
# Install dependencies
pip install perforatedai torch transformers datasets evaluate wandb

# Run baseline experiment
python -m ml.train --experiment-type baseline --dataset samsum --wandb disabled

# Run compressed + PAI experiment
python -m ml.train --experiment-type compressed_pai --dataset samsum --wandb enabled

# Run compressed control (ablation)
python -m ml.train --experiment-type compressed_control --dataset samsum
```

## Experiment Protocol

| Type | Name | Compression | Dendrites | Purpose |
|------|------|-------------|-----------|---------|
| A | Baseline | No | No | Full model reference |
| B | Compressed+PAI | Yes | Yes | Dendritic optimization |
| C | Compressed Control | Yes | No | Ablation study |

**Always run all three** for valid comparisons.

## Directory Structure

```
ml/
├── config.py          # Canonical experiment configuration
├── data.py            # Dataset loading and preprocessing
├── models.py          # Model building and PAI conversion
├── train.py           # Main training script (CLI)
├── eval.py            # Evaluation and comparison
├── pai_utils.py       # PAI helper functions
├── tests/             # Unit tests
└── README.md          # This file
```

## Key Design Decisions

### 1. PAI Controls Stopping

Training uses `max_steps=100000` because PerforatedAI decides when to stop:

```python
# PAI signals completion via tracker
model, improved, restructured, training_complete = tracker.add_validation_score(model, score)
if training_complete:
    break
```

### 2. Step-Based Evaluation

Evaluation runs every N steps (not epochs) for proper PAI feedback:

```python
if global_step % config.eval_steps == 0:
    eval_score = evaluate_model(model, eval_loader)
    # PAI feedback here
```

### 3. Explicit Global Configuration

All PAI globals are set in one place before training:

```python
from ml.config import configure_experiment
configure_experiment(config)  # Sets PBG.n_epochs_to_switch, etc.
```

## Metrics Logged

| Metric | Description |
|--------|-------------|
| ROUGE-1/2/L | Summarization quality |
| Latency | Inference time (ms) |
| Parameters | Total and active count |
| Memory | GPU usage (MB) |

## Example Commands

```bash
# Full training with W&B
python -m ml.train \
  --experiment-type compressed_pai \
  --dataset samsum \
  --model-name t5-small \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --wandb enabled \
  --wandb-project v0-pai-experiments

# Quick smoke test
python -m ml.train \
  --experiment-type baseline \
  --dataset samsum \
  --max-steps 100 \
  --wandb disabled
