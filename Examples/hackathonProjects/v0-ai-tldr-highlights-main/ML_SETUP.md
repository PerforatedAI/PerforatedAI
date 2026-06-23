# ML Setup Guide

Complete guide to setting up the ML pipeline with **Weights & Biases**, **HuggingFace**, **PyTorch**, and **PerforatedAI**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements-ml.txt

# 2. Set environment variables
export WANDB_API_KEY="your_wandb_key"
export HF_TOKEN="your_hf_token"

# 3. Verify setup
python -m ml.verify_setup

# 4. Run quick training test
python -m ml.quick_train
```

## Environment Variables

### Required

| Variable | Description | How to Get |
|----------|-------------|------------|
| `WANDB_API_KEY` | Weights & Biases API key | [wandb.ai/authorize](https://wandb.ai/authorize) |
| `HF_TOKEN` | HuggingFace access token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_PROJECT` | `v0-ai-tldr-highlights` | W&B project name |
| `WANDB_ENTITY` | `lucylow` | W&B team/user |
| `WANDB_MODE` | `online` | `online`, `offline`, or `disabled` |

## Setting Up Weights & Biases

1. **Create Account**: Go to [wandb.ai](https://wandb.ai) and sign up
2. **Get API Key**: Visit [wandb.ai/authorize](https://wandb.ai/authorize)
3. **Set Environment Variable**:
   ```bash
   export WANDB_API_KEY="wandb_v1_xxx..."
   ```
4. **Verify**:
   ```bash
   python -c "import wandb; wandb.login()"
   ```

## Setting Up HuggingFace

1. **Create Account**: Go to [huggingface.co](https://huggingface.co) and sign up
2. **Create Token**: Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Choose Permissions**: Select "Fine-grained" for full access
4. **Set Environment Variable**:
   ```bash
   export HF_TOKEN="hf_xxx..."
   ```
5. **Verify**:
   ```bash
   python -c "from huggingface_hub import HfApi; HfApi().whoami()"
   ```

## Installing PerforatedAI (Optional)

PerforatedAI provides dendritic optimization for neural networks:

```bash
pip install perforatedai
```

Verify installation:
```bash
python -c "import perforatedai as PA; print('PAI installed')"
```

## Running Training

### Basic Training

```bash
python -m ml.train \
  --experiment-type baseline \
  --dataset samsum \
  --model-name t5-small \
  --max-steps 1000
```

### With PerforatedAI

```bash
python -m ml.train \
  --experiment-type compressed_pai \
  --dataset samsum \
  --model-name t5-small \
  --use-pai \
  --compression-ratio 0.75
```

### W&B Sweep

```bash
# Create sweep
wandb sweep ml/sweeps/sweep_pai_core.yaml

# Run agents
wandb agent lucylow/v0-ai-tldr-highlights/SWEEP_ID
```

## Verification Commands

```bash
# Full verification
python -m ml.verify_setup

# Quick training test (50 steps)
python -m ml.quick_train

# Run unit tests
pytest ml/tests/ -v
```

## Troubleshooting

### W&B Login Issues
```bash
# Force re-login
wandb login --relogin

# Use offline mode for testing
export WANDB_MODE=offline
```

### HuggingFace Token Issues
```bash
# Verify token
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"

# Clear cache
rm -rf ~/.cache/huggingface/
```

### CUDA Out of Memory
```bash
# Reduce batch size
python -m ml.train --batch-size 2

# Use gradient checkpointing
python -m ml.train --gradient-checkpointing
```

### PerforatedAI Not Found
```bash
# Install from PyPI
pip install perforatedai

# Or skip PAI (training will still work)
python -m ml.train --experiment-type baseline
```

## Project Structure

```
ml/
├── train.py              # Main training script
├── quick_train.py        # Quick verification training
├── verify_setup.py       # Setup verification
├── config.py             # Configuration classes
├── models.py             # Model builders
├── data.py               # Data loading
├── pai_utils.py          # PerforatedAI utilities
├── eval.py               # Evaluation metrics
├── tests/                # Unit tests
├── sweeps/               # W&B sweep configs
└── configs/              # Training configs
```

## Links

- [W&B Dashboard](https://wandb.ai/lucylow/v0-ai-tldr-highlights)
- [HuggingFace Hub](https://huggingface.co)
- [PerforatedAI Docs](https://perforatedai.com)
- [PyTorch Docs](https://pytorch.org/docs)
