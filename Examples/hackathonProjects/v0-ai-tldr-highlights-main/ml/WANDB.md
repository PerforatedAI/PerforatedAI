# Weights & Biases (W&B) Integration

This document describes how to use W&B for experiment tracking in the v0-ai-tldr-highlights project.

## Quick Start

### 1. Get Your API Key

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. **IMPORTANT**: Never commit your API key to the repository

### 2. Set Environment Variable

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export WANDB_API_KEY="<YOUR_KEY>"

# Or use wandb login
wandb login
```

### 3. Run Training with W&B

```bash
# Basic training with W&B logging
python -m ml.train --experiment-type compressed_pai --dataset samsum --wandb enabled

# Disable W&B logging
python -m ml.train --experiment-type baseline --wandb disabled

# Or set environment variable
WANDB_MODE=disabled python -m ml.train --experiment-type baseline
```

## What Gets Logged

### Metrics
- `train/loss` - Per-step training loss
- `train/lr` - Learning rate
- `eval/score` - Evaluation score
- `model/total_params` - Parameter count
- `pai/restructured` - PAI restructuring events
- `final/best_score` - Final best score

### Artifacts
- Model checkpoints (best and final)
- Training configuration
- Evaluation results JSON

### Tables
- Training history with loss/score curves
- Highlight evaluation results

## W&B Sweeps

Run hyperparameter sweeps:

```bash
# Create sweep
wandb sweep ml/sweeps/wandb_sweep_t5.yaml

# Run agent (copy sweep ID from output)
wandb agent <YOUR_SWEEP_ID>
```

## GitHub Actions Integration

W&B is automatically integrated in CI:

1. Add `WANDB_API_KEY` to GitHub Secrets (Settings → Secrets → Actions)
2. Optionally add `WANDB_ENTITY` for team projects
3. PRs run smoke tests with W&B disabled
4. Main branch pushes run full training with W&B enabled

## Security Best Practices

1. **Never commit API keys** - Use environment variables or secrets
2. **Rotate exposed keys immediately** - If you accidentally expose a key, revoke it at wandb.ai/authorize
3. **Use GitHub Secrets** - For CI/CD, store keys in repository secrets
4. **Set WANDB_MODE=disabled** - In development when you don't need logging

## Viewing Results

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project (default: `v0-ai-tldr-highlights`)
3. View runs, compare experiments, and download artifacts

## Troubleshooting

### "wandb: ERROR api_key not configured"
Set `WANDB_API_KEY` environment variable or run `wandb login`

### "wandb: WARNING Connection failed"
Check your internet connection or set `WANDB_MODE=offline` for offline mode

### "Permission denied"
Ensure you have write access to the W&B project, or create a new project
