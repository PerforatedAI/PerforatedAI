# Selective Synaptic Plasticity for Continual Learning

A biologically-inspired optimizer that prevents catastrophic forgetting through NMDA receptor-like coincidence detection and selective synaptic protection.

## The Problem: Catastrophic Forgetting

When neural networks learn new tasks sequentially, they catastrophically forget previously learned information. This fundamental challenge in continual learning limits the deployment of AI systems in dynamic, real-world environments where learning must be ongoing.

## Our Solution: Selective Plasticity

Inspired by **NMDA receptors** in biological neurons, we developed a selective synaptic plasticity mechanism that:

1. **Detects coincidence** between gradients and activations (like NMDA receptors detect coincident pre/post-synaptic activity)
2. **Tracks parameter importance** using eligibility traces
3. **Protects critical synapses** from being overwritten when learning new tasks

### Key Innovation

```python
# Biological NMDA Receptor Coincidence Detection
if pre_synaptic_active AND post_synaptic_active:
    allow_plasticity = True

# Our Digital Equivalent
if gradient_active AND activation_active:
    gate = 1.0  # Allow weight update
else:
    gate = 0.0  # Protect weight
```

This simple principle yields powerful anti-forgetting capabilities.

## Results

### ⭐ Highlight: 86.55% Task Retention

| Method | Task A Retention | Forgetting Rate | Avg Accuracy |
|--------|------------------|-----------------|--------------|
| **Baseline (Adam)** | 0.00% | 97.51% | 46.64% |
| **SelectivePlasticity + Replay** | **86.55%** | **10.82%** | **89.36%** |

**Key Achievement**: 87% reduction in catastrophic forgetting

### Comprehensive Results

#### Split MNIST (2 Tasks: 0-4 vs 5-9)

| Strategy | Task A Final | Task B Final | Average | Forgetting | Overhead |
|----------|--------------|--------------|---------|------------|----------|
| Baseline | 0.00% | 93.28% | 46.64% | 97.51% | - |
| SelectivePlasticity Only | 0.00% | 92.66% | 46.33% | 97.30% | +3.2% |
| Baseline + Replay | 2.82% | 93.11% | 47.96% | 96.40% | - |
| **SP + Replay** | **86.55%** | **92.16%** | **89.36%** | **10.82%** | +3.2% |

#### Permuted MNIST (10 Tasks)

- **Average Accuracy**: 76.15%
- **Backward Transfer**: -0.07 (vs -0.56 for Adam)
- **Forgetting Reduction**: 87%

#### Computational Efficiency

- **Time Overhead**: Only +3.2%
- **Memory Overhead**: Zero (reuses Adam's momentum buffers)
- **Parameters Added**: None

## Biological Basis

### NMDA Receptor Analogy

In biological neurons, NMDA receptors act as **coincidence detectors**:
- Require both glutamate (pre-synaptic) AND depolarization (post-synaptic)
- Only when both conditions are met, the receptor opens
- This implements Hebb's rule: "neurons that fire together, wire together"

Our optimizer translates this to:
- Gradient magnitude ≈ pre-synaptic activity
- Activation magnitude ≈ post-synaptic activity
- Gate = coincidence detector
- Weight update only when both are active

### Synaptic Protection Mechanisms

1. **Eligibility Traces**: Track which parameters are frequently used
2. **Adaptive Thresholds**: Dynamically adjust protection strength
3. **Surprise Modulation**: Allow plasticity for unexpected patterns
4. **Warmup Period**: Build accurate importance estimates before gating

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib

# Clone repository
git clone https://github.com/perforatedai/PerforatedAI-Examples
cd PerforatedAI-Examples/selectivePlasticity
```

## Usage

### Quick Start (3 Lines)

```python
from selective_plasticity import SelectivePlasticityOptimizer

# Replace your optimizer with SelectivePlasticity
optimizer = SelectivePlasticityOptimizer(
    model.parameters(),
    lr=0.001,
    trace_decay=0.99,      # Eligibility trace decay rate
    plasticity_k=5.0,      # Gate sharpness
    warmup_steps=1000      # Steps before gating activates
)

# Use exactly like any PyTorch optimizer
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Full Example with Experience Replay

```python
import torch
import torch.nn as nn
from selective_plasticity import SelectivePlasticityOptimizer

# Create model
model = SimpleMLP(input_size=784, hidden_size=256, output_size=10)

# Setup optimizer with selective plasticity
optimizer = SelectivePlasticityOptimizer(
    model.parameters(),
    lr=0.001,
    trace_decay=0.99,
    plasticity_k=5.0
)

# Memory buffer for experience replay
memory = MemoryBuffer(capacity=2000)

# Train on multiple tasks
for task_id, task_data in enumerate(tasks):
    for epoch in range(epochs_per_task):
        for data, target in task_data:
            # Mix current data with replayed data
            if len(memory) > 0:
                replay_data, replay_targets = memory.sample(batch_size // 2)
                data = torch.cat([data, replay_data])
                target = torch.cat([target, replay_targets])

            # Standard training step
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            # Update replay buffer
            memory.add_batch(data, target)
```

## Experiments

### Included Experiments

1. **split_mnist.py** - Main experiment demonstrating 86.55% retention
2. **permuted_mnist.py** - Standard continual learning benchmark
3. **hyperparameter_search.py** - Systematic hyperparameter tuning

Run experiments:
```bash
# Split MNIST (quick, 5-10 min)
python experiments/split_mnist.py

# Permuted MNIST (30-45 min)
python experiments/permuted_mnist.py

# Hyperparameter search (2-3 hours)
python experiments/hyperparameter_search.py
```

## Key Findings

### 1. Replay Synergy

SelectivePlasticity **amplifies the effectiveness** of experience replay:
- Replay alone: 2.82% Task A retention
- SP + Replay: 86.55% Task A retention
- **30x improvement** in retention

### 2. Computational Efficiency

Despite sophisticated gating mechanism:
- Only 3.2% time overhead
- Zero memory overhead
- No additional parameters

### 3. Biological Plausibility

Clear mapping to neuroscience:
- NMDA receptor → Coincidence detection
- LTP/LTD → Selective updates
- Synaptic tagging → Eligibility traces
- Metaplasticity → Adaptive thresholds

### 4. Limitations Identified

- **Requires replay**: Without experience replay, retention is near 0%
- **Not a silver bullet**: Works best as complement to other methods
- **Task boundaries needed**: Currently requires knowing when tasks change

## Evolution to DendriticPlasticity

While SelectivePlasticity focuses on **learning rules** (backward pass optimization), we explored combining it with **structural enhancements** (forward pass dendrites).

**See our companion example**: [DendriticPlasticity](../continualLearning/) demonstrates the integration with PerforatedAI's artificial dendrites.

## Reproducing Results

### Option 1: Run Locally

```bash
python experiments/split_mnist.py --epochs 5 --replay-buffer 2000
```

### Option 2: One-Click Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/demo_selective_plasticity.ipynb)

## API Reference

### SelectivePlasticityOptimizer

```python
SelectivePlasticityOptimizer(
    params,                    # Model parameters
    lr=0.001,                  # Learning rate
    trace_decay=0.99,          # Eligibility trace decay (β)
    plasticity_k=5.0,          # Gate temperature (higher = sharper)
    warmup_steps=1000,         # Steps before gating activates
    surprise_weight=0.3,       # Weight for surprise modulation
    alignment_threshold=0.1    # Minimum alignment for gate activation
)
```

**Key Parameters:**
- `trace_decay`: Controls how quickly eligibility fades (0.95-0.999)
- `plasticity_k`: Sharpness of gating function (1.0-10.0)
- `warmup_steps`: Calibration period (500-2000 recommended)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{selectiveplasticity2025,
  title={Selective Synaptic Plasticity for Continual Learning},
  author={},
  year={2025},
  url={https://github.com/perforatedai/PerforatedAI-Examples}
}
```

## Related Work

- **EWC** (Elastic Weight Consolidation): Uses Fisher information
- **SI** (Synaptic Intelligence): Online importance estimation
- **MAS** (Memory Aware Synapses): Output-based importance
- **PackNet**: Network pruning for continual learning

**Our difference**: Biological grounding + zero overhead + replay synergy

## Future Directions

1. **Remove task boundaries**: Automatic task detection
2. **Hierarchical protection**: Different strategies per layer
3. **Meta-learning integration**: Learn optimal gate parameters
4. **Theoretical analysis**: Convergence guarantees and forgetting bounds

## Acknowledgments

Built on research in:
- Neuroscience (NMDA receptors, synaptic plasticity)
- Continual learning (catastrophic forgetting)
- Biologically-inspired AI (Hebbian learning, eligibility traces)

## License

Apache 2.0

---

**Questions or Issues?** Open an issue on GitHub or see our [DendriticPlasticity](../continualLearning/) example for integration with PerforatedAI.
