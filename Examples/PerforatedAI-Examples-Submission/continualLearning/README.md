# DendriticPlasticity: Combining Artificial Dendrites with Selective Synaptic Plasticity for Continual Learning

This example demonstrates the integration of **Perforated AI's Artificial Dendrites** with **Selective Plasticity** mechanisms for continual learning tasks.

## Background

This work builds on our [SelectivePlasticity](../selectivePlasticity/) optimizer, which achieved **86.55% task retention** through NMDA-inspired synaptic protection. While SelectivePlasticity focuses on **learning rules** (backward pass optimization), we explored combining it with **structural enhancements** (forward pass dendrites from PerforatedAI).

**Key Insight**: Dendrites and synaptic plasticity work together in biological brains - why not in artificial networks?

## Overview

### The Problem: Catastrophic Forgetting
When neural networks learn new tasks sequentially, they tend to forget previously learned information - a phenomenon known as catastrophic forgetting. This is a fundamental challenge in continual learning.

### Our Approach: DendriticPlasticity
We combine two biologically-inspired mechanisms:

1. **Artificial Dendrites (PAI)**: Enhance feature representation through dendritic computation
2. **Selective Plasticity**: Protect important synapses from being overwritten

```
┌─────────────────────────────────────────────────────────┐
│                   DendriticPlasticity                   │
├─────────────────────────────────────────────────────────┤
│  Forward Pass                  │  Backward Pass         │
│  ┌──────────────────────────┐  │  ┌───────────────────┐ │
│  │  Perforated AI           │  │  │ SelectivePlasticity│ │
│  │  - Dendritic Modules     │  │  │ - Eligibility Trace│ │
│  │  - Enhanced Computation  │  │  │ - Protection Gate  │ │
│  └──────────────────────────┘  │  └───────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Biological Basis

### Dendrites (Forward Pass Enhancement)
- Biological neurons integrate signals through dendrites before reaching the soma
- Dendrites perform nonlinear computations, not just passive signal transmission
- PAI adds artificial dendritic modules that enhance the network's representational capacity

### Selective Plasticity (Backward Pass Control)
- Inspired by NMDA receptor coincidence detection
- Uses eligibility traces to track parameter usage frequency
- Protects "busy" parameters from being overwritten
- Based on LTP/LTD (Long-Term Potentiation/Depression) mechanisms

## Results

### Split MNIST Experiment (2 Tasks)

| Method | Task 1 | Task 2 | Average | Forgetting | Parameters |
|--------|--------|--------|---------|------------|------------|
| Baseline (Adam) | 0.00% | 98.13% | 49.06% | 98.79% | 269,322 |
| Baseline + Replay | 2.82% | 98.11% | 50.46% | 96.40% | 269,322 |
| PAI (Dendrites) | 0.00% | 98.17% | 49.08% | 99.03% | 538,644 |
| **PAI + Replay** | **2.96%** | **98.35%** | **50.66%** | 96.36% | 538,644 |
| SelectivePlasticity | 0.00% | 88.40% | 44.20% | 93.66% | 269,322 |
| DendriticPlasticity | 0.02% | 88.69% | 44.35% | 93.54% | 538,644 |

### Key Findings

1. **PAI improves single-task performance**: With dendrites, Task 2 accuracy reaches 98.35% vs 98.13% baseline

2. **Replay is essential**: Without experience replay, all methods suffer from catastrophic forgetting (~99%)

3. **PAI + Replay is the best combination**: Achieves the highest average accuracy (50.66%)

4. **Selective Plasticity reduces forgetting rate**: From ~99% to ~94%, though the absolute retention is still low for this extreme task

## Installation

```bash
# Install PerforatedAI
cd PerforatedAI
pip install -e .

# Run the experiment
python continual_learning_experiment.py
```

## Usage

### Basic Usage

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from selective_plasticity import SelectivePlasticityOptimizer

# Create model
model = YourModel()

# Add dendrites
GPA.pc.set_testing_dendrite_capacity(True)
model = UPA.initialize_pai(model)

# Use selective plasticity optimizer
optimizer = SelectivePlasticityOptimizer(
    model.parameters(),
    lr=0.001,
    trace_decay=0.99,
    plasticity_k=5.0
)

# Train as usual
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### With Experience Replay

```python
# See continual_learning_experiment.py for full implementation
memory = MemoryBuffer(capacity=2000)

for task in tasks:
    for data, target in task.train_loader:
        # Add replay data
        if len(memory) > 0:
            replay_data, replay_targets = memory.sample(32)
            data = torch.cat([data, replay_data])
            target = torch.cat([target, replay_targets])

        # Train
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

        # Update buffer
        memory.add_batch(data, target)
```

## Files

```
continualLearning/
├── README.md                          # This file
├── continual_learning_experiment.py   # Main experiment script
├── selective_plasticity/
│   ├── __init__.py
│   ├── selective_plasticity_optimizer.py  # Core optimizer
│   ├── coincidence_gate.py            # Gate functions
│   └── synapse_state.py               # State tracking
└── results/
    └── dendritic_plasticity_results.json
```

## Future Work

1. **Task-aware replay**: Prioritize replay samples from tasks that show most forgetting
2. **Adaptive protection strength**: Dynamically adjust plasticity based on task similarity
3. **Dendritic task routing**: Use different dendrite branches for different tasks
4. **Larger scale experiments**: Test on CIFAR-100, ImageNet subsets

## References

1. Perforated AI: Adding Artificial Dendrites to Neural Networks
2. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC)
3. Zenke et al., "Continual Learning Through Synaptic Intelligence" (SI)
4. NMDA Receptor and Synaptic Plasticity in Neuroscience

## See Also

- **[SelectivePlasticity](../selectivePlasticity/)** - Standalone optimizer achieving 86.55% task retention
  - Detailed biological grounding and mechanism explanation
  - Permuted MNIST and hyperparameter experiments
  - Complete API documentation

- **[PerforatedAI Documentation](https://github.com/perforatedai/PerforatedAI)** - Artificial dendrite framework
  - How to add dendrites to any PyTorch model
  - Dendrite capacity and configuration options

## Authors

- DendriticPlasticity Team
- Built on Perforated AI framework

## License

Apache 2.0 (following PerforatedAI license)
