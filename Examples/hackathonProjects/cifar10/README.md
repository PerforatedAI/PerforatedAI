# CIFAR-10: Baseline vs Dendritic Optimization Comparison

## Overview

This project demonstrates a comprehensive comparison between a traditional convolutional neural network (CNN) and the same architecture enhanced with PerforatedAI's dendritic optimization technology. The experiment is conducted on the CIFAR-10 dataset, a well-established benchmark for image classification tasks.

The notebook provides a full side-by-side comparison showing how dendritic optimization can improve model accuracy while maintaining reasonable computational efficiency.

## What is Dendritic Optimization?

Dendritic optimization is a technique that adds artificial dendrites to neural network layers, allowing models to dynamically grow additional connections during training. This mimics biological neurons' ability to form new connections based on learning experiences. Unlike static neural networks that have a fixed architecture, dendritic models can:

- **Dynamically expand capacity** when performance plateaus
- **Learn complex feature relationships** more effectively
- **Achieve higher accuracy** with controlled parameter growth
- **Adapt to training dynamics** by adding dendrites strategically

## Dataset

**CIFAR-10** is a classic computer vision dataset consisting of:

- 60,000 32x32 color images in 10 classes
- 50,000 training images
- 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Experiment Setup

### Model Architecture

Both models use the same base architecture - a SimpleCNN with:

- 2 convolutional blocks with batch normalization and dropout
- MaxPooling layers
- Fully connected classifier
- Architecture: Conv(16) → Conv(16) → Pool → Conv(32) → Conv(32) → Pool → FC(256) → FC(10)

### Training Configuration

- **Epochs**: 40
- **Batch Size**: 128
- **Learning Rate**: 0.006 (Adam optimizer)
- **Scheduler**: ReduceLROnPlateau (patience: 5, factor: 0.5)
- **Data Augmentation**: Random crop, horizontal flip, normalization

### PAI Configuration

The dendritic model uses aggressive growth settings:

- **Switch Mode**: DOING_HISTORY (monitors training history for growth opportunities)
- **Epochs to Switch**: 2 (checks for dendritic growth every 2 epochs)
- **Max Dendrites**: 30 per layer
- **Max Dendrite Tries**: 16
- **Improvement Threshold**: 0.1% (triggers dendritic expansion when improvement slows)
- **Converted Layers**: Both Conv2d and Linear layers

## Results

The experiment produces comprehensive visualizations and metrics comparing:

### Accuracy Metrics

- **Test Accuracy Progression**: Epoch-by-epoch comparison
- **Best Test Accuracy**: Peak performance achieved by each model
- **Final Test Accuracy**: Performance at the end of training
- **Per-Class Accuracy**: Breakdown by each of the 10 CIFAR-10 classes

### Performance Analysis

- **Precision, Recall, F1-Score**: Comprehensive classification metrics
- **Error Reduction**: Percentage reduction in classification errors
- **Learning Curves**: Training vs test accuracy over time
- **Generalization Gap**: Overfitting analysis

### Efficiency Metrics

- **Parameter Count**: Total trainable parameters in each model
- **Parameter Efficiency**: Accuracy per million parameters
- **Training Time**: Total and per-epoch training duration
- **Dendrite Growth**: Number of dendrites added over time

### Visualizations

The notebook generates extensive visualizations including:

- Accuracy and loss comparison plots
- Dendrite growth curves
- Per-class accuracy breakdowns
- Training efficiency analysis
- Comprehensive summary dashboards
- Radar charts for multi-metric comparison

## Expected Outcomes

Based on the notebook configuration, you can expect:

1. **Improved Accuracy**: The dendritic model typically achieves higher test accuracy than the baseline
2. **Dynamic Growth**: Dendrites are added progressively when training progress stalls
3. **Better Feature Learning**: Enhanced ability to learn complex spatial and class relationships
4. **Controlled Parameter Growth**: Strategic addition of parameters where they provide most benefit

## Installation & Usage

### Prerequisites

```bash
pip install perforatedai
pip install perforatedbp  # Optional: Enhanced backpropagation
pip install torch torchvision
pip install numpy matplotlib scikit-learn
```

### Running the Notebook

Open the Jupyter notebook:

```bash
jupyter notebook cifar10_dendritic_comparison.ipynb
```

Or use VS Code with Jupyter extension support.

### Execution

Run all cells sequentially. The notebook is organized into sections:

1. **Configuration & Setup**: Install dependencies, configure parameters
2. **Baseline Model Training**: Train traditional CNN
3. **Dendritic Optimization Model**: Train PAI-enhanced model
4. **Comparison**: Comprehensive analysis and visualization

## Key Features

- ✅ **Reproducible**: Fixed random seeds for consistent results
- ✅ **Comprehensive Metrics**: Accuracy, precision, recall, F1-score tracking
- ✅ **Device Agnostic**: Supports CUDA, MPS (Apple Silicon), and CPU
- ✅ **Rich Visualizations**: 10+ visualization types for in-depth analysis
- ✅ **Per-Class Analysis**: Detailed breakdown of performance by class
- ✅ **Efficiency Analysis**: Training time and parameter efficiency metrics

## Customization

You can modify the global configuration parameters at the top of the notebook:

```python
# Training parameters
NUM_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.006

# PAI parameters
PAI_MAX_DENDRITES = 30
PAI_EPOCHS_TO_SWITCH = 2
PAI_IMPROVEMENT_THRESHOLD = 0.1
```

## Understanding the Results

The notebook produces several key metrics:

- **Accuracy Gain**: Direct percentage point improvement (e.g., +2.5%)
- **Error Reduction**: Percentage of remaining errors eliminated
- **Parameter Overhead**: Additional parameters used by dendritic model
- **Dendrites Added**: Total number of dendritic connections created

A successful dendritic optimization will show:

- Higher final accuracy than baseline
- Steady dendrite growth during training
- Better per-class accuracy across most classes
- Manageable parameter increase for the accuracy gain

## Notebook Sections

The notebook is structured as follows:

### Part 1: Configuration & Setup

- Installation of dependencies (PerforatedAI, PyTorch)
- Import statements and environment detection
- Global configuration parameters
- Device setup (CUDA/MPS/CPU)
- Data loading with augmentation

### Part 2: Baseline Model Training

- Initialize standard SimpleCNN
- Train for full epoch count
- Track accuracy, loss, precision, recall, F1
- Visualize training dynamics

### Part 3: Dendritic Optimization Model (PerforatedAI)

- Configure PAI settings
- Initialize model with dendrite scaffolding
- Train with dynamic dendrite growth
- Monitor dendrite additions and parameter growth

### Part 4: Comparison - Baseline vs Dendritic Optimization

- Side-by-side accuracy and loss comparison
- Summary statistics and performance gains
- Multi-panel visualizations
- Statistical metrics and radar charts
- Per-class accuracy analysis

### Part 5: Efficiency & Resource Comparison

- Training time analysis
- Parameter efficiency metrics
- Resource utilization comparison

## Technical Notes

- The notebook uses both Conv2d and Linear layer conversion to dendrites
- Perforated backpropagation can be toggled (currently disabled for compatibility)
- Batch normalization layers are tracked but not converted
- The model automatically handles dimensionality for different layer types
- Output dimensions are manually configured for Conv2d `[-1, 0, -1, -1]` and Linear `[-1, 0]`

## Troubleshooting

If the dendritic model doesn't show improvement:

- Increase `PAI_MAX_DENDRITES` to allow more growth
- Decrease `PAI_IMPROVEMENT_THRESHOLD` to trigger growth earlier
- Increase `NUM_EPOCHS` to allow more training time
- Adjust `PAI_EPOCHS_TO_SWITCH` for more/less frequent growth checks
- Try increasing `PAI_MAX_DENDRITE_TRIES` for more connection attempts

## Common Issues

**Issue**: Model not adding dendrites

- Check that `PAI_SWITCH_MODE` is set to `"DOING_HISTORY"`
- Verify improvement threshold isn't too aggressive
- Ensure sufficient epochs for growth opportunities

**Issue**: Out of memory errors

- Reduce `BATCH_SIZE`
- Decrease `PAI_MAX_DENDRITES`
- Use gradient accumulation if needed

**Issue**: Training too slow

- Reduce `NUM_WORKERS` if using CPU
- Decrease `BATCH_SIZE` if memory constrained
- Consider using smaller model or fewer dendrites

## References

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PerforatedAI Documentation: See `/API/` directory in repository
- PyTorch Documentation: https://pytorch.org/docs/

## License

See LICENSE file in the root directory of the repository.
