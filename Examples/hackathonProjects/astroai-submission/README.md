# AstroAI: Exoplanet Transit Detection with Perforated AI Dendritic Optimization

**This project uses Perforated AI's dendritic optimization technology** to enhance neural network performance for exoplanet transit detection from light curve data.

## Overview

AstroAI is an AI-powered exoplanet detection system that analyzes synthetic telescope light curves to identify planetary transits. This project demonstrates the application of **Perforated AI's dendritic optimization** to improve model accuracy and efficiency in astronomical signal detection.

Inspired by NASA's Kepler and TESS missions, AstroAI makes space data analysis accessible while showcasing cutting-edge neural network optimization techniques.

## Hackathon Submission

This project is submitted to the **PyTorch Dendritic Optimization Hackathon** hosted by Perforated AI.

- **Devpost**: [PyTorch Dendritic Optimization Hackathon](https://pytorch-dendritic-optimization.devpost.com/)
- **Submission Repository**: [PerforatedAI/hackathonProjects](https://github.com/PerforatedAI/PerforatedAI/tree/main/Examples/hackathonProjects)

## Features

- **Synthetic Data Generation**: Generate realistic exoplanet transit light curves with customizable parameters
- **Multiple Model Architectures**: MLP, CNN, and LSTM models for transit detection
- **Perforated AI Integration**: Dendritic optimization for improved model performance
- **Interactive Dashboard**: Streamlit web app for real-time simulations
- **Comprehensive Training Pipeline**: Baseline vs PAI comparison with metrics and visualizations

## Results

### Baseline vs Perforated AI Comparison

| Metric | Baseline | PAI (Dendritic) | Improvement |
|--------|----------|-----------------|-------------|
| Test Accuracy | TBD% | TBD% | TBD% |
| Parameters | TBD | TBD | TBD |
| Training Time | TBD | TBD | TBD |

*Results will be updated after running experiments*

![Training Comparison](results/training_comparison.png)
![Accuracy Comparison](results/accuracy_comparison.png)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/astroai-perforatedai.git
cd astroai-perforatedai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Perforated AI
pip install perforatedai
```

## Usage

### Training with Perforated AI

```bash
# Train both baseline and PAI models for comparison
python train.py --epochs 50 --samples 5000 --model mlp

# Train only with Perforated AI
python train.py --pai_only --epochs 100 --model cnn

# Train only baseline for comparison
python train.py --baseline_only --epochs 50
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--samples` | 5000 | Number of training samples |
| `--model` | mlp | Model type: mlp, cnn |
| `--device` | auto | Device: cuda or cpu |
| `--baseline_only` | False | Train only baseline model |
| `--pai_only` | False | Train only PAI model |

### Interactive Demo

```bash
# Launch Streamlit app
streamlit run app.py
```

## Project Structure

```
astroai-perforatedai/
├── app.py              # Streamlit interactive demo
├── model.py            # Neural network architectures
├── simulator.py        # Light curve simulation
├── train.py            # Training script with PAI integration
├── requirements.txt    # Dependencies
├── README.md           # This file
├── results/            # Training results and plots
│   ├── training_comparison.png
│   ├── accuracy_comparison.png
│   ├── baseline_model.pth
│   └── pai_model.pth
└── assets/             # Demo assets
    └── sample_light_curve.png
```

## How Perforated AI Integration Works

### Key Integration Points

1. **Model Initialization**
```python
import perforatedai as pai

model = TransitDetector()
model = pai.initialize_pai(
    model,
    doing_pai=True,
    save_name='PAI_AstroAI',
    making_graphs=True,
    maximizing_score=True
)
```

2. **Optimizer Setup**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
pai.pai_tracker.set_optimizer(optimizer, scheduler)
```

3. **Training Loop Integration**
```python
# After each validation epoch
model, optimizer, training_complete = pai.pai_tracker.add_validation_score(
    val_accuracy, model, optimizer
)
```

### Benefits of Dendritic Optimization

- **Improved Accuracy**: Dendrites add computational capacity where needed
- **Efficient Learning**: Automatic architecture adaptation during training
- **Better Generalization**: Enhanced feature representation

## Technical Details

### Light Curve Simulation

The simulator generates realistic exoplanet transit signals with:
- Configurable orbital periods (2-30 days)
- Variable planet-to-star radius ratios (0.02-0.15)
- Gaussian noise modeling
- Stellar variability trends
- Limb darkening approximation

### Model Architectures

1. **TransitDetector (MLP)**: 4-layer fully connected network with batch normalization
2. **TransitDetectorCNN**: 1D convolutional network for sequential pattern detection
3. **TransitDetectorLSTM**: Bidirectional LSTM for temporal analysis

## Project Story

### Inspiration
Exoplanet discovery through transit photometry has revolutionized our understanding of planetary systems. We wanted to combine this exciting field with cutting-edge AI optimization techniques.

### What We Built
An end-to-end system that generates synthetic transit data, trains neural networks with dendritic optimization, and provides interactive visualizations.

### Challenges
- Modeling realistic noise patterns in light curves
- Balancing model complexity with training efficiency
- Integrating Perforated AI with custom architectures

### What We Learned
- Dendritic optimization can improve model performance without manual architecture tuning
- Time-series astronomical data benefits from specialized preprocessing
- The importance of proper validation in scientific ML applications

### What's Next
- Integration with real NASA Kepler/TESS data
- Multi-planet detection capabilities
- Deployment as a web service for citizen science

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Perforated AI** for the dendritic optimization technology
- NASA Kepler and TESS missions for inspiration
- PyTorch team for the deep learning framework

## Links

- [Perforated AI Documentation](https://www.perforatedai.com/docs)
- [PyTorch Dendritic Optimization Hackathon](https://pytorch-dendritic-optimization.devpost.com/)
- [Perforated AI GitHub](https://github.com/PerforatedAI/PerforatedAI)
