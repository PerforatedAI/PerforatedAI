# Dendritic MobileNet Project

## Overview
This project enhances a **MobileNetV3-Small** computer vision model by integrating **Dendritic Computing** concepts using the [PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) library. 

Developed for the **Perforated AI Dendritic Optimization Hackathon**, this project demonstrates how adding artificial dendrites to standard PyTorch models can potentially improve accuracy or parameter efficiency ("smarter, smaller, cheaper") for edge vision tasks.

## Key Features
- **Dendritic Integration**: Replaces standard linear output layers with `PAINeuronModules` containing biological-inspired dendrites.
- **Hyperparameter Optimization**: Includes a full Weights & Biases (WandB) sweep configuration to find the optimal dendrite count.
- **PyTorch Native**: Built on standard `torch` and `torchvision` libraries, making it easy to integrate into existing workflows.

## Installation

1.  **Clone the Repository** (if you haven't already):
    ```bash
    git clone https://github.com/PerforatedAI/PerforatedAI.git
    cd PerforatedAI/Examples/hackathonProjects/dendritic_mobilenet
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model with default settings (or as part of a sweep):

```bash
# Ensure PerforatedAI is in your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/PerforatedAI

python train.py
```

### Hyperparameter Sweep
This project is configured to run a **WandB Sweep** automatically. The `train.py` script initializes a sweep controller and agent to test:
- **Dendrite Counts**: [2, 4, 8]
- **Batch Sizes**: [32, 64]
- **Learning Rates**: Uniform distribution [0.0001, 0.005]

### Optimal Hyperparameters
Based on our experimental sweeps, the following configuration yielded the best results (82.57% Validation Accuracy):
- **Dendrite Count**: 8
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: ~50 (for full convergence, demo uses fewer)

[W&B Sweep Report](https://wandb.ai/raghunathsundar1-chennai-institute-of-technology/hackathon-dendritic-vision)


## Results

| Metric | Baseline (MobileNetV3-Small) | Dendritic (8 Dendrites) |
| :--- | :--- | :--- |
| **Parameters** | ~1.5M | ~4.1M |
| **Validation Accuracy** | *67.4% (Typical)* | **82.57%** |

*Note: Baseline accuracy is approximate for MobileNetV3-Small on CIFAR-10 without extensive tuning. Dendritic results are from our actual hackathon training runs.*

## Project Structure
- `model.py`: Defines the `DendriticVisionModel` class.
- `train.py`: Main training script with WandB integration.
- `requirements.txt`: Project dependencies.
