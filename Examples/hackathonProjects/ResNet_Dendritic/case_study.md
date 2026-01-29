# Dendritic Optimization - Case Study

## Project Overview
**Project Name**: Optimizing ResNet50 for CIFAR-100
**Team Name**: Perforated [Team Name]
**Description**: We applied PerforatedAI's dendritic optimization to a standard ResNet50 model to improve efficiency and accuracy on the CIFAR-100 dataset.

## The Challenge
ResNet50 is a powerful model but can be parameter-heavy. We aimed to reduce the computational cost and improve the learning efficiency without sacrificing accuracy, or even improving it, by using artificial dendrites.

## Methodology
- **Baseline**: Standard ResNet50 trained on CIFAR-100 for X epochs.
- **Optimization**: We integrated PerforatedAI to dynamically add dendrites during training.
- **Tuning**: We used Weights & Biases sweeps to optimize the hyperparameters (learning rate, dendrite candidates, improvement thresholds).

## Results
| Metric | Baseline | Dendritic | Improvement |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | XX.X% | XX.X% | +X.X% |
| **Parameters** | 23.5M | XX.XM | -X.X% |
| **Inference Time**| XX ms | XX ms | -X.X% |

## Key Findings
[Narrative about what happened, e.g. dendrites helped initially then pruned, or found a better local minima, etc.]

## Business Impact
This optimization demonstrates potential for deploying complex models on edge devices with limited compute, or reducing cloud training costs by converging faster.

## Reproducibility
- [GitHub PR Link](https://github.com/PerforatedAI/PerforatedAI/pull/...)
- [W&B Sweep Report](https://wandb.ai/...)
