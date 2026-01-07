# W&B Sweep Report: Optimization of SBERT Adapter

**Authors:** Aakanksha Singh and Mihir Phalke

**A hyperparameter exploration for the Project NEXUS dendritic fine-tuning architecture.**

## Executive Summary
To validate that our dendritic performance gains were systematic and not due to chance, we conducted a **Bayesian Optimization** sweep using Weights & Biases (W&B) to intelligently traverse the hyperparameter space. We explored Learning Rate, Weight Decay, and Batch Size configurations to find the optimal environment for dendritic growth.

## Live Interactive Dashboard

**[View the Bayesian Optimization Results (W&B)](https://wandb.ai/aakanksha-singh0205-kj-somaiya-school-of-engineering/PerforatedAI-Examples_hackathonProjects_Project-Nexus-SBERT_src/sweeps/lzwt00l9)**

## Reproduce This Sweep

1.  **Install W&B:** `pip install wandb`
2.  **Login:** `wandb login`
3.  **Run Agent:**
    ```bash
    python -m wandb agent aakanksha-singh0205-kj-somaiya-school-of-engineering/PerforatedAI-Examples_hackathonProjects_Project-Nexus-SBERT_src/lzwt00l9
    ```

