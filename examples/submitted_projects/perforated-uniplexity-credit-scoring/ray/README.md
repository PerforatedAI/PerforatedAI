# ⚡ Ray Distributed Training for Dendritic Model

This folder contains the implementation for scaling the Dendritic Model training using [Ray](https://docs.ray.io/en/latest/).

## ⚠️ Requirements
*   **Python 3.8 - 3.11** (Ray does not yet support Python 3.13)
*   **OS**: Linux, macOS, or Windows (WSL recommended for heavy workloads)

## 🛠️ Setup
**Prerequisite:** You must have **Python 3.11** installed.
*   Check versions: `py --list` (Windows) or `python3 --version` (Mac/Linux)
*   **If missing**: [Download Python 3.11](https://www.python.org/downloads/release/python-3119/)

### Option A: Using Conda (Recommended)
```bash
conda create -n ray_env python=3.11 -y
conda activate ray_env
pip install "ray[default]" torch
```

### Option B: Using Standard Python (Windows)
If you installed Python 3.11 separately:
```bash
# 1. Create venv using specific Python version
py -3.11 -m venv ray_venv

# 2. Activate
.\ray_venv\Scripts\activate

# 3. Install
pip install "ray[default]" torch
```

## 🚀 Running Distributed Training
Trains the **Optimized Dendritic Model (529 params)** across available cores/GPUs.

```bash
python ray/ray_trainer.py
```
*Expected Result:* ~98.3% Accuracy, identical to the local "Efficiency King" model but faster on large datasets.

## 📈 Running Hyperparameter Sweep
Uses Ray Tune to find the best configuration.

```bash
python ray/ray_sweep.py
```

## 📁 Files
*   `ray_trainer.py`: Main distributed training entry point.
*   `ray_sweep.py`: Hyperparameter optimization script.
*   `ray_utils.py`: Helper functions for data sharding and metrics.
