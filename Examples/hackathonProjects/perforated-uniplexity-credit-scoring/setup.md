# 🛠️ Complete Setup & Execution Guide

Welcome to the **Perforated Uniplexity Credit Scoring** project. This comprehensive guide covers every step from environment setup to distributed training and edge deployment.

---

## 1. 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9+** (Tested on 3.10)
- **Git** (for version control)
- **Visual Studio Code** (Recommended) or any terminal

---

## 2. 🐍 Environment Setup

Isolate your dependencies to prevent conflicts.

### Windows (PowerShell)
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate the environment
.\venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

*> **Tip:** If you see `(venv)` in your terminal prompt, you're good to go!*

---

## 3. 📦 Install Dependencies

Install all required Machine Learning libraries (PyTorch, Lightning, Transformers, Ray, etc.).

```bash
pip install -r requirements.txt
```

### ⚠️ Troubleshooting Installation
- **Windows Users**: If `ray` installation fails, you can exclude it and strictly run standard PyTorch scripts. However, mostly it works fine on modern Windows.
- **Torch CPU vs CUDA**: By default, pip installs the CPU version of PyTorch on some systems. If you have an NVIDIA GPU, verify your installation at [pytorch.org](https://pytorch.org/).

---

## 4. 📊 Data Pipeline (Crucial Step)

You likely started with a project skeleton. You **must** generate the data before training.

### Step A: Generate Synthetic SME Data
Creates a dataset of 5,000 SMEs based on real-world ERP patterns (Revenue, Expenses, Inventory, etc.).
```bash
python data/generate_data.py
```
*Output: `data/raw/sme_erp_synthetic.csv`*

### Step B: Preprocess & Normalize
Converts CSV data into normalized PyTorch tensors and JSON files for HuggingFace.
```bash
python data/preprocess.py
```
*Output: `data/processed/features.pt`, `labels.pt`, and `hf_dataset/`*

---

## 5. 🧠 Training the Models

We support **4 different training frameworks** to demonstrate flexibility. Choose one:

### Option A: Standard PyTorch (Easiest)
Simple training loop. Good for debugging.
```bash
python training/train_dendritic.py
```
*Checkpoints saved to: `models/dendritic_checkpoint.pt`*

### Option B: PyTorch Lightning (Best Practice)
Structured training with automatic logging and checkpointing.
```bash
python training/train_lightning.py
```
*Logs saved to: `lightning_logs/`*

### Option C: HuggingFace Trainer (Tabular)
Demonstrates integration with the HF ecosystem.
```bash
python training/train_hf_tabular.py
```
*Checkpoints saved to: `models/hf_checkpoints/`*

### Option D: Ray Distributed (Advanced) 🚀
Scales training across multiple CPU cores or GPUs.
```bash
python ray/ray_trainer.py
```
*Results saved to: `models/ray_results/`*

---

## 6. 🔍 Hyperparameter Tuning

Want to find the best model architecture? We have automated sweeps.

### W&B Sweeps (Weights & Biases)
Requires a W&B account (optional).
```bash
python sweeps/run_sweep.py
```

### Ray Tune (Local)
Runs Bayesian Optimization to find the best learning rate and hidden dimensions.
```bash
python ray/ray_sweep.py
```
*Output: Best config printed to console and saved CSV.*

---

## 7. 📉 Evaluation & Results

Compute definitive metrics (Accuracy, ROC-AUC) on the test set.

```bash
python training/evaluate.py
```

**Expected Output:**
```text
Baseline Model Metrics:
  MSE: 0.1824
  Accuracy: 0.7150

Dendritic Model Metrics:
  MSE: 0.1105
  Accuracy: 0.8350
```

---

## 8. 📱 Edge Deployment & Benchmarking

Prove that the model can run on a POS device.

### Benchmark Latency & Memory
Measure inference speed (ms) and RAM usage (MB).
```bash
python edge/benchmark_latency.py
python edge/benchmark_memory.py
```
*Results appended to: `reports/edge_benchmarks.csv`*

### Export to TorchScript
Convert the model to a serialized format for mobile (iOS/Android) deployment.
```bash
python edge/export_torchscript.py
```
*Output: `models/dendritic_torchscript.pt`*

---

## 9. ⚙️ Configuration

You can tweak model parameters without changing code.
Edit **`models/model_config.yaml`**:

```yaml
dendritic:
  input_dim: 5          # Number of features
  hidden_dim: 64        # Size of hidden layers
  num_dendrites: 4      # Branching factor
```

---

## 10. 📂 Project Structure Overview

- **`data/`**: Raw CSVs and processed `.pt` files.
- **`models/`**: Neural network architectures (`dendritic_model.py`, `baseline_model.py`).
- **`training/`**: Scripts for training and evaluation.
- **`ray/`**: Distributed training logic.
- **`edge/`**: Benchmarking and export scripts.
- **`diagrams/`**: System architecture visuals.
- **`reports/`**: CSVs and Markdown reports of experiments.

---

## ❓ FAQ / Troubleshooting

**Q: `FileNotFoundError: ... features.pt`**
A: You skipped Step 4. Run `python data/preprocess.py` first.

**Q: Ray output is messy or hanging.**
A: Ray can be verbose. If it hangs on Windows, ensure you allow Python through the firewall when prompted.

**Q: How do I view the diagrams?**
A: Open the `diagrams/` folder. You'll find:
- `erp_to_credit_pipeline.png` (System Flow)
- `dendrite_architecture.png` (Model Internal)
- `deployment_targets.png` (Edge/Cloud)
