# Phase 1: Baseline Training - Quick Start Guide

This guide will help you complete Phase 1 of the hackathon project.

## Prerequisites

You should have already:
- ✅ Downloaded ESC-50 dataset to `data/ESC-50/`
- ✅ Activated your virtual environment
- ✅ Installed PerforatedAI: `pip install -e .` (from repo root)

## Step 1: Install Dependencies

```bash
cd Examples/hackathonProjects/ravi-audio-classification
pip install -r requirements.txt
```

This installs: PyTorch, librosa, MLflow, and other dependencies.

## Step 2: Preprocess the Dataset

Convert audio files to mel-spectrograms (one-time operation, ~2-3 minutes):

```bash
python preprocess.py
```

This creates:
- `preprocessed/X_train.npy`, `y_train.npy` (training data)
- `preprocessed/X_val.npy`, `y_val.npy` (validation data)
- `preprocessed/X_test.npy`, `y_test.npy` (test data)
- `preprocessed/label_mapping.pkl` (class names)
- `preprocessed/config.pkl` (preprocessing settings)

Expected output:
```
Total samples: 2000
Train set: (1280, 128, 216)
Validation set: (320, 128, 216)
Test set: (400, 128, 216)
```

## Step 3: Train Baseline Model

Train the CNN without dendrites (~1-2 hours on M4 Mac):

```bash
python train_baseline.py
```

Optional arguments:
- `--batch_size 32` (default)
- `--lr 0.001` (learning rate)
- `--epochs 100` (max epochs)
- `--patience 10` (early stopping)

This will:
- Train AudioCNN on spectrograms
- Use early stopping based on validation accuracy
- Save best model to `models/baseline_best.pt`
- Log experiments to MLflow (`mlruns/` folder)
- Generate confusion matrix
- Save results to `models/baseline_results.json`

Expected baseline accuracy: **60-75%** (ESC-50 is challenging!)

## Step 4: View Results

### Option A: Check JSON Results

```bash
cat models/baseline_results.json
```

You'll see:
```json
{
  "model": "Baseline CNN",
  "test_accuracy": 72.5,
  "test_loss": 0.89,
  "best_val_accuracy": 74.2,
  "num_parameters": 1234567,
  "epochs_trained": 45
}
```

### Option B: View MLflow UI

In a new terminal:

```bash
cd Examples/hackathonProjects/ravi-audio-classification
mlflow ui --port 5000
```

Then open http://localhost:5000 in your browser to see:
- Training/validation curves
- Hyperparameters
- Confusion matrix
- Model artifacts

## Troubleshooting

### Error: "No module named 'perforatedai'"
Solution: Install PerforatedAI from repo root:
```bash
cd /Users/ravirai/GitHub/PerforatedAI
pip install -e .
```

### Error: "File not found: data/ESC-50"
Solution: Download ESC-50 first:
```bash
cd data
curl -L -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip esc50.zip && mv ESC-50-master ESC-50 && rm esc50.zip
```

### MPS device not found (on M4 Mac)
The code will automatically fall back to CPU if MPS isn't available. Training will just be slower.

## What's Next?

Once Phase 1 is complete, you'll have:
- ✅ Preprocessed spectrograms saved
- ✅ Baseline CNN trained
- ✅ Baseline accuracy recorded
- ✅ MLflow experiments logged

**Phase 2** will add dendrites to improve this baseline!

## File Structure After Phase 1

```
ravi-audio-classification/
├── requirements.txt           ✓ Created
├── preprocess.py             ✓ Created
├── train_baseline.py         ✓ Created
├── utils/                    ✓ Created
│   ├── __init__.py
│   ├── model.py
│   ├── data_utils.py
│   └── metrics.py
├── data/ESC-50/              ✓ Downloaded
├── preprocessed/             ✓ Created by preprocess.py
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│   ├── label_mapping.pkl
│   └── config.pkl
├── models/                   ✓ Created by training
│   ├── baseline_best.pt
│   ├── baseline_results.json
│   └── baseline_confusion_matrix.png
└── mlruns/                   ✓ Created by MLflow
```

Ready for Phase 2! 🚀
