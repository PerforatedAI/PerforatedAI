"""
Configuration file for ESC-50 audio classification project.
All hyperparameters and paths are defined here.
"""

# ============================================================================
# Data Paths
# ============================================================================
DATA_DIR = 'data/ESC-50'
OUTPUT_DIR = 'preprocessed'
MODELS_DIR = 'models'

# ============================================================================
# Preprocessing Configuration
# ============================================================================
PREPROCESSING = {
    'sample_rate': 22050,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'duration': 5.0,  # seconds
    'test_fold': 5,  # ESC-50 standard: fold 5 for test
    'val_split': 0.2,  # 20% of train_val for validation
    'random_state': 42,  # For reproducibility
}

# ============================================================================
# Model Configuration
# ============================================================================
MODEL = {
    'num_classes': 50,  # ESC-50 has 50 classes
    'input_channels': 1,  # Single channel spectrogram
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING = {
    'batch_size': 32,
    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'max_epochs': 50,
    'patience': 10,  # Early stopping patience
    'num_workers': 2,  # DataLoader workers
    'pin_memory': True,
}

# ============================================================================
# Optimizer Configuration
# ============================================================================
OPTIMIZER = {
    'type': 'Adam',
    'lr': TRAINING['learning_rate'],
    'weight_decay': TRAINING['weight_decay'],
}

# ============================================================================
# Scheduler Configuration
# ============================================================================
SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'mode': 'max',  # Monitor validation accuracy
    'patience': 5,
    'factor': 0.5,  # Reduce LR by half
}

# ============================================================================
# MLflow Configuration
# ============================================================================
MLFLOW = {
    'experiment_name': 'ESC-50-Baseline',
    'tracking_uri': None,  # Use local file store (./mlruns)
}

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE = {
    'prefer_mps': True,  # Use MPS on M4 Mac if available
    'prefer_cuda': True,  # Use CUDA if available
}
