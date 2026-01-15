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
    'type': 'CNN14',  # Options: 'AudioCNN' (simple), 'CNN14' (better), 'SpeechBrain' (pretrained)
    'num_classes': 50,  # ESC-50 has 50 classes
    'input_channels': 1,  # Single channel spectrogram
    'pretrained': False,  # Use pretrained weights (for SpeechBrain model)
    'freeze_encoder': False,  # Freeze encoder layers (for fine-tuning)
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING = {
    'batch_size': 32,
    'learning_rate': 0.0001,  # Lower LR for fine-tuning pretrained model
    'weight_decay': 1e-5,  # Lower weight decay for pretrained models
    'max_epochs': 200,  # Max epochs for both baseline and PAI
    'patience': 15,  # Early stopping patience (increased for better models)
    'num_workers': 2,  # DataLoader workers
    'pin_memory': False,  # Set to False for MPS (not supported on Apple Silicon)
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
    #'experiment_name': 'ESC-50-Baseline',
    'experiment_name': 'ESC-50-Audio',
    'tracking_uri': None,  # Use local file store (./mlruns)
}

# ============================================================================
# PerforatedAI Configuration
# ============================================================================
PAI = {
    'max_dendrites': 5,  # Maximum number of dendrites to add
    'test_mode': False,  # Set True for quick test (3 dendrites), False for real training
    'verbose': True,  # Enable verbose PAI output
    'improvement_threshold': [0.001, 0.0001, 0],  # When to stop adding dendrites
    'forward_function': 'sigmoid',  # Options: 'sigmoid', 'relu', 'tanh'
    'weight_init_multiplier': 0.01,  # Weight initialization for new dendrites
    'use_perforated_backprop': False,  # False=GD (open source), True=PB (requires license)
}

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE = {
    'prefer_mps': True,  # Use MPS on M4 Mac if available
    'prefer_cuda': True,  # Use CUDA if available
}
