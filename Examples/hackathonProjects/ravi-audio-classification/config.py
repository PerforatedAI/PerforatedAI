"""
Configuration for ESC-50 audio classification with PerforatedAI.
"""

# ============================================================================
# Paths
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
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'max_epochs': 200,
    'patience': 15,  # Early stopping patience
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
# PerforatedAI Configuration
# ============================================================================
PAI = {
    'max_dendrites': 5,  # Maximum number of dendrites to add
    'improvement_threshold': [0.001, 0.0001, 0],  # When to stop adding dendrites
    'forward_function': 'sigmoid',  # Dendrite activation function
    'weight_init_multiplier': 0.01,  # Weight initialization for new dendrites
}

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE = {
    'prefer_mps': True,  # Use MPS on Apple Silicon if available
    'prefer_cuda': True,  # Use CUDA if available
}
