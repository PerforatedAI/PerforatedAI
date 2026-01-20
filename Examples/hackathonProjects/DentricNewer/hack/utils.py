import os
import random
import torch
import numpy as np
from pathlib import Path

class BaseConfig:
    PROJECT_NAME = "dendrivit-medical"
    EXPERIMENT_STAGE = "baseline"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    NUM_CLASSES = 2
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 50
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    DATA_PATH = Path("data/medical_images")
    WANDB_PROJECT = "dendrivit-bci"
    LOG_INTERVAL = 10
    MODELS_PATH = Path("models")
    RESULTS_PATH = Path("results")
    
    def __init__(self):
        for path in [self.MODELS_PATH, self.RESULTS_PATH]:
            path.mkdir(exist_ok=True, parents=True)

class DendriticConfig(BaseConfig):
    EXPERIMENT_STAGE = "dendritic"
    DENDRITE_CYCLE_LIMIT = 2
    DENDRITIC_DEPTH = 3
    PRUNING_RATE = 0.75
    FREEZE_LAYERS = 8

# --- MISSING FUNCTIONS ADDED BELOW ---

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Saves the model checkpoint."""
    # Ensure the parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")

# Initialize configs so they are available for import if needed
base_config = BaseConfig()
dendrite_config = DendriticConfig()