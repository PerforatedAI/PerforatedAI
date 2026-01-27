from pathlib import Path
import torch

class BaseConfig:
    PROJECT_NAME = "dendrivit-medical"
    EXPERIMENT_STAGE = "baseline"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    NUM_CLASSES = 24
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

base_config = BaseConfig()
dendrite_config = DendriticConfig()
