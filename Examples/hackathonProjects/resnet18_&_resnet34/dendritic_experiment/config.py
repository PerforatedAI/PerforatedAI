import os

# PAI Credentials
# User should fill these in or set environment variables
PAI_EMAIL = os.getenv("PAIEMAIL", "hacker@perforatedai.com")
PAI_TOKEN = os.getenv("PAITOKEN", "InJ9BjZSB+B+l30bmSzhqOwsXxOx0NRKAe8dtdAqdQcT/pKjmme1fqB1zrnCd5CWNrhJm40PVjaDbIrjR5xU+q2uhcUWX8gk2Kb2lHjafkUnizPXyP+yckbv+UxlU25ZlrvC3XlLu/AZdVKJE7Eov9+4c76sKe2hbRnH1fny2xIPYmy2/m/sY1gxXbhPtTa1mtxk2EgLeo5pRu/eL/7pSXWmEoRmvVorgQEJzt1VYOZyp0vP4bLxF72tOgSjXGBO8SHHcN16CbOVJuIEm3jmEc/AfPyyB+G4TEqhH7UZ0W2R/bnXtNberKqF2bQTuyT26etQw6NEMoXwuugDcrBXEw==")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 250 # Full training run
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EARLY_STOP_PATIENCE = 25 # Give it more room to breathe

# PAI Settings
PAI_SWITCH_MODE = "DOING_HISTORY"  # Adaptive mode: trains until plateau before switching
PAI_IMPROVEMENT_THRESHOLD = 0.001  # From best sweep run
PAI_MAX_DENDRITES = 10  # Increased to allow more growth
PAI_N_EPOCHS = 8  # From best sweep run
PAI_P_EPOCHS = 8  # From best sweep run

# Baseline Constants (DO NOT MODIFY - from completed training runs)
BASELINE_RESNET18_PARAMS = 11227812
BASELINE_RESNET18_ACC = 81.12
BASELINE_RESNET34_PARAMS = 21335972
BASELINE_RESNET34_ACC = 82.34

# Parameter Safety Threshold
PARAM_SAFETY_RATIO = 0.95  # Stop dendrite growth at 95% of ResNet34 params

# Paths
DATA_DIR = "./data"
RESULTS_DIR = "./results"
# Use relative path or environment variable for PAI repo
PAI_REPO_PATH = os.path.join(os.getcwd(), "PAI_repo")
