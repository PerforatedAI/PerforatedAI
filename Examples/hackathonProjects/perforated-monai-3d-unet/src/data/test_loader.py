import sys
from pathlib import Path

# add src/ to python path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

# 🔥 define project root properly
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import bootstrap
from data.dataset_loader import get_dataloaders

DATA_DIR = PROJECT_ROOT / "datasets" / "monai"

print("Using data dir:", DATA_DIR)
print("Exists:", DATA_DIR.exists())

train_loader, val_loader = get_dataloaders(
    data_dir=str(DATA_DIR),
    batch_size=1,
    num_workers=4,
)

batch = next(iter(train_loader))

print("Image shape:", batch["image"].shape)
print("Label shape:", batch["label"].shape)
print("Image dtype:", batch["image"].dtype)
print("Label dtype:", batch["label"].dtype)
