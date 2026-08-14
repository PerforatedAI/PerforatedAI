import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import bootstrap
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd
)

DATA_DIR = PROJECT_ROOT / "datasets" / "monai"

print("Using data dir:", DATA_DIR)
print("Exists:", DATA_DIR.exists())

# 🔥 MINIMAL transform (no spacing, no crops)
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
])

ds = DecathlonDataset(
    root_dir=str(DATA_DIR),
    task="Task01_BrainTumour",
    section="training",
    transform=test_transforms,
    download=False,
)

sample = ds[0]

print("Image shape:", sample["image"].shape)
print("Label shape:", sample["label"].shape)
print("Image dtype:", sample["image"].dtype)
print("Label dtype:", sample["label"].dtype)
