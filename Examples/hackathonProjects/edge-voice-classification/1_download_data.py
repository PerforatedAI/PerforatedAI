"""
Phase 1: Download and prepare Google Speech Commands dataset
"""

import os
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from pathlib import Path
import json

print("="*70)
print("PHASE 1: DOWNLOADING GOOGLE SPEECH COMMANDS DATASET")
print("="*70)

# Create folder structure
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("\n Created folder structure:")
print("   ✓ data/")
print("   ✓ models/")
print("   ✓ results/")

# Download dataset
print("\n Downloading Speech Commands v2 dataset...")
print("(This may take 5-10 minutes on first run)")

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./data", download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]
        
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]

# Create datasets
print("\n Creating train/val/test splits...")
train_set = SubsetSC("training")
val_set = SubsetSC("validation")
test_set = SubsetSC("testing")

print(f"\n✓ Dataset ready!")
print(f"   Train samples: {len(train_set):,}")
print(f"   Val samples:   {len(val_set):,}")
print(f"   Test samples:  {len(test_set):,}")
print(f"   Total: {len(train_set) + len(val_set) + len(test_set):,}")

# Get all unique labels
labels = sorted(list(set([label for _, _, label, *_ in train_set])))
print(f"\n📊 Total classes available: {len(labels)}")

# ✅ USE ONLY 10 CLASSES - Optimized for hackathon
target_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

print(f"\n Using {len(target_labels)} classes for classification")
print(f"   Optimized for: Faster training + Better compression")
print(f"   Classes: {', '.join(target_labels)}")

# Save config
config = {
    "dataset": "Google Speech Commands v2",
    "total_samples": len(train_set) + len(val_set) + len(test_set),
    "train_samples": len(train_set),
    "val_samples": len(val_set),
    "test_samples": len(test_set),
    "total_classes": len(labels),
    "target_classes": len(target_labels),
    "target_labels": target_labels,
    "sample_rate": 16000,
    "audio_length": 1.0,  # 1 second
    "architecture": "DS-CNN",
    "model_type": "Depthwise Separable CNN"
}

with open("results/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n✓ Configuration saved to results/config.json")

# Test loading a sample
print("\n Testing audio loading...")
waveform, sample_rate, label, speaker_id, utterance_num = train_set[0]
print(f"   Sample shape: {waveform.shape}")
print(f"   Sample rate: {sample_rate} Hz")
print(f"   Label: {label}")

print("\n" + "="*70)
print(" PHASE 1 COMPLETE!")
print("="*70)
print("\nNext step: python 2_train_baseline.py")
print("Estimated time: ~45 minutes (10 classes)")
