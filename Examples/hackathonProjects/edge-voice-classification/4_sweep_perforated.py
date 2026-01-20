"""
4_sweep_perforated.py - W&B Sweep for PerforatedAI
With progress bars and proper argument handling
"""

import os
import json
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings("ignore")

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

os.environ["PAIEMAIL"] = "EMAIL"
os.environ["PAITOKEN"] = "TOKEN"
# Dataset classes
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./data", download=False)
        
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

class AudioDataset(Dataset):
    def __init__(self, subset, transform, target_labels, augment=False):
        self.dataset = SubsetSC(subset)
        self.transform = transform
        self.target_labels = target_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(target_labels)}
        self.augment = augment
        
        self.samples = []
        print(f"   Filtering {subset} samples...")
        for i in tqdm(range(len(self.dataset)), desc=f"   Loading {subset}", leave=False, ncols=80):
            _, _, label, *_ = self.dataset[i]
            if label in self.target_labels:
                self.samples.append(i)
        print(f"   ✅ {len(self.samples)} samples ready\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        waveform, sample_rate, label, *_ = self.dataset[real_idx]
        
        # Data augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                waveform = waveform * (0.7 + 0.6 * torch.rand(1))
            
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
            
            if torch.rand(1).item() > 0.5:
                shift = int(waveform.shape[1] * 0.1 * (torch.rand(1).item() - 0.5))
                waveform = torch.roll(waveform, shift, dims=1)
        
        # Pad or truncate to 16000 samples
        if waveform.shape[1] < 16000:
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]
        
        spec = self.transform(waveform)
        return spec, self.label_to_idx[label]

class DSCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DSCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=10, stride=(2, 1), padding=(5, 0))
        self.bn1 = nn.BatchNorm2d(64)
        
        self.ds_conv1 = self._make_ds_conv(64, 64)
        self.ds_conv2 = self._make_ds_conv(64, 64)
        self.ds_conv3 = self._make_ds_conv(64, 64)
        self.ds_conv4 = self._make_ds_conv(64, 64)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def _make_ds_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", ncols=100, leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", ncols=100, leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def train_sweep():
    """Single training run for W&B sweep"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--improvement_threshold', type=float, default=0.005)
    parser.add_argument('--max_dendrites', type=int, default=5)
    parser.add_argument('--n_epochs_to_switch', type=int, default=7)
    args, unknown = parser.parse_known_args()
    
    # Initialize wandb
    run = wandb.init(
        project="speech-commands-perforated",
        config={
            "n_epochs_to_switch": args.n_epochs_to_switch,
            "max_dendrites": args.max_dendrites,
            "improvement_threshold": args.improvement_threshold
        }
    )
    
    config = wandb.config
    
    print(f"\n{'='*70}")
    print(f"SWEEP RUN: {run.name}")
    print(f"  n_epochs_to_switch: {config.n_epochs_to_switch}")
    print(f"  max_dendrites: {config.max_dendrites}")
    print(f"  improvement_threshold: {config.improvement_threshold}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}\n")
    
    # Load config
    with open("results/config.json", "r") as f:
        data_config = json.load(f)
    
    target_labels = data_config["target_labels"]
    
    # Prepare datasets
    print("🔹 Loading datasets...")
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=64
    )
    
    train_dataset = AudioDataset("training", transform, target_labels, augment=True)
    val_dataset = AudioDataset("validation", transform, target_labels, augment=False)
    
    batch_size = 128
    print("🔹 Creating data loaders...\n")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Create model
    print("🔹 Creating model...")
    model = DSCNN(num_classes=len(target_labels))
    
    # Configure PerforatedAI
    print("🔹 Configuring PerforatedAI...")
    GPA.pc.set_verbose(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.append_module_names_to_track(['BatchNorm2d'])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # SWEEP PARAMETERS
    GPA.pc.set_improvement_threshold([
        config.improvement_threshold, 
        config.improvement_threshold / 2, 
        0
    ])
    GPA.pc.set_max_dendrites(config.max_dendrites)
    GPA.pc.set_n_epochs_to_switch(config.n_epochs_to_switch)
    
    print(f"   improvement_threshold: [{config.improvement_threshold}, {config.improvement_threshold/2}, 0]")
    print(f"   max_dendrites: {config.max_dendrites}")
    print(f"   n_epochs_to_switch: {config.n_epochs_to_switch}\n")
    
    # Initialize PAI
    print("🔹 Wrapping with PAI...\n")
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name=f"sweep_{run.id}",
        making_graphs=False,
        maximizing_score=True
    )
    
    model = model.to(device)
    
    # Optimizer & loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=5, verbose=False)
    
    GPA.pai_tracker.set_optimizer_instance(optimizer)
    
    # Training loop
    print("🔹 Starting training (max 80 epochs)...\n")
    max_epochs = 80
    best_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{max_epochs}")
        print("="*70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update PerforatedAI
        GPA.pai_tracker.add_extra_score_without_graphing(train_loss, 'Train Loss')
        GPA.pai_tracker.add_extra_score_without_graphing(train_acc, 'Train Acc')
        
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        model = model.to(device)
        
        if restructured:
            print("\n🌳 DENDRITES ADDED!\n")
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                            factor=0.5, patience=5, verbose=False)
            GPA.pai_tracker.set_optimizer_instance(optimizer)
        
        # Get dendrite count
        dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dendrites": dendrite_count,
            "parameters": param_count,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"   Dendrites: {dendrite_count}")
        
        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            print(f"   ✅ NEW BEST! ({val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   Best: {best_acc:.2f}% (patience: {patience_counter}/{patience})")
        
        scheduler.step(val_acc)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
        
        if training_complete:
            print(f"\n✅ PAI training complete at epoch {epoch}")
            break
    
    # Log final summary
    wandb.summary["best_val_acc"] = best_acc
    wandb.summary["final_dendrites"] = dendrite_count
    wandb.summary["final_parameters"] = param_count
    wandb.summary["epochs_trained"] = epoch
    
    print(f"\n{'='*70}")
    print(f"✅ Sweep run complete!")
    print(f"   Best Val Acc: {best_acc:.2f}%")
    print(f"   Final Dendrites: {dendrite_count}")
    print(f"   Epochs: {epoch}")
    print(f"{'='*70}\n")
    
    wandb.finish()

if __name__ == '__main__':
    train_sweep()

