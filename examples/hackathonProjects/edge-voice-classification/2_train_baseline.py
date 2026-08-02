"""
Phase 2: Optimized Baseline DS-CNN (v3)
SpecAugment (TimeMask + FreqMask)
Label smoothing 0.15
Same as perforated for FAIR comparison
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import time
import wandb

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
        
        #  NEW: SpecAugment
        if augment:
            self.time_mask = T.TimeMasking(time_mask_param=20)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        
        self.samples = []
        print(f"Filtering {subset} samples for target labels...")
        for i in tqdm(range(len(self.dataset)), desc=f"Loading {subset}"):
            _, _, label, *_ = self.dataset[i]
            if label in self.target_labels:
                self.samples.append(i)
        print(f"✓ Kept {len(self.samples)} samples from {len(self.dataset)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        waveform, sample_rate, label, *_ = self.dataset[real_idx]
        
        # Waveform augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                waveform = waveform * (0.7 + 0.6 * torch.rand(1))
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
            if torch.rand(1).item() > 0.5:
                shift = int(waveform.shape[1] * 0.1 * (torch.rand(1).item() - 0.5))
                waveform = torch.roll(waveform, shift, dims=1)
        
        if waveform.shape[1] < 16000:
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]
        
        spec = self.transform(waveform)
        
        #  NEW: Apply SpecAugment AFTER mel spectrogram
        if self.augment:
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)
        
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
    
    pbar = tqdm(loader, desc="Training")
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def main():
    print("="*70)
    print("BASELINE v3: OPTIMIZED (SpecAugment + Label Smoothing 0.15)")
    print("="*70)
    
    with open("results/config.json", "r") as f:
        config = json.load(f)
    
    target_labels = config["target_labels"]
    print(f"\n Target classes: {len(target_labels)} classes")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n Initializing Weights & Biases...")
    wandb.init(
        project="perforatedai-kws-hackathon",
        name="DS-CNN-Baseline-v3-Optimized-10class",
        config={
            "architecture": "DS-CNN",
            "dataset": "Speech Commands v2",
            "num_classes": len(target_labels),
            "batch_size": 128,
            "learning_rate": 0.001,
            "epochs": 50,
            "optimizer": "AdamW",
            "model_type": "baseline_optimized",
            "perforated": False,
            "augmentation": "waveform + SpecAugment",
            "label_smoothing": 0.15,
            "gradient_clipping": 1.0,
            "specaugment_time_mask": 20,
            "specaugment_freq_mask": 10
        },
        tags=["baseline", "ds-cnn", "kws", "esp32", "10class", "v3-optimized", "specaugment"]
    )
    
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=64
    )
    
    print("\n Creating datasets...")
    train_dataset = AudioDataset("training", transform, target_labels, augment=True)
    val_dataset = AudioDataset("validation", transform, target_labels, augment=False)
    test_dataset = AudioDataset("testing", transform, target_labels, augment=False)
    
    print(f"\n✓ Dataset sizes:")
    print(f"   Train: {len(train_dataset):,} (with SpecAugment )")
    print(f"   Val:   {len(val_dataset):,}")
    print(f"   Test:  {len(test_dataset):,}")
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = DSCNN(num_classes=len(target_labels)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_kb_int8 = (total_params * 1) / 1024
    
    print("\n🏗️  Model: DS-CNN (Baseline)")
    print(f"   Parameters: {total_params:,}")
    print(f"   Size (INT8): {model_size_kb_int8:.1f} KB")
    
    wandb.config.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_int8_kb": model_size_kb_int8
    })
    
    #  Label smoothing 0.15
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    num_epochs = 50
    print(f"\n Starting training ({num_epochs} epochs)...")
    print("SpecAugment: TimeMask(20) + FreqMask(10)")
    print("Label Smoothing: 0.15")
    print("Gradient Clipping: 1.0")
    
    best_acc = 0
    best_epoch = 0
    training_history = []
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f" Epoch {epoch}/{num_epochs}")
        print('='*70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        overfit_gap = train_acc - val_acc
        
        print(f"\n Results:")
        print(f"      Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"      Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"      Gap:   {overfit_gap:.2f}%")
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train_val_gap": overfit_gap,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "overfit_gap": overfit_gap
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'models/baseline_v3_optimized_best.pth')
            print(f"\n  New best! (Val Acc: {val_acc:.2f}%)")
            
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_epoch"] = epoch
        
        scheduler.step(val_acc)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f" Time: {training_time/60:.1f} minutes")
    print(f" Best Val Acc: {best_acc:.2f}% (epoch {best_epoch})")
    
    print("\n Testing best model...")
    checkpoint = torch.load('models/baseline_v3_optimized_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"   Test Acc: {test_acc:.2f}%")
    
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc
    })
    
    wandb.run.summary["test_acc"] = test_acc
    wandb.run.summary["training_time_minutes"] = training_time / 60
    
    results = {
        "model": "DS-CNN Baseline v3 Optimized",
        "architecture": "Depthwise Separable CNN",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "training_time_minutes": training_time / 60,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "training_history": training_history,
        "training_config": {
            "augmentation": "waveform + SpecAugment",
            "label_smoothing": 0.15,
            "gradient_clipping": 1.0,
            "optimizer": "AdamW",
            "specaugment_time_mask": 20,
            "specaugment_freq_mask": 10
        }
    }
    
    with open("results/baseline_v3_optimized_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/baseline_v3_optimized_results.json")
    
    wandb.finish()
    
    print("\n" + "="*70)
    print("NEXT STEP: python 3_train_perforated_v3_optimized.py")
    print("="*70)

if __name__ == '__main__':
    main()
