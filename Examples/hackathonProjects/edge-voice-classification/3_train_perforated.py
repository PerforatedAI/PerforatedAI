"""
Phase 3: Train DS-CNN with PerforatedAI
SAVES BEST MODEL ONCE AT END (after training completes)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import time
import wandb
import warnings
import copy

warnings.filterwarnings("ignore")

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

os.environ["PAIEMAIL"] = "EMAIL"
os.environ["PAITOKEN"] = "TOKEN"

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
    
    def __str__(self):
        return "DSCNN"
    
    def __repr__(self):
        return "DSCNN"
    
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
    print("PHASE 3: DS-CNN + PERFORATEDAI")
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
        name="DS-CNN-PAI-Clean",
        config={
            "architecture": "DS-CNN",
            "dataset": "Speech Commands v2",
            "num_classes": len(target_labels),
            "batch_size": 128,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "model_type": "perforated",
            "perforated": True,
            "pai_tracking": "BatchNorm2d",
            "pai_epochs_to_switch": 10,
            "pai_max_dendrites": 5
        },
        tags=["perforated", "ds-cnn", "kws", "clean"]
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

    print(f"\n✓ Datasets ready: Train={len(train_dataset):,}, Val={len(val_dataset):,}, Test={len(test_dataset):,}")

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\n Initializing DS-CNN...")
    model = DSCNN(num_classes=len(target_labels))
    
    baseline_params = sum(p.numel() for p in model.parameters())
    print(f"   Baseline params: {baseline_params:,}")
    
    print("\n  Configuring PerforatedAI...")
    GPA.pc.set_verbose(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.append_module_names_to_track(['BatchNorm2d'])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_improvement_threshold([0.005, 0.001, 0])
    GPA.pc.set_max_dendrites(8)
    GPA.pc.set_n_epochs_to_switch(7)
    
    print("   ✓ PAI configured:")
    print("      - Tracking: BatchNorm2d")
    print("      - n_epochs_to_switch: 7")
    print("      - max_dendrites: 8")
    
    print("\n Converting to PerforatedAI...")
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name="dscnn_perforated_10class",
        making_graphs=True,
        maximizing_score=True
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    initial_size_int8 = (total_params * 1) / 1024
    pai_overhead = total_params - baseline_params

    print(f"\n✓ Model ready!")
    print(f"   Baseline:     {baseline_params:,} params")
    print(f"   After PAI:    {total_params:,} params")
    print(f"   PAI overhead: {pai_overhead:,} params (+{(pai_overhead/baseline_params)*100:.1f}%)")
    print(f"   Size (INT8):  {initial_size_int8:.1f} KB")
    
    wandb.config.update({
        "baseline_parameters": baseline_params,
        "initial_parameters": total_params,
        "pai_overhead": pai_overhead
    })

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("\n  Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=5,
        verbose=False
    )
    
    GPA.pai_tracker.set_optimizer_instance(optimizer)

    print(" Optimizer: AdamW (lr=0.001)")
    print(" Scheduler: ReduceLROnPlateau")
    print(" Clean logs: num_workers=0")

    os.makedirs('models', exist_ok=True)

    max_epochs = 200
    print(f"\n Starting training (max {max_epochs} epochs)...")
    
    best_acc = 0
    best_epoch = 0
    best_val_loss = float('inf')
    best_model_params = 0
    best_model_dendrites = 0
    best_model = None  # Keep best model in memory
    training_history = []
    dendrite_events = []
    start_time = time.time()

    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        print(f"\n{'='*70}")
        print(f" Epoch {epoch}/{max_epochs}")
        print('='*70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        overfit_gap = train_acc - val_acc
        
        print(f"\n    Results:")
        print(f"      Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"      Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"      Gap:   {overfit_gap:.2f}%")
        
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "overfit_gap": overfit_gap
        })
        
        current_params = sum(p.numel() for p in model.parameters())
        dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
        
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train_val_gap": overfit_gap,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "parameters": current_params,
            "dendrites": dendrite_count,
            "size_int8_kb": (current_params * 1) / 1024
        })
        
        GPA.pai_tracker.add_extra_score_without_graphing(train_loss, 'Train Loss')
        GPA.pai_tracker.add_extra_score_without_graphing(train_acc, 'Train Acc')
        
        prev_params = current_params
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        model = model.to(device)
        
        if restructured:
            new_params = sum(p.numel() for p in model.parameters())
            params_added = new_params - prev_params
            
            print("\n    PerforatedAI: DENDRITE ADDED!")
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                weight_decay=1e-4
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=5,
                verbose=False
            )
            
            GPA.pai_tracker.set_optimizer_instance(optimizer)
            
            dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
            
            print(f"      New parameters: {new_params:,}")
            print(f"      Added: {params_added:,}")
            print(f"      Total dendrites: {dendrite_count}")
            print(f"      Size (INT8): {(new_params * 1) / 1024:.1f} KB")
            
            dendrite_events.append({
                "epoch": epoch,
                "dendrite_num": dendrite_count,
                "params_added": params_added,
                "params_total": new_params,
                "val_acc": val_acc
            })
            
            wandb.log({
                "dendrite_addition": epoch,
                "dendrite_count": dendrite_count,
                "params_added_this_dendrite": params_added
            })
        
        # Track best model (in memory only during training)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_params = current_params
            best_model_dendrites = dendrite_count
            
            # Keep copy in memory
            print(f"\n    Copying best model to memory...")
            best_model = copy.deepcopy(model)
            best_model = best_model.to(device)
            
            print(f"    New best! (Val Acc: {val_acc:.2f}%)")
            print(f"    Params: {current_params:,}, Dendrites: {dendrite_count}")
            
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_epoch"] = epoch
        
        if training_complete:
            print("\n PerforatedAI TRAINING COMPLETE!")
            break
        
        scheduler.step(val_acc)

    training_time = time.time() - start_time
    final_params = sum(p.numel() for p in model.parameters())
    final_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)

    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f"  Time: {training_time/60:.1f} minutes")
    print(f" Best Val Acc: {best_acc:.2f}% (epoch {best_epoch})")
    print(f" Dendrites: {final_dendrites} added")
    print(f" Best Model: {best_model_params:,} params")
    
    if dendrite_events:
        print("\n Dendrite Addition Events:")
        for event in dendrite_events:
            print(f"   Epoch {event['epoch']}: Dendrite #{event['dendrite_num']}")
            print(f"      Added {event['params_added']:,} params → {event['params_total']:,} total")
            print(f"      Val acc: {event['val_acc']:.2f}%")

    print("\n" + "="*70)
    print(" TESTING BEST MODEL")
    print("="*70)
    
    # Test the best model
    if best_model is not None:
        print(f"   Testing BEST model from epoch {best_epoch}")
        print(f"   Model: {best_model_params:,} params, {best_model_dendrites} dendrites")
        
        test_loss, test_acc = validate(best_model, test_loader, criterion, device)
        print(f"\n    Test Acc: {test_acc:.2f}%")
        print(f"   (Best val was {best_acc:.2f}% at epoch {best_epoch})")
        
    else:
        print(f"   ⚠️  No best model, using final")
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"\n   Test Acc: {test_acc:.2f}%")
    
    print(f"\n    ESP32 Deployment:")
    print(f"      Size (INT8): {(best_model_params * 1) / 1024:.1f} KB")
    print(f"      Size (FP32): {(best_model_params * 4) / 1024:.1f} KB")
    print(f"      ESP32 SRAM: 260 KB")
    print(f"      Fits: {' YES!' if (best_model_params * 1) / 1024 < 260 else '❌ Too large'}")
    
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc
    })
    
    wandb.run.summary["test_acc"] = test_acc
    wandb.run.summary["training_time_minutes"] = training_time / 60

    #  SAVE BEST MODEL ONCE AT THE END (AFTER TESTING)
    print("\n" + "="*70)
    print(" SAVING BEST MODEL")
    print("="*70)
    
    if best_model is not None:
        print(f"Saving best model from epoch {best_epoch}...")
        
        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': best_model.state_dict(),
            'val_acc': best_acc,
            'val_loss': best_val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'dendrites': best_model_dendrites,
            'params': best_model_params
        }
        
        torch.save(checkpoint, 'models/perforated_best.pth')
        
        file_size_mb = os.path.getsize('models/perforated_best.pth') / (1024*1024)
        print(f"   SAVED: models/perforated_best.pth ({file_size_mb:.2f} MB)")
        print(f"   Epoch: {best_epoch}")
        print(f"   Val acc: {best_acc:.2f}%")
        print(f"   Test acc: {test_acc:.2f}%")
        print(f"   Params: {best_model_params:,}")
        print(f"   Dendrites: {best_model_dendrites}")
    else:
        print("⚠️  No best model to save!")

    # Save results JSON
    results = {
        "model": "DS-CNN_PAI_10class",
        "tracking": "BatchNorm2d",
        "baseline_parameters": baseline_params,
        "best_model_parameters": best_model_params,
        "best_model_dendrites": best_model_dendrites,
        "final_parameters": final_params,
        "final_dendrites": final_dendrites,
        "dendrite_events": dendrite_events,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "training_time_minutes": training_time / 60,
        "num_epochs": epoch,
        "training_history": training_history,
        "esp32_size_int8_kb": (best_model_params * 1) / 1024,
        "esp32_size_fp32_kb": (best_model_params * 4) / 1024,
        "esp32_compatible": (best_model_params * 1) / 1024 < 260,
        "model_saved": "models/perforated_best.pth"
    }

    with open("results/perforated_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n Results JSON: results/perforated_results.json")
    
    wandb.finish()
    
    print("\n" + "="*70)
    print(" ALL DONE!")
    print("="*70)
    print(f"\n Final Summary:")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Model Size:    {(best_model_params * 1) / 1024:.1f} KB (INT8)")
    print(f"   Dendrites:     {best_model_dendrites}")
    print(f"   Best Epoch:    {best_epoch}")
    print(f"   Saved:      models/perforated_best.pth")

if __name__ == '__main__':
    main()
