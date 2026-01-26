import os
import json
import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import random
import numpy as np

# --- CONFIGURATION ---
os.environ["PAIEMAIL"] = "Email"
os.environ["PAITOKEN"] = "Token"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION ---
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

    def __str__(self): return "DSCNN"
    def __repr__(self): return "DSCNN"

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


# --- LOAD BASELINE MODEL (.pth) ---
def load_baseline_model(checkpoint_path, num_classes):
    print("="*70)
    print("LOADING BASELINE MODEL")
    print("="*70)
    print(f" Path: {checkpoint_path}")
    
    model = DSCNN(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" Loaded! (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")
    except Exception as e:
        print(f" Failed: {e}")
        return None
    
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Parameters: {total_params:,}")
    print("="*70)
    
    return model


# --- LOAD PERFORATED MODEL (.pt) ---
def load_perforated_model(save_name, num_classes):
    print("="*70)
    print("LOADING PERFORATED MODEL")
    print("="*70)
    
    model = DSCNN(num_classes=num_classes)
    
    # Configure PAI
    GPA.pc.set_verbose(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.append_module_names_to_track(['BatchNorm2d'])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # Initialize PAI
    model = UPA.initialize_pai(
        model, 
        doing_pai=True,
        save_name=save_name,
        making_graphs=False,
        maximizing_score=True
    )
    
    # Load
    print(f" Loading from: projects/{save_name}/best_model.pt")
    try:
        model = UPA.load_system(model, save_name, 'best_model', True)
        print("✅ Successfully loaded!")
    except Exception as e:
        print(f" Failed: {e}")
        return None
    
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Parameters: {total_params:,}")
    print("="*70)
    
    return model


# --- DATASET (CLEAN + NOISY VERSIONS) ---
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
    def __init__(self, subset, transform, target_labels, augment_type='none'):
        """
        augment_type: 
            'none' = clean audio
            'low_volume' = reduce volume 50-70%
            'background_noise' = add white noise (SNR 10dB)
            'both' = low volume + noise
        """
        self.dataset = SubsetSC(subset)
        self.transform = transform
        self.target_labels = target_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(target_labels)}
        self.augment_type = augment_type
        self.samples = []
        
        print(f" Loading {subset} ({augment_type})...")
        for i in range(len(self.dataset)):
            _, _, label, *_ = self.dataset[i]
            if label in self.target_labels:
                self.samples.append(i)
        print(f"   {len(self.samples)} samples ready")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        waveform, _, label, *_ = self.dataset[real_idx]
        
        # Pad/truncate
        if waveform.shape[1] < 16000:
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]
        
        # Apply augmentation
        if self.augment_type == 'low_volume':
            waveform = waveform * random.uniform(0.3, 0.5)  # 30-50% volume
        
        elif self.augment_type == 'background_noise':
            # Add white noise (SNR ~10dB)
            noise = torch.randn_like(waveform) * 0.01
            waveform = waveform + noise
        
        elif self.augment_type == 'both':
            waveform = waveform * random.uniform(0.3, 0.5)
            noise = torch.randn_like(waveform) * 0.01
            waveform = waveform + noise
        
        return self.transform(waveform), self.label_to_idx[label]


# --- EVALUATE ---
def evaluate(model, loader, device, desc="Testing"):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, ncols=100, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


# --- MAIN ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL COMPARISON: BASELINE vs PERFORATED AI")
    print("="*70)
    
    # Load config
    with open("results/config.json", "r") as f:
        config = json.load(f)
    target_labels = config["target_labels"]
    
    print(f"\n Classes: {target_labels}")
    print(f"  Device: {DEVICE}\n")
    
    # Load both models
    baseline_model = load_baseline_model("models/baseline_v3_optimized_best.pth", len(target_labels))
    perforated_model = load_perforated_model("dscnn_perforated_10class", len(target_labels))
    
    if baseline_model is None or perforated_model is None:
        print("\n Failed to load models. Exiting.")
        exit(1)
    
    # Prepare transform
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, hop_length=256, n_mels=64
    )
    
    # Test scenarios
    test_scenarios = [
        ('clean', 'none'),
        ('low_volume', 'low_volume'),
        ('noisy', 'background_noise'),
        ('challenging', 'both')
    ]
    
    results = {
        'baseline': {},
        'perforated': {}
    }
    
    print("\n" + "="*70)
    print("TESTING ON MULTIPLE SCENARIOS")
    print("="*70)
    
    for scenario_name, augment_type in test_scenarios:
        print(f"\n Scenario: {scenario_name.upper()}")
        print("-"*70)
        
        # Create dataset
        test_dataset = AudioDataset("testing", transform, target_labels, augment_type=augment_type)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        
        # Test baseline
        print("  Testing Baseline...")
        loss_b, acc_b = evaluate(baseline_model, test_loader, DEVICE, f"Baseline ({scenario_name})")
        results['baseline'][scenario_name] = {'loss': loss_b, 'accuracy': acc_b}
        
        # Test perforated
        print("  Testing Perforated...")
        loss_p, acc_p = evaluate(perforated_model, test_loader, DEVICE, f"Perforated ({scenario_name})")
        results['perforated'][scenario_name] = {'loss': loss_p, 'accuracy': acc_p}
        
        # Show results
        improvement = acc_p - acc_b
        print(f"\n  Results:")
        print(f"    Baseline:   {acc_b:.2f}% (loss: {loss_b:.4f})")
        print(f"    Perforated: {acc_p:.2f}% (loss: {loss_p:.4f})")
        print(f"    Δ Accuracy: {improvement:+.2f}%")
        print(f"    Winner: {' PERFORATED' if acc_p > acc_b else '🏆 BASELINE' if acc_b > acc_p else '🤝 TIE'}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\n BASELINE:")
    for scenario in ['clean', 'low_volume', 'noisy', 'challenging']:
        acc = results['baseline'][scenario]['accuracy']
        print(f"  {scenario:12s}: {acc:.2f}%")
    
    print("\n PERFORATED:")
    for scenario in ['clean', 'low_volume', 'noisy', 'challenging']:
        acc = results['perforated'][scenario]['accuracy']
        print(f"  {scenario:12s}: {acc:.2f}%")
    
    print("\n IMPROVEMENTS:")
    for scenario in ['clean', 'low_volume', 'noisy', 'challenging']:
        base_acc = results['baseline'][scenario]['accuracy']
        perf_acc = results['perforated'][scenario]['accuracy']
        diff = perf_acc - base_acc
        print(f"  {scenario:12s}: {diff:+.2f}%")
    
    # Calculate averages
    avg_baseline = sum(results['baseline'][s]['accuracy'] for s in ['clean', 'low_volume', 'noisy', 'challenging']) / 4
    avg_perforated = sum(results['perforated'][s]['accuracy'] for s in ['clean', 'low_volume', 'noisy', 'challenging']) / 4
    
    print("\n" + "="*70)
    print(f"AVERAGE ACCURACY:")
    print(f"  Baseline:   {avg_baseline:.2f}%")
    print(f"  Perforated: {avg_perforated:.2f}%")
    print(f"  Improvement: {avg_perforated - avg_baseline:+.2f}%")
    print("="*70)
    
    # Save results
    results['summary'] = {
        'baseline_avg': avg_baseline,
        'perforated_avg': avg_perforated,
        'improvement': avg_perforated - avg_baseline,
        'baseline_params': sum(p.numel() for p in baseline_model.parameters()),
        'perforated_params': sum(p.numel() for p in perforated_model.parameters())
    }
    
    with open("results/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved: results/comparison_results.json")
    print("\n COMPARISON COMPLETE!\n")
