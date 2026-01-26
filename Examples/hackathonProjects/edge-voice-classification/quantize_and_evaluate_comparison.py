"""
INT8 Quantization Comparison: Baseline vs Perforated
Baseline: Direct quantization
Perforated: BPA → CPA → Quantization (preserves dendrite structure)
"""

import os
import json
import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ============================================================================
# PERFORATED AI IMPORTS
# ============================================================================
os.environ["PAIEMAIL"] = "EMAIL"
os.environ["PAITOKEN"] = "TOKEN"

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from perforatedai import blockwise_perforatedai as BPA
from perforatedai import clean_perforatedai as CPA

DEVICE = torch.device('cpu')  # Quantization requires CPU

# ============================================================================
# DATA LOADING (Same as your training scripts)
# ============================================================================

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
    def __init__(self, subset, transform, target_labels):
        self.dataset = SubsetSC(subset)
        self.transform = transform
        self.target_labels = target_labels
        self.label_to_idx = {label: idx for idx, label in enumerate(target_labels)}
        
        self.samples = []
        print(f"Loading {subset} samples...")
        for i in tqdm(range(len(self.dataset)), desc=f"Loading {subset}"):
            _, _, label, *_ = self.dataset[i]
            if label in self.target_labels:
                self.samples.append(i)
        print(f"✓ {len(self.samples)} samples ready")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        waveform, _, label, *_ = self.dataset[real_idx]
        
        # Pad/truncate to 1 second
        if waveform.shape[1] < 16000:
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        else:
            waveform = waveform[:, :16000]
        
        return self.transform(waveform), self.label_to_idx[label]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_size(model):
    """Calculate model size in KB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def evaluate_model(model, loader, device, desc="Testing"):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, ncols=100, leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1

def quantize_to_int8(model, desc="model"):
    """Apply dynamic INT8 quantization"""
    print(f"🔧 Quantizing {desc} to INT8...")
    quantized = quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    print("✅ Quantization complete!")
    return quantized

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("INT8 QUANTIZATION COMPARISON: BASELINE vs PERFORATED")
    print("="*70)
    
    # Load config
    with open("results/config.json", "r") as f:
        config = json.load(f)
    target_labels = config["target_labels"]
    num_classes = len(target_labels)
    print(f"Classes: {num_classes}")
    print(f"Device: {DEVICE}")
    
    # Prepare data
    print("\n" + "="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        n_mels=64
    )
    
    test_dataset = AudioDataset("testing", transform, target_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    results = {}
    
    # ========================================================================
    # BASELINE MODEL (STANDARD PYTORCH - DIRECT QUANTIZATION)
    # ========================================================================
    
    print("\n" + "="*70)
    print("🔵 BASELINE MODEL (Standard PyTorch)")
    print("="*70)
    
    # Load baseline FP32
    print("\n📥 Loading baseline FP32...")
    baseline_fp32 = DSCNN(num_classes=num_classes)
    checkpoint = torch.load("models/baseline_v3_optimized_best.pth", map_location=DEVICE)
    baseline_fp32.load_state_dict(checkpoint["model_state_dict"])
    baseline_fp32 = baseline_fp32.to(DEVICE)
    baseline_fp32.eval()
    
    fp32_size = get_model_size(baseline_fp32)
    fp32_params, _ = count_parameters(baseline_fp32)
    print(f"✓ Loaded: {fp32_size:.2f} KB ({fp32_params:,} params)")
    
    # Evaluate FP32
    print("📊 Evaluating FP32...")
    baseline_fp32_acc, baseline_fp32_f1 = evaluate_model(
        baseline_fp32, test_loader, DEVICE, "Baseline FP32"
    )
    print(f"  Accuracy: {baseline_fp32_acc:.4f}")
    print(f"  F1 Score: {baseline_fp32_f1:.4f}")
    
    results['baseline_fp32'] = {
        'accuracy': float(baseline_fp32_acc),
        'f1_score': float(baseline_fp32_f1),
        'size_kb': float(fp32_size),
        'parameters': int(fp32_params)
    }
    
    # Quantize baseline to INT8 (DIRECT - no BPA/CPA needed)
    print("\n🔧 Quantizing baseline to INT8 (direct quantization)...")
    baseline_int8 = quantize_to_int8(baseline_fp32, "baseline")
    int8_size = get_model_size(baseline_int8)
    print(f"✓ INT8 Size: {int8_size:.2f} KB")
    print(f"✓ Compression: {(1 - int8_size/fp32_size)*100:.1f}%")
    
    # Evaluate INT8
    print("📊 Evaluating INT8...")
    baseline_int8_acc, baseline_int8_f1 = evaluate_model(
        baseline_int8, test_loader, DEVICE, "Baseline INT8"
    )
    print(f"  Accuracy: {baseline_int8_acc:.4f}")
    print(f"  F1 Score: {baseline_int8_f1:.4f}")
    
    baseline_acc_drop = (baseline_fp32_acc - baseline_int8_acc) * 100
    print(f"  Accuracy Drop: {baseline_acc_drop:.2f}%")
    
    results['baseline_int8'] = {
        'accuracy': float(baseline_int8_acc),
        'f1_score': float(baseline_int8_f1),
        'size_kb': float(int8_size),
        'accuracy_drop_percent': float(baseline_acc_drop),
        'compression_ratio': float((1 - int8_size/fp32_size)*100)
    }
    
    # Save baseline INT8
    os.makedirs("models/quantized", exist_ok=True)
    torch.save(baseline_int8.state_dict(), "models/quantized/baseline_int8.pth")
    print("💾 Saved: models/quantized/baseline_int8.pth")
    
    # ========================================================================
    # PERFORATED MODEL (REQUIRES BPA → CPA → QUANTIZATION)
    # ========================================================================
    
    print("\n" + "="*70)
    print("🌿 PERFORATED MODEL (BPA → CPA → INT8)")
    print("="*70)
    
    # Load perforated FP32
    print("\n📥 Loading perforated FP32...")
    perforated_fp32 = DSCNN(num_classes=num_classes)
    
    # Initialize PAI
    GPA.pc.set_verbose(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.append_module_names_to_track(['BatchNorm2d'])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    perforated_fp32 = UPA.initialize_pai(
        perforated_fp32,
        doing_pai=True,
        save_name="dscnn_int8_eval",
        making_graphs=False,
        maximizing_score=True
    )
    
    save_name = "dscnn_perforated_10class"
    perforated_fp32 = UPA.load_system(perforated_fp32, save_name, 'best_model', True)
    perforated_fp32 = perforated_fp32.to(DEVICE)
    perforated_fp32.eval()
    
    perf_fp32_size = get_model_size(perforated_fp32)
    perf_fp32_params, _ = count_parameters(perforated_fp32)
    print(f"✓ Loaded: {perf_fp32_size:.2f} KB ({perf_fp32_params:,} params)")
    
    # Evaluate FP32
    print("📊 Evaluating FP32...")
    perf_fp32_acc, perf_fp32_f1 = evaluate_model(
        perforated_fp32, test_loader, DEVICE, "Perforated FP32"
    )
    print(f"  Accuracy: {perf_fp32_acc:.4f}")
    print(f"  F1 Score: {perf_fp32_f1:.4f}")
    
    results['perforated_fp32'] = {
        'accuracy': float(perf_fp32_acc),
        'f1_score': float(perf_fp32_f1),
        'size_kb': float(perf_fp32_size),
        'parameters': int(perf_fp32_params)
    }
    
    # ========================================================================
    # CRITICAL: Apply BPA + CPA before quantization (preserves dendrites)
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: APPLYING BPA + CPA (Convert dendrites to blocks)")
    print("="*70)
    
    print("🔧 Applying BPA.blockwise_network()...")
    perforated_fp32 = BPA.blockwise_network(perforated_fp32)
    print("✅ BPA applied!")
    
    print("🔧 Applying CPA.refresh_net()...")
    perforated_fp32 = CPA.refresh_net(perforated_fp32)
    print("✅ CPA applied!")
    
    bpa_params, _ = count_parameters(perforated_fp32)
    bpa_size = get_model_size(perforated_fp32)
    print(f"\n📊 After BPA/CPA:")
    print(f"   Parameters: {bpa_params:,}")
    print(f"   Size: {bpa_size:.2f} KB")
    
    # Now quantize
    print("\n" + "="*70)
    print("STEP 2: QUANTIZATION (After BPA/CPA)")
    print("="*70)
    
    perforated_int8 = quantize_to_int8(perforated_fp32, "perforated")
    perf_int8_size = get_model_size(perforated_int8)
    print(f"✓ INT8 Size: {perf_int8_size:.2f} KB")
    print(f"✓ Compression: {(1 - perf_int8_size/perf_fp32_size)*100:.1f}%")
    
    # Evaluate INT8
    print("📊 Evaluating INT8...")
    perf_int8_acc, perf_int8_f1 = evaluate_model(
        perforated_int8, test_loader, DEVICE, "Perforated INT8"
    )
    print(f"  Accuracy: {perf_int8_acc:.4f}")
    print(f"  F1 Score: {perf_int8_f1:.4f}")
    
    perf_acc_drop = (perf_fp32_acc - perf_int8_acc) * 100
    print(f"  Accuracy Drop: {perf_acc_drop:.2f}%")
    
    results['perforated_int8'] = {
        'accuracy': float(perf_int8_acc),
        'f1_score': float(perf_int8_f1),
        'size_kb': float(perf_int8_size),
        'accuracy_drop_percent': float(perf_acc_drop),
        'compression_ratio': float((1 - perf_int8_size/perf_fp32_size)*100),
        'bpa_cpa_applied': True
    }
    
    # Save perforated INT8
    torch.save(perforated_int8.state_dict(), "models/quantized/perforated_bpa_cpa_int8.pth")
    print("💾 Saved: models/quantized/perforated_bpa_cpa_int8.pth")
    
    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'F1 Score':<12} {'Size (KB)':<12} {'Acc Drop'}")
    print("-" * 80)
    
    print(f"{'Baseline FP32':<30} {baseline_fp32_acc:<12.4f} {baseline_fp32_f1:<12.4f} {fp32_size:<12.2f} {'-'}")
    print(f"{'Baseline INT8 (direct)':<30} {baseline_int8_acc:<12.4f} {baseline_int8_f1:<12.4f} {int8_size:<12.2f} {baseline_acc_drop:.2f}%")
    print()
    print(f"{'Perforated FP32':<30} {perf_fp32_acc:<12.4f} {perf_fp32_f1:<12.4f} {perf_fp32_size:<12.2f} {'-'}")
    print(f"{'Perforated INT8 (BPA+CPA)':<30} {perf_int8_acc:<12.4f} {perf_int8_f1:<12.4f} {perf_int8_size:<12.2f} {perf_acc_drop:.2f}%")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  Baseline quantization impact: {baseline_acc_drop:.2f}% accuracy loss")
    print(f"  Perforated quantization impact: {perf_acc_drop:.2f}% accuracy loss")
    
    if perf_acc_drop < baseline_acc_drop:
        diff = baseline_acc_drop - perf_acc_drop
        print(f"  ✅ Perforated is MORE ROBUST to quantization by {diff:.2f}%!")
    elif perf_acc_drop > baseline_acc_drop:
        diff = perf_acc_drop - baseline_acc_drop
        print(f"  ⚠️  Perforated is LESS ROBUST to quantization by {diff:.2f}%")
    else:
        print(f"  ↔️  Both models equally affected by quantization")
    
    # Save results
    results['summary'] = {
        'baseline_quantization_method': 'direct',
        'perforated_quantization_method': 'BPA + CPA + INT8',
        'baseline_acc_drop': float(baseline_acc_drop),
        'perforated_acc_drop': float(perf_acc_drop),
        'robustness_difference': float(baseline_acc_drop - perf_acc_drop)
    }
    
    with open("results/int8_quantization_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: results/int8_quantization_comparison.json")
    print("\n" + "="*70)
    print("✅ INT8 QUANTIZATION COMPARISON COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()

