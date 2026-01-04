import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load metrics
# Load metrics
baseline_path = 'experiments/baseline_optimized/metrics.json'
dendritic_path = 'experiments/dendritic_optimized/metrics.json'

# Wait for files to exist or default to empty list to prevent crash
if not os.path.exists(baseline_path):
    print(f"⚠️ Baseline (15 epochs) not ready yet.")
    baseline = []
else:
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
    except:
        baseline = []

if not os.path.exists(dendritic_path):
    print(f"⚠️ Dendritic (15 epochs) not ready yet.")
    dendritic = []
else:
    try:
        with open(dendritic_path, 'r') as f:
            dendritic = json.load(f)
    except:
        dendritic = []

print(f"Baseline loaded from: {baseline_path}")
print(f"Baseline has {len(baseline)} epochs")
print(f"Dendritic has {len(dendritic)} epochs")

# Extract data
baseline_epochs = [m['epoch'] for m in baseline]
baseline_loss = [m['train_loss'] for m in baseline]
baseline_spearman = [m['val_spearman'] for m in baseline]

dendritic_epochs = [m['epoch'] for m in dendritic]
dendritic_loss = [m['train_loss'] for m in dendritic]
dendritic_spearman = [m['val_spearman'] for m in dendritic]

# --- Graph 1: Training Loss (Green vs Red) ---
plt.figure(figsize=(10, 6))

# Baseline (Red - what would have happened)
plt.plot(baseline_epochs, baseline_loss, label=f'Baseline SBERT ({len(baseline)} epochs)', 
         linewidth=2.5, color='#D9534F', linestyle='--', alpha=0.8) # Red dashed

# Dendritic (Green - Actual)
plt.plot(dendritic_epochs, dendritic_loss, label=f'Dendritic SBERT ({len(dendritic)} epochs)', 
         linewidth=2.5, color='#5CB85C') # Green

# Add vertical line to show where dendrites were added
# Assuming epoch 10 (index 9) is the switch point based on previous code
switch_epoch = 9
plt.axvline(x=switch_epoch, color='#4A90E2', linestyle='-', linewidth=2, alpha=0.6, 
            label='Dendrites Added')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss: Baseline vs Dendritic SBERT', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('raw_loss_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Loss comparison graph created: raw_loss_comparison.png")

# --- Graph 2: Validation Spearman (Orange vs Blue) ---
plt.figure(figsize=(10, 6))

# Baseline (Blue - what would have happened)
plt.plot(baseline_epochs, baseline_spearman, label=f'Baseline SBERT ({len(baseline)} epochs)', 
         linewidth=2.5, color='#4A90E2', linestyle='--', alpha=0.8) # Blue dashed

# Dendritic (Orange - Actual)
plt.plot(dendritic_epochs, dendritic_spearman, label=f'Dendritic SBERT ({len(dendritic)} epochs)', 
         linewidth=2.5, color='#F0AD4E') # Orange

# Add vertical line
plt.axvline(x=switch_epoch, color='#4A90E2', linestyle='-', linewidth=2, alpha=0.6,
            label='Dendrites Added')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Spearman Correlation', fontsize=12)
plt.title('Validation Performance: Baseline vs Dendritic SBERT', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3)
# plt.ylim([0.884, 0.893]) # Removed hardcoded limits to fit new data
plt.tight_layout()

plt.savefig('spearman_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Spearman comparison graph created: spearman_comparison.png")

print("\n" + "="*60)
print("FINAL STATISTICS")
print("="*60)

if baseline_spearman:
    print(f"Baseline Best Spearman: {max(baseline_spearman):.4f}")
else:
    print("Baseline Best Spearman: N/A (No data yet)")

if dendritic_spearman:
    print(f"Dendritic Best Spearman: {max(dendritic_spearman):.4f}")
else:
    print("Dendritic Best Spearman: N/A (No data yet)")
