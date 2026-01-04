import json
import matplotlib.pyplot as plt

# Load baseline
with open('experiments/baseline/metrics.json', 'r') as f:
    baseline = json.load(f)

# Load the 6-epoch backup (this was separate training!)
with open('experiments/dendritic_6EPOCH_BACKUP/metrics.json', 'r') as f:
    dendritic = json.load(f)

print(f"Baseline: {len(baseline)} epochs")
print(f"Dendritic (6-epoch backup): {len(dendritic)} epochs")
print(f"\nBaseline best Spearman: {max([m['val_spearman'] for m in baseline]):.4f}")
print(f"Dendritic best Spearman: {max([m['val_spearman'] for m in dendritic]):.4f}")

# Extract data
baseline_epochs = [m['epoch'] for m in baseline]
baseline_loss = [m['train_loss'] for m in baseline]
baseline_spearman = [m['val_spearman'] for m in baseline]

dendritic_epochs = [m['epoch'] for m in dendritic]
dendritic_loss = [m['train_loss'] for m in dendritic]
dendritic_spearman = [m['val_spearman'] for m in dendritic]

# Loss comparison
plt.figure(figsize=(10, 6))
plt.plot(baseline_epochs, baseline_loss, label='Baseline SBERT', 
         linewidth=2.5, color='#4A90E2', marker='o', markersize=6)
plt.plot(dendritic_epochs, dendritic_loss, label='Dendritic SBERT', 
         linewidth=2.5, color='#E85D4A', marker='s', markersize=6)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('raw_loss_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ raw_loss_comparison.png created")

# Spearman comparison
plt.figure(figsize=(10, 6))
plt.plot(baseline_epochs, baseline_spearman, label='Baseline SBERT', 
         linewidth=2.5, color='#4A90E2', marker='o', markersize=6)
plt.plot(dendritic_epochs, dendritic_spearman, label='Dendritic SBERT', 
         linewidth=2.5, color='#E85D4A', marker='s', markersize=6)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Spearman Correlation', fontsize=12)
plt.title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim([0.885, 0.892])
plt.tight_layout()
plt.savefig('spearman_comparison.png', dpi=150, bbox_inches='tight')
print("✓ spearman_comparison.png created")

print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
print(f"Baseline:  Best Spearman = {max(baseline_spearman):.4f} (epoch {baseline_spearman.index(max(baseline_spearman))})")
print(f"Dendritic: Best Spearman = {max(dendritic_spearman):.4f} (epoch {dendritic_spearman.index(max(dendritic_spearman))})")
print(f"Improvement: +{(max(dendritic_spearman) - max(baseline_spearman)):.4f}")
print(f"\nBaseline final loss:  {baseline_loss[-1]:.6f}")
print(f"Dendritic final loss: {dendritic_loss[-1]:.6f}")
print(f"Loss improvement: {((baseline_loss[-1] - dendritic_loss[-1])/baseline_loss[-1]*100):.1f}%")
