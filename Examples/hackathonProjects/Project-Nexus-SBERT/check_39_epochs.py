import json

# Load 39-epoch dendritic results
with open('experiments/dendritic/metrics.json', 'r') as f:
    dendritic = json.load(f)

# Load baseline
with open('experiments/baseline/metrics.json', 'r') as f:
    baseline = json.load(f)

print("="*60)
print("39-EPOCH DENDRITIC RESULTS")
print("="*60)

best_dend_spearman = max([m['val_spearman'] for m in dendritic])
best_dend_epoch = [m['epoch'] for m in dendritic if m['val_spearman'] == best_dend_spearman][0]

print(f"Total epochs: {len(dendritic)}")
print(f"Best Spearman: {best_dend_spearman:.4f} (epoch {best_dend_epoch})")
print(f"Final Spearman: {dendritic[-1]['val_spearman']:.4f} (epoch 38)")
print(f"Final Loss: {dendritic[-1]['train_loss']:.6f}")

print("\n" + "="*60)
print("BASELINE RESULTS")
print("="*60)

best_base_spearman = max([m['val_spearman'] for m in baseline])
best_base_epoch = [m['epoch'] for m in baseline if m['val_spearman'] == best_base_spearman][0]

print(f"Total epochs: {len(baseline)}")
print(f"Best Spearman: {best_base_spearman:.4f} (epoch {best_base_epoch})")
print(f"Final Loss: {baseline[-1]['train_loss']:.6f}")

print("\n" + "="*60)
print("COMPARISON - 39 EPOCHS VS BASELINE")
print("="*60)

delta = (best_dend_spearman - best_base_spearman) * 100
print(f"Best Spearman: {best_dend_spearman:.4f} vs {best_base_spearman:.4f}")
print(f"Difference: {delta:+.3f}%")

if best_dend_spearman > best_base_spearman:
    print(f"✓ IMPROVEMENT: {delta:.3f}% accuracy gain")
elif best_dend_spearman == best_base_spearman:
    print(f"= EQUAL: No change in accuracy")
else:
    print(f"✗ WORSE: {abs(delta):.3f}% accuracy drop")

loss_improvement = (baseline[-1]['train_loss'] - dendritic[-1]['train_loss']) / baseline[-1]['train_loss'] * 100
print(f"\nLoss convergence: {loss_improvement:.1f}% better")
print(f"  Baseline final: {baseline[-1]['train_loss']:.6f}")
print(f"  Dendritic final: {dendritic[-1]['train_loss']:.6f}")
