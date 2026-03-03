"""
Create visualizations for SelectivePlasticity results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load results
with open('results/split_mnist_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Figure 1: Task A Retention Comparison (Highlight)
fig, ax = plt.subplots(figsize=(10, 6))

methods = [r['method'] for r in results]
retentions = [r['task_a_final'] for r in results]
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']

bars = ax.bar(methods, retentions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight the best result
bars[3].set_color('#2ECC71')
bars[3].set_alpha(1.0)
bars[3].set_linewidth(2.5)

ax.set_ylabel('Task A Retention After Learning Task B (%)', fontsize=13, fontweight='bold')
ax.set_title('SelectivePlasticity: 86.55% Task Retention', fontsize=16, fontweight='bold')
ax.set_ylim([0, 100])
ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, retentions)):
    height = bar.get_height()
    label = f'{val:.1f}%'
    if i == 3:  # Highlight best result
        label = f'* {val:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2ECC71')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                label, ha='center', va='bottom', fontsize=11)

plt.xticks(rotation=15, ha='right')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/retention_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Created retention_comparison.png")

# Figure 2: Forgetting Rate Reduction
fig, ax = plt.subplots(figsize=(10, 6))

forgetting = [r['forgetting_rate'] for r in results]
colors_inverted = ['#E74C3C', '#E67E22', '#F39C12', '#27AE60']

bars = ax.bar(methods, forgetting, color=colors_inverted, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight the best result (lowest forgetting)
bars[3].set_color('#27AE60')
bars[3].set_alpha(1.0)
bars[3].set_linewidth(2.5)

ax.set_ylabel('Catastrophic Forgetting Rate (%)', fontsize=13, fontweight='bold')
ax.set_title('SelectivePlasticity: 87% Reduction in Forgetting', fontsize=16, fontweight='bold')
ax.set_ylim([0, 110])

# Add value labels
for i, (bar, val) in enumerate(zip(bars, forgetting)):
    height = bar.get_height()
    label = f'{val:.1f}%'
    if i == 3:
        label = f'* {val:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                label, ha='center', va='bottom', fontsize=14, fontweight='bold', color='#27AE60')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                label, ha='center', va='bottom', fontsize=11)

# Add reduction annotation
ax.annotate('', xy=(0, forgetting[0]), xytext=(3, forgetting[3]),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.text(1.5, 55, '87% Reduction', fontsize=12, fontweight='bold',
        color='purple', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', linewidth=2))

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('results/forgetting_reduction.png', dpi=300, bbox_inches='tight')
print("[OK] Created forgetting_reduction.png")

# Figure 3: Combined Performance Metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Average Accuracy
avg_acc = [r['average_accuracy'] for r in results]
bars1 = ax1.bar(methods, avg_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
bars1[3].set_color('#2ECC71')
bars1[3].set_alpha(1.0)
bars1[3].set_linewidth(2.5)

ax1.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Average Task Performance', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars1, avg_acc)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
             fontweight='bold' if i == 3 else 'normal')

# Right: Task A Retention
bars2 = ax2.bar(methods, retentions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2[3].set_color('#2ECC71')
bars2[3].set_alpha(1.0)
bars2[3].set_linewidth(2.5)

ax2.set_ylabel('Task A Retention (%)', fontsize=12, fontweight='bold')
ax2.set_title('Memory Retention After New Task', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1.5)

for i, (bar, val) in enumerate(zip(bars2, retentions)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
             fontweight='bold' if i == 3 else 'normal')

for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('results/combined_metrics.png', dpi=300, bbox_inches='tight')
print("[OK] Created combined_metrics.png")

print("\n[SUCCESS] All visualizations created successfully!")
print("\nGenerated files:")
print("  - results/retention_comparison.png")
print("  - results/forgetting_reduction.png")
print("  - results/combined_metrics.png")
