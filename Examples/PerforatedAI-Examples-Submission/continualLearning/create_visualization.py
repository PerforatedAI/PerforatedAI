"""
Create comparison visualization for DendriticPlasticity results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/dendritic_plasticity_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Prepare data
methods = [r['name'] for r in results]
accuracies = [r['avg_final'] for r in results]
forgetting = [r['forgetting'] for r in results]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Colors for different method categories
colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
          '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA']

# Subplot 1: Average Accuracy
bars1 = ax1.bar(range(len(methods)), accuracies, color=colors, alpha=0.8,
                edgecolor='black', linewidth=1.5)

# Highlight best result (PAI + Replay)
bars1[5].set_color('#2ECC71')
bars1[5].set_alpha(1.0)
bars1[5].set_linewidth(2.5)

ax1.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Continual Learning Performance Comparison', fontsize=15, fontweight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
ax1.set_ylim([0, 60])
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, accuracies)):
    height = bar.get_height()
    label = f'{val:.1f}%'
    if i == 5:  # PAI + Replay
        label = f'* {val:.1f}%'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ECC71')
    else:
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontsize=8)

# Subplot 2: Forgetting Rate
bars2 = ax2.bar(range(len(methods)), forgetting, color=colors, alpha=0.8,
                edgecolor='black', linewidth=1.5)

# Highlight best result (lowest forgetting)
bars2[5].set_color('#2ECC71')
bars2[5].set_alpha(1.0)
bars2[5].set_linewidth(2.5)

ax2.set_ylabel('Task Forgetting Rate (%)', fontsize=13, fontweight='bold')
ax2.set_title('Catastrophic Forgetting Comparison', fontsize=15, fontweight='bold')
ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
ax2.set_ylim([0, 105])
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, forgetting)):
    height = bar.get_height()
    label = f'{val:.1f}%'
    if i == 5:
        label = f'* {val:.1f}%'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ECC71')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                label, ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/comparison_plot.png', dpi=300, bbox_inches='tight')
print("[OK] Created comparison_plot.png")

# Create method categories diagram
fig, ax = plt.subplots(figsize=(14, 8))

# Group methods by category
categories = {
    'Baseline': [0, 1],      # Baseline, Baseline+Replay
    'PAI Only': [4, 5],      # PAI, PAI+Replay
    'SP Only': [2, 3],       # SP, SP+Replay
    'Combined': [6, 7]       # DP, DP+Replay
}

category_colors = {
    'Baseline': '#FF6B6B',
    'PAI Only': '#FFE66D',
    'SP Only': '#F38181',
    'Combined': '#FCBAD3'
}

x_pos = 0
category_positions = []
category_labels = []

for cat_name, indices in categories.items():
    cat_accs = [accuracies[i] for i in indices]
    cat_methods = [methods[i] for i in indices]

    x_positions = [x_pos + i*0.8 for i in range(len(indices))]
    bars = ax.bar(x_positions, cat_accs, width=0.7,
                  color=category_colors[cat_name], alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    # Highlight PAI + Replay
    if cat_name == 'PAI Only' and len(bars) > 1:
        bars[1].set_alpha(1.0)
        bars[1].set_linewidth(2.5)
        bars[1].set_color('#2ECC71')

    # Add labels
    for bar, val, method in zip(bars, cat_accs, cat_methods):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(bar.get_x() + bar.get_width()/2., -3,
                method.replace(' + ', '+\n'), ha='center', va='top',
                fontsize=8, rotation=0)

    category_positions.append(x_pos + 0.4)
    category_labels.append(cat_name)
    x_pos += len(indices) * 0.8 + 1.5

ax.set_ylabel('Average Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('8-Method Systematic Comparison', fontsize=15, fontweight='bold')
ax.set_ylim([0, 60])
ax.set_xlim([-1, x_pos - 1.5])
ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2)

# Add category separators and labels
for i, (pos, label) in enumerate(zip(category_positions, category_labels)):
    if i > 0:
        ax.axvline(x=pos - 1.2, color='gray', linestyle=':', alpha=0.5, linewidth=2)
    ax.text(pos, 58, label, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_xticks([])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/method_categories.png', dpi=300, bbox_inches='tight')
print("[OK] Created method_categories.png")

print("\n[SUCCESS] All visualizations created!")
print("\nGenerated files:")
print("  - results/comparison_plot.png")
print("  - results/method_categories.png")
