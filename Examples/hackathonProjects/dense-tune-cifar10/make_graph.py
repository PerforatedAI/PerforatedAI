import matplotlib.pyplot as plt
import numpy as np

# Dense-Tune CIFAR-10 Results
# Baseline: 84.59% -> Dendritic: 86.85% (Stable improvement)

labels = ['Traditional CNN', 'DendriticCIFARNet']
baseline_scores = [84.59, 84.59]  # Starting point
final_scores = [84.59, 86.85]     # Final accuracy

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color=['#95a5a6', '#f39c12'], alpha=0.5)
rects2 = ax.bar(x + width/2, final_scores, width, label='Final Accuracy', color=['#e74c3c', '#27ae60'])

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('CIFAR-10: Dendritic Optimization Breakthrough', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend()

ax.set_ylim(80, 90)
ax.grid(axis='y', linestyle='--', alpha=0.3)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Annotations
plt.text(1 + width/2, 87.5, "+2.26%\n(14.7% RER)", ha='center', color='green', fontweight='bold', fontsize=11)
plt.text(0 + width/2, 85.5, "BASELINE", ha='center', color='gray', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('Accuracy Improvement.png', dpi=300)
print("Graph saved as 'Accuracy Improvement.png'")
