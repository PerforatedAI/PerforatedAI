import matplotlib.pyplot as plt
import numpy as np

# Original: Peak 85.43% -> Jatuh ke 83.31% (Overfitting)
# Dendritic: Peak 85.98% -> Stabil di 85.82% (Robust)

labels = ['Standard ResNet-50', 'NeuroResNet (PAI)']
peak_scores = [85.43, 85.98]
final_scores = [83.31, 85.82]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, peak_scores, width, label='Peak Accuracy', color=['#95a5a6', '#f1c40f'], alpha=0.5)
rects2 = ax.bar(x + width/2, final_scores, width, label='Final Accuracy (Ep 50)', color=['#e74c3c', '#2ecc71'])

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Stability Test: Preventing Model Collapse (50 Epochs)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend()

ax.set_ylim(80, 88)
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

plt.text(1 + width/2, 86.5, "STABLE (+2.51%)", ha='center', color='green', fontweight='bold', fontsize=12)
plt.text(0 + width/2, 84.0, "COLLAPSED", ha='center', color='red', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('Accuracy Improvement.png', dpi=300)
print("Graph saved as 'Accuracy Improvement.png'")