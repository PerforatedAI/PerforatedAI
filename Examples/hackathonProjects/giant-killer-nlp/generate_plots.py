"""
Generate training visualization plots for README
"""
import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Training data from the README
epochs = list(range(1, 10))
train_loss = [0.9398, 0.7157, 0.6272, 0.5914, 0.4974, 0.4415, 0.4464, 0.3912, 0.3267]
val_loss = [0.6893, 0.6859, 0.6290, 0.6491, 0.6331, 0.5669, 0.7070, 0.8594, 0.9265]
train_acc = [70.00, 70.52, 76.14, 83.72, 83.70, 90.16, 88.90, 91.44, 93.54]
val_acc = [73.20, 83.20, 77.50, 83.70, 84.80, 78.20, 88.90, 91.30, 91.30]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training and Validation Loss
ax1.plot(epochs, train_loss, marker='o', linewidth=2, label='Training Loss', color='#2E86AB', markersize=8)
ax1.plot(epochs, val_loss, marker='s', linewidth=2, label='Validation Loss', color='#A23B72', markersize=8)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Loss', fontweight='bold')
ax1.set_title('Training and Validation Loss Over Epochs', fontweight='bold', pad=15)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 9.5)
ax1.set_xticks(epochs)

# Mark best validation loss point
best_epoch = 6
ax1.scatter([best_epoch], [val_loss[best_epoch-1]], s=200, c='red', marker='*', 
           zorder=5, label='Best Model (Epoch 6)')
ax1.legend(loc='upper right')

# Plot 2: Training and Validation Accuracy
ax2.plot(epochs, train_acc, marker='o', linewidth=2, label='Training Accuracy', color='#06A77D', markersize=8)
ax2.plot(epochs, val_acc, marker='s', linewidth=2, label='Validation Accuracy', color='#F18F01', markersize=8)
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_title('Training and Validation Accuracy Over Epochs', fontweight='bold', pad=15)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 9.5)
ax2.set_ylim(65, 95)
ax2.set_xticks(epochs)

# Mark best validation epoch
ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Best Model')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_curves.png")
plt.close()

# Create a second figure for learning rate and convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 3: Loss Comparison
ax1.plot(epochs, train_loss, marker='o', linewidth=2.5, label='Training Loss', color='#2E86AB', markersize=8)
ax1.plot(epochs, val_loss, marker='s', linewidth=2.5, label='Validation Loss', color='#A23B72', markersize=8)
ax1.fill_between(epochs, train_loss, alpha=0.2, color='#2E86AB')
ax1.fill_between(epochs, val_loss, alpha=0.2, color='#A23B72')
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Loss', fontweight='bold')
ax1.set_title('Loss Convergence', fontweight='bold', pad=15)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 9.5)
ax1.set_xticks(epochs)

# Add annotations for key events
ax1.annotate('Early stopping triggered', 
            xy=(9, val_loss[-1]), xytext=(7, 1.0),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red', fontweight='bold')

# Plot 4: Overfitting Analysis (Gap between train and val)
gap = [abs(train_acc[i] - val_acc[i]) for i in range(len(epochs))]
ax2.bar(epochs, gap, color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Train-Val Accuracy Gap (%)', fontweight='bold')
ax2.set_title('Overfitting Analysis (Train-Val Gap)', fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(0.5, 9.5)
ax2.set_xticks(epochs)
ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Acceptable Gap (<5%)')
ax2.legend(loc='upper left')

# Add average line
avg_gap = np.mean(gap)
ax2.axhline(y=avg_gap, color='blue', linestyle=':', alpha=0.7, linewidth=2, label=f'Average Gap ({avg_gap:.1f}%)')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_analysis.png")
plt.close()

print("\nPlots generated successfully!")
print("- training_curves.png: Main training and validation metrics")
print("- training_analysis.png: Loss convergence and overfitting analysis")
