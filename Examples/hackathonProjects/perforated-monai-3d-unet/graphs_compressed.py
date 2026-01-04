import matplotlib.pyplot as plt

# ✅ FINAL COMPRESSED RESULTS
models = ["Compressed Baseline UNet", "Compressed UNet + Dendrites"]
dice = [0.1547, 0.1851]

baseline, dendritic = dice
improvement = dendritic - baseline
improvement_pct = (improvement / baseline) * 100

plt.figure(figsize=(6, 4))

bars = plt.bar(
    models,
    dice,
    color=["#cfcfcf", "#d89c4a"],
    edgecolor="black",
    linewidth=1.2
)

plt.ylabel("Validation Dice")
plt.title("Accuracy Improvement with Dendritic Optimization")

# Tight y-range like other submissions
plt.ylim(0.14, 0.20)

# Subtle grid
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Value labels
for bar, val in zip(bars, dice):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.002,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Improvement annotation
plt.text(
    0.5,
    0.193,
    f"+{improvement:.3f} Dice  (+{improvement_pct:.1f}%)",
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="#444444"
)

plt.tight_layout()
plt.savefig("accuracy_improvement_compressed.png", dpi=200)
plt.show()
