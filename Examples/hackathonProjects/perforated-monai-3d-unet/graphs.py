import matplotlib.pyplot as plt

models = ["Baseline UNet", "UNet + Dendrites"]
dice = [0.20653, 0.26715]

plt.figure(figsize=(6, 4))

bars = plt.bar(
    models,
    dice,
    color=["#cfcfcf", "#d89c4a"],
    edgecolor="black"
)

plt.ylabel("Validation Dice")
plt.title("Accuracy Improvement with Dendritic Optimization")

# ✅ Correct Y-axis so baseline is visible
plt.ylim(0.18, 0.30)

# Optional: subtle grid like other submissions
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Value labels
for bar, val in zip(bars, dice):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.004,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Optional: improvement annotation
improvement = dice[1] - dice[0]
plt.text(
    0.5,
    0.285,
    f"+{improvement:.3f} Dice (+29.3%)",
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="#444444"
)

plt.tight_layout()
plt.savefig("accuracy_improvement.png", dpi=200)
plt.show()
