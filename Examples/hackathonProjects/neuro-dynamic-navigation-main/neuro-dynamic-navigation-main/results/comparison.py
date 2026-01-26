import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# ⚙️ SETTINGS: EXACT FILENAMES FROM YOUR UPLOAD
# ==========================================
PAI_CSV = "dqn_pai_20260114-164320(3).csv"        # Identified as Reward Data
DOUBLE_CSV = "DOUBLE_DQN_20260114-151701(1).csv"  # Identified as Reward Data
STD_CSV = "STANDARD_DQN_20260114-182014(1).csv"   # Identified as Reward Data

OUTPUT_FILE = "comparison_plot.png"
# ==========================================

def smooth(scalars, weight=0.85):
    """ Smooths the noisy RL data for a cleaner look """
    if len(scalars) == 0: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_graph():
    plt.figure(figsize=(10, 6), dpi=150)
    
    # --- 1. Plot PAI Agent (GREEN) ---
    if os.path.exists(PAI_CSV):
        df = pd.read_csv(PAI_CSV)
        steps = df['Step']
        values = df['Value']
        plt.plot(steps, smooth(values), color='#2ecc71', linewidth=2.5, label='Dendritic PAI (Ours)')
        plt.plot(steps, values, color='#2ecc71', alpha=0.15)
        print(f"✅ Loaded PAI: {PAI_CSV}")
    else:
        print(f"⚠️ Missing: {PAI_CSV}")

    # --- 2. Plot Double DQN (ORANGE) ---
    if os.path.exists(DOUBLE_CSV):
        df = pd.read_csv(DOUBLE_CSV)
        steps = df['Step']
        values = df['Value']
        plt.plot(steps, smooth(values), color='#e67e22', linewidth=2, label='Baseline (Double DQN)')
        plt.plot(steps, values, color='#e67e22', alpha=0.15)
        print(f"✅ Loaded Double: {DOUBLE_CSV}")
    else:
        print(f"⚠️ Missing: {DOUBLE_CSV}")

    # --- 3. Plot Standard DQN (RED) ---
    if os.path.exists(STD_CSV):
        df = pd.read_csv(STD_CSV)
        steps = df['Step']
        values = df['Value']
        plt.plot(steps, smooth(values), color='#e74c3c', linewidth=2, linestyle='--', label='Standard DQN')
        plt.plot(steps, values, color='#e74c3c', alpha=0.1)
        print(f"✅ Loaded Standard: {STD_CSV}")
    else:
        print(f"⚠️ Missing: {STD_CSV}")

    # --- Styling ---
    plt.title("Learning Efficiency: Dendritic PAI vs Baselines", fontsize=14, fontweight='bold')
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"\n🎉 Graph saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_graph()