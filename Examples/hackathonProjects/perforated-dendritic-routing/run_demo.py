import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.baseline_model import BaselineModel
from src.dendritic_router import DendriticRouter

# Configuration (safe defaults)
torch.manual_seed(0)

DIM = 256
BATCH = 128
RUNS = 300
THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 1.0]

# Setup input and model
x = torch.randn(BATCH, DIM)

baseline = BaselineModel(DIM)
baseline.eval()

# Timing utility
def measure_latency(model):
    with torch.no_grad():
        # warm-up
        for _ in range(50):
            model(x)

        start = time.perf_counter()
        for _ in range(RUNS):
            model(x)
        end = time.perf_counter()

    return (end - start) * 1000 / RUNS  # ms

# Baseline reference
baseline_latency = measure_latency(baseline)
baseline_output = baseline(x)

baseline_throughput = BATCH / (baseline_latency / 1000.0)

print("\n=== Baseline Reference ===")
print(f"Latency        : {baseline_latency:.3f} ms")
print(f"Throughput     : {baseline_throughput:.1f} samples/sec")

# Sweep dendritic thresholds
results = []

for t in THRESHOLDS:
    model = DendriticRouter(baseline, threshold=t)
    model.eval()

    latency = measure_latency(model)
    throughput = BATCH / (latency / 1000.0)

    with torch.no_grad():
        y = model(x)

    # correctness metrics
    max_diff = (baseline_output - y).abs().max().item()
    mean_diff = (baseline_output - y).abs().mean().item()

    # routing metrics
    active_fraction = (x.abs().mean(dim=1) > t).float().mean().item()
    effective_compute = active_fraction * 100.0

    results.append({
        "threshold": t,
        "latency": latency,
        "speedup": baseline_latency / latency,
        "throughput": throughput,
        "active_pct": effective_compute,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    })

# Console summary

print("\n=== Dendritic Inference Results ===")
print(
    "thr | latency(ms) | speedup | throughput | active% | maxΔ | meanΔ"
)
print("-" * 72)

for r in results:
    print(
        f"{r['threshold']:.1f} | "
        f"{r['latency']:.3f} | "
        f"{r['speedup']:.2f}x | "
        f"{r['throughput']:.0f} | "
        f"{r['active_pct']:.1f} | "
        f"{r['max_diff']:.2e} | "
        f"{r['mean_diff']:.2e}"
    )

# -----------------------------
# Overhead analysis
# -----------------------------
overhead = results[0]["latency"] / baseline_latency
print("\nRouting Overhead @ 100% active:")
print(f"Overhead ratio: {overhead:.3f}x")

# -----------------------------
# Visualization
# -----------------------------
active = [r["active_pct"] for r in results]
latencies = [r["latency"] for r in results]
speedups = [r["speedup"] for r in results]

plt.figure(figsize=(6, 4))
plt.plot(active, latencies, marker="o")
plt.xlabel("Active computation (%)")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Active Computation")
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_tradeoff.png")

plt.figure(figsize=(6, 4))
plt.plot(active, speedups, marker="o")
plt.xlabel("Active computation (%)")
plt.ylabel("Speedup (×)")
plt.title("Speedup vs Active Computation")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_tradeoff.png")

print("\nSaved plots:")
print(" - latency_tradeoff.png")
print(" - speedup_tradeoff.png")

print("\n[INFO] Demo completed successfully.")
