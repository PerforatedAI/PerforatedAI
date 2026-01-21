"""
Edge Latency & Efficiency Benchmark
Compare Dendritic Model vs. Baseline MLP on CPU (Edge Simulation)
"""
import sys
import os
import time
import torch
import torch.nn as nn
import tracemalloc
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dendritic_model import DendriticModel
from training.train_baseline import BaselineModel

def measure_inference(model, input_sample, num_loops=1000):
    """Measure average latency over N loops"""
    # Warmup
    for _ in range(100):
        _ = model(input_sample)
    
    start_time = time.perf_counter_ns()
    for _ in range(num_loops):
        _ = model(input_sample)
    end_time = time.perf_counter_ns()
    
    total_time_ms = (end_time - start_time) / 1e6
    avg_latency_ms = total_time_ms / num_loops
    return avg_latency_ms

def measure_memory(model, input_sample):
    """Measure peak memory usage"""
    tracemalloc.start()
    _ = model(input_sample)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024  # KB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# -------------------------------------------------------------
# Lite Version (No Soma) - 529 Params
# -------------------------------------------------------------
class DendriticLayerLite(nn.Module):
    def __init__(self, input_dim, output_dim, num_dendrites=4):
        super(DendriticLayerLite, self).__init__()
        self.dendrites = nn.ModuleList([
            nn.Linear(input_dim, output_dim // num_dendrites)
            for _ in range(num_dendrites)
        ])
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        dendrite_outputs = [dendrite(x) for dendrite in self.dendrites]
        combined = torch.cat(dendrite_outputs, dim=-1)
        combined = self.bn(combined)
        return self.activation(combined) # No Soma mixing

class DendriticModelLite(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=24, output_dim=1, num_dendrites=4):
        super(DendriticModelLite, self).__init__()
        # Note: Lite version relies purely on finding specific feature combos
        self.dendritic_layer1 = DendriticLayerLite(input_dim, hidden_dim, num_dendrites)
        self.dendritic_layer2 = DendriticLayerLite(hidden_dim, hidden_dim // 2, num_dendrites)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dendritic_layer1(x)
        x = self.dropout(x)
        x = self.dendritic_layer2(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
# -------------------------------------------------------------

def run_benchmark():
    print("Starting Edge Efficiency Benchmark...")
    
    # Initialize Models
    baseline = BaselineModel(input_dim=5, output_dim=1)
    dendritic_std = DendriticModel(input_dim=5, hidden_dim=24, output_dim=1, num_dendrites=4)
    dendritic_lite = DendriticModelLite(input_dim=5, hidden_dim=24, output_dim=1, num_dendrites=4) # 529 params
    
    baseline.eval()
    dendritic_std.eval()
    dendritic_lite.eval()
    
    # Dummy Input (Single sample for edge simulation)
    input_sample = torch.randn(1, 5)
    
    results = {}
    
    models_to_test = [
        ("Baseline (MLP)", baseline), 
        ("Dendritic (Std)", dendritic_std),
        ("Dendritic (Lite)", dendritic_lite)
    ]

    for name, model in models_to_test:
        print(f"\nBenchmarking {name}...")
        
        # Parameters
        params = count_parameters(model)
        
        # Latency
        latency = measure_inference(model, input_sample)
        
        # Memory
        memory = measure_memory(model, input_sample)
        
        print(f"  - Params: {params:,}")
        print(f"  - Latency: {latency:.4f} ms/sample")
        print(f"  - Memory: {memory:.2f} KB")
        
        results[name] = {
            "parameters": params,
            "latency_ms": latency,
            "memory_kb": memory
        }
        
    # Calculate Improvement
    base_lat = results["Baseline (MLP)"]["latency_ms"]
    
    print("\n" + "="*40)
    for name in ["Dendritic (Std)", "Dendritic (Lite)"]:
        lat = results[name]["latency_ms"]
        speedup = base_lat / lat
        print(f"🏆 {name} vs Baseline: {speedup:.2f}x speedup")
    print("="*40)
    
    # Save Results
    with open("benchmarks/edge_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Results saved to benchmarks/edge_results.json")

if __name__ == "__main__":
    run_benchmark()
