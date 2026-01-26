"""
Edge Inference Latency Benchmark
📱 Measure inference speed for edge deployment
"""
import sys
sys.path.append('..')

import torch
import time
import numpy as np
from models.baseline_model import BaselineModel
from models.dendritic_model import DendriticModel


def benchmark_latency(model, input_shape=(1, 5), num_iterations=1000):
    """Benchmark model inference latency"""
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(*input_shape)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            input_tensor = torch.randn(*input_shape)
            
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99)
    }


def main():
    """Main benchmarking function"""
    
    print("="*60)
    print("Edge Inference Latency Benchmark")
    print("="*60)
    
    # Benchmark baseline model
    print("\nBaseline Model:")
    baseline_model = BaselineModel()
    baseline_results = benchmark_latency(baseline_model)
    for metric, value in baseline_results.items():
        print(f"  {metric}: {value:.4f} ms")
    
    # Benchmark dendritic model
    print("\nDendritic Model:")
    dendritic_model = DendriticModel()
    dendritic_results = benchmark_latency(dendritic_model)
    for metric, value in dendritic_results.items():
        print(f"  {metric}: {value:.4f} ms")
    
    # Save results
    import csv
    with open('../reports/edge_benchmarks.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Metric', 'Value (ms)'])
        
        for metric, value in baseline_results.items():
            writer.writerow(['Baseline', metric, f"{value:.4f}"])
        
        for metric, value in dendritic_results.items():
            writer.writerow(['Dendritic', metric, f"{value:.4f}"])
    
    print("\n✅ Results saved to reports/edge_benchmarks.csv")


if __name__ == "__main__":
    main()
