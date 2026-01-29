"""
Edge Memory Benchmark
📉 Measure model size and memory usage
"""
import sys
sys.path.append('..')

import torch
import os
from models.baseline_model import BaselineModel
from models.dendritic_model import DendriticModel


def get_model_size(model, model_name):
    """Calculate model size in MB"""
    # Save model temporarily
    temp_path = f'temp_{model_name}.pt'
    torch.save(model.state_dict(), temp_path)
    
    # Get file size
    size_bytes = os.path.getsize(temp_path)
    size_mb = size_bytes / (1024 * 1024)
    
    # Clean up
    os.remove(temp_path)
    
    return size_mb


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def estimate_memory_usage(model, input_shape=(1, 5)):
    """Estimate memory usage during inference"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Measure memory before
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Estimate memory (simplified)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    param_memory_mb = param_memory / (1024 * 1024)
    
    return param_memory_mb


def main():
    """Main memory benchmarking function"""
    
    print("="*60)
    print("Edge Memory Benchmark")
    print("="*60)
    
    models = {
        'Baseline': BaselineModel(),
        'Dendritic': DendriticModel()
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{model_name} Model:")
        
        # Model size
        size_mb = get_model_size(model, model_name)
        print(f"  Model Size: {size_mb:.4f} MB")
        
        # Parameter count
        total_params, trainable_params = count_parameters(model)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        
        # Memory usage
        memory_mb = estimate_memory_usage(model)
        print(f"  Estimated Memory Usage: {memory_mb:.4f} MB")
        
        results.append({
            'Model': model_name,
            'Size (MB)': size_mb,
            'Total Params': total_params,
            'Trainable Params': trainable_params,
            'Memory (MB)': memory_mb
        })
    
    # Save results
    import csv
    with open('../reports/edge_benchmarks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])  # Empty line
        writer.writerow(['Model', 'Size (MB)', 'Total Params', 'Trainable Params', 'Memory (MB)'])
        
        for result in results:
            writer.writerow([
                result['Model'],
                f"{result['Size (MB)']:.4f}",
                result['Total Params'],
                result['Trainable Params'],
                f"{result['Memory (MB)']:.4f}"
            ])
    
    print("\n✅ Results appended to reports/edge_benchmarks.csv")


if __name__ == "__main__":
    main()
