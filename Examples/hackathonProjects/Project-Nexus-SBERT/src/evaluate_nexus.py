"""
NEXUS Evaluation & Visualization Suite
Generates the comparison assets for the README.
"""
import torch
import argparse
import os
import time
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def benchmark_inference(model, sentences, batch_size=1, num_runs=3):
    """Measure average inference time per sentence."""
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.encode(sentences, batch_size=batch_size, show_progress_bar=False)
        end_time = time.time()
        times.append((end_time - start_time) / len(sentences) * 1000)  # ms per sentence
    return np.mean(times), np.std(times)

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(baseline_path, dendritic_path, output_dir="assets"):
    print("📊 Starting NEXUS Evaluation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("mteb/stsbenchmark-sts")
    test_data = dataset['test']
    
    evaluator = EmbeddingSimilarityEvaluator(
        [x['sentence1'] for x in test_data],
        [x['sentence2'] for x in test_data],
        [float(x['score'])/5.0 for x in test_data],
        name='sts-test'
    )
    
    # Load Models
    print("Loading Baseline...")
    base_model = SentenceTransformer(baseline_path)
    print("Loading Dendritic...")
    dend_model = SentenceTransformer(dendritic_path)
    
    # 1. Accuracy Evaluation
    print("\n🎯 Evaluating Accuracy...")
    base_score = evaluator(base_model)
    dend_score = evaluator(dend_model)
    
    if isinstance(base_score, dict):
        base_score = base_score.get("sts-test_spearman_cosine", next(iter(base_score.values())))
    if isinstance(dend_score, dict):
        dend_score = dend_score.get("sts-test_spearman_cosine", next(iter(dend_score.values())))
    
    # 2. Latency Benchmarking
    print("⚡ Benchmarking Latency...")
    sample_sentences = [x['sentence1'] for x in test_data[:100]]
    base_latency, base_std = benchmark_inference(base_model, sample_sentences)
    dend_latency, dend_std = benchmark_inference(dend_model, sample_sentences)
    
    # 3. Parameter Count
    base_params = count_parameters(base_model)
    dend_params = count_parameters(dend_model)
    
    # Print Results
    print(f"\n{'='*60}")
    print(f"🏆 NEXUS EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<30} {'Baseline':<15} {'NEXUS':<15} {'Delta'}")
    print(f"{'-'*60}")
    print(f"{'Spearman Correlation':<30} {base_score:<15.4f} {dend_score:<15.4f} {'+' if dend_score > base_score else ''}{(dend_score - base_score):.4f}")
    print(f"{'Latency (ms/query)':<30} {base_latency:<15.2f} {dend_latency:<15.2f} {'+' if dend_latency > base_latency else ''}{(dend_latency - base_latency):.2f}")
    print(f"{'Parameters (M)':<30} {base_params/1e6:<15.2f} {dend_params/1e6:<15.2f} {'+' if dend_params > base_params else ''}{(dend_params - base_params)/1e6:.2f}")
    print(f"{'='*60}\n")
    
    # 4. Generate Visualizations
    print("📊 Generating Visualizations...")
    
    # Performance Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy
    labels = ['Baseline', 'NEXUS\n(Dendritic)']
    scores = [base_score, dend_score]
    colors = ['#6c757d', '#00ff00']
    
    axes[0].bar(labels, scores, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylim(min(scores) - 0.01, max(scores) + 0.01)
    axes[0].set_ylabel('Spearman Correlation', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (label, score) in enumerate(zip(labels, scores)):
        axes[0].text(i, score + 0.001, f'{score:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 2: Efficiency
    metrics = ['Latency\n(ms)', 'Parameters\n(M)']
    baseline_vals = [base_latency, base_params/1e6]
    dendritic_vals = [dend_latency, dend_params/1e6]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, baseline_vals, width, label='Baseline', color='#6c757d', alpha=0.8, edgecolor='black')
    axes[1].bar(x + width/2, dendritic_vals, width, label='NEXUS', color='#00ff00', alpha=0.8, edgecolor='black')
    
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/performance_comparison.png")
    
    # 5. Save Results to File
    with open(f"{output_dir}/evaluation_results.txt", "w") as f:
        f.write("NEXUS EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Baseline Model: {baseline_path}\n")
        f.write(f"Dendritic Model: {dendritic_path}\n\n")
        f.write(f"Spearman Correlation:\n")
        f.write(f"  Baseline: {base_score:.4f}\n")
        f.write(f"  NEXUS:    {dend_score:.4f}\n")
        f.write(f"  Delta:    {(dend_score - base_score):.4f} ({((dend_score - base_score)/base_score * 100):.2f}%)\n\n")
        f.write(f"Latency (ms/query):\n")
        f.write(f"  Baseline: {base_latency:.2f} ± {base_std:.2f}\n")
        f.write(f"  NEXUS:    {dend_latency:.2f} ± {dend_std:.2f}\n")
        f.write(f"  Delta:    {(dend_latency - base_latency):.2f} ms\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Baseline: {base_params:,} ({base_params/1e6:.2f}M)\n")
        f.write(f"  NEXUS:    {dend_params:,} ({dend_params/1e6:.2f}M)\n")
        f.write(f"  Delta:    {(dend_params - base_params):,} ({(dend_params - base_params)/1e6:.2f}M)\n")
    
    print(f"✅ Saved: {output_dir}/evaluation_results.txt")
    print("\n🎉 Evaluation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEXUS Evaluation Suite")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline model checkpoint")
    parser.add_argument("--dendritic", type=str, required=True, help="Path to dendritic model checkpoint")
    parser.add_argument("--output", type=str, default="../assets", help="Output directory for visualizations")
    args = parser.parse_args()
    
    evaluate(args.baseline, args.dendritic, args.output)
