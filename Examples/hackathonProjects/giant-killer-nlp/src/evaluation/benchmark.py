"""
Benchmarking and Evaluation Utilities

This module provides tools for:
1. Computing classification metrics (F1, accuracy, precision, recall)
2. Benchmarking latency on CPU for edge deployment
3. Comparing Dendritic BERT-Tiny vs BERT-Base
4. Memory footprint analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    model_name: str
    num_parameters: int
    model_size_mb: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc_roc: Optional[float]
    avg_latency_ms: float
    std_latency_ms: float
    throughput_samples_per_sec: float


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Optional probability scores for AUC-ROC
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    
    if y_proba is not None:
        try:
            # Use probability of positive class for AUC-ROC
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None
    
    return metrics


def benchmark_latency(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device = torch.device("cpu"),
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Benchmark model latency on CPU for edge deployment.
    
    Target latencies:
    - BERT-Base: ~200ms for 100 sentences
    - Dendritic BERT-Tiny: ~5-10ms for 100 sentences
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        texts: Sample texts for benchmarking
        device: Device to run on (default CPU for edge)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs to exclude
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    model.to(device)
    
    # Prepare input
    sample_texts = texts[:batch_size] if len(texts) >= batch_size else texts
    encodings = tokenizer(
        sample_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    ).to(device)
    
    latencies = []
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
    
    # Timed runs
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "avg_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_samples_per_sec": 1000 / np.mean(latencies) * batch_size,
    }


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and parameter count.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    num_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        "num_parameters": num_parameters,
        "trainable_parameters": trainable_parameters,
        "size_bytes": total_size,
        "size_mb": total_size / (1024 * 1024),
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device for computation
        
    Returns:
        Tuple of (true labels, predictions, probability scores)
    """
    model.eval()
    model.to(device)
    
    all_labels = []
    all_predictions = []
    all_probas = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs["logits"]
            probas = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
    
    all_probas = np.vstack(all_probas)
    
    return all_labels, all_predictions, all_probas


def compare_models(
    tiny_model: nn.Module,
    base_model: nn.Module,
    test_loader: DataLoader,
    tokenizer: Any,
    sample_texts: List[str],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, BenchmarkResults]:
    """
    Compare Dendritic BERT-Tiny against BERT-Base.
    
    This is the core comparison that proves the "Giant-Killer" hypothesis.
    We expect:
    - F1 score gap: < 2%
    - Speed improvement: 15-40x
    
    Args:
        tiny_model: BERT-Tiny with dendritic optimization
        base_model: BERT-Base for comparison
        test_loader: Test data loader
        tokenizer: Tokenizer for latency benchmarking
        sample_texts: Sample texts for latency tests
        device: Device for evaluation
        
    Returns:
        Dictionary with benchmark results for each model
    """
    results = {}
    
    # Evaluate BERT-Tiny
    print("\nEvaluating Dendritic BERT-Tiny...")
    tiny_labels, tiny_preds, tiny_probas = evaluate_model(tiny_model, test_loader, device)
    tiny_metrics = compute_metrics(tiny_labels, tiny_preds, tiny_probas)
    tiny_size = get_model_size(tiny_model)
    tiny_latency = benchmark_latency(tiny_model, tokenizer, sample_texts, device)
    
    results["dendritic_bert_tiny"] = BenchmarkResults(
        model_name="Dendritic BERT-Tiny",
        num_parameters=tiny_size["num_parameters"],
        model_size_mb=tiny_size["size_mb"],
        accuracy=tiny_metrics["accuracy"],
        f1_score=tiny_metrics["f1"],
        precision=tiny_metrics["precision"],
        recall=tiny_metrics["recall"],
        auc_roc=tiny_metrics["auc_roc"],
        avg_latency_ms=tiny_latency["avg_latency_ms"],
        std_latency_ms=tiny_latency["std_latency_ms"],
        throughput_samples_per_sec=tiny_latency["throughput_samples_per_sec"],
    )
    
    # Evaluate BERT-Base
    print("\nEvaluating BERT-Base...")
    base_labels, base_preds, base_probas = evaluate_model(base_model, test_loader, device)
    base_metrics = compute_metrics(base_labels, base_preds, base_probas)
    base_size = get_model_size(base_model)
    base_latency = benchmark_latency(base_model, tokenizer, sample_texts, device)
    
    results["bert_base"] = BenchmarkResults(
        model_name="BERT-Base",
        num_parameters=base_size["num_parameters"],
        model_size_mb=base_size["size_mb"],
        accuracy=base_metrics["accuracy"],
        f1_score=base_metrics["f1"],
        precision=base_metrics["precision"],
        recall=base_metrics["recall"],
        auc_roc=base_metrics["auc_roc"],
        avg_latency_ms=base_latency["avg_latency_ms"],
        std_latency_ms=base_latency["std_latency_ms"],
        throughput_samples_per_sec=base_latency["throughput_samples_per_sec"],
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("GIANT-KILLER COMPARISON RESULTS")
    print("=" * 60)
    
    tiny = results["dendritic_bert_tiny"]
    base = results["bert_base"]
    
    print(f"\n{'Metric':<25} {'BERT-Tiny':<15} {'BERT-Base':<15} {'Gap':<15}")
    print("-" * 70)
    print(f"{'Parameters':<25} {tiny.num_parameters:,}  {base.num_parameters:,}  {base.num_parameters/tiny.num_parameters:.1f}x")
    print(f"{'Model Size (MB)':<25} {tiny.model_size_mb:.2f}           {base.model_size_mb:.2f}          {base.model_size_mb/tiny.model_size_mb:.1f}x")
    print(f"{'F1 Score':<25} {tiny.f1_score:.4f}         {base.f1_score:.4f}        {abs(base.f1_score - tiny.f1_score):.4f}")
    print(f"{'Accuracy':<25} {tiny.accuracy:.4f}         {base.accuracy:.4f}        {abs(base.accuracy - tiny.accuracy):.4f}")
    print(f"{'Latency (ms)':<25} {tiny.avg_latency_ms:.2f}           {base.avg_latency_ms:.2f}        {base.avg_latency_ms/tiny.avg_latency_ms:.1f}x faster")
    print(f"{'Throughput (samples/s)':<25} {tiny.throughput_samples_per_sec:.1f}          {base.throughput_samples_per_sec:.1f}")
    
    # Check if Giant-Killer hypothesis is proven
    f1_gap = abs(base.f1_score - tiny.f1_score)
    speed_improvement = base.avg_latency_ms / tiny.avg_latency_ms
    
    print("\n" + "=" * 60)
    if f1_gap < 0.02 and speed_improvement > 15:
        print("🎉 GIANT-KILLER STATUS: CONFIRMED!")
        print(f"   F1 Gap: {f1_gap:.4f} (< 2% threshold)")
        print(f"   Speed: {speed_improvement:.1f}x faster (> 15x threshold)")
    else:
        print("📊 GIANT-KILLER STATUS: IN PROGRESS")
        if f1_gap >= 0.02:
            print(f"   F1 Gap: {f1_gap:.4f} (need < 2%)")
        if speed_improvement <= 15:
            print(f"   Speed: {speed_improvement:.1f}x (need > 15x)")
    print("=" * 60)
    
    return results


def run_full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    tokenizer: Any,
    sample_texts: List[str],
    device: torch.device = torch.device("cpu"),
    model_name: str = "Model",
) -> BenchmarkResults:
    """
    Run full evaluation on a single model.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        tokenizer: Tokenizer for latency benchmarking
        sample_texts: Sample texts for latency tests
        device: Device for evaluation
        model_name: Name for reporting
        
    Returns:
        BenchmarkResults for the model
    """
    print(f"\nEvaluating {model_name}...")
    
    # Evaluate accuracy
    labels, preds, probas = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probas)
    
    # Get model size
    size_info = get_model_size(model)
    
    # Benchmark latency
    latency_info = benchmark_latency(model, tokenizer, sample_texts, device)
    
    results = BenchmarkResults(
        model_name=model_name,
        num_parameters=size_info["num_parameters"],
        model_size_mb=size_info["size_mb"],
        accuracy=metrics["accuracy"],
        f1_score=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        auc_roc=metrics["auc_roc"],
        avg_latency_ms=latency_info["avg_latency_ms"],
        std_latency_ms=latency_info["std_latency_ms"],
        throughput_samples_per_sec=latency_info["throughput_samples_per_sec"],
    )
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Parameters: {results.num_parameters:,}")
    print(f"  Size: {results.model_size_mb:.2f} MB")
    print(f"  Accuracy: {results.accuracy:.4f}")
    print(f"  F1 Score: {results.f1_score:.4f}")
    print(f"  Precision: {results.precision:.4f}")
    print(f"  Recall: {results.recall:.4f}")
    if results.auc_roc:
        print(f"  AUC-ROC: {results.auc_roc:.4f}")
    print(f"  Latency: {results.avg_latency_ms:.2f} ± {results.std_latency_ms:.2f} ms")
    print(f"  Throughput: {results.throughput_samples_per_sec:.1f} samples/sec")
    
    # Print detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Non-Toxic", "Toxic"]))
    
    return results


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing evaluation utilities...")
    
    # Create sample predictions
    y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4],
                        [0.85, 0.15], [0.2, 0.8], [0.95, 0.05], [0.25, 0.75],
                        [0.1, 0.9], [0.7, 0.3]])
    
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print("\nTest Metrics:")
    for name, value in metrics.items():
        if value is not None:
            print(f"  {name}: {value:.4f}")
