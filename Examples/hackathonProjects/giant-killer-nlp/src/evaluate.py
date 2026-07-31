"""
Giant-Killer NLP: Evaluation and Benchmarking Script

This script evaluates the trained Dendritic BERT-Tiny model and compares
it against BERT-Base to prove the "Giant-Killer" hypothesis.

Success criteria:
- F1 Score gap: < 2%
- Speed improvement: > 15x on CPU

Usage:
    python src/evaluate.py
    python src/evaluate.py --model-path checkpoints/best_model.pt
    python src/evaluate.py --quantize  # Evaluate quantized model
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_jigsaw_dataset, create_dataloaders, get_tokenizer
from models import create_bert_tiny_model, create_bert_base_model
from models.bert_tiny import quantize_model, wrap_with_dendrites
from evaluation import (
    run_full_evaluation,
    compare_models,
    benchmark_latency,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(model_path: str, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    # Load checkpoint to check if it has dendritic weights
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Check if model was trained with dendrites
    has_dendrites = any("dendrite_module" in key or "main_module" in key for key in state_dict.keys())
    
    # Create base model
    model = create_bert_tiny_model(
        num_labels=config["model"]["num_labels"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
    )
    
    # Wrap with dendrites if needed
    if has_dendrites:
        print("   Model was trained with dendrites, wrapping...")
        model = wrap_with_dendrites(model)
    
    # Load state dict (strict=False to ignore shape metadata from dendrites)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Giant-Killer NLP Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Compare against BERT-Base",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Evaluate quantized model",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run latency benchmarks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for evaluation (cpu recommended for edge benchmarks)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    device = torch.device(args.device)
    
    print("=" * 60)
    print("GIANT-KILLER NLP EVALUATION")
    print("Benchmarking Dendritic Optimization")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Model path: {args.model_path}")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer(config["model"]["name"])
    
    # Load dataset
    print("\n2. Loading dataset...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_jigsaw_dataset(sample_size=args.sample_size)
    
    # Create test dataloader
    _, _, test_loader, _ = create_dataloaders(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        tokenizer=tokenizer,
        batch_size=config["data"]["batch_size"],
        max_length=config["data"]["max_length"],
        num_workers=0,
    )
    
    # Load or create model
    print("\n3. Loading model...")
    if os.path.exists(args.model_path):
        print(f"   Loading trained model from {args.model_path}")
        model = load_trained_model(args.model_path, config, device)
    else:
        print("   No trained model found. Creating new BERT-Tiny model...")
        model = create_bert_tiny_model(
            num_labels=config["model"]["num_labels"],
        )
        model = model.to(device)
        model.eval()
    
    # Apply quantization if requested
    if args.quantize:
        print("\n4. Applying quantization...")
        model = quantize_model(model)
    
    # Run benchmarks
    if args.benchmark_only:
        print("\n5. Running latency benchmarks...")
        latency_results = benchmark_latency(
            model=model,
            tokenizer=tokenizer,
            texts=test_texts,
            device=device,
            num_runs=config["evaluation"]["benchmark"]["num_samples"],
            warmup_runs=config["evaluation"]["benchmark"]["warm_up_runs"],
        )
        
        print("\nLatency Results:")
        for key, value in latency_results.items():
            print(f"   {key}: {value:.4f}")
        return
    
    # Run full evaluation
    if args.compare_base:
        print("\n5. Running comparison against BERT-Base...")
        
        # Create BERT-Base model
        print("   Loading BERT-Base (this may take a while)...")
        base_model = create_bert_base_model(num_labels=config["model"]["num_labels"])
        base_model = base_model.to(device)
        base_model.eval()
        
        # Compare models
        results = compare_models(
            tiny_model=model,
            base_model=base_model,
            test_loader=test_loader,
            tokenizer=tokenizer,
            sample_texts=test_texts,
            device=device,
        )
    else:
        print("\n5. Running evaluation...")
        results = run_full_evaluation(
            model=model,
            test_loader=test_loader,
            tokenizer=tokenizer,
            sample_texts=test_texts,
            device=device,
            model_name="Dendritic BERT-Tiny" if not args.quantize else "Quantized BERT-Tiny",
        )
    
    # Save results
    print("\n6. Saving results...")
    results_path = os.path.join(config["logging"]["log_dir"], "evaluation_results.txt")
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    
    with open(results_path, "w") as f:
        f.write("Giant-Killer NLP Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        if hasattr(results, "model_name"):
            f.write(f"Model: {results.model_name}\n")
            f.write(f"Parameters: {results.num_parameters:,}\n")
            f.write(f"Size: {results.model_size_mb:.2f} MB\n")
            f.write(f"Accuracy: {results.accuracy:.4f}\n")
            f.write(f"F1 Score: {results.f1_score:.4f}\n")
            f.write(f"Latency: {results.avg_latency_ms:.2f} ms\n")
        else:
            for name, result in results.items():
                f.write(f"\n{result.model_name}:\n")
                f.write(f"  Parameters: {result.num_parameters:,}\n")
                f.write(f"  F1 Score: {result.f1_score:.4f}\n")
                f.write(f"  Latency: {result.avg_latency_ms:.2f} ms\n")
    
    print(f"   Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
