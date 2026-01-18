"""
Evaluation Scripts for PAI Experiments

Computes comprehensive metrics:
- ROUGE-1/2/L and BERTScore for summarization
- HighlightPrecision@K for application-specific evaluation
- Latency and memory footprints
- Parameter counts and dendrite statistics

Usage:
    from ml.eval import evaluate_checkpoint
    
    results = evaluate_checkpoint("path/to/checkpoint.pt", "validation")
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""
    
    # Summarization metrics
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_f1: float = 0.0
    
    # Highlight metrics
    highlight_precision_at_1: float = 0.0
    highlight_precision_at_5: float = 0.0
    highlight_mrr: float = 0.0
    
    # Efficiency metrics
    latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    memory_mb: float = 0.0
    
    # Model statistics
    total_parameters: int = 0
    active_parameters: int = 0
    active_dendrite_count: int = 0
    parameter_reduction_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_split: str = "validation",
    output_dir: Optional[str] = None,
    dataset_name: str = "samsum",
) -> EvaluationResults:
    """
    Evaluate a saved checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset_split: Data split to evaluate on
        output_dir: Optional output directory for results
        dataset_name: Dataset to use
        
    Returns:
        EvaluationResults with all metrics
    """
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = EvaluationResults()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # Load model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = config.get("model_name", "t5-small")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    from ml.data import load_dataset, preprocess_for_summarization
    examples = preprocess_for_summarization(
        load_dataset(dataset_name, dataset_split, max_samples=100)
    )
    
    # Compute model statistics
    results.total_parameters = sum(p.numel() for p in model.parameters())
    results.active_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute ROUGE scores
    predictions = []
    references = []
    latencies = []
    
    with torch.no_grad():
        for ex in examples[:50]:  # Limit for speed
            inputs = tokenizer(
                ex.source_text,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            # Measure latency
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
            )
            latencies.append((time.time() - start) * 1000)
            
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred)
            references.append(ex.target_text)
    
    # ROUGE
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=predictions, references=references)
        results.rouge1 = scores["rouge1"]
        results.rouge2 = scores["rouge2"]
        results.rougeL = scores["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
    
    # BERTScore
    results.bertscore_f1 = compute_bertscore(references, predictions)
    
    # Latency
    results.latency_ms = sum(latencies) / max(len(latencies), 1)
    
    # Memory
    if torch.cuda.is_available():
        results.memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    logger.info(f"Results: ROUGE-L={results.rougeL:.4f}, Latency={results.latency_ms:.1f}ms")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir) / "eval_results.json"
        results.save(output_path)
        logger.info(f"Results saved to {output_path}")
    
    return results


def compare_experiments(
    baseline_path: str,
    compressed_pai_path: str,
    compressed_control_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare results from all three experiment types.
    
    Args:
        baseline_path: Path to baseline results
        compressed_pai_path: Path to compressed+PAI results
        compressed_control_path: Path to compressed control results
        output_path: Optional path for comparison table
        
    Returns:
        Comparison dictionary
    """
    def load_results(path):
        with open(path) as f:
            return json.load(f)
    
    baseline = load_results(baseline_path)
    pai = load_results(compressed_pai_path)
    control = load_results(compressed_control_path)
    
    comparison = {
        "baseline": baseline,
        "compressed_pai": pai,
        "compressed_control": control,
        "analysis": {
            "rouge_improvement_vs_control": pai["rougeL"] - control["rougeL"],
            "param_reduction_vs_baseline": (baseline["total_parameters"] - pai["total_parameters"]) / baseline["total_parameters"] * 100,
            "latency_reduction_vs_baseline": (baseline["latency_ms"] - pai["latency_ms"]) / baseline["latency_ms"] * 100,
        }
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
    
    return comparison


def compute_rouge(reference_list: List[str], hypothesis_list: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        reference_list: List of reference summaries
        hypothesis_list: List of generated summaries
        
    Returns:
        Dictionary with rouge1, rouge2, rougeL scores
    """
    try:
        from rouge_score import rouge_scorer
        import numpy as np
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for ref, hyp in zip(reference_list, hypothesis_list):
            result = scorer.score(ref, hyp)
            scores['rouge1'].append(result['rouge1'].fmeasure)
            scores['rouge2'].append(result['rouge2'].fmeasure)
            scores['rougeL'].append(result['rougeL'].fmeasure)
        
        return {
            'rouge1': float(np.mean(scores['rouge1'])),
            'rouge2': float(np.mean(scores['rouge2'])),
            'rougeL': float(np.mean(scores['rougeL'])),
        }
    except ImportError:
        logger.warning("rouge_score not installed")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def compute_bertscore(
    reference_list: List[str],
    hypothesis_list: List[str],
    model_type: str = 'microsoft/deberta-xlarge-mnli',
) -> float:
    """
    Compute BERTScore F1.
    
    Args:
        reference_list: List of reference summaries
        hypothesis_list: List of generated summaries
        model_type: Model to use for embeddings
        
    Returns:
        Mean BERTScore F1
    """
    try:
        from bert_score import score as bertscore
        import numpy as np
        
        P, R, F = bertscore(
            hypothesis_list,
            reference_list,
            lang='en',
            model_type=model_type,
            verbose=False,
        )
        return float(np.mean(F.numpy()))
    except ImportError:
        logger.warning("bert-score not installed")
        return 0.0
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return 0.0


def highlight_precision_at_k(
    retrieved: List[List[Tuple[str, int]]],
    ground_truth: List[set],
    k: int = 3,
) -> float:
    """
    Compute Highlight Precision@K.
    
    Measures how well retrieved highlights match ground-truth annotations.
    
    Args:
        retrieved: List of lists of (post_id, sentence_idx) tuples
        ground_truth: List of sets of ground-truth (post_id, sentence_idx) tuples
        k: Number of top results to consider
        
    Returns:
        Mean precision@k
    """
    import numpy as np
    
    precisions = []
    for r, g in zip(retrieved, ground_truth):
        topk = set(r[:k])
        if len(topk) == 0:
            precisions.append(0.0)
        else:
            precisions.append(len(topk.intersection(g)) / k)
    
    return float(np.mean(precisions)) if precisions else 0.0


def highlight_mrr(
    retrieved: List[List[Tuple[str, int]]],
    ground_truth: List[set],
) -> float:
    """
    Compute Mean Reciprocal Rank for highlights.
    
    Args:
        retrieved: List of lists of (post_id, sentence_idx) tuples
        ground_truth: List of sets of ground-truth tuples
        
    Returns:
        Mean Reciprocal Rank
    """
    import numpy as np
    
    reciprocal_ranks = []
    for r, g in zip(retrieved, ground_truth):
        rank = 0
        for i, item in enumerate(r, 1):
            if item in g:
                rank = 1.0 / i
                break
        reciprocal_ranks.append(rank)
    
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
