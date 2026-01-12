#!/usr/bin/env python3
"""
Main experiment runner for PerforatedAI training.

Usage:
    # Run single experiment
    python -m ml.experiments.run_experiment --experiment B --model t5-small
    
    # Run all three experiments
    python -m ml.experiments.run_experiment --experiment all --model t5-small
    
    # Run with W&B sweep
    wandb agent <sweep_id>
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.config.pai_config import PAIConfig, ExperimentType, configure_pai_globals
from ml.models.pai_converter import convert_model_for_pai
from ml.trainers.pai_trainer import PAIDendriticTrainer
from ml.data.data_loader import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PerforatedAI training experiment")
    
    # Experiment selection
    parser.add_argument(
        "--experiment", "-e",
        choices=["A", "B", "C", "all"],
        default="B",
        help="Experiment type: A=baseline, B=compressed+dendrites, C=compressed control"
    )
    
    # Model configuration
    parser.add_argument("--model", "-m", default="t5-small", help="Model name or path")
    parser.add_argument("--dataset", "-d", default="samsum", help="Dataset name")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=50000)
    
    # PAI parameters
    parser.add_argument("--n-epochs-to-switch", type=int, default=5)
    parser.add_argument("--p-epochs-to-switch", type=int, default=3)
    parser.add_argument("--dendrite-capacity", type=int, default=8)
    
    # Output
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--wandb-project", default="ai-tldr-dendritic")
    
    return parser.parse_args()


def run_single_experiment(
    experiment_type: ExperimentType,
    args: argparse.Namespace,
) -> dict:
    """Run a single experiment."""
    
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT {experiment_type.value}: {experiment_type.name}")
    logger.info("=" * 70)
    
    # 1. Create PAI configuration
    config = PAIConfig(
        experiment_type=experiment_type,
        experiment_name=f"summarizer_{args.model.replace('/', '-')}",
        n_epochs_to_switch=args.n_epochs_to_switch,
        p_epochs_to_switch=args.p_epochs_to_switch,
        dendrite_capacity=args.dendrite_capacity,
        maximizing_score=True,
    )
    
    # 2. Configure PAI globals (MUST be done before model conversion)
    configure_pai_globals(config)
    
    # 3. Load tokenizer and model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    # 4. Convert model for PAI (only for experiment B)
    if experiment_type == ExperimentType.COMPRESSED_DENDRITES:
        model = convert_model_for_pai(model, model_type="t5")
    
    # 5. Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset,
        tokenizer,
        batch_size=args.batch_size,
    )
    
    # 6. Create trainer and run
    trainer = PAIDendriticTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
    )
    
    results = trainer.train(max_steps=args.max_steps)
    
    return results


def main():
    args = parse_args()
    
    # Determine which experiments to run
    if args.experiment == "all":
        experiments = [
            ExperimentType.BASELINE,
            ExperimentType.COMPRESSED_DENDRITES,
            ExperimentType.COMPRESSED_CONTROL,
        ]
    else:
        exp_map = {
            "A": ExperimentType.BASELINE,
            "B": ExperimentType.COMPRESSED_DENDRITES,
            "C": ExperimentType.COMPRESSED_CONTROL,
        }
        experiments = [exp_map[args.experiment]]
    
    # Run experiments
    all_results = {}
    for exp_type in experiments:
        results = run_single_experiment(exp_type, args)
        all_results[exp_type.value] = results
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    
    for exp_id, results in all_results.items():
        metrics = results.get("final_metrics", {})
        logger.info(f"\nExperiment {exp_id}:")
        logger.info(f"  ROUGE-L:    {metrics.get('rougeL', 0):.4f}")
        logger.info(f"  Parameters: {metrics.get('total_parameters', 0):,}")
        logger.info(f"  Latency:    {metrics.get('inference_latency_ms', 0):.1f}ms")
    
    return all_results


if __name__ == "__main__":
    main()
