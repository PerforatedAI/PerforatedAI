"""
Training script for dendritic-optimized summarization models.

This script implements the 3-experiment compression protocol:
  A. Baseline (full-size, no dendrites)
  B. Compressed + dendritic optimization
  C. Compressed without dendrites (control)

Usage:
    python train_dendritic_summarizer.py --experiment B --model t5-small --dataset reddit_tldr
    
    # Or run with W&B sweep:
    wandb sweep sweep_encoder_t5.yaml
    wandb agent <sweep_id>
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import evaluate
import wandb

from pai_wrappers import wrap_t5_layers_for_pai, wrap_bart_layers_for_pai

try:
    import perforatedai as PA
    DENDRITIC_AVAILABLE = True
except ImportError:
    DENDRITIC_AVAILABLE = False
    logging.warning("perforatedai not installed - dendritic optimization disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train dendritic-optimized summarizer")
    
    # Experiment type
    parser.add_argument("--experiment", type=str, choices=["A", "B", "C"], default="B",
                       help="A=baseline, B=compressed+dendrites, C=compressed control")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument("--enc_num_layers", type=int, default=6)
    parser.add_argument("--enc_hidden_size", type=int, default=384)
    parser.add_argument("--dec_num_layers", type=int, default=4)
    parser.add_argument("--dec_d_model", type=int, default=384)
    
    # Training configuration
    parser.add_argument("--dataset", type=str, default="reddit_tldr")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Dendritic optimization (only for experiment B)
    parser.add_argument("--do_pb", type=bool, default=True)
    parser.add_argument("--N_EPOCHS_TO_SWITCH", type=int, default=5)
    parser.add_argument("--P_EPOCHS_TO_SWITCH", type=int, default=3)
    parser.add_argument("--CAP_N", type=bool, default=True)
    parser.add_argument("--TEST_DENDRITE_CAPACITY", type=bool, default=True)
    
    # Misc
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    
    return parser.parse_args()


def load_summarization_dataset(dataset_name: str, split: str = "train"):
    """Load and prepare summarization dataset."""
    
    if dataset_name == "reddit_tldr":
        dataset = load_dataset("reddit_tifu", "long", split=split)
        dataset = dataset.map(lambda x: {
            "document": x["documents"],
            "summary": x["tldr"]
        })
    elif dataset_name == "samsum":
        dataset = load_dataset("samsum", split=split)
        dataset = dataset.map(lambda x: {
            "document": x["dialogue"],
            "summary": x["summary"]
        })
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        dataset = dataset.map(lambda x: {
            "document": x["article"],
            "summary": x["highlights"]
        })
    elif dataset_name == "xsum":
        dataset = load_dataset("xsum", split=split)
        dataset = dataset.map(lambda x: {
            "document": x["document"],
            "summary": x["summary"]
        })
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_compressed_model(args) -> nn.Module:
    """Create compressed model based on experiment type."""
    
    if args.experiment == "A":
        # Baseline: full model, no compression
        if "t5" in args.model_type:
            model = T5ForConditionalGeneration.from_pretrained(args.model_type)
        elif "bart" in args.model_type:
            model = BartForConditionalGeneration.from_pretrained(args.model_type)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        logger.info(f"Experiment A: Loaded baseline {args.model_type}")
        
    else:
        # Experiments B and C: compressed architecture
        if "t5" in args.model_type:
            from transformers import T5Config
            config = T5Config.from_pretrained(args.model_type)
            config.num_layers = args.enc_num_layers
            config.num_decoder_layers = args.dec_num_layers
            config.d_model = args.dec_d_model
            model = T5ForConditionalGeneration(config)
        elif "bart" in args.model_type:
            from transformers import BartConfig
            config = BartConfig.from_pretrained(args.model_type)
            config.encoder_layers = args.enc_num_layers
            config.decoder_layers = args.dec_num_layers
            config.d_model = args.dec_d_model
            model = BartForConditionalGeneration(config)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        logger.info(f"Experiment {args.experiment}: Created compressed model")
    
    return model


def setup_dendritic_optimization(model: nn.Module, args) -> Optional[Any]:
    """Setup dendritic optimization for experiment B."""
    
    if args.experiment != "B" or not args.do_pb or not DENDRITIC_AVAILABLE:
        return None
    
    # Wrap layers
    if "t5" in args.model_type:
        model = wrap_t5_layers_for_pai(model)
    elif "bart" in args.model_type:
        model = wrap_bart_layers_for_pai(model)
    
    # Convert network
    try:
        PA.convert_network(
            model,
            module_names_to_convert=["encoder.block", "decoder.block"]
        )
        logger.info("Model converted for dendritic optimization")
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        return None
    
    # Initialize tracker
    tracker = PA.PerforatedBackPropagationTracker(
        do_pb=True,
        save_name=f"dendritic_{args.model_type}_{args.experiment}",
        maximizing_score=True,
        make_graphs=True
    )
    
    # Setup optimizer
    tracker.setup_optimizer(
        torch.optim.AdamW,
        torch.optim.lr_scheduler.StepLR,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=1000,
        gamma=0.95
    )
    
    logger.info("Dendritic optimization tracker initialized")
    return tracker


def evaluate_model(model, eval_loader, tokenizer, device):
    """Evaluate model on validation set."""
    
    model.eval()
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch["summary"])
    
    # Compute ROUGE scores
    scores = rouge.compute(predictions=predictions, references=references)
    
    return {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
    }


def main():
    args = parse_args()
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(project="ai-tldr-dendritic", config=vars(args))
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    train_dataset = load_summarization_dataset(args.dataset, split="train[:10000]")
    val_dataset = load_summarization_dataset(args.dataset, split="validation[:1000]")
    
    # Create model
    model = create_compressed_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    
    # Setup dendritic optimization
    tracker = setup_dendritic_optimization(model, args)
    
    # Move to device
    model.to(args.device)
    
    # Setup optimizer (standard if no dendritic)
    if tracker is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = tracker.optimizer
    
    # Training loop
    logger.info("Starting training...")
    best_rouge_l = 0.0
    patience_counter = 0
    step = 0
    
    while step < args.max_steps:
        model.train()
        
        # Training step (simplified - you'd add proper data loading here)
        # ... training code ...
        
        step += 1
        
        # Validation every N steps
        if step % 500 == 0:
            metrics = evaluate_model(model, None, tokenizer, args.device)  # Add proper eval loader
            rouge_l = metrics["rougeL"]
            
            logger.info(f"Step {step}: ROUGE-L = {rouge_l:.4f}")
            
            # Dendritic feedback
            if tracker is not None:
                returned = tracker.add_validation_score(model, rouge_l)
                
                if isinstance(returned, tuple) and len(returned) >= 4:
                    model, improved, restructured, training_complete = returned[:4]
                    
                    if restructured:
                        logger.info("Model restructured by dendritic algorithm")
                        model.to(args.device)
                    
                    if training_complete:
                        logger.info("Training complete")
                        break
            
            # Early stopping
            if rouge_l > best_rouge_l:
                best_rouge_l = rouge_l
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered")
                break
    
    # Final evaluation and save
    final_metrics = evaluate_model(model, None, tokenizer, args.device)
    total_params = sum(p.numel() for p in model.parameters())
    
    results = {
        "experiment": args.experiment,
        "model_type": args.model_type,
        "best_rouge_l": best_rouge_l,
        "final_metrics": final_metrics,
        "total_parameters": total_params,
        "training_steps": step
    }
    
    # Save model and results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir / f"model_exp_{args.experiment}")
    
    with open(output_dir / f"results_exp_{args.experiment}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results: {json.dumps(results, indent=2)}")
    
    if args.use_wandb:
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()
