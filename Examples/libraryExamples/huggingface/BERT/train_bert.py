#!/usr/bin/env python
"""
Baseline BERT Classification Training Script (without PAI integration)

This script trains a Transformer model (RoBERTa or BERT) on either an IMDB
or SNLI dataset. All configuration is provided via command‑line arguments.
It supports a DSN flag (--dsn) which, when enabled, sets the number of encoder
layers to 0 (i.e. uses only embeddings, sum pooling, and classifier).

Usage example for IMDB:
    python train_bert.py --model_name "prajjwal1/bert-tiny" --dataset imdb --dsn \
       --seed 42 --num_epochs 100 --batch_size 32 --max_len 512 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_IMDB --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy \
       --greater_is_better True --scheduler_type reduce_lr_on_plateau --scheduler_patience 2 \
       --scheduler_factor 0.5 --scheduler_min_lr 1e-7
       
Usage example for SNLI:
    python train_bert.py --model_name "prajjwal1/bert-tiny" --dataset snli --dsn \
       --seed 42 --num_epochs 100 --batch_size 256 --max_len 128 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_SNLI --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy \
       --greater_is_better True --scheduler_type reduce_lr_on_plateau --scheduler_patience 2 \
       --scheduler_factor 0.5 --scheduler_min_lr 1e-7
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    modeling_outputs
)
from datasets import load_dataset, Dataset, DatasetDict

from models import RobertaForSequenceClassificationPB, BertForSequenceClassificationPB

# =============================================================================
# Helper Dataset Class
# =============================================================================
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# =============================================================================
# Utility Functions
# =============================================================================
def count_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name} | {num_params:,}")
    print(f"\nTotal Model Parameters: {total_params:,}")
    return total_params

def load_imdb_dataset(tokenizer, max_len, seed=42, reduce_lines=False, cache_dir="./cached_datasets"):
    """
    Downloads and loads the IMDB dataset directly from Hugging Face.
    Creates a stratified split of the training data to get a dev set.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download the IMDB dataset
    dataset = load_dataset("imdb", cache_dir=cache_dir)
    
    # Function to create stratified split
    def stratified_split(dataset, test_size=0.1, seed=42):
        # Get indices for each class
        pos_indices = np.where(np.array(dataset['label']) == 1)[0]
        neg_indices = np.where(np.array(dataset['label']) == 0)[0]
        
        # Calculate split sizes
        n_pos_test = int(len(pos_indices) * test_size)
        n_neg_test = int(len(neg_indices) * test_size)
        
        # Random split for each class
        np.random.seed(seed)
        pos_test_indices = np.random.choice(pos_indices, n_pos_test, replace=False)
        neg_test_indices = np.random.choice(neg_indices, n_neg_test, replace=False)
        
        # Combine test indices
        test_indices = np.concatenate([pos_test_indices, neg_test_indices])
        train_indices = np.array([i for i in range(len(dataset)) if i not in test_indices])
        
        return dataset.select(train_indices), dataset.select(test_indices)
    
    # Split train set into 90% train, 10% dev while maintaining class balance
    train_data, dev_data = stratified_split(dataset["train"], test_size=0.1, seed=seed)
    test_data = dataset["test"]
    
    # Optionally reduce dataset size for faster testing
    if reduce_lines:
        train_data = train_data.select(range(min(1000, len(train_data))))
        dev_data = dev_data.select(range(min(100, len(dev_data))))
        test_data = test_data.select(range(min(100, len(test_data))))
    
    # Print dataset statistics
    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Print class distribution
    train_labels = train_data["label"]
    train_class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    print("\nClass distribution in train set:")
    for label, count in train_class_counts.items():
        print(f"Class {label}: {count} samples")
    
    # Tokenize datasets
    def tokenize_and_convert(dataset):
        texts = dataset["text"]
        labels = dataset["label"]
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        return SentimentDataset(encodings, labels)
    
    train_dataset = tokenize_and_convert(train_data)
    dev_dataset = tokenize_and_convert(dev_data)
    test_dataset = tokenize_and_convert(test_data)
    
    return train_dataset, dev_dataset, test_dataset


def load_snli_dataset(tokenizer, max_len, seed, reduce_lines=False, cache_dir="./cached_datasets"):
    """
    Loads the SNLI dataset from Hugging Face.
    Filters out examples with label -1 and tokenizes using premise and hypothesis.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset("stanfordnlp/snli", cache_dir=cache_dir)

    # Filter out examples with no gold label.
    dataset = dataset.filter(lambda ex: ex["label"] != -1)
    dataset = dataset.shuffle(seed=seed)

    if reduce_lines:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["validation"] = dataset["validation"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))

    def tokenize_fn(examples):
        return tokenizer(examples["premise"], examples["hypothesis"],
                         truncation=True, padding="max_length", max_length=max_len)

    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    train_dataset = SentimentDataset({k: tokenized_datasets["train"][k] for k in tokenizer.model_input_names},
                                     tokenized_datasets["train"]["label"])
    dev_dataset = SentimentDataset({k: tokenized_datasets["validation"][k] for k in tokenizer.model_input_names},
                                   tokenized_datasets["validation"]["label"])
    test_dataset = SentimentDataset({k: tokenized_datasets["test"][k] for k in tokenizer.model_input_names},
                                    tokenized_datasets["test"]["label"])
    return train_dataset, dev_dataset, test_dataset

def resize_model_hidden_size(config, width_factor):
    if width_factor <= 0 or width_factor > 1.0:
        raise ValueError(f"Width factor must be in range (0, 1.0], got {width_factor}")
    if width_factor == 1.0:
        return config
    print(f"Resizing model hidden dimensions by factor {width_factor}")
    if hasattr(config, "hidden_size"):
        config.hidden_size = int(config.hidden_size * width_factor)
    if hasattr(config, "intermediate_size"):
        config.intermediate_size = int(config.intermediate_size * width_factor)
    if hasattr(config, "num_attention_heads"):
        config.num_attention_heads = max(8, int(config.num_attention_heads * width_factor) // 8 * 8)
    return config


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train a baseline Transformer model (without PAI integration).")
    # Model and tokenizer settings
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., roberta-base or bert-base-uncased)")
    # Dataset selection: imdb or snli
    parser.add_argument("--dataset", type=str, choices=["imdb", "snli"], required=True, help="Dataset type: imdb or snli")
    # DSN flag: if enabled, set number of encoder layers to 0.
    parser.add_argument("--dsn", action="store_true", help="Enable DSN mode (set number of encoder layers to 0)")
    # Set compression for model width
    parser.add_argument("--width", type=float, default=1.0, help="Width factor to shrink the model (between 0 and 1)")
    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Hidden dropout probability")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="Attention dropout probability")
    # Saving and early stopping parameters
    parser.add_argument("--model_save_location", type=str, default=None, help="Directory to save the trained model")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Early stopping threshold")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saves")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy (e.g., epoch or steps)")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluation steps (if strategy is steps)")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy (e.g., epoch or steps)")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total number of saved checkpoints")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_accuracy", help="Metric for selecting best model")
    parser.add_argument("--greater_is_better", type=bool, default=True, help="Whether a higher metric is better")
    parser.add_argument("--scheduler_type", type=str, default="reduce_lr_on_plateau", help="Learning rate scheduler type")
    parser.add_argument("--scheduler_patience", type=int, default=2, help="Scheduler patience")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="Scheduler factor")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-7, help="Minimum learning rate")
    # For testing speed
    parser.add_argument("--reduce_lines_for_testing", action="store_true", help="Reduce dataset size for testing")
    args = parser.parse_args()

    # Set random seeds for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set number of labels and load model configuration.
    if args.dataset == "imdb":
        num_labels = 2
    else:  # snli
        num_labels = 3
        
    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)
    
    # Compression for model width
    if args.width < 1.0:
        config = resize_model_hidden_size(config, args.width)

    # Apply dropout settings.
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    
    # If DSN flag is enabled, set the number of encoder layers to 0.
    if args.dsn:
        print("Using Deep Simple Network mode (no encoder layers)")
        config.num_hidden_layers = 0

    # Load pretrained model
    print(f"Loading pretrained model from {args.model_name}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True)
    
    # Wrap the base model for compatibility with PAI
    if "roberta" in args.model_name:
        model = RobertaForSequenceClassificationPB(base_model, dsn=args.dsn, dropout=args.hidden_dropout_prob)
    elif "bert" in args.model_name:
        model = BertForSequenceClassificationPB(base_model, dsn=args.dsn, dropout=args.hidden_dropout_prob)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Move model to device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Print parameter counts.
    count_model_parameters(model)

    # Load dataset.
    if args.dataset == "imdb":
        train_dataset, dev_dataset, test_dataset = load_imdb_dataset(
            tokenizer, args.max_len, seed=args.seed, reduce_lines=args.reduce_lines_for_testing
        )
    else:  # SNLI
        train_dataset, dev_dataset, test_dataset = load_snli_dataset(
            tokenizer, args.max_len, seed=args.seed, reduce_lines=args.reduce_lines_for_testing
        )

    # Prepare training arguments.
    if args.evaluation_strategy == "steps" and args.eval_steps is None:
        args.eval_steps = int(len(train_dataset) / args.batch_size)
    training_args = TrainingArguments(
        output_dir=args.model_save_location if args.model_save_location else "./results",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=int(1000 / args.batch_size) if not args.reduce_lines_for_testing else 1,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=args.early_stopping,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        lr_scheduler_type=args.scheduler_type,
        lr_scheduler_kwargs={"patience": args.scheduler_patience,
                             "factor": args.scheduler_factor,
                             "min_lr": args.scheduler_min_lr} if args.scheduler_type == "reduce_lr_on_plateau" else None,
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold))
        print("Early stopping enabled.")

    # Create Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=lambda pred: {"eval_accuracy": float((pred.predictions.argmax(-1) == pred.label_ids).mean())},
        callbacks=callbacks
    )

    # Start training.
    print("Starting training...")
    trainer.train()

    # Evaluate on test set.
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = test_results.predictions.argmax(-1)
    test_labels = test_results.label_ids
    test_accuracy = (test_preds == test_labels).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save final model and tokenizer.
    if args.model_save_location:
        print(f"Saving model to {args.model_save_location}...")
        os.makedirs(args.model_save_location, exist_ok=True)
        trainer.save_model(args.model_save_location)
        tokenizer.save_pretrained(args.model_save_location)

if __name__ == "__main__":
    main()
