#!/usr/bin/env python
"""
BERT Classification Training Script with PAI integration

This script trains a Transformer model (RoBERTa or BERT) on IMDB, SNLI,
or AG News using PAI conversion. It uses command‐line arguments for all
configuration parameters.

Features:
  • Supports IMDB, SNLI, and AG News datasets.
  • Includes a flag (--dsn) that, when enabled, sets the number of encoder layers to 0.
  • Uses Perforated Backpropagation implemented through the PerforatedAI library.

Setup for PAI:
    git clone https://github.com/PerforatedAI/PerforatedAI-Transformers.git
    cd PerforatedAI-Transformers
    pip install -e .
    pip install perforatedai evaluate scikit-learn

    export PAIEMAIL=<your_email>
    export PAITOKEN=<your_pai_token>
    export CUDA_VISIBLE_DEVICES=<your_gpu_id>  <-- set this if you have multiple GPUs

Usage example for IMDB:
    python train_bert_pai.py --model_name "prajjwal1/bert-tiny" --dataset imdb --dsn \
       --pai_save_name my_pai_run --switch_mode DOING_HISTORY \
       --n_epochs_to_switch 10 --p_epochs_to_switch 10 --history_lookback 1 \
       --max_dendrites 5 --improvement_threshold 0.0001 \
       --pb_improvement_threshold 0.01 --pb_improvement_threshold_raw 0.001 \
       --unwrapped_modules_confirmed True \
       --seed 42 --num_epochs 100 --batch_size 32 --max_len 512 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_PAI_IMDB --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy --maximizing_score True \
       --greater_is_better True --scheduler_type linear

Usage example for SNLI:
    python train_bert_pai.py --model_name "prajjwal1/bert-tiny" --dataset snli --dsn \
       --pai_save_name my_pai_run --switch_mode DOING_HISTORY \
       --n_epochs_to_switch 10 --p_epochs_to_switch 10 --history_lookback 1 \
       --max_dendrites 5 --improvement_threshold 0.0001 \
       --pb_improvement_threshold 0.01 --pb_improvement_threshold_raw 0.001 \
       --unwrapped_modules_confirmed True \
       --seed 42 --num_epochs 100 --batch_size 256 --max_len 128 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_PAI_SNLI --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy --maximizing_score True \
       --greater_is_better True --scheduler_type linear

Usage example for AG News:
    python train_bert_pai.py --model_name "prajjwal1/bert-tiny" --dataset agnews --dsn \
       --pai_save_name my_pai_run --switch_mode DOING_HISTORY \
       --n_epochs_to_switch 10 --p_epochs_to_switch 10 --history_lookback 1 \
       --max_dendrites 5 --improvement_threshold 0.0001 \
       --pb_improvement_threshold 0.01 --pb_improvement_threshold_raw 0.001 \
       --unwrapped_modules_confirmed True \
       --seed 0 --num_epochs 10 --batch_size 32 --max_len 512 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_PAI_AGNEWS --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy --maximizing_score True \
       --greater_is_better True --scheduler_type linear
"""

import os
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
    BertForSequenceClassification,
    RobertaForSequenceClassification,
)

# Import PAI modules
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# For datasets
from datasets import load_dataset

from models import (
    BertForSequenceClassificationPB,
    RobertaForSequenceClassificationPB,
    ClassifierWrapper,
)


# =============================================================================
# Helper Dataset Class (used for SNLI)
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
    regular_params = 0
    pb_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        if "parentModule" in name or "PBtoTop" in name:
            pb_params += num_params
            print(f"PB: {name} | {num_params:,}")
        elif "pb.layers" in name:
            pass
        else:
            regular_params += num_params
            print(f"Regular: {name} | {num_params:,}")

    total_params = regular_params + pb_params

    print(f"\nRegular Model Parameters: {regular_params:,}")
    print(f"PB Parameters: {pb_params:,}")
    print(f"Total Parameters: {total_params:,}\n")

    return regular_params, pb_params, total_params


def load_agnews_dataset(tokenizer, max_len, seed=42):
    """Load AG News with HF Datasets; tokenized and Trainer-ready."""
    dataset = load_dataset("ag_news")
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=seed)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=max_len
        )

    train_dataset = train_val["train"].map(tokenize_function, batched=True)
    val_dataset = train_val["test"].map(tokenize_function, batched=True)
    test_dataset = dataset["test"].map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, val_dataset, test_dataset


def load_imdb_dataset(
    tokenizer, max_len, seed=42, reduce_lines=False, cache_dir="./cached_datasets"
):
    """
    Downloads and loads the IMDB dataset directly from Hugging Face.
    Creates a stratified split of the training data to get a dev set.
    Returns HF Datasets formatted for Trainer (labels column + torch format).
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset("imdb", cache_dir=cache_dir)

    def stratified_split(dataset_split, test_size=0.1, seed=42):
        labels = np.array(dataset_split["label"])
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        n_pos_test = int(len(pos_idx) * test_size)
        n_neg_test = int(len(neg_idx) * test_size)

        rng = np.random.default_rng(seed)
        pos_test = rng.choice(pos_idx, n_pos_test, replace=False)
        neg_test = rng.choice(neg_idx, n_neg_test, replace=False)

        test_indices = np.concatenate([pos_test, neg_test])
        mask = np.ones(len(dataset_split), dtype=bool)
        mask[test_indices] = False
        train_indices = np.where(mask)[0]

        return dataset_split.select(train_indices), dataset_split.select(test_indices)

    train_data, dev_data = stratified_split(dataset["train"], test_size=0.1, seed=seed)
    test_data = dataset["test"]

    if reduce_lines:
        train_data = train_data.select(range(min(1000, len(train_data))))
        dev_data = dev_data.select(range(min(100, len(dev_data))))
        test_data = test_data.select(range(min(100, len(test_data))))

    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples:   {len(dev_data)}")
    print(f"Test samples:  {len(test_data)}")

    # Tokenize and format
    def tok_map(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_len
        )

    train_dataset = train_data.map(tok_map, batched=True)
    dev_dataset = dev_data.map(tok_map, batched=True)
    test_dataset = test_data.map(tok_map, batched=True)

    # rename 'label' -> 'labels' and set torch format (assign back!)
    train_dataset = train_dataset.rename_column("label", "labels")
    dev_dataset = dev_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, dev_dataset, test_dataset


def load_snli_dataset(
    tokenizer, max_len, seed, reduce_lines=False, cache_dir="./cached_datasets"
):
    """
    Loads the SNLI dataset from Hugging Face.
    Filters out examples with label -1 and tokenizes using premise and hypothesis.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset("stanfordnlp/snli", cache_dir=cache_dir)

    dataset = dataset.filter(lambda ex: ex["label"] != -1)
    dataset = dataset.shuffle(seed=seed)

    if reduce_lines:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["validation"] = dataset["validation"].select(range(100))
        dataset["test"] = dataset["test"].select(range(100))

    def tokenize_fn(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    train_dataset = SentimentDataset(
        {k: tokenized["train"][k] for k in tokenizer.model_input_names},
        tokenized["train"]["label"],
    )
    dev_dataset = SentimentDataset(
        {k: tokenized["validation"][k] for k in tokenizer.model_input_names},
        tokenized["validation"]["label"],
    )
    test_dataset = SentimentDataset(
        {k: tokenized["test"][k] for k in tokenizer.model_input_names},
        tokenized["test"]["label"],
    )
    return train_dataset, dev_dataset, test_dataset


def set_GPA_params(args):
    """
    Set PAI global parameters based on *actual* methods present in PAIConfig.
    """
    GPA.metric = args.metric_for_best_model
    pc = GPA.pc

    pc.set_save_name(args.pai_save_name)

    # Normalize and set switch_mode
    sm = args.switch_mode.upper()
    if sm == "DOING_HISTORY":
        pc.set_switch_mode(pc.DOING_HISTORY)
    elif sm == "DOING_FIXED_SWITCH":
        pc.set_switch_mode(pc.DOING_FIXED_SWITCH)  # constant, not a function
        pc.set_fixed_switch_num(args.fixed_switch_num)
        pc.set_first_fixed_switch_num(args.first_fixed_switch_num)
    else:
        raise ValueError(f"Invalid switch_mode: {args.switch_mode}")

    pc.set_n_epochs_to_switch(args.n_epochs_to_switch)
    pc.set_input_dimensions([-1, -1, 0])
    pc.set_history_lookback(args.history_lookback)
    pc.set_max_dendrites(args.max_dendrites)
    pc.set_testing_dendrite_capacity(False)

    # Thresholds available in this config
    pc.set_improvement_threshold(args.improvement_threshold)
    pc.set_improvement_threshold_raw(args.pb_improvement_threshold_raw)

    pc.set_unwrapped_modules_confirmed(args.unwrapped_modules_confirmed)
    pc.set_weight_decay_accepted(True)

    # Debug level
    pc.set_debugging_input_dimensions(1)


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
        config.num_attention_heads = max(
            8, (int(config.num_attention_heads * width_factor) // 8) * 8
        )
    return config


# =============================================================================
# Main Training Function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer model with PAI integration."
    )
    # Model and tokenizer settings
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., roberta-base or bert-base-uncased)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imdb", "snli", "agnews"],
        required=True,
        help="Dataset type",
    )
    # PAI parameters
    parser.add_argument(
        "--wrap_full_model",
        action="store_true",
        help="If True wrap encoder + classifier; else only wrap classifier",
    )
    parser.add_argument(
        "--maximizing_score",
        type=bool,
        default=True,
        help="If True then maximize score for PAI",
    )
    parser.add_argument(
        "--pai_save_name", type=str, default="default_pai", help="PAI save name"
    )
    parser.add_argument(
        "--switch_mode",
        type=str,
        choices=["DOING_HISTORY", "DOING_FIXED_SWITCH"],
        default="DOING_HISTORY",
        help="PAI switch mode",
    )
    parser.add_argument(
        "--history_lookback", type=int, default=1, help="History lookback for PAI"
    )
    parser.add_argument(
        "--fixed_switch_num",
        type=int,
        default=1,
        help="Fixed switch number (if using DOING_FIXED_SWITCH)",
    )
    parser.add_argument(
        "--first_fixed_switch_num",
        type=int,
        default=1,
        help="First fixed switch number (if using DOING_FIXED_SWITCH)",
    )
    parser.add_argument(
        "--n_epochs_to_switch", type=int, default=10, help="Number of epochs to switch"
    )
    parser.add_argument(
        "--p_epochs_to_switch",
        type=int,
        default=10,
        help="(not used by this PAIConfig)",
    )
    parser.add_argument(
        "--max_dendrites", type=int, default=1, help="Maximum dendrites"
    )
    parser.add_argument(
        "--improvement_threshold",
        type=float,
        default=0.0001,
        help="Relative improvement threshold",
    )
    parser.add_argument(
        "--pb_improvement_threshold",
        type=float,
        default=0.01,
        help="(not used by this PAIConfig)",
    )
    parser.add_argument(
        "--pb_improvement_threshold_raw",
        type=float,
        default=0.001,
        help="Absolute improvement threshold (used)",
    )
    parser.add_argument(
        "--unwrapped_modules_confirmed",
        type=bool,
        default=True,
        help="Unwrapped modules confirmed",
    )
    # DSN flag
    parser.add_argument(
        "--dsn",
        action="store_true",
        help="Enable DSN mode (set number of encoder layers to 0)",
    )
    # Compression for model width
    parser.add_argument(
        "--width",
        type=float,
        default=1.0,
        help="Width factor to shrink the model (0 < width <= 1)",
    )
    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="Hidden dropout probability",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="Attention dropout probability",
    )
    # Saving and early stopping parameters
    parser.add_argument(
        "--model_save_location",
        type=str,
        default=None,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--early_stopping", action="store_true", help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=6, help="Early stopping patience"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Early stopping threshold",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Number of steps between saves"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        help="Evaluation strategy (epoch or steps)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluation steps (if strategy is steps)",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Save strategy (epoch or steps)",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit total number of saved checkpoints",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_accuracy",
        help="Metric for selecting best model",
    )
    parser.add_argument(
        "--greater_is_better",
        type=bool,
        default=True,
        help="Whether a higher metric is better",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (e.g., linear, cosine, reduce_on_plateau)",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=2,
        help="(May be ignored by your HF version)",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.5,
        help="(May be ignored by your HF version)",
    )
    parser.add_argument(
        "--scheduler_min_lr",
        type=float,
        default=1e-7,
        help="(May be ignored by your HF version)",
    )
    # For testing speed
    parser.add_argument(
        "--reduce_lines_for_testing",
        action="store_true",
        help="Reduce dataset size for quick testing",
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set PAI parameters.
    set_GPA_params(args)

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Number of labels by dataset
    if args.dataset == "imdb":
        num_labels = 2
    elif args.dataset == "snli":
        num_labels = 3
    else:  # agnews
        num_labels = 4

    config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels)

    # Compression for model width
    if args.width < 1.0:
        config = resize_model_hidden_size(config, args.width)

    # Dropout settings
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = args.hidden_dropout_prob
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob

    # DSN: set number of encoder layers to 0 (note: some heads may require >=1)
    if args.dsn and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 0

    # Load pretrained model
    print(f"Loading pretrained model from {args.model_name}...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True
    )

    # Wrap the base model for compatibility with PAI
    if "roberta" in args.model_name.lower():
        GPA.pc.append_modules_to_replace([RobertaForSequenceClassification])
        GPA.pc.append_replacement_modules([RobertaForSequenceClassificationPB])
        GPA.pc.append_module_names_to_track(["RobertaEmbeddings"])
        GPA.pc.append_module_names_to_track(["Embedding"])

        if args.wrap_full_model:
            print("Wrapping full model")
            GPA.pc.append_module_names_to_convert(["RobertaLayer"])
            GPA.pc.append_module_names_with_processing(["RobertaLayer"])
            # No PBM processing class available in this build; skipping
        else:
            print("Only adding dendrites to classifier")
            GPA.pc.set_modules_to_convert([])
            GPA.pc.append_module_names_to_track(["RobertaLayer", "RobertaPooler"])

            if hasattr(base_model, "classifier"):
                base_model.classifier = ClassifierWrapper(base_model.classifier)
                GPA.pc.append_module_names_to_convert(["ClassifierWrapper"])

        model = RobertaForSequenceClassificationPB(
            base_model, dsn=args.dsn, dropout=args.hidden_dropout_prob
        )

    elif "bert" in args.model_name.lower():
        GPA.pc.append_modules_to_replace([BertForSequenceClassification])
        GPA.pc.append_replacement_modules([BertForSequenceClassificationPB])
        GPA.pc.append_module_names_to_track(["Embeddings"])
        GPA.pc.append_module_names_to_track(["Embedding"])

        if args.wrap_full_model:
            print("Wrapping full model")
            GPA.pc.append_module_names_to_convert(["TransformerBlock"])
            GPA.pc.append_module_names_with_processing(["TransformerBlock"])
            # No PBM processing class available in this build; skipping
        else:
            print("Only adding dendrites to classifier")
            GPA.pc.set_modules_to_convert([])
            GPA.pc.append_module_names_to_track(["TransformerBlock"])

            if hasattr(base_model, "classifier"):
                base_model.classifier = ClassifierWrapper(base_model.classifier)
                GPA.pc.append_module_names_to_convert(["ClassifierWrapper"])

            if hasattr(base_model, "pre_classifier"):
                base_model.pre_classifier = ClassifierWrapper(base_model.pre_classifier)
                GPA.pc.append_module_names_to_convert(["ClassifierWrapper"])

        model = BertForSequenceClassificationPB(
            base_model, dsn=args.dsn, dropout=args.hidden_dropout_prob
        )

    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Initialize PAI tracking for the model
    UPA.initialize_pai(
        model,
        save_name=GPA.pc.get_save_name(),
        maximizing_score=args.maximizing_score,
    )

    # Set input dimensions for specific layers
    for layer_name, layer in dict(model.named_modules()).items():
        try:
            if "roberta" in args.model_name.lower():
                if layer_name in [
                    "roberta.pooler",
                    "roberta.pooler.dense",
                    "classifier",
                    "classifier.dense",
                    "classifier.out_proj",
                ]:
                    layer.set_this_input_dimensions([-1, 0])
            elif "bert" in args.model_name.lower():
                if layer_name in ["bert.pooler", "bert.pooler.dense", "classifier"]:
                    layer.set_this_input_dimensions([-1, 0])
        except Exception as e:
            print(f"Could not set input dimensions for {layer_name}: {e}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Params summary
    count_model_parameters(model)

    # Load dataset
    if args.dataset == "imdb":
        train_dataset, dev_dataset, test_dataset = load_imdb_dataset(
            tokenizer,
            args.max_len,
            seed=args.seed,
            reduce_lines=args.reduce_lines_for_testing,
        )
    elif args.dataset == "snli":
        train_dataset, dev_dataset, test_dataset = load_snli_dataset(
            tokenizer,
            args.max_len,
            seed=args.seed,
            reduce_lines=args.reduce_lines_for_testing,
        )
    else:  # agnews
        train_dataset, dev_dataset, test_dataset = load_agnews_dataset(
            tokenizer, args.max_len, seed=args.seed
        )

    # TrainingArguments
    if args.evaluation_strategy == "steps" and args.eval_steps is None:
        args.eval_steps = max(1, int(len(train_dataset) / args.batch_size))

    training_args = TrainingArguments(
        output_dir=(
            args.model_save_location if args.model_save_location else "./results"
        ),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=(
            int(1000 / args.batch_size) if not args.reduce_lines_for_testing else 1
        ),
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=args.early_stopping,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        save_total_limit=args.save_total_limit,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,  # correct kw
        eval_steps=args.eval_steps,
        learning_rate=args.lr,
        seed=args.seed,
        lr_scheduler_type=args.scheduler_type,  # keep to strings like "linear"
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
        print("Early stopping enabled.")

    # Trainer
    def _compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {"eval_accuracy": float((preds == pred.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Test
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = test_results.predictions.argmax(-1)
    test_labels = test_results.label_ids
    test_accuracy = (test_preds == test_labels).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model + tokenizer
    if args.model_save_location:
        print(f"Saving model to {args.model_save_location}...")
        os.makedirs(args.model_save_location, exist_ok=True)
        trainer.save_model(args.model_save_location)
        tokenizer.save_pretrained(args.model_save_location)


if __name__ == "__main__":
    main()
