"""
Training script for DendriticBERT with Multi-Benchmark Support.
"""

import argparse
import torch
import numpy as np
import evaluate
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from dendritic_bert_core import DendriticBERT, initialize_dendritic_model
from data_loader import load_glue_task, load_squad
from perforatedai import globals_perforatedai as gpa


def parse_args():
    parser = argparse.ArgumentParser(description="Train DendriticBERT")
    parser.add_argument(
        "--model_name", type=str, default="prajjwal1/bert-tiny"
    )
    parser.add_argument(
        "--benchmark", type=str, default="glue", choices=["glue", "squad"]
    )
    parser.add_argument("--task", type=str, default="sst2", help="GLUE task")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dsn", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dendritic-bert")
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, config=args)
    gpa.save_name = f"DendriticBERT_{args.benchmark}_{args.task}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.benchmark == "glue":
        tokenized_datasets = load_glue_task(args.task, tokenizer)
        num_labels = 1 if args.task == "stsb" else 2
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )
        model = DendriticBERT(hf_model, dsn=args.dsn, task_type="classification")
        metric = evaluate.load("glue", args.task)
    else:
        tokenized_datasets = load_squad(tokenizer)
        hf_model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        model = DendriticBERT(hf_model, dsn=args.dsn, task_type="qa")
        metric = evaluate.load("squad")

    model = initialize_dendritic_model(model)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if args.benchmark == "glue":
            predictions = (
                np.argmax(logits, axis=-1)
                if args.task != "stsb"
                else logits.flatten()
            )
            return metric.compute(predictions=predictions, references=labels)
        return {}  # SQuAD metrics are more complex, handled separately

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].select(range(1000)),
        eval_dataset=tokenized_datasets["validation" if args.benchmark == "glue" else "validation"].select(range(100)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if args.benchmark == "glue" else None,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
