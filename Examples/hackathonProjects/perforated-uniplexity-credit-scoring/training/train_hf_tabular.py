"""
Training script using HuggingFace Trainer
🤗 Leverage HuggingFace ecosystem for training
"""
import sys
sys.path.append('..')

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from models.hf_tabular_model import HFTabularModel, CreditScoringConfig


def train_with_hf():
    """Train model using HuggingFace Trainer"""
    
    # Prepare data
    # Prepare data
    try:
        train_dataset = Dataset.from_json('../data/processed/hf_dataset/train.json')
        eval_dataset = Dataset.from_json('../data/processed/hf_dataset/test.json')
    except Exception:
        print("❌ HF Datasets not found! Run data/preprocess.py first.")
        return

    # Convert to tensors (HuggingFace Trainer usually handles this, but custom models need strict formats)
    # The Generic HFTabularModel expects 'input_features' and 'labels'
    
    # We rename 'features' to 'input_features' if needed, or rely on colator
    # Assuming the JSON has "features" and "default" (label)
    
    def preprocess(examples):
        return {
            'input_features': examples['features'],
            'labels': [[x] for x in examples['default']] # Reshape to (Batch, 1)
        }
    
    train_dataset = train_dataset.map(preprocess, batched=True)
    eval_dataset = eval_dataset.map(preprocess, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_features', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_features', 'labels'])
    
    # Initialize model
    config = CreditScoringConfig()
    model = HFTabularModel(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='../models/hf_checkpoints',
        num_train_epochs=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-3,
        logging_dir='../models/logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to='wandb',
        run_name='hf-tabular-model'
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained('../models/hf_final_model')
    
    print("Training completed!")


if __name__ == "__main__":
    train_with_hf()
