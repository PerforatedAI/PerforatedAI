"""
Training script for Dendritic Sentence Classifier

Usage:
    python train_classifier.py --data-path data/forum_sentences.csv --output models/classifier.pth
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
import json
import logging

from dendritic_classifier import (
    DendriticSentenceClassifier,
    train_with_dendritic_optimization
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceDataset(Dataset):
    """Dataset for sentence classification."""
    
    def __init__(self, sentences: list, labels: list, tokenizer, max_length: int = 128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_path: str):
    """Load training data from CSV."""
    df = pd.read_csv(data_path)
    
    # Map category labels to integers
    label_map = {
        "fact": 0,
        "solution": 1,
        "opinion": 2,
        "question": 3,
        "citation": 4,
        "irrelevant": 5
    }
    
    sentences = df['sentence'].tolist()
    labels = [label_map.get(label, 5) for label in df['category']]  # Default to irrelevant
    
    return sentences, labels


def main():
    parser = argparse.ArgumentParser(description='Train Dendritic Sentence Classifier')
    parser.add_argument('--data-path', type=str, default='data/train.csv', help='Path to training data CSV')
    parser.add_argument('--val-path', type=str, default='data/val.csv', help='Path to validation data CSV')
    parser.add_argument('--output', type=str, default='models/classifier.pth', help='Output model path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    logger.info(f"Loading training data from {args.data_path}")
    train_sentences, train_labels = load_data(args.data_path)
    
    logger.info(f"Loading validation data from {args.val_path}")
    val_sentences, val_labels = load_data(args.val_path)
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SentenceDataset(train_sentences, train_labels, tokenizer)
    val_dataset = SentenceDataset(val_sentences, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    logger.info("Initializing model")
    model = DendriticSentenceClassifier()
    
    # Train with dendritic optimization
    logger.info("Starting training with dendritic optimization")
    results = train_with_dendritic_optimization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results
    }, output_path)
    
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Training results: {json.dumps(results, indent=2)}")
    
    # Save results
    results_path = output_path.parent / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
