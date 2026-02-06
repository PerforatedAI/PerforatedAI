"""
Original Baseline: BERT-Tiny for Toxicity Classification
Without Dendritic Optimization

This script trains a standard BERT-Tiny model without PerforatedAI enhancement.
It serves as the baseline for comparison with the dendritic-enhanced version.

Usage:
    python train_original.py
    
This is equivalent to running:
    python src/train.py --no-dendrites
    
But provides a standalone reference implementation.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import os


class ToxicityDataset(Dataset):
    """Simple toxicity dataset without augmentation."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data(sample_size=5000):
    """Load Civil Comments dataset."""
    print("Loading dataset...")
    dataset = load_dataset("google/civil_comments", split="train[:10000]")
    
    # Extract toxicity labels
    texts = dataset["text"]
    labels = [1 if toxicity >= 0.5 else 0 for toxicity in dataset["toxicity"]]
    
    # Split into train/val/test
    train_size = sample_size
    val_size = 1000
    test_size = 1000
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    
    val_texts = texts[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    
    test_texts = texts[train_size + val_size:train_size + val_size + test_size]
    test_labels = labels[train_size + val_size:train_size + val_size + test_size]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def compute_class_weights_balanced(labels):
    """Compute balanced class weights."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    # Configuration
    MODEL_NAME = "prajjwal1/bert-tiny"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    PATIENCE = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data()
    
    # Compute class weights
    class_weights = compute_class_weights_balanced(train_labels)
    print(f"\nClass weights: {class_weights}")
    print(f"Toxic weight multiplier: {class_weights[1] / class_weights[0]:.2f}x")
    
    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=0.1,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create datasets and dataloaders
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = ToxicityDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Training loop
    print("\n" + "=" * 60)
    print("BASELINE TRAINING (WITHOUT DENDRITES)")
    print("=" * 60)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/baseline_best.pt")
            print("✓ Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        scheduler.step()
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    model.load_state_dict(torch.load("checkpoints/baseline_best.pt"))
    test_loss, test_acc, test_preds, test_labels_array = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_labels_array,
        test_preds,
        target_names=["Non-Toxic", "Toxic"],
        digits=4
    ))
    
    print("\n✓ Training complete!")
    print(f"Best model saved to: checkpoints/baseline_best.pt")


if __name__ == "__main__":
    main()
