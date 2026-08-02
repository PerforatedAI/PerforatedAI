"""
Data preparation module for WikiText-2 dataset.
Handles downloading, tokenization, and batching for language modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import numpy as np


class Vocabulary:
    """Simple vocabulary for word-level language modeling."""

    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()

        # Special tokens
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.EOS_TOKEN = "<eos>"

        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.EOS_IDX = 2

        # Initialize special tokens
        self.word2idx[self.PAD_TOKEN] = self.PAD_IDX
        self.word2idx[self.UNK_TOKEN] = self.UNK_IDX
        self.word2idx[self.EOS_TOKEN] = self.EOS_IDX

        self.idx2word[self.PAD_IDX] = self.PAD_TOKEN
        self.idx2word[self.UNK_IDX] = self.UNK_TOKEN
        self.idx2word[self.EOS_IDX] = self.EOS_TOKEN

    def build_vocab(self, texts):
        """Build vocabulary from list of texts."""
        for text in texts:
            tokens = text.lower().split()
            self.word_counts.update(tokens)
        
        # Get most common words (excluding special tokens)
        # We want max_vocab_size total tokens, including the 3 special tokens
        num_regular_tokens = self.max_vocab_size - 3
        
        # Filter out special tokens from most common words to avoid duplicates
        special_tokens = {self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN}
        filtered_common = []
        for word, count in self.word_counts.most_common():
            if word not in special_tokens:
                filtered_common.append((word, count))
            if len(filtered_common) >= num_regular_tokens:
                break
        
        # Add to vocabulary
        for idx, (word, _) in enumerate(filtered_common, start=3):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"Vocabulary built with {len(self.word2idx)} tokens")
        print(f"Vocabulary indices range from 0 to {max(self.word2idx.values())}")
        return self
    
    def encode(self, text):
        """Convert text to list of token indices."""
        tokens = text.lower().split()
        return [self.word2idx.get(token, self.UNK_IDX) for token in tokens]

    def decode(self, indices):
        """Convert list of indices back to text."""
        return " ".join([self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices])

    def __len__(self):
        return len(self.word2idx)


class WikiTextDataset(Dataset):
    """Dataset for WikiText-2 language modeling."""

    def __init__(self, token_ids, seq_length):
        """
        Args:
            token_ids: List of token indices
            seq_length: Length of each training sequence
        """
        self.token_ids = token_ids
        self.seq_length = seq_length

        # Create sequences
        self.sequences = []
        for i in range(0, len(token_ids) - seq_length - 1, seq_length):
            input_seq = token_ids[i : i + seq_length]
            target_seq = token_ids[i + 1 : i + seq_length + 1]
            self.sequences.append((input_seq, target_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(
            target_seq, dtype=torch.long
        )


def load_wikitext2(seq_length=50, batch_size=32, max_vocab_size=10000):
    """
    Load and preprocess WikiText-2 dataset.

    Args:
        seq_length: Length of sequences for training
        batch_size: Batch size for DataLoader
        max_vocab_size: Maximum vocabulary size

    Returns:
        train_loader, val_loader, test_loader, vocabulary
    """
    print("Loading WikiText-2 dataset...")

    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-v1")

    # Extract text
    train_texts = [text for text in dataset["train"]["text"] if text.strip()]
    val_texts = [text for text in dataset["validation"]["text"] if text.strip()]
    test_texts = [text for text in dataset["test"]["text"] if text.strip()]

    print(f"Train texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}")
    print(f"Test texts: {len(test_texts)}")

    # Build vocabulary from training data
    vocab = Vocabulary(max_vocab_size=max_vocab_size)
    vocab.build_vocab(train_texts)

    # Tokenize all texts
    train_ids = []
    for text in train_texts:
        train_ids.extend(vocab.encode(text))
        train_ids.append(vocab.EOS_IDX)

    val_ids = []
    for text in val_texts:
        val_ids.extend(vocab.encode(text))
        val_ids.append(vocab.EOS_IDX)

    test_ids = []
    for text in test_texts:
        test_ids.extend(vocab.encode(text))
        test_ids.append(vocab.EOS_IDX)

    print(f"Train tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(val_ids)}")
    print(f"Test tokens: {len(test_ids)}")

    # Create datasets
    train_dataset = WikiTextDataset(train_ids, seq_length)
    val_dataset = WikiTextDataset(val_ids, seq_length)
    test_dataset = WikiTextDataset(test_ids, seq_length)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, vocab = load_wikitext2()

    # Print sample batch
    for inputs, targets in train_loader:
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Sample input: {inputs[0][:10]}")
        print(f"Decoded: {vocab.decode(inputs[0][:10].tolist())}")
        break
