"""
Tests for TLDRDataModule

Run with: pytest ml/tests/test_datamodule.py -v
"""

import pytest
import torch


def test_datamodule_setup():
    """Test that datamodule initializes correctly."""
    from ml.data.datamodule import TLDRDataModule
    
    dm = TLDRDataModule.small_synthetic_dataset(size=20)
    dm.setup()
    
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert len(dm.train_dataset) > 0


def test_datamodule_shapes():
    """Test that batches have correct shapes."""
    from ml.data.datamodule import TLDRDataModule
    
    dm = TLDRDataModule.small_synthetic_dataset(size=20)
    dm.setup()
    
    batch = next(iter(dm.train_dataloader()))
    
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'labels' in batch
    assert batch['input_ids'].dim() == 2
    assert batch['input_ids'].shape[0] == dm.batch_size or batch['input_ids'].shape[0] <= 20


def test_sentence_splitting():
    """Test sentence splitting function."""
    from ml.data.datamodule import split_into_sentences
    
    text = "Hello world. This is a test. Another sentence!"
    sentences = split_into_sentences(text)
    
    assert len(sentences) >= 2
    for sent, start, end in sentences:
        assert text[start:end].strip() == sent or sent in text


def test_tokenization_consistency():
    """Test that tokenization is deterministic."""
    from ml.data.datamodule import TLDRDataModule
    
    dm1 = TLDRDataModule.small_synthetic_dataset(size=10)
    dm2 = TLDRDataModule.small_synthetic_dataset(size=10)
    
    dm1.setup()
    dm2.setup()
    
    batch1 = dm1.train_dataset[0]
    batch2 = dm2.train_dataset[0]
    
    assert torch.equal(batch1['input_ids'], batch2['input_ids'])
