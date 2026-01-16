"""
Data utilities for UrbanSound8K dataset.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UrbanSound8KDataset(Dataset):
    """PyTorch Dataset for UrbanSound8K preprocessed data."""
    
    def __init__(self, spectrograms, labels):
        """
        Args:
            spectrograms: numpy array of shape (N, 1, freq, time)
            labels: numpy array of shape (N,)
        """
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


def load_urbansound_data(data_dir='data/urbansound_processed'):
    """
    Load preprocessed UrbanSound8K data.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Dictionary with train/val/test spectrograms and labels
    """
    data_path = os.path.join(data_dir, 'urbansound8k_processed.npz')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at {data_path}\n"
            "Please run preprocess_urbansound.py first."
        )
    
    print(f"Loading UrbanSound8K data from {data_path}...")
    data = np.load(data_path)
    
    data_dict = {
        'train': {
            'spectrograms': data['train_spectrograms'],
            'labels': data['train_labels']
        },
        'val': {
            'spectrograms': data['val_spectrograms'],
            'labels': data['val_labels']
        },
        'test': {
            'spectrograms': data['test_spectrograms'],
            'labels': data['test_labels']
        }
    }
    
    # Load class info
    class_info_path = os.path.join(data_dir, 'class_info.npy')
    if os.path.exists(class_info_path):
        class_info = np.load(class_info_path, allow_pickle=True).item()
        data_dict['class_names'] = class_info['class_names']
        data_dict['num_classes'] = class_info['num_classes']
    
    print("Data loaded successfully!")
    print(f"  Train: {data_dict['train']['spectrograms'].shape}")
    print(f"  Val:   {data_dict['val']['spectrograms'].shape}")
    print(f"  Test:  {data_dict['test']['spectrograms'].shape}")
    
    return data_dict


def create_urbansound_dataloaders(data_dict, batch_size=32, num_workers=2):
    """
    Create PyTorch DataLoaders for UrbanSound8K.
    
    Args:
        data_dict: Dictionary from load_urbansound_data()
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with train/val/test DataLoaders
    """
    datasets = {
        'train': UrbanSound8KDataset(
            data_dict['train']['spectrograms'],
            data_dict['train']['labels']
        ),
        'val': UrbanSound8KDataset(
            data_dict['val']['spectrograms'],
            data_dict['val']['labels']
        ),
        'test': UrbanSound8KDataset(
            data_dict['test']['spectrograms'],
            data_dict['test']['labels']
        )
    }
    
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
    }
    
    return loaders
