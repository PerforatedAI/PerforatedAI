"""
Data loading utilities for ESC-50 dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class ESC50Dataset(Dataset):
    """PyTorch Dataset for preprocessed ESC-50 spectrograms"""
    
    def __init__(self, X, y):
        """
        Args:
            X: numpy array of spectrograms (N, H, W)
            y: numpy array of labels (N,)
        """
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def stratified_subsample(X, y, fraction, random_state=42):
    """
    Subsample data while maintaining class balance (stratified).

    Args:
        X: Features array
        y: Labels array
        fraction: Fraction of data to keep (e.g., 0.75 for 75%)
        random_state: Random seed for reproducibility

    Returns:
        Subsampled X, y arrays
    """
    if fraction >= 1.0:
        return X, y

    # Use train_test_split to do stratified sampling
    # We keep 'fraction' of the data, discard the rest
    X_keep, _, y_keep, _ = train_test_split(
        X, y,
        train_size=fraction,
        stratify=y,
        random_state=random_state
    )

    return X_keep, y_keep


def load_preprocessed_data(data_dir='preprocessed', data_fraction=1.0, random_state=42):
    """
    Load preprocessed spectrograms from disk.

    Args:
        data_dir: Directory containing .npy files
        data_fraction: Fraction of training data to use (0.0-1.0)
                       Only affects training data, val/test remain full size
        random_state: Random seed for reproducible subsampling

    Returns:
        Dictionary with train, val, test splits
    """
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')

    X_val = np.load(f'{data_dir}/X_val.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')

    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')

    # Apply stratified subsampling to training data only
    if data_fraction < 1.0:
        original_size = len(y_train)
        X_train, y_train = stratified_subsample(X_train, y_train, data_fraction, random_state)
        print(f"Subsampled training data: {original_size} -> {len(y_train)} samples ({data_fraction*100:.0f}%)")

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def create_dataloaders(data_dict, batch_size=32, num_workers=2):
    """
    Create PyTorch DataLoaders for train, val, test sets.
    
    Args:
        data_dict: Dictionary from load_preprocessed_data()
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary of DataLoaders
    """
    train_dataset = ESC50Dataset(*data_dict['train'])
    val_dataset = ESC50Dataset(*data_dict['val'])
    test_dataset = ESC50Dataset(*data_dict['test'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
