"""
Data utilities for UrbanSound8K with AST (Audio Spectrogram Transformer).
"""
import os
import pandas as pd
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor


class UrbanSound8KASTDataset(Dataset):
    """PyTorch Dataset for UrbanSound8K with AST preprocessing."""
    
    def __init__(self, metadata_path, feature_extractor, max_length=10.0):
        """
        Args:
            metadata_path: Path to CSV with file paths and labels
            feature_extractor: ASTFeatureExtractor instance
            max_length: Maximum audio length in seconds
        """
        self.metadata = pd.read_csv(metadata_path)
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.target_sample_rate = feature_extractor.sampling_rate
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = row['file_path']
        label = row['classID']
        
        try:
            # Load audio using librosa (more reliable, no torchcodec dependency)
            waveform, sample_rate = librosa.load(
                audio_path,
                sr=self.target_sample_rate,
                mono=True,
                duration=self.max_length
            )
            
            # waveform is already numpy array, mono, and resampled
            
            # Pad or trim to max_length
            max_samples = int(self.max_length * self.target_sample_rate)
            if len(waveform) < max_samples:
                waveform = np.pad(waveform, (0, max_samples - len(waveform)))
            else:
                waveform = waveform[:max_samples]
            
            # Use AST feature extractor
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            )
            
            return {
                'input_values': inputs['input_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a zero tensor on error
            dummy_inputs = self.feature_extractor(
                np.zeros(int(self.max_length * self.target_sample_rate)),
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            )
            return {
                'input_values': dummy_inputs['input_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }


def load_urbansound_ast_data(data_dir='data/urbansound_ast_processed'):
    """
    Load UrbanSound8K metadata for AST.
    
    Args:
        data_dir: Directory containing processed metadata
        
    Returns:
        Dictionary with paths to train/val/test metadata
    """
    train_path = os.path.join(data_dir, 'train_metadata.csv')
    val_path = os.path.join(data_dir, 'val_metadata.csv')
    test_path = os.path.join(data_dir, 'test_metadata.csv')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Metadata files not found in {data_dir}\n"
            "Please run preprocess_urbansound_ast.py first."
        )
    
    print(f"Loading UrbanSound8K metadata for AST from {data_dir}...")
    
    data_dict = {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }
    
    # Load class info
    class_info_path = os.path.join(data_dir, 'class_info.npy')
    if os.path.exists(class_info_path):
        class_info = np.load(class_info_path, allow_pickle=True).item()
        data_dict['class_names'] = class_info['class_names']
        data_dict['num_classes'] = class_info['num_classes']
        data_dict['id2label'] = class_info['id2label']
        data_dict['label2id'] = class_info['label2id']
    
    # Print stats
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(data_dict[split])
        print(f"  {split.capitalize()}: {len(df)} samples")
    
    return data_dict


def ast_collate_fn(batch):
    """Custom collate function for batching AST data."""
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_values': input_values, 'labels': labels}


def create_urbansound_ast_dataloaders(metadata_dict, feature_extractor, 
                                      batch_size=8, num_workers=2, max_length=10.0):
    """
    Create PyTorch DataLoaders for UrbanSound8K with AST.
    
    Args:
        metadata_dict: Dictionary from load_urbansound_ast_data()
        feature_extractor: ASTFeatureExtractor instance
        batch_size: Batch size for training (smaller for AST due to memory)
        num_workers: Number of workers for data loading
        max_length: Maximum audio length in seconds
        
    Returns:
        Dictionary with train/val/test DataLoaders
    """
    datasets = {
        'train': UrbanSound8KASTDataset(
            metadata_dict['train'],
            feature_extractor,
            max_length=max_length
        ),
        'val': UrbanSound8KASTDataset(
            metadata_dict['val'],
            feature_extractor,
            max_length=max_length
        ),
        'test': UrbanSound8KASTDataset(
            metadata_dict['test'],
            feature_extractor,
            max_length=max_length
        )
    }
    
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=ast_collate_fn,
            pin_memory=False
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=ast_collate_fn,
            pin_memory=False
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=ast_collate_fn,
            pin_memory=False
        )
    }
    
    return loaders
