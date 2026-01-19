"""
RAVDESS Dataset Loader for Emotion Recognition

This module handles loading and preprocessing of the RAVDESS dataset,
converting audio files to Mel spectrograms for CNN classification.
"""

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# RAVDESS emotion labels
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTION_TO_IDX = {v: i for i, v in enumerate(EMOTIONS.values())}
IDX_TO_EMOTION = {i: v for i, v in enumerate(EMOTIONS.values())}


class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS audio emotion recognition.
    
    The RAVDESS dataset contains audio files with naming convention:
    {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
    
    We extract emotion from the filename and convert audio to Mel spectrogram.
    """
    
    def __init__(self, file_paths, labels, sr=22050, n_mels=128, 
                 max_len=128, augment=False):
        """
        Args:
            file_paths: List of paths to audio files
            labels: List of emotion labels (integers)
            sr: Sample rate for audio loading
            n_mels: Number of Mel bands
            max_len: Maximum length of spectrogram (time dimension)
            augment: Whether to apply data augmentation
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        audio, sr = librosa.load(file_path, sr=self.sr, duration=3.0)
        
        # Apply augmentation if training
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Convert to Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=self.n_mels,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Pad or truncate to fixed length
        if mel_spec_db.shape[1] < self.max_len:
            pad_width = self.max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :self.max_len]
        
        # Add channel dimension and convert to tensor
        mel_spec_db = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        
        return mel_spec_db, torch.tensor(label, dtype=torch.long)
    
    def _augment_audio(self, audio):
        """Apply random augmentations to audio."""
        # Random noise
        if np.random.random() < 0.5:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
            
        # Random time shift
        if np.random.random() < 0.5:
            shift = np.random.randint(-1000, 1000)
            audio = np.roll(audio, shift)
            
        # Random pitch shift
        if np.random.random() < 0.3:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
            
        return audio


def load_ravdess_data(data_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    Load RAVDESS dataset and split into train/val/test sets.
    
    Args:
        data_dir: Root directory containing RAVDESS audio files
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_files, train_labels, val_files, val_labels, test_files, test_labels
    """
    file_paths = []
    labels = []
    
    # Walk through directory to find all .wav files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Extract emotion from filename
                # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in EMOTIONS:
                        emotion = EMOTIONS[emotion_code]
                        file_paths.append(file_path)
                        labels.append(EMOTION_TO_IDX[emotion])
    
    if len(file_paths) == 0:
        raise ValueError(f"No valid audio files found in {data_dir}. "
                        "Please download the RAVDESS dataset.")
    
    print(f"Found {len(file_paths)} audio files")
    
    # Split into train+val and test
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )
    
    # Split train+val into train and val
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_size,
        random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory containing RAVDESS audio files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load and split data
    (train_files, train_labels, 
     val_files, val_labels, 
     test_files, test_labels) = load_ravdess_data(data_dir)
    
    # Create datasets
    train_dataset = RAVDESSDataset(train_files, train_labels, augment=True)
    val_dataset = RAVDESSDataset(val_files, val_labels, augment=False)
    test_dataset = RAVDESSDataset(test_files, test_labels, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_dataset(num_samples=1000, num_classes=8):
    """
    Create a synthetic dataset for testing when RAVDESS is not available.
    
    This generates random spectrograms with random labels for testing
    the training pipeline before downloading the real dataset.
    """
    print("Creating synthetic dataset for testing...")
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples, num_classes):
            self.num_samples = num_samples
            self.num_classes = num_classes
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random spectrogram
            spec = torch.randn(1, 128, 128)
            # Random label
            label = torch.randint(0, self.num_classes, (1,)).item()
            return spec, torch.tensor(label, dtype=torch.long)
    
    train_dataset = SyntheticDataset(int(num_samples * 0.7), num_classes)
    val_dataset = SyntheticDataset(int(num_samples * 0.15), num_classes)
    test_dataset = SyntheticDataset(int(num_samples * 0.15), num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader
