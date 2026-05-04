"""
Preprocessing pipeline for ESC-50 dataset.
Converts audio files to mel-spectrograms and saves them for fast loading.
"""
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import argparse

import config


def extract_melspectrogram(audio_path, sr=None, n_mels=None, n_fft=None, hop_length=None, max_len=None):
    """
    Convert audio file to mel-spectrogram.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate (defaults to config)
        n_mels: Number of mel bands (defaults to config)
        n_fft: FFT window size (defaults to config)
        hop_length: Hop length (defaults to config)
        max_len: Maximum length in samples (pads or trims)
        
    Returns:
        Mel-spectrogram in dB scale
    """
    # Use config defaults if not provided
    if sr is None:
        sr = config.PREPROCESSING['sample_rate']
    if n_mels is None:
        n_mels = config.PREPROCESSING['n_mels']
    if n_fft is None:
        n_fft = config.PREPROCESSING['n_fft']
    if hop_length is None:
        hop_length = config.PREPROCESSING['hop_length']
    
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=config.PREPROCESSING['duration'])
    
    # Pad or trim to fixed length
    if max_len is not None:
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def preprocess_esc50(data_dir=None, output_dir=None, sr=None, n_mels=None):
    """
    Preprocess all ESC-50 audio files to mel-spectrograms.
    
    Args:
        data_dir: Root directory of ESC-50 dataset (defaults to config)
        output_dir: Directory to save preprocessed files (defaults to config)
        sr: Sample rate (defaults to config)
        n_mels: Number of mel bands (defaults to config)
    """
    # Use config defaults if not provided
    if data_dir is None:
        data_dir = config.DATA_DIR
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if sr is None:
        sr = config.PREPROCESSING['sample_rate']
    if n_mels is None:
        n_mels = config.PREPROCESSING['n_mels']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    meta_path = os.path.join(data_dir, 'meta', 'esc50.csv')
    print(f"Loading metadata from {meta_path}")
    meta_df = pd.read_csv(meta_path)
    
    print(f"Total samples: {len(meta_df)}")
    print(f"Number of classes: {meta_df['target'].nunique()}")
    
    # Configuration from config file
    max_len = int(config.PREPROCESSING['duration'] * sr)
    n_fft = config.PREPROCESSING['n_fft']
    hop_length = config.PREPROCESSING['hop_length']
    test_fold = config.PREPROCESSING['test_fold']
    val_split = config.PREPROCESSING['val_split']
    random_state = config.PREPROCESSING['random_state']
    
    # Preprocess all files
    print("\nPreprocessing audio files...")
    spectrograms = []
    labels = []
    
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        audio_path = os.path.join(data_dir, 'audio', row['filename'])
        
        # Extract spectrogram
        spec = extract_melspectrogram(
            audio_path, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft,
            hop_length=hop_length,
            max_len=max_len
        )
        spectrograms.append(spec)
        labels.append(row['target'])
    
    # Convert to numpy arrays
    X = np.array(spectrograms)
    y = np.array(labels)
    
    print(f"\nData shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split data using fold information
    print("\nSplitting data...")
    train_val_mask = meta_df['fold'] != test_fold
    test_mask = meta_df['fold'] == test_fold
    
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Further split train_val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_split,
        random_state=random_state,
        stratify=y_train_val
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {output_dir}/...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save label mapping
    label_mapping = dict(zip(meta_df['target'], meta_df['category']))
    with open(os.path.join(output_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(label_mapping, f)
    
    # Save preprocessing config
    preprocess_config = {
        'sr': sr,
        'n_mels': n_mels,
        'max_len': max_len,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'num_classes': meta_df['target'].nunique(),
        'test_fold': test_fold
    }
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(preprocess_config, f)
    
    print("\nPreprocessing complete!")
    print(f"Files saved to {output_dir}/")
    print("\nClass distribution:")
    print(f"Train: {np.bincount(y_train)[:5]}...")
    print(f"Val: {np.bincount(y_val)[:5]}...")
    print(f"Test: {np.bincount(y_test)[:5]}...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ESC-50 dataset')
    parser.add_argument('--data_dir', type=str, default=None,
                        help=f'Path to ESC-50 dataset (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=None,
                        help=f'Directory to save preprocessed files (default: {config.OUTPUT_DIR})')
    parser.add_argument('--sr', type=int, default=None,
                        help=f'Sample rate for audio (default: {config.PREPROCESSING["sample_rate"]})')
    parser.add_argument('--n_mels', type=int, default=None,
                        help=f'Number of mel bands (default: {config.PREPROCESSING["n_mels"]})')
    
    args = parser.parse_args()
    
    preprocess_esc50(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sr=args.sr,
        n_mels=args.n_mels
    )
