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


def extract_melspectrogram(audio_path, sr=22050, n_mels=128, max_len=None):
    """
    Convert audio file to mel-spectrogram.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel bands
        max_len: Maximum length in samples (pads or trims)
        
    Returns:
        Mel-spectrogram in dB scale
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=5.0)
    
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
        n_fft=2048,
        hop_length=512
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def preprocess_esc50(data_dir='data/ESC-50', output_dir='preprocessed', sr=22050, n_mels=128):
    """
    Preprocess all ESC-50 audio files to mel-spectrograms.
    
    Args:
        data_dir: Root directory of ESC-50 dataset
        output_dir: Directory to save preprocessed files
        sr: Sample rate
        n_mels: Number of mel bands
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    meta_path = os.path.join(data_dir, 'meta', 'esc50.csv')
    print(f"Loading metadata from {meta_path}")
    meta_df = pd.read_csv(meta_path)
    
    print(f"Total samples: {len(meta_df)}")
    print(f"Number of classes: {meta_df['target'].nunique()}")
    
    # Configuration
    max_len = 5 * sr  # 5 seconds
    
    # Preprocess all files
    print("\nPreprocessing audio files...")
    spectrograms = []
    labels = []
    
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        audio_path = os.path.join(data_dir, 'audio', row['filename'])
        
        # Extract spectrogram
        spec = extract_melspectrogram(audio_path, sr=sr, n_mels=n_mels, max_len=max_len)
        spectrograms.append(spec)
        labels.append(row['target'])
    
    # Convert to numpy arrays
    X = np.array(spectrograms)
    y = np.array(labels)
    
    print(f"\nData shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split data using fold information
    # Use fold 5 for test (standard practice in ESC-50)
    print("\nSplitting data...")
    test_fold = 5
    train_val_mask = meta_df['fold'] != test_fold
    test_mask = meta_df['fold'] == test_fold
    
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Further split train_val into train and validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.2,
        random_state=42,
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
    config = {
        'sr': sr,
        'n_mels': n_mels,
        'max_len': max_len,
        'n_fft': 2048,
        'hop_length': 512,
        'num_classes': meta_df['target'].nunique(),
        'test_fold': test_fold
    }
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    print("\nPreprocessing complete!")
    print(f"Files saved to {output_dir}/")
    print("\nClass distribution:")
    print(f"Train: {np.bincount(y_train)[:5]}...")
    print(f"Val: {np.bincount(y_val)[:5]}...")
    print(f"Test: {np.bincount(y_test)[:5]}...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess ESC-50 dataset')
    parser.add_argument('--data_dir', type=str, default='data/ESC-50',
                        help='Path to ESC-50 dataset')
    parser.add_argument('--output_dir', type=str, default='preprocessed',
                        help='Directory to save preprocessed files')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate for audio')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of mel bands')
    
    args = parser.parse_args()
    
    preprocess_esc50(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sr=args.sr,
        n_mels=args.n_mels
    )
