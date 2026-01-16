"""
Preprocess UrbanSound8K dataset for training.
Convert audio files to mel-spectrograms.
"""
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path


# Configuration
URBANSOUND_DIR = 'data/urbansound'
OUTPUT_DIR = 'data/urbansound_processed'
CSV_PATH = os.path.join(URBANSOUND_DIR, 'UrbanSound8K.csv')

# Audio processing parameters (same as ESC-50 for consistency)
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
DURATION = 4.0  # seconds - pad/trim to 4 seconds

# Class mapping (10 urban sound classes)
CLASSES = [
    'air_conditioner',  # 0
    'car_horn',         # 1
    'children_playing', # 2
    'dog_bark',         # 3
    'drilling',         # 4
    'engine_idling',    # 5
    'gun_shot',         # 6
    'jackhammer',       # 7
    'siren',            # 8
    'street_music'      # 9
]


def extract_melspectrogram(audio_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Extract mel-spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        duration: Target duration in seconds
        
    Returns:
        mel_spec: Mel-spectrogram as numpy array
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Pad or trim to fixed length
        target_length = int(sr * duration)
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            # Trim
            audio = audio[:target_length]
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def preprocess_urbansound8k():
    """
    Preprocess UrbanSound8K dataset.
    
    Dataset structure:
    - 10 classes (0-9)
    - 8,732 audio files
    - Organized in 10 folds (fold1-fold10)
    - Each audio is <=4 seconds
    
    We'll use:
    - Folds 1-8: Training (70%)
    - Fold 9: Validation (10%)
    - Fold 10: Testing (10%)
    """
    
    print("\n" + "="*60)
    print("UrbanSound8K Preprocessing")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    print(f"\nLoading metadata from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(df)}")
    
    # Display class distribution
    print("\nClass distribution:")
    for class_id in range(10):
        count = len(df[df['classID'] == class_id])
        class_name = CLASSES[class_id]
        print(f"  {class_id}: {class_name:20s} - {count:4d} samples")
    
    # Split by folds
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    val_fold = 9
    test_fold = 10
    
    train_df = df[df['fold'].isin(train_folds)]
    val_df = df[df['fold'] == val_fold]
    test_df = df[df['fold'] == test_fold]
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} samples (folds {train_folds})")
    print(f"  Val:   {len(val_df)} samples (fold {val_fold})")
    print(f"  Test:  {len(test_df)} samples (fold {test_fold})")
    
    # Process each split
    data = {}
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\nProcessing {split_name} set...")
        
        spectrograms = []
        labels = []
        filenames = []
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            # Construct audio file path
            audio_filename = row['slice_file_name']
            fold = row['fold']
            audio_path = os.path.join(URBANSOUND_DIR, f'fold{fold}', audio_filename)
            
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                continue
            
            # Extract mel-spectrogram
            mel_spec = extract_melspectrogram(audio_path)
            
            if mel_spec is not None:
                spectrograms.append(mel_spec)
                labels.append(row['classID'])
                filenames.append(audio_filename)
        
        # Convert to numpy arrays
        spectrograms = np.array(spectrograms, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Add channel dimension (batch, 1, freq, time)
        spectrograms = np.expand_dims(spectrograms, axis=1)
        
        print(f"  Spectrograms shape: {spectrograms.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Save processed data
        data[split_name] = {
            'spectrograms': spectrograms,
            'labels': labels,
            'filenames': filenames
        }
    
    # Save all data
    print(f"\nSaving processed data to {OUTPUT_DIR}...")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, 'urbansound8k_processed.npz'),
        train_spectrograms=data['train']['spectrograms'],
        train_labels=data['train']['labels'],
        val_spectrograms=data['val']['spectrograms'],
        val_labels=data['val']['labels'],
        test_spectrograms=data['test']['spectrograms'],
        test_labels=data['test']['labels']
    )
    
    # Save class names
    class_info = {
        'class_names': CLASSES,
        'num_classes': len(CLASSES)
    }
    np.save(os.path.join(OUTPUT_DIR, 'class_info.npy'), class_info, allow_pickle=True)
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nProcessed data saved to: {OUTPUT_DIR}/")
    print(f"  - urbansound8k_processed.npz")
    print(f"  - class_info.npy")
    
    # Print shapes
    print("\nFinal shapes:")
    print(f"  Train: {data['train']['spectrograms'].shape}")
    print(f"  Val:   {data['val']['spectrograms'].shape}")
    print(f"  Test:  {data['test']['spectrograms'].shape}")
    
    return data


if __name__ == '__main__':
    preprocess_urbansound8k()
