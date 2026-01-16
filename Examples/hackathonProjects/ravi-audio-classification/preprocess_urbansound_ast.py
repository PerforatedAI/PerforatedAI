"""
Preprocess UrbanSound8K dataset for AST (Audio Spectrogram Transformer).
AST uses its own feature extraction, so we just organize the audio files.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path


# Configuration
URBANSOUND_DIR = 'data/urbansound'
OUTPUT_DIR = 'data/urbansound_ast_processed'
CSV_PATH = os.path.join(URBANSOUND_DIR, 'UrbanSound8K.csv')

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


def preprocess_urbansound8k_ast():
    """
    Prepare UrbanSound8K dataset for AST training.
    
    AST uses raw audio files, so we just create metadata files
    with paths and labels for train/val/test splits.
    
    Dataset structure:
    - 10 classes (0-9)
    - 8,732 audio files
    - Organized in 10 folds (fold1-fold10)
    
    We'll use:
    - Folds 1-8: Training (70%)
    - Fold 9: Validation (10%)
    - Fold 10: Testing (10%)
    """
    
    print("\n" + "="*60)
    print("UrbanSound8K AST Preprocessing")
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
    
    train_df = df[df['fold'].isin(train_folds)].copy()
    val_df = df[df['fold'] == val_fold].copy()
    test_df = df[df['fold'] == test_fold].copy()
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)} samples (folds {train_folds})")
    print(f"  Val:   {len(val_df)} samples (fold {val_fold})")
    print(f"  Test:  {len(test_df)} samples (fold {test_fold})")
    
    # Add full file paths
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df['file_path'] = split_df.apply(
            lambda row: os.path.join(URBANSOUND_DIR, f"fold{row['fold']}", row['slice_file_name']),
            axis=1
        )
        
        # Verify files exist
        print(f"\nVerifying {split_name} files...")
        missing_files = []
        for idx, row in split_df.iterrows():
            if not os.path.exists(row['file_path']):
                missing_files.append(row['file_path'])
        
        if missing_files:
            print(f"  Warning: {len(missing_files)} files not found")
            for f in missing_files[:5]:
                print(f"    {f}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")
        else:
            print(f"  ✓ All {len(split_df)} files verified")
        
        # Save metadata
        output_path = os.path.join(OUTPUT_DIR, f'{split_name}_metadata.csv')
        split_df[['file_path', 'classID', 'class']].to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")
    
    # Save class info
    class_info = {
        'class_names': CLASSES,
        'num_classes': len(CLASSES),
        'id2label': {i: name for i, name in enumerate(CLASSES)},
        'label2id': {name: i for i, name in enumerate(CLASSES)}
    }
    np.save(os.path.join(OUTPUT_DIR, 'class_info.npy'), class_info, allow_pickle=True)
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nProcessed metadata saved to: {OUTPUT_DIR}/")
    print(f"  - train_metadata.csv ({len(train_df)} samples)")
    print(f"  - val_metadata.csv ({len(val_df)} samples)")
    print(f"  - test_metadata.csv ({len(test_df)} samples)")
    print(f"  - class_info.npy")
    print("\nAST will process audio files on-the-fly during training.")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    preprocess_urbansound8k_ast()
