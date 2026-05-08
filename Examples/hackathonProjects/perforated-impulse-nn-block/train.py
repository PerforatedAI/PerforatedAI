#!/usr/bin/env python3
# Generalized train.py for Edge Impulse custom learning blocks with PerforatedAI

import sys
# 
# SUPPORTED DATA MODALITIES:
# ==========================
# 
# 1. IMAGE DATA (2D classification)
#    - Raw images (RGB/grayscale)
#    - Resized/preprocessed images
#    - Image preprocessing block outputs
#    - Input format: (N, H, W, C) where C=1 (grayscale) or C=3 (RGB)
#    - Uses 2D convolutions with adaptive architecture
# 
# 2. AUDIO DATA (1D/2D classification)
#    - MFCC features (Mel-Frequency Cepstral Coefficients)
#    - MFE features (Mel-Filterbank Energy)
#    - Spectrogram processing block outputs
#    - Input format: (N, time, freq, 1) treated as 2D image
#    - Can use either 1D or 2D convolutions depending on shape
# 
# 3. TIME-SERIES SENSOR DATA (1D classification)
#    - Accelerometer, gyroscope, IMU sensors
#    - Environmental sensors (temperature, humidity, pressure, etc.)
#    - Spectral features (FFT, PSD, etc.)
#    - Raw sensor features
#    - Input format: (N, features) as flattened feature vector
#    - Uses 1D convolutions or direct fully-connected layers
# 
# 4. CUSTOM NUMERIC FEATURES (1D classification)
#    - Hand-engineered features
#    - External precomputed embeddings
#    - Output from custom processing blocks
#    - Input format: (N, features) as any flattened NumPy array
#    - Uses fully-connected architecture
# 
# AUTOMATIC ADAPTATION:
# =====================
# The script automatically detects the input shape from X_train.npy and:
# - Determines if data is 1D (time-series/features) or 2D (images/spectrograms)
# - Selects appropriate convolution type (Conv1d vs Conv2d)
# - Adjusts model architecture accordingly
# - Handles NHWC to NCHW conversion for PyTorch compatibility
# 
# PERFORATED AI INTEGRATION:
# ==========================
# Supports dendritic optimization via PerforatedAI for automatic neural architecture
# search and optimization. Enable with --dendritic-optimization true
# 
# OUTPUT:
# =======
# - Exports to ONNX format (model.onnx) for Edge Impulse deployment
# - Optional onnx2tf conversion to TensorFlow SavedModel
# - Creates compatibility shim for Edge Impulse postprocessing
#
# Exports to ONNX format with optional onnx2tf conversion and TF SavedModel compat shim.

import os
import sys
import json
import time
import argparse
import traceback
import subprocess as _subprocess
import shutil
import glob

# Force PyTorch to use legacy ONNX exporter (pre-torch.export)
os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# PerforatedAI helpers
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser(description='Running custom PyTorch models in Edge Impulse')

# Edge Impulse specific (may be passed but not used)
parser.add_argument('--info-file', type=str, default='')

# Data / run args (Edge Impulse standard) - using dashes for Edge Impulse compatibility
parser.add_argument('--data-directory', type=str, default='data', dest='data_directory')
parser.add_argument('--out-directory', type=str, default='out', dest='out_directory')

# Training hyperparameters
parser.add_argument('--batch-size', type=int, default=128, dest='batch_size')
parser.add_argument('--learning-rate', type=float, default=0.005, dest='learning_rate')
parser.add_argument('--seed', type=int, default=-1, help="Random seed for reproducibility. Use -1 for random (non-deterministic) behavior")
parser.add_argument('--softmax-export', type=str, default='false', help="Include softmax in exported ONNX/TFLite models (default: false for raw logits)", dest='softmax_export')
parser.add_argument('--noise-std', type=str, default='None', help="Gaussian noise augmentation (None/Low/High)", dest='noise_std')

# Data augmentation parameters (SpecAugment for spectrograms)
parser.add_argument('--mask-time-bands', type=str, default='None', help="Time band masking (None/Low/High)", dest='mask_time_bands')
parser.add_argument('--mask-freq-bands', type=str, default='None', help="Frequency band masking (None/Low/High)", dest='mask_freq_bands')
parser.add_argument('--warp-time', type=str, default='false', help="Enable time warping", dest='warp_time')
parser.add_argument('--auto-weight-classes', type=str, default='false', help="Auto weight classes", dest='auto_weight_classes')

# Layer-by-layer architecture parameters (layers 0-19)
for i in range(20):
    parser.add_argument(f'--layer-{i}-type', type=str, default='None', dest=f'layer_{i}_type',
                        choices=['None', 'Dense', '1D Convolution/Pool', '2D Convolution/Pool', 'Dropout', 'Reshape'],
                        help=f"Type of layer {i}")
    # Dense layer parameters
    parser.add_argument(f'--layer-{i}-dense-channels', type=str, default='', dest=f'layer_{i}_dense_channels',
                        help=f"Layer {i} Dense: number of output channels")
    # 1D Conv/Pool parameters
    parser.add_argument(f'--layer-{i}-conv1d-config', type=str, default='', dest=f'layer_{i}_conv1d_config',
                        help=f"Layer {i} 1D Conv/Pool: channels,kernel_size,layer_count (comma-separated)")
    # 2D Conv/Pool parameters
    parser.add_argument(f'--layer-{i}-conv2d-config', type=str, default='', dest=f'layer_{i}_conv2d_config',
                        help=f"Layer {i} 2D Conv/Pool: channels,kernel_size,layer_count (comma-separated)")
    # Dropout parameters
    parser.add_argument(f'--layer-{i}-dropout-rate', type=str, default='', dest=f'layer_{i}_dropout_rate',
                        help=f"Layer {i} Dropout: dropout rate (0.0-1.0)")
    # Reshape parameters
    parser.add_argument(f'--layer-{i}-reshape-columns', type=str, default='', dest=f'layer_{i}_reshape_columns',
                        help=f"Layer {i} Reshape: number of columns (freq bins) to reshape flat input into")

# PerforatedAI dendritic optimization parameters
parser.add_argument('--dendritic-optimization', type=str, required=False, default="true", dest='dendritic_optimization')
parser.add_argument('--switch-speed', type=str, default='slow', help="speed to switch", choices=['extra-slow', 'slow', 'medium', 'fast'], dest='switch_speed')
parser.add_argument('--max-dendrites', type=int, default=1, dest='max_dendrites')
parser.add_argument('--improvement-threshold', type=str, default='medium', choices=['high', 'medium', 'low'], dest='improvement_threshold')
parser.add_argument('--dendrite-weight-initialization-multiplier', type=float, default=0.01, dest='dendrite_weight_initialization_multiplier')
parser.add_argument('--dendrite-forward-function', type=str, default='tanh', choices=['relu','sigmoid','tanh'], dest='dendrite_forward_function')
parser.add_argument('--dendrite-conversion', type=str, default='All Layers', choices=['Linear Only','All Layers'], dest='dendrite_conversion')
parser.add_argument('--improved-dendritic-optimization', type=str, required=False, default="false", dest='improved_dendritic_optimization')
parser.add_argument('--perforated-ai-login-email', type=str, required=False, default="", dest='perforated_ai_login_email')
parser.add_argument('--perforated-ai-token', type=str, required=False, default="", dest='perforated_ai_token')
parser.add_argument('--split-test', type=str, required=False, default="false", dest='split_test', help="Split test set into validation and test. If false, uses full test set for validation.")


# Use parse_known_args to ignore any extra arguments Edge Impulse might pass
args, unknown = parser.parse_known_args()
if unknown:
    print(f"Note: Ignoring unknown arguments: {unknown}")

# Helper functions (defined before use)
def str2bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "t", "yes", "y")

def parse_augmentation_level(level: str, low_val, high_val, none_val=0):
    """Parse None/Low/High augmentation level to numeric value."""
    level = str(level).strip().lower()
    if level in ('low', 'l'):
        return low_val
    elif level in ('high', 'h'):
        return high_val
    else:  # None, empty, or 0
        return none_val

# Parse augmentation levels from None/Low/High to numeric values
noise_std_val = parse_augmentation_level(args.noise_std, low_val=0.2, high_val=0.45, none_val=0.0)
mask_time_bands_val = parse_augmentation_level(args.mask_time_bands, low_val=1, high_val=3, none_val=0)
mask_freq_bands_val = parse_augmentation_level(args.mask_freq_bands, low_val=1, high_val=3, none_val=0)
warp_time_val = str2bool(args.warp_time)
auto_weight_val = str2bool(args.auto_weight_classes)
softmax_export_val = str2bool(args.softmax_export)

# Parse layer configurations early (needed for data type detection)
layer_configs = []
for i in range(20):
    layer_type = getattr(args, f'layer_{i}_type', 'None')
    if layer_type == 'None':
        break  # Stop at first None layer
    
    config = {'type': layer_type}
    
    if layer_type == 'Dense':
        channels_str = getattr(args, f'layer_{i}_dense_channels', '')
        if channels_str:
            config['channels'] = int(channels_str)
            
    elif layer_type == '1D Convolution/Pool':
        config_str = getattr(args, f'layer_{i}_conv1d_config', '')
        if config_str:
            parts = [p.strip() for p in config_str.split(',')]
            if len(parts) >= 1:
                config['channels'] = int(parts[0].strip())
            if len(parts) >= 2:
                config['kernel_size'] = int(parts[1].strip())
            if len(parts) >= 3:
                config['layer_count'] = int(parts[2].strip())
                
    elif layer_type == '2D Convolution/Pool':
        config_str = getattr(args, f'layer_{i}_conv2d_config', '')
        if config_str:
            parts = [p.strip() for p in config_str.split(',')]
            if len(parts) >= 1:
                config['channels'] = int(parts[0].strip())
            if len(parts) >= 2:
                config['kernel_size'] = int(parts[1].strip())
            if len(parts) >= 3:
                config['layer_count'] = int(parts[2].strip())
                
    elif layer_type == 'Dropout':
        rate_str = getattr(args, f'layer_{i}_dropout_rate', '')
        if rate_str:
            config['rate'] = float(rate_str)

    elif layer_type == 'Reshape':
        columns_str = getattr(args, f'layer_{i}_reshape_columns', '')
        if columns_str:
            config['columns'] = int(columns_str)

    layer_configs.append(config)

os.environ["PAIEMAIL"] = args.perforated_ai_login_email
os.environ["PAITOKEN"] = args.perforated_ai_token

os.makedirs(args.out_directory, exist_ok=True)

# -------------------------
# Load data
# -------------------------
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

print(f"Data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}, X_test={X_test.shape}, Y_test={Y_test.shape}")

# -------------------------
# Determine data type and shape
# -------------------------
input_shape = X_train.shape[1:]
num_samples = X_train.shape[0]

# Extract class labels correctly from Edge Impulse format
# Y_train can be:
# 1. One-hot encoded: (N, num_classes) where each row sums to 1
# 2. Edge Impulse metadata: (N, 4) with [label_index, sample_id, start_ms, end_ms]
# 3. Simple labels: (N,) with class indices

if Y_train.ndim > 1 and Y_train.shape[1] > 1:
    # Check if one-hot encoded by seeing if rows sum to 1 (or close to 1)
    row_sums = Y_train.sum(axis=1)
    is_onehot = np.allclose(row_sums, 1.0) and np.all((Y_train == 0) | (Y_train == 1))
    
    if is_onehot:
        # One-hot encoded format
        num_classes = Y_train.shape[1]
        print(f"Detected one-hot encoded labels with {num_classes} classes")
    else:
        # Assume Edge Impulse metadata format: first column is label_index
        num_classes = len(np.unique(Y_train[:, 0]))
        print(f"Detected Edge Impulse metadata format, extracted {num_classes} classes from first column")
else:
    # Simple label array
    num_classes = len(np.unique(Y_train))
    print(f"Detected simple label array with {num_classes} classes")

print(f"Number of classes: {num_classes}")

# Detect data modality based on input shape
# Edge Impulse data formats:
# - Images (NHWC): (batch, height, width, channels) - typically (N, H, W, 1 or 3)
# - Audio spectrograms (NHWC): (batch, time, freq, 1) - e.g. (N, 49, 13, 1) for MFCC
# - Time-series 1D: (batch, features) - e.g. (N, 39) for spectral features
# - Custom features 1D: (batch, features) - any flattened feature vector

data_type = 'unknown'
if len(input_shape) == 3:
    # 3D input: (H, W, C) - Image or spectrogram
    height, width, channels = input_shape
    if height > 1 and width > 1:
        data_type = '2d_image'
        print(f"Detected 2D image/spectrogram data: {input_shape} (H, W, C)")
    else:
        # Edge case: flattened spectrogram
        data_type = '1d_timeseries'
        print(f"Detected flattened data, treating as 1D: {input_shape}")
elif len(input_shape) == 1:
    # 1D input: (features,) - Check if architecture needs 2D treatment
    # If we have Conv2D/MaxPool2D layers, treat as flattened spectrogram
    has_conv2d = any(layer.get('type') in ['Conv2D', 'MaxPool2D', '2D Convolution/Pool'] for layer in layer_configs)
    if has_conv2d:
        data_type = '2d_image'
        print(f"Detected flattened spectrogram with Conv2D architecture: {input_shape}")
    else:
        data_type = '1d_timeseries'
        print(f"Detected 1D time-series/feature data: {input_shape}")
else:
    raise ValueError(f"Unsupported input shape: {input_shape}. Expected 1D (features,) or 3D (H, W, C)")

print(f"Data type: {data_type}, Number of classes: {num_classes}")

# Split test set in half for test and validation, or use full test set for validation
if str2bool(args.split_test):
    print("Splitting test set into validation and test sets")
    split_idx = len(X_test) // 2
    X_val = X_test[:split_idx]
    Y_val = Y_test[:split_idx]
    X_test = X_test[split_idx:]
    Y_test = Y_test[split_idx:]
else:
    print("Using full test set for validation (no separate test set)")
    X_val = X_test
    Y_val = Y_test
    # Keep X_test and Y_test the same for compatibility, but they won't be used separately

# Convert to tensors on device
X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
Y_val = torch.FloatTensor(Y_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
if str2bool(args.split_test):
    test_dataset = TensorDataset(X_test, Y_test)

if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
if str2bool(args.split_test):
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

# -------------------------
# Generalized Model definition
# -------------------------
class _Reshape1D(nn.Module):
    """Reshapes flat (batch, length) input to (batch, columns, rows) for Conv1d.
    Equivalent to Keras Reshape((rows, columns)) followed by Conv1D which uses columns as channels.

    Keras Reshape((rows, columns)) fills row-major: output[b, t, c] = input[b, t*columns + c]
    PyTorch Conv1d expects (batch, channels, length), so we reshape to (batch, rows, columns)
    matching Keras, then permute to (batch, columns, rows) for Conv1d channel-first format.
    """
    def __init__(self, columns: int, rows: int):
        super().__init__()
        self.columns = columns
        self.rows = rows

    def forward(self, x):
        # Match Keras Reshape((rows, columns)) row-major layout, then permute for Conv1d
        return x.view(x.size(0), self.rows, self.columns).permute(0, 2, 1).contiguous()


class _Reshape2D(nn.Module):
    """Reshapes flat (batch, length) input to (batch, 1, rows, columns) for Conv2d.
    Produces a single-channel 2D spatial tensor (H=rows, W=columns) suitable for Conv2d.
    """
    def __init__(self, columns: int, rows: int):
        super().__init__()
        self.columns = columns
        self.rows = rows

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x.view(x.size(0), 1, self.rows, self.columns)


class AdaptiveClassifier(nn.Module):
    """
    Adaptive classifier with flexible layer-by-layer configuration.
    Supports: Dense, 1D Conv/Pool, 2D Conv/Pool, Flatten, Dropout layers.
    """
    def __init__(self, input_shape, classes, data_type='1d_timeseries', layer_configs=None, noise_std=0.2,
                 mask_time_bands=0, mask_freq_bands=0, warp_time=False):
        super(AdaptiveClassifier, self).__init__()
        
        self.input_shape = input_shape
        self.classes = classes
        self.data_type = data_type
        self.noise_std = noise_std
        self.mask_time_bands = mask_time_bands
        self.mask_freq_bands = mask_freq_bands
        self.warp_time = warp_time
        self.softmax_on = False  # When True, output probabilities; when False, output logits
        self._debug_forward = False  # Set to True to print per-layer stats for one sample
        self.layer_configs = layer_configs or []
        
        # Build layers dynamically from configuration
        self.layers = nn.ModuleList()
        self._constrained_conv_layers = []
        self._flatten_layer_indices = []  # Track flatten layers for post-PerforatedAI fixing
        self._flatten_layer_indices = []  # Track flatten layers for post-PerforatedAI fixing
        
        # Track current shape through the network
        if self.data_type == '2d_image':
            if len(input_shape) == 3:
                self.rows, self.columns, self.channels = input_shape
            elif len(input_shape) == 1:
                # Flattened spectrogram - need to reshape like Keras
                # Keras formula: rows = input_length / columns, columns = 13, channels = 1
                self.channels = 1
                self.columns = 13
                self.rows = int(input_shape[0] / self.columns)
            current_shape = (self.channels, self.rows, self.columns)  # PyTorch format: (C, H, W)
            current_size = None
        elif self.data_type == '1d_timeseries':
            if len(input_shape) == 1:
                self.input_length = input_shape[0]
            else:
                self.input_length = np.prod(input_shape)
            # For 1D data, check if we're starting with conv or dense layers
            # If starting with Dense, use current_size; if Conv1d, use current_shape
            has_initial_conv = False
            if layer_configs:
                first_layer_type = layer_configs[0].get('type', 'None')
                has_initial_conv = first_layer_type == '1D Convolution/Pool'
            
            if has_initial_conv:
                # Will be reshaped in forward() if a Reshape layer precedes, otherwise
                # unsqueeze(1) gives (batch, 1, length) as a flat sequence.
                self.freq_bins = 1
                self.time_steps = self.input_length
                current_shape = (1, self.input_length)  # (channels, length) for 1D conv
                current_size = None
            else:
                self.freq_bins = 1
                self.time_steps = self.input_length
                current_shape = None
                current_size = self.input_length
        
        # Build each layer from configuration
        for i, config in enumerate(self.layer_configs):
            layer_type = config.get('type', 'None')
            
            if layer_type == 'Dense':
                channels = int(config.get('channels', 64))
                # If we're coming from conv layers (current_shape is set), automatically flatten
                if current_shape is not None:
                    self.layers.append(nn.Flatten())
                    self._flatten_layer_indices.append(len(self.layers) - 1)
                    if len(current_shape) == 3:
                        current_size = current_shape[0] * current_shape[1] * current_shape[2]
                    elif len(current_shape) == 2:
                        current_size = current_shape[0] * current_shape[1]
                    current_shape = None
                
                self.layers.append(nn.Linear(current_size, channels))
                self.layers.append(nn.ReLU())
                current_size = channels
                
            elif layer_type == '1D Convolution/Pool':
                channels = int(config.get('channels', 16))
                kernel_size = int(config.get('kernel_size', 3))
                layer_count = int(config.get('layer_count', 1))
                
                # Auto-flatten if coming from 2D conv
                if current_shape is not None and len(current_shape) == 3:
                    # Coming from 2D conv, need to flatten to 1D
                    self.layers.append(nn.Flatten())
                    self._flatten_layer_indices.append(len(self.layers) - 1)
                    current_size = current_shape[0] * current_shape[1] * current_shape[2]
                    current_shape = None
                
                in_channels = current_shape[0] if current_shape else 1
                current_length = current_shape[1] if current_shape else current_size
                
                # Add layer_count conv+relu layers
                # Note: no MaxNorm constraint on Conv1d (Keras Conv1D has no kernel_constraint by default)
                for _ in range(layer_count):
                    conv = nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size//2)
                    self.layers.append(conv)
                    self.layers.append(nn.ReLU())
                    in_channels = channels
                
                # Single pooling at the end.
                # Pad to even length first (matches Keras padding='same') so the
                # exported ONNX contains a plain Pad+MaxPool instead of
                # MaxPool(ceil_mode=1), which onnx2tf wraps in a PartitionedCall
                # that the TFLite PTQ calibrator cannot trace.
                _pad_len = current_length % 2
                if _pad_len:
                    self.layers.append(nn.ConstantPad1d((0, _pad_len), 0))
                self.layers.append(nn.MaxPool1d(2))
                current_length = (current_length + 1) // 2
                
                current_shape = (channels, current_length)
                current_size = None
                
            elif layer_type == '2D Convolution/Pool':
                channels = int(config.get('channels', 16))
                kernel_size = int(config.get('kernel_size', 3))
                layer_count = int(config.get('layer_count', 1))
                
                in_channels = current_shape[0] if current_shape else 1
                current_rows = current_shape[1] if len(current_shape) > 1 else 1
                current_cols = current_shape[2] if len(current_shape) > 2 else 1
                
                # Add layer_count conv+relu layers
                for _ in range(layer_count):
                    # Keras padding='same' with odd kernel size: padding = (kernel_size - 1) // 2
                    padding = (kernel_size - 1) // 2
                    conv = nn.Conv2d(in_channels, channels, kernel_size, padding=padding)
                    self.layers.append(conv)
                    self._constrained_conv_layers.append(conv)
                    self.layers.append(nn.ReLU())
                    in_channels = channels
                
                # Single pooling at the end.
                # Pad to even dimensions first (matches Keras padding='same') so
                # the exported ONNX contains a plain Pad+MaxPool instead of
                # MaxPool(ceil_mode=1), which onnx2tf wraps in a PartitionedCall
                # that the TFLite PTQ calibrator cannot trace.
                _pad_h = current_rows % 2
                _pad_w = current_cols % 2
                if _pad_h or _pad_w:
                    self.layers.append(nn.ZeroPad2d((0, _pad_w, 0, _pad_h)))  # (left, right, top, bottom)
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_rows = (current_rows + 1) // 2
                current_cols = (current_cols + 1) // 2
                
                current_shape = (channels, current_rows, current_cols)
                current_size = None
                
            elif layer_type == 'Dropout':
                rate = float(config.get('rate', 0.5))
                self.layers.append(nn.Dropout(rate))

            elif layer_type == 'Reshape':
                if 'columns' not in config:
                    raise ValueError("Reshape layer requires 'columns' to be specified (e.g. 13 for MFCC, 40 for motion).")
                columns = int(config['columns'])
                # Flatten first if coming from conv layers
                if current_shape is not None:
                    self.layers.append(nn.Flatten())
                    self._flatten_layer_indices.append(len(self.layers) - 1)
                    if len(current_shape) == 3:
                        current_size = current_shape[0] * current_shape[1] * current_shape[2]
                    elif len(current_shape) == 2:
                        current_size = current_shape[0] * current_shape[1]
                    current_shape = None
                rows = current_size // columns
                # Check if the next layer is 2D Conv to decide reshape format
                next_type = self.layer_configs[i + 1].get('type', '') if i + 1 < len(self.layer_configs) else ''
                if next_type == '2D Convolution/Pool':
                    # Produce (batch, 1, rows, columns) for Conv2d
                    self.layers.append(_Reshape2D(columns, rows))
                    self.freq_bins = columns
                    self.time_steps = rows
                    current_shape = (1, rows, columns)  # (C=1, H=rows, W=columns) for Conv2d
                else:
                    # Produce (batch, columns, rows) for Conv1d
                    self.layers.append(_Reshape1D(columns, rows))
                    self.freq_bins = columns
                    self.time_steps = rows
                    current_shape = (columns, rows)  # (channels, length) for Conv1d
                current_size = None

        # Final output layer
        if current_size is None:
            # Need to flatten
            self.layers.append(nn.Flatten())
            if len(current_shape) == 3:
                current_size = current_shape[0] * current_shape[1] * current_shape[2]
            elif len(current_shape) == 2:
                current_size = current_shape[0] * current_shape[1]
        
        self.layers.append(nn.Linear(current_size, classes))

        # Apply Xavier/Glorot uniform initialization to match Keras default
        # PyTorch Conv1d/Linear default is kaiming_uniform(a=sqrt(5)) which gives ~2x smaller
        # weights than Keras's glorot_uniform, causing slower/worse convergence.
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _max_norm_constraint_tensor(self, weight: torch.Tensor, max_value=1.0) -> torch.Tensor:
        if weight.dim() == 3:
            # 1D conv: (out_channels, in_channels, kernel_size)
            norm = weight.norm(2, dim=(1, 2), keepdim=True)
        elif weight.dim() == 4:
            # 2D conv: (out_channels, in_channels, kernel_h, kernel_w)
            norm = weight.norm(2, dim=(1, 2, 3), keepdim=True)
        else:
            return weight
        desired = torch.clamp(norm, max=max_value)
        scale = (desired / (norm + 1e-8))
        return weight * scale
    
    def enforce_max_norm(self, max_value=1.0):
        for module in self._constrained_conv_layers:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    if module.weight.dim() == 3:
                        # 1D conv
                        norm = module.weight.norm(2, dim=(1, 2), keepdim=True)
                    elif module.weight.dim() == 4:
                        # 2D conv
                        norm = module.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                    else:
                        continue
                    desired = torch.clamp(norm, max=max_value)
                    module.weight *= (desired / (norm + 1e-8))
    
    def forward(self, x):
        # Add noise FIRST to match Keras (applied on flattened 1D input before reshape)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Reshape input to match expected format (after noise, like Keras)
        if self.data_type == '2d_image':
            # Reshape from (batch, height, width, channels) to (batch, channels, height, width)
            # OR from flattened (batch, features) to (batch, channels, height, width)
            if x.dim() == 2:
                # Flattened input like (batch, 624) - need to reshape to spectrogram
                # Keras does: GaussianNoise -> Reshape((rows, columns, channels))
                batch_size = x.size(0)
                x = x.view(batch_size, self.rows, self.columns, self.channels)
                # Convert to PyTorch format: (batch, channels, height, width)
                x = x.permute(0, 3, 1, 2)
            elif x.dim() == 4 and x.size(1) != self.channels:
                # NHWC format: (batch, height, width, channels) -> (batch, channels, height, width)
                x = x.permute(0, 3, 1, 2)
        
        # Apply SpecAugment-style masking during training (for spectrograms)
        if self.training and self.data_type == '2d_image':
            # Time masking (mT_num_time_masks, T_time_mask_max_consecutive)
            # Low: 1 mask, max 1 consecutive | High: 3 masks, max 2 consecutive
            if self.mask_time_bands > 0:
                t_max_consecutive = 1 if self.mask_time_bands == 1 else 2
                for _ in range(self.mask_time_bands):
                    time_dim = x.size(-2) if x.dim() == 4 else x.size(-1)
                    if time_dim > t_max_consecutive:
                        t = torch.randint(0, time_dim - t_max_consecutive, (1,)).item()
                        if x.dim() == 4:  # (N, C, H, W) or (N, H, W, C)
                            x[:, :, t:t+t_max_consecutive, :] = 0
                        else:
                            x[:, t:t+t_max_consecutive] = 0
            
            # Frequency masking (mF_num_freq_masks, F_freq_mask_max_consecutive)
            # Low: 1 mask, max 4 consecutive | High: 3 masks, max 4 consecutive
            if self.mask_freq_bands > 0:
                f_max_consecutive = 4  # Same for both Low and High
                for _ in range(self.mask_freq_bands):
                    freq_dim = x.size(-1)
                    if freq_dim > f_max_consecutive:
                        f = torch.randint(0, freq_dim - f_max_consecutive, (1,)).item()
                        if x.dim() == 4:
                            x[:, :, :, f:f+f_max_consecutive] = 0
                        else:
                            x[:, :, f:f+f_max_consecutive] = 0
            
            # Time warping (sparse image warp) - matches TensorFlow SpecAugment
            # Based on tf.contrib.image.sparse_image_warp and dense_image_warp
            if self.warp_time:
                # Only apply if spectrogram is large enough
                if x.dim() == 4 and x.size(-2) > 10:
                    batch_size, channels, time_dim, freq_dim = x.shape
                    
                    # Create a flow field with random time warp
                    # W_time_warp_max_distance = 6 (from Keras implementation)
                    center_time = time_dim // 2
                    warp_amount = torch.randint(-6, 7, (batch_size,), device=x.device).float()
                    
                    # Create a simple 1D warp centered at the middle
                    # This warps the center of the time axis by a random amount
                    time_indices = torch.arange(time_dim, device=x.device, dtype=torch.float32)
                    
                    for b in range(batch_size):
                        # Compute warped coordinates (bilinear interpolation)
                        warp = warp_amount[b]
                        if abs(warp) > 0.1:  # Only warp if significant
                            # Apply Gaussian-like warp centered at middle
                            sigma = time_dim / 4.0
                            gaussian = torch.exp(-((time_indices - center_time) ** 2) / (2 * sigma ** 2))
                            warped_indices = time_indices + warp * gaussian
                            warped_indices = torch.clamp(warped_indices, 0, time_dim - 1)
                            
                            # Bilinear interpolation for warping
                            floor_indices = warped_indices.long()
                            ceil_indices = torch.clamp(floor_indices + 1, 0, time_dim - 1)
                            alpha = warped_indices - floor_indices.float()
                            
                            # Interpolate across time dimension for each channel and freq
                            warped_x = torch.zeros_like(x[b])
                            for t in range(time_dim):
                                floor_idx = floor_indices[t]
                                ceil_idx = ceil_indices[t]
                                a = alpha[t]
                                warped_x[:, t, :] = (1 - a) * x[b, :, floor_idx, :] + a * x[b, :, ceil_idx, :]
                            
                            x[b] = warped_x
        
        # Pass through all configured layers
        if self._debug_forward:
            with torch.no_grad():
                _dbg = x[0:1].detach().float()
                print(f"[FWD] input  shape={tuple(_dbg.shape)} min={_dbg.min().item():.4f} max={_dbg.max().item():.4f} mean={_dbg.mean().item():.4f}")
        for i, layer in enumerate(self.layers):
            # Check actual layer type (may be wrapped by PAIModule)
            actual_layer = layer.main_module if hasattr(layer, 'main_module') else layer
            
            if isinstance(actual_layer, _Reshape1D):
                x = layer(x)
            elif isinstance(actual_layer, _Reshape2D):
                x = layer(x)
            elif isinstance(actual_layer, nn.Conv1d):
                # If no explicit Reshape layer was used, unsqueeze to (batch, 1, length)
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = layer(x)
                # Check for dimension issues after PerforatedAI transforms
                if x.dim() != 3 and x.dim() != 2:
                    raise RuntimeError(
                        f"Conv1d layer at index {i} produced {x.dim()}D output with shape {x.shape}, "
                        f"expected 3D (batch, channels, length) or 2D after PerforatedAI transforms.")
            elif isinstance(actual_layer, nn.Conv2d):
                # Input should already be in correct format from data prep
                x = layer(x)
            elif isinstance(actual_layer, nn.MaxPool1d):
                # Ensure input is 3D for MaxPool1d
                if x.dim() == 2:
                    x = x.unsqueeze(0)  # Add batch dimension if lost
                x = layer(x)
            elif isinstance(actual_layer, nn.MaxPool2d):
                # Ensure input is 4D for MaxPool2d
                if x.dim() == 3:
                    x = x.unsqueeze(0)  # Add batch dimension if lost
                x = layer(x)
            elif i in self._flatten_layer_indices:
                # Use index-based detection instead of isinstance - PerforatedAI transforms break isinstance checks
                # Handle case where batch dimension was lost due to PerforatedAI transforms
                if x.dim() == 2:
                    # Could be (batch, features) already flat, or (channels, length) with batch squeezed
                    # Assume it's (channels, length) and add batch dimension back
                    x = x.unsqueeze(0)  # Now (1, channels, length)
                
                x = layer(x)
                # Ensure output is truly flat
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
            elif isinstance(actual_layer, nn.Linear):
                # Before Linear layer, ensure input is 2D
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = layer(x)
            else:
                x = layer(x)
            if self._debug_forward:
                with torch.no_grad():
                    _dbg = x[0:1].detach().float()
                    _lname = type(actual_layer).__name__
                    print(f"[FWD] layer[{i}] {_lname:20s} shape={tuple(_dbg.shape)} min={_dbg.min().item():.4f} max={_dbg.max().item():.4f} mean={_dbg.mean().item():.4f} unique={_dbg.unique().numel()}")
        
        # Apply softmax only in export mode to output probabilities for Edge Impulse
        # During training, output raw logits for CrossEntropyLoss compatibility
        if self._debug_forward:
            with torch.no_grad():
                _dbg = x[0:1].detach().float()
                print(f"[FWD] output shape={tuple(_dbg.shape)} min={_dbg.min().item():.4f} max={_dbg.max().item():.4f} mean={_dbg.mean().item():.4f}")
        if self.softmax_on:
            x = torch.nn.functional.softmax(x, dim=1)
        
        return x

# -------------------------
# Main training loop
# -------------------------
def main(config):
    # config is the argparse.Namespace (args)
    # PerforatedAI config
    if args.improvement_threshold == 'high':
        thresh = [0.01, 0.001, 0.0001, 0]
    elif args.improvement_threshold == 'medium':
        thresh = [0.001, 0.0001, 0]
    else:
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)
    GPA.pc.set_candidate_weight_initialization_multiplier(args.dendrite_weight_initialization_multiplier)
    if args.dendrite_forward_function == 'sigmoid':
        pai_forward_function = torch.sigmoid
    elif args.dendrite_forward_function == 'relu':
        pai_forward_function = torch.relu
    else:
        pai_forward_function = torch.tanh
    GPA.pc.set_pai_forward_function(pai_forward_function)

    if args.dendrite_conversion == 'All Layers':
        GPA.pc.set_modules_to_perforate([nn.Conv2d, nn.Linear])
        GPA.pc.set_modules_to_track([])
    else:
        GPA.pc.set_modules_to_perforate([nn.Linear])
        GPA.pc.set_modules_to_track([nn.Conv2d])

    GPA.pc.set_max_dendrites(args.max_dendrites if str2bool(args.dendritic_optimization) else 0)
    GPA.pc.set_perforated_backpropagation(str2bool(args.improved_dendritic_optimization))
    GPA.pc.set_dendrite_update_mode(True)
    GPA.pc.set_initial_correlation_batches(40)
    GPA.pc.set_max_dendrite_tries(2)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    print(f"\nUsing {len(layer_configs)} layer configurations:", flush=True)
    for i, cfg in enumerate(layer_configs):
        print(f"  Layer {i}: {cfg}", flush=True)
    
    # Instantiate adaptive model based on detected data type
    model = AdaptiveClassifier(
        input_shape=input_shape,
        classes=num_classes,
        data_type=data_type,
        layer_configs=layer_configs,
        noise_std=noise_std_val,
        mask_time_bands=mask_time_bands_val,
        mask_freq_bands=mask_freq_bands_val,
        warp_time=warp_time_val
    ).to(device)

    GPA.pc.set_testing_dendrite_capacity(False)

    if args.switch_speed == 'fast':
        GPA.pc.set_n_epochs_to_switch(10)
    elif args.switch_speed == 'medium':
        GPA.pc.set_n_epochs_to_switch(25)
    elif args.switch_speed == 'slow':
        GPA.pc.set_n_epochs_to_switch(100)
    elif args.switch_speed == 'extra-slow':
        GPA.pc.set_n_epochs_to_switch(250)

    GPA.pc.set_verbose(False)
    GPA.pc.set_silent(True)

    model = UPA.perforate_model(model)
    print(model)
    print(f"Total parameters (before PAI): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
   
    # Set output dimensions for Conv1d layers if they are being optimized with dendrites
    for layer in model.layers:
        if isinstance(layer, nn.Conv1d):
            # Check if it's a PAIModule (PerforatedAI wrapped layer) by type name
            if type(layer).__name__ == 'PAINeuronModule' and isinstance(layer.main_module, nn.Conv1d):
                layer.set_this_output_dimensions([-1, 0, -1])
    print(f"Total parameters (after PAI): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params':model.parameters(), 'lr':args.learning_rate, 'betas':(0.9, 0.999), 'eps':1e-7}  # eps=1e-7 matches Keras legacy Adam default (PyTorch default is 1e-8)
    schedArgs = {'mode':'max', 'patience': int(GPA.pc.get_n_epochs_to_switch()*0.75)}
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    optimizer = torch.optim.Adam(**optimArgs)

    # Compute class weights if auto_weight_classes is enabled
    class_weights = None
    if auto_weight_val:
        # Extract labels from Y_train
        if Y_train.dim() > 1:
            # Check if one-hot encoded
            row_sums = Y_train.sum(dim=1)
            is_onehot = torch.allclose(row_sums, torch.ones_like(row_sums)) and torch.all((Y_train == 0) | (Y_train == 1))
            
            if is_onehot:
                train_labels = torch.argmax(Y_train, dim=1)
            else:
                # Assume first column is label index
                train_labels = Y_train[:, 0].long()
        else:
            train_labels = Y_train.long()
        
        # Compute class weights inversely proportional to class frequencies
        class_counts = torch.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Handle Edge Impulse label format
            if labels.dim() > 1:
                if labels.size(1) == 4:
                    # Edge Impulse 4-column format: first column is label_index
                    labels = labels[:, 0].long()
                elif labels.size(1) > 1:
                    # One-hot encoded: convert to class indices
                    labels = torch.argmax(labels, dim=1)
            else:
                # Simple label array
                labels = labels.long()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if hasattr(model, 'enforce_max_norm'):
                model.enforce_max_norm(max_value=1.0)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total

    def test(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predicted = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Handle Edge Impulse label format
                if labels.dim() > 1:
                    if labels.size(1) == 4:
                        # Edge Impulse 4-column format: first column is label_index
                        labels = labels[:, 0].long()
                    elif labels.size(1) > 1:
                        # One-hot encoded: convert to class indices
                        labels = torch.argmax(labels, dim=1)
                else:
                    # Simple label array
                    labels = labels.long()
                
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predicted.append(predicted.cpu())
        return running_loss / total, correct / total

    # Training loop
    first_val_acc = 0
    first_test_acc = 0
    max_val_acc = 0
    max_test_acc = 0
    first_param_count = UPA.count_params(model)

    epoch = -1
    while True:
        epoch += 1
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test(model, val_loader, criterion, device)
        if str2bool(args.split_test):
            test_loss, test_acc = test(model, test_loader, criterion, device)
        else:
            test_acc = val_acc  # Use validation as test when not split
        if(val_acc > max_val_acc):
            max_val_acc = val_acc
            max_test_acc = test_acc
        GPA.pai_tracker.add_extra_score(train_acc, 'Train')
        if str2bool(args.split_test):
            GPA.pai_tracker.add_extra_score(test_acc, 'Test')
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        model.to(device)
        if training_complete:  # Stop at 100 epochs to match Keras
            break
        elif restructured:
            if first_test_acc == 0:
                first_val_acc = val_acc
                first_test_acc = test_acc
            print('Restructured dendritic architecture')
            optimArgs = {'params':model.parameters(), 'lr':args.learning_rate, 'betas':(0.9, 0.999), 'eps':1e-7}
            schedArgs = {'mode':'max', 'patience': int(GPA.pc.get_n_epochs_to_switch()*0.75)}
            optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.2e} | Dendrite Count: {GPA.pai_tracker.member_vars.get("num_dendrites_added", "N/A")}')
        
    if str2bool(args.split_test):
        test_loss, test_acc = test(model, test_loader, criterion, device)
    else:
        test_acc = val_acc  # Use validation as test when not split

    """
    if str2bool(args.dendritic_optimization):
        print(f'First architecture: Val Acc: {first_val_acc:.4f}, Test Acc: {first_test_acc:.4f}, params: {first_param_count}')
        print(f'Best architecture: Val Acc: {max_val_acc:.4f}, Test Acc: {max_test_acc:.4f}, params: {UPA.count_params(model)} Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')
    else:
        print(f'Final architecture: Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, params: {UPA.count_params(model)} Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')
    """
    model = model.cpu()
    model.eval()
    
    from perforatedai import blockwise_perforatedai as BPA
    from perforatedai import clean_perforatedai as CPA
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = BPA.blockwise_network(model)
        model  = CPA.refresh_net(model)
    

    # Test forward pass and capture activation statistics
    model.eval()
    
    # Fix padding for conv layers if needed
    for layer in model.layers:
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            if hasattr(layer, 'layer_array'):
                for conv in layer.layer_array:
                    if hasattr(conv, 'padding') and conv.padding == 'same':
                        if isinstance(conv.kernel_size, tuple):
                            padding = tuple((k - 1) // 2 for k in conv.kernel_size)
                        else:
                            padding = (conv.kernel_size - 1) // 2
                        conv.padding = padding

    # Model already in eval mode and on CPU from baseline evaluation
    # Control softmax in exported models: only enable if softmax_export is True
    # Default (False) exports raw logits - argmax still works, avoids quantization issues
    model.softmax_on = softmax_export_val
    
    # Export ONNX with batch size 1 (important for Edge Impulse)
    # Calculate total input size for ONNX export
    if data_type == '2d_image':
        # For images/spectrograms: flatten (H, W, C) to single dimension
        onnx_input_size = int(np.prod(input_shape))
    else:
        # For 1D data: use shape as-is
        onnx_input_size = int(np.prod(input_shape))
    
    onnx_path = os.path.join(args.out_directory, 'model.onnx')
    dummy_input = torch.randn((1, onnx_input_size))
    # Use legacy ONNX exporter to avoid torch.export issues with dynamic shapes
    # Dynamic axes allow variable batch size for validation
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    # Export the full model including PAI dendritic modules.
    # PAI's forward pass is purely standard ops (Conv, Mul, Add), so
    # torch.onnx.export traces it cleanly into ONNX initializers without
    # producing any PartitionedCall in the TF graph.
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=dynamic_axes,
                      dynamo=False)

    print("Exported ONNX to", onnx_path)
    
    import onnxruntime as ort

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    def test_onnx(loader, dataset_name):
        correct = 0
        total = 0
        
        # Check if model supports dynamic batch size
        input_shape = ort_session.get_inputs()[0].shape
        supports_batch = input_shape[0] == 'batch_size' or input_shape[0] is None or isinstance(input_shape[0], str)
        
        for inputs, labels in loader:
            # Convert to numpy and flatten
            inputs_np = inputs.cpu().numpy()
            batch_size = inputs_np.shape[0]
            inputs_flat = inputs_np.reshape(batch_size, -1)
            
            # Handle Edge Impulse label format
            labels = labels.cpu().numpy()
            if labels.ndim > 1:
                if labels.shape[1] == 4:
                    labels = labels[:, 0].astype(int)
                elif labels.shape[1] > 1:
                    labels = np.argmax(labels, axis=1)
            else:
                labels = labels.astype(int)
            
            if supports_batch:
                # Process entire batch at once
                ort_inputs = {ort_session.get_inputs()[0].name: inputs_flat}
                ort_outputs = ort_session.run(None, ort_inputs)
                predictions = ort_outputs[0]
                predicted = np.argmax(predictions, axis=1)
                correct += (predicted == labels).sum()
            else:
                # Process one sample at a time (batch size = 1)
                for i in range(batch_size):
                    sample = inputs_flat[i:i+1]
                    ort_inputs = {ort_session.get_inputs()[0].name: sample}
                    ort_outputs = ort_session.run(None, ort_inputs)
                    prediction = ort_outputs[0]
                    predicted = np.argmax(prediction, axis=1)[0]
                    correct += (predicted == labels[i])
            
            total += batch_size
        
        accuracy = correct / total
        print(f"{dataset_name} Accuracy (ONNX): {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    # Test on validation and test sets
    val_acc_onnx = test_onnx(val_loader, "Validation")
    if str2bool(args.split_test):
        test_acc_onnx = test_onnx(test_loader, "Test")

    import warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TFLite conversion is CPU-only; avoids driver mismatch crash
    import tensorflow as tf
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    import subprocess
    
    # EI-compatible conversion helpers (mirrors ei_tensorflow.conversion, adapted for SavedModel)
    def convert_float32(saved_model_dir, dir_path, filename):
        print('Converting TFLite float32 model...', flush=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tflite_model = converter.convert()
        with open(os.path.join(dir_path, filename), 'wb') as f:
            f.write(tflite_model)
        print(f"✓ Created float32 TFLite model: {os.path.join(dir_path, filename)}")
        return tflite_model

    _cal_call_count = [0]
    _cal_first_shape = [None]
    _cal_first_min   = [None]
    _cal_first_max   = [None]

    def convert_int8_io_int8(saved_model_dir, dataset_generator, dir_path, filename):
        print('Converting TFLite int8 quantized model...', flush=True)

        # Per https://ai.google.dev/edge/litert/performance/post_training_quantization
        # "Integer with float fallback (using default float input/output)":
        #   converter = TFLiteConverter.from_saved_model(saved_model_dir)
        #   converter.optimizations = [Optimize.DEFAULT]
        #   converter.representative_dataset = representative_dataset
        #   tflite_quant_model = converter.convert()
        # No target_spec, no SELECT_TF_OPS, no inference_input/output_type, no from_concrete_functions.
        converter_quantize = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quantize.representative_dataset = dataset_generator
        # NOTE: stdout/stderr NOT suppressed — we need to see fully_quantize: and any warnings
        tflite_quant_model = converter_quantize.convert()
        print(f"  Calibration samples fed: {_cal_call_count[0]}" if _cal_call_count[0] > 0
              else "  WARNING: representative_dataset was never called (calibration skipped)")
        if _cal_call_count[0] > 0:
            print(f"  First cal sample: shape={_cal_first_shape[0]}, range=[{_cal_first_min[0]:.4f}, {_cal_first_max[0]:.4f}]")

        # Dump ALL tensor quantization params from the resulting model
        _diag_interp = tf.lite.Interpreter(model_content=tflite_quant_model)
        try:
            _diag_interp.allocate_tensors()
        except (RuntimeError, Exception) as _xnn_e:
            if 'XNNPACK' in str(_xnn_e):
                _diag_interp = tf.lite.Interpreter(
                    model_content=tflite_quant_model,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
                _diag_interp.allocate_tensors()
            else:
                raise
        _all_td = _diag_interp.get_tensor_details()
        _all_ops = _diag_interp._get_ops_details()
        print(f"  INT8 model: {len(_all_td)} tensors, {len(_all_ops)} ops")
        print("  All tensor quant params:")
        for _td in _all_td:
            _sc = _td['quantization_parameters']['scales']
            _zp = _td['quantization_parameters']['zero_points']
            if len(_sc) == 0:
                _sc_str = 'none'
            elif len(_sc) == 1:
                _sc_str = f"{float(_sc[0]):.4e}"
            else:
                _sc_str = f"[{len(_sc)}ch min={float(_sc.min()):.3e} max={float(_sc.max()):.3e}]"
            _zp_str = (f"{int(_zp[0])}" if len(_zp) == 1
                       else f"[{len(_zp)}ch]" if len(_zp) > 1 else 'none')
            _flag = ' *** SENTINEL ***' if len(_sc) > 0 and float(np.max(np.abs(_sc))) > 1e10 else ''
            print(f"    [{_td['index']:2d}] {_td['name'][:45]:<45} "
                  f"{str(_td['dtype']):<25} scale={_sc_str:<35} zp={_zp_str}{_flag}")
        del _diag_interp

        with open(os.path.join(dir_path, filename), 'wb') as f:
            f.write(tflite_quant_model)
        print(f"✓ Created int8 quantized model: {os.path.join(dir_path, filename)}")
        return tflite_quant_model

    # Calibration data generator: handles NCHW→NHWC (Conv2D) and NCL→NLC (Conv1D)
    # transpositions because onnx2tf produces channel-last SavedModels.
    # Yields a plain list per the official litert post_training_quantization docs.
    _cal_input_shape = [None]
    def representative_dataset():
        cal_shape = _cal_input_shape[0]
        for batch_idx, (data, _) in enumerate(train_loader):
            if batch_idx >= 100:
                break
            data_np = data.cpu().numpy()
            for i in range(data_np.shape[0]):
                sample = data_np[i:i+1]
                if cal_shape is not None and len(cal_shape) == 4:
                    sample = np.transpose(sample, (0, 2, 3, 1))  # NCHW → NHWC
                elif cal_shape is not None and len(cal_shape) == 3:
                    sample = np.transpose(sample, (0, 2, 1))     # NCL → NLC
                else:
                    sample = sample.reshape(1, -1)
                if _cal_call_count[0] == 0:
                    _cal_first_shape[0] = sample.shape
                    _cal_first_min[0]   = float(sample.min())
                    _cal_first_max[0]   = float(sample.max())
                _cal_call_count[0] += 1
                yield [sample.astype(np.float32)]

    # Simplify ONNX model first to fold standalone Constant ops into consuming ops.
    # This prevents onnx2tf from generating arith.constant tensors that TFLite's
    # INT8 quantizer can't scale, which otherwise causes fully_quantize:0 and
    # sentinel scales on internal tensors.
    import onnx, onnxsim
    _onnx_model = onnx.load(onnx_path)
    _const_before = sum(1 for n in _onnx_model.graph.node if n.op_type == 'Constant')
    print(f"ONNX Constant nodes before simplification: {_const_before}")
    from collections import Counter as _Counter
    _ops_before = _Counter(n.op_type for n in _onnx_model.graph.node)
    print(f"ONNX op types before: {dict(_ops_before)}")
    _simplified, _check = onnxsim.simplify(_onnx_model)
    _const_after = sum(1 for n in _simplified.graph.node if n.op_type == 'Constant')
    _ops_after = _Counter(n.op_type for n in _simplified.graph.node)
    print(f"ONNX simplified (constant folding OK: {_check}), Constant nodes after: {_const_after}")
    print(f"ONNX op types after: {dict(_ops_after)}")
    if _const_after > 0:
        _const_names = [n.output[0] for n in _simplified.graph.node if n.op_type == 'Constant']
        print(f"  Remaining Constant outputs: {_const_names[:20]}")

    onnx_path_for_tf = os.path.join(args.out_directory, 'model_simplified.onnx')
    onnx.save(_simplified, onnx_path_for_tf)

    # Convert ONNX to TF SavedModel using onnx2tf
    saved_model_dir = os.path.join(args.out_directory, 'saved_model_temp')
    print(f"Converting ONNX to TensorFlow SavedModel using onnx2tf...")
    result = subprocess.run(
        [sys.executable, "-m", "onnx2tf", "-i", onnx_path_for_tf, "-o", saved_model_dir, "-osd"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"onnx2tf stderr: {result.stderr}")
        raise Exception(f"onnx2tf conversion failed with return code {result.returncode}")
    print(f"Converted ONNX to TF SavedModel at {saved_model_dir}")

    tflite_float_model = convert_float32(saved_model_dir, args.out_directory, 'model.tflite')
    tflite_float_path = os.path.join(args.out_directory, 'model.tflite')

    # Populate calibration shape AND input key from float32 model
    _cal_interp = tf.lite.Interpreter(model_content=tflite_float_model)
    _cal_interp.allocate_tensors()
    _cal_input_shape[0] = _cal_interp.get_input_details()[0]['shape']
    print(f"Calibration input shape: {_cal_input_shape[0]}")
    del _cal_interp

    # Sanity-check: run 3 training samples through float32 TFLite to verify the
    # SavedModel conversion is numerically sound before committing to INT8 cal.
    _sane_interp = tf.lite.Interpreter(model_content=tflite_float_model)
    _sane_interp.allocate_tensors()
    _sane_in  = _sane_interp.get_input_details()[0]
    _sane_out = _sane_interp.get_output_details()[0]
    _sane_n = 0
    print("Float32 TFLite sanity check (first 3 training samples):")
    for _sane_b, (_sane_x, _sane_y) in enumerate(train_loader):
        if _sane_b >= 1:
            break
        for _sane_i in range(min(3, _sane_x.shape[0])):
            _ss = _sane_x[_sane_i:_sane_i+1].cpu().numpy()
            _cal_sh = _cal_input_shape[0]
            if len(_cal_sh) == 4:
                _ss = np.transpose(_ss, (0, 2, 3, 1))
            elif len(_cal_sh) == 3:
                _ss = np.transpose(_ss, (0, 2, 1))
            else:
                _ss = _ss.reshape(1, -1)
            _sane_interp.set_tensor(_sane_in['index'], _ss.astype(np.float32))
            _sane_interp.invoke()
            _so = _sane_interp.get_tensor(_sane_out['index'])
            _true_label = int(_sane_y[_sane_i].cpu().numpy().argmax()) if _sane_y[_sane_i].ndim > 0 and _sane_y[_sane_i].shape[-1] > 1 else int(_sane_y[_sane_i].cpu().numpy().flat[0])
            print(f"  [{_sane_i}] input range [{_ss.min():.4f}, {_ss.max():.4f}] "
                  f"→ logits {_so[0]} pred={int(np.argmax(_so[0]))} true={_true_label}")
    del _sane_interp

    # Define test function for TFLite models
    def test_tflite(model_path, loader, dataset_name):
        with contextlib.redirect_stderr(io.StringIO()):
            interpreter = tf.lite.Interpreter(model_path=model_path)
        
        try:
            interpreter.allocate_tensors()
        except (RuntimeError, Exception) as alloc_error:
            if 'XNNPACK' in str(alloc_error):
                # XNNPACK can't handle this INT8 subgraph (common with Conv2D).
                # Retry using built-in CPU kernels only (no default delegates).
                with contextlib.redirect_stderr(io.StringIO()):
                    interpreter = tf.lite.Interpreter(
                        model_path=model_path,
                        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
                interpreter.allocate_tensors()
            else:
                print(f"\n✗ TENSOR ALLOCATION FAILED for {model_path}")
                print(f"Error: {alloc_error}")
                raise
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        correct = 0
        total = 0
        all_preds = []
        
        for inputs, labels in loader:
            inputs_np = inputs.cpu().numpy()  # PyTorch: (B, C, H, W) or (B, features)
            batch_size = inputs_np.shape[0]
            
            # Handle Edge Impulse label format
            labels = labels.cpu().numpy()
            if labels.ndim > 1:
                if labels.shape[1] == 4:
                    labels = labels[:, 0].astype(int)
                elif labels.shape[1] > 1:
                    labels = np.argmax(labels, axis=1)
            else:
                labels = labels.astype(int)
            
            # Process one sample at a time
            for i in range(batch_size):
                sample = inputs_np[i:i+1]
                if len(input_details[0]['shape']) == 4:
                    # Conv2D: onnx2tf model expects NHWC; PyTorch data is NCHW
                    sample = np.transpose(sample, (0, 2, 3, 1))
                elif len(input_details[0]['shape']) == 3:
                    # Conv1D: onnx2tf model expects NLC; PyTorch data is NCL
                    sample = np.transpose(sample, (0, 2, 1))
                else:
                    sample = sample.reshape(1, -1)
                sample = sample.astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], sample)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                predicted = np.argmax(output[0])
                all_preds.append(int(predicted))
                correct += (predicted == labels[i])
                total += 1
        
        accuracy = correct / total
        print(f"{dataset_name}: {accuracy:.4f} ({correct}/{total})")
        if accuracy < 0.5:
            # Show prediction distribution to distinguish quantization collapse vs random noise
            import collections
            pred_counts = collections.Counter(all_preds)
            print(f"  Prediction distribution (top 5): {pred_counts.most_common(5)}")
            # Check for tensors with sentinel scales (never calibrated)
            all_tensor_details = interpreter.get_tensor_details()
            bad_tensors = [
                (d['index'], d['name'][:50], d['dtype'], d['quantization_parameters']['scales'])
                for d in all_tensor_details
                if len(d['quantization_parameters']['scales']) > 0
                and float(np.max(np.abs(d['quantization_parameters']['scales']))) > 1e10
            ]
            if bad_tensors:
                print(f"  WARNING: {len(bad_tensors)}/{len(all_tensor_details)} tensors with sentinel/giant scales:")
                for idx, name, dtype, scales in bad_tensors:
                    print(f"    [{idx}] {name} ({dtype}): scale={scales}")
                # Dump op graph with tensor indices so we can see if bad tensors are active
                print("  Op graph (!= bad tensor with sentinel/giant scale):")
                _bad_idx_set = {idx for idx, _, _, _ in bad_tensors}
                _dtype_map = {d['index']: d['dtype'] for d in all_tensor_details}
                _scale_map = {d['index']: d['quantization_parameters']['scales'] for d in all_tensor_details}
                def _fmt_t(t):
                    dt = _dtype_map.get(t, None)
                    dname = ('f32' if dt == np.float32 else
                             'i8'  if dt == np.int8   else
                             'i32' if dt == np.int32  else '?')
                    sc = _scale_map.get(t, [])
                    sc_str = f"{float(sc[0]):.3e}" if len(sc) == 1 else f"[{len(sc)}ch]" if len(sc) > 1 else 'noscale'
                    mark = '!' if t in _bad_idx_set else ''
                    return f"{mark}t{t}:{dname}(s={sc_str})"
                for op_idx, op in enumerate(interpreter._get_ops_details()):
                    ins  = [_fmt_t(i) for i in op['inputs']  if i >= 0]
                    outs = [_fmt_t(o) for o in op['outputs'] if o >= 0]
                    print(f"    op{op_idx} {op['op_name']}: [{', '.join(ins)}] -> [{', '.join(outs)}]")

            # Run one sample through the model and dump every intermediate tensor value.
            # Uses experimental_preserve_all_tensors so internal buffers aren't freed.
            print("  --- Intermediate tensor value dump (sample 0) ---")
            try:
                _dbg_interp = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_preserve_all_tensors=True)
                try:
                    _dbg_interp.allocate_tensors()
                except (RuntimeError, Exception) as _dae:
                    if 'XNNPACK' in str(_dae):
                        _dbg_interp = tf.lite.Interpreter(
                            model_path=model_path,
                            experimental_preserve_all_tensors=True,
                            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
                        _dbg_interp.allocate_tensors()
                    else:
                        raise
                _dbg_in = _dbg_interp.get_input_details()[0]
                for _dbg_inputs, _dbg_labels in loader:
                    _dbg_sample = _dbg_inputs[0:1].cpu().numpy()
                    if len(_dbg_in['shape']) == 4:
                        _dbg_sample = np.transpose(_dbg_sample, (0, 2, 3, 1))
                    elif len(_dbg_in['shape']) == 3:
                        _dbg_sample = np.transpose(_dbg_sample, (0, 2, 1))
                    else:
                        _dbg_sample = _dbg_sample.reshape(1, -1)
                    _dbg_interp.set_tensor(_dbg_in['index'], _dbg_sample.astype(np.float32))
                    _dbg_interp.invoke()
                    _dbg_td = _dbg_interp.get_tensor_details()
                    print(f"  All {len(_dbg_td)} tensors after invoke:")
                    for _t in _dbg_td:
                        _val = _dbg_interp.get_tensor(_t['index'])
                        _unique = len(np.unique(_val))
                        _vmin = float(_val.min())
                        _vmax = float(_val.max())
                        _vmean = float(_val.mean())
                        _flag2 = ' <-- ALL SAME' if _unique == 1 else (' <-- ALL ZERO' if _vmax == 0 and _vmin == 0 else '')
                        print(f"    [{_t['index']:2d}] {_t['name'][:42]:<42} shape={str(_val.shape):<18} "
                              f"dtype={str(_t['dtype']):<10} "
                              f"min={_vmin:+.4e} max={_vmax:+.4e} mean={_vmean:+.4e} "
                              f"unique={_unique}{_flag2}")
                    break
                del _dbg_interp
            except Exception as _dbg_e:
                print(f"  Intermediate tensor dump failed: {_dbg_e}")
            if not bad_tensors:
                print(f"  All {len(all_tensor_details)} tensors have valid scales (calibration OK)")
        return accuracy
    
    # Test on full X_split_test dataset
    X_test_full = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
    Y_test_full = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))
    X_test_full = torch.FloatTensor(X_test_full)
    Y_test_full = torch.FloatTensor(Y_test_full)
    full_test_dataset = TensorDataset(X_test_full, Y_test_full)
    full_test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    full_test_acc_float = test_tflite(tflite_float_path, full_test_loader, "Float32 TFLite")

    full_test_acc_int8 = None
    _cal_call_count[0] = 0  # reset counter before INT8 conversion
    tflite_int8_model = convert_int8_io_int8(saved_model_dir, representative_dataset,
                                              args.out_directory, 'model_quantized_int8_io.tflite')
    if tflite_int8_model is not None:
        tflite_int8_path = os.path.join(args.out_directory, 'model_quantized_int8_io.tflite')
        # Print PyTorch ground-truth intermediate values for the first test sample
        print("\n" + "="*60)
        print("PYTORCH FORWARD PASS (ground truth, first test sample)")
        print("="*60)
        _debug_sample = X_test_full[0:1].to(next(model.parameters()).device)
        model.eval()
        model._debug_forward = True
        with torch.no_grad():
            model(_debug_sample)
        model._debug_forward = False
        print("="*60 + "\n")
        full_test_acc_int8 = test_tflite(tflite_int8_path, full_test_loader, "INT8 TFLite")
    
    # Print improvement percentages then final comparison
    if str2bool(args.dendritic_optimization):
        print('Reduction in misclassifications because of dendrites')
        if first_val_acc >= 1.0:
            print(f'Validation: N/A (already perfect accuracy)')
        else:
            val_reduction = 100.0 * ((max_val_acc - first_val_acc) / (1 - first_val_acc))
            print(f'Validation: {val_reduction:.2f}%')
        if first_test_acc >= 1.0:
            print(f'Test: N/A (already perfect accuracy)')
        else:
            test_reduction = 100.0 * ((max_test_acc - first_test_acc) / (1 - first_test_acc))
            print(f'Test: {test_reduction:.2f}%')

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    if str2bool(args.dendritic_optimization):
        if str2bool(args.split_test):
            print(f"PyTorch (pre-dendrite)  - Val: {first_val_acc:.4f}, Test: {first_test_acc:.4f}")
        else:
            print(f"PyTorch (pre-dendrite)  - Val: {first_val_acc:.4f}")
    if str2bool(args.split_test):
        print(f"PyTorch (post-dendrite) - Val: {max_val_acc:.4f}, Test: {max_test_acc:.4f}")
        print(f"ONNX                   - Val: {val_acc_onnx:.4f}, Test: {test_acc_onnx:.4f}")
    else:
        print(f"PyTorch (post-dendrite) - Val: {max_val_acc:.4f}")
        print(f"ONNX                   - Val: {val_acc_onnx:.4f}")
    print(f"TFLite Float32          - Full: {full_test_acc_float:.4f}")
    if full_test_acc_int8 is not None:
        print(f"TFLite INT8             - Full: {full_test_acc_int8:.4f}")
    else:
        print(f"TFLite INT8             - Full: FAILED (see error above)")

    print("="*60 + "\n")


if __name__ == "__main__":
    main(args)
