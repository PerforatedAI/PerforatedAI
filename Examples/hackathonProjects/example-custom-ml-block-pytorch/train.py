#!/usr/bin/env python3
# Generalized train.py for Edge Impulse custom learning blocks with PerforatedAI

import sys
print("This is version 3.57", flush=True)
print("="*60, flush=True)
sys.stdout.flush()
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
parser.add_argument('--noise-std', type=str, default='None', help="Gaussian noise augmentation (None/Low/High)", dest='noise_std')

# Data augmentation parameters (SpecAugment for spectrograms)
parser.add_argument('--mask-time-bands', type=str, default='None', help="Time band masking (None/Low/High)", dest='mask_time_bands')
parser.add_argument('--mask-freq-bands', type=str, default='None', help="Frequency band masking (None/Low/High)", dest='mask_freq_bands')
parser.add_argument('--warp-time', type=str, default='false', help="Enable time warping", dest='warp_time')
parser.add_argument('--auto-weight-classes', type=str, default='false', help="Auto weight classes", dest='auto_weight_classes')

# Layer-by-layer architecture parameters (layers 0-19)
for i in range(20):
    parser.add_argument(f'--layer-{i}-type', type=str, default='None', dest=f'layer_{i}_type',
                        choices=['None', 'Dense', '1D Convolution/Pool', '2D Convolution/Pool', 'Dropout'],
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

# PerforatedAI dendritic optimization parameters
parser.add_argument('--dendritic-optimization', type=str, required=False, default="true", dest='dendritic_optimization')
parser.add_argument('--switch-speed', type=str, default='extra-slow', help="speed to switch", choices=['extra-slow', 'slow', 'medium', 'fast'], dest='switch_speed')
parser.add_argument('--max-dendrites', type=int, default=3, dest='max_dendrites')
parser.add_argument('--improvement-threshold', type=str, default='medium', choices=['high', 'medium', 'low'], dest='improvement_threshold')
parser.add_argument('--dendrite-weight-initialization-multiplier', type=float, default=0.01, dest='dendrite_weight_initialization_multiplier')
parser.add_argument('--dendrite-forward-function', type=str, default='tanh', choices=['relu','sigmoid','tanh'], dest='dendrite_forward_function')
parser.add_argument('--dendrite-conversion', type=str, default='All Layers', choices=['Linear Only','All Layers'], dest='dendrite_conversion')
parser.add_argument('--improved-dendritic-optimization', type=str, required=False, default="false", dest='improved_dendritic_optimization')
parser.add_argument('--perforated-ai-token', type=str, required=False, default="", dest='perforated_ai_token')
parser.add_argument('--confirm-quant-score', type=str, required=False, default="false", dest='confirm_quant_score', help="Test int8 quantization accuracy locally")

# Use parse_known_args to ignore any extra arguments Edge Impulse might pass
args, unknown = parser.parse_known_args()
if unknown:
    print(f"Note: Ignoring unknown arguments: {unknown}")

# Print all parsed arguments for debugging
print("\n=== Parsed Arguments ===", flush=True)
for arg, value in vars(args).items():
    print(f"{arg}: {value}", flush=True)
print("========================\n", flush=True)

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

print(f"Parsed augmentation settings:")
print(f"  Noise std: {args.noise_std} -> {noise_std_val}")
print(f"  Mask time bands: {args.mask_time_bands} -> {mask_time_bands_val}")
print(f"  Mask freq bands: {args.mask_freq_bands} -> {mask_freq_bands_val}")
print(f"  Warp time: {args.warp_time} -> {warp_time_val}")
print(f"  Auto weight classes: {args.auto_weight_classes} -> {auto_weight_val}\n")

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
    
    layer_configs.append(config)

print(f"Parsed {len(layer_configs)} layer configurations")

os.environ["PAIEMAIL"] = "user@edgeimpulse.com"
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
# Y_train format: (N, 4) with columns [label_index, sample_id, start_ms, end_ms]
# OR for one-hot: (N, num_classes)
if Y_train.ndim > 1 and Y_train.shape[1] > 1:
    # Check if it's the 4-column format or one-hot encoded
    if Y_train.shape[1] == 4:
        # Edge Impulse 4-column format: first column is label_index
        num_classes = len(np.unique(Y_train[:, 0]))
    else:
        # One-hot encoded format
        num_classes = Y_train.shape[1]
else:
    # Simple label array
    num_classes = len(np.unique(Y_train))

print(f"Number of classes: {num_classes}")
print(f"Y_train format: {Y_train.shape}, Y_train sample: {Y_train[:3]}")
print(f"Unique labels in Y_train: {np.unique(Y_train[:, 0]) if Y_train.ndim > 1 and Y_train.shape[1] >= 1 else np.unique(Y_train)}")

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

# Split test set in half for test and validation (same logic as original)
split_idx = len(X_test) // 2
X_val = X_test[:split_idx]
Y_val = Y_test[:split_idx]
X_test = X_test[split_idx:]
Y_test = Y_test[split_idx:]

# Convert to tensors on device
X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
Y_val = torch.FloatTensor(Y_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

# -------------------------
# Generalized Model definition
# -------------------------
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
        self.export_mode = False  # When True, output probabilities; when False, output logits
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
            current_shape = (1, self.input_length)  # (channels, length) for 1D conv
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
                for _ in range(layer_count):
                    conv = nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size//2)
                    self.layers.append(conv)
                    self._constrained_conv_layers.append(conv)
                    self.layers.append(nn.ReLU())
                    in_channels = channels
                
                # Single pooling at the end
                self.layers.append(nn.MaxPool1d(2))
                current_length = current_length // 2
                
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
                
                # Single pooling at the end
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_rows = current_rows // 2
                current_cols = current_cols // 2
                
                current_shape = (channels, current_rows, current_cols)
                current_size = None
                
            elif layer_type == 'Dropout':
                rate = float(config.get('rate', 0.5))
                self.layers.append(nn.Dropout(rate))
        
        # Final output layer
        if current_size is None:
            # Need to flatten
            self.layers.append(nn.Flatten())
            if len(current_shape) == 3:
                current_size = current_shape[0] * current_shape[1] * current_shape[2]
            elif len(current_shape) == 2:
                current_size = current_shape[0] * current_shape[1]
        
        self.layers.append(nn.Linear(current_size, classes))
    
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
        for i, layer in enumerate(self.layers):
            # Check actual layer type (may be wrapped by PAIModule)
            actual_layer = layer.main_module if hasattr(layer, 'main_module') else layer
            
            if isinstance(actual_layer, nn.Conv1d):
                # PerforatedAI handles the unsqueeze, just pass through
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
        
        # Apply softmax only in export mode to output probabilities for Edge Impulse
        # During training, output raw logits for CrossEntropyLoss compatibility
        if self.export_mode:
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
        GPA.pc.set_modules_to_convert([nn.Conv2d, nn.Conv1d, nn.Linear])
        GPA.pc.set_modules_to_track([])
    else:
        GPA.pc.set_modules_to_convert([nn.Linear])
        GPA.pc.set_modules_to_track([nn.Conv2d, nn.Conv1d])


    GPA.pc.set_max_dendrites(args.max_dendrites if str2bool(args.dendritic_optimization) else 0)
    print(f"Max dendrites set to: {GPA.pc.get_max_dendrites()}", flush=True)
    GPA.pc.set_perforated_backpropagation(str2bool(args.improved_dendritic_optimization))
    GPA.pc.set_dendrite_update_mode(True)
    GPA.pc.set_initial_correlation_batches(40)
    GPA.pc.set_max_dendrite_tries(1)
    
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

    # Print model architecture and parameter count BEFORE initialize_pai
    print(model)
    print(f"Total parameters (before PAI): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model = UPA.initialize_pai(model)
    
    # Re-verify max_dendrites after initialization
    print(f"Max dendrites after initialize_pai: {GPA.pc.get_max_dendrites()}", flush=True)
    if GPA.pc.get_max_dendrites() != (args.max_dendrites if str2bool(args.dendritic_optimization) else 0):
        print(f"WARNING: Max dendrites changed during initialization! Re-setting to {args.max_dendrites}...", flush=True)
        GPA.pc.set_max_dendrites(args.max_dendrites if str2bool(args.dendritic_optimization) else 0)
        print(f"Max dendrites now: {GPA.pc.get_max_dendrites()}", flush=True)
    
    # Set output dimensions for Conv1d layers if they are being optimized with dendrites
    for layer in model.layers:
        if isinstance(layer, nn.Conv1d):
            # Check if it's a PAIModule (PerforatedAI wrapped layer) by type name
            if type(layer).__name__ == 'PAINeuronModule' and isinstance(layer.main_module, nn.Conv1d):
                layer.set_this_output_dimensions([-1, 0, -1])
    print(f"Total parameters (after PAI): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Debug: Print model configuration
    print(f"\n[DEBUG] PyTorch Model Configuration:")
    print(f"[DEBUG] Input shape: {input_shape}, Data type: {data_type}")
    print(f"[DEBUG] Reshape to: rows={model.rows if hasattr(model, 'rows') else 'N/A'}, cols={model.columns if hasattr(model, 'columns') else 'N/A'}, channels={model.channels if hasattr(model, 'channels') else 'N/A'}")
    print(f"[DEBUG] Noise std: {noise_std_val}")
    print(f"[DEBUG] Mask time bands: {mask_time_bands_val}, Mask freq bands: {mask_freq_bands_val}, Warp time: {warp_time_val}")
    print(f"[DEBUG] Layer count: {len(model.layers)}")
    for i, layer in enumerate(model.layers):
        actual_layer = layer.main_module if hasattr(layer, 'main_module') else layer
        print(f"[DEBUG]   Layer {i}: {type(actual_layer).__name__}")
    print()
    
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params':model.parameters(), 'lr':args.learning_rate, 'betas':(0.9, 0.999)}
    schedArgs = {'mode':'max', 'patience': int(GPA.pc.get_n_epochs_to_switch()*0.75)}  # Set to 1000 to match Keras (no LR scheduling)
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # Compute class weights if auto_weight_classes is enabled
    class_weights = None
    if auto_weight_val:
        # Extract labels from Y_train_tensor
        if Y_train_tensor.dim() > 1:
            if Y_train_tensor.size(1) == 4:
                train_labels = Y_train_tensor[:, 0].long()
            else:
                train_labels = torch.argmax(Y_train_tensor, dim=1)
        else:
            train_labels = Y_train_tensor.long()
        
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
        test_loss, test_acc = test(model, test_loader, criterion, device)
        if(val_acc > max_val_acc):
            max_val_acc = val_acc
            max_test_acc = test_acc

        GPA.pai_tracker.add_extra_score(train_acc, 'Train')
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
            optimArgs = {'params':model.parameters(), 'lr':args.learning_rate, 'betas':(0.9, 0.999)}
            schedArgs = {'mode':'max', 'patience': int(GPA.pc.get_n_epochs_to_switch()*0.75)}
            optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

        print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Dendrite Count: {GPA.pai_tracker.member_vars.get("num_dendrites_added", "N/A")}')
        print(f'[DEBUG] Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f'\n[DEBUG] Final Test: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}')


    if str2bool(args.dendritic_optimization):
        print(f'First architecture: Val Acc: {first_val_acc:.4f}, Test Acc: {first_test_acc:.4f}, params: {first_param_count}')
        print(f'Best architecture: Val Acc: {max_val_acc:.4f}, Test Acc: {max_test_acc:.4f}, params: {UPA.count_params(model)} Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')
        print('Reduction in misclassifications because of dendrites')
        print(f'Validation: {(100.0*((max_val_acc-first_val_acc)/(1-first_val_acc))):.2f}%')
        print(f'Test: {(100.0*((max_test_acc-first_test_acc)/(1-first_test_acc))):.2f}%')
    else:
        print(f'Final architecture: Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, params: {UPA.count_params(model)} Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')

    # -------------------------
    # Get baseline PyTorch scores BEFORE blockwise/refresh transformations
    # -------------------------
    # Move model to CPU and eval mode to get reference scores
    model = model.cpu()
    model.eval()
    model.export_mode = False  # Use logits for evaluation
    
    print("\n=== PyTorch Baseline (before ONNX export transformations) ===", flush=True)
    pytorch_val_loss, pytorch_val_acc = test(model, val_loader, criterion, torch.device('cpu'))
    pytorch_test_loss, pytorch_test_acc = test(model, test_loader, criterion, torch.device('cpu'))
    print(f"PyTorch (eval mode) - Val: {pytorch_val_acc:.4f}, Test: {pytorch_test_acc:.4f}")
    print("="*50 + "\n", flush=True)

    # -------------------------
    # Print best_test_scores table
    # -------------------------
    """
    import pandas as pd
    best_test_scores_path = os.path.join('PAI', 'PAIbest_test_scores.csv')
    if os.path.exists(best_test_scores_path):
        print("\n" + "="*60)
        print("BEST TEST SCORES TABLE (from PAI/PAIbest_test_scores.csv)")
        print("="*60)
        df = pd.read_csv(best_test_scores_path)
        print(df.to_string(index=False))
        print("="*60 + "\n")
    else:
        print(f"\nNote: best_test_scores.csv not found at {best_test_scores_path}\n")
    """
    from perforatedai import blockwise_perforatedai as BPA
    from perforatedai import clean_perforatedai as CPA
    #model = UPA.load_system(model, 'PAI', 'best_model', True)
    print("before blockwise and refresh:")
    print(model)
    model = BPA.blockwise_network(model)
    model  = CPA.refresh_net(model)
    print("after blockwise and refresh:")
    print(model)
    # ============================================================================
    # COMPREHENSIVE DEBUGGING: Inspect ALL weights and activations
    # ============================================================================
    print("\n" + "="*80)
    print("DEBUGGING: Analyzing model after dendritic transformations")
    print("="*80)
    
    # 1. Print complete model structure
    print("\n--- MODEL STRUCTURE ---")
    for i, (name, module) in enumerate(model.named_modules()):
        print(f"{i:3d}. {name:50s} -> {type(module).__name__}")
    
    # 2. Analyze ALL parameters with detailed statistics
    print("\n--- PARAMETER ANALYSIS (BEFORE any clipping) ---")
    print(f"{'Module Name':<60s} {'Shape':<20s} {'Min':<12s} {'Max':<12s} {'Mean':<12s} {'Std':<12s}")
    print("-" * 120)
    
    extreme_params = []
    all_param_stats = []
    
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            
            with torch.no_grad():
                p_min = param.min().item()
                p_max = param.max().item()
                p_mean = param.mean().item()
                p_std = param.std().item()
                p_abs_max = param.abs().max().item()
                
                stats = {
                    'name': full_name,
                    'shape': str(list(param.shape)),
                    'min': p_min,
                    'max': p_max,
                    'mean': p_mean,
                    'std': p_std,
                    'abs_max': p_abs_max
                }
                all_param_stats.append(stats)
                
                # Flag extreme parameters
                if p_abs_max > 100 or abs(p_mean) > 100:
                    extreme_params.append(stats)
                
                # Print with scientific notation for large values
                print(f"{full_name:<60s} {str(list(param.shape)):<20s} {p_min:12.4e} {p_max:12.4e} {p_mean:12.4e} {p_std:12.4e}")
    
    if extreme_params:
        print("\n⚠️  WARNING: Found parameters with extreme values:")
        for stat in extreme_params:
            print(f"  - {stat['name']}: abs_max={stat['abs_max']:.2e}, mean={stat['mean']:.2e}")
    else:
        print("\n✓ All parameters have reasonable magnitudes (< 100)")
    
    # 3. Test forward pass and capture activation statistics
    print("\n--- ACTIVATION ANALYSIS (using calibration sample) ---")
    model.eval()
    model.export_mode = False  # Don't apply softmax yet
    
    # Get a sample from training data for activation analysis
    sample_input, _ = next(iter(train_loader))
    sample_input = sample_input[0:1].cpu()  # Take just one sample and move to CPU
    sample_flat = sample_input.view(1, -1)
    
    # Hook to capture activations
    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    activations[name] = {
                        'shape': list(output.shape),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'abs_max': output.abs().max().item()
                    }
        return hook
    
    # Register hooks on all modules
    hooks = []
    for name, module in model.named_modules():
        if name:  # Skip root module
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run forward pass
    with torch.no_grad():
        output = model(sample_flat)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print activation statistics
    print(f"{'Module Name':<60s} {'Shape':<20s} {'Min':<12s} {'Max':<12s} {'Abs Max':<12s}")
    print("-" * 120)
    
    extreme_activations = []
    for name, stats in activations.items():
        print(f"{name:<60s} {str(stats['shape']):<20s} {stats['min']:12.4e} {stats['max']:12.4e} {stats['abs_max']:12.4e}")
        
        if stats['abs_max'] > 1000:
            extreme_activations.append({'name': name, **stats})
    
    if extreme_activations:
        print("\n⚠️  WARNING: Found activations with extreme values:")
        for act in extreme_activations:
            print(f"  - {act['name']}: abs_max={act['abs_max']:.2e}, range=[{act['min']:.2e}, {act['max']:.2e}]")
    else:
        print("\n✓ All activations have reasonable magnitudes (< 1000)")
    
    # 4. Check for tanh operations specifically
    print("\n--- CHECKING FOR TANH AND DENDRITIC OPERATIONS ---")
    found_tanh = False
    found_dendritic = False
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'tanh' in module_type.lower() or 'Tanh' in module_type:
            print(f"  Found Tanh: {name} -> {module_type}")
            found_tanh = True
        if 'dendrit' in module_type.lower() or 'Dendrit' in module_type:
            print(f"  Found Dendritic: {name} -> {module_type}")
            found_dendritic = True
    
    if not found_tanh and not found_dendritic:
        print("  No explicit Tanh or Dendritic modules found")
        print("  (They may be embedded in custom forward functions)")
    
    print("="*80 + "\n")
    
    # Now apply weight constraints based on what we found
    print("--- APPLYING WEIGHT CONSTRAINTS ---")
    if extreme_params:
        print("Found extreme parameters - applying aggressive clipping to [-1, 1]")
        clip_range = 1.0
    elif extreme_activations:
        print("Found extreme activations - applying moderate clipping to [-10, 10]")
        clip_range = 10.0
    else:
        print("No extreme values found - applying conservative clipping to [-100, 100]")
        clip_range = 100.0
    
    constrained_params = 0
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            with torch.no_grad():
                orig_max = param.abs().max().item()
                param.clamp_(-clip_range, clip_range)
                if orig_max > clip_range:
                    print(f"  Clipped {full_name}: {orig_max:.2e} → {clip_range:.2f}")
                    constrained_params += 1
    
    print(f"Constrained {constrained_params} parameters to [±{clip_range}]")
    print("="*80 + "\n")
    
    # ============================================================================
    # INSPECT skip_weights - These are likely causing the 1e36 scales
    # ============================================================================
    print("--- INSPECTING SKIP_WEIGHTS ---")
    skip_weights_found = False
    for name, module in model.named_modules():
        if hasattr(module, 'skip_weights'):
            skip_weights_found = True
            print(f"\nModule: {name}")
            print(f"  skip_weights type: {type(module.skip_weights)}")
            
            if isinstance(module.skip_weights, nn.ParameterList):
                print(f"  Length: {len(module.skip_weights)}")
                for i, param in enumerate(module.skip_weights):
                    print(f"  Parameter {i}:")
                    print(f"    Shape: {param.shape}")
                    print(f"    Min: {param.min().item():.6e}, Max: {param.max().item():.6e}")
                    print(f"    Mean: {param.mean().item():.6e}, Std: {param.std().item():.6e}")
                
                # Try to remove empty skip_weights
                if len(module.skip_weights) == 0:
                    print(f"  ⚠️  EMPTY ParameterList - attempting to remove...")
                    try:
                        delattr(module, 'skip_weights')
                        print(f"  ✓ Successfully removed empty skip_weights")
                    except Exception as e:
                        print(f"  ✗ Failed to remove: {e}")
            else:
                print(f"  Not a ParameterList - it's a {type(module.skip_weights)}")
    
    if not skip_weights_found:
        print("  No skip_weights found in any module")
    print("="*80 + "\n")
    
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
    # Now enable export mode for ONNX (outputs probabilities)
    model.export_mode = True
    
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
    
    # Test dummy input works before ONNX export
    print("\n=== Testing dummy input before ONNX export ===", flush=True)
    print(f"Model layers: {len(model.layers)}", flush=True)
    for i, layer in enumerate(model.layers):
        actual_layer = layer.main_module if hasattr(layer, 'main_module') else layer
        has_main = hasattr(layer, 'main_module')
        print(f"  Layer {i}: {type(layer).__name__} (has_main_module={has_main}) -> {type(actual_layer).__name__}", flush=True)
    
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"Dummy input shape: {dummy_input.shape}", flush=True)
            print(f"Model output shape: {test_output.shape}", flush=True)
            print("Dummy input test passed!", flush=True)
    except Exception as e:
        print(f"ERROR: Dummy input test failed: {e}", flush=True)
        print("This model cannot be exported to ONNX.", flush=True)
        return
    print("="*50 + "\n", flush=True)
    
    # Use legacy ONNX exporter to avoid torch.export issues with dynamic shapes
    # Dynamic axes allow variable batch size for validation
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    # Try with dynamo=False first (PyTorch 2.1+), fallback to legacy exporter
    try:
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes,
                          dynamo=False)  # Use legacy exporter
    except TypeError:
        # Older PyTorch version, dynamo parameter doesn't exist
        # Set environment variable to force legacy exporter
        os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)
    print("Exported ONNX to", onnx_path)
    
    # ============================================================================
    # INSPECT skip_weights AFTER export - These are likely causing the 1e36 scales
    # ============================================================================
    print("\n" + "="*80)
    print("INSPECTING SKIP_WEIGHTS (post-export for debugging)")
    print("="*80)
    skip_weights_found = False
    for name, module in model.named_modules():
        if hasattr(module, 'skip_weights'):
            skip_weights_found = True
            print(f"\nModule: {name}")
            print(f"  skip_weights type: {type(module.skip_weights)}")
            
            if isinstance(module.skip_weights, nn.ParameterList):
                print(f"  Length: {len(module.skip_weights)}")
                for i, param in enumerate(module.skip_weights):
                    print(f"  Parameter {i}:")
                    print(f"    Shape: {param.shape}")
                    print(f"    Min: {param.min().item():.6e}, Max: {param.max().item():.6e}")
                    print(f"    Mean: {param.mean().item():.6e}, Std: {param.std().item():.6e}")
            else:
                print(f"  Not a ParameterList - it's a {type(module.skip_weights)}")
    
    if not skip_weights_found:
        print("  No skip_weights found in any module")
    print("="*80 + "\n")
    
    # -------------------------
    # DEBUG: Inspect ONNX model for extreme initializer values
    # -------------------------
    print("\n=== INSPECTING ONNX MODEL INITIALIZERS ===")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        
        print(f"ONNX Opset: {onnx_model.opset_import[0].version}")
        print(f"Initializers (constants): {len(onnx_model.graph.initializer)}")
        
        extreme_initializers = []
        for init in onnx_model.graph.initializer:
            # Convert to numpy array
            from onnx.numpy_helper import to_array
            arr = to_array(init)
            
            if arr.size > 0:
                abs_max = np.abs(arr).max()
                abs_mean = np.abs(arr).mean()
                
                if abs_max > 1000 or abs_mean > 100:
                    extreme_initializers.append({
                        'name': init.name,
                        'shape': arr.shape,
                        'abs_max': abs_max,
                        'abs_mean': abs_mean,
                        'min': arr.min(),
                        'max': arr.max()
                    })
                    print(f"\n⚠️  EXTREME INITIALIZER: {init.name}")
                    print(f"    Shape: {arr.shape}")
                    print(f"    Range: [{arr.min():.6e}, {arr.max():.6e}]")
                    print(f"    Abs max: {abs_max:.6e}, Abs mean: {abs_mean:.6e}")
        
        if not extreme_initializers:
            print("\n✓ All ONNX initializers have reasonable values")
        else:
            print(f"\n⚠️  Found {len(extreme_initializers)} initializers with extreme values!")
            
    except Exception as e:
        print(f"Failed to inspect ONNX model: {e}")
    
    print("="*80 + "\n")
    
    # -------------------------
    # Validate ONNX model accuracy
    # -------------------------
    print("\n" + "="*60)
    print("VALIDATING ONNX MODEL")
    print("="*60)
    
    try:
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
        test_acc_onnx = test_onnx(test_loader, "Test")
        
        # Compare with PyTorch model results (use fresh eval, not training max)
        print("\nComparison:")
        print(f"PyTorch  - Val: {pytorch_val_acc:.4f}, Test: {pytorch_test_acc:.4f}")
        print(f"ONNX     - Val: {val_acc_onnx:.4f}, Test: {test_acc_onnx:.4f}")
        
        val_diff = abs(pytorch_val_acc - val_acc_onnx)
        test_diff = abs(pytorch_test_acc - test_acc_onnx)
        
        if val_diff < 0.001 and test_diff < 0.001:
            print("✓ ONNX model matches PyTorch model (difference < 0.1%)")
        elif val_diff < 0.01 and test_diff < 0.01:
            print("⚠ ONNX model has small difference from PyTorch (difference < 1%)")
        else:
            print(f"⚠ WARNING: ONNX model differs significantly (val diff: {val_diff:.4f}, test diff: {test_diff:.4f})")
        
        print("="*60 + "\n")
        
    except ImportError:
        print("⚠ onnxruntime not installed, skipping ONNX validation")
        print("  Install with: pip install onnxruntime")
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠ Error validating ONNX model: {e}")
        traceback.print_exc()
        print("="*60 + "\n")
    
    # -------------------------
    # Test INT8 Quantization (similar to Edge Impulse)
    # -------------------------
    if str2bool(args.confirm_quant_score):
        print("\n" + "="*60)
        print("TESTING INT8 QUANTIZATION")
        print("="*60)
        
        try:
            import tensorflow as tf
            import subprocess
            
            # Convert ONNX to TFLite with int8 quantization
            def representative_dataset():
                """Generate representative data for quantization calibration"""
                activation_stats = {'min': float('inf'), 'max': float('-inf'), 'samples': 0}
                
                for batch_idx, (data, _) in enumerate(train_loader):
                    if batch_idx >= 100:  # Use 100 batches for calibration
                        break
                    data_np = data.cpu().numpy()
                    for i in range(data_np.shape[0]):
                        sample = data_np[i:i+1].reshape(1, -1).astype(np.float32)
                        activation_stats['min'] = min(activation_stats['min'], sample.min())
                        activation_stats['max'] = max(activation_stats['max'], sample.max())
                        activation_stats['samples'] += 1
                        yield [sample]
                
                print(f"\nCalibration dataset statistics:")
                print(f"  Samples: {activation_stats['samples']}")
                print(f"  Input range: [{activation_stats['min']:.4f}, {activation_stats['max']:.4f}]")
                print(f"  Input span: {activation_stats['max'] - activation_stats['min']:.4f}")

            
            # Convert ONNX to TF SavedModel using onnx2tf
            saved_model_dir = os.path.join(args.out_directory, 'saved_model_temp')
            print(f"Converting ONNX to TensorFlow SavedModel using onnx2tf...")
            result = subprocess.run(
                [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", saved_model_dir, "-osd"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                print(f"onnx2tf stderr: {result.stderr}")
                raise Exception(f"onnx2tf conversion failed with return code {result.returncode}")
            print(f"Converted ONNX to TF SavedModel at {saved_model_dir}")
            
            # Convert to TFLite with int8 quantization
            print("\nAttempting int8 quantization...")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            try:
                tflite_model = converter.convert()
                tflite_path = os.path.join(args.out_directory, 'model_int8.tflite')
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                print(f"✓ Successfully created int8 quantized model: {tflite_path}")
            except Exception as quant_error:
                print(f"\n✗ INT8 QUANTIZATION FAILED")
                print(f"Error: {quant_error}")
                print("\nThis is the same error Edge Impulse encounters.")
                print("The dendritic network likely produces activation ranges that are incompatible with int8 quantization.")
                print("\nPossible solutions:")
                print("  1. Skip dendritic optimization for Edge Impulse deployment")
                print("  2. Add activation clipping in the dendritic modules")
                print("  3. Use mixed precision (int8 weights, float32 activations)")
                raise
            
            # Test int8 quantized model
            print("\nTesting int8 quantized model...")
            
            # Inspect TFLite model before allocating tensors
            try:
                import flatbuffers
                from tensorflow.lite.python import schema_py_generated as schema_fb
                
                with open(tflite_path, 'rb') as f:
                    buf = bytearray(f.read())
                
                model = schema_fb.Model.GetRootAsModel(buf, 0)
                
                print(f"\nTFLite Model Info:")
                print(f"  Version: {model.Version()}")
                print(f"  Subgraphs: {model.SubgraphsLength()}")
                
                subgraph = model.Subgraphs(0)
                print(f"  Tensors: {subgraph.TensorsLength()}")
                print(f"  Operators: {subgraph.OperatorsLength()}")
                
                # Check quantization parameters for each tensor
                print(f"\nTensor Quantization Details:")
                for i in range(subgraph.TensorsLength()):
                    tensor = subgraph.Tensors(i)
                    name = tensor.Name().decode('utf-8') if tensor.Name() else f"tensor_{i}"
                    quant = tensor.Quantization()
                    
                    if quant and quant.ScaleLength() > 0:
                        scales = [quant.Scale(j) for j in range(quant.ScaleLength())]
                        zero_points = [quant.ZeroPoint(j) for j in range(quant.ZeroPointLength())]
                        print(f"  Tensor {i} ({name}): scale={scales}, zero_point={zero_points}")
                        
                        # Check for problematic quantization scales
                        for scale in scales:
                            if scale <= 0 or scale > 1e6:
                                print(f"    ⚠️  WARNING: Tensor {i} has extreme scale: {scale}")
                
            except Exception as inspect_error:
                print(f"Could not inspect TFLite model: {inspect_error}")
            
            # Now try to allocate tensors
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            
            try:
                interpreter.allocate_tensors()
                print("\n✓ Tensor allocation successful")
            except (RuntimeError, Exception) as alloc_error:
                print(f"\n✗ TENSOR ALLOCATION FAILED")
                print(f"Error: {alloc_error}")
                print("\nThis is the exact error Edge Impulse encountered.")
                print("The quantized model has invalid tensor configurations.")
                print("\nLikely cause: PerforatedAI dendritic modules create activation ranges")
                print("that cannot be properly quantized to int8.")
                raise
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            def test_int8(loader, dataset_name):
                correct = 0
                total = 0
                
                for inputs, labels in loader:
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
                    
                    # Process one sample at a time
                    for i in range(batch_size):
                        sample = inputs_flat[i:i+1].astype(np.float32)
                        
                        # Quantize input if needed
                        if input_details[0]['dtype'] == np.int8:
                            input_scale, input_zero_point = input_details[0]['quantization']
                            sample = (sample / input_scale + input_zero_point).astype(np.int8)
                        
                        interpreter.set_tensor(input_details[0]['index'], sample)
                        interpreter.invoke()
                        output = interpreter.get_tensor(output_details[0]['index'])
                        
                        # Dequantize output if needed
                        if output_details[0]['dtype'] == np.int8:
                            output_scale, output_zero_point = output_details[0]['quantization']
                            output = (output.astype(np.float32) - output_zero_point) * output_scale
                        
                        predicted = np.argmax(output[0])
                        correct += (predicted == labels[i])
                        total += 1
                
                accuracy = correct / total
                print(f"{dataset_name} Accuracy (INT8): {accuracy:.4f} ({correct}/{total})")
                return accuracy
            
            # Test on full X_split_test dataset
            X_test_full = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
            Y_test_full = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))
            print(f"Full test dataset size: {X_test_full.shape[0]} samples")
            print(f"(Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)})")
            X_test_full = torch.FloatTensor(X_test_full)
            Y_test_full = torch.FloatTensor(Y_test_full)
            full_test_dataset = TensorDataset(X_test_full, Y_test_full)
            full_test_loader = DataLoader(full_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            
            full_test_acc_int8 = test_int8(full_test_loader, "Full Test Set")
            
            print("\nComparison:")
            print(f"PyTorch  - Val: {max_val_acc:.4f}, Test: {max_test_acc:.4f}")
            print(f"ONNX     - Val: {val_acc_onnx:.4f}, Test: {test_acc_onnx:.4f}")
            print(f"INT8     - Full: {full_test_acc_int8:.4f}")

            print("="*60 + "\n")
            
        except ImportError as ie:
            print(f"⚠ Missing dependencies for int8 testing: {ie}")
            print("  Install with: pip install tensorflow onnx2tf")
            print("="*60 + "\n")
        except Exception as e:
            print(f"⚠ Error during int8 quantization test: {e}")
            traceback.print_exc()
            print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Skipping int8 quantization test (use --confirm-quant-score true to enable)")
        print("="*60 + "\n")

    """
    # Attempt to run onnx2tf (so EI won't need to re-run it)
    saved_model_dir = os.path.join(args.out_directory, 'onnx2tf-out')
    try:
        cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", saved_model_dir]
        print("[train] Running onnx2tf:", " ".join(cmd))
        proc = _subprocess.run(cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE, text=True)
        print("[train] onnx2tf stdout:\n", proc.stdout)
        print("[train] onnx2tf stderr:\n", proc.stderr)
        if proc.returncode == 0 and os.path.isdir(saved_model_dir):
            print("[train] onnx2tf conversion succeeded, SavedModel at:", saved_model_dir)
        else:
            print("[train] onnx2tf conversion returned code", proc.returncode, "- continuing to create compat shim (EI may convert later).")
    except Exception:
        print("[train] Failed to invoke onnx2tf CLI; continuing and creating compat shim.")
        traceback.print_exc()

    # Create compat shim SavedModel at out/onnx2tf-out that exposes .layers (used by EI embeddings)
    try:
        # Import TF lazily so environments without TF can still run training portion
        import tensorflow as tf
    except Exception:
        tf = None
        print("[train] TensorFlow not available in this environment; cannot create compat shim. If running inside EI, ensure TF is installed.")

    if tf is not None:
        try:
            print("[compat-shim] Target SavedModel dir:", saved_model_dir)
            loaded = None
            if os.path.isdir(saved_model_dir):
                try:
                    loaded = tf.saved_model.load(saved_model_dir)
                    print("[compat-shim] Loaded existing SavedModel from", saved_model_dir)
                except Exception:
                    print("[compat-shim] Failed to load existing SavedModel (will still create shim):")
                    traceback.print_exc()
            else:
                print("[compat-shim] No existing SavedModel at", saved_model_dir, "- creating shim fallback.")

            # Try to find a backend callable
            backend_fn = None
            if loaded is not None:
                try:
                    sigs = getattr(loaded, 'signatures', None)
                    if sigs and 'serving_default' in sigs:
                        backend_fn = sigs['serving_default']
                        print("[compat-shim] Using signatures['serving_default'] as backend function.")
                except Exception:
                    pass

            if backend_fn is None and loaded is not None:
                try:
                    candidate = tf.function(lambda x: loaded(x))
                    il = int(input_length) if 'input_length' in locals() else 624
                    _ = candidate.get_concrete_function(tf.TensorSpec([1, il], tf.float32))
                    backend_fn = candidate
                    print("[compat-shim] Using tf.function(lambda x: loaded(x)) as backend.")
                except Exception:
                    pass

            if backend_fn is None and loaded is not None:
                for name in dir(loaded):
                    if name.startswith('_'):
                        continue
                    attr = getattr(loaded, name)
                    if callable(attr):
                        backend_fn = attr
                        print(f"[compat-shim] Using callable attribute '{name}' from loaded SavedModel as backend.")
                        break

            if backend_fn is None:
                print("[compat-shim] No backend callable found. Creating fallback backend that returns zeros.")
                try:
                    out_dim = int(num_classes)
                except Exception:
                    out_dim = 3

                @tf.function(input_signature=[tf.TensorSpec([1, onnx_input_size], tf.float32)])
                def backend_fn(x):
                    return tf.zeros((1, out_dim), dtype=tf.float32)

            # Determine output dim
            out_dim = None
            try:
                test_inp = tf.constant(np.zeros((1, onnx_input_size), dtype=np.float32))
                # Call backend and be robust to different return types (Tensor, np.array, dict, tuple, list, etc.)
                try:
                    test_out = backend_fn(test_inp)
                except Exception:
                    # Some callables may succeed but raise on first call signature; try a gentle fallback
                    try:
                        maybe = backend_fn(test_inp)
                        test_out = maybe
                    except Exception:
                        test_out = None

                # Normalize outputs: flatten nested structures (dict/list/tuple) and pick first element if needed
                try:
                    flat_outputs = tf.nest.flatten(test_out)
                    if flat_outputs:
                        candidate = flat_outputs[0]
                    else:
                        candidate = None
                except Exception:
                    # tf.nest.flatten may throw if test_out is None or unexpected; fall back to simple checks
                    if isinstance(test_out, dict):
                        vals = list(test_out.values())
                        candidate = vals[0] if vals else None
                    elif isinstance(test_out, (list, tuple)):
                        candidate = test_out[0] if test_out else None
                    else:
                        candidate = test_out

                if candidate is not None:
                    # Ensure candidate is a tensor-like object we can inspect
                    try:
                        to = tf.convert_to_tensor(candidate)
                    except Exception:
                        # If candidate is a mapping (unlikely after flatten) attempt to get its first value
                        if isinstance(candidate, dict):
                            vals = list(candidate.values())
                            if vals:
                                to = tf.convert_to_tensor(vals[0])
                            else:
                                to = None
                        else:
                            to = None

                    if to is not None:
                        out_shape = to.shape
                        if len(out_shape) >= 2:
                            out_dim = int(out_shape[-1])
                        else:
                            out_dim = int(out_shape[0])
                        print("[compat-shim] Detected output dim:", out_dim)
                    else:
                        print("[compat-shim] Candidate output not convertible to tensor; will use fallback.")
                else:
                    print("[compat-shim] No candidate output obtained from backend; will use fallback.")
            except Exception:
                print("[compat-shim] Could not infer output dim; will use fallback if needed.")
                traceback.print_exc()

            if out_dim is None:
                try:
                    out_dim = int(num_classes)
                except Exception:
                    out_dim = 3
                print("[compat-shim] Using fallback output dim:", out_dim)

            # Define LayerContainer and compat module
            class LayerContainer(tf.Module):
                def __init__(self):
                    super().__init__()
                    self.layer_0 = tf.keras.layers.Layer(name='compat_dummy_layer')
                def __len__(self):
                    return 1
                def __getitem__(self, idx):
                    if idx == 0:
                        return getattr(self, 'layer_0')
                    raise IndexError

            class CompatModule(tf.Module):
                def __init__(self, backend_callable, output_dim, input_size):
                    super().__init__()
                    self.layers = LayerContainer()
                    self._backend = backend_callable
                    self._output_dim = output_dim
                    self._input_size = input_size

                @tf.function(input_signature=[tf.TensorSpec([1, None], tf.float32)])
                def serving_default(self, x):
                    # Reshape input if needed to match expected size
                    x = tf.reshape(x, [1, self._input_size])
                    try:
                        res = self._backend(x)
                    except Exception:
                        res = self._backend(x)
                    if isinstance(res, dict):
                        res = list(res.values())[0]
                    res_t = tf.convert_to_tensor(res, dtype=tf.float32)
                    try:
                        res_t = tf.reshape(res_t, [1, self._output_dim])
                    except Exception:
                        pass
                    return res_t

            compat = CompatModule(backend_fn, out_dim, onnx_input_size)

            # Overwrite existing SavedModel dir
            try:
                if os.path.isdir(saved_model_dir):
                    shutil.rmtree(saved_model_dir)
                    print("[compat-shim] Removed existing SavedModel dir to be replaced with compat shim.")
            except Exception:
                print("[compat-shim] Failed to remove existing SavedModel dir; continuing.")

            try:
                tf.saved_model.save(compat, saved_model_dir, signatures={'serving_default': compat.serving_default})
                print("[compat-shim] Saved compat shim at:", saved_model_dir)
                print("[compat-shim] Now tf.saved_model.load(saved_model_dir) will return an object with .layers.")
            except Exception:
                print("[compat-shim] Failed to save compat shim:")
                traceback.print_exc()

        except Exception:
            print("[compat-shim] Unexpected error during shim creation:")
            traceback.print_exc()
    else:
        print("[train] TensorFlow not present — compat shim not created.")
    """
if __name__ == "__main__":
    main(args)
