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
parser.add_argument('--batch-size', type=int, default=32, dest='batch_size')
parser.add_argument('--learning-rate', type=float, default=0.001, dest='learning_rate')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.1)

# Model architecture parameters
parser.add_argument('--num-conv', type=int, default=3, choices=[1,2,3,4], help="Number of conv layers for 2D data", dest='num_conv')
parser.add_argument('--num-linear', type=int, default=2, choices=[1,2,3], help="Number of linear layers", dest='num_linear')
parser.add_argument('--network-width', type=float, default=2, help="Width multiplier for channels", dest='network_width')
parser.add_argument('--noise-std', type=float, default=0, help="Gaussian noise stddev during training", dest='noise_std')
parser.add_argument('--channel-growth-mode', type=int, default=5, choices=[0,1,2,3,4,5], help="Channel growth pattern", dest='channel_growth_mode')

# PerforatedAI dendritic optimization parameters
parser.add_argument('--dendritic-optimization', type=str, required=False, default="true", dest='dendritic_optimization')
parser.add_argument('--switch-speed', type=str, default='slow', help="speed to switch", choices=['slow', 'medium', 'fast'], dest='switch_speed')
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

os.environ["PAIEMAIL"] = "user@edgeimpulse.com"
os.environ["PAITOKEN"] = args.perforated_ai_token

os.makedirs(args.out_directory, exist_ok=True)

def str2bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "t", "yes", "y")

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
    # 1D input: (features,) - Time-series or custom features
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
    Adaptive classifier that works with:
    - 1D data: time-series, sensor data, custom features
    - 2D data: images, spectrograms, MFCC/MFE features
    """
    def __init__(self, input_shape, classes, data_type='1d_timeseries', num_conv=2, num_linear=1, 
                 width=1.0, linear_dropout=0.5, noise_std=0.2, growth_mode=0):
        super(AdaptiveClassifier, self).__init__()
        
        self.input_shape = input_shape
        self.classes = classes
        self.data_type = data_type
        self.num_conv = num_conv
        self.num_linear = num_linear
        self.width = width
        self.linear_dropout = linear_dropout
        self.noise_std = noise_std
        self.export_mode = False  # When True, output probabilities; when False, output logits
        
        # Channel growth patterns
        if growth_mode == 0:
            base_channels = [8, 16, 32, 64]
        elif growth_mode == 1:
            base_channels = [8, 16, 24, 32]
        elif growth_mode == 2:
            base_channels = [8, 16, 16, 32]
        elif growth_mode == 3:
            base_channels = [8, 16, 16, 16]
        elif growth_mode == 4:
            base_channels = [8, 8, 8, 8]
        elif growth_mode == 5:
            base_channels = [8, 8, 8, 16]
        self.channel_sizes = [max(1, int(ch * width)) for ch in base_channels]
        
        if self.data_type == '2d_image':
            # For 2D image/spectrogram data (H, W, C)
            self.rows, self.columns, self.channels = input_shape
            self.conv_blocks = nn.ModuleList()
            self._constrained_conv_layers = []
            
            if num_conv > 0:
                in_channels = self.channels
                for i in range(num_conv):
                    out_channels = self.channel_sizes[i]
                    conv_block = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=3, padding='same'),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Dropout(0.25)
                    )
                    self.conv_blocks.append(conv_block)
                    self._constrained_conv_layers.append(conv_block[0])
                    in_channels = out_channels
                
                # Calculate flattened size after conv layers
                final_rows = self.rows // (2 ** num_conv)
                final_cols = self.columns // (2 ** num_conv)
                flattened_size = self.channel_sizes[num_conv - 1] * final_rows * final_cols
            else:
                # No conv layers - flatten input
                flattened_size = self.rows * self.columns * self.channels
                
        elif self.data_type == '1d_timeseries':
            # For 1D time-series or feature data
            if len(input_shape) == 1:
                self.input_length = input_shape[0]
            else:
                # If somehow 3D but should be treated as 1D
                self.input_length = np.prod(input_shape)
            
            self.conv_blocks = nn.ModuleList()
            self._constrained_conv_layers = []
            
            # Optional 1D convolutions for time-series
            if num_conv > 0 and self.input_length >= 16:
                in_channels = 1
                current_length = self.input_length
                
                for i in range(num_conv):
                    out_channels = self.channel_sizes[i]
                    # Use 1D convolutions for time-series
                    conv_block = nn.Sequential(
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2, stride=2),
                        nn.Dropout(0.25)
                    )
                    self.conv_blocks.append(conv_block)
                    self._constrained_conv_layers.append(conv_block[0])
                    in_channels = out_channels
                    current_length = current_length // 2
                
                flattened_size = self.channel_sizes[num_conv - 1] * current_length
            else:
                # No conv layers or input too small - use input directly
                flattened_size = self.input_length
        
        # Linear layers (same for both 1D and 2D)
        self.linear_layers = nn.ModuleList()
        linear_sizes = self._calculate_linear_sizes(flattened_size, classes, num_linear)
        
        for i in range(num_linear - 1):
            self.linear_layers.append(nn.Sequential(
                nn.Dropout(linear_dropout),
                nn.Linear(linear_sizes[i], linear_sizes[i + 1]),
                nn.ReLU()
            ))
        
        self.linear_layers.append(nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(linear_sizes[-2], linear_sizes[-1])
        ))
    
    def _calculate_linear_sizes(self, input_size, output_size, num_layers):
        if num_layers == 1:
            return [input_size, output_size]
        sizes = [input_size]
        log_start = torch.log(torch.tensor(float(input_size)))
        log_end = torch.log(torch.tensor(float(output_size)))
        for i in range(1, num_layers):
            ratio = i / num_layers
            log_size = log_start + (log_end - log_start) * ratio
            size = max(output_size, int(torch.exp(log_size).item()))
            sizes.append(size)
        sizes.append(output_size)
        return sizes
    
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
        # Add noise during training if specified
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        if self.data_type == '2d_image':
            # Reshape for 2D convolutions: (N, H, W, C) -> (N, C, H, W)
            if x.dim() == 2:
                # Flat input, reshape to image
                x = x.view(-1, self.channels, self.rows, self.columns)
            elif x.dim() == 4:
                # Already 4D (N, H, W, C), permute to (N, C, H, W)
                x = x.permute(0, 3, 1, 2)
            
            # Pass through conv blocks
            if len(self.conv_blocks) > 0:
                for conv_block in self.conv_blocks:
                    x = conv_block(x)
            x = x.view(x.size(0), -1)
            
        elif self.data_type == '1d_timeseries':
            # Handle 1D time-series data
            if x.dim() == 2:
                # (N, features) format
                if len(self.conv_blocks) > 0:
                    # Reshape to (N, 1, features) for 1D conv
                    x = x.unsqueeze(1)
                    for conv_block in self.conv_blocks:
                        x = conv_block(x)
                    x = x.view(x.size(0), -1)
                # else: already in correct shape for linear layers
            elif x.dim() == 4:
                # If somehow 4D, flatten appropriately
                x = x.view(x.size(0), -1)
                if len(self.conv_blocks) > 0:
                    x = x.unsqueeze(1)
                    for conv_block in self.conv_blocks:
                        x = conv_block(x)
                    x = x.view(x.size(0), -1)
        
        # Pass through linear layers
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        
        # Apply softmax only in export mode to output probabilities for Edge Impulse
        # During training, output raw logits for CrossEntropyLoss compatibility
        if self.export_mode:
            # Clamp logits to prevent numerical instability in softmax and int8 quantization
            x = torch.clamp(x, min=-10.0, max=10.0)
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
    # Instantiate adaptive model based on detected data type
    model = AdaptiveClassifier(
        input_shape=input_shape,
        classes=num_classes,
        data_type=data_type,
        num_conv=args.num_conv,
        num_linear=args.num_linear,
        width=args.network_width,
        linear_dropout=args.dropout,
        noise_std=args.noise_std,
        growth_mode=args.channel_growth_mode
    ).to(device)

    GPA.pc.set_testing_dendrite_capacity(False)

    if args.switch_speed == 'fast':
        GPA.pc.set_n_epochs_to_switch(10)
    elif args.switch_speed == 'medium':
        GPA.pc.set_n_epochs_to_switch(25)
    else:
        GPA.pc.set_n_epochs_to_switch(100)

    GPA.pc.set_verbose(False)
    GPA.pc.set_silent(True)


    model = UPA.initialize_pai(model)
    
    # Re-verify max_dendrites after initialization
    print(f"Max dendrites after initialize_pai: {GPA.pc.get_max_dendrites()}", flush=True)
    if GPA.pc.get_max_dendrites() != (args.max_dendrites if str2bool(args.dendritic_optimization) else 0):
        print(f"WARNING: Max dendrites changed during initialization! Re-setting to {args.max_dendrites}...", flush=True)
        GPA.pc.set_max_dendrites(args.max_dendrites if str2bool(args.dendritic_optimization) else 0)
        print(f"Max dendrites now: {GPA.pc.get_max_dendrites()}", flush=True)
    
    # Set output dimensions for Conv1d layers if they are being optimized with dendrites
    for block in model.conv_blocks:
        print(block)
        if hasattr(block, '__iter__') and len(block) > 0:
            first_layer = block[0]
            # Check if it's a PAIModule (PerforatedAI wrapped layer) by type name
            if type(first_layer).__name__ == 'PAINeuronModule' and isinstance(first_layer.main_module, nn.Conv1d):
                first_layer.set_this_output_dimensions([-1, 0, -1])
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params':model.parameters(), 'lr':args.learning_rate, 'betas':(0.9, 0.999)}
    schedArgs = {'mode':'max', 'patience': 5}
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    criterion = nn.CrossEntropyLoss()

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
        if training_complete:
            break
        elif restructured:
            if first_test_acc == 0:
                first_val_acc = val_acc
                first_test_acc = test_acc
            print('Restructured dendritic architecture')
            optimArgs = {'params':model.parameters(), 'lr':args.learning_rate, 'betas':(0.9, 0.999)}
            schedArgs = {'mode':'max', 'patience': 5}
            optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

        print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Dendrite Count: {GPA.pai_tracker.member_vars.get("num_dendrites_added", "N/A")}')

    test_loss, test_acc = test(model, test_loader, criterion, device)

    if str2bool(args.dendritic_optimization):
        print(f'First architecture: Val Acc: {first_val_acc:.4f}, Test Acc: {first_test_acc:.4f}, params: {first_param_count}')
        print(f'Best architecture: Val Acc: {max_val_acc:.4f}, Test Acc: {max_test_acc:.4f}, params: {UPA.count_params(model)} Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')
        print('Reduction in misclassifications because of dendrites')
        print(f'Validation: {(100.0*((max_val_acc-first_val_acc)/(1-first_val_acc))):.2f}%')
        print(f'Test: {(100.0*((max_test_acc-first_test_acc)/(1-first_test_acc))):.2f}%')
    else:
        print(f'Final architecture: Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, params: {UPA.count_params(model)} Dendrite Count: {GPA.pai_tracker.member_vars["num_dendrites_added"]}')

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
    model = BPA.blockwise_network(model)
    model  = CPA.refresh_net(model)
    
    # Fix padding for conv layers if needed
    for block in model.conv_blocks:
        if hasattr(block, '__iter__'):
            for layer in block:
                if isinstance(layer, (nn.Conv2d, nn.Conv1d)) and hasattr(layer, 'layer_array'):
                    for conv in layer.layer_array:
                        if hasattr(conv, 'padding') and conv.padding == 'same':
                            if isinstance(conv.kernel_size, tuple):
                                padding = tuple((k - 1) // 2 for k in conv.kernel_size)
                            else:
                                padding = (conv.kernel_size - 1) // 2
                            conv.padding = padding

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
    
    # Put model in eval mode and enable export mode for softmax output
    model.eval()
    model.export_mode = True  # Output probabilities instead of logits
    
    # Move model to CPU for export
    model = model.cpu()
    
    # Check model output range with some sample data for diagnostics
    print("\n=== Checking model output range ===", flush=True)
    with torch.no_grad():
        sample_outputs = []
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 5:  # Check first 5 batches
                break
            data = data.view(data.size(0), -1).cpu()
            output = model(data)
            sample_outputs.append(output)
        
        all_outputs = torch.cat(sample_outputs, dim=0)
        print(f"Output min: {all_outputs.min().item():.4f}", flush=True)
        print(f"Output max: {all_outputs.max().item():.4f}", flush=True)
        print(f"Output mean: {all_outputs.mean().item():.4f}", flush=True)
        print(f"Output std: {all_outputs.std().item():.4f}", flush=True)
    print("===================================\n", flush=True)
    
    # Use legacy ONNX exporter to avoid torch.export issues with dynamic shapes
    # Dynamic axes allow variable batch size for validation
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    
    # Try with dynamo=False first (PyTorch 2.1+), fallback to legacy exporter
    try:
        torch.onnx.export(model.cpu(),
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
        torch.onnx.export(model.cpu(),
                          dummy_input,
                          onnx_path,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes=dynamic_axes)
    print("Exported ONNX to", onnx_path)
    
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
        
        # Compare with PyTorch model results
        print("\nComparison:")
        print(f"PyTorch  - Val: {max_val_acc:.4f}, Test: {max_test_acc:.4f}")
        print(f"ONNX     - Val: {val_acc_onnx:.4f}, Test: {test_acc_onnx:.4f}")
        
        val_diff = abs(max_val_acc - val_acc_onnx)
        test_diff = abs(max_test_acc - test_acc_onnx)
        
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
                for batch_idx, (data, _) in enumerate(train_loader):
                    if batch_idx >= 100:  # Use 100 batches for calibration
                        break
                    data_np = data.cpu().numpy()
                    for i in range(data_np.shape[0]):
                        sample = data_np[i:i+1].reshape(1, -1).astype(np.float32)
                        yield [sample]
            
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
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            tflite_path = os.path.join(args.out_directory, 'model_int8.tflite')
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Created int8 quantized model: {tflite_path}")
            
            # Test int8 quantized model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
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
