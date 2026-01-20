import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.transforms as T
import numpy as np
import argparse, os, sys, random, logging

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

import wandb
wandb.login()

sweep_config = {"method": "random"}
metric = {"name": "Final Max Val", "goal": "maximize"}
sweep_config["metric"] = metric
"""
before leaving - keep the same params and stuff but add the change for epoch and max counter to see if that second grpah with score by count looks better.
add a check for if the dendrite count is the max count or why it stopped

last night - saved combo as oldWorking.  added all the loop stuff from the old one and running with the same settings.
"""

parameters_dict = {

# Associated values for sweeping

# Dropout can be especially important if your training is already higher than your validation
"dropout": {"values": [0.1, 0.3, 0.5]},
# num conv layers
"num_conv": {"values": [1,2,3]},
# num linear layers
"num_linear": {"values": [1, 2]},
# network width
"network_width": {"values": [1/4, 1/2, 1, 2, 4]},

#"noise_std": {"values": [0,0.1,0.2,0.4]},
"noise_std": {"values": [0,0.1,0.2]},

"switch_speed": {"values": [10,25,100]},

#"learning_rate_multiplier": {"values": [0.1,0.5, 1, 5, 10]},
"learning_rate_multiplier": {"values": [1, 5, 10]},

"growth_mode": {"values": [0, 1, 2, 3, 4, 5]},


# Max Dendrites
"max_dendrites": {"values": [1, 2, 3, 4]},
# Used for all dendritic models:

# Speed of improvement required to prevent switching
# 0.1 means dendrites will switch after score stops improving by 10% over recent history
# This extra-early early-stopping sometimes enables high final scores as well as
# achieving top scores with fewer epochs
#"improvement_threshold": {"values": [0, 1, 2]},
"improvement_threshold": {"values": [1, 2]},
# Multiplier to initialize dendrite weights
"candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]},
# Forward function for dendrites
"pai_forward_function": {"values": [0, 1, 2]},

"conversion": {"values": [0, 1]},


# Only used with Perforated Backpropagation add-on

# A setting for dendritic connectivity
"dendrite_graph_mode": {"values": [0, 1]},
# A setting for dendritic learning rule
#"dendrite_update_mode": {"values": [0, 1]},
"dendrite_update_mode": {"values": [1]},
# No dendrites, GD dendrites or CC dendrites
"dendrite_mode": {"values": [0, 1, 2]},

}


sweep_config["parameters"] = parameters_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Running custom PyTorch models in Edge Impulse')
parser.add_argument('--data-directory', type=str, default='data')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default = 0.001)
parser.add_argument('--out-directory', type=str, default='out')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_name', type=str, required=True,
                    help="Pass a sweep ID to join an existing sweep.")


args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

# Split test set in half for test and validation
split_idx = len(X_test) // 2
X_val = X_test[:split_idx]
Y_val = Y_test[:split_idx]
X_test = X_test[split_idx:]
Y_test = Y_test[split_idx:]

X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
Y_val = torch.FloatTensor(Y_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)

# create data loaders
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

# Set determinism if needed
if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    drop_last=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
)

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, input_length, classes, num_conv=2, num_linear=1, width=1.0, linear_dropout=0.5, noise_std = 0.2, growth_mode=0):
        """
        Args:
            input_length: Length of input audio features
            classes: Number of output classes
            num_conv: Number of convolutional blocks (0-3)
            num_linear: Number of linear layers (1-3)
            width: Width multiplier for channels (0.0625-8.0)
            linear_dropout: Dropout rate for linear layers (0.0-1.0)
        """
        super(AudioClassifier, self).__init__()
        
        # Validate parameters
        assert 0 <= num_conv <= 4, "num_conv must be between 0 and 4"
        assert 1 <= num_linear <= 2, "num_linear must be between 1 and 2"
        assert 0.0625 <= width <= 8.0, "width must be between 0.0625 and 8.0"
        assert 0.0 <= linear_dropout <= 1.0, "linear_dropout must be between 0.0 and 1.0"
        
        self.input_length = input_length
        self.classes = classes
        self.num_conv = num_conv
        self.num_linear = num_linear
        self.width = width
        self.linear_dropout = linear_dropout
        
        # Data augmentation - GaussianNoise
        self.noise_std = noise_std
        
        # Reshape parameters
        self.channels = 1
        self.columns = 13
        self.rows = int(input_length / (self.columns * self.channels))
        
        # Calculate channel sizes with width multiplier
        if(growth_mode == 0):
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
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        self.conv_hooks = []
        
        if num_conv > 0:
            in_channels = self.channels
            for i in range(num_conv):
                out_channels = self.channel_sizes[i]
                
                conv_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding='same'
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.25)
                )
                
                self.conv_blocks.append(conv_block)
                
                # Register weight constraint hook
                conv_layer = conv_block[0]
                hook = conv_layer.register_forward_pre_hook(
                    lambda module, input: self._max_norm_constraint(module)
                )
                self.conv_hooks.append(hook)
                
                in_channels = out_channels
        
        # Calculate flattened size after conv layers
        if num_conv > 0:
            final_rows = self.rows // (2 ** num_conv)
            final_cols = self.columns // (2 ** num_conv)
            flattened_size = self.channel_sizes[num_conv - 1] * final_rows * final_cols
        else:
            # No conv layers, input goes directly to linear layers
            flattened_size = input_length
        
        # Build linear layers
        self.linear_layers = nn.ModuleList()
        
        # Calculate linear layer sizes
        linear_sizes = self._calculate_linear_sizes(flattened_size, classes, num_linear)
        
        for i in range(num_linear - 1):
            self.linear_layers.append(nn.Sequential(
                nn.Dropout(linear_dropout),
                nn.Linear(linear_sizes[i], linear_sizes[i + 1]),
                nn.ReLU()
            ))
        
        # Final output layer (with dropout before it)
        self.linear_layers.append(nn.Sequential(
            nn.Dropout(linear_dropout),
            nn.Linear(linear_sizes[-2], linear_sizes[-1])
        ))
    
    def _calculate_linear_sizes(self, input_size, output_size, num_layers):
        """Calculate intermediate layer sizes for linear layers"""
        if num_layers == 1:
            return [input_size, output_size]
        
        # Create logarithmically spaced sizes from input to output
        sizes = [input_size]
        
        # Calculate intermediate sizes
        log_start = torch.log(torch.tensor(float(input_size)))
        log_end = torch.log(torch.tensor(float(output_size)))
        
        for i in range(1, num_layers):
            ratio = i / num_layers
            log_size = log_start + (log_end - log_start) * ratio
            size = max(output_size, int(torch.exp(log_size).item()))
            sizes.append(size)
        
        sizes.append(output_size)
        return sizes
    
    def _max_norm_constraint(self, module, max_value=1.0):
        """Apply MaxNorm constraint to conv layers"""
        if hasattr(module, 'weight'):
            with torch.no_grad():
                norm = module.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                desired = torch.clamp(norm, max=max_value)
                module.weight *= (desired / (norm + 1e-8))
    
    def forward(self, x):
        # Add Gaussian noise during training
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Apply convolutional layers if any
        if self.num_conv > 0:
            # Reshape: (batch, input_length) -> (batch, channels, rows, columns)
            x = x.view(-1, self.channels, self.rows, self.columns)
            
            for conv_block in self.conv_blocks:
                x = conv_block(x)
            
            # Flatten
            x = x.view(x.size(0), -1)
        
        # Apply linear layers
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        
        return x

from types import SimpleNamespace

def parse_config_string(name_str, parameters_dict):
    prefix = "Dendrites-"
    if not name_str.startswith(prefix):
        raise ValueError("name_str must start with 'Dendrites-'")
    rest = name_str[len(prefix):]
    if "_" in rest:
        dend_token, rest = rest.split("_", 1)
    else:
        dend_token, rest = rest, ""
    result = {"dendrite_mode": dend_token}
    tokens = rest.split("_") if rest else []
    excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
    keys = [k for k in parameters_dict.keys() if k not in excluded]
    for key in keys:
        if not tokens:
            break
        result[key] = tokens.pop(0)
    if tokens:
        result["_extras"] = tokens
    return SimpleNamespace(**result)

def main():
    name_str = args.model_name
    config = parse_config_string(name_str, parameters_dict=parameters_dict)   
    
    
    if config.improvement_threshold == '0':
        thresh = [0.01, 0.001, 0.0001, 0]
    elif config.improvement_threshold == '1':
        thresh = [0.001, 0.0001, 0]
    elif config.improvement_threshold == '2':
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)
    GPA.pc.set_candidate_weight_initialization_multiplier(
        float(config.candidate_weight_initialization_multiplier)
    )
    if config.pai_forward_function == '0':
        pai_forward_function = torch.sigmoid
    elif config.pai_forward_function == '1':
        pai_forward_function = torch.relu
    elif config.pai_forward_function == '2':
        pai_forward_function = torch.tanh
    GPA.pc.set_pai_forward_function(pai_forward_function)
    if(bool(config.dendrite_graph_mode)):
        GPA.pc.set_dendrite_graph_mode(True)
    else:
        GPA.pc.set_dendrite_graph_mode(False)
    if config.conversion == '0':
        GPA.pc.set_modules_to_convert([nn.Conv2d, nn.Linear])
        GPA.pc.set_modules_to_track([])
    elif config.conversion == '1':
        GPA.pc.set_modules_to_convert([nn.Linear])
        GPA.pc.set_modules_to_track([nn.Conv2d])
    
    if(config.max_dendrites == '4'):
        max_dendrites = 100
    else:
        max_dendrites = int(config.max_dendrites)
    
    if config.dendrite_mode == '0':
        GPA.pc.set_max_dendrites(0)
    else:
        GPA.pc.set_max_dendrites(max_dendrites)
    if int(config.dendrite_mode) < 2:
        GPA.pc.set_perforated_backpropagation(False)
        GPA.pc.set_dendrite_update_mode(True)
    else:
        GPA.pc.set_perforated_backpropagation(True)
        if(int(config.dendrite_update_mode)):
            GPA.pc.set_dendrite_update_mode(True)
        else:
            GPA.pc.set_dendrite_update_mode(False)
    GPA.pc.set_initial_correlation_batches(40)
    excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
    keys = [k for k in parameters_dict.keys() if k not in excluded]
    #name_str = "Dendrites-" + str(wandb.config.dendrite_mode) + "_" + "_".join(
    #str(wandb.config[k]) for k in keys if k in wandb.config
    #)
    #run.name = name_str
    
    
    # Initialize model
    input_length = 624
    classes = 3
    model = AudioClassifier(input_length, classes, num_conv=int(config.num_conv), num_linear=int(config.num_linear), width=float(config.network_width), linear_dropout=float(config.dropout), noise_std=float(config.noise_std), growth_mode=int(config.growth_mode)).to(device)

    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_n_epochs_to_switch(int(config.switch_speed))
    GPA.pc.set_verbose(True)
    
    preface = "trained/"
    
    if(int(config.dendrite_mode) < 2):
        model = UPA.initialize_pai(model, save_name=name_str)
        model = UPA.load_system(model, preface + name_str, 'best_model', True)
        param_count = UPA.count_params(model)
        dendrite_count = GPA.pai_tracker.member_vars["num_dendrites_added"]
    else:
        from perforatedbp import network_pbp as PBN
        model = PBN.load_pai_model(model, preface + name_str + '/best_model_pai.pt')   
        parameters = model.named_parameters()
        unique_params = {p.data_ptr(): p for name, p in parameters if 'parent_module' not in name}.values()
        param_count = sum(p.numel() for p in unique_params)
        dendrite_count = len(model.linear_layers[0][1].layer_array) - 1
    print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    def test(model, loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Convert one-hot labels to class indices if needed
                if labels.dim() > 1 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)
                
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    val_loss, val_acc = test(model, val_loader, criterion, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"val_acc: {val_acc}")
    print(f"test_acc: {test_acc}")
    print(f"params: {param_count}")
    print(f"Dendrite Count: {dendrite_count}")
    
    for i, block in enumerate(model.conv_blocks):
        conv = block[0].main_module
        if conv.padding == 'same':
            # Calculate explicit padding
            if isinstance(conv.kernel_size, tuple):
                padding = tuple((k - 1) // 2 for k in conv.kernel_size)
            else:
                padding = (conv.kernel_size - 1) // 2
            
            # Set the explicit padding
            conv.padding = padding
            print(f"Updated padding for conv_blocks[{i}][0].main_module to {padding}")
    
    torch.onnx.export(model.cpu(),
                  torch.randn((32, 624)),
                  os.path.join(args.model_name + '.model.onnx'),
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])
    

if __name__ == "__main__":
    main()
"""
def run():
    try:
        with wandb.init(config=sweep_config) as run:
            main(run)
    except Exception:
        import pdb
        pdb.post_mortem()
if __name__ == "__main__":
    # Count is how many runs to perform.
    project="Dendritic Edge Impulse Audio - Combo"
    if args.sweep_id == "main":
        sweep_id = wandb.sweep(sweep_config, project=project)
        print("\nInitialized sweep. Use --sweep_id", sweep_id, "to join on other machines.\n")
        # Optionally run the agent on this machine as well
        wandb.agent(sweep_id, run, count=400)
    else:
        # Join the existing sweep as an agent
        wandb.agent(args.sweep_id, run, count=400, project=project)
"""

# For quantization (PyTorch equivalent)
# disable_per_channel_quantization = False
# If you need quantization:
# model_quantized = torch.quantization.quantize_dynamic(
#     model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
# )
