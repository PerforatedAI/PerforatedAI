import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.transforms as T
import numpy as np
import argparse, os, sys, random, logging

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Running custom PyTorch models in Edge Impulse')
parser.add_argument('--data-directory', type=str, default='data')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default = 0.001)
parser.add_argument('--out-directory', type=str, default='out')
parser.add_argument('--seed', type=int, default=0)
args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)

# create data loaders
train_dataset = TensorDataset(X_train, Y_train)
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

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False
)

# Model architecture
class AudioClassifier(nn.Module):
    def __init__(self, input_length, classes):
        super(AudioClassifier, self).__init__()
        
        self.input_length = input_length
        self.classes = classes
        
        # Data augmentation - GaussianNoise
        self.noise_std = 0.2
        
        # Reshape parameters
        self.channels = 1
        self.columns = 13
        self.rows = int(input_length / (self.columns * self.channels))
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=8,
            kernel_size=3,
            padding='same'
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding='same'
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Calculate flattened size after conv layers
        # After 2 maxpool layers with stride 2
        final_rows = self.rows // 4
        final_cols = self.columns // 4
        flattened_size = 16 * final_rows * final_cols
        
        # Fully connected layer
        self.fc = nn.Linear(flattened_size, classes)
        
        # Apply weight constraints (MaxNorm)
        self._apply_weight_constraints()
    
    def _apply_weight_constraints(self):
        """Apply MaxNorm constraint to conv layers"""
        def max_norm_constraint(module, max_value=1.0):
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    norm = module.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=max_value)
                    module.weight *= (desired / (norm + 1e-8))
        
        # Register hooks for weight constraints
        self.conv1.register_forward_pre_hook(
            lambda module, input: max_norm_constraint(module)
        )
        self.conv2.register_forward_pre_hook(
            lambda module, input: max_norm_constraint(module)
        )
    
    def forward(self, x):
        # Add Gaussian noise during training
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Reshape: (batch, input_length) -> (batch, channels, rows, columns)
        x = x.view(-1, self.channels, self.rows, self.columns)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x

# Initialize model
input_length = 624
classes = 3
model = AudioClassifier(input_length, classes).to(device)

GPA.pc.set_testing_dendrite_capacity(False)
GPA.pc.set_n_epochs_to_switch(50)

GPA.pc.set_verbose(True) # This gives more details when printing the model

model = UPA.initialize_pai(model)
print(model)

# Optimizer (Adam with same parameters as Keras)
optimizer = optim.Adam(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.999)
)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Convert one-hot labels to class indices if needed
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

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

training = False

if(training):
    # Training loop
    epoch = -1
    while True:
        epoch += 1
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        GPA.pai_tracker.add_extra_score(train_acc, 'Train')
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(test_acc, 
        model) # .module if its a dataParallel
        model.to(device)
        if(training_complete):
            break
        elif(restructured):
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999)
            )
            GPA.pai_tracker.set_optimizer_instance(optimizer)

        print(f'Epoch [{epoch+1}/{args.epochs}] '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    print(model)
else:
    model = UPA.load_system(model, 'PAI', 'best_model', True)

import pdb; pdb.set_trace()

torch.onnx.export(model.cpu(),
                  torch.randn(tuple([1] + list(X_train.shape[1:]))),
                  os.path.join(args.out_directory, 'model.onnx'),
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

# For quantization (PyTorch equivalent)
# disable_per_channel_quantization = False
# If you need quantization:
# model_quantized = torch.quantization.quantize_dynamic(
#     model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
# )
