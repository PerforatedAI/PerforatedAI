import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T

# Assuming you have PyTorch datasets ready
# train_dataset and validation_dataset should be torch.utils.data.Dataset objects

EPOCHS = args.epochs or 100
LEARNING_RATE = args.learning_rate or 0.005
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32

# Set determinism if needed
if ENSURE_DETERMINISM:
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=not ENSURE_DETERMINISM,
    drop_last=False
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioClassifier(input_length, classes).to(device)

# Optimizer (Adam with same parameters as Keras)
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999)
)

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

def validate(model, loader, criterion, device):
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

# Training loop
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, validation_loader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{EPOCHS}] '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

torch.onnx.export(model.cpu(),
                  torch.randn(tuple([1] + list(X_train.shape[1:]))),
                  os.path.join(args.out_directory, 'model.onnx'),
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

# Save model
torch.save(model.state_dict(), 'model_weights.pth')

# For quantization (PyTorch equivalent)
# disable_per_channel_quantization = False
# If you need quantization:
# model_quantized = torch.quantization.quantize_dynamic(
#     model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
# )
