"""
CNN Model for Audio Classification on Mel-Spectrograms
"""
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    """
    Convolutional Neural Network for audio classification.
    
    Processes mel-spectrograms (128 mel bands x ~216 time steps) and outputs
    class probabilities for 50 environmental sound categories.
    """
    
    def __init__(self, num_classes=50, input_channels=1):
        super(AudioCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate size after 3 poolings: 128 x 216 -> 16 x 27
        self.fc1 = nn.Linear(128 * 16 * 27, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Input: (batch, 1, 128, 216)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (batch, 32, 64, 108)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (batch, 64, 32, 54)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (batch, 128, 16, 27)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
