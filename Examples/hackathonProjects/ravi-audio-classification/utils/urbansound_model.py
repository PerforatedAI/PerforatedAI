"""
CNN14 model for UrbanSound8K classification (10 classes).
Optimized for use with PerforatedAI dendrites.
"""
import torch.nn as nn


class CNN14UrbanSound(nn.Module):
    """
    CNN14 implementation for UrbanSound8K, optimized for PerforatedAI.
    Based on the same architecture as ESC-50 but with 10 output classes.
    
    Uses Sequential blocks (Conv+BN+ReLU+Pool) which PAI can convert effectively.
    """
    
    def __init__(self, num_classes=10):
        super(CNN14UrbanSound, self).__init__()
        
        # Convolutional blocks using Sequential for PAI compatibility
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input: (batch, 1, 128, 173) for 4-second audio
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
