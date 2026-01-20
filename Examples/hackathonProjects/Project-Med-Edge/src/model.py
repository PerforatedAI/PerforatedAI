import torch
import torch.nn as nn

class DermoNet_Edge(nn.Module):
    """
    Micro-CNN for Blood Cell Classification (Edge Deployment)
    
    Uses TWO DROPOUT LAYERS like the winning MNIST submission:
    - dropout1 = 0.25 after conv layers
    - dropout2 = 0.5 after fc1
    """
    def __init__(self, channels=3, classes=7, dropout1=0.25, dropout2=0.5):
        super(DermoNet_Edge, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Dropout after features (like MNIST dropout1)
        self.dropout1 = nn.Dropout(dropout1)
        
        # 28 -> 14 -> 7 -> 3
        self.flat_dim = 32 * 3 * 3
        
        self.fc1 = nn.Linear(self.flat_dim, 32)
        
        # Dropout after fc1 (like MNIST dropout2)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.fc2 = nn.Linear(32, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout1(x)  # First dropout after conv
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)  # Second dropout after fc1
        x = self.fc2(x)
        return x
