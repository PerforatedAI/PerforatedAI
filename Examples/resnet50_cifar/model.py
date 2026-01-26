import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=100):
    # Use standard ResNet50 but modify for CIFAR (32x32)
    # We pass weights=None to train from scratch (baseline requirement usually)
    # If the hackathon allows pretrained, we could start with weights='DEFAULT'
    model = models.resnet50(weights=None)
    
    # Modify the first convergence layer for small images (32x32)
    # Standard ResNet uses 7x7 kernel, stride 2, padding 3 which is too aggressive for 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the maxpool layer (usually follows conv1)
    model.maxpool = nn.Identity()
    
    # Modify the final FC layer for the correct number of classes
    model.fc = nn.Linear(2048, num_classes)
    
    return model
