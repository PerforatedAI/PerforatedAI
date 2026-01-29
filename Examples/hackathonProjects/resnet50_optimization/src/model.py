import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes=100, arch='resnet50'):
    # Use standard ResNet but modify for CIFAR (32x32)
    # We pass weights=None to train from scratch (baseline requirement usually)
    
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
    elif arch == 'resnet34':
        model = models.resnet34(weights=None)
    elif arch == 'resnet50':
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Modify the first convergence layer for small images (32x32)
    # Standard ResNet uses 7x7 kernel, stride 2, padding 3 which is too aggressive for 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the maxpool layer (usually follows conv1)
    model.maxpool = nn.Identity()
    
    # Modify the final FC layer for the correct number of classes
    # ResNet18/34 have 512 input features, ResNet50/101/152 have 2048
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
