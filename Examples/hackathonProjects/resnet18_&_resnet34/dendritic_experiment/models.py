import torch
import torch.nn as nn
from torchvision import models

def get_resnet18(pretrained=True, num_classes=100):
    """
    Returns a ResNet18 model.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Modify the final fully connected layer for CIFAR-100
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_resnet34(pretrained=True, num_classes=100):
    """
    Returns a ResNet34 model.
    """
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)
    
    # Modify the final fully connected layer for CIFAR-100
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

