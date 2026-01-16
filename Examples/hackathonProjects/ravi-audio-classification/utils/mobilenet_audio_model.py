"""
MobileNetV2 adapted for audio classification.
Uses pretrained ImageNet weights and fine-tunes on mel-spectrograms.
"""
import torch
import torch.nn as nn
from torchvision import models


class MobileNetAudio(nn.Module):
    """
    MobileNetV2 adapted for audio classification.
    Pretrained on ImageNet, fine-tuned on mel-spectrograms.
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNetAudio, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            print("✓ Loaded MobileNetV2 pretrained on ImageNet")
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)
        
        # Modify first conv layer to accept 1-channel input (grayscale mel-spectrogram)
        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1,  # 1 channel for grayscale mel-spectrogram
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize new conv layer weights by averaging RGB channels
        if pretrained:
            with torch.no_grad():
                self.mobilenet.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace classifier for our number of classes
        self.mobilenet.classifier[1] = nn.Linear(
            self.mobilenet.last_channel,
            num_classes
        )
    
    def forward(self, x):
        return self.mobilenet(x)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
