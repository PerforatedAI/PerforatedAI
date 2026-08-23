"""
MobileNetV3 Pre-FC wrapper for PerforatedAI.

This module provides a wrapper class that adds a perforable pre-FC layer
before the final classifier in MobileNetV3 models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from perforatedai import globals_perforatedai as GPA


__all__ = ["MobileNetV3PAI"]


def convert_inplace_to_false(model):
    """Convert all inplace operations in model to non-inplace.
    
    This is necessary for PerforatedAI gradient tracking compatibility.
    
    Parameters
    ----------
    model : nn.Module
        The model to modify
    """
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
    return model


class MobileNetV3PAI(nn.Module):
    """PerforatedAI-compatible MobileNetV3 wrapper.
    
    Adds a pre_fc layer before the final classifier that can be perforated.
    The pre_fc layer maintains the same dimensionality as the features output.
    
    Architecture:
        features -> avgpool -> flatten -> pre_fc -> ReLU -> classifier
    """

    def __init__(self, other_mobilenet):
        """Initialize MobileNetV3PAI from existing MobileNetV3 model.

        Parameters
        ----------
        other_mobilenet : torchvision.models.mobilenetv3.MobileNetV3
            An existing MobileNetV3 model to convert to PAI-compatible format.
        """
        super(MobileNetV3PAI, self).__init__()

        # Convert all inplace operations to non-inplace for PerforatedAI compatibility
        other_mobilenet = convert_inplace_to_false(other_mobilenet)

        # Copy the exact components from the original module
        self.features = other_mobilenet.features
        self.avgpool = other_mobilenet.avgpool
        
        # Determine the feature dimension from the classifier
        # MobileNetV3 classifier is Sequential(Linear, Hardswish, Dropout, Linear)
        # The first Linear layer's in_features gives us the dimension
        if isinstance(other_mobilenet.classifier, nn.Sequential):
            for module in other_mobilenet.classifier:
                if isinstance(module, nn.Linear):
                    fc_in_features = module.in_features
                    break
        else:
            # Fallback if classifier structure is different
            fc_in_features = other_mobilenet.classifier.in_features
        
        # Create pre_fc layer with dimensions matching features output
        self.pre_fc = nn.Linear(fc_in_features, fc_in_features)
        
        # Keep the original classifier
        self.classifier = other_mobilenet.classifier

    def _forward_impl(self, x):
        """Implementation of the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the network.

        Returns
        -------
        torch.Tensor
            Output tensor from the network.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Add pre_fc layer with ReLU activation
        x = self.pre_fc(x)
        x = F.relu(x)
        
        x = self.classifier(x)

        return x

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the network.

        Returns
        -------
        torch.Tensor
            Output tensor from the network.
        """
        return self._forward_impl(x)
