"""
EfficientNet Pre-FC wrapper for PerforatedAI.

This module provides a wrapper class that adds a perforable pre-FC layer
before the final classifier in EfficientNet models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from perforatedai import globals_perforatedai as GPA


__all__ = ["EfficientNetPAI"]


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


class EfficientNetPAI(nn.Module):
    """PerforatedAI-compatible EfficientNet wrapper.
    
    Adds a pre_fc layer before the final classifier that can be perforated.
    The pre_fc layer maintains the same dimensionality as the features output.
    
    Architecture:
        features -> avgpool -> flatten -> pre_fc -> ReLU -> classifier
    """

    def __init__(self, other_efficientnet):
        """Initialize EfficientNetPAI from existing EfficientNet model.

        Parameters
        ----------
        other_efficientnet : torchvision.models.efficientnet.EfficientNet
            An existing EfficientNet model to convert to PAI-compatible format.
        """
        super(EfficientNetPAI, self).__init__()

        # Convert all inplace operations to non-inplace for PerforatedAI compatibility
        other_efficientnet = convert_inplace_to_false(other_efficientnet)

        # Copy the exact components from the original module
        self.features = other_efficientnet.features
        self.avgpool = other_efficientnet.avgpool
        
        # Determine the feature dimension from the classifier
        # EfficientNet classifier is Sequential(Dropout, Linear(...))
        # The Linear layer's in_features gives us the dimension
        if isinstance(other_efficientnet.classifier, nn.Sequential):
            for module in other_efficientnet.classifier:
                if isinstance(module, nn.Linear):
                    fc_in_features = module.in_features
                    break
        else:
            # Fallback if classifier structure is different
            fc_in_features = other_efficientnet.classifier.in_features
        
        # Create pre_fc layer with dimensions matching features output
        self.pre_fc = nn.Linear(fc_in_features, fc_in_features)
        
        # Keep the original classifier
        self.classifier = other_efficientnet.classifier

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
