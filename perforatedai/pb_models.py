# Copyright (c) 2025 Perforated AI

import math
import pdb
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models.resnet as resnet_pt

from perforatedai import pb_globals as PBG

"""
Details on processors found in customization.md in API
But they exist to enable simplicity in adding dendrites to modules
where forward is not one tensor in and one tensor out

Even though this is one class, what really happens is that the main module has
one instance, which will use post_n1 and post_n2 and then each new Dendrite
node gets a unique separate individual instance to use pre_d and post_d
"""


# General multi output processor for any number that ignores later ones
class MultiOutputProcessor:
    """Processor for handling multiple outputs, ignoring later ones."""
    
    def post_n1(self, *args, **kwargs):
        """Extract first output and store extra outputs."""
        out = args[0][0]
        extra_out = args[0][1:]
        self.extra_out = extra_out
        return out
        
    def post_n2(self, *args, **kwargs):
        """Combine output with stored extra outputs."""
        out = args[0]
        if isinstance(self.extra_out, tuple):
            return (out,) + self.extra_out
        else:
            return (out,) + (self.extra_out,)
            
    def pre_d(self, *args, **kwargs):
        """Pass through arguments unchanged for dendrite preprocessing."""
        return args, kwargs
        
    def post_d(self, *args, **kwargs):
        """Extract first output for dendrite postprocessing."""
        out = args[0][0]
        return out
        
    def clear_processor(self):
        """Clear stored processor state."""
        if hasattr(self, 'extra_out'):
            delattr(self, 'extra_out')


# LSTMCellProcessor defined here to use as example of how to setup processing
# functions for more complex situations
class LSTMCellProcessor:
    """Processor for LSTM cells to handle hidden and cell states."""
    
    # The neuron does eventually need to return h_t and c_t, but h_t gets
    # modified by the Dendrite nodes first so it needs to be extracted in
    # post_n1, and then gets added back in post_n2
    
    def post_n1(self, *args, **kwargs):
        """Extract hidden state and store cell state temporarily.
        
        post_n1 is called right after the main module is called before any 
        Dendrite processing. It should return only the part of the output 
        that you want to do Dendrite learning for.
        """
        h_t = args[0][0]
        c_t = args[0][1]
        # Store the cell state temporarily and just use the hidden state
        # to do Dendrite functions
        self.c_t_n = c_t
        return h_t
        
    def post_n2(self, *args, **kwargs):
        """Combine modified hidden state with stored cell state.
        
        post_n2 is called right before passing final value forward, should 
        return everything that gets returned from main module.
        h_t at this point has been modified with Dendrite processing.
        """
        h_t = args[0]
        return h_t, self.c_t_n
    
    def pre_d(self, *args, **kwargs):
        """Filter input for Dendrite processing.
        
        Input to pre_d will be (input, (h_t, c_t))
        pre_d does filtering to make sure Dendrite is getting the right input.
        This typically would be done in the training loop.
        For example, with an LSTM this is where you check if its the first 
        iteration or not and either pass the Dendrite the regular args to the 
        neuron or pass the Dendrite its own internal state.
        """
        h_t = args[1][0]
        # If its the initial step then just use the normal input and zeros
        if h_t.sum() == 0:
            return args, kwargs
        # If its not the first one then return the input it got with its own
        # h_t and c_t to replace neurons
        else:
            return (args[0], (self.h_t_d, self.c_t_d)), kwargs
        
    def post_d(self, *args, **kwargs):
        """Process Dendrite output and save state for next iteration.
        
        For post processing post_d just getting passed the output, which is 
        (h_t, c_t). Then it wants to only pass along h_t as the output for the 
        function to be passed to the neuron while retaining both h_t and c_t.
        post_d saves what needs to be saved for next time and passes forward 
        only the Dendrite part that will be added to the neuron.
        """
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t
    
    def clear_processor(self):
        """Clear all saved processor state."""
        if hasattr(self, 'h_t_d'):
            delattr(self, 'h_t_d')
        if hasattr(self, 'c_t_d'):
            delattr(self, 'c_t_d')
        if hasattr(self, 'c_t_n'):
            delattr(self, 'c_t_n')


class ResNetPAI(nn.Module):
    """PB-compatible ResNet wrapper.
    
    All normalization layers should be wrapped in a PAISequential, or other 
    wrapped module. When working with a predefined model the following shows 
    an example of how to create a module for MODULES_TO_REPLACE.
    """
    
    def __init__(self, other_resnet):
        """Initialize ResNetPAI from existing ResNet model."""
        super(ResNetPAI, self).__init__()
        
        # For the most part, just copy the exact values from the original module
        self._norm_layer = other_resnet._norm_layer
        self.inplanes = other_resnet.inplanes
        self.dilation = other_resnet.dilation
        self.groups = other_resnet.groups
        self.base_width = other_resnet.base_width

        # For the component to be changed, define a PAISequential with the old
        # modules included
        self.b1 = PBG.PAISequential([
            other_resnet.conv1,
            other_resnet.bn1]
        )

        self.relu = other_resnet.relu
        self.maxpool = other_resnet.maxpool
        
        for i in range(1, 5):
            layer_name = 'layer' + str(i)
            original_layer = getattr(other_resnet, layer_name)
            pb_layer = self._make_layer_pb(original_layer, other_resnet, i)
            setattr(self, layer_name, pb_layer)
            
        self.avgpool = other_resnet.avgpool
        self.fc = other_resnet.fc

    def _make_layer_pb(self, other_block_set, other_resnet, block_id):
        """Convert ResNet layer blocks to PB-compatible format.
        
        This might not be needed now that the blocks are being converted.
        """
        layers = []
        for i in range(len(other_block_set)):
            block_type = type(other_block_set[i])
            if block_type == resnet_pt.BasicBlock:
                layers.append(other_block_set[i])
            elif block_type == resnet_pt.Bottleneck:
                layers.append(other_block_set[i])
            else:
                print('Your resnet uses a block type that has not been '
                      'accounted for. Customization might be required.')
                layer_name = 'layer' + str(block_id)
                print(type(getattr(other_resnet, layer_name)))
                pdb.set_trace()
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """Implementation of the forward pass."""
        # Modified b1 rather than conv1 and bn1
        x = self.b1(x)
        # Rest of forward remains the same
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        """Forward pass through the network."""
        return self._forward_impl(x)
