# Copyright (c) 2025 Perforated AI

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models.resnet as resnetPT
import math
import pdb
from itertools import chain
from perforatedai import pb_globals as PBG

'''
Details on processors found in customization.md in API
But they exist to enable simplcity in adding dendrites to modules
where forward is not one tensor in and one tensor out

Even though this is one class, what really happens is that the main module has one instance,
which will use post_n1 and post_n2 and then each new Dendrite node gets a unique sepearte individual isntance to use pre_d and post_d
'''

# General multi output processor for any number that ignores later ones
class multiOutputProcesser():
    def post_n1(self, *args, **kwargs):
        out = args[0][0]
        extraOut = args[0][1:]
        self.extraOut = extraOut
        return out
    def post_n2(self, *args, **kwargs):
        out = args[0]
        if(type(self.extraOut) == tuple):
            return (out,) + self.extraOut
        else:
            return (out,) + (self.extraOut,)
    def pre_d(self, *args, **kwargs):
        return args, kwargs
    def post_d(self, *args, **kwargs):
        out = args[0][0]
        return out
    def clear_processor(self):
        if hasattr(self, 'extraOut'):
            delattr(self, 'extraOut')

#LSTMCellProcessor defined here to use as example of how to setup processing functions for more complex situations
class LSTMCellProcessor():
    # The neuron does eventually need to return h_t and c__t, but h_t gets modified py the Dendrite
    # nodes first so it needs to be extracted in post_n1, and then gets added back in post_n2
    
    # post_n1 is called right after the main module is called before any Dendrite processing. 
    # It should return only the part of the output that you want to do Dendrite learning for.  
    def post_n1(self, *args, **kwargs):
        h_t = args[0][0]
        c_t = args[0][1]
        #store the cell state temporarily and just use the hidden state to do Dendrite functions
        self.c_t_n = c_t
        return h_t
    # post_n2 is called right before passing final value forward, should return everything that gets returned from main module
    # h_t at this point has been modified with Dendrite processing
    def post_n2(self, *args, **kwargs):
        h_t = args[0]
        return h_t, self.c_t_n
    
    # Input to pre_d will be (input, (h_t, c_t))
    # pre_d does filtering to make sure Dendrite is getting the right input.
    # This typically would be done in the training loop.  
    # For example, with an LSTM this is where you check if its the first iteration or not and either pass the Dendrite
    # the regular args to the neuron or pass the Dendrite its own internal state.
    def pre_d(self, *args, **kwargs):
        h_t = args[1][0]
        # If its the initial step then just use the normal input and zeros
        if(h_t.sum() == 0):
            return args, kwargs
        # If its not the first one then return the input it got with its own h_t and c_t to replace neuronss
        else:
            return (args[0], (self.h_t_d, self.c_t_d)), kwargs
        
    # For post processsing post_d just getting passed the output, which is (h_t,c_t).
    # Then it wants to only pass along h_t as the output for the function to be passed to the neuron while retaining both h_t and c_t.
    # post_d saves what needs to be saved for next time and passes forward only the Dendrite part that will be added to the neuron
    def post_d(self, *args, **kwargs):
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t
    
    # clear_processor must clear all saved values
    def clear_processor(self):
        if hasattr(self, 'h_t_d'):
            delattr(self, 'h_t_d')
        if hasattr(self, 'c_t_d'):
            delattr(self, 'c_t_d')
        if hasattr(self, 'c_t_n'):
            delattr(self, 'c_t_n')


# All normalization layers should be wrapped in a PBSequential, or other wrapped modeule
# When working with a predefined model the following shows an example of how to create a module for modulesToReplace
class ResNetPB(nn.Module):
    def __init__(self, otherResNet):
        super(ResNetPB, self).__init__()
        
        # For the most part, just copy the exact values from the original module
        self._norm_layer = otherResNet._norm_layer
        self.inplanes = otherResNet.inplanes
        self.dilation = otherResNet.dilation
        self.groups = otherResNet.groups
        self.base_width = otherResNet.base_width

        # For the component to be changed, define a PBSequential with the old modules included
        self.b1 = PBG.PBSequential([
             otherResNet.conv1,
             otherResNet.bn1]
        )

        self.relu = otherResNet.relu
        self.maxpool = otherResNet.maxpool
        for i in range(1,5):
            setattr(self, 'layer' + str(i), self._make_layerPB(getattr(otherResNet,'layer' + str(i)),otherResNet, i))
        self.avgpool = otherResNet.avgpool
        self.fc = otherResNet.fc

    # This might not be needed now that the blocks are being converted
    def _make_layerPB(self, otherBlockSet,otherResNet, blockID):
        layers = []
        for i in range(len(otherBlockSet)):
            if(type(otherBlockSet[i]) == resnetPT.BasicBlock):
                layers.append((otherBlockSet[i]))
            elif(type(otherBlockSet[i]) == resnetPT.Bottleneck):
                layers.append((otherBlockSet[i]))
            else:
                print('your resnet uses a block type that has not been accounted for.  customization might be required')
                print(type(getattr(otherResNet,'layer' + str(blockID))))
                pdb.set_trace()
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Modified b1 rather chan conv1 and bn 1
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
        return self._forward_impl(x)
