# Copyright (c) 2025 Perforated AI

import math
import pdb
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models.resnet as resnet_pt
from abc import ABC, abstractmethod

from perforatedai import globals_perforatedai as GPA

"""
Details on processors can be found in customization.md in the API directory.

They exist to enable simplicity in adding dendrites to modules where
forward() is not one tensor in and one tensor out.

The main module has one instance, which uses post_n1 and post_n2
and each new Dendrite node gets a unique instance to use pre_d and post_d.
"""


class PAIProcessor(ABC):
    """
    Abstract base class for processing neuron and dendrite operations.

    Processors handle state management and data flow between neurons and
    dendrites, allowing for custom pre/post processing of modules which have
    multiple inputs and outputs, rather than the default single tensor input/output.
    Subclasses should implement the five core processing methods to handle
    their specific state management needs.
    """

    @abstractmethod
    def post_n1(self, *args, **kwargs):
        """
        Post-process neuron output before dendrite processing.

        Called immediately after the main module/neuron is executed and before
        any dendrite processing occurs. This method should extract and return
        only the tensor of the neuron output that should be seen by
        dendrite operations.

        Parameters
        ----------
        *args : tuple
            Positional arguments, typically containing the neuron output.
        **kwargs : dict
            Keyword arguments from the neuron output.

        Returns
        -------
        Any
            The filtered output to be passed to dendrite processing.
        """
        pass

    @abstractmethod
    def post_n2(self, *args, **kwargs):
        """
        Post-process dendrite-modified output before final return.

        Called after dendrite processing is complete and before passing the
        final value forward in the network. This method should combine the
        dendrite-modified output with any stored state to produce the complete
        output that matches the expected format of the main module.

        Parameters
        ----------
        *args : tuple
            Positional arguments containing the dendrite-modified output.
        **kwargs : dict
            Keyword arguments from the processing chain.

        Returns
        -------
        Any
            The complete output in the format expected by downstream components.
        """
        pass

    @abstractmethod
    def pre_d(self, *args, **kwargs):
        """
        Pre-process input before dendrite operations.

        Filters and prepares inputs for dendrite processing. This method handles
        special cases such as initial time steps vs. subsequent iterations,
        ensuring dendrites receive the appropriate inputs (e.g., external inputs
        vs. internal recurrent state).

        Parameters
        ----------
        *args : tuple
            Positional arguments containing inputs to the PAI module.
        **kwargs : dict
            Keyword arguments containing inputs to the PAI module.

        Returns
        -------
        tuple
            A tuple of (processed_args, processed_kwargs) to pass to dendrite.
        """
        pass

    @abstractmethod
    def post_d(self, *args, **kwargs):
        """
        Post-process dendrite output and manage state.

        Processes the output from dendrite operations, storing any state needed
        for future iterations and returning only the portion that should be
        combined with the neuron output. E.g. this is where recurrent state is
        saved for the next time step.

        Parameters
        ----------
        *args : tuple
            Positional arguments containing the dendrite output.
        **kwargs : dict
            Keyword arguments from the dendrite output.

        Returns
        -------
        Any
            The filtered dendrite output to be added to the neuron output.
        """
        pass

    @abstractmethod
    def clear_processor(self):
        """
        Clear all internal processor state.

        Resets the processor by removing all stored state variables. Must
        be called before saving or safe_tensors will run into errors.
        Implementations should safely check for attribute existence before
        deletion to avoid errors.
        """
        pass


# General multi output processor for any number that ignores later ones
class MultiOutputProcessor:
    """Processor for handling multiple outputs, ignoring later ones."""

    def post_n1(self, *args, **kwargs):
        """Saves extra outputs and returns the first output.

        Parameters
        ----------
        *args : tuple
            Contains the modules output tuple.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            The first tensor of the tuple
        """
        out = args[0][0]
        extra_out = args[0][1:]
        self.extra_out = extra_out
        return out

    def post_n2(self, *args, **kwargs):
        """Combine output with stored extra outputs.

        Parameters
        ----------
        *args : torch.tensor
            The first tensor combined with dendrite output.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        tuple
            The recombined output tuple wth the new first output modified
        """
        out = args[0]
        if isinstance(self.extra_out, tuple):
            return (out,) + self.extra_out
        else:
            return (out,) + (self.extra_out,)

    def pre_d(self, *args, **kwargs):
        """Pass through arguments unchanged for dendrite preprocessing.

        Parameters
        ----------
        *args : tuple
            Positional arguments containing inputs to the PAI module.
        **kwargs : dict
            Keyword arguments containing inputs to the PAI module.

        Returns
        -------
        args : tuple
            Positional arguments containing inputs to the PAI module.
        kwargs : dict
            Keyword arguments containing inputs to the PAI module.
        """
        return args, kwargs

    def post_d(self, *args, **kwargs):
        """Extract first output for dendrite postprocessing.

        Parameters
        ----------
        *args : tuple
            Contains the dendrite modules output tuple.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            The first tensor of the tuple
        """
        out = args[0][0]
        return out

    def clear_processor(self):
        """Clear stored processor state."""

        if hasattr(self, "extra_out"):
            delattr(self, "extra_out")

class LSTMCellProcessor(PAIProcessor):
    """Processor for LSTM cells to handle hidden and cell states."""

    def post_n1(self, *args, **kwargs):
        """
        Extract hidden state from LSTM output for dendrite processing.

        Separates the hidden state (h_t) from the cell state (c_t) in the
        LSTM output tuple. Stores the cell state temporarily since only the
        hidden state should be modified by dendrites.

        Parameters
        ----------
        *args : tuple
            Contains LSTM output tuple (h_t, c_t) as first element.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Hidden state h_t to be passed to dendrite processing.
        """
        h_t = args[0][0]
        c_t = args[0][1]
        # Store the cell state temporarily and just use the hidden state
        # to do Dendrite functions
        self.c_t_n = c_t
        return h_t

    def post_n2(self, *args, **kwargs):
        """
        Recombine dendrite-modified hidden state with cell state.

        Takes the hidden state that has been modified by dendrite operations
        and combines it with the stored cell state to produce the complete
        LSTM output tuple.

        Parameters
        ----------
        *args : tuple
            Contains the dendrite-modified hidden state h_t.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        tuple
            Complete LSTM output (h_t, c_t) where h_t has been modified.
        """
        h_t = args[0]
        return h_t, self.c_t_n

    def pre_d(self, *args, **kwargs):
        """
        Filter LSTMCell input for dendrite based on initialization state.

        Checks if this is the first time step (all zeros in h_t) or a
        subsequent step. For the first step, passes through the original
        inputs. For subsequent steps, replaces the neuron's hidden state
        with the dendrite's own internal state from the previous iteration.

        Parameters
        ----------
        *args : tuple
            Contains (input, (h_t, c_t)) where input is the external input
            and (h_t, c_t) is the neuron's recurrent state.
        **kwargs : dict
            Keyword arguments to pass through.

        Returns
        -------
        tuple
            ((processed_input, processed_state), kwargs) for dendrite call.
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
        """
        Extract and store dendrite's LSTM state for next iteration.

        Separates the dendrite's hidden and cell states from its output tuple,
        stores both for use in the next time step, and returns only the hidden
        state to be combined with the neuron's output.

        Parameters
        ----------
        *args : tuple
            Contains dendrite LSTM output tuple (h_t, c_t).
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Hidden state h_t to be added to the neuron output.
        """
        h_t = args[0][0]
        c_t = args[0][1]
        self.h_t_d = h_t
        self.c_t_d = c_t
        return h_t

    def clear_processor(self):
        """
        Clear all stored LSTM states.

        Removes dendrite hidden state (h_t_d), dendrite cell state (c_t_d),
        and temporarily stored neuron cell state (c_t_n). Safe to call even
        if attributes don't exist.
        """
        if hasattr(self, "h_t_d"):
            delattr(self, "h_t_d")
        if hasattr(self, "c_t_d"):
            delattr(self, "c_t_d")
        if hasattr(self, "c_t_n"):
            delattr(self, "c_t_n")



class LSTMProcessor(PAIProcessor):
    """Processor for LSTM to handle hidden and output states."""

    def post_n1(self, *args, **kwargs):
        """
        Extract hidden state from LSTM output for dendrite processing.

        Separates the hidden state from the output in the
        LSTM output tuple. Stores the hidden state temporarily since only the
        output state should be modified by dendrites.

        Parameters
        ----------
        *args : tuple
            Contains LSTM output tuple (output, hidden) as first element.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Output state to be passed to dendrite processing.
        """
        output = args[0][0]
        hidden = args[0][1]
        # Store the hidden state temporarily and just use the output state
        # to do Dendrite functions
        self.hidden_n = hidden
        return output

    def post_n2(self, *args, **kwargs):
        """
        Recombine dendrite-modified output with hidden tuple.

        Takes the output state that has been modified by dendrite operations
        and combines it with the stored hidden state to produce the complete
        LSTM output tuple.

        Parameters
        ----------
        *args : tuple
            Contains the dendrite-modified output state.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        tuple
            Complete LSTM output (output, hidden) where output has been modified.
        """
        output = args[0]
        return output, self.hidden_n

    def pre_d(self, *args, **kwargs):
        """
        LSTM input is just the tensor which also goes to the dendrite

        Parameters
        ----------
        *args : 
            Input tensor
        **kwargs : dict
            Empty

        Returns
        -------
        tuple
            (output, hidden)
        """
        return args, kwargs
        
    def post_d(self, *args, **kwargs):
        """
        Extract dendrite's output to combine.

        Parameters
        ----------
        *args : tuple
            Contains dendrite LSTM output tuple (output, hidden).
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Output state to be added to the neuron output.
        """
        output = args[0][0]
        hidden = args[0][1]
        return output

    def clear_processor(self):
        """
        Clear all stored LSTM states.

        """
        if hasattr(self, "hidden_n"):
            delattr(self, "hidden_n")


class LSTMProcessorLastHidden(PAIProcessor):
    """Processor for LSTM to forward the last hidden."""

    def post_n1(self, *args, **kwargs):
        """
        Extract the last hidden to combine with dendrites

        Parameters
        ----------
        *args : tuple
            Contains LSTM output tuple (output, hidden) as first element.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Output state to be passed to dendrite processing.
        """
        ignored_output = args[0][0]
        last_hidden = args[0][1][-1]

        return last_hidden

    def post_n2(self, *args, **kwargs):
        """
        Recombine dendrite-modified last hidden, and append None just to maintain output format

        Parameters
        ----------
        *args : tuple
            Contains the dendrite-modified output state.
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        tuple
            Complete LSTM output (output, hidden) where output has been modified.
        """
        combined_last_hidden = args[0]
        return None, combined_last_hidden

    def pre_d(self, *args, **kwargs):
        """
        LSTM input is just the tensor which also goes to the dendrite

        Parameters
        ----------
        *args : 
            Input tensor
        **kwargs : dict
            Empty

        Returns
        -------
        tuple
            (output, hidden)
        """
        return args, kwargs
        
    def post_d(self, *args, **kwargs):
        """
        Extract extract the dendrites last hidden to combine with neurons.

        Parameters
        ----------
        *args : tuple
            Contains dendrite LSTM output tuple (output, hidden).
        **kwargs : dict
            Unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Output state to be added to the neuron output.
        """
        ignored_output = args[0][0]
        last_hidden = args[0][1][-1]
        return last_hidden

    def clear_processor(self):
        # Nothing is stored
        pass

class ResNetPAI(nn.Module):
    """PB-compatible ResNet wrapper.

    All normalization layers should be wrapped in a PAISequential, or other
    wrapped module. When working with a predefined model the following shows
    an example of how to create a module for modules_to_replace.
    """

    def __init__(self, other_resnet):
        """Initialize ResNetPAI from existing ResNet model.

        Parameters
        ----------
        *args : other_resnet : torchvision.models.resnet.ResNet
            An existing ResNet model to convert to PAI-compatible format.
        """
        super(ResNetPAI, self).__init__()

        # For the most part, just copy the exact values from the original module
        self._norm_layer = other_resnet._norm_layer
        self.inplanes = other_resnet.inplanes
        self.dilation = other_resnet.dilation
        self.groups = other_resnet.groups
        self.base_width = other_resnet.base_width

        # For the component to be changed, define a PAISequential with the old
        # modules included
        self.b1 = GPA.PAISequential([other_resnet.conv1, other_resnet.bn1])

        self.relu = other_resnet.relu
        self.maxpool = other_resnet.maxpool

        for i in range(1, 5):
            layer_name = "layer" + str(i)
            original_layer = getattr(other_resnet, layer_name)
            pb_layer = self._make_layer_pb(original_layer, other_resnet, i)
            setattr(self, layer_name, pb_layer)

        self.avgpool = other_resnet.avgpool
        self.fc = other_resnet.fc

    def _make_layer_pb(self, other_block_set, other_resnet, block_id):
        """Convert ResNet layer blocks to PB-compatible format.

        Parameters
        ----------
        other_block_set : torch.vision.models.resnet.any_block
            A set of blocks from the original ResNet model.
        other_resnet : torchvision.models.resnet.ResNet
            The original ResNet model.
        block_id : int
            The layer number being converted.
        Returns
        -------
        nn.Sequential
            A sequential container with the converted blocks.
        """
        layers = []
        for i in range(len(other_block_set)):
            block_type = type(other_block_set[i])
            if block_type == resnet_pt.BasicBlock:
                layers.append(other_block_set[i])
            elif block_type == resnet_pt.Bottleneck:
                layers.append(other_block_set[i])
            else:
                print(
                    "Your resnet uses a block type that has not been "
                    "accounted for. Customization might be required."
                )
                layer_name = "layer" + str(block_id)
                print(type(getattr(other_resnet, layer_name)))
                pdb.set_trace()
        return nn.Sequential(*layers)

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
