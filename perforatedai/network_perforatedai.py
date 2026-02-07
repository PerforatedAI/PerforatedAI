from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import sys

from safetensors.torch import load_file
import copy

import torch.nn as nn
import torch
import pdb

from threading import Thread


doing_threading = False
loaded_full_print = False


def convert_network(net, layer_name=""):
    # If the net itself has a substitution make that substitution first
    if type(net) in GPA.pc.get_modules_to_replace():
        net = UPA.replace_predefined_modules(net)
    # If the net itself should be converted make the converstion
    if type(net) in GPA.pc.get_modules_to_convert():
        if layer_name == "":
            print(
                "converting a single layer without a name, add a layer_name param to the call"
            )
            sys.exit(-1)
        net = PerforatedModule(net, layer_name)
    # Otherwise, check the module recursively if there are other modules to convert
    else:
        net = UPA.convert_module(
            net, 0, "", [], [], PerforatedModule, PAITrackedModule
        )
    return net


def get_pai_modules(net, depth):
    all_members = net.__dir__()
    this_list = []
    if issubclass(type(net), nn.Sequential) or issubclass(type(net), nn.ModuleList):
        for submodule_id, layer in net.named_children():
            if net.get_submodule(submodule_id) is net:
                continue
            if type(net.get_submodule(submodule_id)) is PerforatedModule:
                this_list = this_list + [net.get_submodule(submodule_id)]
            else:
                this_list = this_list + get_pai_modules(
                    net.get_submodule(submodule_id), depth + 1
                )
    else:
        for member in all_members:
            if getattr(net, member, None) is net:
                continue
            if type(getattr(net, member, None)) is PerforatedModule:
                this_list = this_list + [getattr(net, member)]
            elif issubclass(type(getattr(net, member, None)), nn.Module):
                this_list = this_list + get_pai_modules(getattr(net, member), depth + 1)
    return this_list


def load_pai_model_from_dict(net, state_dict):
    pai_modules = get_pai_modules(net, 0)
    if pai_modules == []:
        print("No PAI modules were found something went wrong with convert network")
        import pdb

        pdb.set_trace()
        sys.exit()
    for module in pai_modules:
        # Set up name to be what will be saved in the state dict
        module_name = UPA.get_module_base_name(module)
        # Then instantiate as many Dendrites as were created during training
        num_cycles = int(state_dict[module_name + ".num_cycles"].item())
        # extract node index from state_dict
        nodeCount = 10
        # also extract view tuple
        if num_cycles > 0:
            module.simulate_cycles(num_cycles, nodeCount)
        if not module.processor is None:
            processor = copy.deepcopy(module.processor)
            processor.pre = module.processor.post_n1
            processor.post = module.processor.post_n2
            module.processor_array.append(processor)
        else:
            module.processor_array.append(None)
        
        # Create ParameterList for skip_weights based on num_cycles
        num_params = num_cycles // 2
        skip_weights_list = nn.ParameterList()
        for i in range(num_params):
            param_key = module_name + f".skip_weights.{i}"
            if param_key in state_dict:
                param = nn.Parameter(torch.randn(state_dict[param_key].shape))
                skip_weights_list.append(param)
        module.skip_weights = skip_weights_list
        
        # module.register_buffer('skip_weights', torch.zeros(state_dict[module_name + '.skip_weights'].shape))
        module.register_buffer("view_tuple", state_dict[module_name + ".view_tuple"])

    net.load_state_dict(state_dict)

    for module in pai_modules:
        temp = tuple(module.view_tuple.tolist())
        del module.view_tuple
        module.view_tuple = temp

    return net
    # figure out if doing this 'thread' stuff is actually helping at all.
    # If its not just get rid of it to simplify things.
    # to test this will have to first get load_pai_model actually set up and working then run a test with and #without threading.


def load_pai_model(net, filename):
    net = convert_network(net)
    state_dict = load_file(filename)
    return load_pai_model_from_dict(net, state_dict)


class PerforatedModule(nn.Module):
    def __init__(self, original_module, name):
        super(PerforatedModule, self).__init__()
        self.name = name
        self.register_buffer("node_index", torch.tensor(-1))
        self.register_buffer("num_cycles", torch.tensor(-1))
        self.register_buffer("view_tuple", torch.tensor(-1))
        self.processor_array = []
        self.processor = None
        self.layer_array = nn.ModuleList([original_module])
        # If this original module has processing functions save the processor
        if type(original_module) in GPA.pc.get_modules_with_processing():
            module_index = GPA.pc.get_modules_with_processing().index(
                type(original_module)
            )
            self.processor = GPA.pc.get_modules_processing_classes()[module_index]()
        elif (
            type(original_module).__name__ in GPA.pc.get_module_names_with_processing()
        ):
            module_index = GPA.pc.get_module_names_with_processing().index(
                type(original_module).__name__
            )
            self.processor = GPA.pc.get_module_by_name_processing_classes()[
                module_index
            ]()
        
    def simulate_cycles(self, num_cycles, nodeCount):
        for i in range(0, num_cycles, 2):
            self.layer_array.append(copy.deepcopy(self.layer_array[0]))
            if not self.processor is None:
                processor = copy.deepcopy(self.processor)
                processor.pre = self.processor.pre_d
                processor.post = self.processor.post_d
                self.processor_array.append(processor)
            else:
                self.processor_array.append(None)

    def process_and_forward(self, *args2, **kwargs2):
        c = args2[0]
        dendrite_outs = args2[1]
        args2 = args2[2:]
        if self.processor_array[c] != None:
            out_values = self.processor_array[c].pre(*args2, **kwargs2)
        out_values = self.layer_array[c](*args2, **kwargs2)
        if self.processor_array[c] != None:
            out = self.processor_array[c].post(out_values)
        else:
            out = out_values
        dendrite_outs[c] = out

    def process_and_pre(self, *args, **kwargs):
        dendrite_outs = args[0]
        args = args[1:]
        out = self.layer_array[-1].forward(*args, **kwargs)
        if not self.processor_array[-1] is None:
            out = self.processor_array[-1].pre(out)
        dendrite_outs[len(self.layer_array) - 1] = out

    def forward(self, *args, **kwargs):
        # this is currently false anyway, just remove the doing multi idea
        doing_multi = doing_threading
        dendrite_outs = [None] * len(self.layer_array)
        threads = {}
        for c in range(0, len(self.layer_array) - 1):
            args2, kwargs2 = args, kwargs
            if doing_multi:
                threads[c] = Thread(
                    target=self.process_and_forward,
                    args=(c, dendrite_outs, *args),
                    kwargs=kwargs,
                )
            else:
                self.process_and_forward(c, dendrite_outs, *args2, **kwargs2)
        if doing_multi:
            threads[len(self.layer_array) - 1] = Thread(
                target=self.process_and_pre, args=(dendrite_outs, *args), kwargs=kwargs
            )
        else:
            self.process_and_pre(dendrite_outs, *args, **kwargs)
        if doing_multi:
            for i in range(len(dendrite_outs)):
                threads[i].start()
            for i in range(len(dendrite_outs)):
                threads[i].join()
        for out_index in range(0, len(self.layer_array)):
            current_out = dendrite_outs[out_index]

            if len(self.layer_array) > 1 and hasattr(self, 'skip_weights'):
                for in_index in range(0, out_index):
                    # Use out_index - 1 because skip_weights[0] is never used
                    current_out = (
                        current_out
                        + self.skip_weights[out_index - 1][in_index, :]
                        .reshape(self.view_tuple)
                        .to(current_out.device)
                        * dendrite_outs[in_index]
                    )
                if out_index < len(self.layer_array) - 1:
                    current_out = GPA.pc.get_pai_forward_function()(current_out)
            dendrite_outs[out_index] = current_out
        if not self.processor_array[-1] is None:
            current_out = self.processor_array[-1].post(current_out)
        return current_out


class PAITrackedModule(nn.Module):
    """Wrapper for modules you don't want to add dendrites to. Ensures all modules are accounted for."""

    def __init__(self, start_module, name):
        """Initialize PAITrackedModule.

        This function sets up the tracked neuron module to wrap the start_module
        without adding dendrites.

        Parameters
        ----------
        start_module : nn.Module
            The module to wrap.
        name : str
            The name of the neuron module.
        """
        super(PAITrackedModule, self).__init__()

        if isinstance(start_module, nn.Module):
            self.main_module = start_module
        else:
            print("start_module must be nn.Module: %s" % name)
            print(type(start_module))
            print(start_module)
            sys.exit(-1)
        self.name = name

        self.type = "tracked_module"

    def __getattr__(self, name):
        """Get member variables from the main module.

        Parameters
        ----------
        name : str
            The name of the variable to retrieve.
        Returns
        -------
        The requested variable.

        Notes
        -----
        This method first attempts to retrieve the attribute from the PAINeuronModule instance.
        If it fails, it tries to get the attribute from the wrapped main_module.
        This allows seamless access to the main module's attributes without modifying original code.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.main_module, name)

    def forward(self, *args, **kwargs):
        """Forward pass for tracked layer.

        Parameters
        ----------
        *args : tuple
            Positional arguments for the forward pass.
        **kwargs : dict
            Keyword arguments for the forward pass.

        Returns
        -------
        Any
            The output of the module

        Notes
        -----
            The output of this forward function will have the same format as the output
            of the original module
        """
        return self.main_module(*args, **kwargs)

    def __str__(self):
        """String representation of the layer.

        Parameters
        ----------
        None

        Returns
        -------
        str
            String representation of the layer.

        Notes
        -----
        Setting for verbose changes level of details in the string output.
        """

        if GPA.pc.get_verbose():
            total_string = self.main_module.__str__()
            total_string = "PAITrackedLayer(" + total_string + ")"
            return total_string
        else:
            total_string = self.main_module.__str__()
            total_string = "PAITrackedLayer(" + total_string + ")"
            return total_string

    def __repr__(self):
        """Representation of the layer."""
        return self.__str__()
