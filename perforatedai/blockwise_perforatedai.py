# Copyright (c) 2025 Perforated AI

from perforatedai import globals_perforatedai as GPA
from perforatedai import modules_perforatedai as PA
from perforatedai import utils_perforatedai as UPA
import torch.nn as nn
import torch
import pdb
import numpy as np
import string
import copy

# This is the cleaner inference version of PAI modules
class PAILayer(nn.Module):
    def __init__(
        self,
        layer_array,
        processor_array,
        dendrites_to_top,
        dendrites_to_dendrites,
        node_index,
        num_cycles,
        view_tuple,
    ):
        super(PAILayer, self).__init__()
        self.layer_array = layer_array
        self.register_buffer("num_cycles", num_cycles)
        self.register_buffer("view_tuple", torch.tensor(view_tuple))

        self.processor_array = processor_array
        if dendrites_to_dendrites:
            self.skip_weights = dendrites_to_dendrites
        else:
            """
            This will only be the case if there is less than 2 dendrites, in these cases an empty array
            should still be added so that dendrites_to_top is included at the correct index
            """
            self.skip_weights = nn.ParameterList([torch.zeros(1, 1, 1)])
        if dendrites_to_top:
            self.skip_weights.append(dendrites_to_top[len(dendrites_to_top) - 1])
        else:
            self.skip_weights = nn.ParameterList()
        
        # Delete skip_weights if it's empty (only 1 layer, no skip connections)
        if len(self.skip_weights) == 0:
            delattr(self, 'skip_weights')

        self.node_index = node_index
        self.internal_nonlinearity = GPA.pc.get_pai_forward_function()

def unWrap_params(model):
    for p in model.parameters():
        if "wrapped" in p.__dir__():
            del p.wrapped

# This converts one training PAI module into an inference PAI module
def convert_to_pai_layer_block(pretrained_dendrite):
    unWrap_params(pretrained_dendrite)
    layer_array = []
    processor_array = []
    for layer_id in range(len(pretrained_dendrite.dendrite_module.layers)):
        layer_array.append(pretrained_dendrite.dendrite_module.layers[layer_id])
        if pretrained_dendrite.dendrite_module.processors == []:
            processor_array.append(None)
        else:
            if not pretrained_dendrite.dendrite_module.processors[layer_id] is None:
                pretrained_dendrite.dendrite_module.processors[layer_id].pre = (
                    pretrained_dendrite.dendrite_module.processors[layer_id].pre_d
                )
                pretrained_dendrite.dendrite_module.processors[layer_id].post = (
                    pretrained_dendrite.dendrite_module.processors[layer_id].post_d
                )
            processor_array.append(
                pretrained_dendrite.dendrite_module.processors[layer_id]
            )
    layer_array.append(pretrained_dendrite.main_module)
    if not pretrained_dendrite.processor is None:
        pretrained_dendrite.processor.pre = pretrained_dendrite.processor.post_n1
        pretrained_dendrite.processor.post = pretrained_dendrite.processor.post_n2
    processor_array.append(pretrained_dendrite.processor)

    view_tuple = []
    for dim in range(
        len(
            pretrained_dendrite.dendrite_module.dendrite_values[0].this_output_dimensions
        )
    ):
        if (
            dim
            == pretrained_dendrite.dendrite_module.dendrite_values[0].this_node_index
        ):
            view_tuple.append(-1)
            continue
        view_tuple.append(1)
    return PAILayer(
        nn.Sequential(*layer_array),
        processor_array,
        pretrained_dendrite.dendrites_to_top,
        pretrained_dendrite.dendrite_module.dendrites_to_dendrites,
        pretrained_dendrite.this_node_index,
        pretrained_dendrite.dendrite_module.num_cycles,
        view_tuple,
    )


def get_pretrained_pai_attr(pretrained_dendrite, member):
    if pretrained_dendrite is None:
        return None
    else:
        return getattr(pretrained_dendrite, member)


def get_pretrained_pai_var(pretrained_dendrite, submodule_id):
    if pretrained_dendrite is None:
        return None
    else:
        return pretrained_dendrite[submodule_id]

# This optimizes a network recursively from training modules to inference modules
def optimize_module(net, depth, name_so_far, converted_list):
    all_members = net.__dir__()
    if issubclass(type(net), nn.Sequential) or issubclass(type(net), nn.ModuleList):
        for submodule_id, layer in net.named_children():
            if type(net.get_submodule(submodule_id)) is PA.PAINeuronModule:
                if GPA.pc.get_extra_verbose():
                    print(
                        "Seq sub is PAI so optimizing: %s" % name_so_far
                        + "["
                        + str(submodule_id)
                        + "]"
                    )
                setattr(
                    net,
                    submodule_id,
                    convert_to_pai_layer_block(net.get_submodule(submodule_id)),
                )
            else:
                if net != net.get_submodule(submodule_id):
                    # this currently just always returns false, not sure what it was for
                    converted_list += [name_so_far + "[" + str(submodule_id) + "]"]
                    setattr(
                        net,
                        submodule_id,
                        optimize_module(
                            net.get_submodule(submodule_id),
                            depth + 1,
                            name_so_far + "[" + str(submodule_id) + "]",
                            converted_list,
                        ),
                    )
                else:
                    if GPA.pc.get_extra_verbose():
                        print(
                            "%s is a self pointer so skipping"
                            % (name_so_far + "[" + str(submodule_id) + "]")
                        )
    else:
        for member in all_members:
            try:
                getattr(net, member, None)
            except:
                continue
            sub_name = name_so_far + "." + member
            if (
                sub_name in GPA.pc.get_module_names_to_not_save()
                or sub_name in converted_list
            ):
                if GPA.pc.get_extra_verbose():
                    print("Skipping %s during save" % sub_name)
                continue
            if type(getattr(net, member, None)) is PA.PAINeuronModule:
                if GPA.pc.get_extra_verbose():
                    print(
                        "Sub is in conversion list so initiating optimization for: %s"
                        % name_so_far
                        + "."
                        + member
                    )
                setattr(net, member, convert_to_pai_layer_block(getattr(net, member)))
            elif issubclass(type(getattr(net, member, None)), nn.Module):
                if net != getattr(net, member):
                    converted_list += [sub_name]
                    setattr(
                        net,
                        member,
                        optimize_module(
                            getattr(net, member),
                            depth + 1,
                            sub_name,
                            converted_list,
                        ),
                    )
                else:
                    if GPA.pc.get_extra_verbose():
                        print("%s is a self pointer so skipping" % (sub_name))
    return net

def blockwise_network(net):
    return optimize_module(net, 0, "", [])
