# Copyright (c) 2025 Perforated AI
from perforatedai import globals_perforatedai as GPA

import copy

import torch.nn as nn
import torch
import pdb

from threading import Thread

doing_threading = False

# This is one implimentation of the forward function of PAI modules that 
# has an option to use python threading
class PAIModulePyThread(nn.Module):
    def __init__(self, original_module):
        super(PAIModulePyThread, self).__init__()
        self.layer_array = original_module.layer_array
        self.processor_array = original_module.processor_array
        self.skip_weights = original_module.skip_weights
        self.register_buffer("node_index", original_module.node_index.clone().detach())
        self.register_buffer("num_cycles", original_module.num_cycles)
        self.register_buffer("view_tuple", original_module.view_tuple)

    def process_and_forward(self, *args2, **kwargs2):
        c = args2[0]
        dendrite_outs = args2[1]
        args2 = args2[2:]
        if self.processor_array[c] != None:
            args2, kwargs2 = self.processor_array[c].pre(*args2, **kwargs2)
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
            if len(self.layer_array) > 1:
                for in_index in range(0, out_index):
                    current_out = current_out + (
                        self.skip_weights[out_index][in_index, :]
                        .view(self.view_tuple.tolist())
                        .to(current_out.device)
                        * dendrite_outs[in_index]
                    )
                if out_index < len(self.layer_array) - 1:
                    current_out = GPA.pc.get_pai_forward_function()(current_out)
            dendrite_outs[out_index] = current_out
        if not self.processor_array[-1] is None:
            current_out = self.processor_array[-1].post(current_out)
        return current_out


def get_pretrained_pai_attr(pretrained_dendrite, member):
    if pretrained_dendrite is None:
        return None
    else:
        return getattr(pretrained_dendrite, member)


def get_pretrained_pai_var(pretrained_dendrite, submodule_id):
    if pretrained_dendrite is None:
        return None
    else:
        return pretrained_dendrite.get_submodule(submodule_id)

ModuleType = PAIModulePyThread
doing_threading = False

def make_module(module):
    return ModuleType(module)

# This Refreshes a PAI network with the PyThread Module
def refresh_pai(net, depth, name_so_far, converted_list):
    if GPA.pc.get_extra_verbose():
        print("CL calling convert on %s depth %d" % (net, depth))
        print(
            "CL calling convert on %s: %s, depth %d"
            % (name_so_far, type(net).__name__, depth)
        )
    if type(net) is ModuleType:
        if GPA.pc.get_extra_verbose():
            print(
                "this is only being called because something in your model is pointed to twice by two different variables.  Highest thing on the list is one of the duplicates"
            )
        return net
    all_members = net.__dir__()
    if (
        issubclass(type(net), nn.Sequential)
        or issubclass(type(net), nn.ModuleList)
        or issubclass(type(net), list)
    ):
        for submodule_id, layer in net.named_children():
            if net != net.get_submodule(submodule_id):
                converted_list += [name_so_far + "[" + str(submodule_id) + "]"]
                setattr(
                    net,
                    submodule_id,
                    refresh_pai(
                        net.get_submodule(submodule_id),
                        depth + 1,
                        name_so_far + "[" + str(submodule_id) + "]",
                        converted_list,
                    ),
                )
            if type(net.get_submodule(submodule_id)).__name__ == "PAILayer":
                setattr(
                    net,
                    submodule_id,
                    make_module(get_pretrained_pai_var(net, submodule_id)),
                )
    elif type(net) in GPA.pc.get_modules_to_track():
        return net
    else:
        for member in all_members:
            try:
                getattr(net, member, None)
            except:
                continue
            sub_name = name_so_far + "." + member

            if member == "device" or member == "dtype":
                continue
            if sub_name in GPA.pc.get_module_names_to_not_save():
                continue
            if name_so_far == "":
                if (
                    sub_name in GPA.pc.get_module_names_to_not_save()
                    or sub_name in converted_list
                ):
                    if GPA.pc.get_extra_verbose():
                        print("Skipping %s during save" % sub_name)
                    continue

            if (
                issubclass(type(getattr(net, member, None)), nn.Module)
                or member == "layer_array"
            ):
                converted_list += [sub_name]
                if net != getattr(net, member):
                    setattr(
                        net,
                        member,
                        refresh_pai(
                            getattr(net, member), depth + 1, sub_name, converted_list
                        ),
                    )
            if type(getattr(net, member, None)).__name__ == "PAILayer":
                setattr(net, member, make_module(get_pretrained_pai_attr(net, member)))
    if type(net).__name__ == "PAILayer":
        net = make_module(net)
    return net

def refresh_net(pretrained_dendrite):

    net = refresh_pai(pretrained_dendrite, 0, "", [])
    return net
