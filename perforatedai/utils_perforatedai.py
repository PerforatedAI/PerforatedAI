# Copyright (c) 2025 Perforated AI

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import sys
import numpy as np
import pdb
import os
import time
import warnings
from collections import defaultdict

from perforatedai import globals_perforatedai as GPA
from perforatedai import modules_perforatedai as PA
from perforatedai import tracker_perforatedai as TPA

try:
    from perforatedbp import utils_pbp as UPB

except Exception as e:
    pass
import copy

from safetensors.torch import load_file
from safetensors.torch import save_file
from safetensors.torch import safe_open


def initialize_pai(
    model,
    doing_pai=True,
    save_name="PAI",
    making_graphs=True,
    maximizing_score=True,
    num_classes=10000000000,
    values_per_train_epoch=-1,
    values_per_val_epoch=-1,
    zooming_graph=True,
):
    """Main function to initialize the network to add dendrites

    This kicks off the entire Perforated AI process to add
    the scaffolding to the network to be able to add dendrites

    Parameters
    ----------
    model : nn.Module
        The neural network model to initialize.
    doing_pai : bool, optional
        Whether to actually add dendrites, by default True
    save_name : str, optional
        The name to save the model under, by default "PAI"
    making_graphs : bool, optional
        Whether to create graphs during training, by default True
    maximizing_score : bool, optional
        Whether to maximize the score during training, by default True
        setting to false is for when the score is a loss to be minimized
    num_classes : int, optional
        The number of output classes, unused in current version
    values_per_train_epoch : int, optional
        The number of values to look back for graphing
        during training, by default -1 (all values).
    values_per_val_epoch : int, optional
        The number of values to look back for graphing
        during validation, by default -1 (all values).
    zooming_graph : bool, optional
        Whether to enable zooming on the graphs, by default True

    Returns
    -------
    model : nn.Module
        The modified model with dendrite scaffolding added if doing_pai is True

    """
    GPA.pai_tracker = TPA.PAINeuronModuleTracker(
        doing_pai=doing_pai, save_name=save_name
    )
    GPA.pc.set_save_name(save_name)
    model = GPA.pai_tracker.initialize(
        model,
        doing_pai=doing_pai,
        save_name=save_name,
        making_graphs=making_graphs,
        maximizing_score=maximizing_score,
        num_classes=num_classes,
        values_per_train_epoch=-values_per_train_epoch,
        values_per_val_epoch=values_per_val_epoch,
        zooming_graph=zooming_graph,
    )
    return model


def get_pai_modules(net, depth):
    """Get a list of all neuron modules

    Parameters
    ----------
    net : nn.Module
        The module to search.
    depth : int
        The current depth in the recursion.

    Returns
    -------
    list
        A list of all PAI neuron modules found in the network.

    """
    all_members = net.__dir__()
    this_list = []
    if issubclass(type(net), nn.Sequential) or issubclass(type(net), nn.ModuleList):
        for submodule_id, layer in net.named_children():
            # If there is a self pointer ignore it
            if net.get_submodule(submodule_id) is net:
                continue
            if type(net.get_submodule(submodule_id)) is PA.PAINeuronModule:
                this_list = this_list + [net.get_submodule(submodule_id)]
            else:
                this_list = this_list + get_pai_modules(
                    net.get_submodule(submodule_id), depth + 1
                )
    else:
        for member in all_members:
            # if the getter fails or it is a self pointer ignore it
            try:
                if getattr(net, member, None) is net:
                    continue
            except:
                continue
            if type(getattr(net, member, None)) is PA.PAINeuronModule:
                this_list = this_list + [getattr(net, member)]
            elif (
                issubclass(type(getattr(net, member, None)), nn.Module)
                or issubclass(type(getattr(net, member, None)), nn.Sequential)
                or issubclass(type(getattr(net, member, None)), nn.ModuleList)
            ):
                this_list = this_list + get_pai_modules(getattr(net, member), depth + 1)

    return this_list


def get_tracked_modules(net, depth):
    """Get a list of all tracked modules

    Parameters
    ----------
    net : nn.Module
        The module to search.
    depth : int
        The current depth in the recursion.

    Returns
    -------
    list
        A list of all tracked modules found in the network.

    """
    all_members = net.__dir__()
    this_list = []
    if issubclass(type(net), nn.Sequential) or issubclass(type(net), nn.ModuleList):
        for submodule_id, layer in net.named_children():
            if net.get_submodule(submodule_id) is net:
                continue
            if type(net.get_submodule(submodule_id)) is PA.TrackedNeuronModule:
                this_list = this_list + [net.get_submodule(submodule_id)]
            else:
                this_list = this_list + get_tracked_modules(
                    net.get_submodule(submodule_id), depth + 1
                )
    else:
        for member in all_members:
            # if the getter fails or it is a self pointer ignore it
            try:
                if getattr(net, member, None) is net:
                    continue
            except:
                continue
            if type(getattr(net, member, None)) is PA.TrackedNeuronModule:
                this_list = this_list + [getattr(net, member)]
            elif issubclass(type(getattr(net, member, None)), nn.Module):
                this_list = this_list + get_tracked_modules(
                    getattr(net, member), depth + 1
                )
    return this_list


def get_pai_module_params(net, depth):
    """Get a list of all neuron module parameters

    Parameters
    ----------
    net : nn.Module
        The module to search.
    depth : int
        The current depth in the recursion.

    Returns
    -------
    list
        A list of all parameters of neuron modules found in this module.

    """

    all_members = net.__dir__()
    this_list = []
    if issubclass(type(net), nn.Sequential) or issubclass(type(net), nn.ModuleList):
        for submodule_id, layer in net.named_children():
            if isinstance(net.get_submodule(submodule_id), PA.PAINeuronModule):  #
                for param in net.get_submodule(submodule_id).parameters():
                    if param.requires_grad:
                        this_list = this_list + [param]
            else:
                this_list = this_list + get_pai_module_params(
                    net.get_submodule(submodule_id), depth + 1
                )
    else:
        for member in all_members:
            if getattr(net, member, None) == net:
                continue
            if isinstance(getattr(net, member, None), PA.PAINeuronModule):
                for param in getattr(net, member).parameters():
                    if param.requires_grad:
                        this_list = this_list + [param]
            elif issubclass(type(getattr(net, member, None)), nn.Module):
                this_list = this_list + get_pai_module_params(
                    getattr(net, member), depth + 1
                )
    return this_list


def get_pai_network_params(net):
    """Get a list of all neuron module parameters

    Parameters
    ----------
    net : nn.Module
        The full model to search.

    Returns
    -------
    list
        A list of all parameters of neuron modules found in the network.

    """
    param_list = get_pai_module_params(net, 0)
    return param_list


def replace_predefined_modules(start_module):
    """Replace a module with the module from globals list

    Parameters
    ----------
    start_module : nn.Module
        The module to replace.

    Returns
    -------
    nn.Module
        The replaced module.

    """
    index = GPA.pc.get_modules_to_replace().index(type(start_module))
    return GPA.pc.get_replacement_modules()[index](start_module)


def convert_module(net, depth, name_so_far, converted_list, converted_names_list):
    """Recursive function to do all conversion of modules to wrappers of modules

    This is the function that goes through all of the module lists from
    the globals file and does all the conversion and replacements to
    setup the dendrite scaffolding as instructed.

    Parameters
    ----------
    net : nn.Module
        The module to convert.
    depth : int
        The current depth in the recursion.
    name_so_far : str
        The name of the module so far in the recursion.
    converted_list : list
        A list of already converted module ids to avoid infinite loops.
    converted_names_list : list
        A corresponding list to help debug duplicate conversions

    Returns
    -------
    nn.Module
        The converted module.

    """
    if GPA.pc.get_verbose():
        print("calling convert on %s depth %d" % (net, depth))
        print(
            "calling convert on %s: %s, depth %d"
            % (name_so_far, type(net).__name__, depth)
        )
    if isinstance(net, PA.PAINeuronModule) or isinstance(net, PA.TrackedNeuronModule):
        if GPA.pc.get_verbose():
            print(
                "This is only being called because something in your model "
                "is pointed to twice by two different variables. Highest "
                "thing on the list is one of the duplicates"
            )
        return net
    all_members = net.__dir__()
    if GPA.pc.get_extra_verbose():
        print("all members:")
        for member in all_members:
            print(" - %s" % member)
    if issubclass(type(net), nn.Sequential) or issubclass(type(net), nn.ModuleList):
        for submodule_id, layer in net.named_children():
            sub_name = name_so_far + "." + str(submodule_id)
            if sub_name in GPA.pc.get_module_ids_to_track():
                if GPA.pc.get_verbose():
                    print("Seq ID is in track IDs: %s" % sub_name)
                setattr(
                    net,
                    submodule_id,
                    PA.TrackedNeuronModule(net.get_submodule(submodule_id), sub_name),
                )
                continue
            if sub_name in GPA.pc.get_module_ids_to_convert():
                if GPA.pc.get_verbose():
                    print("Seq ID is in convert IDs: %s" % sub_name)
                setattr(
                    net,
                    submodule_id,
                    PA.PAINeuronModule(net.get_submodule(submodule_id), sub_name),
                )
                continue
            if type(net.get_submodule(submodule_id)) in GPA.pc.get_modules_to_replace():
                if GPA.pc.get_verbose():
                    print(
                        "Seq sub is in replacement module so replacing: %s" % sub_name
                    )
                setattr(
                    net,
                    submodule_id,
                    replace_predefined_modules(net.get_submodule(submodule_id)),
                )
            if (
                type(net.get_submodule(submodule_id)) in GPA.pc.get_modules_to_track()
            ) or (
                type(net.get_submodule(submodule_id)).__name__
                in GPA.pc.get_module_names_to_track()
            ):
                if GPA.pc.get_verbose():
                    print(
                        "Seq sub is in tracking list so initiating tracked for: %s"
                        % sub_name
                    )
                setattr(
                    net,
                    submodule_id,
                    PA.TrackedNeuronModule(net.get_submodule(submodule_id), sub_name),
                )
            elif (
                type(net.get_submodule(submodule_id)) in GPA.pc.get_modules_to_convert()
                or type(net.get_submodule(submodule_id)).__name__
                in GPA.pc.get_module_names_to_convert()
            ):
                if GPA.pc.get_verbose():
                    print(
                        "Seq sub is in conversion list so initing PAI for: "
                        "%s" % sub_name
                    )
                if (
                    issubclass(
                        type(net.get_submodule(submodule_id)),
                        torch.nn.modules.batchnorm._BatchNorm,
                    )
                    or issubclass(
                        type(net.get_submodule(submodule_id)),
                        torch.nn.modules.instancenorm._InstanceNorm,
                    )
                    or issubclass(
                        type(net.get_submodule(submodule_id)),
                        torch.nn.modules.normalization.LayerNorm,
                    )
                ):
                    print(
                        "You have an unwrapped normalization layer, this "
                        "is not recommended: " + name_so_far
                    )
                    pdb.set_trace()
                setattr(
                    net,
                    submodule_id,
                    PA.PAINeuronModule(net.get_submodule(submodule_id), sub_name),
                )
            else:
                if net != net.get_submodule(submodule_id):
                    converted_list += [id(net.get_submodule(submodule_id))]
                    converted_names_list += [sub_name]
                    if GPA.pc.get_verbose():
                        print(
                            "sub is module but in no lists so going deeper: %s"
                            % sub_name
                        )

                    setattr(
                        net,
                        submodule_id,
                        convert_module(
                            net.get_submodule(submodule_id),
                            depth + 1,
                            sub_name,
                            converted_list,
                            converted_names_list,
                        ),
                    )
                # else:
                # print('%s is a self pointer so skipping' % (name_so_far + '[' + str(submodule_id) + ']'))
    elif type(net) in GPA.pc.get_modules_to_track():
        # print('skipping type for returning from call to: %s' % (name_so_far))
        return net
    else:
        for member in all_members:
            # Immediately check if able to get the member, if not skip it
            try:
                getattr(net, member, None)
            except:
                continue
            sub_name = name_so_far + "." + member
            if sub_name in GPA.pc.get_module_ids_to_track():
                if GPA.pc.get_verbose():
                    print("Seq ID is in track IDs: %s" % sub_name)
                setattr(
                    net, member, PA.TrackedNeuronModule(getattr(net, member), sub_name)
                )
                continue
            if sub_name in GPA.pc.get_module_ids_to_convert():
                if GPA.pc.get_verbose():
                    print("Seq ID is in convert IDs: %s" % sub_name)
                setattr(net, member, PA.PAINeuronModule(getattr(net, member), sub_name))
                continue
            if id(getattr(net, member, None)) == id(net):
                if GPA.pc.get_verbose():
                    print("member sub is a self pointer: %s" % sub_name)
                continue
            if sub_name in GPA.pc.get_module_names_to_not_save():
                if GPA.pc.get_verbose():
                    print("Skipping %s during convert" % sub_name)
                else:
                    if sub_name == ".base_model":
                        print(
                            "By default skipping base_model. See "
                            '"Safetensors Errors" section of '
                            "customization.md to include it."
                        )
                continue
            if id(getattr(net, member, None)) in converted_list:
                print(
                    "The following module has a duplicate pointer within "
                    "your model: %s" % sub_name
                )
                print(
                    "It is shared with: %s"
                    % converted_names_list[
                        converted_list.index(id(getattr(net, member, None)))
                    ]
                )
                print(
                    "One of these must be added to "
                    "GPA.pc.append_module_names_to_not_save() (with the .)"
                )
                pdb.set_trace()
                sys.exit(0)

            if type(getattr(net, member, None)) in GPA.pc.get_modules_to_replace():
                if GPA.pc.get_verbose():
                    print("sub is in replacement module so replacing: %s" % sub_name)
                setattr(
                    net, member, replace_predefined_modules(getattr(net, member, None))
                )
            if (
                type(getattr(net, member, None)) in GPA.pc.get_modules_to_track()
                or type(getattr(net, member, None)).__name__
                in GPA.pc.get_module_names_to_track()
                or sub_name in GPA.pc.get_module_ids_to_track()
            ):
                if GPA.pc.get_verbose():
                    print(
                        "sub is in tracking list so initiating tracked for: %s"
                        % sub_name
                    )
                setattr(
                    net, member, PA.TrackedNeuronModule(getattr(net, member), sub_name)
                )
            elif (
                type(getattr(net, member, None)) in GPA.pc.get_modules_to_convert()
                or type(getattr(net, member, None)).__name__
                in GPA.pc.get_module_names_to_convert()
                or (sub_name in GPA.pc.get_module_ids_to_convert())
            ):
                if GPA.pc.get_verbose():
                    print(
                        "sub is in conversion list so initiating PAI for: %s" % sub_name
                    )
                setattr(
                    net,
                    member,
                    PA.PAINeuronModule(getattr(net, member), sub_name),
                )
            elif (
                issubclass(type(getattr(net, member, None)), nn.Module)
                or issubclass(type(getattr(net, member, None)), nn.Sequential)
                or issubclass(type(getattr(net, member, None)), nn.ModuleList)
            ):
                if net != getattr(net, member):
                    converted_list += [id(getattr(net, member))]
                    converted_names_list += [sub_name]
                    if GPA.pc.get_verbose():
                        print(
                            "sub is module but in no lists so going deeper: %s"
                            % sub_name
                        )
                    setattr(
                        net,
                        member,
                        convert_module(
                            getattr(net, member),
                            depth + 1,
                            sub_name,
                            converted_list,
                            converted_names_list,
                        ),
                    )
            if (
                issubclass(
                    type(getattr(net, member, None)),
                    torch.nn.modules.batchnorm._BatchNorm,
                )
                or issubclass(
                    type(getattr(net, member, None)),
                    torch.nn.modules.instancenorm._InstanceNorm,
                )
                or issubclass(
                    type(getattr(net, member, None)),
                    torch.nn.modules.normalization.LayerNorm,
                )
            ):
                if not GPA.pc.get_unwrapped_modules_confirmed():
                    print(
                        "potentially found a norm Layer that wont be "
                        "converted, this is not recommended: %s" % (sub_name)
                    )
                    print(
                        "Set GPA.pc.set_unwrapped_modules_confirmed(True) to skip "
                        "this next time"
                    )
                    print(
                        "Type 'net' + enter to inspect your network and "
                        "see what the module type containing this layer is."
                    )
                    print("Then do one of the following:")
                    print(
                        " - Add the module type to "
                        "GPA.pc.get_module_names_to_convert() to wrap it entirely"
                    )
                    print(
                        " - If the norm layer is part of a sequential wrap "
                        "it and the previous layer in a PAISequential"
                    )
                    print(
                        " - If you do not want to add dendrites to this "
                        "module add the type to GPA.pc.get_module_names_to_track()"
                    )
                    pdb.set_trace()
            else:
                if GPA.pc.get_verbose():
                    if member[0] != "_" or GPA.pc.get_extra_verbose() is True:
                        print("not calling convert on %s depth %d" % (member, depth))
    if GPA.pc.get_verbose():
        print("returning from call to: %s" % (name_so_far))
    return net


def convert_network(net, layer_name=""):
    """Function that calls convert_module and checks results

    Parameters
    ----------
    net : nn.Module
        The network to convert.
    layer_name : str, optional
        The name of the layer if converting a single layer, by default ""

    Returns
    -------
    nn.Module
        The converted network.

    """
    if GPA.pc.get_perforated_backpropagation():
        UPB.initialize_pb()
    if type(net) in GPA.pc.get_modules_to_replace():
        net = replace_predefined_modules(net)
    if (type(net) in GPA.pc.get_modules_to_convert()) or (
        type(net).__name__ in GPA.pc.get_module_names_to_convert()
    ):
        if layer_name == "":
            print(
                "converting a single layer without a name, add a "
                "layer_name param to the call"
            )
            sys.exit(-1)
        net = PA.PAINeuronModule(net, layer_name)
    else:
        net = convert_module(net, 0, "", [], [])
    if GPA.pai_tracker.member_vars["doing_pai"]:
        missed_ones = []
        tracked_ones = []
        for name, param in net.named_parameters():
            wrapped = "wrapped" in param.__dir__()
            if wrapped:
                if GPA.pc.get_verbose():
                    print("param %s is now wrapped" % (name))
            else:
                tracked = "tracked" in param.__dir__()
                if tracked:
                    tracked_ones.append(name)
                else:
                    missed_ones.append(name)
        if (
            len(missed_ones) != 0 or len(tracked_ones) != 0
        ) and GPA.pc.get_unwrapped_modules_confirmed() is False:
            print(
                "\n------------------------------------------------------------------"
            )
            print(
                "The following params are not wrapped.\n------------------------------------------------------------------"
            )
            for name in tracked_ones:
                print("." + name)
            print(
                "\n------------------------------------------------------------------"
            )
            print(
                "The following params are not tracked or wrapped.\n------------------------------------------------------------------"
            )
            for name in missed_ones:
                print("." + name)
            print(
                "\n------------------------------------------------------------------"
            )
            print(
                "Modules that are not wrapped will not have Dendrites to optimize them"
            )
            print(
                "Modules modules that are not tracked can cause errors and is NOT recommended"
            )
            print(
                "Any modules in the second list should be added to module_names_to_track"
            )
            print(
                "------------------------------------------------------------------\nType 'c' + enter to continue the run to confirm you do not want them to be refined"
            )
            print(
                "Set GPA.pc.set_unwrapped_modules_confirmed(True) to skip this next time"
            )
            print(
                "Type 'net' + enter to inspect your network and see what the module types of these values are to add them to PGB.module_names_to_convert"
            )
            # If did miss some then set trace to debug
            if len(missed_ones) != 0:
                pdb.set_trace()
            print("confirmed")
    net.register_buffer("tracker_string", torch.tensor([], dtype=torch.uint8))
    return net


def string_to_tensor(string):
    """Helper function to convert a layer_tracker into a string

    This is required for safetensors saving

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    torch.Tensor
        The converted tensor.

    """
    ords = list(map(ord, string))
    ords = torch.tensor(ords, dtype=torch.uint8)
    return ords


def string_from_tensor(string_tensor):
    """Convert a tensor back into a string

    Parameters
    ----------
    string_tensor : torch.Tensor
        The tensor to convert.

    Returns
    -------
    str
        The converted string.

    """
    ords = string_tensor.tolist()
    to_return = ""
    # Doing block processing like this helps with memory errors
    while len(ords) != 0:
        remaining_ords = ords[100000:]
        ords = ords[:100000]
        to_append = "".join(map(chr, ords))
        to_return = to_return + to_append
        ords = remaining_ords
    return to_return


def save_system(net, folder, name):
    """Save the entire system

    This saves the network itself as well as the tracker information

    Parameters
    ----------
    net : nn.Module
        The network to save.
    folder : str
        The folder to save the network in.
    name : str
        The name to save the network under.

    Returns
    -------
    None

    """
    if GPA.pc.get_verbose():
        print("saving system %s" % name)
    temp = string_to_tensor(GPA.pai_tracker.to_string())
    if hasattr(net, "tracker_string"):
        net.tracker_string = string_to_tensor(GPA.pai_tracker.to_string()).to(
            next(net.parameters()).device
        )
    else:
        net.register_buffer(
            "tracker_string",
            string_to_tensor(GPA.pai_tracker.to_string()).to(
                next(net.parameters()).device
            ),
        )
    # Before saving the tracker must be cleared to not contain pointers to the
    # models modules
    old_list = GPA.pai_tracker.neuron_module_vector
    GPA.pai_tracker.neuron_module_vector = []
    save_net(net, folder, name)
    GPA.pai_tracker.neuron_module_vector = old_list
    pai_save_system(net, folder, name)


def load_system(
    net,
    folder,
    name,
    load_from_restart=False,
    switch_call=False,
    load_from_manual_save=False,
):
    """Load the entire system

    This is what should be used to load a saved system and restart training

    Parameters
    ----------
    net : nn.Module
        The network to load into.
    folder : str
        The folder to load the network from.
    name : str
        The name to load the network from.
    load_from_restart : bool, optional
        Whether this is being loaded from an automatic restart, by default False
    switch_call : bool, optional
        Whether this is being called from a switch, by default False
    load_from_manual_save : bool, optional
        Whether this is being loaded from a manual save, by default False

    Returns
    -------
    nn.Module
        The loaded network.

    Notes
    -----
    If you manually call save_system then load_from_manual_save should be True

    """
    if GPA.pc.get_verbose():
        print("loading system %s" % name)
    net = load_net(net, folder, name)
    GPA.pai_tracker.reset_module_vector(net, load_from_restart)

    GPA.pai_tracker.from_string(string_from_tensor(net.tracker_string))
    GPA.pai_tracker.saved_time = time.time()
    GPA.pai_tracker.loaded = True
    GPA.pai_tracker.member_vars["current_best_validation_score"] = 0
    GPA.pai_tracker.member_vars["epoch_last_improved"] = GPA.pai_tracker.member_vars[
        "num_epochs_run"
    ]
    if GPA.pc.get_verbose():
        print(
            "after loading epoch last improved is %d mode is %c"
            % (
                GPA.pai_tracker.member_vars["epoch_last_improved"],
                GPA.pai_tracker.member_vars["mode"],
            )
        )
    # Saves always take place before the call to start_epoch so call it here
    # when loading to correct off by 1 problems
    if (not switch_call) and (not load_from_manual_save):
        GPA.pai_tracker.start_epoch(internal_call=True)
    return net


import json
from collections import defaultdict
from safetensors.torch import save_file, safe_open
import torch


def save_model_with_weight_tying(model, filepath):
    """Save model with safetensors while handling weight tying automatically"""
    state_dict = model.state_dict()

    # Find all weight tied parameters
    tensor_to_keys = defaultdict(list)
    for key, tensor in state_dict.items():
        # Use tensor data pointer as unique identifier
        tensor_id = tensor.data_ptr()
        tensor_to_keys[tensor_id].append(key)

    # Find tied weights (tensors referenced by multiple keys)
    tied_weights = {}
    keys_to_remove = set()
    for tensor_id, keys in tensor_to_keys.items():
        if len(keys) > 1 and not tensor_id == 0:
            # Multiple keys reference the same tensor - this is weight tying
            # Sort keys for deterministic ordering
            keys = sorted(keys)
            primary_key = keys[0]  # Keep the first key
            for secondary_key in keys[1:]:
                tied_weights[secondary_key] = primary_key
                keys_to_remove.add(secondary_key)

    # Remove tied weights from state_dict (keep only primary references)
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k not in keys_to_remove
    }

    # Create metadata for weight tying information
    metadata = {}
    if tied_weights:
        # Store weight tying info as JSON string in metadata
        metadata["weight_tying"] = json.dumps(tied_weights)
    save_file(filtered_state_dict, filepath, metadata=metadata)
    print(f"Saved model with {len(tied_weights)} weight tying relationships")
    return tied_weights


def load_model_with_weight_tying(model, filepath):
    """Load model from safetensors while restoring weight tying"""
    with safe_open(filepath, framework="pt") as f:
        metadata = f.metadata()
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    # Restore weight tying if metadata exists
    tied_weights = {}
    if metadata and "weight_tying" in metadata:
        tied_weights = json.loads(metadata["weight_tying"])
        for secondary_key, primary_key in tied_weights.items():
            if primary_key in state_dict:
                # Restore the tied reference
                state_dict[secondary_key] = state_dict[primary_key]
                print(f"Restored weight tying: {secondary_key} -> {primary_key}")



    # Handle tracker_string loading with flexible key matching
    tracker_key = None
    if "tracker_string" in state_dict:
        tracker_key = "tracker_string"
    else:
        # Search for keys containing "tracker_string"
        tracker_keys = [key for key in state_dict.keys() if "tracker_string" in key]
        if len(tracker_keys) == 1:
            tracker_key = tracker_keys[0]
        elif len(tracker_keys) > 1:
            print(f"Error: Multiple tracker_string keys found: {tracker_keys}")
            pdb.set_trace()
        else:
            print("Error: No tracker_string found in state_dict")
            pdb.set_trace()
    # To restore the tracker the string may have changed lengths
    # so handle that before load_state_dict
    if hasattr(model, "tracker_string"):
        model.tracker_string = state_dict[tracker_key]
    else:
        model.register_buffer("tracker_string", state_dict[tracker_key])

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)

    # Re-establish weight tying at the model level
    # This ensures the actual tensor objects are shared, not just copied
    if tied_weights:
        for secondary_key, primary_key in tied_weights.items():
            try:
                # Navigate to primary parameter
                primary_param = model
                for attr in primary_key.split("."):
                    if attr.isdigit():
                        primary_param = primary_param[int(attr)]
                    else:
                        primary_param = getattr(primary_param, attr)

                # Navigate to secondary parameter's parent
                secondary_param = model
                secondary_attrs = secondary_key.split(".")
                for attr in secondary_attrs[:-1]:
                    if attr.isdigit():
                        secondary_param = secondary_param[int(attr)]
                    else:
                        secondary_param = getattr(secondary_param, attr)

                # Actually tie the weights by sharing the tensor
                final_attr = secondary_attrs[-1]
                if final_attr.isdigit():
                    secondary_param[int(final_attr)] = primary_param
                else:
                    setattr(secondary_param, final_attr, primary_param)

                print(f"Re-tied weights: {secondary_key} = {primary_key}")
            except (AttributeError, IndexError) as e:
                pdb.set_trace()
                print(
                    f"Warning: Could not re-tie {secondary_key} -> {primary_key}: {e}"
                )

    return model


def save_net(net, folder, name):
    """Save the network

    This is called within save_system after the tracker has been
    turned into a single tensor to be saved as a part of the network

    Parameters
    ----------
    net : nn.Module
        The network to save.
    folder : str
        The folder to save the network in.
    name : str
        The name to save the network under.

    Returns
    -------
    None

    """
    # If running a DDP only save with first thread
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return
    if not os.path.isdir(folder):
        os.makedirs(folder)
    save_point = folder + "/"
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    for param in net.parameters():
        param.data = param.data.contiguous()
    if GPA.pc.get_using_safe_tensors():
        if GPA.pc.get_weight_tying_experimental():
            save_model_with_weight_tying(net, save_point + name + ".pt")
        else:
            save_file(net.state_dict(), save_point + name + ".pt")
    else:
        torch.save(net, save_point + name + ".pt")


def manual_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print(f"Warning: {name} not found in model state_dict")
            continue
        if isinstance(param, torch.nn.Parameter):
            # Backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except Exception as e:
            print(f"Error loading {name}: {e}")
    print("Manual load complete")


def load_net(net, folder, name):
    """load the network

    This is called within load_system after the tracker has been
    loaded

    Parameters
    ----------
    net : nn.Module
        The network to save.
    folder : str
        The folder to save the network in.
    name : str
        The name to save the network under.

    Returns
    -------
    nn.Module
        The loaded network.

    """
    save_point = folder + "/"
    if GPA.pc.get_using_safe_tensors():
        if GPA.pc.get_weight_tying_experimental():
            return load_model_with_weight_tying(net, save_point + name + ".pt")
        else:
            state_dict = load_file(save_point + name + ".pt")
    else:
        # Different versions of torch require this change
        try:
            state_dict = torch.load(
                save_point + name + ".pt",
                map_location=torch.device("cpu"),
                weights_only=False,
            ).state_dict()
        except:
            try:
                state_dict = torch.load(
                    save_point + name + ".pt", map_location=torch.device("cpu")
                ).state_dict()
            except:
                state_dict = torch.load(
                    save_point + name + ".pt", map_location=torch.device("cpu")
                )
    return load_net_from_dict(net, state_dict)


def load_net_from_dict(net, state_dict):
    """load the network

    This is called within load_net

    Parameters
    ----------
    net : nn.Module
        The network to save.
    state_dict : dict
        The state dictionary to load.

    Returns
    -------
    nn.Module
        The loaded network.

    """
    pai_modules = get_pai_modules(net, 0)
    if pai_modules == []:
        print(
            "PAI load_net and load_system uses a state_dict so it must be "
            "called with a net after initialize_pai has been called"
        )
        import pdb

        pdb.set_trace()
        sys.exit(-1)
    for module in pai_modules:
        # Set up name to be what will be saved in the state dict
        module_name = module.name
        # This should always be true
        if module_name[0] == ".":
            # strip "."
            module_name = module_name[1:]
        # If it was a dataparallel it will also have a module at the start
        # so strip that for loading
        if module_name[:6] == "module":
            module_name = module_name[7:]
        module.clear_dendrites()
        for tracker in module.dendrite_module.dendrite_values:
            try:
                tracker.setup_arrays(
                    len(
                        state_dict[
                            module_name + ".dendrite_module.dendrite_values.0.shape"
                        ]
                    )
                )
            except Exception as e:
                print(e)
                print(
                    "This value is missing from the state dict\n"
                    "When missing this value it typically means you\n"
                    "converted a module but didn't actually use it in\n"
                    "your forward and backward pass"
                )
                print("module was: %s" % module.name)
                print(
                    "check your model definition and forward function and "
                    "ensure this module is being used properly"
                )
                print(
                    "with GPA.pc.set_verbose(True) you can confirm this is the case if"
                    'you do not see a "setting d shape for" this module at the first training batch.'
                )
                print(
                    "If this is the case, and it is correct to not be passing data through it\n"
                    "Set it to be a tracked module with:\n"
                    'GPA.pc.append_module_ids_to_track(["%s"]) to leave it out '
                    % module.name
                )
                print(
                    "\nThis can also happen if you adjusted your model "
                    "definition after calling initialize_pai"
                )
                print(
                    "for example with torch.compile. If the module name "
                    "printed above does not contain all modules leading "
                    "to the main definition"
                )
                print(
                    "this is likely the case for your problem. Fix by "
                    "calling initialize_pai after all other model "
                    "initialization steps"
                )
                first_key = next(iter(state_dict.keys()))

                print(
                    "\nA Third reason this can happen is if the model where you called initialize_pai\n"
                    "and the model within add_validation_score are not the same. \n"
                    "Check if the module above and %s have the same prefix\n"
                    % first_key
                )
                print(
                    "if one starts with .model or .base etc and the other does not, this is the problem."
                )
                import pdb

                pdb.set_trace()

        # Perform as many cycles as the state dict has
        num_cycles = int(state_dict[module_name + ".dendrite_module.num_cycles"].item())
        if num_cycles > 0:
            simulate_cycles(module, num_cycles, doing_pai=True)
    # Handle tracker_string loading with flexible key matching
    tracker_key = None
    if "tracker_string" in state_dict:
        tracker_key = "tracker_string"
    else:
        # Search for keys containing "tracker_string"
        tracker_keys = [key for key in state_dict.keys() if "tracker_string" in key]
        if len(tracker_keys) == 1:
            tracker_key = tracker_keys[0]
        elif len(tracker_keys) > 1:
            print(f"Error: Multiple tracker_string keys found: {tracker_keys}")
            pdb.set_trace()
        else:
            print("Error: No tracker_string found in state_dict")
            pdb.set_trace()
    
    if hasattr(net, "tracker_string"):
        net.tracker_string = state_dict[tracker_key]
    else:
        net.register_buffer("tracker_string", state_dict[tracker_key])
    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        """
        When modules have high depth to them (i.e. modules within modules not number of layers)
        PyTorch can have trouble loading state dicts even when they are correct.
        This is a workaround to manually load the state dict if this happens.
        """
        if set(net.state_dict().keys()) == set(state_dict.keys()):
            print("Attempting manual loading of state_dict")
            manual_load_state_dict(net, state_dict)
        else:
            print(f"Error loading state_dict: {e}")
            print("net state dict is:")
            print(net.state_dict())
            print("loaded state dict is:")
            print(state_dict)
            print(
                "Try to check differences.  Likely is caused by a module not "
                "being converted that should be or vice versa"
            )
            pdb.set_trace()
    net.to(GPA.pc.get_device())
    return net


def pai_save_system(net, folder, name):
    """Save the entire system with scaffolding removed

    This is used for the final network for inference after training

    Parameters
    ----------
    net : nn.Module
        The network to save.
    folder : str
        The folder to save the network in.
    name : str
        The name to save the network under.

    Returns
    -------
    None

    """
    net.member_vars = {}
    for member_var in GPA.pai_tracker.member_vars:
        if member_var == "scheduler_instance" or member_var == "optimizer_instance":
            continue
        net.member_vars[member_var] = GPA.pai_tracker.member_vars[member_var]
    pai_save_net(net, folder, name)


def deep_copy_pai(net):
    """Deep copy a PAI network


    Parameters
    ----------
    net : nn.Module
        The network to copy.

    Returns
    -------
    nn.Module
        The copied network.

    Notes
    ----
    This is required because processors must be cleared before calling copy

    """
    GPA.pai_tracker.clear_all_processors()
    return copy.deepcopy(net)


def pai_save_net(net, folder, name):
    """Save the entire system with scaffolding removed

    This is called within pai_save_system after the tracker has been
    turned into a single tensor to be saved as a part of the network


    Parameters
    ----------
    net : nn.Module
        The network to save.
    folder : str
        The folder to save the network in.
    name : str
        The name to save the network under.

    Returns
    -------
    None

    Notes
    ----
    For open source implementation this is not as important since
    minimal values are already being used.

    """

    if GPA.pc.get_perforated_backpropagation():
        UPB.pb_save_net(net, folder, name)
    else:
        return


def simulate_cycles(module, num_cycles, doing_pai):
    """Simulate dendrite addition cycles

    Simulate the back and forth processes of adding dendrites to build a
    pretrained dendrite model before loading weights.  Required for loading
    dendrite save files from non dendrite initial models.

    Parameters
    ----------
    module : PA.PAINeuronModule
        The module to simulate cycles on.
    num_cycles : int
        The number of cycles to simulate.
    doing_pai : bool
        Whether to actually do the simulation.

    Returns
    -------
    None

    """

    check_skipped = GPA.pc.get_checked_skipped_modules()
    if doing_pai is False:
        return
    GPA.pc.set_checked_skipped_modules(True)
    mode = "n"
    for i in range(num_cycles):
        if mode == "n":
            module.set_mode("p")
            module.create_new_dendrite_module()
            mode = "p"
        else:
            module.set_mode("n")
            mode = "n"
    GPA.pc.set_checked_skipped_modules(check_skipped)


def count_params(net):
    """Count the number of parameters in the network

    If doing perforated backpropagation this calls the PB function
    which does not count scaffolding parameters since the final model
    will not have them.

    Parameters
    ----------
    net : nn.Module
        The network to count parameters in.

    Returns
    -------
    int
        The number of parameters in the network.

    """
    if GPA.pc.get_perforated_backpropagation():
        return UPB.pb_count_params(net)
    return sum(p.numel() for p in net.parameters())


def change_learning_modes(net, folder, name, doing_pai):
    """Change between neuron and dendrite learning modes

    High level steps for entire system to switch back and forth between
    neuron learning and dendrite learning

    Parameters
    ----------
    net : nn.Module
        The network to change modes on.
    folder : str
        The folder to save/load the network in/from.
    name : str
        The name to save/load the network under.
    doing_pai : bool
        Whether to add dendrites when changing modes.

    Returns
    -------
    int
        The number of parameters in the network.

    Notes
    -----
    If doing_pai is False this just allows training to continue longer rather than early stopping

    """
    # If not adding dendrites this just allows training to continue longer with flags
    # every time early stopping should be occurring
    if doing_pai is False:
        GPA.pai_tracker.member_vars["switch_epochs"].append(
            GPA.pai_tracker.member_vars["num_epochs_run"]
        )
        GPA.pai_tracker.member_vars["last_switch"] = GPA.pai_tracker.member_vars[
            "switch_epochs"
        ][-1]
        GPA.pai_tracker.reset_vals_for_score_reset()
        return net
    if GPA.pai_tracker.member_vars["mode"] == "n":
        current_epoch = GPA.pai_tracker.member_vars["num_epochs_run"]
        overwritten_epochs = GPA.pai_tracker.member_vars["overwritten_epochs"]
        overwritten_extra = GPA.pai_tracker.member_vars["extra_scores"]
        if GPA.pc.get_drawing_pai():
            overwritten_val = GPA.pai_tracker.member_vars["accuracies"]
        else:
            overwritten_val = GPA.pai_tracker.member_vars["neuron_accuracies"]
        """
        If true don't load the best system
        because it will delete dendrites if the previous best was better than
        the current best
        """
        if not GPA.pc.get_silent():
            print("Importing best Model for switch to PA...")
        net = load_system(net, folder, name, switch_call=True)
        GPA.pai_tracker.set_dendrite_training()
        GPA.pai_tracker.member_vars["overwritten_epochs"] = overwritten_epochs
        GPA.pai_tracker.member_vars["overwritten_epochs"] += (
            current_epoch - GPA.pai_tracker.member_vars["num_epochs_run"]
        )
        GPA.pai_tracker.member_vars["total_epochs_run"] = (
            GPA.pai_tracker.member_vars["num_epochs_run"]
            + GPA.pai_tracker.member_vars["overwritten_epochs"]
        )

        if GPA.pc.get_save_old_graph_scores():
            GPA.pai_tracker.member_vars["overwritten_extras"].append(overwritten_extra)
            GPA.pai_tracker.member_vars["overwritten_vals"].append(overwritten_val)
        else:
            GPA.pai_tracker.member_vars["overwritten_extras"] = [overwritten_extra]
            GPA.pai_tracker.member_vars["overwritten_vals"] = [overwritten_val]
        if GPA.pc.get_drawing_pai():
            GPA.pai_tracker.member_vars["n_switch_epochs"].append(
                GPA.pai_tracker.member_vars["num_epochs_run"]
            )
        else:
            if len(GPA.pai_tracker.member_vars["switch_epochs"]) == 0:
                GPA.pai_tracker.member_vars["n_switch_epochs"].append(
                    GPA.pai_tracker.member_vars["num_epochs_run"]
                )
            else:
                GPA.pai_tracker.member_vars["n_switch_epochs"].append(
                    GPA.pai_tracker.member_vars["n_switch_epochs"][-1]
                    + (
                        (GPA.pai_tracker.member_vars["num_epochs_run"])
                        - (GPA.pai_tracker.member_vars["switch_epochs"][-1])
                    )
                )

        GPA.pai_tracker.member_vars["switch_epochs"].append(
            GPA.pai_tracker.member_vars["num_epochs_run"]
        )
        GPA.pai_tracker.member_vars["last_switch"] = GPA.pai_tracker.member_vars[
            "switch_epochs"
        ][-1]

        # Because open source version is only doing neuron training for
        # gradient descent dendrites, switch back to n mode right away
        if (
            not GPA.pc.get_perforated_backpropagation()
        ) or GPA.pc.get_no_extra_n_modes():
            net = change_learning_modes(net, folder, name, doing_pai)
    else:
        if not GPA.pc.get_silent():
            print("Switching back to N...")
        set_best = GPA.pai_tracker.member_vars["current_n_set_global_best"]
        GPA.pai_tracker.set_neuron_training()
        if len(GPA.pai_tracker.member_vars["p_switch_epochs"]) == 0:
            GPA.pai_tracker.member_vars["p_switch_epochs"].append(
                (
                    (GPA.pai_tracker.member_vars["num_epochs_run"] - 1)
                    - (GPA.pai_tracker.member_vars["switch_epochs"][-1])
                )
            )
        else:
            GPA.pai_tracker.member_vars["p_switch_epochs"].append(
                GPA.pai_tracker.member_vars["p_switch_epochs"][-1]
                + (
                    (GPA.pai_tracker.member_vars["num_epochs_run"])
                    - (GPA.pai_tracker.member_vars["switch_epochs"][-1])
                )
            )
        GPA.pai_tracker.member_vars["switch_epochs"].append(
            GPA.pai_tracker.member_vars["num_epochs_run"]
        )
        GPA.pai_tracker.member_vars["last_switch"] = GPA.pai_tracker.member_vars[
            "switch_epochs"
        ][-1]
        # Will be false for open source implementation
        if GPA.pc.get_retain_all_dendrites() or (
            GPA.pc.get_learn_dendrites_live() and set_best
        ):
            if not GPA.pc.get_silent():
                print(
                    "Saving model before starting normal training to "
                    "retain PBNodes regardless of next N Phase results"
                )
            save_system(net, folder, name)
        # if its just doing P for learn PAI live then switch back immediately
        if GPA.pc.get_perforated_backpropagation() and GPA.pc.get_no_extra_n_modes():
            net = change_learning_modes(net, folder, name, doing_pai)

    GPA.pai_tracker.member_vars["param_counts"].append(count_params(net))

    return net
