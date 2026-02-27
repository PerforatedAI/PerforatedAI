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
from perforatedai import clean_perforatedai as CL
from perforatedai import blockwise_perforatedai as BPA
from perforatedai import network_perforatedai as NPA

try:
    from perforatedbp import utils_pbp as UPB

except ModuleNotFoundError as e:
    # Only pass if perforatedbp package itself is missing
    if e.name == 'perforatedbp':
        pass
    else:
        # perforatedbp exists but is missing a dependency
        raise
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


def convert_module(
    net,
    depth,
    name_so_far,
    converted_list,
    converted_names_list,
    neuron_module_class,
    tracked_module_class,
):
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
    if isinstance(net, neuron_module_class) or (
        (tracked_module_class is not None) and isinstance(net, tracked_module_class)
    ):
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
                if tracked_module_class is None:
                    continue
                setattr(
                    net,
                    submodule_id,
                    tracked_module_class(net.get_submodule(submodule_id), sub_name),
                )
                continue
            if sub_name in GPA.pc.get_module_ids_to_convert():
                if GPA.pc.get_verbose():
                    print("Seq ID is in convert IDs: %s" % sub_name)
                setattr(
                    net,
                    submodule_id,
                    neuron_module_class(net.get_submodule(submodule_id), sub_name),
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
                if tracked_module_class is None:
                    continue
                setattr(
                    net,
                    submodule_id,
                    tracked_module_class(net.get_submodule(submodule_id), sub_name),
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
                    neuron_module_class(net.get_submodule(submodule_id), sub_name),
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
                            neuron_module_class,
                            tracked_module_class,
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
                if tracked_module_class is None:
                    continue
                setattr(
                    net, member, tracked_module_class(getattr(net, member), sub_name)
                )
                continue
            if sub_name in GPA.pc.get_module_ids_to_convert():
                if GPA.pc.get_verbose():
                    print("Seq ID is in convert IDs: %s" % sub_name)
                setattr(
                    net, member, neuron_module_class(getattr(net, member), sub_name)
                )
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
                    "One of these must be selected to not be saved by calling, for example:"
                )
                print("GPA.pc.append_module_names_to_not_save(%s)" % sub_name)
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
                if tracked_module_class is None:
                    continue
                setattr(
                    net, member, tracked_module_class(getattr(net, member), sub_name)
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
                    neuron_module_class(getattr(net, member), sub_name),
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
                            neuron_module_class,
                            tracked_module_class,
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
        net = convert_module(
            net, 0, "", [], [], PA.PAINeuronModule, PA.TrackedNeuronModule
        )
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

def save_pai_net(net, folder, name):
    """Save the final pai network

    This can be called after training to save the final network
    with all scaffolding removed so only the refined weights remain

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
    # if running a DDP only save with first thread
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return

    # print('calling save: %s' % name)
    # GPA.pai_tracker.archive_layer()
    # These deep copys are required or the real model will also have its layers replaced
    net = prepare_final_model(net)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    save_point = folder + "/"
    if not os.path.isdir(save_point):
        os.mkdir(save_point)

    if GPA.pc.get_using_safe_tensors():
        if(GPA.pc.get_weight_tying_experimental()):
            save_model_with_weight_tying(net, save_point + name + "_pai.pt")
        else:
            save_file(net.state_dict(), save_point + name + "_pai.pt")
    else:
        torch.save(net, save_point + name + "_pai.pt")

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


def get_module_base_name(module):
    module_name = module.name
    # This should always be true
    if module_name[0] == ".":
        # strip "."
        module_name = module_name[1:]
    # If it was a dataparallel it will also have a module at the start
    # so strip that for loading
    if module_name[:6] == "module":
        module_name = module_name[7:]
    return module_name


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
    if(GPA.pc.get_verbose()):
        print('loading net from dict')
    pai_modules = get_pai_modules(net, 0)
    if pai_modules == []:
        print(
            "PAI load_net and load_system uses a state_dict so it must be\n"
            "called with a net after initialize_pai has been called"
        )
        print("This is being flagged because you are attempting to load a model\n"
            "that does not have any pai_modules in it.  Confirm that you are calling\n"
            "initialize_pai on the correct model, and the same model is the one\n"
            "being passed into add_validation_score"
        )
        import pdb

        pdb.set_trace()
        sys.exit(-1)
    if(GPA.pc.get_verbose()):
        print('setting up arrays and simulating cycles for %d pai modules' % len(pai_modules))
    for module in pai_modules:
        # Set up name to be what will be saved in the state dict
        module_name = get_module_base_name(module)
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
                    "your forward and backward pass."
                )
                print("module was: %s" % module.name)
                print("There are many reasons this can happen:")
                print(
                    "\n1 - check your model definition and forward function and "
                    "ensure this module is being used properly"
                )
                print(
                    "with GPA.pc.set_verbose(True) you can confirm this is the case if\n"
                    'you do not see a "setting d shape for" this module at the first training batch.'
                )
                print(
                    "If this is the case, and it is correct to not be passing data through it\n"
                    "Set it to be a tracked module with:\n"
                    'GPA.pc.append_module_ids_to_track(["%s"]) to leave it out '
                    % module.name
                )
                print(
                    "\n2 - This can happen if you adjusted your model "
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
                    "\n3 - This can happen is if the model where you called initialize_pai\n"
                    "and the model within add_validation_score are not the same. \n"
                    "Check if the module above and .%s have the same prefix\n"
                    % first_key
                )
                print(
                    "if one starts with .model or .base etc and the other does not, this is the problem."
                )
                
                print("\n4 - If you are using this module but then not actually including\n"
                    "the correct output tensor in the forward.  For example\n"
                    "if you are using an LSTM and forwarding hidden instead of otput\n"
                    "but your processors are set up to work with output"
                )
                print("\n5 - if you are not properly calling backward at all."
                    " If this is the first module in your network it is more"
                    "likely this is the problem"
                )
                print("\n6 - You have converted a module that is in a frozen"
                    " part of the network and thus no gradients are flowing"
                )
                print("\n7 - You are running multiple experiments at once with the same save_name."
                      " When running concurrent trials be sure to add save_name=<unique_name> to initialize_pai."
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

    # Clear gradients before saving the model
    if ((GPA.pai_tracker.member_vars["optimizer_instance"]) is not None) and \
        (GPA.pai_tracker.member_vars["optimizer_instance"] != []):
        GPA.pai_tracker.member_vars["optimizer_instance"].zero_grad()
    GPA.pai_tracker.clear_all_processors()
    return copy.deepcopy(net)


def prepare_final_model(net):
    """Prepare model for final save by removing scaffolding.
    
    This performs all cleanup steps to convert a PAI model with scaffolding
    into a clean final model ready for inference or distribution.
    
    Parameters
    ----------
    net : nn.Module
        The network to prepare.
        
    Returns
    -------
    nn.Module
        The cleaned model with scaffolding removed.
    """
    # Deep copy and clean the model (removes scaffolding)
    net = deep_copy_pai(net)
    net = BPA.blockwise_network(net)
    net = deep_copy_pai(net)
    net = CL.refresh_net(net)
    
    # Remove tracker_string (not needed for final model)
    if hasattr(net, 'tracker_string'):
        del net.tracker_string
    
    # Make parameters contiguous
    for param in net.parameters():
        param.data = param.data.contiguous()
    
    return net


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
    parameters = net.named_parameters()
    unique_params = {p.data_ptr(): p for name, p in parameters if 'parent_module' not in name}.values()
    return sum(p.numel() for p in unique_params)
    

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


def find_param_name_by_id(model, param_id):
    """
    This is only used for debugging.
    Return the fully-qualified parameter name (e.g. "layer1.conv.weight")
    for the parameter whose id matches param_id. Returns None if not found.

    This uses model.named_parameters(), which already recurses through submodules.
    """
    for name, p in model.named_parameters(recurse=True):
        if id(p) == param_id:
            return "." + name
    return None


def add_method_delegation_to_module(wrapper_module, method_name):
    """Add delegating methods to a wrapper module that has a main_module attribute.
    
    This adds the specified methods to the wrapper module instance so they
    properly delegate to the wrapped main_module. Works for any wrapper module
    (TrackedNeuronModule, PAINeuronModule, etc.) that has a main_module attribute.
    
    Args:
        wrapper_module: A wrapper module instance with a main_module attribute
        method_name: The method name to delegate (e.g., '_gradient_checkpointing_func')
    """
    import types
    
    if hasattr(wrapper_module.main_module, method_name):
        # Create a delegating method that forwards to main_module
        def make_delegated_method(name):
            def delegated_method(self, *args, **kwargs):
                main_module_attr = getattr(self.main_module, name, None)
                if main_module_attr is None:
                    raise AttributeError(
                        f"'{type(self.main_module).__name__}' object has no attribute '{name}'"
                    )
                if callable(main_module_attr):
                    return main_module_attr(*args, **kwargs)
                return main_module_attr
            return delegated_method
        
        # Bind it to this specific instance
        setattr(wrapper_module, method_name, types.MethodType(
            make_delegated_method(method_name), 
            wrapper_module
        ))


def apply_method_delegation_to_model(model, method_name, main_module_type):
    """Recursively apply method delegation to all wrapper modules with main_module in a model.
    
    This traverses the entire model and adds method delegation for any module that has
    a main_module attribute and optionally matches specified types.
    
    Args:
        model: The PyTorch model to traverse
        method_name: The method name to delegate (e.g., '_gradient_checkpointing_func')
        main_module_type: main_module type name to filter by.
                          Example: 'Qwen2DecoderLayer'
    
    Example:
        # Apply gradient checkpointing delegation to all decoder layers
        apply_method_delegation_to_model(
            model, 
            '_gradient_checkpointing_func',
            main_module_type='Qwen2DecoderLayer'
        )
    """
    count = 0
    for name, module in model.named_modules():
        # Check if module has main_module attribute (it's a wrapper)
        if hasattr(module, 'main_module'):
            # Check if we should apply based on main_module type
            should_apply = True
            if main_module_type is not None:
                main_module_type_name = type(module.main_module).__name__
                should_apply = main_module_type_name == main_module_type
            
            if should_apply:
                add_method_delegation_to_module(module, method_name)
                count += 1
    
    print(f"[PAI] Applied method delegation to {count} wrapper module instances")


def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects to strings.
    
    Parameters
    ----------
    obj : any
        The object to convert
        
    Returns
    -------
    any
        JSON-serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        # Convert non-serializable types to string
        return str(obj)


def extract_gpa_config():
    """Extract all configuration from GPA.pc by calling all get_* methods.
    
    Returns
    -------
    dict
        Dictionary with all GPA.pc configuration values and type metadata
        
    Examples
    --------
    >>> config = extract_gpa_config()
    >>> # Returns: {'max_dendrites': 10, 'device': 'cuda', '_types': {...}}
    """
    config = {}
    config_types = {}
    
    # Get all attributes from GPA.pc
    for attr_name in dir(GPA.pc):
        # Check if it starts with 'get_'
        if attr_name.startswith('get_'):
            try:
                # Get the method
                method = getattr(GPA.pc, attr_name)
                
                # Check if it's callable
                if callable(method):
                    # Call it and store result with key as name without 'get_'
                    key = attr_name[4:]  # Remove 'get_' prefix
                    value = method()
                    
                    # Check if this is an array (has corresponding append_ method)
                    append_method_name = f'append_{key}'
                    is_array = hasattr(GPA.pc, append_method_name)
                    
                    if is_array and isinstance(value, (list, tuple)):
                        # Store array element type
                        if len(value) > 0:
                            element_type = type(value[0]).__name__
                        else:
                            element_type = None  # empty array, no conversion needed
                        config_types[key] = {'is_array': True, 'element_type': element_type}
                    else:
                        # Store value type
                        config_types[key] = {'is_array': False, 'type': type(value).__name__}
                    
                    # Make sure value is JSON serializable
                    config[key] = make_json_serializable(value)
            except Exception as e:
                # Skip if method fails
                if GPA.pc.get_verbose():
                    print(f"Skipping {attr_name}: {e}")
                continue
    
    # Add types metadata to config
    config['_types'] = config_types
    
    return config


def convert_to_type(value, type_name):
    """Convert a value to the specified type.
    
    Parameters
    ----------
    value : any
        The value to convert
    type_name : str
        The target type name
        
    Returns
    -------
    any
        The converted value
    """
    if type_name == 'NoneType' or value is None:
        return None
    elif type_name == 'bool':
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value)
    elif type_name == 'int':
        return int(value)
    elif type_name == 'float':
        return float(value)
    elif type_name == 'str':
        return str(value)
    elif type_name == 'list':
        if not isinstance(value, list):
            return [value]
        return value
    elif type_name == 'dict':
        if not isinstance(value, dict):
            return {}
        return value
    elif type_name == 'type':
        # Handle type objects - convert string representation back to type
        if isinstance(value, str):
            # Try to evaluate the type string (e.g., "<class 'torch.nn.Linear'>")
            # Extract the class path from the string
            if value.startswith("<class '") and value.endswith("'>"):
                class_path = value[8:-2]  # Extract 'torch.nn.Linear' from "<class 'torch.nn.Linear'>"
                parts = class_path.split('.')
                # Try to import and get the type
                try:
                    module_name = '.'.join(parts[:-1])
                    class_name = parts[-1]
                    module = __import__(module_name, fromlist=[class_name])
                    return getattr(module, class_name)
                except Exception as e:
                    print(f"Warning: Could not convert type string '{value}' to actual type: {e}")
                    return value
            return value
        return value
    elif type_name == 'dtype':
        # Handle torch dtype objects
        if isinstance(value, str):
            # Convert string like "torch.float32" to actual dtype
            import torch
            try:
                # Try to get the dtype from torch module
                if value.startswith('torch.'):
                    dtype_name = value.split('.')[1]  # Get 'float32' from 'torch.float32'
                    return getattr(torch, dtype_name)
                else:
                    return getattr(torch, value)
            except Exception as e:
                print(f"Warning: Could not convert dtype string '{value}' to actual dtype: {e}")
                return value
        return value
    elif type_name == 'device':
        # Handle torch device objects
        if isinstance(value, str):
            # Convert string like "cuda" or "cpu" to torch.device
            import torch
            try:
                return torch.device(value)
            except Exception as e:
                print(f"Warning: Could not convert device string '{value}' to actual device: {e}")
                return value
        return value
    elif type_name == 'builtin_function_or_method':
        # Handle torch functions like torch.sigmoid, torch.relu, etc.
        if isinstance(value, str):
            # Parse string like "<built-in method sigmoid of type object at 0x...>"
            # to extract the function name
            import torch
            try:
                if '<built-in method ' in value and ' of type object' in value:
                    # Extract function name between '<built-in method ' and ' of type object'
                    start = value.find('<built-in method ') + len('<built-in method ')
                    end = value.find(' of type object')
                    func_name = value[start:end]
                    # Try to get the function from torch module
                    if hasattr(torch, func_name):
                        return getattr(torch, func_name)
                    else:
                        print(f"Warning: torch.{func_name} not found")
                        return value
                else:
                    return value
            except Exception as e:
                print(f"Warning: Could not convert builtin function string '{value}': {e}")
                return value
        return value
    else:
        # Unknown type - error and debug
        print(f"ERROR: Unknown type '{type_name}' for value: {value}")
        print(f"Type of value is: {type(value).__name__}")
        import pdb
        pdb.set_trace()
        return value


def convert_to_type_array(value, element_type):
    """Convert an array's elements to the specified type.
    
    Parameters
    ----------
    value : list or tuple
        The array to convert
    element_type : str or None
        The target type name for elements, None if array was empty
        
    Returns
    -------
    list
        The array with converted elements
    """
    if not isinstance(value, (list, tuple)):
        return value
    # If element_type is None (empty array), no conversion needed
    if element_type is None:
        return list(value) if isinstance(value, tuple) else value
    return [convert_to_type(item, element_type) for item in value]


def set_gpa_config(config):
    """Set GPA.pc configuration by calling all set_* methods.
    
    This is the reverse of extract_gpa_config(). It takes a configuration
    dictionary and calls the corresponding set_* methods on GPA.pc.
    Uses type metadata to ensure values are converted to the correct type.
    
    Parameters
    ----------
    config : dict
        Dictionary with configuration values (keys without 'set_' prefix)
        and optional '_types' metadata
        
    Examples
    --------
    >>> config = {'verbose': True, 'device': 'cuda'}
    >>> set_gpa_config(config)
    # Calls GPA.pc.set_verbose(True), GPA.pc.set_device('cuda'), etc.
    """
    set_count = 0
    skip_count = 0
    
    # Extract type information
    config_types = config.get('_types', {})
    
    for key, value in config.items():
        # Skip the types metadata
        if key == '_types':
            continue
            
        # Construct the set method name
        set_method_name = f'set_{key}'
        
        # Check if the set method exists
        if hasattr(GPA.pc, set_method_name):
            try:
                method = getattr(GPA.pc, set_method_name)
                if callable(method):
                    # Convert value to correct type if we have type info
                    if key in config_types:
                        type_info = config_types[key]
                        if type_info.get('is_array', False):
                            # Convert array elements to correct type
                            element_type = type_info.get('element_type', 'str')
                            value = convert_to_type_array(value, element_type)
                        else:
                            # Convert single value to correct type
                            value_type = type_info.get('type', 'str')
                            value = convert_to_type(value, value_type)
                    
                    method(value)
                    set_count += 1
                    if GPA.pc.get_verbose():
                        print(f"Set {key} = {value}")
            except Exception as e:
                skip_count += 1
                if GPA.pc.get_verbose():
                    print(f"Failed to set {key}: {e}")
        else:
            skip_count += 1
            if GPA.pc.get_verbose():
                print(f"No setter found for {key} (looking for {set_method_name})")
    
    if GPA.pc.get_verbose():
        print(f"Applied {set_count} PAI configuration settings ({skip_count} skipped)")
    
    return set_count


try:
    from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, HfApi

    def upload_to_huggingface(
        model, 
        repo_id, 
        license="apache-2.0",
        pipeline_tag=None,
        repo_url=None,
        tags=None,
        include_pai_config=True,
        **kwargs
    ):
        """Upload a model to HuggingFace Hub.
        
        Uploads model weights and PAI configuration to HuggingFace Hub.
        The configuration is saved in config.json and can be restored when loading.
        
        Parameters
        ----------
        model : nn.Module
            The model to upload
        repo_id : str
            Repository ID (format: "username/model-name")
        license : str, optional
            License for the model card, by default "apache-2.0"
        pipeline_tag : str, optional
            Pipeline tag for the model (e.g., "text-classification", "image-classification")
        repo_url : str, optional
            URL to the model's repository/documentation
        tags : list, optional
            List of tags for the model card
        include_pai_config : bool, optional
            Whether to include all GPA.pc configuration in the model config, by default True
        **kwargs
            Additional arguments passed to HfApi (token, private, etc.)
            
        Returns
        -------
        str
            URL of the uploaded model
            
        Examples
        --------
        >>> url = upload_to_huggingface(
        ...     model, 
        ...     "username/my-model",
        ...     license="mit",
        ...     pipeline_tag="image-classification",
        ...     tags=["pytorch", "vision"]
        ... )
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install it with: pip install huggingface_hub"
            )
        
        import tempfile
        import os
        
        # Prepare model same way as save_pai_net does
        model = prepare_final_model(model)
        
        # Calculate parameter count
        param_count = count_params(model)
        
        # Format parameter count for tags (e.g., "11m" for 11 million)
        if param_count >= 1e9:
            param_tag = f"{param_count/1e9:.0f}b"
        elif param_count >= 1e6:
            param_tag = f"{param_count/1e6:.0f}m"
        elif param_count >= 1e3:
            param_tag = f"{param_count/1e3:.0f}k"
        else:
            param_tag = str(param_count)
        
        # Create a temporary directory for files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model weights
            model_path = os.path.join(tmpdir, "model.safetensors")
            save_file(model.state_dict(), model_path)
            
            # Create config with PAI configuration
            config = {}
            if include_pai_config:
                pai_config = extract_gpa_config()
                config['pai_config'] = pai_config
                if GPA.pc.get_verbose():
                    print(f"Extracted {len(pai_config)} PAI configuration parameters")
            
            # Add parameter count at top level
            config['num_parameters'] = param_count
            
            # Add metadata
            if license:
                config['license'] = license
            if pipeline_tag:
                config['pipeline_tag'] = pipeline_tag
            if repo_url:
                config['repo_url'] = repo_url
            
            # Add tags with parameter count
            if tags is None:
                tags = []
            elif not isinstance(tags, list):
                tags = [tags]
            else:
                tags = tags.copy()  # Don't modify the original list
            
            # Add perforated-ai tag if not present
            if "perforated-ai" not in tags:
                tags.insert(0, "perforated-ai")
            
            # Add parameter count tag if not present
            if param_tag not in tags:
                tags.append(param_tag)
            
            config['tags'] = tags
            
            # Save config.json
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Upload to HuggingFace
            api = HfApi()
            
            # Extract token from kwargs if present
            token = kwargs.pop('token', None)
            private = kwargs.pop('private', None)
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
            except Exception as e:
                print(f"Repo may already exist: {e}")
            
            # Upload folder
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                token=token,
                **kwargs
            )
        
        print(f"Model uploaded to: https://huggingface.co/{repo_id}")
        if include_pai_config:
            print(f"PAI configuration saved in config.json")
        print(f"To reload, use: model = from_hf_pretrained(model, '{repo_id}')")
        
        return f"https://huggingface.co/{repo_id}"

    def from_hf_pretrained(net, repo_id):
        """Load a PerforatedAI model from HuggingFace Hub using PyTorchModelHubMixin.
        
        Args:
            net: The base model architecture (will be converted to PAI format)
            repo_id: HuggingFace Hub repository ID (e.g., "username/model-name")
        
        Returns:
            net: The loaded model with PAI modules initialized
        """
        
        # Wrap in a class that inherits from PyTorchModelHubMixin
        class PAIHFModel(net.__class__, PyTorchModelHubMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        
        # Create an instance that can use from_pretrained
        wrapped_net = PAIHFModel.__new__(PAIHFModel)
        wrapped_net.__dict__ = net.__dict__
        wrapped_net.__class__ = PAIHFModel
        
        # Download config.json to restore PAI configuration
        try:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'pai_config' in config:
                    #print(f"Restoring PAI configuration from HuggingFace")
                    set_gpa_config(config['pai_config'])
                else:
                    print("Warning: No pai_config found in config.json")
        except Exception as e:
            print(f"Warning: Could not load PAI config from HuggingFace: {e}")
        
        # Download model files from HuggingFace
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(model_path)
        wrapped_net = NPA.convert_network(wrapped_net)
        wrapped_net = NPA.load_pai_model_from_dict(wrapped_net, state_dict)
        return wrapped_net
except:
    def upload_to_huggingface(*args, **kwargs):
        raise ImportError(
            "huggingface_hub is required for upload_to_huggingface. "
            "Install it with: pip install huggingface_hub"
        )
