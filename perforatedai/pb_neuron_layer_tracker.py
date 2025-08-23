# Copyright (c) 2025 Perforated AI

import io
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pydoc import locate

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from perforatedai import pb_globals as PBG
from perforatedai import pb_layer as PB
from perforatedai import pb_utils as PBU

mpl.use("Agg")


class PAINeuronModuleTracker:
    """
    Manager class that tracks all neuron layers and dendrite layers,
    controls when new dendrites are added, and communicates signals to modules.
    """

    def __init__(
        self,
        doing_pai,
        save_name,
        making_graphs=True,
        param_vals_setting=-1,
        values_per_train_epoch=-1,
        values_per_val_epoch=-1,
    ):
        """Initialize the tracker with default settings."""
        # Dict of member vars and their types for saving
        self.member_vars = {}
        self.member_var_types = {}

        # Whether or not PAI will be running
        self.member_vars["doing_pai"] = doing_pai
        self.member_var_types["doing_pai"] = "bool"

        # How many Dendrites have been added
        self.member_vars["num_dendrites_added"] = 0
        self.member_var_types["num_dendrites_added"] = "int"

        # How many cycles have been run, *2 or *2+1 of the above
        self.member_vars["num_cycles"] = 0
        self.member_var_types["num_cycles"] = "int"

        # Pointers to all neuron wrapped modules
        self.neuron_module_vector = []

        # Pointers to all non neuron modules for tracking
        self.tracked_neuron_module_vector = []

        # Neuron training or dendrite training mode
        self.member_vars["mode"] = "n"
        self.member_var_types["mode"] = "string"

        # Number of epochs run excluding overwritten epochs
        self.member_vars["num_epochs_run"] = -1
        self.member_var_types["num_epochs_run"] = "int"

        # Number including overwritten epochs
        self.member_vars["total_epochs_run"] = -1
        self.member_var_types["total_epochs_run"] = "int"

        # Last epoch that validation/correlation score was improved
        self.member_vars["epoch_last_improved"] = 0
        self.member_var_types["epoch_last_improved"] = "int"

        # Running validation accuracy
        self.member_vars["running_accuracy"] = 0
        self.member_var_types["running_accuracy"] = "float"

        # True if maxing validation, False if minimizing Loss
        self.member_vars["maximizing_score"] = True
        self.member_var_types["maximizing_score"] = "bool"

        # Mode for switching back and forth between learning modes
        self.member_vars["switch_mode"] = PBG.SWITCH_MODE
        self.member_var_types["switch_mode"] = "int"

        # Epoch of the last switch
        self.member_vars["last_switch"] = 0
        self.member_var_types["last_switch"] = "int"

        # Highest validation score from current cycle
        self.member_vars["current_best_validation_score"] = 0
        self.member_var_types["current_best_validation_score"] = "float"

        # Last epoch where the learning rate was updated
        self.member_vars["initial_lr_test_epoch_count"] = -1
        self.member_var_types["initial_lr_test_epoch_count"] = "int"

        # Highest validation score of full run
        self.member_vars["global_best_validation_score"] = 0
        self.member_var_types["global_best_validation_score"] = "float"

        # List of switch epochs
        self.member_vars["switch_epochs"] = []
        self.member_var_types["switch_epochs"] = "int array"

        # Parameter counts at each network structure
        self.member_vars["param_counts"] = []
        self.member_var_types["param_counts"] = "int array"

        # List of epochs where switch was made to neuron training
        self.member_vars["n_switch_epochs"] = []
        self.member_var_types["n_switch_epochs"] = "int array"

        # List of epochs where switch was made to dendrite training
        self.member_vars["p_switch_epochs"] = []
        self.member_var_types["p_switch_epochs"] = "int array"

        # List of validation accuracies
        self.member_vars["accuracies"] = []
        self.member_var_types["accuracies"] = "float array"

        # List of epochs where score improved for scheduler updates
        self.member_vars["last_improved_accuracies"] = []
        self.member_var_types["last_improved_accuracies"] = "int array"

        # List of test accuracy scores registered
        self.member_vars["test_accuracies"] = []
        self.member_var_types["test_accuracies"] = "float array"

        # List of accuracies registered during neuron training
        self.member_vars["n_accuracies"] = []
        self.member_var_types["n_accuracies"] = "float array"

        # List of accuracies registered during dendrite training
        self.member_vars["p_accuracies"] = []
        self.member_var_types["p_accuracies"] = "float array"

        # Running average accuracies from recent epochs
        self.member_vars["running_accuracies"] = []
        self.member_var_types["running_accuracies"] = "float array"

        # List of additional scores recorded
        self.member_vars["extra_scores"] = {}
        self.member_var_types["extra_scores"] = "float array dictionary"

        # Extra scores not set to be graphed
        self.member_vars["extra_scores_without_graphing"] = {}
        self.member_var_types["extra_scores_without_graphing"] = (
            "float array dictionary"
        )

        # List of test scores
        self.member_vars["test_scores"] = []
        self.member_var_types["test_scores"] = "float array"

        # Extra scores calculated during neuron training
        self.member_vars["n_extra_scores"] = {}
        self.member_var_types["n_extra_scores"] = "float array dictionary"

        # List of training losses calculated
        self.member_vars["training_loss"] = []
        self.member_var_types["training_loss"] = "float array"

        # List of learning rates at each epoch
        self.member_vars["training_learning_rates"] = []
        self.member_var_types["training_learning_rates"] = "float array"

        # Best dendrite scores
        self.member_vars["best_scores"] = []
        self.member_var_types["best_scores"] = "float array array"

        # Current dendrite scores
        self.member_vars["current_scores"] = []
        self.member_var_types["current_scores"] = "float array array"

        # Times for neuron training epochs
        self.member_vars["n_epoch_times"] = []
        self.member_var_types["n_epoch_times"] = "float array"

        # Timing values
        self.member_vars["p_epoch_times"] = []
        self.member_var_types["p_epoch_times"] = "float array"
        self.member_vars["n_train_times"] = []
        self.member_var_types["n_train_times"] = "float array"
        self.member_vars["p_train_times"] = []
        self.member_var_types["p_train_times"] = "float array"
        self.member_vars["n_val_times"] = []
        self.member_var_types["n_val_times"] = "float array"
        self.member_vars["p_val_times"] = []
        self.member_var_types["p_val_times"] = "float array"

        # Setting for tracking timing
        self.member_vars["manual_train_switch"] = False
        self.member_var_types["manual_train_switch"] = "bool"

        # Tracking scores overwritten when reloading best model
        self.member_vars["overwritten_extras"] = []
        self.member_var_types["overwritten_extras"] = "float array dictionary array"
        self.member_vars["overwritten_vals"] = []
        self.member_var_types["overwritten_vals"] = "float array array"
        self.member_vars["overwritten_epochs"] = 0
        self.member_var_types["overwritten_epochs"] = "int"

        # Setting for determining scores
        self.member_vars["param_vals_setting"] = PBG.PARAM_VALS_SETTING
        self.member_var_types["param_vals_setting"] = "int"

        # Optimizer and scheduler types and instances
        self.member_vars["optimizer"] = None
        self.member_var_types["optimizer"] = "type"
        self.member_vars["scheduler"] = None
        self.member_var_types["scheduler"] = "type"
        self.member_vars["optimizer_instance"] = None
        self.member_var_types["optimizer_instance"] = "empty array"
        self.member_vars["scheduler_instance"] = None
        self.member_var_types["scheduler_instance"] = "empty array"

        # Flag for if the tracker was loaded
        self.loaded = False

        # Settings for tracking learning rates
        self.member_vars["current_n_learning_rate_initial_skip_steps"] = 0
        self.member_var_types["current_n_learning_rate_initial_skip_steps"] = "int"
        self.member_vars["last_max_learning_rate_steps"] = 0
        self.member_var_types["last_max_learning_rate_steps"] = "int"
        self.member_vars["last_max_learning_rate_value"] = -1
        self.member_var_types["last_max_learning_rate_value"] = "float"
        self.member_vars["current_cycle_lr_max_scores"] = []
        self.member_var_types["current_cycle_lr_max_scores"] = "float array"
        self.member_vars["current_step_count"] = 0
        self.member_var_types["current_step_count"] = "int"
        self.member_vars["committed_to_initial_rate"] = True
        self.member_var_types["committed_to_initial_rate"] = "bool"
        self.member_vars["current_n_set_global_best"] = True

        # Flag for if current dendrite achieved highest global score
        self.member_var_types["current_n_set_global_best"] = "bool"

        # Number of tries adding this dendrite count
        self.member_vars["num_dendrite_tries"] = 0
        self.member_var_types["num_dendrite_tries"] = "int"

        # Count of batches per epoch
        self.values_per_train_epoch = values_per_train_epoch
        self.values_per_val_epoch = values_per_val_epoch

        self.save_name = save_name
        self.making_graphs = making_graphs

        self.start_time = time.time()
        self.saved_time = 0
        self.start_epoch(internal_call=True)

        if PBG.VERBOSE:
            print(f'Initializing with switch_mode {self.member_vars["switch_mode"]}')

    def to_string(self):
        """Convert tracker values to string for saving with safetensors."""
        full_string = ""
        for var in self.member_vars:
            full_string += var + ","
            if self.member_vars[var] is None:
                full_string += "None"
                full_string += "\n"
            elif self.member_var_types[var] == "bool":
                full_string += str(self.member_vars[var])
                full_string += "\n"
            elif self.member_var_types[var] in ("int", "float", "string"):
                full_string += str(self.member_vars[var])
                full_string += "\n"
            elif self.member_var_types[var] == "type":
                name = (
                    self.member_vars[var].__module__
                    + "."
                    + self.member_vars[var].__name__
                )
                full_string += str(self.member_vars[var])
                full_string += "\n"
            elif self.member_var_types[var] == "empty array":
                full_string += "[]"
                full_string += "\n"
            elif self.member_var_types[var] in ("int array", "float array"):
                full_string += "\n"
                string = ""
                for val in self.member_vars[var]:
                    string += str(val) + ","
                # Remove the last comma
                string = string[:-1]
                full_string += string
                full_string += "\n"
            elif self.member_var_types[var] == "float array dictionary array":
                full_string += "\n"
                for array in self.member_vars[var]:
                    for key in array:
                        string = key + ","
                        for val in array[key]:
                            string += str(val) + ","
                        # Remove the last comma
                        string = string[:-1]
                        full_string += string
                        full_string += "\n"
                    full_string += "endkey"
                    full_string += "\n"
                full_string += "endarray"
                full_string += "\n"
            elif self.member_var_types[var] == "float array dictionary":
                full_string += "\n"
                for key in self.member_vars[var]:
                    string = key + ","
                    for val in self.member_vars[var][key]:
                        string += str(val) + ","
                    # Remove the last comma
                    string = string[:-1]
                    full_string += string
                    full_string += "\n"
                full_string += "end"
                full_string += "\n"
            elif self.member_var_types[var] == "float array array":
                full_string += "\n"
                for array in self.member_vars[var]:
                    string = ""
                    for val in array:
                        string += str(val) + ","
                    # Remove the last comma
                    string = string[:-1]
                    full_string += string
                    full_string += "\n"
                full_string += "end"
                full_string += "\n"
            else:
                print("Did not find a member variable")
                import pdb

                pdb.set_trace()
        return full_string

    def from_string(self, string):
        """Load tracker values from string."""
        f = io.StringIO(string)
        while True:
            line = f.readline()
            if not line:
                break
            vals = line.split(",")
            var = vals[0]

            if self.member_var_types[var] == "bool":
                val = vals[1][:-1]
                if val == "True":
                    self.member_vars[var] = True
                elif val == "False":
                    self.member_vars[var] = False
                elif val == "1":
                    self.member_vars[var] = 1
                elif val == "0":
                    self.member_vars[var] = 0
                else:
                    print("Something went wrong with loading")
                    import pdb

                    pdb.set_trace()
            elif self.member_var_types[var] == "int":
                val = vals[1]
                self.member_vars[var] = int(val)
            elif self.member_var_types[var] == "float":
                val = vals[1]
                self.member_vars[var] = float(val)
            elif self.member_var_types[var] == "string":
                val = vals[1][:-1]
                self.member_vars[var] = val
            elif self.member_var_types[var] == "type":
                # Ignore loading types, tracker should have them set up
                continue
            elif self.member_var_types[var] == "empty array":
                val = vals[1]
                self.member_vars[var] = []
            elif self.member_var_types[var] == "int array":
                vals = f.readline()[:-1].split(",")
                self.member_vars[var] = []
                if vals[0] == "":
                    continue
                for val in vals:
                    self.member_vars[var].append(int(val))
            elif self.member_var_types[var] == "float array":
                vals = f.readline()[:-1].split(",")
                self.member_vars[var] = []
                if vals[0] == "":
                    continue
                for val in vals:
                    self.member_vars[var].append(float(val))
            elif self.member_var_types[var] == "float array dictionary array":
                self.member_vars[var] = []
                line2 = f.readline()[:-1]
                while line2 != "endarray":
                    temp = {}
                    while line2 != "endkey":
                        vals = line2.split(",")
                        name = vals[0]
                        temp[name] = []
                        vals = vals[1:]
                        for val in vals:
                            temp[name].append(float(val))
                        line2 = f.readline()[:-1]
                    self.member_vars[var].append(temp)
                    line2 = f.readline()[:-1]
            elif self.member_var_types[var] == "float array dictionary":
                self.member_vars[var] = {}
                line2 = f.readline()[:-1]
                while line2 != "end":
                    vals = line2.split(",")
                    name = vals[0]
                    self.member_vars[var][name] = []
                    vals = vals[1:]
                    for val in vals:
                        self.member_vars[var][name].append(float(val))
                    line2 = f.readline()[:-1]
            elif self.member_var_types[var] == "float array array":
                self.member_vars[var] = []
                line2 = f.readline()[:-1]
                while line2 != "end":
                    vals = line2.split(",")
                    self.member_vars[var].append([])
                    if line2:
                        for val in vals:
                            self.member_vars[var][-1].append(float(val))
                    line2 = f.readline()[:-1]
            else:
                print("Did not find a member variable")
                import pdb

                pdb.set_trace()

    def save_tracker_settings(self):
        """
        Save tracker settings for DistributedDataParallel use.
        Instructions for use are in API customization.md
        """
        if not os.path.isdir(self.save_name):
            os.makedirs(self.save_name)
        f = open(self.save_name + "/array_dims.csv", "w")
        for layer in self.neuron_module_vector:
            f.write(
                f"{layer.name},{layer.dendrite_module.dendrite_values[0].out_channels}\n"
            )
        f.close()
        if not PBG.SILENT:
            print("Tracker settings saved.")
            print("You may now delete save_tracker_settings")

    def initialize_tracker_settings(self):
        """Initialize tracker settings from saved file."""
        channels = {}
        if not os.path.exists(self.save_name + "/array_dims.csv"):
            print(
                "You must call save_tracker_settings before "
                "initialize_tracker_settings"
            )
            print("Follow instructions in customization.md")
            import pdb

            pdb.set_trace()
        f = open(self.save_name + "/array_dims.csv", "r")
        for line in f:
            channels[line.split(",")[0]] = int(line.split(",")[1])
        for layer in self.neuron_module_vector:
            layer.dendrite_module.dendrite_values[0].setup_arrays(channels[layer.name])

    def set_optimizer_instance(self, optimizer_instance):
        """Set optimizer instance directly."""
        try:
            if (
                optimizer_instance.param_groups[0]["weight_decay"] > 0
                and PBG.WEIGHT_DECAY_ACCEPTED is False
            ):
                print(
                    "For PAI training it is recommended to not use "
                    "weight decay in your optimizer"
                )
                print(
                    "Set PBG.WEIGHT_DECAY_ACCEPTED = True to ignore this "
                    "warning or c to continue"
                )
                PBG.WEIGHT_DECAY_ACCEPTED = True
                import pdb

                pdb.set_trace()
        except:
            pass
        self.member_vars["optimizer_instance"] = optimizer_instance

    def set_optimizer(self, optimizer):
        """Set optimizer type."""
        self.member_vars["optimizer"] = optimizer

    def set_scheduler(self, scheduler):
        """Set scheduler type."""
        if scheduler is not torch.optim.lr_scheduler.ReduceLROnPlateau:
            if PBG.VERBOSE:
                print("Not using ReduceLROnPlateau, this is not recommended")
        self.member_vars["scheduler"] = scheduler

    def increment_scheduler(self, num_ticks, mode):
        """
        Increment the scheduler a set number of times.
        Used for finding best initial learning rate when adding dendrites.
        """
        current_steps = 0
        current_ticker = 0

        for param_group in PBG.pai_tracker.member_vars[
            "optimizer_instance"
        ].param_groups:
            learning_rate1 = param_group["lr"]

        if PBG.VERBOSE:
            print("Using scheduler:")
            print(type(self.member_vars["scheduler_instance"]))

        while current_ticker < num_ticks:
            if PBG.VERBOSE:
                print(
                    f"Lower start rate initial {learning_rate1} "
                    f'stepping {PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} times'
                )

            if (
                type(self.member_vars["scheduler_instance"])
                is torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if mode == "steplearning_rate":
                    # Step with counter as last improved accuracy
                    self.member_vars["scheduler_instance"].step(
                        metrics=self.member_vars["last_improved_accuracies"][
                            PBG.pai_tracker.steps_after_switch() - 1
                        ]
                    )
                elif mode == "incrementepoch_count":
                    # Step with improved epoch counts up to current location
                    self.member_vars["scheduler_instance"].step(
                        metrics=self.member_vars["last_improved_accuracies"][
                            -((num_ticks - 1) - current_ticker) - 1
                        ]
                    )
            else:
                self.member_vars["scheduler_instance"].step()

            for param_group in PBG.pai_tracker.member_vars[
                "optimizer_instance"
            ].param_groups:
                learning_rate2 = param_group["lr"]

            if learning_rate2 != learning_rate1:
                current_steps += 1
                learning_rate1 = learning_rate2
                if mode == "steplearning_rate":
                    current_ticker += 1
                if PBG.VERBOSE:
                    print(f"1 step {current_steps} to {learning_rate2}")

            if mode == "incrementepoch_count":
                current_ticker += 1

        return current_steps, learning_rate1

    def setup_optimizer(self, net, opt_args, sched_args=None):
        """Initialize the optimizer and scheduler when added."""
        if "weight_decay" in opt_args and not PBG.WEIGHT_DECAY_ACCEPTED:
            print(
                "For PAI training it is recommended to not use "
                "weight decay in your optimizer"
            )
            print(
                "Set PBG.WEIGHT_DECAY_ACCEPTED = True to ignore this "
                "warning or c to continue"
            )
            PBG.WEIGHT_DECAY_ACCEPTED = True
            import pdb

            pdb.set_trace()

        if "model" not in opt_args.keys():
            if self.member_vars["mode"] == "n":
                opt_args["params"] = filter(lambda p: p.requires_grad, net.parameters())
            else:
                opt_args["params"] = PBU.getPBNetworkParams(net)

        optimizer = self.member_vars["optimizer"](**opt_args)
        self.member_vars["optimizer_instance"] = optimizer

        if self.member_vars["scheduler"] is not None:
            self.member_vars["scheduler_instance"] = self.member_vars["scheduler"](
                optimizer, **sched_args
            )
            current_steps = 0

            for param_group in PBG.pai_tracker.member_vars[
                "optimizer_instance"
            ].param_groups:
                learning_rate1 = param_group["lr"]

            if PBG.VERBOSE:
                print(
                    f"Resetting scheduler with {PBG.pai_tracker.steps_after_switch()} "
                    f"steps and {PBG.INITIAL_HISTORY_AFTER_SWITCHES} initial ticks to skip"
                )

            # Find setting of previously used learning rate before adding dendrites
            if (
                PBG.pai_tracker.member_vars[
                    "current_n_learning_rate_initial_skip_steps"
                ]
                != 0
            ):
                additional_steps, learning_rate1 = self.increment_scheduler(
                    PBG.pai_tracker.member_vars[
                        "current_n_learning_rate_initial_skip_steps"
                    ],
                    "steplearning_rate",
                )
                current_steps += additional_steps

            if self.member_vars["mode"] == "n" or PBG.LEARN_DENDRITES_LIVE:
                initial = PBG.INITIAL_HISTORY_AFTER_SWITCHES
            else:
                initial = 0

            if PBG.pai_tracker.steps_after_switch() > initial:
                # Minus extra 1 because this gets called after start epoch
                additional_steps, learning_rate1 = self.increment_scheduler(
                    (PBG.pai_tracker.steps_after_switch() - initial) - 1,
                    "incrementepoch_count",
                )
                current_steps += additional_steps

            if PBG.VERBOSE:
                print(
                    f"Scheduler update loop with {current_steps} "
                    f"ended with {learning_rate1}"
                )
                print(
                    f"Scheduler ended with {current_steps} steps "
                    f"and lr of {learning_rate1}"
                )

            self.member_vars["current_step_count"] = current_steps
            return optimizer, self.member_vars["scheduler_instance"]
        else:
            return optimizer

    def clear_optimizer_and_scheduler(self):
        """Clear the instances for saving."""
        self.member_vars["optimizer_instance"] = None
        self.member_vars["scheduler_instance"] = None

    def switch_time(self):
        """
        Based on settings and scores, determine if it's time to switch
        between neuron and dendrite training.
        """
        switch_phrase = "No mode, this should never be the case."
        if self.member_vars["switch_mode"] == PBG.DOING_SWITCH_EVERY_TIME:
            switch_phrase = "DOING_SWITCH_EVERY_TIME"
        elif self.member_vars["switch_mode"] == PBG.DOING_HISTORY:
            switch_phrase = "DOING_HISTORY"
        elif self.member_vars["switch_mode"] == PBG.DOING_FIXED_SWITCH:
            switch_phrase = "DOING_FIXED_SWITCH"
        elif self.member_vars["switch_mode"] == PBG.DOING_NO_SWITCH:
            switch_phrase = "DOING_NO_SWITCH"

        if not PBG.SILENT:
            print(
                f'Checking PAI switch with mode {self.member_vars["mode"]}, '
                f'switch mode {switch_phrase}, epoch {self.member_vars["num_epochs_run"]}, '
                f'last improved epoch {self.member_vars["epoch_last_improved"]}, '
                f'total epochs {self.member_vars["total_epochs_run"]}, '
                f'n: {PBG.N_EPOCHS_TO_SWITCH}, num_cycles: {self.member_vars["num_cycles"]}'
            )

        if self.member_vars["switch_mode"] == PBG.DOING_NO_SWITCH:
            if not PBG.SILENT:
                print("Returning False - doing no switch mode")
            return False

        if self.member_vars["switch_mode"] == PBG.DOING_SWITCH_EVERY_TIME:
            if not PBG.SILENT:
                print("Returning True - switching every time")
            return True

        if (
            ((self.member_vars["mode"] == "n") or PBG.LEARN_DENDRITES_LIVE)
            and (self.member_vars["switch_mode"] == PBG.DOING_HISTORY)
            and (PBG.pai_tracker.member_vars["committed_to_initial_rate"] is False)
            and (PBG.DONT_GIVE_UP_UNLESS_LEARNING_RATE_LOWERED)
            and (
                self.member_vars["current_n_learning_rate_initial_skip_steps"]
                < self.member_vars["last_max_learning_rate_steps"]
            )
            and self.member_vars["scheduler"] is not None
        ):
            if not PBG.SILENT:
                print(
                    f"Returning False since no first step yet and comparing "
                    f'initial {self.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                    f'to last max {self.member_vars["last_max_learning_rate_steps"]}'
                )
            return False

        cap_switch = False
        if len(self.member_vars["switch_epochs"]) == 0:
            this_count = self.member_vars["num_epochs_run"]
        else:
            this_count = (
                self.member_vars["num_epochs_run"]
                - self.member_vars["switch_epochs"][-1]
            )

        if self.member_vars["switch_mode"] == PBG.DOING_HISTORY and (
            (
                (self.member_vars["mode"] == "n")
                and (
                    self.member_vars["num_epochs_run"]
                    - self.member_vars["epoch_last_improved"]
                    >= PBG.N_EPOCHS_TO_SWITCH
                )
                and this_count
                >= PBG.INITIAL_HISTORY_AFTER_SWITCHES + PBG.N_EPOCHS_TO_SWITCH
            )
            or cap_switch
        ):
            if not PBG.SILENT:
                print("Returning True - History and last improved is hit")
            return True

        if self.member_vars["switch_mode"] == PBG.DOING_FIXED_SWITCH and (
            (self.member_vars["total_epochs_run"] % PBG.FIXED_SWITCH_NUM == 0)
            and self.member_vars["num_epochs_run"] >= PBG.FIRST_FIXED_SWITCH_NUM
        ):
            if not PBG.SILENT:
                print("Returning True - Fixed switch number is hit")
            return True

        if not PBG.SILENT:
            print("Returning False - no triggers to switch have been hit")
        return False

    def steps_after_switch(self):
        """Based on settings, return value for steps since a switch."""
        if self.member_vars["param_vals_setting"] == PBG.PARAM_VALS_BY_TOTAL_EPOCH:
            return self.member_vars["num_epochs_run"]
        elif self.member_vars["param_vals_setting"] == PBG.PARAM_VALS_BY_UPDATE_EPOCH:
            return self.member_vars["num_epochs_run"] - self.member_vars["last_switch"]
        elif (
            self.member_vars["param_vals_setting"]
            == PBG.PARAM_VALS_BY_NEURON_EPOCH_START
        ):
            if self.member_vars["mode"] == "p":
                return (
                    self.member_vars["num_epochs_run"] - self.member_vars["last_switch"]
                )
            else:
                return self.member_vars["num_epochs_run"]
        else:
            print(
                f'{self.member_vars["param_vals_setting"]} is not a valid param vals option'
            )
            import pdb

            pdb.set_trace()

    def add_pai_neuron_module(self, new_module, initial_add=True):
        """Add neuron layers to internal vectors."""
        # If it's a duplicate, ignore the second addition
        if new_module in self.neuron_module_vector:
            return
        self.neuron_module_vector.append(new_module)
        if self.member_vars["doing_pai"]:
            PB.set_wrapped_params(new_module)
        if initial_add:
            self.member_vars["best_scores"].append([])
            self.member_vars["current_scores"].append([])

    def add_tracked_neuron_module(self, new_module, initial_add=True):
        """Add tracked layers to internal vectors."""
        # If it's a duplicate, ignore the second addition
        if new_module in self.tracked_neuron_module_vector:
            return
        self.tracked_neuron_module_vector.append(new_module)
        if self.member_vars["doing_pai"]:
            PB.set_tracked_params(new_module)

    def reset_module_vector(self, net, load_from_restart):
        """Clear internal vectors."""
        self.neuron_module_vector = []
        self.tracked_neuron_module_vector = []
        this_list = PBU.get_pai_modules(net, 0)
        for module in this_list:
            self.add_pai_neuron_module(module, initial_add=load_from_restart)
        this_list = PBU.get_tracked_modules(net, 0)
        for module in this_list:
            self.add_tracked_neuron_module(module, initial_add=load_from_restart)

    def reset_vals_for_score_reset(self):
        """Reset values if resetting scores."""
        if PBG.FIND_BEST_LR:
            self.member_vars["committed_to_initial_rate"] = False
        self.member_vars["current_n_set_global_best"] = False
        # Don't reset global best, but do reset current best
        self.member_vars["current_best_validation_score"] = 0
        self.member_vars["initial_lr_test_epoch_count"] = -1

    def set_dendrite_training(self):
        """Signal all layers to start dendrite training."""
        if PBG.VERBOSE:
            print("Calling set_dendrite_training")

        for layer in self.neuron_module_vector[:]:
            worked = layer.set_mode("p")
            """
            worked is False when a layer was added to the neuron module vector
            but then it's never actually been used. This can happen when
            you have set a layer to have requires_grad = False or when
            you have a module as a member variable but it's not actually
            part of the network. Should be moved to be a tracked layer
            rather than a neuron layer.
            """
            if not worked:
                self.neuron_module_vector.remove(layer)

        for layer in self.tracked_neuron_module_vector[:]:
            worked = layer.set_mode("p")

        self.create_new_dendrite_module()
        self.member_vars["mode"] = "p"
        self.member_vars["current_n_learning_rate_initial_skip_steps"] = 0

        if PBG.LEARN_DENDRITES_LIVE:
            self.reset_vals_for_score_reset()

        self.member_vars["last_max_learning_rate_steps"] = self.member_vars[
            "current_step_count"
        ]

        PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"] = []
        PBG.pai_tracker.member_vars["num_cycles"] += 1

    def set_neuron_training(self):
        """Signal all layers to start neuron training."""
        for module in self.neuron_module_vector:
            module.set_mode("n")
        for module in self.tracked_neuron_module_vector[:]:
            module.set_mode("n")

        self.member_vars["mode"] = "n"
        self.member_vars["num_dendrites_added"] += 1
        self.member_vars["current_n_learning_rate_initial_skip_steps"] = 0
        self.reset_vals_for_score_reset()

        self.member_vars["current_cycle_lr_max_scores"] = []
        if PBG.LEARN_DENDRITES_LIVE:
            self.member_vars["last_max_learning_rate_steps"] = self.member_vars[
                "current_step_count"
            ]
        PBG.pai_tracker.member_vars["num_cycles"] += 1

        if PBG.RESET_BEST_SCORE_ON_SWITCH:
            PBG.pai_tracker.member_vars["current_best_validation_score"] = 0
            PBG.pai_tracker.member_vars["running_accuracy"] = 0

    def start_epoch(self, internal_call=False):
        """Perform steps for when a new training epoch is about to begin."""
        if self.member_vars["manual_train_switch"] and internal_call:
            return

        if not internal_call and not self.member_vars["manual_train_switch"]:
            self.member_vars["manual_train_switch"] = True
            self.saved_time = 0
            self.member_vars["num_epochs_run"] = -1
            self.member_vars["total_epochs_run"] = -1

        end = time.time()
        if self.member_vars["manual_train_switch"]:
            if self.saved_time != 0:
                if self.member_vars["mode"] == "p":
                    self.member_vars["p_val_times"].append(end - self.saved_time)
                else:
                    self.member_vars["n_val_times"].append(end - self.saved_time)

        if self.member_vars["mode"] == "p":
            for layer in self.neuron_module_vector:
                for m in range(0, PBG.GLOBAL_CANDIDATES):
                    with torch.no_grad():
                        if PBG.VERBOSE:
                            print(f"Resetting score for {layer.name}")
                        layer.dendrite_module.dendrite_values[
                            m
                        ].best_score_improved_this_epoch = (
                            layer.dendrite_module.dendrite_values[
                                m
                            ].best_score_improved_this_epoch
                            * 0
                        )
                        layer.dendrite_module.dendrite_values[
                            m
                        ].nodes_best_improved_this_epoch = (
                            layer.dendrite_module.dendrite_values[
                                m
                            ].nodes_best_improved_this_epoch
                            * 0
                        )

        self.member_vars["num_epochs_run"] += 1
        self.member_vars["total_epochs_run"] = (
            self.member_vars["num_epochs_run"] + self.member_vars["overwritten_epochs"]
        )
        self.saved_time = end

    def stop_epoch(self, internal_call=False):
        """Perform steps when a training epoch has completed."""
        end = time.time()
        if self.member_vars["manual_train_switch"] and internal_call:
            return

        if self.member_vars["manual_train_switch"]:
            if self.member_vars["mode"] == "p":
                self.member_vars["p_train_times"].append(end - self.saved_time)
            else:
                self.member_vars["n_train_times"].append(end - self.saved_time)
        else:
            if self.member_vars["mode"] == "p":
                self.member_vars["p_epoch_times"].append(end - self.saved_time)
            else:
                self.member_vars["n_epoch_times"].append(end - self.saved_time)

        self.saved_time = end

    def initialize(
        self,
        model,
        doing_pai=True,
        save_name="PB",
        making_graphs=True,
        maximizing_score=True,
        num_classes=10000,
        values_per_train_epoch=-1,
        values_per_val_epoch=-1,
        zooming_graph=True,
    ):
        """Setup the tracker with initial settings."""
        model = PBU.convert_network(model)
        self.member_vars["doing_pai"] = doing_pai
        self.member_vars["maximizing_score"] = maximizing_score
        self.save_name = save_name
        self.zooming_graph = zooming_graph
        self.making_graphs = making_graphs

        if not self.loaded:
            self.member_vars["running_accuracy"] = (1.0 / num_classes) * 100

        self.values_per_train_epoch = values_per_train_epoch
        self.values_per_val_epoch = values_per_val_epoch

        if PBG.TESTING_DENDRITE_CAPACITY:
            if not PBG.SILENT:
                print("Running a test of Dendrite Capacity.")
            PBG.SWITCH_MODE = PBG.DOING_SWITCH_EVERY_TIME
            self.member_vars["switch_mode"] = PBG.SWITCH_MODE
            PBG.RETAIN_ALL_DENDRITES = True
            PBG.MAX_DENDRITE_TRIES = 1000
            PBG.MAX_DENDRITES = 1000
        else:
            if not PBG.SILENT:
                print("Running Dendrite Experiment")
        return model

    def save_graphs(self, extra_string=""):
        """
        Function to save graphs for all the values.
        TODO: clean this up, add comments, and separate into more functions
        """
        if not self.making_graphs:
            return

        save_folder = "./" + self.save_name + "/"

        plt.ioff()
        fig = plt.figure(figsize=(28, 14))

        # Plot with accuracy scores
        ax = plt.subplot(221)
        df1 = None

        for list_id in range(len(self.member_vars["overwritten_extras"])):
            for extra_id in self.member_vars["overwritten_extras"][list_id]:
                ax.plot(
                    np.arange(
                        len(self.member_vars["overwritten_extras"][list_id][extra_id])
                    ),
                    self.member_vars["overwritten_extras"][list_id][extra_id],
                    "r",
                )
            ax.plot(
                np.arange(len(self.member_vars["overwritten_vals"][list_id])),
                self.member_vars["overwritten_vals"][list_id],
                "b",
            )

        if PBG.DRAWING_PAI:
            accuracies = self.member_vars["accuracies"]
            extra_scores = self.member_vars["extra_scores"]
        else:
            accuracies = self.member_vars["n_accuracies"]
            extra_scores = self.member_vars["extra_scores"]

        ax.plot(np.arange(len(accuracies)), accuracies, label="Validation Scores")
        ax.plot(
            np.arange(len(self.member_vars["running_accuracies"])),
            self.member_vars["running_accuracies"],
            label="Validation Running Scores",
        )

        for extra_score in extra_scores:
            ax.plot(
                np.arange(len(extra_scores[extra_score])),
                extra_scores[extra_score],
                label=extra_score,
            )

        plt.title(save_folder + "/" + self.save_name + "Scores")
        plt.xlabel("Epochs")
        plt.ylabel("Score")

        # Add point at epoch last improved
        last_improved = self.member_vars["epoch_last_improved"]
        if PBG.DRAWING_PAI:
            ax.plot(
                last_improved,
                self.member_vars["global_best_validation_score"],
                "bo",
                label="Global best (y)",
            )
            ax.plot(
                last_improved,
                accuracies[last_improved],
                "go",
                label="Epoch Last Improved",
            )
        else:
            if self.member_vars["mode"] == "n":
                missed_time = self.member_vars["num_epochs_run"] - last_improved
                ax.plot(
                    (len(self.member_vars["n_accuracies"]) - 1) - missed_time,
                    self.member_vars["n_accuracies"][-(missed_time + 1)],
                    "go",
                    label="Epoch Last Improved",
                )

        pd1 = pd.DataFrame(
            {"Epochs": np.arange(len(accuracies)), "Validation Scores": accuracies}
        )
        pd2 = pd.DataFrame(
            {
                "Epochs": np.arange(len(self.member_vars["running_accuracies"])),
                "Validation Running Scores": self.member_vars["running_accuracies"],
            }
        )
        pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

        for extra_score in extra_scores:
            pd2 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(extra_scores[extra_score])),
                    extra_score: extra_scores[extra_score],
                }
            )
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

        extra_scores_without_graphing = self.member_vars[
            "extra_scores_without_graphing"
        ]
        for extra_score in extra_scores_without_graphing:
            pd2 = pd.DataFrame(
                {
                    "Epochs": np.arange(
                        len(extra_scores_without_graphing[extra_score])
                    ),
                    extra_score: extra_scores_without_graphing[extra_score],
                }
            )
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

        pd1.to_csv(
            save_folder + "/" + self.save_name + extra_string + "Scores.csv",
            index=False,
        )
        pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
        del pd1, pd2

        # Set y min and max to zoom in on important part of axis
        if (
            len(self.member_vars["switch_epochs"]) > 0
            and self.member_vars["switch_epochs"][0] > 0
            and self.zooming_graph
        ):
            if PBG.pai_tracker.member_vars["maximizing_score"]:
                min_val = np.array(
                    accuracies[0 : self.member_vars["switch_epochs"][0]]
                ).mean()
                for extra_score in extra_scores:
                    min_pot = np.array(
                        extra_scores[extra_score][
                            0 : self.member_vars["switch_epochs"][0]
                        ]
                    ).mean()
                    if min_pot < min_val:
                        min_val = min_pot
                ax.set_ylim(ymin=min_val)
            else:
                max_val = np.array(
                    accuracies[0 : self.member_vars["switch_epochs"][0]]
                ).mean()
                for extra_score in extra_scores:
                    max_pot = np.array(
                        extra_scores[extra_score][
                            0 : self.member_vars["switch_epochs"][0]
                        ]
                    ).mean()
                    if max_pot > max_val:
                        max_val = max_pot
                ax.set_ylim(ymax=max_val)

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        if PBG.DRAWING_PAI and self.member_vars["doing_pai"]:
            color = "r"
            for switcher in self.member_vars["switch_epochs"]:
                plt.axvline(x=switcher, ymin=0, ymax=1, color=color)
                if color == "r":
                    color = "b"
                else:
                    color = "r"
        else:
            for switcher in self.member_vars["n_switch_epochs"]:
                plt.axvline(x=switcher, ymin=0, ymax=1, color="b")

        # Plot the times for each training epoch
        ax = plt.subplot(222)
        if self.member_vars["manual_train_switch"]:
            ax.plot(
                np.arange(len(self.member_vars["n_train_times"])),
                self.member_vars["n_train_times"],
                label="Normal Epoch Train Times",
            )
            ax.plot(
                np.arange(len(self.member_vars["p_train_times"])),
                self.member_vars["p_train_times"],
                label="PAI Epoch Train Times",
            )
            ax.plot(
                np.arange(len(self.member_vars["n_val_times"])),
                self.member_vars["n_val_times"],
                label="Normal Epoch Val Times",
            )
            ax.plot(
                np.arange(len(self.member_vars["p_val_times"])),
                self.member_vars["p_val_times"],
                label="PAI Epoch Val Times",
            )

            plt.title(
                save_folder + "/" + self.save_name + "times (by train() and eval())"
            )
            plt.xlabel("Iteration")
            plt.ylabel("Epoch Time in Seconds ")
            ax.set_ylim(ymin=0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            pd1 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(self.member_vars["n_train_times"])),
                    "Normal Epoch Train Times": self.member_vars["n_train_times"],
                }
            )
            pd2 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(self.member_vars["p_train_times"])),
                    "PAI Epoch Train Times": self.member_vars["p_train_times"],
                }
            )
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

            pd2 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(self.member_vars["n_val_times"])),
                    "Normal Epoch Val Times": self.member_vars["n_val_times"],
                }
            )
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

            pd2 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(self.member_vars["p_val_times"])),
                    "PAI Epoch Val Times": self.member_vars["p_val_times"],
                }
            )
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

            pd1.to_csv(
                save_folder + "/" + self.save_name + extra_string + "Times.csv",
                index=False,
            )
            pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
            del pd1, pd2
        else:
            ax.plot(
                np.arange(len(self.member_vars["n_epoch_times"])),
                self.member_vars["n_epoch_times"],
                label="Normal Epoch Times",
            )
            ax.plot(
                np.arange(len(self.member_vars["p_epoch_times"])),
                self.member_vars["p_epoch_times"],
                label="PAI Epoch Times",
            )

            plt.title(
                save_folder + "/" + self.save_name + "times (by train() and eval())"
            )
            plt.xlabel("Iteration")
            plt.ylabel("Epoch Time in Seconds ")
            ax.set_ylim(ymin=0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            pd1 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(self.member_vars["n_epoch_times"])),
                    "Normal Epoch Times": self.member_vars["n_epoch_times"],
                }
            )
            pd2 = pd.DataFrame(
                {
                    "Epochs": np.arange(len(self.member_vars["p_epoch_times"])),
                    "PAI Epoch Times": self.member_vars["p_epoch_times"],
                }
            )
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

            pd1.to_csv(
                save_folder + "/" + self.save_name + extra_string + "Times.csv",
                index=False,
            )
            pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
            del pd1, pd2

        if self.values_per_train_epoch != -1 and self.values_per_val_epoch != -1:
            ax2 = ax.twinx()  # Second axes sharing same x-axis
            ax2.set_ylabel("Single Datapoint Time in Seconds")

            ax2.plot(
                np.arange(len(self.member_vars["n_train_times"])),
                np.array(self.member_vars["n_train_times"])
                / self.values_per_train_epoch,
                linestyle="dashed",
                label="Normal Train Item Times",
            )
            ax2.plot(
                np.arange(len(self.member_vars["p_train_times"])),
                np.array(self.member_vars["p_train_times"])
                / self.values_per_train_epoch,
                linestyle="dashed",
                label="PAI Train Item Times",
            )
            ax2.plot(
                np.arange(len(self.member_vars["n_val_times"])),
                np.array(self.member_vars["n_val_times"]) / self.values_per_val_epoch,
                linestyle="dashed",
                label="Normal Val Item Times",
            )
            ax2.plot(
                np.arange(len(self.member_vars["p_val_times"])),
                np.array(self.member_vars["p_val_times"]) / self.values_per_val_epoch,
                linestyle="dashed",
                label="PAI Val Item Times",
            )
            ax2.tick_params(axis="y")
            ax2.set_ylim(ymin=0)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot learning rates for each training epoch
        ax = plt.subplot(223)
        ax.plot(
            np.arange(len(self.member_vars["training_learning_rates"])),
            self.member_vars["training_learning_rates"],
            label="learning_rate",
        )
        plt.title(save_folder + "/" + self.save_name + "learning_rate")
        plt.xlabel("Epochs")
        plt.ylabel("learning_rate")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        pd1 = pd.DataFrame(
            {
                "Epochs": np.arange(len(self.member_vars["training_learning_rates"])),
                "learning_rate": self.member_vars["training_learning_rates"],
            }
        )
        pd1.to_csv(
            save_folder + "/" + self.save_name + extra_string + "learning_rate.csv",
            index=False,
        )
        pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
        del pd1

        pd1 = pd.DataFrame(
            {
                "Switch Number": np.arange(len(self.member_vars["switch_epochs"])),
                "Switch Epoch": self.member_vars["switch_epochs"],
            }
        )
        pd1.to_csv(
            save_folder + "/" + self.save_name + extra_string + "switch_epochs.csv",
            index=False,
        )
        pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
        del pd1

        pd1 = pd.DataFrame(
            {
                "Switch Number": np.arange(len(self.member_vars["param_counts"])),
                "Param Count": self.member_vars["param_counts"],
            }
        )
        pd1.to_csv(
            save_folder + "/" + self.save_name + extra_string + "param_counts.csv",
            index=False,
        )
        pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
        del pd1

        # Create best_test_scores.csv file
        test_scores = self.member_vars["test_scores"]
        # If not tracking test scores, use validation scores
        if len(self.member_vars["test_scores"]) == 0:
            test_scores = self.member_vars["accuracies"]

        if len(test_scores) != len(self.member_vars["accuracies"]):
            print("Your test scores are not the same length as validation scores")
            print(
                "add_test_score should only be included once, use add_extra_score for other variables"
            )

        switch_counts = len(self.member_vars["switch_epochs"])
        best_test = []
        best_valid = []
        associated_params = []

        for switch in range(0, switch_counts, 2):
            start_index = 0
            if switch != 0:
                start_index = self.member_vars["switch_epochs"][switch - 1] + 1
            end_index = self.member_vars["switch_epochs"][switch] + 1

            if PBG.pai_tracker.member_vars["maximizing_score"]:
                best_valid_index = start_index + np.argmax(
                    self.member_vars["accuracies"][start_index:end_index]
                )
            else:
                best_valid_index = start_index + np.argmin(
                    self.member_vars["accuracies"][start_index:end_index]
                )

            best_valid_score = self.member_vars["accuracies"][best_valid_index]
            best_test_score = test_scores[best_valid_index]
            best_valid.append(best_valid_score)
            best_test.append(best_test_score)
            associated_params.append(self.member_vars["param_counts"][switch])

        # If in neuron training mode but not the very first epoch
        if self.member_vars["mode"] == "n" and (
            (len(self.member_vars["switch_epochs"]) == 0)
            or (
                self.member_vars["switch_epochs"][-1] + 1
                != len(self.member_vars["accuracies"])
            )
        ):
            start_index = 0
            if len(self.member_vars["switch_epochs"]) != 0:
                start_index = self.member_vars["switch_epochs"][-1] + 1

            if PBG.pai_tracker.member_vars["maximizing_score"]:
                best_valid_index = start_index + np.argmax(
                    self.member_vars["accuracies"][start_index:]
                )
            else:
                best_valid_index = start_index + np.argmin(
                    self.member_vars["accuracies"][start_index:]
                )

            best_valid_score = self.member_vars["accuracies"][best_valid_index]
            best_test_score = test_scores[best_valid_index]
            best_valid.append(best_valid_score)
            best_test.append(best_test_score)
            associated_params.append(self.member_vars["param_counts"][-1])

        pd1 = pd.DataFrame(
            {
                "Param Counts": associated_params,
                "Max Valid Scores": best_valid,
                "Max Test Scores": best_test,
            }
        )
        pd1.to_csv(
            save_folder + "/" + self.save_name + extra_string + "best_test_scores.csv",
            index=False,
        )
        pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
        del pd1

        # Plot dendrite learning scores
        ax = plt.subplot(224)
        if self.member_vars["doing_pai"]:
            pd1 = None
            pd2 = None
            num_colors = len(self.neuron_module_vector)

            if (
                len(self.neuron_module_vector) > 0
                and len(self.member_vars["current_scores"][0]) != 0
            ):
                num_colors *= 2

            cm = plt.get_cmap("gist_rainbow")
            ax.set_prop_cycle(
                "color", [cm(1.0 * i / num_colors) for i in range(num_colors)]
            )

            for layer_id in range(len(self.neuron_module_vector)):
                ax.plot(
                    np.arange(len(self.member_vars["best_scores"][layer_id])),
                    self.member_vars["best_scores"][layer_id],
                    label=self.neuron_module_vector[layer_id].name,
                )

                pd2 = pd.DataFrame(
                    {
                        "Epochs": np.arange(
                            len(self.member_vars["best_scores"][layer_id])
                        ),
                        f"Best ever for all nodes Layer {self.neuron_module_vector[layer_id].name}": self.member_vars[
                            "best_scores"
                        ][
                            layer_id
                        ],
                    }
                )

                if pd1 is None:
                    pd1 = pd2
                else:
                    pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

                if len(self.member_vars["current_scores"][layer_id]) != 0:
                    ax.plot(
                        np.arange(len(self.member_vars["current_scores"][layer_id])),
                        self.member_vars["current_scores"][layer_id],
                        label=f"Best current for all Nodes Layer {self.neuron_module_vector[layer_id].name}",
                    )

                pd2 = pd.DataFrame(
                    {
                        "Epochs": np.arange(
                            len(self.member_vars["current_scores"][layer_id])
                        ),
                        f"Best current for all nodes Layer {self.neuron_module_vector[layer_id].name}": self.member_vars[
                            "current_scores"
                        ][
                            layer_id
                        ],
                    }
                )
                pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)

            plt.title(save_folder + "/" + self.save_name + " Best PBScores")
            plt.xlabel("Epochs")
            plt.ylabel("Best PBScore")
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                ncol=math.ceil(len(self.neuron_module_vector) / 30),
            )

            for switcher in self.member_vars["p_switch_epochs"]:
                plt.axvline(x=switcher, ymin=0, ymax=1, color="r")

            if self.member_vars["mode"] == "p":
                missed_time = self.member_vars["num_epochs_run"] - last_improved
                plt.axvline(
                    x=(len(self.member_vars["best_scores"][0]) - (missed_time + 1)),
                    ymin=0,
                    ymax=1,
                    color="g",
                )

            pd1.to_csv(
                save_folder + "/" + self.save_name + extra_string + "Best PBScores.csv",
                index=False,
            )
            pd1.to_csv("pd.csv", float_format="%.2f", na_rep="NAN!")
            del pd1, pd2

        fig.tight_layout()
        plt.savefig(save_folder + "/" + self.save_name + extra_string + ".png")
        plt.close("all")

    def add_loss(self, loss):
        """Add loss to tracking vectors."""
        if not isinstance(loss, (float, int)):
            loss = loss.item()
        self.member_vars["training_loss"].append(loss)

    def add_learning_rate(self, learning_rate):
        """Add learning rate to tracking vectors."""
        if not isinstance(learning_rate, (float, int)):
            learning_rate = learning_rate.item()
        self.member_vars["training_learning_rates"].append(learning_rate)

    def add_extra_score(self, score, extra_score_name):
        """Add extra score to tracking vectors."""
        if not isinstance(score, (float, int)):
            try:
                score = score.item()
            except:
                print(
                    "Scores added for Perforated Backpropagation should be "
                    "float, int, or tensor, yours is a:"
                )
                print(type(score))
                print("in add_extra_score")
                import pdb

                pdb.set_trace()

        if PBG.VERBOSE:
            print(f"Adding extra score {extra_score_name} of {float(score)}")

        if extra_score_name not in self.member_vars["extra_scores"]:
            self.member_vars["extra_scores"][extra_score_name] = []
        self.member_vars["extra_scores"][extra_score_name].append(score)

        if self.member_vars["mode"] == "n":
            if extra_score_name not in self.member_vars["n_extra_scores"]:
                self.member_vars["n_extra_scores"][extra_score_name] = []
            self.member_vars["n_extra_scores"][extra_score_name].append(score)

    def add_extra_score_without_graphing(self, score, extra_score_name):
        """Add extra score without graphing to tracking vectors."""
        if not isinstance(score, (float, int)):
            try:
                score = score.item()
            except:
                print(
                    "Scores added for Perforated Backpropagation should be "
                    "float, int, or tensor, yours is a:"
                )
                print(type(score))
                print("in add_extra_score_without_graphing")
                import pdb

                pdb.set_trace()

        if PBG.VERBOSE:
            print(f"Adding extra score {extra_score_name} of {float(score)}")

        if extra_score_name not in self.member_vars["extra_scores_without_graphing"]:
            self.member_vars["extra_scores_without_graphing"][extra_score_name] = []
        self.member_vars["extra_scores_without_graphing"][extra_score_name].append(
            score
        )

    def add_test_score(self, score, extra_score_name):
        """Add test score to tracking vectors."""
        self.add_extra_score(score, extra_score_name)

        if not isinstance(score, (float, int)):
            try:
                score = score.item()
            except:
                print(
                    "Scores added for Perforated Backpropagation should be "
                    "float, int, or tensor, yours is a:"
                )
                print(type(score))
                print("in add_test_score")
                import pdb

                pdb.set_trace()

        if PBG.VERBOSE:
            print(f"Adding test score {extra_score_name} of {float(score)}")
        self.member_vars["test_scores"].append(score)

    def add_validation_score(self, accuracy, net, force_switch=False):
        """
        Function to add the validation score. This is complex because it
        determines neuron and dendrite switching.
        WARNING: Do not call self anywhere in this function. When systems
        get loaded the actual tracker you are working with can change.
        TODO: clean this up, add comments, and separate into more functions
        """
        save_name = PBG.SAVE_NAME

        for param_group in PBG.pai_tracker.member_vars[
            "optimizer_instance"
        ].param_groups:
            learning_rate = param_group["lr"]
        PBG.pai_tracker.add_learning_rate(learning_rate)

        if len(PBG.pai_tracker.member_vars["param_counts"]) == 0:
            pytorch_total_params = sum(p.numel() for p in net.parameters())
            PBG.pai_tracker.member_vars["param_counts"].append(pytorch_total_params)

        if not PBG.SILENT:
            print(f"Adding validation score {accuracy:.8f}")

        # Make sure you are passing in the model and not the dataparallel wrapper
        if issubclass(type(net), nn.DataParallel):
            print("Need to call .module when using add validation score")
            import pdb

            pdb.set_trace()
            sys.exit(-1)

        if "module" in net.__dir__():
            print("Need to call .module when using add validation score")
            import pdb

            pdb.set_trace()
            sys.exit(-1)

        if not isinstance(accuracy, (float, int)):
            try:
                accuracy = accuracy.item()
            except:
                print(
                    "Scores added for Perforated Backpropagation should be "
                    "float, int, or tensor, yours is a:"
                )
                print(type(accuracy))
                print("in add_validation_score")
                import pdb

                pdb.set_trace()

        file_name = "best_model"
        if len(PBG.pai_tracker.member_vars["switch_epochs"]) == 0:
            epochs_since_cycle_switch = PBG.pai_tracker.member_vars["num_epochs_run"]
        else:
            epochs_since_cycle_switch = (
                PBG.pai_tracker.member_vars["num_epochs_run"]
                - PBG.pai_tracker.member_vars["switch_epochs"][-1]
            )

        # Don't update running accuracy during dendrite training
        if PBG.pai_tracker.member_vars["mode"] == "n" or PBG.LEARN_DENDRITES_LIVE:
            if epochs_since_cycle_switch < PBG.INITIAL_HISTORY_AFTER_SWITCHES:
                if epochs_since_cycle_switch == 0:
                    PBG.pai_tracker.member_vars["running_accuracy"] = accuracy
                else:
                    PBG.pai_tracker.member_vars[
                        "running_accuracy"
                    ] = PBG.pai_tracker.member_vars["running_accuracy"] * (
                        1 - (1.0 / (epochs_since_cycle_switch + 1))
                    ) + accuracy * (
                        1.0 / (epochs_since_cycle_switch + 1)
                    )
            else:
                PBG.pai_tracker.member_vars[
                    "running_accuracy"
                ] = PBG.pai_tracker.member_vars["running_accuracy"] * (
                    1.0 - 1.0 / PBG.HISTORY_LOOKBACK
                ) + accuracy * (
                    1.0 / PBG.HISTORY_LOOKBACK
                )

        PBG.pai_tracker.member_vars["accuracies"].append(accuracy)
        if PBG.pai_tracker.member_vars["mode"] == "n":
            PBG.pai_tracker.member_vars["n_accuracies"].append(accuracy)

        if (
            PBG.DRAWING_PAI
            or PBG.pai_tracker.member_vars["mode"] == "n"
            or PBG.LEARN_DENDRITES_LIVE
        ):
            PBG.pai_tracker.member_vars["running_accuracies"].append(
                PBG.pai_tracker.member_vars["running_accuracy"]
            )

        PBG.pai_tracker.stop_epoch(internal_call=True)

        # If it is neuron training mode
        if PBG.pai_tracker.member_vars["mode"] == "n" or PBG.LEARN_DENDRITES_LIVE:
            # Check if score improved
            score_improved = False
            if PBG.pai_tracker.member_vars["maximizing_score"]:
                score_improved = (
                    PBG.pai_tracker.member_vars["running_accuracy"]
                    * (1.0 - PBG.IMPROVEMENT_THRESHOLD)
                    > PBG.pai_tracker.member_vars["current_best_validation_score"]
                    and PBG.pai_tracker.member_vars["running_accuracy"]
                    - PBG.IMPROVEMENT_THRESHOLD_RAW
                    > PBG.pai_tracker.member_vars["current_best_validation_score"]
                )
            else:
                score_improved = (
                    PBG.pai_tracker.member_vars["running_accuracy"]
                    * (1.0 + PBG.IMPROVEMENT_THRESHOLD)
                    < PBG.pai_tracker.member_vars["current_best_validation_score"]
                    and (
                        PBG.pai_tracker.member_vars["running_accuracy"]
                        + PBG.IMPROVEMENT_THRESHOLD_RAW
                    )
                    < PBG.pai_tracker.member_vars["current_best_validation_score"]
                )

            enough_time = (
                epochs_since_cycle_switch > PBG.INITIAL_HISTORY_AFTER_SWITCHES
            ) or (
                PBG.pai_tracker.member_vars["switch_mode"]
                == PBG.DOING_SWITCH_EVERY_TIME
            )

            if (
                score_improved
                or PBG.pai_tracker.member_vars["current_best_validation_score"] == 0
            ) and enough_time:

                if PBG.pai_tracker.member_vars["maximizing_score"]:
                    if PBG.VERBOSE:
                        print(
                            f"\n\nGot score of {accuracy:.10f} "
                            f'(average {PBG.pai_tracker.member_vars["running_accuracy"]}, '
                            f"*{1-PBG.IMPROVEMENT_THRESHOLD}="
                            f'{PBG.pai_tracker.member_vars["running_accuracy"]*(1.0 - PBG.IMPROVEMENT_THRESHOLD)}) '
                            f'which is higher than {PBG.pai_tracker.member_vars["current_best_validation_score"]:.10f} '
                            f"by {PBG.IMPROVEMENT_THRESHOLD_RAW} so setting epoch to "
                            f'{PBG.pai_tracker.member_vars["num_epochs_run"]}\n\n'
                        )
                else:
                    if PBG.VERBOSE:
                        print(
                            f"\n\nGot score of {accuracy:.10f} "
                            f'(average {PBG.pai_tracker.member_vars["running_accuracy"]}, '
                            f"*{1+PBG.IMPROVEMENT_THRESHOLD}="
                            f'{PBG.pai_tracker.member_vars["running_accuracy"]*(1.0 + PBG.IMPROVEMENT_THRESHOLD)}) '
                            f'which is lower than {PBG.pai_tracker.member_vars["current_best_validation_score"]:.10f} '
                            f'so setting epoch to {PBG.pai_tracker.member_vars["num_epochs_run"]}\n\n'
                        )

                # Set the new best score
                PBG.pai_tracker.member_vars["current_best_validation_score"] = (
                    PBG.pai_tracker.member_vars["running_accuracy"]
                )

                # Check if global best
                is_global_best = False
                if PBG.pai_tracker.member_vars["maximizing_score"]:
                    is_global_best = (
                        PBG.pai_tracker.member_vars["current_best_validation_score"]
                        > PBG.pai_tracker.member_vars["global_best_validation_score"]
                    )
                else:
                    is_global_best = (
                        PBG.pai_tracker.member_vars["current_best_validation_score"]
                        < PBG.pai_tracker.member_vars["global_best_validation_score"]
                    )

                if (
                    is_global_best
                    or PBG.pai_tracker.member_vars["global_best_validation_score"] == 0
                ):
                    if PBG.VERBOSE:
                        print(
                            f"This also beats global best of "
                            f'{PBG.pai_tracker.member_vars["global_best_validation_score"]} so saving'
                        )
                    PBG.pai_tracker.member_vars["global_best_validation_score"] = (
                        PBG.pai_tracker.member_vars["current_best_validation_score"]
                    )
                    PBG.pai_tracker.member_vars["current_n_set_global_best"] = True
                    PBU.save_system(net, save_name, file_name)
                    if PBG.PAI_SAVES:
                        PBU.pai_save_system(net, save_name, file_name)

                PBG.pai_tracker.member_vars["epoch_last_improved"] = (
                    PBG.pai_tracker.member_vars["num_epochs_run"]
                )
                if PBG.VERBOSE:
                    print(
                        f'2 epoch improved is {PBG.pai_tracker.member_vars["epoch_last_improved"]}'
                    )
            else:
                if PBG.VERBOSE:
                    print("Not saving new best because:")
                    if epochs_since_cycle_switch <= PBG.INITIAL_HISTORY_AFTER_SWITCHES:
                        print(
                            f"Not enough history since switch {epochs_since_cycle_switch} <= "
                            f"{PBG.INITIAL_HISTORY_AFTER_SWITCHES}"
                        )
                    elif PBG.pai_tracker.member_vars["maximizing_score"]:
                        print(
                            f"Got score of {accuracy} "
                            f'(average {PBG.pai_tracker.member_vars["running_accuracy"]}, '
                            f"*{1-PBG.IMPROVEMENT_THRESHOLD}="
                            f'{PBG.pai_tracker.member_vars["running_accuracy"]*(1.0 - PBG.IMPROVEMENT_THRESHOLD)}) '
                            f"which is not higher than "
                            f'{PBG.pai_tracker.member_vars["current_best_validation_score"]}'
                        )
                    else:
                        print(
                            f"Got score of {accuracy} "
                            f'(average {PBG.pai_tracker.member_vars["running_accuracy"]}, '
                            f"*{1+PBG.IMPROVEMENT_THRESHOLD}="
                            f'{PBG.pai_tracker.member_vars["running_accuracy"]*(1.0 + PBG.IMPROVEMENT_THRESHOLD)}) '
                            f"which is not lower than "
                            f'{PBG.pai_tracker.member_vars["current_best_validation_score"]}'
                        )

                # If it's the first epoch, save a model
                if len(PBG.pai_tracker.member_vars["accuracies"]) == 1:
                    if PBG.VERBOSE:
                        print("Saving first model or all models")
                    PBU.save_system(net, save_name, file_name)
                    if PBG.PAI_SAVES:
                        PBU.pai_save_system(net, save_name, file_name)

        # Save the latest model
        if PBG.TEST_SAVES:
            PBU.save_system(net, save_name, "latest")
        if PBG.PAI_SAVES:
            PBU.pai_save_system(net, save_name, "latest")

        PBG.pai_tracker.member_vars["last_improved_accuracies"].append(
            PBG.pai_tracker.member_vars["epoch_last_improved"]
        )
        restructured = False

        # If it is time to switch based on scores and counter
        if PBG.pai_tracker.switch_time() or force_switch:
            # If testing dendrite capacity switch after enough dendrites added
            if (
                (PBG.pai_tracker.member_vars["mode"] == "n")
                and (PBG.pai_tracker.member_vars["num_dendrites_added"] > 3)
                and PBG.TESTING_DENDRITE_CAPACITY
            ):
                PBG.pai_tracker.save_graphs()
                print(
                    "Successfully added 3 dendrites with "
                    "PBG.TESTING_DENDRITE_CAPACITY = True (default). "
                    "You may now set that to False and run a real experiment."
                )
                import pdb

                pdb.set_trace()
                return net, False, True

            # If doing neuron training but this dendrite count didn't improve
            if (
                (PBG.pai_tracker.member_vars["mode"] == "n") or PBG.LEARN_DENDRITES_LIVE
            ) and (PBG.pai_tracker.member_vars["current_n_set_global_best"] is False):

                if PBG.VERBOSE:
                    print(
                        f"Planning to switch to p mode but best beat last: "
                        f'{PBG.pai_tracker.member_vars["current_n_set_global_best"]} '
                        f"current start lr steps: "
                        f'{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                        f"and last maximum lr steps: "
                        f'{PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                        f'for rate: {PBG.pai_tracker.member_vars["last_max_learning_rate_value"]:.8f}'
                    )

                now = datetime.now()
                dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                if PBG.VERBOSE:
                    print(
                        f'1 saving break {dt_string}_noImprove_lr_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                    )

                PBG.pai_tracker.save_graphs(
                    f'{dt_string}_noImprove_lr_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                )

                if (
                    PBG.pai_tracker.member_vars["num_dendrite_tries"]
                    < PBG.MAX_DENDRITE_TRIES
                ):
                    if not PBG.SILENT:
                        print(
                            f"Dendrites did not improve but current tries "
                            f'{PBG.pai_tracker.member_vars["num_dendrite_tries"]} '
                            f"is less than max tries {PBG.MAX_DENDRITE_TRIES} "
                            f"so loading last switch and trying new Dendrites."
                        )
                    old_tries = PBG.pai_tracker.member_vars["num_dendrite_tries"]
                    # Load best model from previous n mode
                    net = PBU.change_learning_modes(
                        net,
                        save_name,
                        file_name,
                        PBG.pai_tracker.member_vars["doing_pai"],
                    )
                    PBG.pai_tracker.member_vars["num_dendrite_tries"] = old_tries + 1
                else:
                    if not PBG.SILENT:
                        print(
                            f"Dendrites did not improve system and "
                            f'{PBG.pai_tracker.member_vars["num_dendrite_tries"]} >= '
                            f"{PBG.MAX_DENDRITE_TRIES} so returning training_complete."
                        )
                        print(
                            "You should now exit your training loop and "
                            "best_model will be your final model for inference"
                        )
                    PBU.load_system(net, save_name, file_name, switch_call=True)
                    PBG.pai_tracker.save_graphs()
                    PBU.pai_save_system(net, save_name, "final_clean")
                    return net, True, True

            # Else if did improve, keep dendrites and switch to new p mode
            else:
                if PBG.VERBOSE:
                    print(
                        f"Calling SWITCH_MODE with "
                        f'{PBG.pai_tracker.member_vars["current_n_set_global_best"]}, '
                        f'{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}, '
                        f'{PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]}, '
                        f'{PBG.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                    )

                if (PBG.pai_tracker.member_vars["mode"] == "n") and (
                    PBG.MAX_DENDRITES
                    == PBG.pai_tracker.member_vars["num_dendrites_added"]
                ):
                    if not PBG.SILENT:
                        print(
                            f"Last Dendrites were good and this hit the max of {PBG.MAX_DENDRITES}"
                        )
                    PBU.load_system(net, save_name, file_name, switch_call=True)
                    PBG.pai_tracker.save_graphs()
                    PBU.pai_save_system(net, save_name, "final_clean")
                    return net, True, True

                if PBG.pai_tracker.member_vars["mode"] == "n":
                    PBG.pai_tracker.member_vars["num_dendrite_tries"] = 0
                    if PBG.VERBOSE:
                        print(
                            "Adding new dendrites without resetting which means "
                            "the last ones improved. Resetting num_dendrite_tries"
                        )

                PBG.pai_tracker.save_graphs(
                    f'_beforeSwitch_{len(PBG.pai_tracker.member_vars["switch_epochs"])}'
                )

                if PBG.TEST_SAVES:
                    PBU.save_system(
                        net,
                        save_name,
                        f'beforeSwitch_{len(PBG.pai_tracker.member_vars["switch_epochs"])}',
                    )
                    # Copy current best model from this set of dendrites
                    shutil.copyfile(
                        f"{save_name}/best_model.pt",
                        f'{save_name}/best_model_beforeSwitch_{len(PBG.pai_tracker.member_vars["switch_epochs"])}.pt',
                    )
                    if PBG.EXTRA_VERBOSE:
                        import pdb

                        pdb.set_trace()

                    net = PBU.change_learning_modes(
                        net,
                        save_name,
                        file_name,
                        PBG.pai_tracker.member_vars["doing_pai"],
                    )

            # If restructured is true, clear scheduler/optimizer before saving
            restructured = True
            PBG.pai_tracker.clear_optimizer_and_scheduler()

            # Save the model from after the switch
            PBU.save_system(
                net,
                save_name,
                f'switch_{len(PBG.pai_tracker.member_vars["switch_epochs"])}',
            )

        # If not time to switch and you have a scheduler, increment it
        elif PBG.pai_tracker.member_vars["scheduler"] is not None:
            """
            Process for finding best initial learning rate for dendrites:
            1. Start at default rate
            2. Learn at that rate until scheduler increments twice
            3. Save that version, start dendrites at LR current increment - 1
            4. Repeat 2 and 3 until version has worse final score at set LR
            5. Load previous model with best accuracy at that LR as initial rate
            """
            for param_group in PBG.pai_tracker.member_vars[
                "optimizer_instance"
            ].param_groups:
                learning_rate1 = param_group["lr"]

            if (
                type(PBG.pai_tracker.member_vars["scheduler_instance"])
                is torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if (
                    epochs_since_cycle_switch > PBG.INITIAL_HISTORY_AFTER_SWITCHES
                    or PBG.pai_tracker.member_vars["mode"] == "p"
                ):
                    if PBG.VERBOSE:
                        print(
                            f"Updating scheduler with last improved "
                            f'{PBG.pai_tracker.member_vars["epoch_last_improved"]} '
                            f'from current {PBG.pai_tracker.member_vars["num_epochs_run"]}'
                        )
                    if PBG.pai_tracker.member_vars["scheduler"] is not None:
                        PBG.pai_tracker.member_vars["scheduler_instance"].step(
                            metrics=accuracy
                        )
                        if (
                            PBG.pai_tracker.member_vars["scheduler"]
                            is torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            if PBG.VERBOSE:
                                print(
                                    f"Scheduler is now at "
                                    f'{PBG.pai_tracker.member_vars["scheduler_instance"].num_bad_epochs} bad epochs'
                                )
                else:
                    if PBG.VERBOSE:
                        print("Not stepping optimizer since hasnt initialized")

            elif PBG.pai_tracker.member_vars["scheduler"] is not None:
                if (
                    epochs_since_cycle_switch > PBG.INITIAL_HISTORY_AFTER_SWITCHES
                    or PBG.pai_tracker.member_vars["mode"] == "p"
                ):
                    if PBG.VERBOSE:
                        print(
                            f"Incrementing scheduler to count "
                            f'{PBG.pai_tracker.member_vars["scheduler_instance"]._step_count}'
                        )
                    PBG.pai_tracker.member_vars["scheduler_instance"].step()
                    if (
                        PBG.pai_tracker.member_vars["scheduler"]
                        is torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        if PBG.VERBOSE:
                            print(
                                f"Scheduler is now at "
                                f'{PBG.pai_tracker.member_vars["scheduler_instance"].num_bad_epochs} bad epochs'
                            )

            if (
                epochs_since_cycle_switch <= PBG.INITIAL_HISTORY_AFTER_SWITCHES
                and PBG.pai_tracker.member_vars["mode"] == "n"
            ):
                if PBG.VERBOSE:
                    print(
                        f"Not stepping with history {PBG.INITIAL_HISTORY_AFTER_SWITCHES} "
                        f"and current {epochs_since_cycle_switch}"
                    )

            for param_group in PBG.pai_tracker.member_vars[
                "optimizer_instance"
            ].param_groups:
                learning_rate2 = param_group["lr"]

            stepped = False
            at_last_count = False

            if PBG.VERBOSE:
                print(
                    f"Checking if at last with scores "
                    f'{len(PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"])}, '
                    f"count since switch {epochs_since_cycle_switch} "
                    f"and last total lr step count "
                    f'{PBG.pai_tracker.member_vars["initial_lr_test_epoch_count"]}'
                )

            # Check if at double or exactly the test count
            if (
                len(PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"]) == 0
                and epochs_since_cycle_switch
                == PBG.pai_tracker.member_vars["initial_lr_test_epoch_count"] * 2
            ) or (
                len(PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"]) == 1
                and epochs_since_cycle_switch
                == PBG.pai_tracker.member_vars["initial_lr_test_epoch_count"]
            ):
                at_last_count = True

            if PBG.VERBOSE:
                print(
                    f"At last count {at_last_count} with count {epochs_since_cycle_switch} "
                    f'and last LR count {PBG.pai_tracker.member_vars["initial_lr_test_epoch_count"]}'
                )

            if learning_rate1 != learning_rate2:
                stepped = True
                PBG.pai_tracker.member_vars["current_step_count"] += 1

                if PBG.VERBOSE:
                    print(
                        f"Learning rate just stepped to {learning_rate2:.10e} "
                        f'with {PBG.pai_tracker.member_vars["current_step_count"]} total steps'
                    )

                if (
                    PBG.pai_tracker.member_vars["current_step_count"]
                    == PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]
                ):
                    if PBG.VERBOSE:
                        print(
                            f'{PBG.pai_tracker.member_vars["current_step_count"]} '
                            f"steps is the max of the last switch mode"
                        )
                    # Set it when 1->2 gets to 2, not when 0->1 hits 2 as stopping point
                    if (
                        PBG.pai_tracker.member_vars["current_step_count"]
                        - PBG.pai_tracker.member_vars[
                            "current_n_learning_rate_initial_skip_steps"
                        ]
                        == 1
                    ):
                        PBG.pai_tracker.member_vars["initial_lr_test_epoch_count"] = (
                            epochs_since_cycle_switch
                        )

            if PBG.VERBOSE:
                print(
                    f"Learning rates were {learning_rate1:.8e} and {learning_rate2:.8e} "
                    f'started with {PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}, '
                    f'and is now at {PBG.pai_tracker.member_vars["current_step_count"]} '
                    f'committed {PBG.pai_tracker.member_vars["committed_to_initial_rate"]} '
                    f"then either this (non zero) or eventually comparing to "
                    f'{PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                    f'steps or rate {PBG.pai_tracker.member_vars["last_max_learning_rate_value"]:.8f}'
                )

            # If learning rate just stepped, check restart at lower rate
            if (
                (PBG.pai_tracker.member_vars["scheduler"] is not None)
                and
                # If potentially might have higher accuracy
                (
                    (PBG.pai_tracker.member_vars["mode"] == "n")
                    or PBG.LEARN_DENDRITES_LIVE
                )
                and
                # And learning rate just stepped
                (stepped or at_last_count)
            ):

                # If hasn't committed to a learning rate for this cycle yet
                if not PBG.pai_tracker.member_vars["committed_to_initial_rate"]:
                    best_score_so_far = PBG.pai_tracker.member_vars[
                        "global_best_validation_score"
                    ]

                    if PBG.VERBOSE:
                        print(
                            f"In statements to check next learning rate with "
                            f"stepped {stepped} and max count {at_last_count}"
                        )

                    # If no scores saved for this dendrite and initial LR test did second step
                    if len(
                        PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"]
                    ) == 0 and (
                        PBG.pai_tracker.member_vars["current_step_count"]
                        - PBG.pai_tracker.member_vars[
                            "current_n_learning_rate_initial_skip_steps"
                        ]
                        == 2
                        or at_last_count
                    ):

                        restructured = True
                        PBG.pai_tracker.clear_optimizer_and_scheduler()

                        # Save system for this initial condition
                        old_global = PBG.pai_tracker.member_vars[
                            "global_best_validation_score"
                        ]
                        old_accuracy = PBG.pai_tracker.member_vars[
                            "current_best_validation_score"
                        ]
                        old_counts = PBG.pai_tracker.member_vars[
                            "initial_lr_test_epoch_count"
                        ]
                        skip1 = PBG.pai_tracker.member_vars[
                            "current_n_learning_rate_initial_skip_steps"
                        ]

                        now = datetime.now()
                        dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                        PBG.pai_tracker.save_graphs(
                            f'{dt_string}_PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                        )

                        if PBG.TEST_SAVES:
                            PBU.save_system(
                                net,
                                save_name,
                                f'PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                            )

                        if PBG.VERBOSE:
                            print(
                                f"Saving with initial steps: {dt_string}_PBCount_"
                                f'{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                                f'{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                                f"with current best {old_accuracy}"
                            )

                        # Load back at start and try with lower initial learning rate
                        net = PBU.load_system(
                            net,
                            save_name,
                            f'switch_{len(PBG.pai_tracker.member_vars["switch_epochs"])}',
                            switch_call=True,
                        )
                        PBG.pai_tracker.member_vars[
                            "current_n_learning_rate_initial_skip_steps"
                        ] = (skip1 + 1)
                        PBG.pai_tracker.member_vars[
                            "current_cycle_lr_max_scores"
                        ].append(old_accuracy)
                        PBG.pai_tracker.member_vars["global_best_validation_score"] = (
                            old_global
                        )
                        PBG.pai_tracker.member_vars["initial_lr_test_epoch_count"] = (
                            old_counts
                        )

                    # If there is one score already, this is first step at next score
                    elif (
                        len(PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"])
                        == 1
                    ):
                        PBG.pai_tracker.member_vars[
                            "current_cycle_lr_max_scores"
                        ].append(
                            PBG.pai_tracker.member_vars["current_best_validation_score"]
                        )

                        # If this LR's score was worse than last LR's score
                        lr_score_worse = False
                        if PBG.pai_tracker.member_vars["maximizing_score"]:
                            lr_score_worse = (
                                PBG.pai_tracker.member_vars[
                                    "current_cycle_lr_max_scores"
                                ][0]
                                > PBG.pai_tracker.member_vars[
                                    "current_cycle_lr_max_scores"
                                ][1]
                            )
                        else:
                            lr_score_worse = (
                                PBG.pai_tracker.member_vars[
                                    "current_cycle_lr_max_scores"
                                ][0]
                                < PBG.pai_tracker.member_vars[
                                    "current_cycle_lr_max_scores"
                                ][1]
                            )

                        if lr_score_worse:
                            restructured = True
                            PBG.pai_tracker.clear_optimizer_and_scheduler()

                            if PBG.VERBOSE:
                                print(
                                    f'Got initial {PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]-1} '
                                    f'step score {PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"][0]} '
                                    f'and {PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                                    f'score at step {PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"][1]} '
                                    f"so loading old score"
                                )

                            prior_best = PBG.pai_tracker.member_vars[
                                "current_cycle_lr_max_scores"
                            ][0]

                            now = datetime.now()
                            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                            PBG.pai_tracker.save_graphs(
                                f'{dt_string}_PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                            )

                            if PBG.TEST_SAVES:
                                PBU.save_system(
                                    net,
                                    save_name,
                                    f'PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                                )

                            if PBG.VERBOSE:
                                print(
                                    f"Saving with initial steps: {dt_string}_PBCount_"
                                    f'{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                                    f'{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                                )

                            if PBG.TEST_SAVES:
                                net = PBU.load_system(
                                    net,
                                    save_name,
                                    f'PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]-1}',
                                    switch_call=True,
                                )

                            # Save graphs for chosen one
                            now = datetime.now()
                            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                            PBG.pai_tracker.save_graphs(
                                f'{dt_string}_PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}PICKED'
                            )

                            if PBG.TEST_SAVES:
                                PBU.save_system(
                                    net,
                                    save_name,
                                    f'PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                                )

                            if PBG.VERBOSE:
                                print(
                                    f"Saving with initial steps: {dt_string}_PBCount_"
                                    f'{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                                    f'{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                                )

                            PBG.pai_tracker.member_vars["committed_to_initial_rate"] = (
                                True
                            )
                            PBG.pai_tracker.member_vars[
                                "last_max_learning_rate_steps"
                            ] = PBG.pai_tracker.member_vars["current_step_count"]
                            PBG.pai_tracker.member_vars[
                                "last_max_learning_rate_value"
                            ] = learning_rate2
                            PBG.pai_tracker.member_vars[
                                "current_best_validation_score"
                            ] = prior_best

                            if PBG.VERBOSE:
                                print(
                                    f"Setting last max steps to "
                                    f'{PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                                    f'and lr {PBG.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                                )

                        else:  # Current LR score is better
                            if PBG.VERBOSE:
                                print(
                                    f'Got initial {PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]-1} '
                                    f'step score {PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"][0]} '
                                    f'and {PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                                    f'score at step {PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"][1]} '
                                    f"so NOT loading old score and continuing with this score"
                                )

                            if (
                                at_last_count
                            ):  # If this is the last one, set it to be picked
                                restructured = True
                                PBG.pai_tracker.clear_optimizer_and_scheduler()

                                now = datetime.now()
                                dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                                PBG.pai_tracker.save_graphs(
                                    f'{dt_string}_PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}PICKED'
                                )

                                if PBG.TEST_SAVES:
                                    PBU.save_system(
                                        net,
                                        save_name,
                                        f'PBCount_{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                                    )

                                if PBG.VERBOSE:
                                    print(
                                        f"Saving with initial steps: {dt_string}_PBCount_"
                                        f'{PBG.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                                        f'{PBG.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                                    )

                                PBG.pai_tracker.member_vars[
                                    "committed_to_initial_rate"
                                ] = True
                                PBG.pai_tracker.member_vars[
                                    "last_max_learning_rate_steps"
                                ] = PBG.pai_tracker.member_vars["current_step_count"]
                                PBG.pai_tracker.member_vars[
                                    "last_max_learning_rate_value"
                                ] = learning_rate2

                                if PBG.VERBOSE:
                                    print(
                                        f"Setting last max steps to "
                                        f'{PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                                        f'and lr {PBG.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                                    )

                        PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"] = []

                    elif (
                        len(PBG.pai_tracker.member_vars["current_cycle_lr_max_scores"])
                        == 2
                    ):
                        print(
                            "Should never be here. Please let Perforated AI know if this happened."
                        )
                        import pdb

                        pdb.set_trace()

                    PBG.pai_tracker.member_vars["global_best_validation_score"] = (
                        best_score_so_far
                    )

                else:
                    if PBG.VERBOSE:
                        print(
                            f"Setting last max steps to "
                            f'{PBG.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                            f'and lr {PBG.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                        )
                    PBG.pai_tracker.member_vars["last_max_learning_rate_steps"] += 1
                    PBG.pai_tracker.member_vars["last_max_learning_rate_value"] = (
                        learning_rate2
                    )

        PBG.pai_tracker.start_epoch(internal_call=True)
        PBG.pai_tracker.save_graphs()

        if restructured:
            PBG.pai_tracker.member_vars["epoch_last_improved"] = (
                PBG.pai_tracker.member_vars["num_epochs_run"]
            )
            if PBG.VERBOSE:
                print(
                    f"Setting epoch last improved to "
                    f'{PBG.pai_tracker.member_vars["epoch_last_improved"]}'
                )

            now = datetime.now()
            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

            if PBG.VERBOSE:
                print("Not saving restructure right now")

            for param in net.parameters():
                param.data = param.data.contiguous()

        if PBG.VERBOSE:
            print(
                f"Completed adding score. Restructured is {restructured}, "
                f"\ncurrent switch list is:"
            )
            print(PBG.pai_tracker.member_vars["switch_epochs"])

        # Always False for training complete if nothing triggered that training is over
        return net, restructured, False

    def clear_all_processors(self):
        """Clear all processors from modules."""
        for module in self.neuron_module_vector:
            module.clear_processors()

    def create_new_dendrite_module(self):
        """Add dendrite module to all neuron modules."""
        for module in self.neuron_module_vector:
            module.create_new_dendrite_module()
