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
import pdb

from perforatedai import globals_perforatedai as GPA
from perforatedai import modules_perforatedai as PA
from perforatedai import utils_perforatedai as UPA


try:
    from perforatedbp import tracker_pbp as TPB
except ModuleNotFoundError:
    pass  # Module not found, pass silently
except ImportError as e:
    print(f"Import error occurred: {e}")

mpl.use("Agg")

# Status constants for restructuring during add_validation_score
NO_MODEL_UPDATE = 0
NETWORK_RESTRUCTURED = 1
TRAINING_COMPLETE = 2

# Status constant for each batch
STEP_CLEARED = 0
STEP_CALLED = 1


def update_restructuring_status(old_status, new_status):
    """Update restructured variable during add_validation_score

    Update the restructuring status based on the new status.
    If the new status is that there was not an update,
    dont overwrite the old status which may show there was an update.

    Parameters
    ----------
    old_status : int
        The old restructuring status.
    new_status : int
        The new restructuring status.

    Returns
    -------
    int
        The updated restructuring status.

    """
    if new_status == NETWORK_RESTRUCTURED or new_status == TRAINING_COMPLETE:
        return NETWORK_RESTRUCTURED
    else:
        return old_status


def update_learning_rate():
    """Update the learning rate in the tracker."""
    for param_group in GPA.pai_tracker.member_vars["optimizer_instance"].param_groups:
        learning_rate = param_group["lr"]
    GPA.pai_tracker.add_learning_rate(learning_rate)


def update_param_count(net):
    """Update the parameter count in the tracker if not already set.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model to count parameters for.
    Returns
    -------
    None
    """
    if len(GPA.pai_tracker.member_vars["param_counts"]) == 0:
        GPA.pai_tracker.member_vars["param_counts"].append(UPA.count_params(net))


def check_input_problems(net, accuracy):
    """Check for potential input problems in add_validation_score.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model to check.
    accuracy : float, int, or torch.Tensor
        The accuracy score to validate.

    Returns
    -------
    float
        The validated accuracy score.

    """

    # Make sure you are passing in the model and not the dataparallel wrapper
    if issubclass(type(net), nn.DataParallel):
        print("Need to call .module when using add validation score")
        pdb.set_trace()
        sys.exit(-1)

    if "module" in net.__dir__():
        print("Need to call .module when using add validation score")
        pdb.set_trace()
        sys.exit(-1)

    if not isinstance(accuracy, (float, int)):
        try:
            accuracy = accuracy.item()
        except:
            print(
                "Scores added for add_validation_score should be "
                "float, int, or tensor, yours is a:"
            )
            print(type(accuracy))
            pdb.set_trace()
            sys.exit(-1)
    return accuracy


def update_running_accuracy(accuracy, epochs_since_cycle_switch):
    """Add the new accuracy to the tracker.

    Parameters
    ----------
    accuracy : float, int, or torch.Tensor
        The accuracy score to add.
    epochs_since_cycle_switch : int
        The number of epochs since the last cycle switch.

    Returns
    -------
    None

    """
    # Only update running_accuracy when neurons are being updated
    if GPA.pai_tracker.member_vars["mode"] == "n" or GPA.pc.get_learn_dendrites_live():
        if epochs_since_cycle_switch < GPA.pc.get_initial_history_after_switches():
            if epochs_since_cycle_switch == 0:
                GPA.pai_tracker.member_vars["running_accuracy"] = accuracy
            else:
                GPA.pai_tracker.member_vars[
                    "running_accuracy"
                ] = GPA.pai_tracker.member_vars["running_accuracy"] * (
                    1 - (1.0 / (epochs_since_cycle_switch + 1))
                ) + accuracy * (
                    1.0 / (epochs_since_cycle_switch + 1)
                )
        else:
            GPA.pai_tracker.member_vars[
                "running_accuracy"
            ] = GPA.pai_tracker.member_vars["running_accuracy"] * (
                1.0 - 1.0 / GPA.pc.get_history_lookback()
            ) + accuracy * (
                1.0 / GPA.pc.get_history_lookback()
            )

    GPA.pai_tracker.member_vars["accuracies"].append(accuracy)
    if GPA.pai_tracker.member_vars["mode"] == "n":
        GPA.pai_tracker.member_vars["n_accuracies"].append(accuracy)

    if (
        GPA.pc.get_drawing_pai()
        or GPA.pai_tracker.member_vars["mode"] == "n"
        or GPA.pc.get_learn_dendrites_live()
    ):
        GPA.pai_tracker.member_vars["running_accuracies"].append(
            GPA.pai_tracker.member_vars["running_accuracy"]
        )


def score_beats_current_best(new_score, old_score):
    """Check if the new score beats the current best score.

    Parameters
    ----------
    new_score : float
        The new score to compare.
    old_score : float
        The old score to compare against.

    Returns
    -------
    bool
        True if the new score beats the old score, False otherwise.

    Notes
    -----
    Must beat the old score by the margins set in globals for improvement thresholds.

    """
    return (
        GPA.pai_tracker.member_vars["maximizing_score"]
        and (new_score * (1.0 - GPA.pc.get_improvement_threshold()) > old_score)
        and new_score - GPA.pc.get_improvement_threshold_raw() > old_score
    ) or (
        (not GPA.pai_tracker.member_vars["maximizing_score"])
        and (new_score * (1.0 + GPA.pc.get_improvement_threshold()) < old_score)
        and (new_score + GPA.pc.get_improvement_threshold_raw()) < old_score
    )


def check_new_best(net, accuracy, epochs_since_cycle_switch):
    """Check if the new accuracy is a new best.

    Performs saves if new best score is found.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model being trained.
    accuracy : float
        The accuracy score to check.
    epochs_since_cycle_switch : int
        The number of epochs since the last cycle switch.

    Returns
    -------
    None

    """
    score_improved = score_beats_current_best(
        GPA.pai_tracker.member_vars["running_accuracy"],
        GPA.pai_tracker.member_vars["current_best_validation_score"],
    )

    enough_time = (
        epochs_since_cycle_switch > GPA.pc.get_initial_history_after_switches()
    ) or (GPA.pai_tracker.member_vars["switch_mode"] == GPA.pc.DOING_SWITCH_EVERY_TIME)

    if (
        score_improved
        or GPA.pai_tracker.member_vars["current_best_validation_score"] == 0
    ) and enough_time:

        if GPA.pai_tracker.member_vars["maximizing_score"]:
            if GPA.pc.get_verbose():
                print(
                    f"\n\nGot score of {accuracy:.10f} "
                    f'(average {GPA.pai_tracker.member_vars["running_accuracy"]}, '
                    f"*{1-GPA.pc.get_improvement_threshold()}="
                    f'{GPA.pai_tracker.member_vars["running_accuracy"]*(1.0 - GPA.pc.get_improvement_threshold())}) '
                    f'which is higher than {GPA.pai_tracker.member_vars["current_best_validation_score"]:.10f} '
                    f"by {GPA.pc.get_improvement_threshold_raw()} so setting epoch to "
                    f'{GPA.pai_tracker.member_vars["num_epochs_run"]}\n\n'
                )
        else:
            if GPA.pc.get_verbose():
                print(
                    f"\n\nGot score of {accuracy:.10f} "
                    f'(average {GPA.pai_tracker.member_vars["running_accuracy"]}, '
                    f"*{1+GPA.pc.get_improvement_threshold()}="
                    f'{GPA.pai_tracker.member_vars["running_accuracy"]*(1.0 + GPA.pc.get_improvement_threshold())}) '
                    f'which is lower than {GPA.pai_tracker.member_vars["current_best_validation_score"]:.10f} '
                    f'so setting epoch to {GPA.pai_tracker.member_vars["num_epochs_run"]}\n\n'
                )

        # Set the new best score
        GPA.pai_tracker.member_vars["current_best_validation_score"] = (
            GPA.pai_tracker.member_vars["running_accuracy"]
        )
        GPA.pai_tracker.member_vars["epoch_last_improved"] = (
            GPA.pai_tracker.member_vars["num_epochs_run"]
        )
        if GPA.pc.get_verbose():
            print(
                f'2 epoch improved is {GPA.pai_tracker.member_vars["epoch_last_improved"]}'
            )
        # Immediately update this list before saving so loading will have it correctly
        GPA.pai_tracker.member_vars["last_improved_accuracies"].append(
            GPA.pai_tracker.member_vars["epoch_last_improved"]
        )
        # Check if global best
        is_global_best = score_beats_current_best(
            GPA.pai_tracker.member_vars["current_best_validation_score"],
            GPA.pai_tracker.member_vars["global_best_validation_score"],
        )

        if (
            is_global_best
            or GPA.pai_tracker.member_vars["global_best_validation_score"] == 0
        ):
            if GPA.pc.get_verbose():
                print(
                    f"This also beats global best of "
                    f'{GPA.pai_tracker.member_vars["global_best_validation_score"]} so saving'
                )
            GPA.pai_tracker.member_vars["global_best_validation_score"] = (
                GPA.pai_tracker.member_vars["current_best_validation_score"]
            )
            GPA.pai_tracker.member_vars["current_n_set_global_best"] = True
            UPA.save_system(net, GPA.pc.get_save_name(), "best_model")
            if GPA.pc.get_pai_saves():
                UPA.pai_save_system(net, GPA.pc.get_save_name(), "best_model")
    else:
        if GPA.pc.get_verbose():
            print("Not saving new best because:")
            if epochs_since_cycle_switch <= GPA.pc.get_initial_history_after_switches():
                print(
                    f"Not enough history since switch {epochs_since_cycle_switch} <= "
                    f"{GPA.pc.get_initial_history_after_switches()}"
                )
            elif GPA.pai_tracker.member_vars["maximizing_score"]:
                print(
                    f"Got score of {accuracy} "
                    f'(average {GPA.pai_tracker.member_vars["running_accuracy"]}, '
                    f"*{1-GPA.pc.get_improvement_threshold()}="
                    f'{GPA.pai_tracker.member_vars["running_accuracy"]*(1.0 - GPA.pc.get_improvement_threshold())}) '
                    f"which is not higher than "
                    f'{GPA.pai_tracker.member_vars["current_best_validation_score"]}'
                )
            else:
                print(
                    f"Got score of {accuracy} "
                    f'(average {GPA.pai_tracker.member_vars["running_accuracy"]}, '
                    f"*{1+GPA.pc.get_improvement_threshold()}="
                    f'{GPA.pai_tracker.member_vars["running_accuracy"]*(1.0 + GPA.pc.get_improvement_threshold())}) '
                    f"which is not lower than "
                    f'{GPA.pai_tracker.member_vars["current_best_validation_score"]}'
                )
        GPA.pai_tracker.member_vars["last_improved_accuracies"].append(
            GPA.pai_tracker.member_vars["epoch_last_improved"]
        )
        # If it's the first epoch, save as best anyway
        if len(GPA.pai_tracker.member_vars["accuracies"]) == 1:
            if GPA.pc.get_verbose():
                print("Saving first model or all models")
            UPA.save_system(net, GPA.pc.get_save_name(), "best_model")
            if GPA.pc.get_pai_saves():
                UPA.pai_save_system(net, GPA.pc.get_save_name(), "best_model")


def process_no_improvement(net):
    """Handle the case where no improvement is observed.

    If the new dendrite did not improve scores, but its time to switch modes
    either trigger the end of learning or reset to the previous dendrite
    to try again.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model being trained.

    Returns
    -------
    int
        The status of restructuring or training completion.
    torch.nn.Module
        The potentially modified neural network model.

    """
    if GPA.pc.get_verbose():
        print(
            f"Planning to switch to p mode but best beat last: "
            f'{GPA.pai_tracker.member_vars["current_n_set_global_best"]} '
            f"current start lr steps: "
            f'{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
            f"and last maximum lr steps: "
            f'{GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
            f'for rate: {GPA.pai_tracker.member_vars["last_max_learning_rate_value"]:.8f}'
        )

    now = datetime.now()
    dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

    if GPA.pc.get_verbose():
        print(
            f'1 saving break {dt_string}_noImprove_lr_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
        )

    GPA.pai_tracker.save_graphs(
        f'{dt_string}_noImprove_lr_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
    )

    if (
        GPA.pai_tracker.member_vars["num_dendrite_tries"]
        < GPA.pc.get_max_dendrite_tries()
    ):
        if not GPA.pc.get_silent():
            print(
                f"Dendrites did not improve but current tries "
                f'{GPA.pai_tracker.member_vars["num_dendrite_tries"]} '
                f"is less than max tries {GPA.pc.get_max_dendrite_tries()} "
                f"so loading last switch and trying new Dendrites."
            )
        old_tries = GPA.pai_tracker.member_vars["num_dendrite_tries"]
        # Load best model from previous n mode
        net = UPA.change_learning_modes(
            net,
            GPA.pc.get_save_name(),
            "best_model",
            GPA.pai_tracker.member_vars["doing_pai"],
        )
        GPA.pai_tracker.member_vars["num_dendrite_tries"] = old_tries + 1
        return NETWORK_RESTRUCTURED, net
    else:
        if not GPA.pc.get_silent():
            print(
                f"Dendrites did not improve system and "
                f'{GPA.pai_tracker.member_vars["num_dendrite_tries"]} >= '
                f"{GPA.pc.get_max_dendrite_tries()} so returning training_complete."
            )
            print(
                "You should now exit your training loop and "
                "best_model will be your final model for inference"
            )
        UPA.load_system(net, GPA.pc.get_save_name(), "best_model", switch_call=True)
        GPA.pai_tracker.save_graphs()
        UPA.pai_save_system(net, GPA.pc.get_save_name(), "final_clean")
        return TRAINING_COMPLETE, net


def process_final_network(net):
    """When the max number of dendrites has been hit load the best_model and return

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model being trained.

    Returns
    -------
    torch.nn.Module
        The final neural network model.
    """

    if not GPA.pc.get_silent():
        print(
            f"Last Dendrites were good and this hit the max of {GPA.pc.get_max_dendrites()}"
        )
    GPA.pai_tracker.save_graphs("before_final")
    UPA.load_system(net, GPA.pc.get_save_name(), "best_model", switch_call=True)
    GPA.pai_tracker.save_graphs()
    UPA.pai_save_system(net, GPA.pc.get_save_name(), "final_clean")
    return net


def process_scheduler_update(net, accuracy, epochs_since_cycle_switch):
    """Updates the scheduler

    This increments the scheduler, but if we are automatically sweeping
    to find the best initial learning rate for a new set of dendrites
    this function also triggers the network at addition time to
    try the next value.

    Process for finding best initial learning rate for dendrites:
    1. Start at default rate
    2. Learn at that rate until scheduler increments twice
    3. Save that version, start dendrites at LR current increment - 1
    4. Repeat 2 and 3 until version has worse final score at set LR
    5. Load previous model with best accuracy at that LR as initial rate

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model being trained.
    accuracy : float
        The accuracy of the model at the current learning rate.
    epochs_since_cycle_switch : int
        The number of epochs since the last cycle switch.

    Returns
    -------
    int
        The status of restructuring or training completion.
    torch.nn.Module
        The potentially modified neural network model.
    """

    restructured = False
    for param_group in GPA.pai_tracker.member_vars["optimizer_instance"].param_groups:
        learning_rate1 = param_group["lr"]

    if (
        type(GPA.pai_tracker.member_vars["scheduler_instance"])
        is torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        if (
            epochs_since_cycle_switch > GPA.pc.get_initial_history_after_switches()
            or GPA.pai_tracker.member_vars["mode"] == "p"
        ):
            if GPA.pc.get_verbose():
                print(
                    f"Updating scheduler with last improved "
                    f'{GPA.pai_tracker.member_vars["epoch_last_improved"]} '
                    f'from current {GPA.pai_tracker.member_vars["num_epochs_run"]}'
                )
            if GPA.pai_tracker.member_vars["scheduler"] is not None:
                GPA.pai_tracker.member_vars["scheduler_instance"].step(metrics=accuracy)
                if (
                    GPA.pai_tracker.member_vars["scheduler"]
                    is torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if GPA.pc.get_verbose():
                        print(
                            f"Scheduler is now at "
                            f'{GPA.pai_tracker.member_vars["scheduler_instance"].num_bad_epochs} bad epochs'
                        )
        else:
            if GPA.pc.get_verbose():
                print("Not stepping optimizer since hasnt initialized")

    elif GPA.pai_tracker.member_vars["scheduler"] is not None:
        if (
            epochs_since_cycle_switch > GPA.pc.get_initial_history_after_switches()
            or GPA.pai_tracker.member_vars["mode"] == "p"
        ):
            if GPA.pc.get_verbose():
                print(
                    f"Incrementing scheduler to count "
                    f'{GPA.pai_tracker.member_vars["scheduler_instance"]._step_count}'
                )
            GPA.pai_tracker.member_vars["scheduler_instance"].step()
            if (
                GPA.pai_tracker.member_vars["scheduler"]
                is torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if GPA.pc.get_verbose():
                    print(
                        f"Scheduler is now at "
                        f'{GPA.pai_tracker.member_vars["scheduler_instance"].num_bad_epochs} bad epochs'
                    )

    if (
        epochs_since_cycle_switch <= GPA.pc.get_initial_history_after_switches()
        and GPA.pai_tracker.member_vars["mode"] == "n"
    ):
        if GPA.pc.get_verbose():
            print(
                f"Not stepping with history {GPA.pc.get_initial_history_after_switches()} "
                f"and current {epochs_since_cycle_switch}"
            )

    for param_group in GPA.pai_tracker.member_vars["optimizer_instance"].param_groups:
        learning_rate2 = param_group["lr"]

    stepped = False
    at_last_count = False

    if GPA.pc.get_verbose():
        print(
            f"Checking if at last with scores "
            f'{len(GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"])}, '
            f"count since switch {epochs_since_cycle_switch} "
            f"and last total lr step count "
            f'{GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"]}'
        )

    # Check if at double or exactly the test count
    if (
        len(GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"]) == 0
        and epochs_since_cycle_switch
        == GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"] * 2
    ) or (
        len(GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"]) == 1
        and epochs_since_cycle_switch
        == GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"]
    ):
        at_last_count = True

    if GPA.pc.get_verbose():
        print(
            f"At last count {at_last_count} with count {epochs_since_cycle_switch} "
            f'and last LR count {GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"]}'
        )

    if learning_rate1 != learning_rate2:
        stepped = True
        GPA.pai_tracker.member_vars["current_step_count"] += 1

        if GPA.pc.get_verbose():
            print(
                f"Learning rate just stepped to {learning_rate2:.10e} "
                f'with {GPA.pai_tracker.member_vars["current_step_count"]} total steps'
            )

        if (
            GPA.pai_tracker.member_vars["current_step_count"]
            == GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]
        ):
            if GPA.pc.get_verbose():
                print(
                    f'{GPA.pai_tracker.member_vars["current_step_count"]} '
                    f"steps is the max of the last switch mode"
                )
            # Set it when 1->2 gets to 2, not when 0->1 hits 2 as stopping point
            if (
                GPA.pai_tracker.member_vars["current_step_count"]
                - GPA.pai_tracker.member_vars[
                    "current_n_learning_rate_initial_skip_steps"
                ]
                == 1
            ):
                GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"] = (
                    epochs_since_cycle_switch
                )

    if GPA.pc.get_verbose():
        print(
            f"Learning rates were {learning_rate1:.8e} and {learning_rate2:.8e} "
            f'started with {GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}, '
            f'and is now at {GPA.pai_tracker.member_vars["current_step_count"]} '
            f'committed {GPA.pai_tracker.member_vars["committed_to_initial_rate"]} '
            f"then either this (non zero) or eventually comparing to "
            f'{GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
            f'steps or rate {GPA.pai_tracker.member_vars["last_max_learning_rate_value"]:.8f}'
        )

    # If learning rate just stepped, check restart at lower rate
    if (
        (GPA.pai_tracker.member_vars["scheduler"] is not None)
        and
        # If potentially might have higher accuracy
        (
            (GPA.pai_tracker.member_vars["mode"] == "n")
            or GPA.pc.get_learn_dendrites_live()
        )
        and
        # And learning rate just stepped
        (stepped or at_last_count)
    ):

        # If hasn't committed to a learning rate for this cycle yet
        if not GPA.pai_tracker.member_vars["committed_to_initial_rate"]:
            best_score_so_far = GPA.pai_tracker.member_vars[
                "global_best_validation_score"
            ]

            if GPA.pc.get_verbose():
                print(
                    f"In statements to check next learning rate with "
                    f"stepped {stepped} and max count {at_last_count}"
                )

            # If no scores saved for this dendrite and initial LR test did second step
            if len(
                GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"]
            ) == 0 and (
                GPA.pai_tracker.member_vars["current_step_count"]
                - GPA.pai_tracker.member_vars[
                    "current_n_learning_rate_initial_skip_steps"
                ]
                == 2
                or at_last_count
            ):

                restructured = True
                GPA.pai_tracker.clear_optimizer_and_scheduler()

                # Save system for this initial condition
                old_global = GPA.pai_tracker.member_vars["global_best_validation_score"]
                old_accuracy = GPA.pai_tracker.member_vars[
                    "current_best_validation_score"
                ]
                old_counts = GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"]
                skip1 = GPA.pai_tracker.member_vars[
                    "current_n_learning_rate_initial_skip_steps"
                ]

                now = datetime.now()
                dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                GPA.pai_tracker.save_graphs(
                    f'{dt_string}_PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                )

                if GPA.pc.get_test_saves():
                    UPA.save_system(
                        net,
                        GPA.pc.get_save_name(),
                        f'PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                    )

                if GPA.pc.get_verbose():
                    print(
                        f"Saving with initial steps: {dt_string}_PBCount_"
                        f'{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                        f'{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                        f"with current best {old_accuracy}"
                    )

                # Load back at start and try with lower initial learning rate
                net = UPA.load_system(
                    net,
                    GPA.pc.get_save_name(),
                    f'switch_{len(GPA.pai_tracker.member_vars["switch_epochs"])}',
                    switch_call=True,
                )
                GPA.pai_tracker.member_vars[
                    "current_n_learning_rate_initial_skip_steps"
                ] = (skip1 + 1)
                GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"].append(
                    old_accuracy
                )
                GPA.pai_tracker.member_vars["global_best_validation_score"] = old_global
                GPA.pai_tracker.member_vars["initial_lr_test_epoch_count"] = old_counts

            # If there is one score already, this is first step at next score
            elif len(GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"]) == 1:
                GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"].append(
                    GPA.pai_tracker.member_vars["current_best_validation_score"]
                )

                # If this LR's score was worse than last LR's score
                lr_score_worse = False
                if GPA.pai_tracker.member_vars["maximizing_score"]:
                    lr_score_worse = (
                        GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][0]
                        > GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][1]
                    )
                else:
                    lr_score_worse = (
                        GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][0]
                        < GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][1]
                    )

                if lr_score_worse:
                    restructured = True
                    GPA.pai_tracker.clear_optimizer_and_scheduler()

                    if GPA.pc.get_verbose():
                        print(
                            f'Got initial {GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]-1} '
                            f'step score {GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][0]} '
                            f'and {GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                            f'score at step {GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][1]} '
                            f"so loading old score"
                        )

                    prior_best = GPA.pai_tracker.member_vars[
                        "current_cycle_lr_max_scores"
                    ][0]

                    now = datetime.now()
                    dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                    GPA.pai_tracker.save_graphs(
                        f'{dt_string}_PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                    )

                    if GPA.pc.get_test_saves():
                        UPA.save_system(
                            net,
                            GPA.pc.get_save_name(),
                            f'PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                        )

                    if GPA.pc.get_verbose():
                        print(
                            f"Saving with initial steps: {dt_string}_PBCount_"
                            f'{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                            f'{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                        )

                    if GPA.pc.get_test_saves():
                        net = UPA.load_system(
                            net,
                            GPA.pc.get_save_name(),
                            f'PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]-1}',
                            switch_call=True,
                        )

                    # Save graphs for chosen one
                    now = datetime.now()
                    dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                    GPA.pai_tracker.save_graphs(
                        f'{dt_string}_PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}PICKED'
                    )

                    if GPA.pc.get_test_saves():
                        UPA.save_system(
                            net,
                            GPA.pc.get_save_name(),
                            f'PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                        )

                    if GPA.pc.get_verbose():
                        print(
                            f"Saving with initial steps: {dt_string}_PBCount_"
                            f'{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                            f'{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                        )

                    GPA.pai_tracker.member_vars["committed_to_initial_rate"] = True
                    GPA.pai_tracker.member_vars["last_max_learning_rate_steps"] = (
                        GPA.pai_tracker.member_vars["current_step_count"]
                    )
                    GPA.pai_tracker.member_vars["last_max_learning_rate_value"] = (
                        learning_rate2
                    )
                    GPA.pai_tracker.member_vars["current_best_validation_score"] = (
                        prior_best
                    )

                    if GPA.pc.get_verbose():
                        print(
                            f"Setting last max steps to "
                            f'{GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                            f'and lr {GPA.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                        )

                else:  # Current LR score is better
                    if GPA.pc.get_verbose():
                        print(
                            f'Got initial {GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]-1} '
                            f'step score {GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][0]} '
                            f'and {GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                            f'score at step {GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"][1]} '
                            f"so NOT loading old score and continuing with this score"
                        )

                    if at_last_count:  # If this is the last one, set it to be picked
                        restructured = True
                        GPA.pai_tracker.clear_optimizer_and_scheduler()

                        now = datetime.now()
                        dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

                        GPA.pai_tracker.save_graphs(
                            f'{dt_string}_PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}PICKED'
                        )

                        if GPA.pc.get_test_saves():
                            UPA.save_system(
                                net,
                                GPA.pc.get_save_name(),
                                f'PBCount_{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}',
                            )

                        if GPA.pc.get_verbose():
                            print(
                                f"Saving with initial steps: {dt_string}_PBCount_"
                                f'{GPA.pai_tracker.member_vars["num_dendrites_added"]}_startSteps_'
                                f'{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}'
                            )

                        GPA.pai_tracker.member_vars["committed_to_initial_rate"] = True
                        GPA.pai_tracker.member_vars["last_max_learning_rate_steps"] = (
                            GPA.pai_tracker.member_vars["current_step_count"]
                        )
                        GPA.pai_tracker.member_vars["last_max_learning_rate_value"] = (
                            learning_rate2
                        )

                        if GPA.pc.get_verbose():
                            print(
                                f"Setting last max steps to "
                                f'{GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                                f'and lr {GPA.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                            )

                GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"] = []

            elif len(GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"]) == 2:
                print(
                    "Should never be here. Please let Perforated AI know if this happened."
                )
                pdb.set_trace()

            GPA.pai_tracker.member_vars["global_best_validation_score"] = (
                best_score_so_far
            )

        else:
            if GPA.pc.get_verbose():
                print(
                    f"Setting last max steps to "
                    f'{GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]} '
                    f'and lr {GPA.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                )
            GPA.pai_tracker.member_vars["last_max_learning_rate_steps"] += 1
            GPA.pai_tracker.member_vars["last_max_learning_rate_value"] = learning_rate2
    if restructured:
        return NETWORK_RESTRUCTURED, net
    else:
        return NO_MODEL_UPDATE, net


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
        """Initialize the tracker

        Parameters
        ----------
        doing_pai : bool
            Whether or not dendrites should be used.
        save_name : str
            The base name for saving models and graphs.
        making_graphs : bool, optional
            Whether or not to generate graphs, by default True.
        param_vals_setting : int, optional
            Parameter values setting, by default -1.
        values_per_train_epoch : int, optional
            The number of values to look back for graphing
            during training, by default -1 (all values).
        values_per_val_epoch : int, optional
            The number of values to look back for graphing
            during validation, by default -1 (all values).
        Returns
        -------
        None
        """

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
        self.member_vars["switch_mode"] = GPA.pc.get_switch_mode()
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
        self.member_vars["param_vals_setting"] = GPA.pc.get_param_vals_setting()
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

        # flag for 
        self.member_vars["step_status"] = STEP_CLEARED
        self.member_var_types["step_status"] = "int"


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
        self.member_vars["best_mean_score_improved_this_epoch"] = 0
        self.member_var_types["best_mean_score_improved_this_epoch"] = "int"

        # Flag for if current dendrite achieved highest global score
        self.member_vars["current_n_set_global_best"] = True
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

        if GPA.pc.get_verbose():
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
                pdb.set_trace()
        return full_string

    def from_string(self, string):
        """Load tracker values from string.

        Parameters
        ----------
        string : str
            The string to load from.
        """
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

                pdb.set_trace()

    def from_string_debug(self, string):
        """Debug function to print tracker values from string without loading them.

        Parameters
        ----------
        string : str
            The string to debug load from.
        """
        f = io.StringIO(string)
        print("=== DEBUGGING TRACKER VARIABLES ===")

        while True:
            line = f.readline()
            if not line:
                break
            vals = line.split(",")
            var = vals[0]

            print(f"\nVariable: {var}")
            print(f"Type: {self.member_var_types.get(var, 'UNKNOWN TYPE')}")
            print(f"Current value: {self.member_vars.get(var, 'NOT SET')}")

            if self.member_var_types.get(var) == "bool":
                val = vals[1][:-1]
                print(f"Would set to: {val} -> {val == 'True'}")

            elif self.member_var_types.get(var) == "int":
                val = vals[1]
                print(f"Would set to: {int(val)}")

            elif self.member_var_types.get(var) == "float":
                val = vals[1]
                print(f"Would set to: {float(val)}")

            elif self.member_var_types.get(var) == "string":
                val = vals[1][:-1]
                print(f"Would set to: '{val}'")

            elif self.member_var_types.get(var) == "type":
                print("Would skip (type loading)")

            elif self.member_var_types.get(var) == "empty array":
                val = vals[1]
                print(f"Would set to: [] (empty array)")

            elif self.member_var_types.get(var) == "int array":
                vals_line = f.readline()[:-1].split(",")
                print(f"Would set to int array with {len(vals_line)} elements:")
                if vals_line[0] != "":
                    print(
                        f"  Elements: {vals_line[:5]}{'...' if len(vals_line) > 5 else ''}"
                    )
                else:
                    print("  Empty array")

            elif self.member_var_types.get(var) == "float array":
                vals_line = f.readline()[:-1].split(",")
                print(f"Would set to float array with {len(vals_line)} elements:")
                if vals_line[0] != "":
                    print(
                        f"  Elements: {vals_line[:5]}{'...' if len(vals_line) > 5 else ''}"
                    )
                else:
                    print("  Empty array")

            elif self.member_var_types.get(var) == "float array dictionary array":
                print("Would process float array dictionary array:")
                array_count = 0
                line2 = f.readline()[:-1]
                while line2 != "endarray":
                    key_count = 0
                    while line2 != "endkey":
                        vals_dict = line2.split(",")
                        name = vals_dict[0]
                        print(
                            f"  Array {array_count}, Key '{name}': {len(vals_dict)-1} elements"
                        )
                        key_count += 1
                        line2 = f.readline()[:-1]
                    print(f"  Array {array_count} has {key_count} keys")
                    array_count += 1
                    line2 = f.readline()[:-1]
                print(f"  Total arrays: {array_count}")

            elif self.member_var_types.get(var) == "float array dictionary":
                print("Would process float array dictionary:")
                line2 = f.readline()[:-1]
                key_count = 0
                while line2 != "end":
                    vals_dict = line2.split(",")
                    name = vals_dict[0]
                    print(f"  Key '{name}': {len(vals_dict)-1} elements")
                    key_count += 1
                    line2 = f.readline()[:-1]
                print(f"  Total keys: {key_count}")

            elif self.member_var_types.get(var) == "float array array":
                print("Would process float array array:")
                line2 = f.readline()[:-1]
                array_count = 0
                while line2 != "end":
                    if line2:
                        vals_array = line2.split(",")
                        print(f"  Array {array_count}: {len(vals_array)} elements")
                    else:
                        print(f"  Array {array_count}: empty")
                    array_count += 1
                    line2 = f.readline()[:-1]
                print(f"  Total arrays: {array_count}")

            else:
                print(f"UNKNOWN TYPE: {self.member_var_types.get(var, 'NOT FOUND')}")

        print("\n=== END DEBUG ===")

    def save_tracker_settings(self):
        """Save tracker settings for DistributedDataParallel use.

        Saves settings in save_name/array_dims.csv

        Parameters
        ----------
        None
        Returns
        -------
        None

        -----
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
        if not GPA.pc.get_silent():
            print("Tracker settings saved.")
            print("You may now delete save_tracker_settings")

    def initialize_tracker_settings(self):
        """Initialize tracker settings from saved file.

        This function loads tracker settings from a CSV file and applies them
        to the layers the tracker is managing.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        channels = {}
        if not os.path.exists(self.save_name + "/array_dims.csv"):
            print(
                "You must call save_tracker_settings before "
                "initialize_tracker_settings"
            )
            print("Follow instructions in customization.md")
            pdb.set_trace()
        f = open(self.save_name + "/array_dims.csv", "r")
        for line in f:
            channels[line.split(",")[0]] = int(line.split(",")[1])
        for layer in self.neuron_module_vector:
            layer.dendrite_module.dendrite_values[0].setup_arrays(channels[layer.name])

    def set_optimizer_instance(self, optimizer_instance):
        """Set optimizer instance directly.

        Parameters
        ----------
        optimizer_instance : object
            The optimizer instance to set.

        Returns
        -------
        None

        """

        try:
            for param_group in optimizer_instance.param_groups:
                if (
                    param_group["weight_decay"] > 0
                    and GPA.pc.get_weight_decay_accepted() is False
                ):
                    print(
                        "For PAI training it is recommended to not use "
                        "weight decay in your optimizer"
                    )
                    print(
                        "Set GPA.pc.set_weight_decay_accepted(True) to ignore this "
                        "warning or c to continue"
                    )
                    GPA.pc.set_weight_decay_accepted(True)
                    pdb.set_trace()
        except:
            pass
        self.member_vars["optimizer_instance"] = optimizer_instance
        if GPA.pc.get_perforated_backpropagation():
            TPB.setup_optimizer_pb(self.member_vars["optimizer_instance"])

    def set_optimizer(self, optimizer):
        """Set optimizer type to be initialized later

        Parameters
        ----------
        optimizer : object
            The optimizer type to set.

        Returns
        -------
        None

        """
        self.member_vars["optimizer"] = optimizer

    def set_scheduler(self, scheduler):
        """Set scheduler type to be initialized later

        Parameters
        ----------
        scheduler : object
            The scheduler type to set.

        Returns
        -------
        None

        """
        if scheduler is not torch.optim.lr_scheduler.ReduceLROnPlateau:
            if GPA.pc.get_verbose():
                print("Not using ReduceLROnPlateau, this is not recommended")
        self.member_vars["scheduler"] = scheduler

    def increment_scheduler(self, num_ticks, mode):
        """Increment the scheduler a set number of times.

        Used for finding best initial learning rate when adding dendrites.

        Parameters
        ----------
        num_ticks : int
            The number of scheduler steps to take.
        mode : str
            The mode for stepping the scheduler. Options are:
            - "step_learning_rate": Step based on improved accuracy epochs
            - "increment_epoch_count": Step based on total epoch count

        Returns
        -------
        current_steps : int
            The number of learning rate changes that occurred.
        learning_rate1 : float
            The final learning rate after stepping.

        """

        current_steps = 0
        current_ticker = 0

        for param_group in GPA.pai_tracker.member_vars[
            "optimizer_instance"
        ].param_groups:
            learning_rate1 = param_group["lr"]

        if GPA.pc.get_verbose():
            print("Using scheduler:")
            print(type(self.member_vars["scheduler_instance"]))

        while current_ticker < num_ticks:
            if GPA.pc.get_verbose():
                print(
                    f"Lower start rate initial {learning_rate1} "
                    f'stepping {GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]} times'
                )

            if (
                type(self.member_vars["scheduler_instance"])
                is torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if mode == "step_learning_rate":
                    # Step with counter as last improved accuracy
                    self.member_vars["scheduler_instance"].step(
                        metrics=self.member_vars["last_improved_accuracies"][
                            GPA.pai_tracker.steps_after_switch() - 1
                        ]
                    )
                elif mode == "increment_epoch_count":
                    # Step with improved epoch counts up to current location
                    self.member_vars["scheduler_instance"].step(
                        metrics=self.member_vars["last_improved_accuracies"][
                            -((num_ticks - 1) - current_ticker) - 1
                        ]
                    )
            else:
                self.member_vars["scheduler_instance"].step()

            for param_group in GPA.pai_tracker.member_vars[
                "optimizer_instance"
            ].param_groups:
                learning_rate2 = param_group["lr"]

            if learning_rate2 != learning_rate1:
                current_steps += 1
                learning_rate1 = learning_rate2
                if mode == "step_learning_rate":
                    current_ticker += 1
                if GPA.pc.get_verbose():
                    print(f"1 step {current_steps} to {learning_rate2}")

            if mode == "increment_epoch_count":
                current_ticker += 1

        return current_steps, learning_rate1

    def setup_optimizer(self, net, opt_args, sched_args=None):
        """Initialize the optimizer and scheduler when added.

        Parameters
        ----------
        net : object
            The neural network model.
        opt_args : dict
            The arguments for the optimizer.
        sched_args : dict, optional
            The arguments for the scheduler, by default None.

        Returns
        -------
        optimizer : object
            The initialized optimizer instance.
        scheduler : object, optional
            The initialized scheduler instance, if a scheduler was set.

        """
        if "weight_decay" in opt_args and not GPA.pc.get_weight_decay_accepted():
            print(
                "For PAI training it is recommended to not use "
                "weight decay in your optimizer"
            )
            print(
                "Set GPA.pc.set_weight_decay_accepted(True) to ignore this "
                "warning or c to continue"
            )
            GPA.pc.set_weight_decay_accepted(True)
            pdb.set_trace()

        if "model" not in opt_args.keys():
            if self.member_vars["mode"] == "n":
                opt_args["params"] = filter(lambda p: p.requires_grad, net.parameters())
            else:
                opt_args["params"] = UPA.get_pai_network_params(net)

        optimizer = self.member_vars["optimizer"](**opt_args)
        self.set_optimizer_instance(optimizer)

        if self.member_vars["scheduler"] is not None:
            self.member_vars["scheduler_instance"] = self.member_vars["scheduler"](
                optimizer, **sched_args
            )
            current_steps = 0

            for param_group in GPA.pai_tracker.member_vars[
                "optimizer_instance"
            ].param_groups:
                learning_rate1 = param_group["lr"]

            if GPA.pc.get_verbose():
                print(
                    f"Resetting scheduler with {GPA.pai_tracker.steps_after_switch()} "
                    f"steps and {GPA.pc.get_initial_history_after_switches()} initial ticks to skip"
                )

            # Find setting of previously used learning rate before adding dendrites
            if (
                GPA.pai_tracker.member_vars[
                    "current_n_learning_rate_initial_skip_steps"
                ]
                != 0
            ):
                additional_steps, learning_rate1 = self.increment_scheduler(
                    GPA.pai_tracker.member_vars[
                        "current_n_learning_rate_initial_skip_steps"
                    ],
                    "step_learning_rate",
                )
                current_steps += additional_steps

            if self.member_vars["mode"] == "n" or GPA.pc.get_learn_dendrites_live():
                initial = GPA.pc.get_initial_history_after_switches()
            else:
                initial = 0

            if GPA.pai_tracker.steps_after_switch() > initial:
                # Minus extra 1 because this gets called after start epoch
                additional_steps, learning_rate1 = self.increment_scheduler(
                    (GPA.pai_tracker.steps_after_switch() - initial) - 1,
                    "increment_epoch_count",
                )
                current_steps += additional_steps

            if GPA.pc.get_verbose():
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
        """Determine if it's time to switch between neuron and dendrite training.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if it's time to switch, False otherwise.

        Notes
        -----
        Based on current settings and history of scores.
        """

        switch_phrase = "No mode, this should never be the case."
        switch_number = GPA.pc.get_n_epochs_to_switch()
        if self.member_vars["switch_mode"] == GPA.pc.DOING_SWITCH_EVERY_TIME:
            switch_phrase = "DOING_SWITCH_EVERY_TIME"
        elif self.member_vars["switch_mode"] == GPA.pc.DOING_HISTORY:
            switch_phrase = "DOING_HISTORY"
        elif self.member_vars["switch_mode"] == GPA.pc.DOING_FIXED_SWITCH:
            switch_phrase = "DOING_FIXED_SWITCH"
            switch_number = GPA.pc.get_fixed_switch_num()
        elif self.member_vars["switch_mode"] == GPA.pc.DOING_NO_SWITCH:
            switch_phrase = "DOING_NO_SWITCH"
        else:
            print(
                "A switch mode must be set.  Check your settings for GPA.pc.set_switch_mode()."
            )
            pdb.set_trace()
        if not GPA.pc.get_silent():

            print(
                f'Checking PAI switch with mode {self.member_vars["mode"]}, '
                f'switch mode {switch_phrase}, epoch {self.member_vars["num_epochs_run"]}, '
                f'last improved epoch {self.member_vars["epoch_last_improved"]}, '
                f'total epochs {self.member_vars["total_epochs_run"]}, '
                f'n: {switch_number}, num_cycles: {self.member_vars["num_cycles"]}'
            )
        if GPA.pc.get_perforated_backpropagation():
            # this will fill in epoch last improved
            TPB.best_pai_score_improved_this_epoch(self)  ## CLOSED ONLY
        if self.member_vars["switch_mode"] == GPA.pc.DOING_NO_SWITCH:
            if not GPA.pc.get_silent():
                print("Returning False - doing no switch mode")
            return False

        if self.member_vars["switch_mode"] == GPA.pc.DOING_SWITCH_EVERY_TIME:
            if not GPA.pc.get_silent():
                print("Returning True - switching every time")
            return True

        if (
            ((self.member_vars["mode"] == "n") or GPA.pc.get_learn_dendrites_live())
            and (self.member_vars["switch_mode"] == GPA.pc.DOING_HISTORY)
            and (GPA.pai_tracker.member_vars["committed_to_initial_rate"] is False)
            and (GPA.pc.get_dont_give_up_unless_learning_rate_lowered())
            and (
                self.member_vars["current_n_learning_rate_initial_skip_steps"]
                < self.member_vars["last_max_learning_rate_steps"]
            )
            and self.member_vars["scheduler"] is not None
        ):
            if not GPA.pc.get_silent():
                print(
                    f"Returning False since no first step yet and comparing "
                    f'initial {self.member_vars["current_n_learning_rate_initial_skip_steps"]} '
                    f'to last max {self.member_vars["last_max_learning_rate_steps"]}'
                )
            return False

        if len(self.member_vars["switch_epochs"]) == 0:
            this_count = self.member_vars["num_epochs_run"]
        else:
            this_count = (
                self.member_vars["num_epochs_run"]
                - self.member_vars["switch_epochs"][-1]
            )
        cap_switch = False
        if GPA.pc.get_perforated_backpropagation():
            cap_switch = TPB.check_cap_switch(self, this_count)

        if self.member_vars["switch_mode"] == GPA.pc.DOING_HISTORY and (
            (
                (self.member_vars["mode"] == "n")
                and (
                    self.member_vars["num_epochs_run"]
                    - self.member_vars["epoch_last_improved"]
                    >= GPA.pc.get_n_epochs_to_switch()
                )
                and this_count
                >= GPA.pc.get_initial_history_after_switches()
                + GPA.pc.get_n_epochs_to_switch()
            )
            or (GPA.pc.get_perforated_backpropagation() and TPB.history_switch(self))
            or cap_switch
        ):
            if not GPA.pc.get_silent():
                print("Returning True - History and last improved is hit")
            return True

        if self.member_vars["switch_mode"] == GPA.pc.DOING_FIXED_SWITCH and (
            (
                self.member_vars["total_epochs_run"] % GPA.pc.get_fixed_switch_num()
                == GPA.pc.get_fixed_switch_num() - 1
            )
            and self.member_vars["num_epochs_run"]
            >= GPA.pc.get_first_fixed_switch_num() - 1
        ):
            if not GPA.pc.get_silent():
                print("Returning True - Fixed switch number is hit")
            return True

        if not GPA.pc.get_silent():
            print("Returning False - no triggers to switch have been hit")
        return False

    def steps_after_switch(self):
        """Based on settings, return value for steps since a switch.

        Different options for param vals setting determine what is returned.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of epochs since the last switch, or total epochs run,
            depending on settings.

        """
        if self.member_vars["param_vals_setting"] == GPA.pc.PARAM_VALS_BY_TOTAL_EPOCH:
            return self.member_vars["num_epochs_run"]
        elif (
            self.member_vars["param_vals_setting"] == GPA.pc.PARAM_VALS_BY_UPDATE_EPOCH
        ):
            return self.member_vars["num_epochs_run"] - self.member_vars["last_switch"]
        elif (
            self.member_vars["param_vals_setting"]
            == GPA.pc.PARAM_VALS_BY_NEURON_EPOCH_START
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
            pdb.set_trace()

    def add_pai_neuron_module(self, new_module, initial_add=True):
        """Add neuron modules to internal vectors.

        Parameters
        ----------
        new_module : object
            The new module to add.
        initial_add : bool, optional
            Whether this is the initial addition rather than loading from file

        Returns
        -------
        None

        """

        # If it's a duplicate, ignore the second addition
        if new_module in self.neuron_module_vector:
            return
        self.neuron_module_vector.append(new_module)
        if self.member_vars["doing_pai"]:
            PA.set_wrapped_params(new_module)
        if initial_add:
            self.member_vars["best_scores"].append([])
            self.member_vars["current_scores"].append([])

    def add_tracked_neuron_module(self, new_module, initial_add=True):
        """Add tracked modules to internal vectors

        Parameters
        ----------
        new_module : object
            The new module to add.
        initial_add : bool, optional
            Whether this is the initial addition rather than loading from file

        Returns
        -------
        None

        """
        # If it's a duplicate, ignore the second addition
        if new_module in self.tracked_neuron_module_vector:
            return
        self.tracked_neuron_module_vector.append(new_module)
        if self.member_vars["doing_pai"]:
            PA.set_tracked_params(new_module)

    def reset_module_vector(self, net, load_from_restart):
        """Clear internal vectors and reset from network.

        Parameters
        ----------
        net : object
            The neural network model.
        load_from_restart : bool
            Whether loading from a restart file.

        Returns
        -------
        None

        """
        self.neuron_module_vector = []
        self.tracked_neuron_module_vector = []
        this_list = UPA.get_pai_modules(net, 0)
        for module in this_list:
            self.add_pai_neuron_module(module, initial_add=load_from_restart)
        this_list = UPA.get_tracked_modules(net, 0)
        for module in this_list:
            self.add_tracked_neuron_module(module, initial_add=load_from_restart)

    def reset_vals_for_score_reset(self):
        """Reset cycle scores for new cycle."""

        if GPA.pc.get_find_best_lr():
            self.member_vars["committed_to_initial_rate"] = False
        # If retaining all dendrties always say that the current dendrites set global best for saving and loading
        if GPA.pc.get_retain_all_dendrites():
            self.member_vars["current_n_set_global_best"] = True
            self.member_vars["global_best_validation_score"] = 0
        else:
            self.member_vars["current_n_set_global_best"] = False

        # Don't reset global best, but do reset current best
        self.member_vars["current_best_validation_score"] = 0
        self.member_vars["initial_lr_test_epoch_count"] = -1

    def set_dendrite_training(self):
        """Signal all layers to start dendrite training."""
        if GPA.pc.get_verbose():
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

        if GPA.pc.get_learn_dendrites_live():
            self.reset_vals_for_score_reset()

        self.member_vars["last_max_learning_rate_steps"] = self.member_vars[
            "current_step_count"
        ]

        GPA.pai_tracker.member_vars["current_cycle_lr_max_scores"] = []
        GPA.pai_tracker.member_vars["num_cycles"] += 1

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
        if GPA.pc.get_learn_dendrites_live():
            self.member_vars["last_max_learning_rate_steps"] = self.member_vars[
                "current_step_count"
            ]
        GPA.pai_tracker.member_vars["num_cycles"] += 1

        if GPA.pc.get_reset_best_score_on_switch():
            GPA.pai_tracker.member_vars["current_best_validation_score"] = 0
            GPA.pai_tracker.member_vars["running_accuracy"] = 0

    def start_epoch(self, internal_call=False):
        """Perform steps for when a new training epoch is about to begin.

        Parameters
        ----------
        internal_call : bool, optional
            Whether this is an internal call or manual call

        Returns
        -------
        None

        Notes
        -----
        If you ever need to call this manually, set internal_call to False.

        """
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
                for m in range(0, GPA.pc.get_global_candidates()):
                    with torch.no_grad():
                        if GPA.pc.get_verbose():
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
            if GPA.pc.get_perforated_backpropagation():
                self.member_vars["best_mean_score_improved_this_epoch"] = 0
        self.member_vars["num_epochs_run"] += 1
        self.member_vars["total_epochs_run"] = (
            self.member_vars["num_epochs_run"] + self.member_vars["overwritten_epochs"]
        )
        self.saved_time = end

    def stop_epoch(self, internal_call=False):
        """Perform steps when a training epoch has completed.

        Parameters
        ----------
        internal_call : bool, optional
            Whether this is an internal call or manual call

        Returns
        -------
        None

        Notes
        -----
        If you ever need to call this manually, set internal_call to False.

        """
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
        save_name="PAI",
        making_graphs=True,
        maximizing_score=True,
        num_classes=10000,
        values_per_train_epoch=-1,
        values_per_val_epoch=-1,
        zooming_graph=True,
    ):
        """Setup the tracker with initial settings.


        Parameters
        ----------
        model : object
            The neural network model.
        doing_pai : bool, optional
            Whether to add dendrites, by default True.
        save_name : str, optional
            The name under which to save the model.
        making_graphs : bool, optional
            Whether to make graphs, by default True.
        maximizing_score : bool, optional
            Whether to maximize the score, by default True.
        num_classes : int, optional
            The number of classes in the dataset, unused
        values_per_train_epoch : int, optional
            The number of values to look back for graphing
            during training, by default -1 (all values).
        values_per_val_epoch : int, optional
            The number of values to look back for graphing
            during validation, by default -1 (all values).
        zooming_graph : bool, optional
            Whether to zoom on graphs, by default True.

        """
        model = UPA.convert_network(model)
        self.member_vars["doing_pai"] = doing_pai
        self.member_vars["maximizing_score"] = maximizing_score
        self.save_name = save_name
        self.zooming_graph = zooming_graph
        self.making_graphs = making_graphs

        if not self.loaded:
            self.member_vars["running_accuracy"] = (1.0 / num_classes) * 100

        self.values_per_train_epoch = values_per_train_epoch
        self.values_per_val_epoch = values_per_val_epoch

        if GPA.pc.get_testing_dendrite_capacity():
            if not GPA.pc.get_silent():
                print("Running a test of Dendrite Capacity.")
            GPA.pc.set_switch_mode(GPA.pc.DOING_SWITCH_EVERY_TIME)
            self.member_vars["switch_mode"] = GPA.pc.get_switch_mode()
            GPA.pc.set_retain_all_dendrites(True)
            GPA.pc.set_max_dendrite_tries(1000)
            GPA.pc.set_max_dendrites(1000)
            if GPA.pc.get_perforated_backpropagation():
                GPA.pc.set_initial_correlation_batches(1)
        else:
            if not GPA.pc.get_silent():
                print("Running Dendrite Experiment")
        return model

    def generate_accuracy_plots(self, ax, save_folder, extra_string):
        """
        Generate plots and csvs for accuracy

        Parameters
        ----------
        ax : object
            The matplotlib axis to plot on.
        save_folder : str
            The folder to save the plots and csvs in.
        extra_string : str
            An extra string to append to the filenames.

        Returns
        -------
        None

        """

        # If scores are being saved for epochs that get overwritten, plot them
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

        # Determine which accuracy vector to use
        if GPA.pc.get_drawing_pai():
            accuracies = self.member_vars["accuracies"]
        else:
            accuracies = self.member_vars["n_accuracies"]

        # Get pointer to additional scores being saved
        extra_scores = self.member_vars["extra_scores"]

        # Plot the main accuracy scores
        ax.plot(np.arange(len(accuracies)), accuracies, label="Validation Scores")
        ax.plot(
            np.arange(len(self.member_vars["running_accuracies"])),
            self.member_vars["running_accuracies"],
            label="Validation Running Scores",
        )

        # Plot additional scores
        for extra_score in extra_scores:
            ax.plot(
                np.arange(len(extra_scores[extra_score])),
                extra_scores[extra_score],
                label=extra_score,
            )

        plt.title(save_folder + "/" + self.save_name + "Scores")
        plt.xlabel("Epochs")
        plt.ylabel("Score")

        # Add point at epoch last improved and best validation score
        if GPA.pc.get_drawing_pai():
            ax.plot(
                self.member_vars["epoch_last_improved"],
                self.member_vars["global_best_validation_score"],
                "bo",
                label="Global best (y)",
            )
            ax.plot(
                self.member_vars["epoch_last_improved"],
                accuracies[self.member_vars["epoch_last_improved"]],
                "go",
                label="Epoch Last Improved",
            )
        else:
            if self.member_vars["mode"] == "n":
                missed_time = (
                    self.member_vars["num_epochs_run"]
                    - self.member_vars["epoch_last_improved"]
                )
                ax.plot(
                    (len(self.member_vars["n_accuracies"]) - 1) - missed_time,
                    self.member_vars["n_accuracies"][-(missed_time + 1)],
                    "go",
                    label="Epoch Last Improved",
                )

        # Generate csv file for the values graphed
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
        del pd1, pd2

        # Set y min and max to zoom in on important part of axis
        if (
            len(self.member_vars["switch_epochs"]) > 0
            and self.member_vars["switch_epochs"][0] > 0
            and self.zooming_graph
        ):
            if GPA.pai_tracker.member_vars["maximizing_score"]:
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

        # Draw vertical lines for epochs where a dendrite switch occurred
        if GPA.pc.get_drawing_pai() and self.member_vars["doing_pai"]:
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

    def generate_time_plots(self, ax, save_folder, extra_string):
        """
        Generate plots and csvs for timing

        Parameters
        ----------
        ax : object
            The matplotlib axis to plot on.
        save_folder : str
            The folder to save the plots and csvs in.
        extra_string : str
            An extra string to append to the filenames.

        Returns
        -------
        None

        """
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

    def generate_learning_rate_plots(self, ax, save_folder, extra_string):
        """
        Generate plots and csvs for learning rate

        Parameters
        ----------
        ax : object
            The matplotlib axis to plot on.
        save_folder : str
            The folder to save the plots and csvs in.
        extra_string : str
            An extra string to append to the filenames.

        Returns
        -------
        None

        """
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
        del pd1

    def generate_dendrite_learning_plots(self, ax, save_folder, extra_string):
        """
        Generate dendrite score plots for the tracker.
        Also saves csv files associated with the plots.
        """
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
                        label=f"Current:{self.neuron_module_vector[layer_id].name}",
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
                ncol=max(1, math.ceil(len(self.neuron_module_vector) / 30)),
            )
            for switcher in self.member_vars["p_switch_epochs"]:
                plt.axvline(x=switcher, ymin=0, ymax=1, color="r")

            if self.member_vars["mode"] == "p":
                missed_time = (
                    self.member_vars["num_epochs_run"]
                    - self.member_vars["epoch_last_improved"]
                )
                plt.axvline(
                    x=(len(self.member_vars["best_scores"][0]) - (missed_time + 1)),
                    ymin=0,
                    ymax=1,
                    color="g",
                )

            # pd1 here will be none if no PB layers are created
            if pd1 is not None:
                pd1.to_csv(
                    save_folder
                    + "/"
                    + self.save_name
                    + extra_string
                    + "Best PBScores.csv",
                    index=False,
                )
            del pd1, pd2

    def generate_extra_csv_files(self, save_folder, extra_string):
        """
        Generate additional csvs

        Parameters
        ----------
        save_folder : str
            The folder to save the plots and csvs in.
        extra_string : str
            An extra string to append to the filenames.

        Returns
        -------
        None

        """
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
        del pd1

        """
        Create best_test_scores.csv file
        When working with dendrites there is a tradeoff between additional param count and test score improvement.
        This file will help track that tradeoff by recording the best test scores for architecture version.
        The test score that gets recorded here is not the best test score calculated,
        instead it is the test score that was calculated during the epoch when the best validation score was found.
        """
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

            if GPA.pai_tracker.member_vars["maximizing_score"]:
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
            if self.member_vars["doing_pai"]:
                associated_params.append(self.member_vars["param_counts"][switch])
            else:
                associated_params.append(self.member_vars["param_counts"][-1])

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

            if GPA.pai_tracker.member_vars["maximizing_score"]:
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
        del pd1

    def save_graphs(self, extra_string=""):
        """
        Save graphs and csvs for all the values the tracker records

        Parameters
        ----------
        extra_string : str
            An extra string to append to the filenames.

        Returns
        -------
        None

        """
        if not self.making_graphs:
            return

        save_folder = "./" + self.save_name + "/"

        plt.ioff()
        fig = plt.figure(figsize=(28, 14))

        # Plot with accuracy scores
        ax = plt.subplot(221)
        self.generate_accuracy_plots(ax, save_folder, extra_string)

        # Plot dendrite learning scores
        ax = plt.subplot(222)
        self.generate_dendrite_learning_plots(ax, save_folder, extra_string)

        if GPA.pc.get_drawing_extra_graphs():
            # Plot learning rates for each training epoch
            ax = plt.subplot(223)
            self.generate_learning_rate_plots(ax, save_folder, extra_string)

            # Plot the times for each training epoch
            ax = plt.subplot(224)
            self.generate_time_plots(ax, save_folder, extra_string)

        # Generate extra CSV files
        self.generate_extra_csv_files(save_folder, extra_string)

        fig.tight_layout()
        plt.savefig(save_folder + "/" + self.save_name + extra_string + ".png")
        plt.close("all")

    def add_loss(self, loss):
        """Add loss to tracking vectors.

        Parameters
        ----------
        loss : float or int
            The loss value to add.

        Returns
        -------
        None

        """
        if not isinstance(loss, (float, int)):
            loss = loss.item()
        self.member_vars["training_loss"].append(loss)

    def add_learning_rate(self, learning_rate):
        """Add learning rate to tracking vectors.

        Parameters
        ----------
        learning_rate : float or int
            The learning rate value to add.

        Returns
        -------
        None

        """
        if not isinstance(learning_rate, (float, int)):
            learning_rate = learning_rate.item()
        self.member_vars["training_learning_rates"].append(learning_rate)

    def add_extra_score(self, score, extra_score_name):
        """Add extra score to tracking vectors.

        Parameters
        ----------
        score : float or int
            The score value to add.

        extra_score_name : str
            The name of the extra score.

        Returns
        -------
        None

        """
        if not isinstance(score, (float, int)):
            try:
                score = score.item()
            except:
                print(
                    "Scores added for Perforated Backpropagation should be "
                    "float, int, or tensor, yours is a:"
                )
                print(type(score))
                pdb.set_trace()

        if GPA.pc.get_verbose():
            print(f"Adding extra score {extra_score_name} of {float(score)}")

        if extra_score_name not in self.member_vars["extra_scores"]:
            self.member_vars["extra_scores"][extra_score_name] = []
        self.member_vars["extra_scores"][extra_score_name].append(score)

        if self.member_vars["mode"] == "n":
            if extra_score_name not in self.member_vars["n_extra_scores"]:
                self.member_vars["n_extra_scores"][extra_score_name] = []
            self.member_vars["n_extra_scores"][extra_score_name].append(score)

    def add_extra_score_without_graphing(self, score, extra_score_name):
        """Add extra score without graphing to tracking vectors.

        Parameters
        ----------
        score : float or int
            The score value to add.

        extra_score_name : str
            The name of the extra score.

        Returns
        -------
        None

        """
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
                pdb.set_trace()

        if GPA.pc.get_verbose():
            print(f"Adding extra score {extra_score_name} of {float(score)}")

        if extra_score_name not in self.member_vars["extra_scores_without_graphing"]:
            self.member_vars["extra_scores_without_graphing"][extra_score_name] = []
        self.member_vars["extra_scores_without_graphing"][extra_score_name].append(
            score
        )

    def add_test_score(self, score, extra_score_name):
        """Add test score to tracking vectors.

        Parameters
        ----------
        score : float or int
            The score value to add.

        extra_score_name : str
            The name of the extra score.

        Returns
        -------
        None

        Notes
        -----
        This function is a wrapper around `add_extra_score` that separates
        test score for adding to best_test_scores.csv.

        """
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
                pdb.set_trace()

        if GPA.pc.get_verbose():
            print(f"Adding test score {extra_score_name} of {float(score)}")
        self.member_vars["test_scores"].append(score)

    def add_validation_score(self, accuracy, net, force_switch=False):
        """Function to add the validation score.

        This is complex because it determines neuron and dendrite switching.

        Parameters
        ----------
        accuracy : float or int
            The accuracy or loss value to add.
        net : object
            The neural network model.
        force_switch : bool, optional
            Whether to force a switch, by default False.

        Returns
        -------
        net : object
            The potentially modified neural network model.
        training_complete : bool
            Whether training is complete.
        restructured : bool
            Whether the model has been restructured.

        Notes
        -----
        WARNING: Do not call self anywhere in this function. When systems
        get loaded the actual tracker you are working with can change.
        """

        if not GPA.pc.get_silent():
            print(f"Adding validation score {accuracy:.8f}")

        update_learning_rate()
        update_param_count(net)

        accuracy = check_input_problems(net, accuracy)

        if len(GPA.pai_tracker.member_vars["switch_epochs"]) == 0:
            epochs_since_cycle_switch = GPA.pai_tracker.member_vars["num_epochs_run"]
        else:
            epochs_since_cycle_switch = (
                GPA.pai_tracker.member_vars["num_epochs_run"]
                - GPA.pai_tracker.member_vars["switch_epochs"][-1]
            )

        update_running_accuracy(accuracy, epochs_since_cycle_switch)
        if GPA.pc.get_perforated_backpropagation():
            TPB.update_pb_scores(self)

        GPA.pai_tracker.stop_epoch(internal_call=True)

        # If it is neuron training mode
        if (
            GPA.pai_tracker.member_vars["mode"] == "n"
            or GPA.pc.get_learn_dendrites_live()
        ):
            check_new_best(net, accuracy, epochs_since_cycle_switch)
        elif GPA.pc.get_perforated_backpropagation():
            TPB.check_best_pai_score_improvement()

        # Save the latest model
        if GPA.pc.get_test_saves():
            UPA.save_system(net, GPA.pc.get_save_name(), "latest")
        if GPA.pc.get_pai_saves():
            UPA.pai_save_system(net, GPA.pc.get_save_name(), "latest")

        restructuring_status_value = NO_MODEL_UPDATE
        # If it is time to switch based on scores and counter or a manual switch
        if GPA.pai_tracker.switch_time() or force_switch:
            # If testing dendrite capacity switch after enough dendrites added
            if (
                (GPA.pai_tracker.member_vars["mode"] == "n")
                and (GPA.pai_tracker.member_vars["num_dendrites_added"] > 2)
                and GPA.pc.get_testing_dendrite_capacity()
            ):
                GPA.pai_tracker.save_graphs()
                print(
                    "Successfully added 3 dendrites with "
                    "GPA.pc.set_testing_dendrite_capacity(True) (default). "
                    "You may now set that to False and run a real experiment."
                )
                pdb.set_trace()
                return net, False, True

            # If doing neuron training but this dendrite count didn't improve
            if (
                (GPA.pai_tracker.member_vars["mode"] == "n")
                or GPA.pc.get_learn_dendrites_live()
            ) and (GPA.pai_tracker.member_vars["current_n_set_global_best"] is False):
                new_restructuring_status_value, net = process_no_improvement(net)
                # if this was the final try return that training is complete
                if new_restructuring_status_value == TRAINING_COMPLETE:
                    return net, True, True
                else:
                    restructuring_status_value = update_restructuring_status(
                        restructuring_status_value, new_restructuring_status_value
                    )
            # Else if did improve, do a normal switch process
            else:
                if GPA.pc.get_verbose():
                    print(
                        f"Calling switch_mode with "
                        f'{GPA.pai_tracker.member_vars["current_n_set_global_best"]}, '
                        f'{GPA.pai_tracker.member_vars["current_n_learning_rate_initial_skip_steps"]}, '
                        f'{GPA.pai_tracker.member_vars["last_max_learning_rate_steps"]}, '
                        f'{GPA.pai_tracker.member_vars["last_max_learning_rate_value"]}'
                    )
                # If the max number of dendrites has been hit or not doing pai and adding dendtites
                # then return rather than adding more
                if (
                    (GPA.pai_tracker.member_vars["mode"] == "n")
                    and (
                        GPA.pc.get_max_dendrites()
                        == GPA.pai_tracker.member_vars["num_dendrites_added"]
                    )
                ) or (GPA.pai_tracker.member_vars["doing_pai"] is False):
                    net = process_final_network(net)
                    return net, True, True

                # Otherwise if its neuron training mode reset the counter of failed dendrites
                if GPA.pai_tracker.member_vars["mode"] == "n":
                    GPA.pai_tracker.member_vars["num_dendrite_tries"] = 0
                    if GPA.pc.get_verbose():
                        print(
                            "Adding new dendrites without resetting which means "
                            "the last ones improved. Resetting num_dendrite_tries"
                        )

                GPA.pai_tracker.save_graphs(
                    f'_beforeSwitch_{len(GPA.pai_tracker.member_vars["switch_epochs"])}'
                )

                if GPA.pc.get_test_saves():
                    UPA.save_system(
                        net,
                        GPA.pc.get_save_name(),
                        f'beforeSwitch_{len(GPA.pai_tracker.member_vars["switch_epochs"])}',
                    )
                    # Copy current best model from this set of dendrites
                    shutil.copyfile(
                        f"{GPA.pc.get_save_name()}/best_model.pt",
                        f'{GPA.pc.get_save_name()}/best_model_beforeSwitch_{len(GPA.pai_tracker.member_vars["switch_epochs"])}.pt',
                    )

                net = UPA.change_learning_modes(
                    net,
                    GPA.pc.get_save_name(),
                    "best_model",
                    GPA.pai_tracker.member_vars["doing_pai"],
                )
                restructuring_status_value = NETWORK_RESTRUCTURED

            # If restructured is true, clear scheduler/optimizer before saving
            if restructuring_status_value != NETWORK_RESTRUCTURED:
                print(
                    "Restructured should always be triggered here, let us know if you encounter this situation"
                )
                pdb.set_trace()

            # Since there is a restructuring optimizer and scheduler must be reinitialized after return
            GPA.pai_tracker.clear_optimizer_and_scheduler()

            # Save the model from after the switch
            UPA.save_system(
                net,
                GPA.pc.get_save_name(),
                f'switch_{len(GPA.pai_tracker.member_vars["switch_epochs"])}',
            )

        # If not time to switch and you have a scheduler, perform the update step
        elif GPA.pai_tracker.member_vars["scheduler"] is not None:
            new_restructuring_status_value, net = process_scheduler_update(
                net, accuracy, epochs_since_cycle_switch
            )
            restructuring_status_value = update_restructuring_status(
                restructuring_status_value, new_restructuring_status_value
            )

        GPA.pai_tracker.start_epoch(internal_call=True)
        GPA.pai_tracker.save_graphs()

        if restructuring_status_value == NETWORK_RESTRUCTURED:
            GPA.pai_tracker.member_vars["epoch_last_improved"] = (
                GPA.pai_tracker.member_vars["num_epochs_run"]
            )
            if GPA.pc.get_verbose():
                print(
                    f"Setting epoch last improved to "
                    f'{GPA.pai_tracker.member_vars["epoch_last_improved"]}'
                )

            now = datetime.now()
            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")

            if GPA.pc.get_verbose():
                print("Not saving restructure right now")

            for param in net.parameters():
                param.data = param.data.contiguous()

        if GPA.pc.get_verbose():
            print(
                f"Completed adding score. Restructured is {restructuring_status_value}, "
                f"\ncurrent switch list is:"
            )
            print(GPA.pai_tracker.member_vars["switch_epochs"])

        # Always False for training complete if nothing triggered that training is over
        return net, restructuring_status_value, False

    def clear_all_processors(self):
        """Clear all processors from modules."""
        for module in self.neuron_module_vector:
            module.clear_processors()

    def create_new_dendrite_module(self):
        """Add dendrite module to all neuron modules."""
        for module in self.neuron_module_vector:
            module.create_new_dendrite_module()

    def apply_pb_grads(self):
        """Apply perforated backpropagation gradients to all modules."""
        if self.member_vars["mode"] == "p":
            for module in self.neuron_module_vector:
                module.apply_pb_grads()

    def apply_pb_zero(self):
        """Apply perforated backpropagation zero gradients to all modules."""
        if self.member_vars["mode"] == "p":
            for module in self.neuron_module_vector:
                module.apply_pb_zero()
