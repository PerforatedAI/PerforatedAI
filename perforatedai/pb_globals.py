# Copyright (c) 2025 Perforated AI
"""PAI configuration file."""

import math
import sys

import torch
import torch.nn as nn

### Global Constants

# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Debug settings
debugging_input_dimensions = 0
# Debugging input tensor sizes.
# This will slow things down very slightly and is not necessary but can help
# catch when dimensions were not filled in correctly.
confirm_correct_sizes = False

# Confirmation flags for non-recommended options
unwrapped_modules_confirmed = False
weight_decay_accepted = False
checked_skipped_modules = False

# Verbosity settings
verbose = False
extra_verbose = False
# Suppress all PAI prints
silent = False

# Analysis settings
save_old_graph_scores = True

# Testing settings
testing_dendrite_capacity = True

# File format settings
using_safe_tensors = True

# In place for future implementation options of adding multiple candidate
# dendrites together
global_candidates = 1

# Graph and visualization settings
# A graph setting which can be set to false if you want to do your own
# training visualizations
drawing_pai = True
# Saving test intermediary models, good for experimentation, bad for memory
test_saves = True
# To be filled in later. pai_saves will remove some extra scaffolding for
# slight memory and speed improvements
pai_saves = False

# Input dimensions needs to be set every time. It is set to what format of
# planes you are expecting.
# Neuron index should be set to 0, variable indexes should be set to -1.
# For example, if your format is [batchsize, nodes, x, y]
# input_dimensions is [-1, 0, -1, -1].
# if your format is, [batchsize, time index, nodes] input_dimensions is
# [-1, -1, 0]
input_dimensions = [-1, 0, -1, -1]

# Improvement thresholds
# Percentage improvement increase needed to call a new best validation score
improvement_threshold = 0.0001
# Raw increase needed
improvement_threshold_raw = 1e-5

# Weight initialization settings
# Multiplier when randomizing dendrite weights
candidate_weight_initialization_multiplier = 0.01

# SWITCH MODE SETTINGS

# Add dendrites every time to debug implementation
doing_switch_every_time = 0

# Switch when validation hasn't improved over x epochs
doing_history = 1
# Epochs to try before deciding to load previous best and add dendrites
# Be sure this is higher than scheduler patience
n_epochs_to_switch = 10
# Number to average validation scores over
history_lookback = 1
# Amount of epochs to run after adding a new set of dendrites before checking
# to add more
initial_history_after_switches = 0

# Switch after a fixed number of epochs
doing_fixed_switch = 2
# Number of epochs to complete before switching
fixed_switch_num = 250
# An additional flag if you want your first switch to occur later than all the
# rest for initial pretraining
first_fixed_switch_num = 249

# A setting to not add dendrites and just do regular training
# Warning, this will also never trigger training_complete
doing_no_switch = 3

# Default switch mode
switch_mode = doing_history

# Reset settings
# Resets score on switch
# This can be useful if you need many epochs to catch up to the best score
# from the previous version after adding dendrites
reset_best_score_on_switch = False

# Advanced settings
# Not used in open source implementation, leave as default
learn_dendrites_live = False
no_extra_n_modes = True

# Data type for new modules and dendrite to dendrite / dendrite to neuron
# weights
d_type = torch.float

# Dendrite retention settings
# A setting to keep dendrites even if they do not improve scores
retain_all_dendrites = False

# Learning rate management
# A setting to automatically sweep over previously used learning rates when
# adding new dendrites
# Sometimes it's best to go back to initial LR, but often its best to start
# at a lower LR
find_best_lr = True
# Enforces the above even if the previous epoch didn't lower the learning rate
dont_give_up_unless_learning_rate_lowered = True

# Dendrite attempt settings
# Set to 1 if you want to quit as soon as one dendrite fails
# Higher values will try new random dendrite weights this many times before
# accepting that more dendrites don't improve
max_dendrite_tries = 5
# Max dendrites to add even if they do continue improving scores
max_dendrites = 100

# Scheduler parameter settings
# Have learning rate params be by total epoch
param_vals_by_total_epoch = 0
# Reset the params at every switch
param_vals_by_update_epoch = 1
# Reset params for dendrite starts but not for normal restarts
# Not used for open source version
param_vals_by_neuron_epoch_start = 2
# Default setting
param_vals_setting = param_vals_by_update_epoch

# Activation function settings
# The activation function to use for dendrites
pb_forward_function = torch.sigmoid

### Global Modules

class PAISequential(nn.Sequential):
    """
    Sequential module wrapper for PAI.
    
    This takes in an array of layers. For example:
    
        PAISequential([nn.Linear(2 * hidden_dim, seq_width),
                     nn.LayerNorm(seq_width)])
    
    This should be used for:
        - all normalization layers
    This can be used for:
        - final output layer and softmax
    """
    
    def __init__(self, layer_array):
        super(PAISequential, self).__init__()
        self.model = nn.Sequential(*layer_array)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

### Global objects and variables

# Pointer to the PAI Tracker which handles adding dendrites
pai_tracker = []

# Lists for module types and names to add dendrites to
modules_to_convert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
module_names_to_convert = ['PAISequential']

# All modules should either be converted or tracked to ensure all modules
# are accounted for
modules_to_track = []
module_names_to_track = []
module_ids_to_track = []

# Replacement modules happen before the conversion,
# so replaced modules will then also be run through the conversion steps
# These are for modules that need to be replaced before addition of dendrites
# See the resnet example in pb_models
modules_to_replace = []
# Modules to replace the above modules with
replacement_modules = []

# Dendrites default to modules which are one tensor input and one tensor
# output in forward()
# Other modules require to be labeled as modules with processing and assigned
# processing classes
# This can be done by module type or module name see customization.md in API
# for example
modules_with_processing = []
modules_processing_classes = []
module_names_with_processing = []
module_by_name_processing_classes = []

# Module names can be added to this list to not convert a specific module
# even though other ones are converted
# It is recommended to not use this, but if you are working with pretrained
# model that you can't edit and a module is causing problems, you can add it
# here to skip it
module_names_to_skip = []

# Similarly here as above. Some huggingface models have multiple pointers to
# the same modules which cause problems
# If you want to only save one of the multiple pointers you can set which ones
# not to save here
module_names_to_not_save = ['.base_model']


