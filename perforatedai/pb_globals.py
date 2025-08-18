# Copyright (c) 2025 Perforated AI
"""PAI configuration file."""

import math
import sys

import torch
import torch.nn as nn

### Global Constants

# Device configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Debug settings
DEBUGGING_INPUT_DIMENSIONS = 0
# Debugging input tensor sizes.
# This will slow things down very slightly and is not necessary but can help
# catch when dimensions were not filled in correctly.
CONFIRM_CORRECT_SIZES = False

# Confirmation flags for non-recommended options
UNWRAPPED_MODULES_CONFIRMED = False
WEIGHT_DECAY_ACCEPTED = False
CHECKED_SKIPPED_MODULES = False

# Verbosity settings
VERBOSE = False
EXTRA_VERBOSE = False
# Suppress all PAI prints
SILENT = False

# Analysis settings
SAVE_OLD_GRAPH_SCORES = True

# Testing settings
TESTING_DENDRITE_CAPACITY = True

# File format settings
USING_SAFE_TENSORS = True

# In place for future implementation options of adding multiple candidate
# dendrites together
GLOBAL_CANDIDATES = 1

# Graph and visualization settings
# A graph setting which can be set to false if you want to do your own
# training visualizations
DRAWING_PAI = True
# Saving test intermediary models, good for experimentation, bad for memory
TEST_SAVES = True
# To be filled in later. pai_saves will remove some extra scaffolding for
# slight memory and speed improvements
PAI_SAVES = False

# Input dimensions needs to be set every time. It is set to what format of
# planes you are expecting.
# Neuron index should be set to 0, variable indexes should be set to -1.
# For example, if your format is [batchsize, nodes, x, y]
# input_dimensions is [-1, 0, -1, -1].
# if your format is, [batchsize, time index, nodes] input_dimensions is
# [-1, -1, 0]
INPUT_DIMENSIONS = [-1, 0, -1, -1]

# Improvement thresholds
# Percentage improvement increase needed to call a new best validation score
IMPROVEMENT_THRESHOLD = 0.0001
# Raw increase needed
IMPROVEMENT_THRESHOLD_RAW = 1e-5

# Weight initialization settings
# Multiplier when randomizing dendrite weights
CANDIDATE_WEIGHT_INITIALIZATION_MULTIPLIER = 0.01

# SWITCH MODE SETTINGS

# Add dendrites every time to debug implementation
DOING_SWITCH_EVERY_TIME = 0

# Switch when validation hasn't improved over x epochs
DOING_HISTORY = 1
# Epochs to try before deciding to load previous best and add dendrites
# Be sure this is higher than scheduler patience
N_EPOCHS_TO_SWITCH = 10
# Number to average validation scores over
HISTORY_LOOKBACK = 1
# Amount of epochs to run after adding a new set of dendrites before checking
# to add more
INITIAL_HISTORY_AFTER_SWITCHES = 0

# Switch after a fixed number of epochs
DOING_FIXED_SWITCH = 2
# Number of epochs to complete before switching
FIXED_SWITCH_NUM = 250
# An additional flag if you want your first switch to occur later than all the
# rest for initial pretraining
FIRST_FIXED_SWITCH_NUM = 249

# A setting to not add dendrites and just do regular training
# Warning, this will also never trigger training_complete
DOING_NO_SWITCH = 3

# Default switch mode
SWITCH_MODE = DOING_HISTORY

# Reset settings
# Resets score on switch
# This can be useful if you need many epochs to catch up to the best score
# from the previous version after adding dendrites
RESET_BEST_SCORE_ON_SWITCH = False

# Advanced settings
# Not used in open source implementation, leave as default
LEARN_DENDRITES_LIVE = False
NO_EXTRA_N_MODES = True

# Data type for new modules and dendrite to dendrite / dendrite to neuron
# weights
D_TYPE = torch.float

# Dendrite retention settings
# A setting to keep dendrites even if they do not improve scores
RETAIN_ALL_DENDRITES = False

# Learning rate management
# A setting to automatically sweep over previously used learning rates when
# adding new dendrites
# Sometimes it's best to go back to initial LR, but often its best to start
# at a lower LR
FIND_BEST_LR = True
# Enforces the above even if the previous epoch didn't lower the learning rate
DONT_GIVE_UP_UNLESS_LEARNING_RATE_LOWERED = True

# Dendrite attempt settings
# Set to 1 if you want to quit as soon as one dendrite fails
# Higher values will try new random dendrite weights this many times before
# accepting that more dendrites don't improve
MAX_DENDRITE_TRIES = 5
# Max dendrites to add even if they do continue improving scores
MAX_DENDRITES = 100

# Scheduler parameter settings
# Have learning rate params be by total epoch
PARAM_VALS_BY_TOTAL_EPOCH = 0
# Reset the params at every switch
PARAM_VALS_BY_UPDATE_EPOCH = 1
# Reset params for dendrite starts but not for normal restarts
# Not used for open source version
PARAM_VALS_BY_NEURON_EPOCH_START = 2
# Default setting
PARAM_VALS_SETTING = PARAM_VALS_BY_UPDATE_EPOCH

# Activation function settings
# The activation function to use for dendrites
PB_FORWARD_FUNCTION = torch.sigmoid

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


