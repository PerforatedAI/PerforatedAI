# Copyright (c) 2025 Perforated AI
"""PAI configuration file."""

import math
import sys

import torch
import torch.nn as nn

### Global Constants


class PAIConfig:
    """Configuration class for PAI settings."""

    def __init__(self):
        ### Global Constants
        # Device configuration
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # User should never set this manually
        self.save_name = "PAI"

        # Debug settings
        self.debugging_input_dimensions = 0
        # Debugging input tensor sizes.
        # This will slow things down very slightly and is not necessary but can help
        # catch when dimensions were not filled in correctly.
        self.confirm_correct_sizes = False

        # Confirmation flags for non-recommended options
        self.unwrapped_modules_confirmed = False
        self.weight_decay_accepted = False
        self.checked_skipped_modules = False

        # Verbosity settings
        self.verbose = False
        self.extra_verbose = False
        # Suppress all PAI prints
        self.silent = False

        # Analysis settings
        self.save_old_graph_scores = True

        # Testing settings
        self.testing_dendrite_capacity = True

        # File format settings
        self.using_safe_tensors = True

        # In place for future implementation options of adding multiple candidate
        # dendrites together
        self.global_candidates = 1

        # Graph and visualization settings
        # A graph setting which can be set to false if you want to do your own
        # training visualizations
        self.drawing_pai = True
        # Saving test intermediary models, good for experimentation, bad for memory
        self.test_saves = True
        # To be filled in later. pai_saves will remove some extra scaffolding for
        # slight memory and speed improvements
        self.pai_saves = False

        # Input dimensions needs to be set every time. It is set to what format of
        # planes you are expecting.
        # Neuron index should be set to 0, variable indexes should be set to -1.
        # For example, if your format is [batchsize, nodes, x, y]
        # input_dimensions is [-1, 0, -1, -1].
        # if your format is, [batchsize, time index, nodes] input_dimensions is
        # [-1, -1, 0]
        self.input_dimensions = [-1, 0, -1, -1]

        # Improvement thresholds
        # Percentage improvement increase needed to call a new best validation score
        self.improvement_threshold = 0.0001
        # Raw increase needed
        self.improvement_threshold_raw = 1e-5

        # Weight initialization settings
        # Multiplier when randomizing dendrite weights
        self.candidate_weight_initialization_multiplier = 0.01

        # SWITCH MODE SETTINGS

        # Add dendrites every time to debug implementation
        self.DOING_SWITCH_EVERY_TIME = 0

        # Switch when validation hasn't improved over x epochs
        self.DOING_HISTORY = 1
        # Epochs to try before deciding to load previous best and add dendrites
        # Be sure this is higher than scheduler patience
        self.n_epochs_to_switch = 10
        # Number to average validation scores over
        self.history_lookback = 1
        # Amount of epochs to run after adding a new set of dendrites before checking
        # to add more
        self.initial_history_after_switches = 0

        # Switch after a fixed number of epochs
        self.DOING_FIXED_SWITCH = 2
        # Number of epochs to complete before switching
        self.fixed_switch_num = 250
        # An additional flag if you want your first switch to occur later than all the
        # rest for initial pretraining
        self.first_fixed_switch_num = 249

        # A setting to not add dendrites and just do regular training
        # Warning, this will also never trigger training_complete
        self.DOING_NO_SWITCH = 3

        # Default switch mode
        switch_mode = self.DOING_HISTORY

        # Reset settings
        # Resets score on switch
        # This can be useful if you need many epochs to catch up to the best score
        # from the previous version after adding dendrites
        self.reset_best_score_on_switch = False

        # Advanced settings
        # Not used in open source implementation, leave as default
        self.learn_dendrites_live = False
        self.no_extra_n_modes = True

        # Data type for new modules and dendrite to dendrite / dendrite to neuron
        # weights
        self.d_type = torch.float

        # Dendrite retention settings
        # A setting to keep dendrites even if they do not improve scores
        self.retain_all_dendrites = False

        # Learning rate management
        # A setting to automatically sweep over previously used learning rates when
        # adding new dendrites
        # Sometimes it's best to go back to initial LR, but often its best to start
        # at a lower LR
        self.find_best_lr = True
        # Enforces the above even if the previous epoch didn't lower the learning rate
        self.dont_give_up_unless_learning_rate_lowered = True

        # Dendrite attempt settings
        # Set to 1 if you want to quit as soon as one dendrite fails
        # Higher values will try new random dendrite weights this many times before
        # accepting that more dendrites don't improve
        self.max_dendrite_tries = 2
        # Max dendrites to add even if they do continue improving scores
        self.max_dendrites = 100

        # Scheduler parameter settings
        # Have learning rate params be by total epoch
        self.PARAM_VALS_BY_TOTAL_EPOCH = 0
        # Reset the params at every switch
        self.PARAM_VALS_BY_UPDATE_EPOCH = 1
        # Reset params for dendrite starts but not for normal restarts
        # Not used for open source version
        self.PARAM_VALS_BY_NEURON_EPOCH_START = 2
        # Default setting
        self.param_vals_setting = self.PARAM_VALS_BY_UPDATE_EPOCH

        # Activation function settings
        # The activation function to use for dendrites
        self.pb_forward_function = torch.sigmoid

        # Lists for module types and names to add dendrites to
        # For these lists no specifier means type, name is module name
        # and ids is the individual modules id, eg. model.conv2
        self.modules_to_convert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
        self.module_names_to_convert = ["PAISequential"]
        self.module_ids_to_convert = []

        # All modules should either be converted or tracked to ensure all modules
        # are accounted for
        self.modules_to_track = []
        self.module_names_to_track = []
        # IDs are for if you want to pass only a single module by its assigned ID rather than the module type by name
        self.module_ids_to_track = []

        # Replacement modules happen before the conversion,
        # so replaced modules will then also be run through the conversion steps
        # These are for modules that need to be replaced before addition of dendrites
        # See the resnet example in models_perforatedai
        self.modules_to_replace = []
        # Modules to replace the above modules with
        self.replacement_modules = []

        # Dendrites default to modules which are one tensor input and one tensor
        # output in forward()
        # Other modules require to be labeled as modules with processing and assigned
        # processing classes
        # This can be done by module type or module name see customization.md in API
        # for example
        self.modules_with_processing = []
        self.modules_processing_classes = []
        self.module_names_with_processing = []
        self.module_by_name_processing_classes = []

        # Similarly here as above. Some huggingface models have multiple pointers to
        # the same modules which cause problems
        # If you want to only save one of the multiple pointers you can set which ones
        # not to save here
        self.module_names_to_not_save = [".base_model"]

        self.perforated_backpropagation = False

    # Setters and Getters for PAIConfig class

    # use_cuda
    def set_use_cuda(self, value):
        self.use_cuda = value

    def get_use_cuda(self):
        return self.use_cuda

    # device
    def set_device(self, value):
        self.device = value

    def get_device(self):
        return self.device

    def get_save_name(self):
        return self.save_name

    # debugging_input_dimensions
    def set_debugging_input_dimensions(self, value):
        self.debugging_input_dimensions = value

    def get_debugging_input_dimensions(self):
        return self.debugging_input_dimensions

    # confirm_correct_sizes
    def set_confirm_correct_sizes(self, value):
        self.confirm_correct_sizes = value

    def get_confirm_correct_sizes(self):
        return self.confirm_correct_sizes

    # unwrapped_modules_confirmed
    def set_unwrapped_modules_confirmed(self, value):
        self.unwrapped_modules_confirmed = value

    def get_unwrapped_modules_confirmed(self):
        return self.unwrapped_modules_confirmed

    # weight_decay_accepted
    def set_weight_decay_accepted(self, value):
        self.weight_decay_accepted = value

    def get_weight_decay_accepted(self):
        return self.weight_decay_accepted

    # checked_skipped_modules
    def set_checked_skipped_modules(self, value):
        self.checked_skipped_modules = value

    def get_checked_skipped_modules(self):
        return self.checked_skipped_modules

    # verbose
    def set_verbose(self, value):
        self.verbose = value

    def get_verbose(self):
        return self.verbose

    # extra_verbose
    def set_extra_verbose(self, value):
        self.extra_verbose = value

    def get_extra_verbose(self):
        return self.extra_verbose

    # silent
    def set_silent(self, value):
        self.silent = value

    def get_silent(self):
        return self.silent

    # save_old_graph_scores
    def set_save_old_graph_scores(self, value):
        self.save_old_graph_scores = value

    def get_save_old_graph_scores(self):
        return self.save_old_graph_scores

    # testing_dendrite_capacity
    def set_testing_dendrite_capacity(self, value):
        self.testing_dendrite_capacity = value

    def get_testing_dendrite_capacity(self):
        return self.testing_dendrite_capacity

    # using_safe_tensors
    def set_using_safe_tensors(self, value):
        self.using_safe_tensors = value

    def get_using_safe_tensors(self):
        return self.using_safe_tensors

    # global_candidates
    def set_global_candidates(self, value):
        self.global_candidates = value

    def get_global_candidates(self):
        return self.global_candidates

    # drawing_pai
    def set_drawing_pai(self, value):
        self.drawing_pai = value

    def get_drawing_pai(self):
        return self.drawing_pai

    # test_saves
    def set_test_saves(self, value):
        self.test_saves = value

    def get_test_saves(self):
        return self.test_saves

    # pai_saves
    def set_pai_saves(self, value):
        self.pai_saves = value

    def get_pai_saves(self):
        return self.pai_saves

    # input_dimensions
    def set_input_dimensions(self, value):
        self.input_dimensions = value

    def get_input_dimensions(self):
        return self.input_dimensions

    # improvement_threshold
    def set_improvement_threshold(self, value):
        self.improvement_threshold = value

    def get_improvement_threshold(self):
        return self.improvement_threshold

    # improvement_threshold_raw
    def set_improvement_threshold_raw(self, value):
        self.improvement_threshold_raw = value

    def get_improvement_threshold_raw(self):
        return self.improvement_threshold_raw

    # candidate_weight_initialization_multiplier
    def set_candidate_weight_initialization_multiplier(self, value):
        self.candidate_weight_initialization_multiplier = value

    def get_candidate_weight_initialization_multiplier(self):
        return self.candidate_weight_initialization_multiplier

    # n_epochs_to_switch
    def set_n_epochs_to_switch(self, value):
        self.n_epochs_to_switch = value

    def get_n_epochs_to_switch(self):
        return self.n_epochs_to_switch

    # history_lookback
    def set_history_lookback(self, value):
        self.history_lookback = value

    def get_history_lookback(self):
        return self.history_lookback

    # initial_history_after_switches
    def set_initial_history_after_switches(self, value):
        self.initial_history_after_switches = value

    def get_initial_history_after_switches(self):
        return self.initial_history_after_switches

    # fixed_switch_num
    def set_fixed_switch_num(self, value):
        self.fixed_switch_num = value

    def get_fixed_switch_num(self):
        return self.fixed_switch_num

    # first_fixed_switch_num
    def set_first_fixed_switch_num(self, value):
        self.first_fixed_switch_num = value

    def get_first_fixed_switch_num(self):
        return self.first_fixed_switch_num

    # switch_mode
    def set_switch_mode(self, value):
        self.switch_mode = value

    def get_switch_mode(self):
        return self.switch_mode

    # reset_best_score_on_switch
    def set_reset_best_score_on_switch(self, value):
        self.reset_best_score_on_switch = value

    def get_reset_best_score_on_switch(self):
        return self.reset_best_score_on_switch

    # learn_dendrites_live
    def set_learn_dendrites_live(self, value):
        self.learn_dendrites_live = value

    def get_learn_dendrites_live(self):
        return self.learn_dendrites_live

    # no_extra_n_modes
    def set_no_extra_n_modes(self, value):
        self.no_extra_n_modes = value

    def get_no_extra_n_modes(self):
        return self.no_extra_n_modes

    # d_type
    def set_d_type(self, value):
        self.d_type = value

    def get_d_type(self):
        return self.d_type

    # retain_all_dendrites
    def set_retain_all_dendrites(self, value):
        self.retain_all_dendrites = value

    def get_retain_all_dendrites(self):
        return self.retain_all_dendrites

    # find_best_lr
    def set_find_best_lr(self, value):
        self.find_best_lr = value

    def get_find_best_lr(self):
        return self.find_best_lr

    # dont_give_up_unless_learning_rate_lowered
    def set_dont_give_up_unless_learning_rate_lowered(self, value):
        self.dont_give_up_unless_learning_rate_lowered = value

    def get_dont_give_up_unless_learning_rate_lowered(self):
        return self.dont_give_up_unless_learning_rate_lowered

    # max_dendrite_tries
    def set_max_dendrite_tries(self, value):
        self.max_dendrite_tries = value

    def get_max_dendrite_tries(self):
        return self.max_dendrite_tries

    # max_dendrites
    def set_max_dendrites(self, value):
        self.max_dendrites = value

    def get_max_dendrites(self):
        return self.max_dendrites

    # param_vals_setting
    def set_param_vals_setting(self, value):
        self.param_vals_setting = value

    def get_param_vals_setting(self):
        return self.param_vals_setting

    # pb_forward_function
    def set_pb_forward_function(self, value):
        self.pb_forward_function = value

    def get_pb_forward_function(self):
        return self.pb_forward_function

    # modules_to_convert
    def set_modules_to_convert(self, value):
        self.modules_to_convert = value

    def get_modules_to_convert(self):
        return self.modules_to_convert

    def append_modules_to_convert(self, value):
        self.modules_to_convert += value

    # module_names_to_convert
    def set_module_names_to_convert(self, value):
        self.module_names_to_convert = value

    def get_module_names_to_convert(self):
        return self.module_names_to_convert

    def append_module_names_to_convert(self, value):
        self.module_names_to_convert += value

    # module_ids_to_convert
    def set_module_ids_to_convert(self, value):
        self.module_ids_to_convert = value

    def get_module_ids_to_convert(self):
        return self.module_ids_to_convert

    def append_module_ids_to_convert(self, value):
        self.module_ids_to_convert += value

    # modules_to_track
    def set_modules_to_track(self, value):
        self.modules_to_track = value

    def get_modules_to_track(self):
        return self.modules_to_track

    def append_modules_to_track(self, value):
        self.modules_to_track += value

    # module_names_to_track
    def set_module_names_to_track(self, value):
        self.module_names_to_track = value

    def get_module_names_to_track(self):
        return self.module_names_to_track

    def append_module_names_to_track(self, value):
        self.module_names_to_track += value

    # module_ids_to_track
    def set_module_ids_to_track(self, value):
        self.module_ids_to_track = value

    def get_module_ids_to_track(self):
        return self.module_ids_to_track

    def append_module_ids_to_track(self, value):
        self.module_ids_to_track += value

    # modules_to_replace
    def set_modules_to_replace(self, value):
        self.modules_to_replace = value

    def get_modules_to_replace(self):
        return self.modules_to_replace

    def append_modules_to_replace(self, value):
        self.modules_to_replace += value

    # replacement_modules
    def set_replacement_modules(self, value):
        self.replacement_modules = value

    def get_replacement_modules(self):
        return self.replacement_modules

    def append_replacement_modules(self, value):
        self.replacement_modules += value

    # modules_with_processing
    def set_modules_with_processing(self, value):
        self.modules_with_processing = value

    def get_modules_with_processing(self):
        return self.modules_with_processing

    def append_modules_with_processing(self, value):
        self.modules_with_processing += value

    # modules_processing_classes
    def set_modules_processing_classes(self, value):
        self.modules_processing_classes = value

    def get_modules_processing_classes(self):
        return self.modules_processing_classes

    def append_modules_processing_classes(self, value):
        self.modules_processing_classes += value

    # module_names_with_processing
    def set_module_names_with_processing(self, value):
        self.module_names_with_processing = value

    def get_module_names_with_processing(self):
        return self.module_names_with_processing

    def append_module_names_with_processing(self, value):
        self.module_names_with_processing += value

    # module_by_name_processing_classes
    def set_module_by_name_processing_classes(self, value):
        self.module_by_name_processing_classes = value

    def get_module_by_name_processing_classes(self):
        return self.module_by_name_processing_classes

    def append_module_by_name_processing_classes(self, value):
        self.module_by_name_processing_classes += value

    # module_names_to_not_save
    def set_module_names_to_not_save(self, value):
        self.module_names_to_not_save = value

    def get_module_names_to_not_save(self):
        return self.module_names_to_not_save

    def append_module_names_to_not_save(self, value):
        self.module_names_to_not_save += value

    # perforated_backpropagation
    def set_perforated_backpropagation(self, value):
        self.perforated_backpropagation = value

    def get_perforated_backpropagation(self):
        return self.perforated_backpropagation


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

### Global Modules
pc = PAIConfig()
# Pointer to the PAI Tracker which handles adding dendrites
pai_tracker = []


def add_pbp_var(obj, var_name, initial_value):
    private_name = f"_{var_name}"

    # Add the private variable to the instance
    setattr(obj, private_name, initial_value)

    # Define getter and setter
    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        setattr(self, private_name, value)

    # Attach methods to the instance
    setattr(obj, f"get_{var_name}", getter.__get__(obj))
    setattr(obj, f"set_{var_name}", setter.__get__(obj))


# This will be set to true if perforated backpropagation is available
# Do not just set this to True without the library and a license, it will cause errors
try:
    import perforatedbp.globals_pbp as perforatedbp_globals

    print("Building dendrites with Perforated Backpropagation")

    pc.set_perforated_backpropagation(True)
    # This is default to True for open source version
    # But defaults to False for perforated backpropagation
    pc.set_no_extra_n_modes(False)

    # Loop through the vars module's attributes and add them dynamically
    for var_name in dir(perforatedbp_globals):
        if not var_name.startswith("_"):
            add_pbp_var(pc, var_name, getattr(perforatedbp_globals, var_name))

except ImportError:
    print("Building dendrites without Perforated Backpropagation")
