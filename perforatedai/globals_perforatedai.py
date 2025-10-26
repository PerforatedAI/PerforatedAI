# Copyright (c) 2025 Perforated AI
"""PAI configuration file.

This module provides configuration classes and utilities for Perforated AI (PAI),
including device settings, dendrite management, module conversion options,
and training parameters.
"""

import math
import sys

import torch
import torch.nn as nn


def add_pai_config_var_functions(obj, var_name, initial_value, list_type=False):
    """Dynamically add a property with getter and setter to an object.

    This function adds a private variable along with getter and setter methods
    to a given object instance. Used for integrating initial and Perforated
    Backpropagation variables into the PAIConfig class.

    Parameters
    ----------
    obj : object
        The object to which the property will be added.
    var_name : str
        Name of the variable/property to create.
    initial_value : any
        Initial value for the property.

    Returns
    -------
    None

    Notes
    -----
    Creates three attributes on obj:
        - _{var_name}: private storage
        - get_{var_name}: getter method
        - set_{var_name}: setter method
    """
    private_name = f"_{var_name}"

    # Add the private variable to the instance
    setattr(obj, private_name, initial_value)

    # Define getter and setter and appender

    def getter_val(self):
        """Get the current value of the property.

        If the property a individual value but is set to be a list,
        return the element corresponding to the
        current number of dendrites added. Otherwise, return the value directly.

        Returns
        -------
        any
            Current value of the property.

        Notes:
        -----
        Many variables have optimal settings that must change as dendrites are added
        this enables those values to be dynamically set very easily.
        """
        global pai_tracker
        if type(getattr(self, private_name)) is list:
            return getattr(self, private_name)[
                min(
                    len(getattr(self, private_name)) - 1,
                    pai_tracker.member_vars["num_dendrites_added"],
                )
            ]
        return getattr(self, private_name)

    def getter_list(self):
        return getattr(self, private_name)

    def setter(self, value):
        """Set the value of the property."""
        setattr(self, private_name, value)

    def appender(self, value):
        """Append a value to the property if it is a list."""
        if isinstance(getattr(self, private_name), list):
            setattr(self, private_name, getattr(self, private_name) + value)
            print(
                'New list value of "{}": {}'.format(
                    private_name, getattr(self, private_name)
                )
            )
        else:
            raise TypeError(f"Cannot append to non-list attribute '{var_name}'")

    # Attach methods to the instance
    if list_type:
        setattr(obj, f"get_{var_name}", getter_list.__get__(obj))
    else:
        setattr(obj, f"get_{var_name}", getter_val.__get__(obj))
    setattr(obj, f"set_{var_name}", setter.__get__(obj))
    setattr(obj, f"append_{var_name}", appender.__get__(obj))


class PAIConfig:
    """Configuration class for PAI settings.

    This class manages all configuration parameters for the Perforated AI system,
    including device settings, dendrite behavior, module conversion rules,
    training parameters, and debugging options.

    Attributes
    ----------
    use_cuda : bool
        Whether CUDA is available and should be used.
    device : torch.device
        The device to use for computation (CPU, CUDA, etc.).
    save_name : str
        Name used for saving models (should not be set manually).
    debugging_input_dimensions : int
        Debug level for input dimension checking.
    confirm_correct_sizes : bool
        Whether to verify tensor sizes during execution.
    unwrapped_modules_confirmed : bool
        Confirmation flag for using unwrapped modules.
    weight_decay_accepted : bool
        Confirmation flag for accepting weight decay.
    checked_skipped_modules : bool
        Whether skipped modules have been verified.
    verbose : bool
        Enable verbose logging output.
    extra_verbose : bool
        Enable extra verbose logging output.
    silent : bool
        Suppress all PAI print statements.
    save_old_graph_scores : bool
        Whether to save historical graph scores.
    testing_dendrite_capacity : bool
        Enable dendrite capacity testing mode.
    using_safe_tensors : bool
        Use safe tensors file format for saving.
    global_candidates : int
        Number of global candidate dendrites.
    drawing_pai : bool
        Enable PAI visualization graphs.
    test_saves : bool
        Save intermediary test models.
    pai_saves : bool
        Save PAI-specific format models.
    input_dimensions : list
        Format specification for input tensor dimensions.
    improvement_threshold : float
        Relative improvement threshold for validation scores.
    improvement_threshold_raw : float
        Absolute improvement threshold for validation scores.
    candidate_weight_initialization_multiplier : float
        Multiplier for random dendrite weight initialization.
    DOING_SWITCH_EVERY_TIME : int
        Constant for switch mode: add dendrites every epoch.
    DOING_HISTORY : int
        Constant for switch mode: add dendrites based on validation history.
    n_epochs_to_switch : int
        Number of epochs without improvement before switching.
    history_lookback : int
        Number of epochs to average for validation history.
    initial_history_after_switches : int
        Epochs to wait after adding dendrites before beggining checks.
    DOING_FIXED_SWITCH : int
        Constant for switch mode: add dendrites at fixed intervals.
    fixed_switch_num : int
        Number of epochs between fixed switches.
    first_fixed_switch_num : int
        Number of epochs before first switch (for pretraining).
    DOING_NO_SWITCH : int
        Constant for switch mode: never add dendrites.
    switch_mode : int
        Current switch mode setting.
    reset_best_score_on_switch : bool
        Whether to reset best score when adding dendrites.
    learn_dendrites_live : bool
        Enable live dendrite learning (advanced feature).
    no_extra_n_modes : bool
        Disable extra neuron modes (advanced feature).
    d_type : torch.dtype
        Data type for dendrite weights.
    retain_all_dendrites : bool
        Keep dendrites even if they don't improve performance.
    find_best_lr : bool
        Automatically sweep learning rates when adding dendrites.
    dont_give_up_unless_learning_rate_lowered : bool
        Ensure search lowers learning rate at least once.
    max_dendrite_tries : int
        Maximum attempts to add dendrites with random initializations.
    max_dendrites : int
        Maximum total number of dendrites to add.
    PARAM_VALS_BY_TOTAL_EPOCH : int
        Constant: scheduler params tracked by total epochs.
    PARAM_VALS_BY_UPDATE_EPOCH : int
        Constant: scheduler params reset at each switch.
    PARAM_VALS_BY_NEURON_EPOCH_START : int
        Constant: scheduler params reset for neuron starts only.
    param_vals_setting : int
        Current parameter tracking mode.
    pai_forward_function : callable
        Activation function used for dendrites.
    modules_to_convert : list
        Module types to convert to PAI modules.
    module_names_to_convert : list
        Module names to convert to PAI modules.
    module_ids_to_convert : list
        Specific module IDs to convert to PAI modules.
    modules_to_track : list
        Module types to track but not convert.
    module_names_to_track : list
        Module names to track but not convert.
    module_ids_to_track : list
        Specific module IDs to track but not convert.
    modules_to_replace : list
        Module types to replace before conversion.
    replacement_modules : list
        Replacement modules for modules_to_replace.
    modules_with_processing : list
        Module types requiring custom processing.
    modules_processing_classes : list
        Processing classes for modules_with_processing.
    module_names_with_processing : list
        Module names requiring custom processing.
    module_by_name_processing_classes : list
        Processing classes for module_names_with_processing.
    module_names_to_not_save : list
        Module names to exclude from saving.
    perforated_backpropagation : bool
        Whether Perforated Backpropagation is enabled.
    """

    def __init__(self):
        """Initialize PAIConfig with default settings."""
        ### Global Constants
        # Device configuration
        self.use_cuda = torch.cuda.is_available()
        add_pai_config_var_functions(self, "use_cuda", self.use_cuda)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        add_pai_config_var_functions(self, "device", self.device)

        # User should never set this manually
        self.save_name = "PAI"
        add_pai_config_var_functions(self, "save_name", self.save_name)

        # Debug settings
        self.debugging_input_dimensions = 0
        add_pai_config_var_functions(
            self, "debugging_input_dimensions", self.debugging_input_dimensions
        )
        # Debugging input tensor sizes.
        # This will slow things down very slightly and is not necessary but can help
        # catch when dimensions were not filled in correctly.
        self.confirm_correct_sizes = False
        add_pai_config_var_functions(
            self, "confirm_correct_sizes", self.confirm_correct_sizes
        )

        # Confirmation flags for non-recommended options
        self.unwrapped_modules_confirmed = False
        add_pai_config_var_functions(
            self, "unwrapped_modules_confirmed", self.unwrapped_modules_confirmed
        )
        self.weight_decay_accepted = False
        add_pai_config_var_functions(
            self, "weight_decay_accepted", self.weight_decay_accepted
        )
        self.checked_skipped_modules = False
        add_pai_config_var_functions(
            self, "checked_skipped_modules", self.checked_skipped_modules
        )

        # Verbosity settings
        self.verbose = False
        add_pai_config_var_functions(self, "verbose", self.verbose)
        self.extra_verbose = False
        add_pai_config_var_functions(self, "extra_verbose", self.extra_verbose)
        # Suppress all PAI prints
        self.silent = False
        add_pai_config_var_functions(self, "silent", self.silent)
        # Analysis settings
        self.save_old_graph_scores = True
        add_pai_config_var_functions(
            self, "save_old_graph_scores", self.save_old_graph_scores
        )

        # Testing settings
        self.testing_dendrite_capacity = True
        add_pai_config_var_functions(
            self, "testing_dendrite_capacity", self.testing_dendrite_capacity
        )

        # File format settings
        self.using_safe_tensors = True
        add_pai_config_var_functions(
            self, "using_safe_tensors", self.using_safe_tensors
        )

        # In place for future implementation options of adding multiple candidate
        # dendrites together
        self.global_candidates = 1
        add_pai_config_var_functions(self, "global_candidates", self.global_candidates)

        # Graph and visualization settings
        # A graph setting which can be set to false if you want to do your own
        # training visualizations
        self.drawing_pai = True
        add_pai_config_var_functions(self, "drawing_pai", self.drawing_pai)

        # Saving test intermediary models, good for experimentation, bad for memory
        self.test_saves = True
        add_pai_config_var_functions(self, "test_saves", self.test_saves)
        # To be filled in later. pai_saves will remove some extra scaffolding for
        # slight memory and speed improvements
        self.pai_saves = False
        add_pai_config_var_functions(self, "pai_saves", self.pai_saves)

        # Input dimensions needs to be set every time. It is set to what format of
        # planes you are expecting.
        # Neuron index should be set to 0, variable indexes should be set to -1.
        # For example, if your format is [batchsize, nodes, x, y]
        # input_dimensions is [-1, 0, -1, -1].
        # if your format is, [batchsize, time index, nodes] input_dimensions is
        # [-1, -1, 0]
        self.input_dimensions = [-1, 0, -1, -1]
        add_pai_config_var_functions(
            self, "input_dimensions", self.input_dimensions, list_type=True
        )

        # Improvement thresholds
        # Percentage improvement increase needed to call a new best validation score
        self.improvement_threshold = [0.001, 0.0001, 0.0]
        add_pai_config_var_functions(
            self, "improvement_threshold", self.improvement_threshold
        )

        # Raw increase needed
        self.improvement_threshold_raw = 1e-5
        add_pai_config_var_functions(
            self, "improvement_threshold_raw", self.improvement_threshold_raw
        )

        # Weight initialization settings
        # Multiplier when randomizing dendrite weights
        self.candidate_weight_initialization_multiplier = 0.01
        add_pai_config_var_functions(
            self,
            "candidate_weight_initialization_multiplier",
            self.candidate_weight_initialization_multiplier,
        )
        # Multiplier when randomizing dendrite weights
        self.candidate_weight_init_by_main = False
        add_pai_config_var_functions(
            self,
            "candidate_weight_init_by_main",
            self.candidate_weight_init_by_main,
        )

        # SWITCH MODE SETTINGS

        # Add dendrites every time to debug implementation
        self.DOING_SWITCH_EVERY_TIME = 0

        # Switch when validation hasn't improved over x epochs
        self.DOING_HISTORY = 1
        # Epochs to try before deciding to load previous best and add dendrites
        # Be sure this is higher than scheduler patience
        self.n_epochs_to_switch = 10
        add_pai_config_var_functions(
            self, "n_epochs_to_switch", self.n_epochs_to_switch
        )
        # Number to average validation scores over
        self.history_lookback = 1
        add_pai_config_var_functions(self, "history_lookback", self.history_lookback)
        # Amount of epochs to run after adding a new set of dendrites before checking
        # to add more
        self.initial_history_after_switches = 0
        add_pai_config_var_functions(
            self, "initial_history_after_switches", self.initial_history_after_switches
        )

        # Switch after a fixed number of epochs
        self.DOING_FIXED_SWITCH = 2
        # Number of epochs to complete before switching
        self.fixed_switch_num = 250
        add_pai_config_var_functions(self, "fixed_switch_num", self.fixed_switch_num)
        # An additional flag if you want your first switch to occur later than all the
        # rest for initial pretraining.  This is a new minimum, if its lower than
        # the above it will be ignored.
        self.first_fixed_switch_num = 0
        add_pai_config_var_functions(
            self, "first_fixed_switch_num", self.first_fixed_switch_num
        )

        # A setting to not add dendrites and just do regular training
        # Warning, this will also never trigger training_complete
        self.DOING_NO_SWITCH = 3

        # Default switch mode
        self.switch_mode = self.DOING_HISTORY
        add_pai_config_var_functions(self, "switch_mode", self.switch_mode)

        # Reset settings
        # Resets score on switch
        # This can be useful if you need many epochs to catch up to the best score
        # from the previous version after adding dendrites
        self.reset_best_score_on_switch = False
        add_pai_config_var_functions(
            self, "reset_best_score_on_switch", self.reset_best_score_on_switch
        )

        # Advanced settings
        # Not used in open source implementation, leave as default
        self.learn_dendrites_live = False
        add_pai_config_var_functions(
            self, "learn_dendrites_live", self.learn_dendrites_live
        )
        self.no_extra_n_modes = True
        add_pai_config_var_functions(self, "no_extra_n_modes", self.no_extra_n_modes)

        # Data type for new modules and dendrite to dendrite / dendrite to neuron
        # weights
        self.d_type = torch.float
        add_pai_config_var_functions(self, "d_type", self.d_type)

        # Dendrite retention settings
        # A setting to keep dendrites even if they do not improve scores
        self.retain_all_dendrites = False
        add_pai_config_var_functions(
            self, "retain_all_dendrites", self.retain_all_dendrites
        )

        # Learning rate management
        # A setting to automatically sweep over previously used learning rates when
        # adding new dendrites
        # Sometimes it's best to go back to initial LR, but often its best to start
        # at a lower LR
        self.find_best_lr = True
        add_pai_config_var_functions(self, "find_best_lr", self.find_best_lr)
        # Enforces the above even if the previous epoch didn't lower the learning rate
        self.dont_give_up_unless_learning_rate_lowered = True
        add_pai_config_var_functions(
            self,
            "dont_give_up_unless_learning_rate_lowered",
            self.dont_give_up_unless_learning_rate_lowered,
        )

        # Dendrite attempt settings
        # Set to 1 if you want to quit as soon as one dendrite fails
        # Higher values will try new random dendrite weights this many times before
        # accepting that more dendrites don't improve
        self.max_dendrite_tries = 2
        add_pai_config_var_functions(
            self, "max_dendrite_tries", self.max_dendrite_tries
        )
        # Max dendrites to add even if they do continue improving scores
        self.max_dendrites = 100
        add_pai_config_var_functions(self, "max_dendrites", self.max_dendrites)

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
        add_pai_config_var_functions(
            self, "param_vals_setting", self.param_vals_setting
        )

        # Activation function settings
        # The activation function to use for dendrites
        self.pai_forward_function = torch.sigmoid
        add_pai_config_var_functions(
            self, "pai_forward_function", self.pai_forward_function
        )

        # Lists for module types and names to add dendrites to
        # For these lists no specifier means type, name is module name
        # and ids is the individual modules id, eg. model.conv2
        self.modules_to_convert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
        add_pai_config_var_functions(
            self, "modules_to_convert", self.modules_to_convert, list_type=True
        )
        self.module_names_to_convert = ["PAISequential"]
        add_pai_config_var_functions(
            self,
            "module_names_to_convert",
            self.module_names_to_convert,
            list_type=True,
        )
        self.module_ids_to_convert = []
        add_pai_config_var_functions(
            self, "module_ids_to_convert", self.module_ids_to_convert, list_type=True
        )

        # All modules should either be converted or tracked to ensure all modules
        # are accounted for
        self.modules_to_track = []
        add_pai_config_var_functions(
            self, "modules_to_track", self.modules_to_track, list_type=True
        )
        self.module_names_to_track = []
        add_pai_config_var_functions(
            self, "module_names_to_track", self.module_names_to_track, list_type=True
        )
        # IDs are for if you want to pass only a single module by its assigned ID rather than the module type by name
        self.module_ids_to_track = []
        add_pai_config_var_functions(
            self, "module_ids_to_track", self.module_ids_to_track, list_type=True
        )

        # Replacement modules happen before the conversion,
        # so replaced modules will then also be run through the conversion steps
        # These are for modules that need to be replaced before addition of dendrites
        # See the resnet example in models_perforatedai
        self.modules_to_replace = []
        add_pai_config_var_functions(
            self, "modules_to_replace", self.modules_to_replace, list_type=True
        )
        # Modules to replace the above modules with
        self.replacement_modules = []
        add_pai_config_var_functions(
            self, "replacement_modules", self.replacement_modules, list_type=True
        )

        # Dendrites default to modules which are one tensor input and one tensor
        # output in forward()
        # Other modules require to be labeled as modules with processing and assigned
        # processing classes
        # This can be done by module type or module name see customization.md in API
        # for example
        self.modules_with_processing = []
        add_pai_config_var_functions(
            self,
            "modules_with_processing",
            self.modules_with_processing,
            list_type=True,
        )
        self.modules_processing_classes = []
        add_pai_config_var_functions(
            self,
            "modules_processing_classes",
            self.modules_processing_classes,
            list_type=True,
        )
        self.module_names_with_processing = []
        add_pai_config_var_functions(
            self,
            "module_names_with_processing",
            self.module_names_with_processing,
            list_type=True,
        )
        self.module_by_name_processing_classes = []
        add_pai_config_var_functions(
            self,
            "module_by_name_processing_classes",
            self.module_by_name_processing_classes,
            list_type=True,
        )

        # Similarly here as above. Some huggingface models have multiple pointers to
        # the same modules which cause problems
        # If you want to only save one of the multiple pointers you can set which ones
        # not to save here
        self.module_names_to_not_save = [".base_model"]
        add_pai_config_var_functions(
            self,
            "module_names_to_not_save",
            self.module_names_to_not_save,
            list_type=True,
        )

        # Perforated Backpropagation settings
        self.perforated_backpropagation = False
        add_pai_config_var_functions(
            self, "perforated_backpropagation", self.perforated_backpropagation
        )


class PAISequential(nn.Sequential):
    """Sequential module wrapper for PAI.

    This wrapper takes an array of layers and creates a sequential container
    that is compatible with PAI's dendrite addition system. It should be used
    for normalization layers and can be used for final output layers.

    Parameters
    ----------
    layer_array : list
        List of PyTorch nn.Module objects to be executed sequentially.

    Examples
    --------
    >>> layers = [nn.Linear(2 * hidden_dim, seq_width),
    ...           nn.LayerNorm(seq_width)]
    >>> sequential_block = PAISequential(layers)

    Notes
    -----
    This should be used for:
        - All normalization layers (LayerNorm, BatchNorm, etc.)
    This can be used for:
        - Final output layer and softmax combinations
    """

    def __init__(self, layer_array):
        """Initialize PAISequential with a list of layers.

        Parameters
        ----------
        layer_array : list
            List of PyTorch modules to execute in sequence.
        """
        super(PAISequential, self).__init__()
        self.model = nn.Sequential(*layer_array)

    def forward(self, *args, **kwargs):
        """Forward pass through the sequential layers.

        Parameters
        ----------
        *args
            Positional arguments passed to the first layer.
        **kwargs
            Keyword arguments passed to the layers.

        Returns
        -------
        torch.Tensor
            Output from the final layer in the sequence.
        """
        return self.model(*args, **kwargs)


### Global objects and variables

### Global Modules
pc = PAIConfig()
"""Global PAIConfig instance.

This is the primary configuration object used throughout the PAI system.
Modify settings through this instance to control PAI behavior.
"""

"""Pointer to the PAI Tracker.

This will be populated with the PAI Tracker instance which handles
the addition of dendrites during training. Initially an empty list.
"""
pai_tracker = []

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
            add_pai_config_var_functions(
                pc, var_name, getattr(perforatedbp_globals, var_name)
            )

except ImportError:
    print("Building dendrites without Perforated Backpropagation")
