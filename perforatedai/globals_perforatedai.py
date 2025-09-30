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

### Global Constants


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
        self.switch_mode = self.DOING_HISTORY

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
        self.pai_forward_function = torch.sigmoid

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

    def set_use_cuda(self, value):
        """Set whether to use CUDA for computation.

        Parameters
        ----------
        value : bool
            True to use CUDA if available, False to use CPU.

        Returns
        -------
        None
        """
        self.use_cuda = value

    def get_use_cuda(self):
        """Get whether CUDA is being used.

        Returns
        -------
        bool
            True if CUDA is enabled, False otherwise.
        """
        return self.use_cuda

    # device
    def set_device(self, value):
        """Set the computation device.

        Parameters
        ----------
        value : torch.device
            The device to use for computation (e.g., torch.device('cuda:0')).

        Returns
        -------
        None
        """
        self.device = value

    def get_device(self):
        """Get the current computation device.

        Returns
        -------
        torch.device
            The device being used for computation.
        """
        return self.device

    def get_save_name(self):
        """Get the model save name.

        Returns
        -------
        str
            The name used for saving models.
        """
        return self.save_name

    # debugging_input_dimensions
    def set_debugging_input_dimensions(self, value):
        """Set the input dimensions debugging level.

        Parameters
        ----------
        value : int
            Debug level for input dimension checking.

        Returns
        -------
        None
        """
        self.debugging_input_dimensions = value

    def get_debugging_input_dimensions(self):
        """Get the input dimensions debugging level.

        Returns
        -------
        int
            Current debug level for input dimensions.

        Notes
        -----
        Should only be set to 0 or 1 manually
            - 0: Not debugging
            - 1: initialize debugging to step 1
        """
        return self.debugging_input_dimensions

    # confirm_correct_sizes
    def set_confirm_correct_sizes(self, value):
        """Set whether to confirm tensor sizes during execution.

        Parameters
        ----------
        value : bool
            True to enable size verification, False to disable.

        Returns
        -------
        None
        """
        self.confirm_correct_sizes = value

    def get_confirm_correct_sizes(self):
        """Get whether tensor size confirmation is enabled.

        Returns
        -------
        bool
            True if size verification is enabled, False otherwise.
        """
        return self.confirm_correct_sizes

    # unwrapped_modules_confirmed
    def set_unwrapped_modules_confirmed(self, value):
        """Set unwrapped modules confirmation flag.

        Parameters
        ----------
        value : bool
            True to confirm use of unwrapped modules.

        Returns
        -------
        None
        """
        self.unwrapped_modules_confirmed = value

    def get_unwrapped_modules_confirmed(self):
        """Get unwrapped modules confirmation status.

        Returns
        -------
        bool
            True if unwrapped modules are confirmed, False otherwise.
        """
        return self.unwrapped_modules_confirmed

    # weight_decay_accepted
    def set_weight_decay_accepted(self, value):
        """Set weight decay acceptance flag.

        Parameters
        ----------
        value : bool
            True to accept weight decay usage.

        Returns
        -------
        None
        """
        self.weight_decay_accepted = value

    def get_weight_decay_accepted(self):
        """Get weight decay acceptance status.

        Returns
        -------
        bool
            True if weight decay is accepted, False otherwise.
        """
        return self.weight_decay_accepted

    # checked_skipped_modules
    def set_checked_skipped_modules(self, value):
        """Set whether skipped modules have been checked.

        Parameters
        ----------
        value : bool
            True if skipped modules have been verified.

        Returns
        -------
        None
        """
        self.checked_skipped_modules = value

    def get_checked_skipped_modules(self):
        """Get whether skipped modules have been checked.

        Returns
        -------
        bool
            True if skipped modules have been verified, False otherwise.
        """
        return self.checked_skipped_modules

    # verbose
    def set_verbose(self, value):
        """Set verbose logging mode.

        Parameters
        ----------
        value : bool
            True to enable verbose output, False to disable.

        Returns
        -------
        None
        """
        self.verbose = value

    def get_verbose(self):
        """Get verbose logging status.

        Returns
        -------
        bool
            True if verbose mode is enabled, False otherwise.
        """
        return self.verbose

    # extra_verbose
    def set_extra_verbose(self, value):
        """Set extra verbose logging mode.

        Parameters
        ----------
        value : bool
            True to enable extra verbose output, False to disable.

        Returns
        -------
        None
        """
        self.extra_verbose = value

    def get_extra_verbose(self):
        """Get extra verbose logging status.

        Returns
        -------
        bool
            True if extra verbose mode is enabled, False otherwise.
        """
        return self.extra_verbose

    # silent
    def set_silent(self, value):
        """Set silent mode to suppress all PAI output.

        Parameters
        ----------
        value : bool
            True to suppress all output, False for normal output.

        Returns
        -------
        None
        """
        self.silent = value

    def get_silent(self):
        """Get silent mode status.

        Returns
        -------
        bool
            True if silent mode is enabled, False otherwise.
        """
        return self.silent

    # save_old_graph_scores
    def set_save_old_graph_scores(self, value):
        """Set whether to save historical graph scores.

        Parameters
        ----------
        value : bool
            True to save old scores, False to discard them.

        Returns
        -------
        None
        """
        self.save_old_graph_scores = value

    def get_save_old_graph_scores(self):
        """Get whether historical graph scores are saved.

        Returns
        -------
        bool
            True if old scores are saved, False otherwise.
        """
        return self.save_old_graph_scores

    # testing_dendrite_capacity
    def set_testing_dendrite_capacity(self, value):
        """Set dendrite capacity testing mode.

        Parameters
        ----------
        value : bool
            True to enable capacity testing, False to disable.

        Returns
        -------
        None
        """
        self.testing_dendrite_capacity = value

    def get_testing_dendrite_capacity(self):
        """Get dendrite capacity testing status.

        Returns
        -------
        bool
            True if capacity testing is enabled, False otherwise.
        """
        return self.testing_dendrite_capacity

    # using_safe_tensors
    def set_using_safe_tensors(self, value):
        """Set whether to use safe tensors file format.

        Parameters
        ----------
        value : bool
            True to use safe tensors format, False for standard format.

        Returns
        -------
        None
        """
        self.using_safe_tensors = value

    def get_using_safe_tensors(self):
        """Get whether safe tensors format is being used.

        Returns
        -------
        bool
            True if safe tensors are enabled, False otherwise.
        """
        return self.using_safe_tensors

    # global_candidates
    def set_global_candidates(self, value):
        """Set the number of global candidate dendrites.

        Parameters
        ----------
        value : int
            Number of global candidates to use.

        Returns
        -------
        None
        """
        self.global_candidates = value

    def get_global_candidates(self):
        """Get the number of global candidate dendrites.

        Returns
        -------
        int
            Number of global candidates configured.
        """
        return self.global_candidates

    # drawing_pai
    def set_drawing_pai(self, value):
        """Set whether to enable PAI visualization graphs.

        Parameters
        ----------
        value : bool
            True to enable graph drawing, False to disable.

        Returns
        -------
        None
        """
        self.drawing_pai = value

    def get_drawing_pai(self):
        """Get whether PAI visualization is enabled.

        Returns
        -------
        bool
            True if graph drawing is enabled, False otherwise.
        """
        return self.drawing_pai

    # test_saves
    def set_test_saves(self, value):
        """Set whether to save intermediary test models.

        Parameters
        ----------
        value : bool
            True to save test models, False to skip.

        Returns
        -------
        None
        """
        self.test_saves = value

    def get_test_saves(self):
        """Get whether test models are being saved.

        Returns
        -------
        bool
            True if test saves are enabled, False otherwise.
        """
        return self.test_saves

    # pai_saves
    def set_pai_saves(self, value):
        """Set whether to use PAI-specific save format.

        Parameters
        ----------
        value : bool
            True to use PAI save format with optimizations.

        Returns
        -------
        None
        """
        self.pai_saves = value

    def get_pai_saves(self):
        """Get whether PAI-specific saves are enabled.

        Returns
        -------
        bool
            True if PAI saves are enabled, False otherwise.
        """
        return self.pai_saves

    # input_dimensions
    def set_input_dimensions(self, value):
        """Set the input tensor dimensions format.

        Parameters
        ----------
        value : list of int
            Dimension format specification. Use 0 for neuron index,
            -1 for variable dimensions. Example: [-1, 0, -1, -1] for
            [batch_size, neurons, height, width].

        Returns
        -------
        None
        """
        self.input_dimensions = value

    def get_input_dimensions(self):
        """Get the input tensor dimensions format.

        Returns
        -------
        list of int
            Current dimension format specification.
        """
        return self.input_dimensions

    # improvement_threshold
    def set_improvement_threshold(self, value):
        """Set the relative improvement threshold for validation scores.

        Parameters
        ----------
        value : float
            Percentage improvement needed to declare a new best score.

        Returns
        -------
        None
        """
        self.improvement_threshold = value

    def get_improvement_threshold(self):
        """Get the relative improvement threshold.

        Returns
        -------
        float
            Current percentage improvement threshold.
        """
        return self.improvement_threshold

    # improvement_threshold_raw
    def set_improvement_threshold_raw(self, value):
        """Set the absolute improvement threshold for validation scores.

        Parameters
        ----------
        value : float
            Absolute improvement needed to declare a new best score.

        Returns
        -------
        None
        """
        self.improvement_threshold_raw = value

    def get_improvement_threshold_raw(self):
        """Get the absolute improvement threshold.

        Returns
        -------
        float
            Current absolute improvement threshold.
        """
        return self.improvement_threshold_raw

    # candidate_weight_initialization_multiplier
    def set_candidate_weight_initialization_multiplier(self, value):
        """Set the multiplier for random dendrite weight initialization.

        Parameters
        ----------
        value : float
            Multiplier applied to random weights during initialization.

        Returns
        -------
        None
        """
        self.candidate_weight_initialization_multiplier = value

    def get_candidate_weight_initialization_multiplier(self):
        """Get the dendrite weight initialization multiplier.

        Returns
        -------
        float
            Current weight initialization multiplier.
        """
        return self.candidate_weight_initialization_multiplier

    # n_epochs_to_switch
    def set_n_epochs_to_switch(self, value):
        """Set epochs to wait before switching when validation doesn't improve.

        Parameters
        ----------
        value : int
            Number of epochs without improvement before adding dendrites.

        Returns
        -------
        None
        """
        self.n_epochs_to_switch = value

    def get_n_epochs_to_switch(self):
        """Get the number of epochs before switching.

        Returns
        -------
        int
            Epochs to wait before adding dendrites.
        """
        return self.n_epochs_to_switch

    # history_lookback
    def set_history_lookback(self, value):
        """Set the number of epochs to average for validation history.

        Parameters
        ----------
        value : int
            Number of epochs to include in validation average.

        Returns
        -------
        None
        """
        self.history_lookback = value

    def get_history_lookback(self):
        """Get the history lookback period.

        Returns
        -------
        int
            Number of epochs in validation history average.
        """
        return self.history_lookback

    # initial_history_after_switches
    def set_initial_history_after_switches(self, value):
        """Set epochs to wait after adding dendrites before checking again.

        Parameters
        ----------
        value : int
            Number of epochs to wait after switch before next check.

        Returns
        -------
        None
        """
        self.initial_history_after_switches = value

    def get_initial_history_after_switches(self):
        """Get the post-switch waiting period.

        Returns
        -------
        int
            Epochs to wait after adding dendrites.
        """
        return self.initial_history_after_switches

    # fixed_switch_num
    def set_fixed_switch_num(self, value):
        """Set the fixed number of epochs between switches.

        Parameters
        ----------
        value : int
            Number of epochs to complete before each switch.

        Returns
        -------
        None
        """
        self.fixed_switch_num = value

    def get_fixed_switch_num(self):
        """Get the fixed switch interval.

        Returns
        -------
        int
            Number of epochs between switches.
        """
        return self.fixed_switch_num

    # first_fixed_switch_num
    def set_first_fixed_switch_num(self, value):
        """Set epochs before the first switch (for pretraining).

        Parameters
        ----------
        value : int
            Number of epochs before first switch occurs.

        Returns
        -------
        None
        """
        self.first_fixed_switch_num = value

    def get_first_fixed_switch_num(self):
        """Get the first switch timing.

        Returns
        -------
        int
            Epochs before first switch.
        """
        return self.first_fixed_switch_num

    # switch_mode
    def set_switch_mode(self, value):
        """Set the dendrite addition switch mode.

        Parameters
        ----------
        value : int
            Switch mode constant (DOING_HISTORY, DOING_FIXED_SWITCH,
            DOING_SWITCH_EVERY_TIME, or DOING_NO_SWITCH).

        Returns
        -------
        None
        """
        self.switch_mode = value

    def get_switch_mode(self):
        """Get the current switch mode.

        Returns
        -------
        int
            Current switch mode constant.
        """
        return self.switch_mode

    # reset_best_score_on_switch
    def set_reset_best_score_on_switch(self, value):
        """Set whether to reset best score when adding dendrites.

        Parameters
        ----------
        value : bool
            True to reset score on switch, False to keep it.

        Returns
        -------
        None
        """
        self.reset_best_score_on_switch = value

    def get_reset_best_score_on_switch(self):
        """Get whether best score is reset on switch.

        Returns
        -------
        bool
            True if score resets on switch, False otherwise.
        """
        return self.reset_best_score_on_switch

    # learn_dendrites_live
    def set_learn_dendrites_live(self, value):
        """Set whether to enable live dendrite learning (advanced feature).

        Parameters
        ----------
        value : bool
            True to enable live learning, False to disable.

        Returns
        -------
        None
        """
        self.learn_dendrites_live = value

    def get_learn_dendrites_live(self):
        """Get whether live dendrite learning is enabled.

        Returns
        -------
        bool
            True if live learning is enabled, False otherwise.
        """
        return self.learn_dendrites_live

    # no_extra_n_modes
    def set_no_extra_n_modes(self, value):
        """Set whether to disable extra neuron modes (advanced feature).

        Parameters
        ----------
        value : bool
            True to disable extra modes, False to enable.

        Returns
        -------
        None
        """
        self.no_extra_n_modes = value

    def get_no_extra_n_modes(self):
        """Get whether extra neuron modes are disabled.

        Returns
        -------
        bool
            True if extra modes are disabled, False otherwise.
        """
        return self.no_extra_n_modes

    # d_type
    def set_d_type(self, value):
        """Set the data type for dendrite weights.

        Parameters
        ----------
        value : torch.dtype
            PyTorch data type for dendrite tensors.

        Returns
        -------
        None
        """
        self.d_type = value

    def get_d_type(self):
        """Get the dendrite weight data type.

        Returns
        -------
        torch.dtype
            Current data type for dendrite weights.
        """
        return self.d_type

    # retain_all_dendrites
    def set_retain_all_dendrites(self, value):
        """Set whether to keep all dendrites regardless of performance.

        Parameters
        ----------
        value : bool
            True to keep all dendrites, False to remove non-improving ones.

        Returns
        -------
        None
        """
        self.retain_all_dendrites = value

    def get_retain_all_dendrites(self):
        """Get whether all dendrites are retained.

        Returns
        -------
        bool
            True if all dendrites are kept, False otherwise.
        """
        return self.retain_all_dendrites

    # find_best_lr
    def set_find_best_lr(self, value):
        """Set whether to automatically sweep learning rates when adding dendrites.

        Parameters
        ----------
        value : bool
            True to enable automatic LR sweeping, False to disable.

        Returns
        -------
        None
        """
        self.find_best_lr = value

    def get_find_best_lr(self):
        """Get whether automatic LR sweeping is enabled.

        Returns
        -------
        bool
            True if LR sweeping is enabled, False otherwise.
        """
        return self.find_best_lr

    # dont_give_up_unless_learning_rate_lowered
    def set_dont_give_up_unless_learning_rate_lowered(self, value):
        """Set whether to enforce LR sweep even if previous epoch didn't lower LR.

        Parameters
        ----------
        value : bool
            True to enforce LR sweep, False for normal behavior.

        Returns
        -------
        None
        """
        self.dont_give_up_unless_learning_rate_lowered = value

    def get_dont_give_up_unless_learning_rate_lowered(self):
        """Get whether LR sweep enforcement is enabled.

        Returns
        -------
        bool
            True if LR sweep is enforced, False otherwise.
        """
        return self.dont_give_up_unless_learning_rate_lowered

    # max_dendrite_tries
    def set_max_dendrite_tries(self, value):
        """Set maximum attempts to add dendrites with random initializations.

        Parameters
        ----------
        value : int
            Maximum number of random initialization attempts.

        Returns
        -------
        None
        """
        self.max_dendrite_tries = value

    def get_max_dendrite_tries(self):
        """Get the maximum dendrite addition attempts.

        Returns
        -------
        int
            Maximum number of tries with random weights.
        """
        return self.max_dendrite_tries

    # max_dendrites
    def set_max_dendrites(self, value):
        """Set maximum total number of dendrites to add.

        Parameters
        ----------
        value : int
            Maximum total dendrites allowed.

        Returns
        -------
        None
        """
        self.max_dendrites = value

    def get_max_dendrites(self):
        """Get the maximum number of dendrites.

        Returns
        -------
        int
            Maximum total dendrites that can be added.
        """
        return self.max_dendrites

    # param_vals_setting
    def set_param_vals_setting(self, value):
        """Set the scheduler parameter tracking mode.

        Parameters
        ----------
        value : int
            Parameter tracking mode (PARAM_VALS_BY_TOTAL_EPOCH,
            PARAM_VALS_BY_UPDATE_EPOCH, or PARAM_VALS_BY_NEURON_EPOCH_START).

        Returns
        -------
        None
        """
        self.param_vals_setting = value

    def get_param_vals_setting(self):
        """Get the current parameter tracking mode.

        Returns
        -------
        int
            Current scheduler parameter tracking mode.
        """
        return self.param_vals_setting

    # pai_forward_function
    def set_pai_forward_function(self, value):
        """Set the activation function for dendrites.

        Parameters
        ----------
        value : callable
            Activation function (e.g., torch.sigmoid, torch.relu).

        Returns
        -------
        None
        """
        self.pai_forward_function = value

    def get_pai_forward_function(self):
        """Get the dendrite activation function.

        Returns
        -------
        callable
            Current activation function for dendrites.
        """
        return self.pai_forward_function

    # modules_to_convert
    def set_modules_to_convert(self, value):
        """Set the list of module types to convert to PAI modules.

        Parameters
        ----------
        value : list
            List of PyTorch module types (e.g., [nn.Linear, nn.Conv2d]).

        Returns
        -------
        None
        """
        self.modules_to_convert = value

    def get_modules_to_convert(self):
        """Get the list of module types to convert.

        Returns
        -------
        list
            Module types that will be converted to PAI modules.
        """
        return self.modules_to_convert

    def append_modules_to_convert(self, value):
        """Append module types to the conversion list.

        Parameters
        ----------
        value : list
            Module types to add to the conversion list.

        Returns
        -------
        None
        """
        self.modules_to_convert += value

    # module_names_to_convert
    def set_module_names_to_convert(self, value):
        """Set the list of module names to convert to PAI modules.

        Parameters
        ----------
        value : list of str
            List of module class names (e.g., ["PAISequential"]).

        Returns
        -------
        None
        """
        self.module_names_to_convert = value

    def get_module_names_to_convert(self):
        """Get the list of module names to convert.

        Returns
        -------
        list of str
            Module names that will be converted to PAI modules.
        """
        return self.module_names_to_convert

    def append_module_names_to_convert(self, value):
        """Append module names to the conversion list.

        Parameters
        ----------
        value : list of str
            Module names to add to the conversion list.

        Returns
        -------
        None
        """
        self.module_names_to_convert += value

    # module_ids_to_convert
    def set_module_ids_to_convert(self, value):
        """Set the list of specific module IDs to convert.

        Parameters
        ----------
        value : list of str
            List of module IDs (e.g., ["model.conv2", "model.layer1"]).

        Returns
        -------
        None
        """
        self.module_ids_to_convert = value

    def get_module_ids_to_convert(self):
        """Get the list of module IDs to convert.

        Returns
        -------
        list of str
            Specific module IDs that will be converted.
        """
        return self.module_ids_to_convert

    def append_module_ids_to_convert(self, value):
        """Append module IDs to the conversion list.

        Parameters
        ----------
        value : list of str
            Module IDs to add to the conversion list.

        Returns
        -------
        None
        """
        self.module_ids_to_convert += value

    # modules_to_track
    def set_modules_to_track(self, value):
        """Set the list of module types to track but not convert.

        Parameters
        ----------
        value : list
            List of PyTorch module types to track.

        Returns
        -------
        None
        """
        self.modules_to_track = value

    def get_modules_to_track(self):
        """Get the list of module types being tracked.

        Returns
        -------
        list
            Module types that are tracked but not converted.
        """
        return self.modules_to_track

    def append_modules_to_track(self, value):
        """Append module types to the tracking list.

        Parameters
        ----------
        value : list
            Module types to add to the tracking list.

        Returns
        -------
        None
        """
        self.modules_to_track += value

    # module_names_to_track
    def set_module_names_to_track(self, value):
        """Set the list of module names to track but not convert.

        Parameters
        ----------
        value : list of str
            List of module class names to track.

        Returns
        -------
        None
        """
        self.module_names_to_track = value

    def get_module_names_to_track(self):
        """Get the list of module names being tracked.

        Returns
        -------
        list of str
            Module names that are tracked but not converted.
        """
        return self.module_names_to_track

    def append_module_names_to_track(self, value):
        """Append module names to the tracking list.

        Parameters
        ----------
        value : list of str
            Module names to add to the tracking list.

        Returns
        -------
        None
        """
        self.module_names_to_track += value

    # module_ids_to_track
    def set_module_ids_to_track(self, value):
        """Set the list of specific module IDs to track.

        Parameters
        ----------
        value : list of str
            List of module IDs to track.

        Returns
        -------
        None
        """
        self.module_ids_to_track = value

    def get_module_ids_to_track(self):
        """Get the list of module IDs being tracked.

        Returns
        -------
        list of str
            Specific module IDs that are tracked.
        """
        return self.module_ids_to_track

    def append_module_ids_to_track(self, value):
        """Append module IDs to the tracking list.

        Parameters
        ----------
        value : list of str
            Module IDs to add to the tracking list.

        Returns
        -------
        None
        """
        self.module_ids_to_track += value

    # modules_to_replace
    def set_modules_to_replace(self, value):
        """Set the list of module types to replace before conversion.

        Parameters
        ----------
        value : list
            List of PyTorch module types to replace.

        Returns
        -------
        None
        """
        self.modules_to_replace = value

    def get_modules_to_replace(self):
        """Get the list of module types to be replaced.

        Returns
        -------
        list
            Module types that will be replaced before conversion.
        """
        return self.modules_to_replace

    def append_modules_to_replace(self, value):
        """Append module types to the replacement list.

        Parameters
        ----------
        value : list
            Module types to add to the replacement list.

        Returns
        -------
        None
        """
        self.modules_to_replace += value

    # replacement_modules
    def set_replacement_modules(self, value):
        """Set the list of replacement modules.

        Parameters
        ----------
        value : list
            List of module types to use as replacements.

        Returns
        -------
        None
        """
        self.replacement_modules = value

    def get_replacement_modules(self):
        """Get the list of replacement modules.

        Returns
        -------
        list
            Module types used as replacements.
        """
        return self.replacement_modules

    def append_replacement_modules(self, value):
        """Append replacement modules to the list.

        Parameters
        ----------
        value : list
            Replacement modules to add to the list.

        Returns
        -------
        None
        """
        self.replacement_modules += value

    # modules_with_processing
    def set_modules_with_processing(self, value):
        """Set the list of module types requiring custom processing.

        Parameters
        ----------
        value : list
            List of module types that need custom processing.

        Returns
        -------
        None
        """
        self.modules_with_processing = value

    def get_modules_with_processing(self):
        """Get the list of modules with custom processing.

        Returns
        -------
        list
            Module types that require custom processing.
        """
        return self.modules_with_processing

    def append_modules_with_processing(self, value):
        """Append module types to the custom processing list.

        Parameters
        ----------
        value : list
            Module types to add to the processing list.

        Returns
        -------
        None
        """
        self.modules_with_processing += value

    # modules_processing_classes
    def set_modules_processing_classes(self, value):
        """Set the list of processing classes for custom modules.

        Parameters
        ----------
        value : list
            List of processing class types.

        Returns
        -------
        None
        """
        self.modules_processing_classes = value

    def get_modules_processing_classes(self):
        """Get the list of processing classes.

        Returns
        -------
        list
            Processing classes for custom modules.
        """
        return self.modules_processing_classes

    def append_modules_processing_classes(self, value):
        """Append processing classes to the list.

        Parameters
        ----------
        value : list
            Processing classes to add.

        Returns
        -------
        None
        """
        self.modules_processing_classes += value

    # module_names_with_processing
    def set_module_names_with_processing(self, value):
        """Set the list of module names requiring custom processing.

        Parameters
        ----------
        value : list of str
            List of module names that need custom processing.

        Returns
        -------
        None
        """
        self.module_names_with_processing = value

    def get_module_names_with_processing(self):
        """Get the list of module names with custom processing.

        Returns
        -------
        list of str
            Module names that require custom processing.
        """
        return self.module_names_with_processing

    def append_module_names_with_processing(self, value):
        """Append module names to the custom processing list.

        Parameters
        ----------
        value : list of str
            Module names to add to the processing list.

        Returns
        -------
        None
        """
        self.module_names_with_processing += value

    # module_by_name_processing_classes
    def set_module_by_name_processing_classes(self, value):
        """Set the list of processing classes for named modules.

        Parameters
        ----------
        value : list
            List of processing class types for named modules.

        Returns
        -------
        None
        """
        self.module_by_name_processing_classes = value

    def get_module_by_name_processing_classes(self):
        """Get the list of processing classes for named modules.

        Returns
        -------
        list
            Processing classes for named modules.
        """
        return self.module_by_name_processing_classes

    def append_module_by_name_processing_classes(self, value):
        """Append processing classes for named modules.

        Parameters
        ----------
        value : list
            Processing classes to add for named modules.

        Returns
        -------
        None
        """
        self.module_by_name_processing_classes += value

    # module_names_to_not_save
    def set_module_names_to_not_save(self, value):
        """Set the list of module names to exclude from saving.

        Parameters
        ----------
        value : list of str
            List of module names to skip during save operations.

        Returns
        -------
        None
        """
        self.module_names_to_not_save = value

    def get_module_names_to_not_save(self):
        """Get the list of module names excluded from saving.

        Returns
        -------
        list of str
            Module names that won't be saved.
        """
        return self.module_names_to_not_save

    def append_module_names_to_not_save(self, value):
        """Append module names to the exclusion list.

        Parameters
        ----------
        value : list of str
            Module names to add to the exclusion list.

        Returns
        -------
        None
        """
        self.module_names_to_not_save += value

    # perforated_backpropagation
    def set_perforated_backpropagation(self, value):
        """Set whether Perforated Backpropagation is enabled.

        Parameters
        ----------
        value : bool
            True to enable Perforated Backpropagation, False to disable.

        Returns
        -------
        None
        """
        self.perforated_backpropagation = value

    def get_perforated_backpropagation(self):
        """Get whether Perforated Backpropagation is enabled.

        Returns
        -------
        bool
            True if Perforated Backpropagation is enabled, False otherwise.
        """
        return self.perforated_backpropagation


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


def add_pbp_var(obj, var_name, initial_value):
    """Dynamically add a property with getter and setter to an object.

    This function adds a private variable along with getter and setter methods
    to a given object instance. Used for integrating Perforated Backpropagation
    variables into the PAIConfig class.

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
