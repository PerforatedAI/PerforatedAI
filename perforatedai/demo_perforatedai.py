# Copyright (c) 2025 Perforated AI

"""Demo mode: add one dendrite to a model with a single call to perforate_model.

    model = UPA.perforate_model(
        model,
        demo_mode=True,
        demo_phase_steps=EPOCHS * len(train_loader),
        demo_sample_input=next(iter(train_loader))[0],
    )

No other PAI calls are needed (no setup_optimizer, set_optimizer_instance or
add_validation_score). Demo mode works like this:

1. One Linear or Conv2d layer near the output gets a dendrite. Every other
   module with parameters is tracked.
2. The dendrite is created right away by UPA.initialize_dendrites, which runs
   the sample batch forward and backward once. Its output weights start at zero.
3. The optimizer and schedulers the user creates afterwards are captured by
   wrapping their constructors. Right after the optimizer is built the
   dendrite is frozen, so the optimizer includes it like any other parameter
   and phase 1 is the user's normal training run.
4. After demo_phase_steps optimizer steps the dendrite is unfrozen, and the
   optimizer and every scheduler on it restart at LR_RESTART_FACTOR times their
   starting learning rates. Phase 2 replays the original learning rate schedule.

The user's script should run twice its usual number of epochs. Demo mode only
uses open source dendrites (no Perforated Backpropagation).
"""

import copy

import torch
import torch.nn as nn

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Learning rates restart at this fraction of their starting values when the dendrite turns on
LR_RESTART_FACTOR = 0.25
# The output layer gets the dendrite unless it holds at least this share of the total network parameters
OUTPUT_LAYER_MAX_SHARE = 0.25
# Param group entries restored at each restart; the LR_KEYS ones are also scaled
LR_KEYS = ("lr", "initial_lr", "max_lr", "min_lr")
OTHER_RESTORED_KEYS = ("momentum", "betas")


class DemoState:
    """Everything demo mode remembers between perforate_model and the phase switch."""

    def __init__(self):
        self.phase_steps = 0
        self.target_name = ""
        self.target_params = 0
        self.total_params = 0
        self.model = None
        # The single PAINeuronModule and the ids of its own (neuron) parameters
        self.module = None
        self.neuron_ids = set()
        # The dendrite's parameters, which start training after the switch
        self.trainable = []
        self.optimizer = None
        self.optimizer_initial_state = {}
        self.optimizers_seen = 0
        # Schedulers created since the last optimizer step
        self.new_schedulers = []
        # (scheduler, state_dict) taken before the first optimizer step
        self.scheduler_snapshots = []
        # Param group learning rates taken before the first optimizer step
        self.group_snapshot = None
        self.steps_this_phase = 0
        self.phase = 1


state = DemoState()
# The optimizer and scheduler constructors are wrapped once per process
wrappers_installed = False


def setup_demo_mode(model, phase_steps, sample_input):
    """Configure PAI to perforate a single module. Runs before the model is converted.

    Parameters
    ----------
    model : nn.Module
        The unconverted model passed to perforate_model.
    phase_steps : int
        Optimizer steps in the user's usual training run.
    sample_input : torch.Tensor
        One batch of the model's inputs, checked here and used by start_demo_mode.

    Returns
    -------
    None
    """
    # Forget any earlier demo run in this process, e.g. a re-run notebook cell
    state.__init__()
    if phase_steps is None or int(phase_steps) <= 0:
        raise ValueError(
            "demo_mode needs demo_phase_steps: the number of optimizer steps in "
            "your usual training run, e.g. demo_phase_steps=EPOCHS * len(train_loader)"
        )
    if not torch.is_tensor(sample_input):
        raise ValueError(
            "demo_mode needs demo_sample_input: one batch of your model's inputs, "
            "e.g. demo_sample_input=next(iter(train_loader))[0]"
        )
    if not hasattr(torch.optim.Optimizer, "register_step_pre_hook"):
        raise RuntimeError("demo_mode needs PyTorch 2.0 or newer")
    if GPA.pc.get_perforated_backpropagation():
        print(
            "PAI demo mode uses gradient descent dendrites, so Perforated Backpropagation "
            "is turned off for this run."
        )
        GPA.pc.set_perforated_backpropagation(False)
    state.phase_steps = int(phase_steps)

    # New dendrite tensors are created on PAI's device and dtype, so match the model
    first_param = next(model.parameters())
    GPA.pc.set_device(first_param.device)
    GPA.pc.set_d_type(first_param.dtype)
    # Skip the interactive target picker and the 3-dendrite capacity test
    GPA.pc.set_configuration_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(False)

    target = choose_module_to_perforate(model)
    set_conversion_lists(model, target)
    state.target_name = target
    state.target_params = count_parameters(model.get_submodule(target[1:]))
    state.total_params = count_parameters(model)


def count_parameters(module):
    """Number of parameters in a module (shared parameters counted once)."""
    return sum(param.numel() for param in module.parameters())


def choose_module_to_perforate(model):
    """Return the module id (e.g. ".fc2") of the layer that gets the dendrite.

    This is the last Linear or Conv2d layer, unless that layer holds at least
    OUTPUT_LAYER_MAX_SHARE of the parameters (e.g. a large vocabulary head), in
    which case it is the Linear or Conv2d layer before it.

    Parameters
    ----------
    model : nn.Module
        The unconverted model.

    Returns
    -------
    str
        Module id of the chosen layer.
    """
    # 1. List every layer that can get a dendrite: all nn.Linear and nn.Conv2d
    # layers in the model.
    #
    # One exception: nn.MultiheadAttention contains a Linear called out_proj, but
    # MultiheadAttention.forward() reads out_proj.weight directly and never calls
    # out_proj(...). A dendrite on out_proj would never run, so skip any layer
    # whose parent is a MultiheadAttention. `parents` maps each module to the
    # module that contains it so we can check that.
    parents = {
        child: parent for parent in model.modules() for child in parent.children()
    }
    # named_modules() lists modules in the order they were created in __init__.
    # Output layers are almost always created last, so candidates[-1] is taken to
    # be the output layer and candidates[-2] the layer before it.
    candidates = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
        and not isinstance(parents.get(module), nn.MultiheadAttention)
    ]
    if len(candidates) == 0:
        raise ValueError(
            "Demo mode needs an nn.Linear or nn.Conv2d layer to add a dendrite to"
        )

    # 2. Use the output layer unless it is a large share of the whole model.
    # A dendrite is a copy of the layer it is added to, so it adds as many
    # parameters as that layer has. If the output layer holds at least
    # OUTPUT_LAYER_MAX_SHARE of all parameters (e.g. a language model's
    # vocabulary head), use the layer before it instead. count_parameters counts
    # shared weights once, so a head that shares its weight with the embedding
    # counts that weight as part of the head and once in the total.
    output_layer = model.get_submodule(candidates[-1])
    output_share = count_parameters(output_layer) / count_parameters(model)
    # Module ids are named_modules() names with a leading "." ("fc2" -> ".fc2")
    if output_share >= OUTPUT_LAYER_MAX_SHARE and len(candidates) > 1:
        return "." + candidates[-2]
    # The output layer is small enough, or it is the only Linear/Conv2d layer
    return "." + candidates[-1]


def set_conversion_lists(model, target):
    """Perforate only `target` and track every other module that has parameters.

    Modules on the path from the model down to `target` are not tracked, since
    that would hide `target` from the converter. Parameters those modules own
    directly (e.g. a cls_token on the model itself) are tracked by id instead.

    Parameters
    ----------
    model : nn.Module
        The unconverted model.
    target : str
        Module id of the layer that gets the dendrite.

    Returns
    -------
    None
    """
    path = {target[:i] for i in range(1, len(target)) if target[i] == "."}
    tracked_modules = []
    tracked_parameters = []

    def visit(module, module_id):
        for name, _ in module.named_parameters(recurse=False):
            tracked_parameters.append(module_id + "." + name)
        for name, child in module.named_children():
            child_id = module_id + "." + name
            if child_id == target:
                continue
            if child_id in path:
                visit(child, child_id)
            elif isinstance(child, (nn.ParameterList, nn.ParameterDict)):
                # Wrapping these would break iterating over them, so track their entries
                for param_name, _ in child.named_parameters():
                    tracked_parameters.append(child_id + "." + param_name)
            elif any(True for _ in child.parameters()):
                tracked_modules.append(child_id)

    visit(model, "")
    GPA.pc.set_modules_to_perforate([])
    GPA.pc.set_module_names_to_perforate([])
    GPA.pc.set_modules_to_track([])
    GPA.pc.set_module_names_to_track([])
    GPA.pc.set_module_ids_to_perforate([target])
    GPA.pc.set_module_ids_to_track(tracked_modules)
    GPA.pc.set_parameter_ids_to_track(tracked_parameters)


def start_demo_mode(model, sample_input):
    """Create the dendrite and start watching for the optimizer and schedulers.

    Runs after perforate_model has converted the model.

    Parameters
    ----------
    model : nn.Module
        The converted model.
    sample_input : torch.Tensor
        One batch of the model's inputs.

    Returns
    -------
    None
    """
    pai_modules = UPA.get_pai_modules(model, 0)
    if len(pai_modules) != 1:
        raise RuntimeError(
            "Demo mode expected one perforated module but found %d" % len(pai_modules)
        )
    state.model = model
    state.module = pai_modules[0]
    state.neuron_ids = {id(param) for param in state.module.main_module.parameters()}
    if GPA.pai_tracker.member_vars["doing_pai"]:
        if isinstance(state.module.main_module, nn.Linear):
            state.module.pai_dimensions_check = state.module.register_forward_pre_hook(
                set_linear_output_dimensions
            )
        UPA.initialize_dendrites(model, sample_input)
        remember_dendrite_params()
    install_patches()
    state.module.pai_optimizer_check = state.module.register_forward_pre_hook(
        check_optimizer_was_created
    )
    print_start_message()


def set_linear_output_dimensions(module, args):
    """One-time hook: match a perforated Linear layer's output dimensions to its input.

    A Linear layer's outputs are on the last dimension of whatever it is given,
    so a layer that sees (batch, features) needs [-1, 0] and one that sees
    (batch, tokens, features), as in a transformer, needs [-1, -1, 0]. This runs
    during initialize_dendrites' sample batch, before filter_backward checks the
    output dimensions against the gradient.
    """
    module.pai_dimensions_check.remove()
    del module.pai_dimensions_check
    rank = args[0].dim()
    module.set_this_output_dimensions([-1] * (rank - 1) + [0])


def remember_dendrite_params():
    """Keep a list of the dendrite's parameters, which train after the switch.

    That is the dendrite layer itself plus the weights connecting it to the
    neurons (dendrites_to_top) and to later dendrites (dendrites_to_dendrites,
    empty with a single dendrite). They are frozen once the user's optimizer
    exists (see freeze_dendrite). parent_module is only a template for new
    dendrites and never runs, so it is frozen right away.

    Returns
    -------
    None
    """
    module = state.module
    layer_params = list(module.dendrite_module.layers.parameters())
    connection_params = list(module.dendrites_to_top.parameters()) + list(
        module.dendrite_module.dendrites_to_dendrites.parameters()
    )
    state.trainable = [p for p in layer_params + connection_params if p.requires_grad]
    for param in module.dendrite_module.parent_module.parameters():
        param.requires_grad_(False)


def freeze_dendrite():
    """Freeze the dendrite's parameters until the end of phase 1.

    This runs right after the user's optimizer is built, not before, so the
    optimizer was set up exactly as usual and includes these parameters like any
    other trainable parameter. While frozen they get no gradients, so phase 1
    trains exactly like the original model.

    Returns
    -------
    None
    """
    for param in state.trainable:
        param.requires_grad_(False)
        # Drop any gradient from a backward pass before the optimizer existed
        param.grad = None


def install_patches():
    """Wrap the optimizer and scheduler constructors so demo mode sees what the user creates."""
    global wrappers_installed
    if wrappers_installed:
        return
    wrappers_installed = True
    patch_init(torch.optim.Optimizer, wrap_optimizer_init)
    schedulers = torch.optim.lr_scheduler
    # ReduceLROnPlateau, SequentialLR and ChainedScheduler do not call
    # LRScheduler.__init__, so they each need their own wrapper
    for scheduler_class in (
        schedulers.LRScheduler,
        schedulers.ReduceLROnPlateau,
        schedulers.SequentialLR,
        schedulers.ChainedScheduler,
    ):
        if "__init__" in scheduler_class.__dict__:
            patch_init(scheduler_class, wrap_scheduler_init)


def patch_init(cls, make_wrapper):
    """Replace cls.__init__ with make_wrapper(original __init__)."""
    cls.__init__ = make_wrapper(cls.__dict__["__init__"])


def wrap_optimizer_init(original_init):
    """Optimizer.__init__ wrapper: build the optimizer exactly as usual, then capture it."""

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        optimizer_created(self)

    return init


def wrap_scheduler_init(original_init):
    """Scheduler __init__ wrapper: remember the scheduler until the next optimizer step."""

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not any(scheduler is self for scheduler in state.new_schedulers):
            state.new_schedulers.append(self)

    return init


def optimizer_created(optimizer):
    """Capture the optimizer that trains the perforated layer and warn about unsupported setups.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        An optimizer that just finished its __init__.

    Returns
    -------
    None
    """
    model_ids = {id(param) for param in state.model.parameters()}
    optimizer_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    if len(optimizer_ids & model_ids) == 0:
        # An optimizer for something other than this model
        return
    state.optimizers_seen += 1
    if "muon" in type(optimizer).__name__.lower():
        print(
            "PAI demo WARNING: Muon optimizers are not supported by demo mode. "
            "The dendrite parameters may not train as intended."
        )
    if state.optimizers_seen == 2:
        print(
            "PAI demo WARNING: more than one optimizer trains this model. Demo mode "
            "only restarts the first one that trains %s." % state.target_name
        )
    if state.optimizer is not None or len(optimizer_ids & state.neuron_ids) == 0:
        return
    state.optimizer = optimizer
    GPA.pai_tracker.set_optimizer_instance(optimizer)
    optimizer.register_step_pre_hook(before_optimizer_step)
    if any(id(param) not in optimizer_ids for param in state.trainable):
        print(
            "PAI demo WARNING: this optimizer does not include the dendrite "
            "parameters, so the dendrite cannot train. Build the optimizer from "
            "model.parameters()."
        )
    freeze_dendrite()


def check_optimizer_was_created(module, args):
    """One-time check at the first training forward pass of the perforated module."""
    if not module.training:
        return
    module.pai_optimizer_check.remove()
    del module.pai_optimizer_check
    if state.optimizer is None:
        print(
            "PAI demo WARNING: no optimizer for this model was created after "
            "perforate_model, so the dendrite will never turn on. Create your "
            "optimizer after calling perforate_model."
        )


def before_optimizer_step(optimizer, args, kwargs):
    """Runs before every step of the captured optimizer: count steps and switch phases."""
    if optimizer is not state.optimizer:
        # An optimizer from an earlier demo run in this process
        return
    if state.group_snapshot is None:
        # First step: every scheduler has been created and none has stepped yet
        state.group_snapshot = [
            {
                key: copy.deepcopy(group[key])
                for key in LR_KEYS + OTHER_RESTORED_KEYS
                if key in group
            }
            for group in optimizer.param_groups
        ]
        # Restarts return the optimizer state to this point. It is empty for most
        # optimizers, but some (e.g. Adagrad) fill it in at the end of their __init__.
        state.optimizer_initial_state = {
            param: copy.deepcopy(param_state)
            for param, param_state in optimizer.state.items()
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print(
                "PAI demo WARNING: distributed training is not supported by demo "
                "mode. The dendrite gradients will not be synchronized."
            )
    remember_new_schedulers(optimizer)
    if state.steps_this_phase == state.phase_steps:
        end_phase()
    state.steps_this_phase += 1


def remember_new_schedulers(optimizer):
    """Snapshot schedulers created since the last step so restarts can rewind them."""
    for scheduler in state.new_schedulers:
        if scheduler_optimizer(scheduler) is optimizer:
            state.scheduler_snapshots.append(
                (scheduler, copy.deepcopy(scheduler.state_dict()))
            )
    state.new_schedulers = []


def scheduler_optimizer(scheduler):
    """The optimizer a scheduler controls (composite schedulers ask their first child)."""
    if getattr(scheduler, "optimizer", None) is not None:
        return scheduler.optimizer
    children = getattr(scheduler, "_schedulers", [])
    return scheduler_optimizer(children[0]) if len(children) > 0 else None


def end_phase():
    """Turn the dendrite on (first phase only) and restart the optimizer and schedulers."""
    if state.phase == 1:
        for param in state.trainable:
            param.requires_grad_(True)
    restart_optimizer_and_schedulers()
    print_phase_message()
    state.phase += 1
    state.steps_this_phase = 0


def restart_optimizer_and_schedulers():
    """Rewind the learning rate schedule to its start, scaled by LR_RESTART_FACTOR.

    Every scheduler goes back to its state before the first optimizer step, each
    param group's learning rates go back to LR_RESTART_FACTOR times their values
    at that point, and the optimizer state (momentum and so on) goes back to how
    it was when the optimizer was created. The user's scheduler.step() calls then
    replay the original schedule at the lower learning rate.

    Returns
    -------
    None
    """
    optimizer = state.optimizer
    for scheduler, saved_state in state.scheduler_snapshots:
        scheduler.load_state_dict(copy.deepcopy(saved_state))
    # Scale each scheduler's own lists once. Composite schedulers such as
    # SequentialLR already rewound their children above.
    for scheduler, _ in state.scheduler_snapshots:
        for attribute in ("base_lrs", "_last_lr", "max_lrs"):
            values = getattr(scheduler, attribute, None)
            if isinstance(values, list):
                setattr(scheduler, attribute, [v * LR_RESTART_FACTOR for v in values])
    for group, saved in zip(optimizer.param_groups, state.group_snapshot):
        for key, value in saved.items():
            if key in LR_KEYS:
                group[key] = value * LR_RESTART_FACTOR
            else:
                group[key] = copy.deepcopy(value)
    optimizer.state.clear()
    for param, param_state in state.optimizer_initial_state.items():
        optimizer.state[param] = copy.deepcopy(param_state)


def print_start_message():
    """Explain what demo mode is about to do."""
    if GPA.pc.get_silent():
        return
    share = 100.0 * state.target_params / state.total_params
    print("-" * 70)
    if GPA.pai_tracker.member_vars["doing_pai"]:
        print("PAI demo mode")
        print(
            "  Dendrite added to %s (%d of %d parameters, %.2f%%). It stays frozen"
            % (state.target_name, state.target_params, state.total_params, share)
        )
        print("  during phase 1; every other layer is tracked.")
    else:
        print("PAI demo mode, control run: same phases and restarts, no dendrite.")
    print("  Phase 1: your normal training for %d optimizer steps." % state.phase_steps)
    print(
        "  Phase 2: the dendrite trains, and your optimizer and schedulers restart at"
    )
    print(
        "           %gx their starting learning rate, replaying the same schedule."
        % LR_RESTART_FACTOR
    )
    print("  Run twice your usual number of epochs, and create the optimizer and")
    print("  scheduler after this call.")
    print("  Not supported: multiple optimizers, Muon, EMA or averaged model copies,")
    print("  DataParallel/DDP, or code that sets the learning rate by hand each step.")
    print("  Demo mode switches after a fixed number of steps, not on validation")
    print("  scores. If your model already overfits, the dendrite adds capacity and")
    print("  can make validation worse.")
    print("-" * 70)


def print_phase_message():
    """Announce the end of a phase."""
    if GPA.pc.get_silent():
        return
    print("-" * 70)
    if state.phase == 1 and GPA.pai_tracker.member_vars["doing_pai"]:
        print(
            "PAI demo: phase 1 finished after %d optimizer steps." % state.phase_steps
        )
        print(
            "  The dendrite on %s is now training, and the learning rate restarted"
            % state.target_name
        )
        print("  at %gx its starting value." % LR_RESTART_FACTOR)
        print("  Compare your validation score at the end of training with the score")
        print("  just before this message.")
    else:
        print(
            "PAI demo: phase %d finished. Restarting the learning rate schedule at %gx"
            % (state.phase, LR_RESTART_FACTOR)
        )
        print("  its starting value (no new dendrite).")
    print("-" * 70)
