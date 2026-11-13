"""Analysis of how a PAI run ended, with suggestions for what to try next.

When training ends because the newest dendrites did not improve the validation
score on any of their max_dendrite_tries tries, the tracker calls
analyze_outcome with the network that still holds those dendrites, before it
deletes them by reloading the best model. Each function in OUTCOME_CHECKS
looks at that network and the results of the run, and returns a suggestion,
or None when it has nothing to suggest. The results that the CSV files are
written from are in GPA.pai_tracker.member_vars, and the files themselves are
in the GPA.pc.get_save_name() folder. To add a check, write a function with
the same arguments as check_dendrite_lr_multiplier and append it to
OUTCOME_CHECKS.
"""

from perforatedai import globals_perforatedai as GPA
from perforatedai import modules_perforatedai as PA
from perforatedai import utils_perforatedai as UPA

# Below this, the newest dendrites' output weights are too small for the dendrites
# to change the network (see newest_dendrite_to_top_ratios). Fine-tuning bert-mini
# at a learning rate of 2e-5, a dendrite trained at 1x that rate reached 0.0003
# after a cycle, at 10x 0.008 and at 30x 0.18. Training a CNN on MNIST from
# scratch, the largest ratio of any module was at least 0.007 at every epoch.
SMALL_DENDRITE_TO_TOP_RATIO = 0.001


def newest_dendrite_to_top_ratios(net):
    """Measure the output weights (dendrites_to_top) of the newest dendrite of each perforated module.

    For each output of a module, the newest dendrite's output weight is divided
    by the norm of the module's weights for that output, which is about the size
    of that output for inputs of unit size. The module's ratio is the median over
    its outputs.

    Parameters
    ----------
    net : nn.Module
        The network, with dendrites added.

    Returns
    -------
    dict
        Module name to ratio, for the modules with dendrites whose weight has the
        outputs as its first dimension.
    """
    ratios = {}
    for module in UPA.get_pai_modules(net, 0):
        if module.dendrite_modules_added == 0:
            continue
        weight = getattr(module.main_module, "weight", None)
        to_top = module.dendrites_to_top[-1][-1].detach().abs()
        if weight is None or weight.dim() < 2 or weight.shape[0] != to_top.shape[0]:
            continue
        output_size = weight.detach().flatten(1).norm(dim=1).clamp_min(1e-12)
        ratios[module.name] = (to_top / output_size.to(to_top.device)).median().item()
    return ratios


def check_dendrite_lr_multiplier(net, dendrites_added):
    """Suggest a higher dendrite_lr_multiplier if the most recently deleted dendrites stayed too small.

    A dendrite's output weights (dendrites_to_top) start at 0. With a small
    learning rate, such as when fine-tuning a pretrained model, they barely grow,
    so the dendrites cannot change the network or improve its score. When
    dendrites were kept, the score may then have improved only because training
    restarted after each one was added.

    Parameters
    ----------
    net : nn.Module
        The network holding the most recently deleted dendrites.
    dendrites_added : bool
        Whether the final model kept at least one dendrite.

    Returns
    -------
    str or None
        The suggestion, or None.
    """
    ratios = newest_dendrite_to_top_ratios(net)
    if not ratios or max(ratios.values()) >= SMALL_DENDRITE_TO_TOP_RATIO:
        return None
    multiplier = GPA.pc.get_dendrite_lr_multiplier()
    finding = (
        f"{'The' if dendrites_added else 'No dendrites were kept, and the'} output "
        f"weights (dendrites_to_top) of the last dendrites tried stayed very small: "
        f"at most {max(ratios.values()):.1e} of the size of the module weights. "
        f"These weights start at 0, and with a small learning rate, such as when "
        f"fine-tuning a pretrained model, they barely grow, so the dendrites cannot "
        f"change the network."
    )
    if dendrites_added:
        finding += (
            " The dendrites that were kept may have improved the score only because "
            "training restarted after each one was added."
        )
    optimizer = GPA.pai_tracker.member_vars.get("optimizer_instance")
    if multiplier != 1 and _trains_dendrites_at_neuron_lr(optimizer):
        return (
            f"{finding} dendrite_lr_multiplier is {multiplier:g}, but the optimizer "
            f"trains the dendrites at the learning rate of the neurons. Build its "
            f"parameters with UPA.get_parameter_groups(params, lr), both where you "
            f"first create it and where you create it again after each restructure."
        )
    if multiplier == 1:
        setting = "GPA.pc.set_dendrite_lr_multiplier(10), then 30 or 100"
    else:
        setting = f"GPA.pc.set_dendrite_lr_multiplier({multiplier * 3:g})"
    suggestion = (
        f"{finding} Next, try {setting} (or dendrite_lr_multiplier in your config "
        f"file) to train the dendrites at that multiple of the learning rate."
    )
    if not UPA.parameter_groups_requested:
        suggestion += (
            " setup_optimizer applies it. If you build the optimizer yourself for "
            "set_optimizer_instance, build its parameters with "
            "UPA.get_parameter_groups(params, lr), both where you first create it "
            "and where you create it again after each restructure."
        )
    return suggestion


def _trains_dendrites_at_neuron_lr(optimizer):
    """Return whether a parameter group of the optimizer mixes dendrite and neuron parameters."""
    for param_group in getattr(optimizer, "param_groups", []):
        dendrite = [PA.is_dendrite_parameter(p) for p in param_group["params"]]
        if any(dendrite) and not all(dendrite):
            return True
    return False


OUTCOME_CHECKS = [check_dendrite_lr_multiplier]


def analyze_outcome(net, dendrites_added):
    """Run every check in OUTCOME_CHECKS and return their suggestions.

    A check that raises an error is skipped, so that it cannot stop training
    from ending normally.

    Parameters
    ----------
    net : nn.Module
        The network holding the most recently deleted dendrites.
    dendrites_added : bool
        Whether the final model kept at least one dendrite.

    Returns
    -------
    list of str
        The suggestions of all checks that made one.
    """
    suggestions = []
    for check in OUTCOME_CHECKS:
        try:
            suggestion = check(net, dendrites_added)
        except Exception as e:
            print(f"Skipping the outcome check {check.__name__}: {e}")
            continue
        if suggestion:
            suggestions.append(suggestion)
    return suggestions
