from __future__ import annotations

from typing import Iterator

# probably should make cleanup_helper_functions in perforatedai library
from perforatedbp.cleanup_helper_functions import is_in_dendrite_training_mode


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from perforatedai import modules_perforatedai as MPA
    from perforatedai import tracker_perforatedai as TPA
    from perforatedai import globals_perforatedai as GPA

def register_epoch_hooks() -> None:
    # Imported at call time (not module top level) to avoid a circular import, since the
    # ``TYPE_CHECKING`` block above deliberately keeps ``globals_perforatedai`` out of the
    # runtime import graph.  ``GPA`` is needed here because it is used in the body below,
    # which ``from __future__ import annotations`` does not cover.
    from perforatedai import globals_perforatedai as GPA
    start = GPA.pc.get_epoch_start_callbacks()
    stop = GPA.pc.get_epoch_stop_callbacks()
    start.clear();
    stop.clear()  # so repeats do not accumulate
    start.append(notify_epoch_start)
    stop.append(notify_epoch_stop)

def notify_epoch_start(tracker: TPA.PAINeuronModuleTracker) -> None:
    """
    TODO this should be a method of another class
    Call ``on_epoch_start(tracker)`` on each dendrite module's ``DendriteLoss`` object.

    See :func:`_dendrite_losses_in_training_mode` for exactly which objects are notified and
    the dendrite-mode gate.  This signature (tracker only) matches what the ``start_epoch``
    boundary loop calls, so it can be registered directly in ``GPA.pc``'s
    ``epoch_start_callbacks``.
    """
    for dendrite_loss in _dendrite_losses_in_training_mode(tracker):
        dendrite_loss.on_epoch_start(tracker)


def notify_epoch_stop(tracker: TPA.PAINeuronModuleTracker) -> None:
    """Call ``on_epoch_stop(tracker)`` on each dendrite module's ``DendriteLoss`` object.

    See :func:`_dendrite_losses_in_training_mode` for exactly which objects are notified and
    the dendrite-mode gate.  This signature (tracker only) matches what the ``stop_epoch``
    boundary loop calls, so it can be registered directly in ``GPA.pc``'s
    ``epoch_stop_callbacks``.
    """
    for dendrite_loss in _dendrite_losses_in_training_mode(tracker):
        dendrite_loss.on_epoch_stop(tracker)



def _dendrite_losses_in_training_mode(
        neuron_module_tracker: TPA.PAINeuronModuleTracker,
) -> 'Iterator[DendriteLoss]':
    """
    Yield each dendrite module's ``DendriteLoss`` object, but only in dendrite mode.

    The library constructs one ``DendriteLoss`` object per ``PAIDendriteModule`` (and only
    where perforated backpropagation is enabled), reachable as
    ``neuron_module.dendrite_module.dendrite_loss`` for each entry in
    ``tracker.neuron_module_vector``.  Yields nothing unless the tracker is in dendrite
    training mode, so the setup and load-path boundary calls (which occur in neuron mode) are
    naturally ignored; a dendrite module lacking a ``dendrite_loss`` attribute is skipped.

    Args:
        tracker: The ``PAINeuronModuleTracker`` whose ``neuron_module_vector`` owns the
            per-layer ``DendriteLoss`` objects.

    Yields:
        Each dendrite module's ``DendriteLoss`` object, in ``neuron_module_vector`` order.
    """
    if not is_in_dendrite_training_mode(neuron_module_tracker):
        return
    for neuron_module in neuron_module_tracker.neuron_module_vector:
        dendrite_loss: 'DendriteLoss | None' = getattr(
            neuron_module.dendrite_module, "dendrite_loss", None
        )
        if dendrite_loss is not None:
            yield dendrite_loss
