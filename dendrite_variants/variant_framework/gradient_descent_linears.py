import torch
import torch.nn as nn
from perforatedai import globals_perforatedai as GPA

try:
    from perforatedbp import modules_pbp as MPB
    from perforatedbp import tracker_pbp as TPB
except ModuleNotFoundError as e:
    # Only pass if perforatedbp package itself is missing
    if e.name == "perforatedbp":
        pass
    else:
        # perforatedbp exists but is missing a dependency
        raise

def create_linear_dendrite(original_module):
    assert isinstance(original_module, nn.Linear), (
        f"create_linear_dendrite expected nn.Linear, got {type(original_module)}"
    )
    return nn.Linear(original_module.in_features, original_module.out_features)

def calculate_and_backwards_dendrite_loss_gd(values):
    """Standard gradient descent alternative to CC: train the dendrite candidate using
    the parent error directly as the gradient signal.
    """
    global _gd_epoch_loss_sum, _gd_epoch_loss_count
    device_index = values.device
    with torch.no_grad():
        MPB.check_dendrite_outs(values, device_index)
        parent_d = values.current_parent_d[device_index][0].detach().clone()

    dendrite_outs = values.dendrite_outs[device_index][0]

    # parent_d == dL/d(neuron_out); dendrite_out is added to neuron_out, so
    # dL/d(dendrite_out) == parent_d.  Summing the inner product gives that gradient.
    loss = (dendrite_outs * parent_d).sum()

    with torch.no_grad():
        new_ema = 0.01 * loss.item() + 0.99 * values.module_loss[device_index].item()
        values.module_loss[device_index] = new_ema

    loss.backward()

def best_gd_score_improved_this_epoch(tracker, first_call=True):
    if tracker.member_vars["mode"] == "n":
        return False
    got_a_best = False
    for layer in tracker.neuron_module_vector:
        values = layer.dendrite_module.dendrite_values[0]
        device_index = values.device
        current_loss = values.module_loss[device_index].item()
        best_loss = values.best_module_loss[0].item()
        if best_loss == 0.0:
            values.best_module_loss[0] = current_loss
            got_a_best = True
            print("[GD] %s: first epoch, setting best_loss=%.6f" % (layer.name, current_loss))
        else:
            pct_threshold = GPA.pc.get_pai_improvement_threshold()
            raw_threshold = GPA.pc.get_pai_improvement_threshold_raw()
            pct_improved = (best_loss - current_loss) / (abs(best_loss) + 1e-12) >= pct_threshold
            raw_improved = (best_loss - current_loss) >= raw_threshold
            if pct_improved and raw_improved:
                values.best_module_loss[0] = current_loss
                got_a_best = True
                print("[GD] %s: improved  current=%.6f  best=%.6f  pct_drop=%.2f%%"
                      % (layer.name, current_loss, best_loss,
                         100.0 * (best_loss - current_loss) / (abs(best_loss) + 1e-12)))
            else:
                print("[GD] %s: no improvement  current=%.6f  best=%.6f  pct_improved=%s  raw_improved=%s"
                      % (layer.name, current_loss, best_loss, pct_improved, raw_improved))
    print("[GD] best_gd_score_improved_this_epoch returning %s" % got_a_best)
    return got_a_best


def initialize_variant_dendrite():
    GPA.pai_tracker.set_create_dendrite_global(create_linear_dendrite)
    # Perforated Backpropagation is required to do variants that require more than just gradient descent
    if GPA.pc.get_perforated_backpropagation():
        # CC values registered by MPB already include the GD subset; just wire the loss.
        MPB.dendrite_loss_fn = calculate_and_backwards_dendrite_loss_gd
        MPB.register_dendrite_variant_values(
            tensor_values=["normal_pass_average_d", "parents_average_d_vector"],
            single_values=["breaking", "locked", "module_loss", "best_module_loss"],
            reinit_skip_values=["best_module_loss"],
        )

        # Wire the per-epoch improvement check used by the tracker.
        TPB.best_pai_score_improved_this_epoch_fn = best_gd_score_improved_this_epoch
    