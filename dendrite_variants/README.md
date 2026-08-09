# Dendrite Variants

Dendrite variants let you customize how PerforatedAI creates and trains dendrite candidates. Instead of using the default Cascade Correlation algorithm or gradient descent with deep-copied parent modules, you can plug in any architecture and any local learning rule.

---

## What a Variant Is

By default, when PerforatedAI adds a dendrite to a layer it deep-copies the parent module and trains it using gradient descent. By default when Perforated Backpropagation is also engaged with the perforatedbp library dendrites are trained with the Cascade Correlation learning rule.  A variant replaces one or both of those defaults:

- **Custom architecture** — return any `nn.Module` you want as the candidate dendrite instead of a deep copy
- **Custom loss** — replace the per-batch CC loss with your own function (requires Perforated Backpropagation)

---

## What You Need to Implement

### Minimum requirement (no PBP)

If you only want to control what module is used for each candidate, you only need one thing:

```python
def initialize_variant_dendrite():
    GPA.pai_tracker.set_create_dendrite_global(my_create_fn)
```

Where `my_create_fn(original_module)` receives the parent module and returns a new `nn.Module`.

That is all that is required. Standard gradient descent will train the candidate automatically, the perforatedai library will create connections from the dendrites to the neurons, and the perforatedai library will determine when to add additional dendrites to existing neurons.

### Full variant (custom architecture + custom loss via PBP)

If Perforated Backpropagation is installed and enabled, you can also replace the per-batch loss function and the per-epoch scoring function:

```python
def initialize_variant_dendrite():
    GPA.pai_tracker.set_create_dendrite_global(my_create_fn)
    if GPA.pc.get_perforated_backpropagation():
        MPB.dendrite_loss_fn = my_loss_fn
        MPB.register_dendrite_variant_values(
            tensor_values=[...],      # extra per-neuron tensor buffers on DendriteValueTracker
            single_values=[...],      # extra scalar buffers
            reinit_skip_values=[...], # names that should NOT be zeroed at the start of each P cycle
        )
        TPB.best_pai_score_improved_this_epoch_fn = my_score_fn
```

**`my_loss_fn(values)`** — called every batch during P mode. `values` is a `DendriteValueTracker` where all dendrite values are tracked over batches and epochs. This function will receive the local loss that the neuron has calculated, use it to compute a dendrite loss, and call `.backward()` through the dendrite module. Any custom tensors you registered via `register_dendrite_variant_values` are available as attributes within `values`.

**`my_score_fn(tracker, first_call=True)`** — called once per epoch. Loop over `tracker.neuron_module_vector`, read state from all DendriteValueTracker in the network, and return `True` if any layer improved. This controls the patience counter and when best weights are saved. Use `GPA.pc.get_pai_improvement_threshold()` and `GPA.pc.get_pai_improvement_threshold_raw()` to set thresholds for your score improvements.

**`register_dendrite_variant_values`** — optional. Only needed if your loss or scoring functions need to store state on `DendriteValueTracker` between batches or epochs. Add names to `reinit_skip_values` for any buffers that track a running best — otherwise they will be zeroed at the start of every cycle when `restructured` returns True from `add_validation_score`.

---

## Adding a Variant to a Training Pipeline

The only changes to a standard PerforatedAI script are:

**1. Optionally restrict which layers get perforated** (if your variant only supports certain layer types be sure to only pick those ones):

```python
GPA.pc.set_module_names_to_perforate(["Linear"])
GPA.pc.set_module_names_to_track(["Conv2d"])
```

**2. Call `initialize_variant_dendrite` after `perforate_model`:**

```python
model = UPA.perforate_model(model)

import variant_framework.my_variant as MV
MV.initialize_variant_dendrite()
```

Everything after this — optimizer setup, training loop, `add_extra_score`, `add_validation_score` — is identical to any other PAI script.

---

## Example

`mnist_perforatedai_variant.py` demonstrates the GD linear variant on MNIST/EMNIST and is meant as a drop-in testbed to verify a variant works before integrating it into a larger pipeline.

```bash
cd dendrite_variants
python mnist_perforatedai_variant.py
python mnist_perforatedai_variant.py --dataset EMNIST
```

The `variant_framework/` directory contains the an example variant implementation which just implements dendrites as a simple linear layer using typical gradient descent as the local loss rule. See `variant_framework/README.md` for details on that specific variant.
