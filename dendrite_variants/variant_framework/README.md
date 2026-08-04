# Variant Framework

This directory contains `gradient_descent_linears.py` — a simple reference implementation of a dendrite variant to use as a framework to build your own custom dendrites.  The Gradient Descent Linear Dendrite instantiates all dendrites as `nn.Linear` layers and trains them with standard gradient descent by plugging into the Perforated Backpropagation system, but then just using the standard chain rule of backpropagation when reaching the dendrite's output.

For a full explanation of how variants work and how to build your own, see the `dendrite_variants/README.md` one level up.

---

## What This Variant Does

- **`create_linear_dendrite`** — creates a new `nn.Linear` with the same input/output size as the parent layer.  This will only allow dendrites to be added to linear modules.
- **`calculate_and_backwards_dendrite_loss_gd`** — calculates gradient descent inner product loss from the loss passed to it with Perforated Backpropagation
- **`best_gd_score_improved_this_epoch`** — tracks a per-layer EMA loss and reports improvement using the standard PAI threshold globals values
- **`initialize_variant_dendrite`** — wires all of the above in; call this after `perforate_model`
