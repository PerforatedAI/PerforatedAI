# Dendritron Variant

The Dendritron variant changes the architecture of the temporary dendrite
candidates that PerforatedAI creates for `torch.nn.Linear` layers. A Dendritron
is a residual, routed mixture of small specialist networks. The router selects
which specialists contribute to each input, while PerforatedAI continues to
control when candidates are created, how they are trained, and whether they are
retained as permanent dendrites.

This is an architecture-only variant. It does not replace PerforatedAI's
training lifecycle, validation logic, optimizer integration, or the selected
learning rule.

## Start here: the three levels of terminology

These terms refer to different objects:

- **Parent module:** an existing `nn.Linear` in the user's model. PerforatedAI
  wraps it and continues to compute its normal output.
- **Dendrite candidate:** a temporary module that PerforatedAI creates and
  evaluates for a wrapped parent. With this variant registered, that temporary
  module is a `DendritronLinear` rather than a deep copy of the parent linear.
- **Specialist branch:** one of the internal expert networks inside a single
  `DendritronLinear`. The router mixes `top_k` of these specialists for each
  input position.
- **Accepted dendrite:** a candidate that PerforatedAI copies into the wrapped
  module's persistent dendrite collection. PerforatedAI combines accepted
  dendrite outputs with the parent module's output using platform-managed
  connections.

In short, one PerforatedAI dendrite candidate contains several Dendritron
specialist branches. A specialist branch is not itself a PerforatedAI dendrite.

## What changes and what stays the same

The variant changes only this factory decision:

```text
Default candidate factory:     parent nn.Linear -> copied nn.Linear candidate
Dendritron candidate factory:  parent nn.Linear -> DendritronLinear candidate
```

It does not automatically change:

- the user's model definition, dataset, task loss, or validation metric;
- which modules PerforatedAI wraps;
- the PerforatedAI N/P training-cycle decisions;
- candidate acceptance and model restructuring;
- the optimizer or scheduler lifecycle;
- Perforated Backpropagation's local loss or epoch scoring function.

With the open-source `perforatedai` package, the platform uses its standard
gradient-descent candidate path. When the separately licensed `perforatedbp`
package is installed and enabled, the same Dendritron candidate architecture is
trained by the configured Perforated Backpropagation rule.

## Why this candidate architecture exists

A copied linear candidate applies one learned affine transformation to every
input. A Dendritron candidate tests a different capacity hypothesis: inputs may
benefit from different nonlinear transformations, and a learned router may be
able to select and combine those transformations locally.

The specialist branches provide alternative transformations. The router makes
their use input-dependent. The residual path preserves a direct route from the
candidate input to its output, while the post-projection lets the mixed
specialist representation be recombined before the final activation.

This makes a Dendritron a richer candidate than a copied `nn.Linear`, but richer
also means more parameters and compute. The architecture is offered so that the
hypothesis can be measured inside PerforatedAI's normal candidate-selection
process. Its inclusion is not evidence that it will outperform a linear
candidate on every task.

## Architecture and data flow

For an input whose last dimension is `in_features`, a `DendritronLinear` follows
this path:

```text
input (..., in_features)
  |
  +-- router: Linear(in_features, branches) -> softmax -> top-k weights
  |
  +-- specialist 0: Linear(in, hidden) -> GELU -> Linear(hidden, out) --+
  +-- specialist 1: Linear(in, hidden) -> GELU -> Linear(hidden, out) --+-- weighted mixture
  +-- ...                                                               |
  +-- specialist B-1                                                    |
  |                                                                     v
  |                                                        post-projection Linear(out, out)
  |
  +-- residual: Identity when in == out, otherwise Linear(in, out)
                                                                        |
                  GELU(post-projection(mixture) + residual) <-----------+
                                    |
                                    v
                         output (..., out_features)
```

Routing happens independently at every input position. For example, an input
with shape `[batch, tokens, features]` receives one specialist selection per
`[batch, token]` position. All leading dimensions are preserved.

The router first produces a probability for every specialist. It keeps the
largest `top_k` probabilities, sets the remaining mixture weights to zero, and
renormalizes the selected weights. Only selected specialist outputs contribute
to the mixture.

Important: the current reference implementation evaluates every specialist
before applying the sparse mixture. The routing is sparse in contribution and
branch gradient flow, but it is not yet a sparse-compute implementation and
should not be expected to reduce inference latency.

### Concrete shape example

Suppose the parent is `nn.Linear(128, 64)` and the Dendritron uses its default
`branches=4`, `top_k=2`, and `hidden_features=128`. For an input batch shaped
`[32, 128]`:

1. The router produces `[32, 4]` probabilities.
2. Exactly two mixture weights per row remain nonzero after top-k selection.
3. Each specialist produces a `[32, 64]` output.
4. The four specialist outputs are stacked as `[32, 4, 64]` and reduced with
   the sparse weights to `[32, 64]`.
5. The post-projection preserves `[32, 64]`.
6. Because the parent is non-square, a learned residual projection maps the
   input from `[32, 128]` to `[32, 64]`.
7. The projected mixture and residual are added and passed through GELU,
   producing the required `[32, 64]` candidate output.

## How it participates in the PerforatedAI lifecycle

Registering the variant does not immediately insert a Dendritron into the
model. It installs a candidate factory for future dendrite cycles:

1. Configure PerforatedAI and select `Linear` for perforation.
2. Call `UPA.perforate_model(model)` to wrap the selected parent modules.
3. Call `dendritron.initialize_variant_dendrite(...)` to register the factory on
   those wrappers.
4. Normal or **N mode** trains/evaluates the current model.
5. When PerforatedAI enters candidate or **P mode**, it asks the registered
   factory to create a `DendritronLinear` for each eligible wrapped layer.
6. The active learning rule trains the temporary candidate.
7. When PerforatedAI returns to N mode, the platform retains the selected best
   candidate as an accepted dendrite and restructures the model.
8. The optimizer and scheduler must be rebuilt for the returned model whenever
   `add_validation_score` reports `restructured=True`.

The platform may repeat this process, allowing a wrapped parent to accumulate
more than one accepted Dendritron over successive cycles.

## Requirements and supported scope

- Python environment capable of running the current PerforatedAI `develop`
  branch
- PyTorch and the `perforatedai` package from this repository
- `nn.Linear` parent modules only
- inputs whose last dimension equals the parent's `in_features`
- optional licensed `perforatedbp` for Perforated Backpropagation training

Convolutional, recurrent, attention, and arbitrary custom parent modules are not
accepted by this factory. They can still be tracked by PerforatedAI, but they
must not be selected for Dendritron perforation.

## Installation

### Test the contribution branch before merge

```bash
git clone https://github.com/RichardAragon/PerforatedAI.git
cd PerforatedAI
git checkout agent/add-dendritron-variant
python -m pip install -e .
```

### Test `develop` after merge

```bash
git clone https://github.com/PerforatedAI/PerforatedAI.git
cd PerforatedAI
git checkout develop
python -m pip install -e .
```

Run scripts that import this variant from the repository root, or add the
repository root to `PYTHONPATH`. The editable install exposes the published
`perforatedai` library; the `dendrite_variants/` examples are intentionally
repository-only and are not installed as a top-level PyPI package.

### Package and import names

The related names are similar but not interchangeable:

| Purpose | Install name | Python import |
| --- | --- | --- |
| Open-source platform | `perforatedai` | `perforatedai` |
| Licensed learning-rule package | `perforatedbp` | `perforatedbp` |
| Repository-only Dendritron variant | not separately installed | `dendrite_variants.dendritron` |

For licensed Perforated Backpropagation testing, install the current package in
the same environment:

```bash
python -m pip install --upgrade perforatedbp
```

The full N -> P -> N lifecycle was tested with `perforatedbp==3.2.5`. Older
releases may not contain the dendrite-variant registration API used by the
current `develop` branch. Supply licensed-package credentials through the
environment as directed by PerforatedAI; never commit them to source files,
shell scripts, notebooks, logs, or configuration files.

## Minimal integration

Configure the supported parent type before wrapping the model. Register the
variant only after `perforate_model`:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from dendrite_variants.dendritron import dendritron

# Dendritron candidates currently support nn.Linear parents only.
GPA.pc.set_module_names_to_perforate(["Linear"])

# Convolutions may be tracked without receiving Dendritron candidates.
GPA.pc.set_module_names_to_track(["Conv2d"])

model = UPA.perforate_model(model)

# This must follow perforate_model so the factory reaches the created wrappers.
dendritron.initialize_variant_dendrite(
    branches=4,
    top_k=2,
    hidden_features=None,
)
```

Everything else remains a normal PerforatedAI pipeline. In particular, keep the
model returned by `add_validation_score` and rebuild the optimizer when the
model is restructured:

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

GPA.pai_tracker.set_optimizer(optim.Adadelta)
GPA.pai_tracker.set_scheduler(StepLR)

optimizer_args = {"params": model.parameters(), "lr": 1.0}
scheduler_args = {"step_size": 1, "gamma": 0.7}
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
    model, optimizer_args, scheduler_args
)

for epoch in range(1, max_epochs + 1):
    train_one_epoch(model, train_loader, optimizer)
    validation_accuracy = evaluate_accuracy(model, validation_loader)

    model, restructured, training_complete = (
        GPA.pai_tracker.add_validation_score(validation_accuracy, model)
    )
    model.to(device)

    if restructured:
        optimizer_args = {"params": model.parameters(), "lr": 1.0}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimizer_args, scheduler_args
        )

    if training_complete:
        break
```

`train_one_epoch` and `evaluate_accuracy` above stand for the application's
existing training and validation functions. Do not call `perforate_model` again
inside the epoch loop.

## Configuration reference

```python
dendritron.initialize_variant_dendrite(
    branches=4,
    top_k=2,
    hidden_features=None,
)
```

| Argument | Meaning | Default | Tradeoff |
| --- | --- | --- | --- |
| `branches` | Number of internal specialist networks | `4` | More branches increase capacity, parameters, and compute. Must be at least 1. |
| `top_k` | Specialists with nonzero mixture weight per input position | `2` | Lower values make routing more selective. Must be between 1 and `branches`. |
| `hidden_features` | Width of each specialist's hidden layer | `max(in_features, out_features)` | Larger values increase specialist capacity and memory. Must be positive. |

The candidate mirrors whether the parent `nn.Linear` uses a bias, and it is
created on the parent's device and dtype. If `in_features == out_features`, the
residual path is an identity. Otherwise, the residual path is a learned,
bias-free linear projection.

## Diagnostics and optional routing utilities

After a Dendritron has completed at least one forward pass, inspect its most
recent routing behavior:

```python
from dendrite_variants.dendritron.dendritron import DendritronLinear

for name, module in model.named_modules():
    if isinstance(module, DendritronLinear):
        print(name, module.routing_metrics())
        active = (module.last_sparse_weights > 0).sum(dim=-1)
        print("active specialists per input:", active.unique().tolist())
```

`routing_metrics()` returns:

- `router_entropy`: mean entropy of the router's soft probabilities;
- `min_branch_utilization`: lowest mean soft probability among specialists;
- `max_branch_utilization`: highest mean soft probability among specialists.

These are diagnostics, not acceptance scores used by PerforatedAI.

`DendritronLinear.balance_loss()` returns a differentiable penalty for unequal
soft branch usage. The architecture-only integration does not add that penalty
to the task loss or the Perforated Backpropagation local loss. Doing so would
change the learning rule and should be treated as a separate, explicitly tested
variant rather than silently enabled here.

## How to verify that Dendritrons are actually being created

Immediately after `initialize_variant_dendrite`, the count may still be zero
because the function registers a factory; it does not create candidates. Check
during P mode or after the first accepted-dendrite cycle:

```python
from dendrite_variants.dendritron.dendritron import DendritronLinear

dendritrons = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, DendritronLinear)
]
print("Dendritron modules:", [name for name, _ in dendritrons])
```

The focused unit tests can be run from the repository root:

```bash
python -m unittest dendrite_variants.dendritron.test_dendritron
```

They cover shape preservation, gradients, top-k routing, non-square residuals,
candidate recreation, global registration, checkpoint reconstruction, and
invalid configuration.

## Troubleshooting

### `TypeError: ... expected nn.Linear or DendritronLinear`

An unsupported parent type was selected for perforation. Call
`GPA.pc.set_module_names_to_perforate(["Linear"])` before `perforate_model` and
track other module types separately if needed.

### No `DendritronLinear` appears immediately after initialization

This is expected. Initialization registers the factory. PerforatedAI calls it
when the model enters a dendrite-candidate cycle. Continue through validation
and model restructuring, then inspect the model during P mode or after a
candidate has been accepted.

### Candidates are ordinary `nn.Linear` modules

Confirm that `initialize_variant_dendrite` is called after `perforate_model` and
that the checkout includes the factory-persistence fix in this contribution.
That fix preserves the registered factory when PerforatedAI reconstructs a
wrapped module from saved state.

### `ModuleNotFoundError: dendrite_variants`

Run the script from the repository root or add that root to `PYTHONPATH`.
Installing `perforatedai` from PyPI alone does not install this repository-only
example directory.

### Licensed PBP reports missing variant attributes

Upgrade the licensed package in the same Python environment:

```bash
python -m pip install --upgrade perforatedbp
```

Version `3.2.5` was used for the documented full-cycle test.

### Training stops updating after restructuring

Always retain the model returned by `add_validation_score`. When
`restructured=True`, rebuild the optimizer and scheduler against that returned
model's parameters, as shown in the integration example.

### One specialist dominates the router

Inspect `routing_metrics()` over representative batches. `balance_loss()` is
available for experiments, but it is intentionally not applied by this
architecture-only variant. Any routing regularizer requires its own controlled
benchmark because it changes the optimization objective.

### Expected speedup does not appear

The current implementation computes all specialist outputs and sparsifies only
their mixture. It is a reference architecture for testing routed specialist
geometry inside PerforatedAI, not a fused or conditional-execution kernel.

## Validation scope and current evidence

The contribution has been checked in three ways:

- focused unit tests for the module and factory contract;
- an open-source PerforatedAI lifecycle smoke test that created both candidate
  and best-candidate `DendritronLinear` modules;
- a licensed `perforatedbp==3.2.5` synthetic lifecycle test that completed
  N -> P -> N, preserved the factory through reconstruction, changed candidate
  parameters, retained Dendritron candidate types, and reached
  `training_complete`.

The licensed run validates integration mechanics on synthetic data. It is not a
claim that this architecture improves accuracy, convergence, memory use, or
training speed on a production dataset. Those questions require controlled
task-specific benchmarks against the same model, data split, seeds, training
budget, and acceptance settings.

## API summary

- `DendritronLinear(...)`: the routed specialist candidate module.
- `create_dendritron_dendrite(parent, ...)`: factory accepting `nn.Linear` or an
  existing `DendritronLinear` and returning a fresh compatible candidate.
- `initialize_variant_dendrite(...)`: registers the configured factory globally
  on the currently perforated modules; call it after `perforate_model`.
- `DendritronLinear.routing_metrics()`: detached diagnostics from the latest
  forward pass.
- `DendritronLinear.balance_loss()`: optional differentiable branch-usage
  regularizer that is not automatically added to any training objective.
