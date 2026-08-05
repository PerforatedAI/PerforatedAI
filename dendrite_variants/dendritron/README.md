# Dendritron Variant

This directory provides a Dendritron candidate architecture for PerforatedAI.
Each candidate uses four specialist branches by default, a learned top-2 sparse
router, a post-projection, a residual path, and GELU activation.

The variant changes candidate architecture only. It deliberately does not
replace PerforatedAI's learning rule or epoch scoring:

- with the open-source `perforatedai` package, candidates use the platform's
  standard gradient-descent path;
- when the separately licensed `perforatedbp` package is installed and enabled,
  candidates use its configured Perforated Backpropagation learning rule.

## Install from the contribution branch

```bash
git clone https://github.com/RichardAragon/PerforatedAI.git
cd PerforatedAI
git checkout agent/add-dendritron-variant
python -m pip install -e .
```

After this contribution is merged, replace the checkout command with
`git checkout develop` when testing the development branch.

Run scripts that import this repository-only variant from the checkout root (or
add that root to `PYTHONPATH`). The editable install exposes the published
`perforatedai` library; the `dendrite_variants/` examples are intentionally not
part of the PyPI package.

Package naming is intentionally different in three places:

- PyPI distribution and Python import: `perforatedai`
- optional Perforated Backpropagation distribution and import: `perforatedbp`
- this repository-only variant module:
  `dendrite_variants.dendritron.dendritron`

Do not commit Perforated Backpropagation credentials. Supply any licensed-package
credentials through the environment as directed by PerforatedAI.

For licensed Perforated Backpropagation testing, install the current package in
the same environment:

```bash
python -m pip install --upgrade perforatedbp
```

The full N -> P -> N lifecycle was tested with `perforatedbp==3.2.5`. Older
releases may not contain the dendrite-variant registration API used by the
current `develop` branch.

## Add the variant to a training script

Configure linear-only perforation, wrap the model, and then initialize the
variant:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

GPA.pc.set_module_names_to_perforate(["Linear"])
model = UPA.perforate_model(model)

from dendrite_variants.dendritron import dendritron

dendritron.initialize_variant_dendrite()
```

Optional constructor settings can be supplied during initialization:

```python
dendritron.initialize_variant_dendrite(
    branches=4,
    top_k=2,
    hidden_features=None,  # defaults to max(in_features, out_features)
)
```

`initialize_variant_dendrite` must be called after `perforate_model`, because it
registers the candidate factory on the tracked modules created during
perforation. The remainder of the optimizer, training, and validation pipeline
is unchanged.

## Architecture contract

`create_dendritron_dendrite(parent)` accepts an `nn.Linear` and returns an
`nn.Module` with matching input and output dimensions. Non-square linears use a
learned residual projection; square linears use an identity residual. The
implementation preserves all leading tensor dimensions, including sequence or
spatially grouped inputs ending in `in_features`.

The factory also accepts an existing `DendritronLinear`. PerforatedAI uses that
path when it creates the best-candidate copy, so both candidate modules have the
same Dendritron configuration.

## Focused test

From the repository root:

```bash
python -m unittest dendrite_variants.dendritron.test_dendritron
```
