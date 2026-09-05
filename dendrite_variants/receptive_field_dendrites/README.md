# receptive_field_dendrites

Dendrite factory and model components based on the architecture introduced in:

> **Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning**  
> Poirazi et al.

Each soma receives no direct input (zero receptive field); all input signal flows through sparse dendrite modules grown by PAI. This variant replaces the dense candidate module with a masked linear module whose active synapses are constrained by a receptive-field mask. The mask is applied in the forward pass, so the connectivity remains part of the module behavior while the original parent parameters remain untouched.

The implementation supports four connectivity modes:

- `all_to_all`: every input feature is connected to every output unit. This mode is useful as a baseline or when you want the dendrite to behave like a dense local transformation without any spatial sparsity.
- `random`: each output unit receives exactly `synapses` random connections from the input vector. This gives a sparse, statistically uniform receptive field while keeping the mask simple and fast to generate.
- `somatic`: each soma shares a single spatial center for all of its dendrites. All dendrites for that soma sample from the same local patch in the image, which encourages a consistent local neighborhood across the soma's candidate branches.
- `dendritic`: each dendrite picks its own independent spatial center. This creates distinct local receptive fields per dendrite and is the most expressive mode when the objective is to encourage different branches to specialize on different image regions.

In all spatial modes, the patch is sampled over the image shape provided through `img_shape`, which should be the input tensor layout as `(width, height, channels)`. The `allocate_synapses` helper grows a mask around a chosen center until it reaches the requested number of synapses, expanding outward when the initial patch is too small.

## How to use it

The variant is initialized by calling `initialize_variant_dendrite(...)` after `perforate_model` has created the PAI tracker state. The factory function is `create_rf_dendrite`, which is what PerforatedAI calls whenever it needs a new dendrite candidate.

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from receptive_field_dendrites import initialize_variant_dendrite

model = UPA.perforate_model(model)

initialize_variant_dendrite(
    synapses=8,
    rf_mode='random',
    img_shape=(28, 28, 1),
)

# or, for spatially localized patches:
# initialize_variant_dendrite(synapses=8, rf_mode='somatic', img_shape=(28, 28, 1))
# initialize_variant_dendrite(synapses=8, rf_mode='dendritic', img_shape=(28, 28, 1))
# initialize_variant_dendrite(synapses=8, rf_mode='all_to_all', img_shape=(28, 28, 1))
```

A typical training pipeline then continues exactly the same way as any other PerforatedAI variant:

```python
GPA.pc.set_module_names_to_perforate(['Linear'])
# or restrict to the layer types you want to grow dendrites on

model = UPA.perforate_model(model)
initialize_variant_dendrite(synapses=12, rf_mode='somatic', img_shape=(28, 28, 1))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# train loop, add_extra_score(...), add_validation_score(...), etc.
```

The `rf_mode` should match the pattern you want for candidate connectivity:

- use `'all_to_all'` for dense, baseline-style candidate masks;
- use `'random'` when each dendrite should be sparse but not spatially organized;
- use `'somatic'` when all dendrites from one soma should share locality;
- use `'dendritic'` when each dendrite should have its own local region and specialization.

This makes the variant easy to swap into an existing PAI workflow without changing the surrounding training loop, optimizer setup, or growth logic.
