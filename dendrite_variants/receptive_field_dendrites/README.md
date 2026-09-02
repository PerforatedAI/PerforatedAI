# receptive_field_dendrites

Dendrite factory and model components based on the architecture introduced in:

> **Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning**  
> Poirazi et al.

Each soma receives no direct input (zero receptive field); all input signal flows through sparse dendrite modules grown by PAI. Four connectivity modes are supported: `all_to_all`, `random`, `somatic` (spatially localized patch shared across a soma's dendrites), and `dendritic` (independent patch per dendrite).
