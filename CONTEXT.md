# PerforatedAI

Glossary of terms used when selecting which parts of a network get dendrites.

## Language

**Trainable module**:
A module that owns at least one parameter with `requires_grad=True`, either directly or in any submodule. Only trainable modules may be selected for perforation.
_Avoid_: "has parameters" (a frozen module has parameters but is not trainable)

**Selection list**:
One of the five unordered lists of module ids, module names or parameter ids that say what gets perforated or tracked: `module_ids_to_perforate`, `module_ids_to_track`, `module_names_to_perforate`, `module_names_to_track`, `parameter_ids_to_track`.

**Python-supplied item**:
An entry in a selection list that came from a `set_` or `append_` call in the user's script, as opposed to the saved JSON config. It always wins over the JSON, and the config CLI cannot change it.
_Avoid_: "manual" (the CLI is also a manual edit)

**Ineligible module**:
A module that is not trainable. It cannot be selected for perforation in the config CLI. Tracking is unaffected.

**Fused pair**:
A **head** and the **norm** that immediately follows it, treated as a single unit when dendrites are added so the data is normalized while dendrites train. The user's model code does not change; the pair behaves exactly as it did before.
_Avoid_: "wrapped" (reserved for a module that has been given dendrites), "sequential" (a container, not the concept)

**Head**:
The first module of a **fused pair**. It must be a **trainable module**.

**Norm**:
The second module of a **fused pair**, the one whose input is exactly the head's output. Usually a normalization layer, but any module may be a norm.

**Fuse**:
To turn a head and a norm into a **fused pair**.
