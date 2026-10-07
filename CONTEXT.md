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

**Block**:
A module type that has submodules and occurs more than once in the model, such as a ResNet `BasicBlock` or a transformer layer. Identified by class name, like a type rule.
_Avoid_: using "block" for a `PAISequential` container, or for the inference-time `PAILayer`

**Relative path**:
The path from one instance of a block down to a module inside it, such as `.conv2`. It exists only in the instances of the block that actually contain it.

**Block rule**:
A mode (perforate or track) applied to a relative path inside every instance of a block. It outranks a type rule and is outranked by a mode set directly on that module by id.
_Avoid_: "scoped rule", "template" (the instance shown while editing is only for display)
