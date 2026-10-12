---
name: perforatedai-libraries-transformers
description: "HuggingFace Transformers integration for PerforatedAI. Handles the Trainer-specific differences: perforate_model=True, GPA.pc.set_library_validation_score, eval_strategy, and num_train_epochs as the per-cycle LR schedule length. Use when the user's script uses HuggingFace Trainer."
---

# PerforatedAI — HuggingFace Transformers Integration

This skill handles PAI integration when the user is using the **HuggingFace `Trainer`** (from the `transformers` library). It replaces the standard optimizer, training loop, and restructuring steps from the main perforatedai skill.

---

## Step T-1: Verify Transformers Compatibility

The standard transformers package does **not** include the hooks that PAI requires. Using PAI with HuggingFace Trainer requires the **transformers-perforated** library.

**Check if already installed:** Ask the user if they have transformers-perforated in their environment. If yes, skip this step.

**If not installed:** Tell them: "PAI integration with Trainer requires transformers-perforated. See the PerforatedAI documentation for installation details."

---

## Step T-2: Add Imports

Add these imports (same as any PAI integration):

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

---

## Step T-3: Configure PAI and Convert the Model

Add this block **after the model is created and before the Trainer is constructed**.

### Set the metric PAI should watch

With the HF Trainer you cannot call `add_validation_score()` directly — the Trainer manages evaluation internally. Instead, tell PAI which metric key to read from the Trainer's evaluation output:

```python
GPA.pc.set_library_validation_score("eval_loss")          # metric PAI uses for dendrite decisions
GPA.pc.set_library_extra_scores(["train_loss"])           # optional: extra scores logged each epoch (graphed)
GPA.pc.set_library_extra_scores_without_graphing([])      # optional: extra scores logged but not graphed
```

- Use `"eval_accuracy"` (or `"eval_" + <key returned by compute_metrics>`) when maximizing
- Use `"eval_loss"` when minimizing loss
- `"train_loss"` is the mean training loss over the epoch since the previous evaluation
- Any other key in the Trainer's eval metrics can be used as an extra score
- Use the setters. Direct assignment (`GPA.metric = ...` or `GPA.pc.library_validation_score = ...`) is **ignored** — the Trainer reads `GPA.pc.get_library_validation_score()`. If you see `GPA.metric` in an existing script, replace it with `set_library_validation_score`
- **Set these before calling `perforate_model`**

### Other configuration (same as a basic script)

```python
GPA.pc.set_testing_dendrite_capacity(True)   # Start True for initial capacity check; switch to False for real training
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(3)             # Patience window; keep well below num_train_epochs (see Step T-4)

# Optional: restrict which modules get dendrites
# GPA.pc.append_module_names_to_track(['ViTModel', 'ViTEncoder'])
```

### Convert the model

```python
model = UPA.perforate_model(
    model,
    save_name="my_model_dendritic",
    maximizing_score=True,   # True for accuracy; False for loss
    making_graphs=True,
)
```

---

## Step T-4: TrainingArguments

```python
training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",   # Required — PAI reads a score after every epoch
    num_train_epochs=10,     # Length of the LR schedule per PAI cycle — use the user's normal (non-PAI) value
    ...
)
```

### `eval_strategy="epoch"` is required

The default `eval_strategy` is `"no"`. If no evaluation runs, PAI never receives a score and **training never ends** — there is no error, it just keeps running. `"steps"` technically works, but PAI then treats every evaluation as an epoch, and a fractional `eval_steps` (e.g. `0.1`) never fires under PAI. Always use `"epoch"`.

### `num_train_epochs` sets the scheduler, not when training ends

PAI decides when training ends. `num_train_epochs` is still used: it sets `num_training_steps` for the LR scheduler (warmup and decay), and the scheduler is **rebuilt with that same length after every PAI restructure**. So each neuron or dendrite cycle gets a fresh schedule of `num_train_epochs` epochs.

- Keep `num_train_epochs` at the value the user would use for a normal training run.
- **Never set it to a huge number** (e.g. `1000000`, an older recommendation). That stretches warmup and decay across the whole value, so the LR barely moves (or, with warmup, stays near zero).
- With the default `linear` scheduler, LR reaches 0 at the end of each `num_train_epochs` window and **stays at 0 until the next restructure**. Choose switch settings so that wait is short:
  - `DOING_HISTORY`: keep `n_epochs_to_switch` well below `num_train_epochs`. The flat LR causes a plateau, and PAI waits `n_epochs_to_switch` epochs at LR 0 before switching.
  - `DOING_FIXED_SWITCH`: set `set_first_fixed_switch_num` and `set_n_fixed_switch_num` equal to `num_train_epochs`, so each switch happens right when the schedule ends.
  - With `lr_scheduler_type="constant"`, `num_train_epochs` has no effect on the LR.

---

## Step T-5: Trainer Constructor

The single most important HF-specific change is passing `perforate_model=True` to the Trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    perforate_model=True,        # THE key addition — without this PAI has no effect
)
trainer.train()
```

Without this flag the Trainer runs normally and PAI never activates. The default is `False`.

> Older versions of transformers-perforated named this argument `using_perforatedai=True`. If the Trainer raises `TypeError: ... unexpected keyword argument 'perforate_model'`, the user has an older install — have them update transformers-perforated, or use `using_perforatedai=True` for that version.

---

## Step T-6: Check the User's Script Before They Run It

Go through the user's script and fix or flag each item. Several of these fail **silently**: the run doesn't crash, it just never ends or never adds dendrites.

| Check | What goes wrong otherwise |
|---|---|
| `perforate_model=True` is passed to `Trainer` (or to their `Trainer` subclass) | PAI never activates — normal training, no dendrites |
| `eval_strategy="epoch"` is set (default is `"no"`) | No scores reach PAI — **training runs forever** |
| An `eval_dataset` is passed | Trainer raises an error at construction |
| `set_library_validation_score(...)` names a key the eval metrics actually contain (`"eval_"` + the `compute_metrics` key, or `"eval_loss"`), and `prediction_loss_only` is not `True` when using a `compute_metrics` key | `ValueError: Validation score '...' not found in metrics` after the first epoch |
| No `GPA.metric = ...` or direct attribute assignment of library scores | Setting is ignored, the key stays `""`, and the run fails after the first epoch with `Validation score '' not found in metrics` |
| `maximizing_score` matches the chosen metric (True for accuracy, False for loss) | Dendrite decisions optimize the wrong direction |
| `num_train_epochs` is the user's normal value, not huge | LR schedule is effectively flat or stuck in warmup |
| `n_epochs_to_switch` (HISTORY) is well below `num_train_epochs`, or fixed switch numbers (FIXED) equal it | Long stretches at LR 0 between cycles |
| If the user relies on `save_strategy`, `load_best_model_at_end`, or `EarlyStoppingCallback`, tell them these are disabled or ignored under PAI | They'll expect HF checkpoints or early stopping that won't happen. PAI saves the best model under `save_name` |

---

## What PAI Handles Automatically with Trainer

When `perforate_model=True` is set, the transformers-perforated library handles these internally — **do not add them manually**:

- Calling `GPA.pai_tracker.add_validation_score()` after each evaluation, plus any configured extra scores
- Detecting when training is complete and stopping the loop (`num_train_epochs` does not stop training)
- Rebuilding the optimizer (same `TrainingArguments` settings) and the LR scheduler (same `num_train_epochs` length) after each restructure
- Disabling HF checkpoint saving (`save_strategy` is forced to `"no"`, since HF checkpoints can't reload a restructured model)

This means you do **not** write a manual restructuring loop or manage the optimizer lifecycle yourself.

---

## Custom Loop with a HuggingFace Model (Not Using Trainer)

If the user is using a HuggingFace model (e.g., `AutoModelForImageClassification`) but **writing their own training loop** (no `Trainer`), the transformers-perforated library's automatic handling does **not** apply. In that case, go back to the standard perforatedai skill and follow the normal steps (optimizer setup, `add_validation_score`, restructuring loop, `model.to(device)` after restructuring, etc.).

---

## Minimal Complete Example

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from transformers import Trainer, TrainingArguments

# --- after model is created ---

GPA.pc.set_library_validation_score("eval_accuracy")  # "eval_" + key returned by compute_metrics
GPA.pc.set_library_extra_scores(["train_loss"])       # optional
GPA.pc.set_testing_dendrite_capacity(True)            # True for initial check; False for real training
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(3)                      # well below num_train_epochs

model = UPA.perforate_model(
    model,
    save_name="my_model_dendritic",
    maximizing_score=True,
    making_graphs=True,
)

training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",
    num_train_epochs=10,     # LR schedule length per PAI cycle; PAI decides when training ends
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    perforate_model=True,
)
trainer.train()
```

---

## Step T-7: Verify

Tell the user to run their script. With `set_testing_dendrite_capacity(True)`, PAI will run a 7-epoch test and print:

```
Successfully added 3 dendrites with GPA.pc.set_testing_dendrite_capacity(True) (default).
You may now set that to False and run a real experiment.
```

Once they see this, change `set_testing_dendrite_capacity(True)` → `set_testing_dendrite_capacity(False)` in their script and tell them to run full training.

---

## Reference Examples

Working example using this integration:
- `examples/library_examples/huggingface/mnist/mnist_huggingface_perforatedai.py` — Simple CNN with a custom Trainer subclass (`mnist_huggingface.py` in the same folder is the non-PAI baseline; the folder `README.md` walks through the changes)
