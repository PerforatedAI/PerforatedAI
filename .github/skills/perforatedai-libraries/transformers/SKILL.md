---
name: perforatedai-libraries-transformers
description: "HuggingFace Transformers integration for PerforatedAI. Handles the Trainer-specific differences: using_perforatedai=True, GPA.metric, eval_strategy. Use when the user's script uses HuggingFace Trainer."
---

# PerforatedAI — HuggingFace Transformers Integration

This skill handles PAI integration when the user is using the **HuggingFace `Trainer`** (from the `transformers` library). It replaces the standard optimizer, training loop, and restructuring steps from the main perforatedai skill.

---

## Step T-1: Install the Patched Fork

The stock `pip install transformers` does **not** work with PAI. A patched fork is required. It internally handles the epoch lifecycle hooks that PAI needs.

Tell the user:

```bash
git clone https://github.com/PerforatedAI/transformers-perforated.git
cd transformers-perforated
pip install -e .
cd ..
pip install perforatedai
```

> **Do not** use the stock transformers when running PAI-enabled scripts. The patched fork is a drop-in replacement — existing code that imports `transformers` will continue to work.

If they already have the fork installed, skip this step.

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
GPA.pc.set_library_validation_score("eval_loss")
GPA.pc.set_library_extra_scores(["train_loss"])
```

- Use `"eval_accuracy"` (or whatever your `compute_metrics` returns) when maximizing
- Use `"eval_loss"` when minimizing loss
- **Set this before calling `perforate_model`**

### Other configuration (same as a basic script)

```python
GPA.pc.set_testing_dendrite_capacity(True)   # Start True for initial capacity check; switch to False for real training
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(10)            # Adjust based on your training length

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

PAI needs a validation score after every epoch. Set `eval_strategy="epoch"`:

```python
training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",   # Required — PAI reads per-epoch scores
    # num_train_epochs — do NOT set this to a huge number; the fork manages training termination internally
    ...
)
```

**Do NOT set `num_train_epochs=1000000`** — the fork handles training termination automatically.

---

## Step T-5: Trainer Constructor

The single most important HF-specific change is passing `using_perforatedai=True` to the Trainer:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    using_perforatedai=True,        # THE key addition — without this PAI has no effect
)
trainer.train()
```

Without this flag the Trainer runs normally and PAI never activates.

---

## What the Fork Handles Automatically

When `using_perforatedai=True` is set, the patched fork handles these internally — **do not add them manually**:

- Calling `GPA.pai_tracker.add_validation_score()` after each epoch
- Detecting when training is complete and stopping the loop
- Reinitializing the optimizer after model restructuring

This means you do **not** write a manual restructuring loop, set a huge epoch count, or manage the optimizer lifecycle yourself.

---

## Custom Loop with a HuggingFace Model (Not Using Trainer)

If the user is using a HuggingFace model (e.g., `AutoModelForImageClassification`) but **writing their own training loop** (no `Trainer`), the fork's automatic handling does **not** apply. In that case, go back to the standard perforatedai skill and follow the normal steps (optimizer setup, `add_validation_score`, restructuring loop, `model.to(device)` after restructuring, etc.).

---

## Minimal Complete Example

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from transformers import Trainer, TrainingArguments

# --- after model is created ---

GPA.metric = "eval_accuracy"                          # metric key from compute_metrics
GPA.pc.set_testing_dendrite_capacity(True)            # True for initial check; False for real training
GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
GPA.pc.set_n_epochs_to_switch(10)

model = UPA.perforate_model(
    model,
    save_name="my_model_dendritic",
    maximizing_score=True,
    making_graphs=True,
)

training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    using_perforatedai=True,
)
trainer.train()
```

---

## Step T-6: Verify

Tell the user to run their script. With `set_testing_dendrite_capacity(True)`, PAI will run a 7-epoch test and print:

```
Successfully added 3 dendrites with GPA.pc.set_testing_dendrite_capacity(True) (default).
You may now set that to False and run a real experiment.
```

Once they see this, change `set_testing_dendrite_capacity(True)` → `set_testing_dendrite_capacity(False)` in their script and tell them to run full training.

---

## Reference Examples

Working examples using this integration:
- `Examples/libraryExamples/huggingface/BERT/train_bert_pai.py` — BERT/RoBERTa classification
- `Examples/libraryExamples/huggingface/mnist/mnist_huggingface_perforatedai.py` — Simple CNN with custom Trainer subclass
- `Examples/libraryExamples/huggingface/ViT Demo Example/` — Step-by-step walkthrough of adding PAI to the official HF image classification script

See `Examples/libraryExamples/huggingface/README.md` for the full integration guide.
