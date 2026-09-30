---
name: perforatedai
description: "Add artificial dendrites to a PyTorch model with the PerforatedAI library - the main integration workflow. Trigger: 'Perforate my model' (interactive setup: analyze the model, record a baseline, add the ~10 lines of PAI integration, verify dendrites train). Also use when configuring PAI settings or working with PAINeuronModule / PAIDendriteModule. Sibling skills own the other phases: perforatedai-debugging for crashes and 'debug my perforated model', perforatedai-deploy for inference loading and ONNX/TFLite/TorchScript export, perforatedai-analyze for tuning a completed run, perforatedai-pilot for a controlled baseline-vs-dendrites comparison with no tuning."
---

# PerforatedAI Dendrite Network Skill

## Available Resources

This skill references the following resources in the PerforatedAI repository on GitHub:

- **API Documentation**: [api/](https://github.com/PerforatedAI/PerforatedAI/tree/main/api) - All API guides and references
- **Source Code**: [perforatedai/](https://github.com/PerforatedAI/PerforatedAI/tree/main/perforatedai) - Complete PerforatedAI library implementation
- **Examples**: [examples/](https://github.com/PerforatedAI/PerforatedAI/tree/main/examples) - Working code examples for various architectures

Feel free to reference these when helping users debug or understand implementation details.

## Entry Points

### 1. "Perforate my model" - Interactive Setup

When the user says **"Perforate my model"**, start the interactive setup process.

**IMPORTANT: Your role is to analyze code and make edits. Never run the user's training script - only ask them to run it and provide output.**

**First, check if PAI is already partially integrated:**

Ask for their training script path, then read it and check for:

- PAI imports: `from perforatedai import globals_perforatedai as GPA` and `utils_perforatedai as UPA`
- Configuration calls: `GPA.pc.set_*` functions
- Model initialization: `UPA.perforate_model()` call
- Optimizer setup: `GPA.pai_tracker.setup_optimizer()` or `set_optimizer_instance()`
- Training loop: `GPA.pai_tracker.add_validation_score()` call

**If PAI integration is already present:**

Tell them: "I see you already have some PerforatedAI integration! Let me check what's complete and what's missing..."

Analyze what's been done and report:

- ✅ Completed steps (e.g., "Imports added", "Configuration set", "Model initialized")
- ❌ Missing steps (e.g., "Optimizer setup missing", "Training loop not updated")

Then ask: "Would you like me to:

1. Complete the missing integration steps
2. Debug/optimize your existing setup (say 'Debug my perforated model')
3. Start fresh with a different approach"

Based on their choice:

- **Option 1**: Continue from the first incomplete step
- **Option 2**: Jump to the "Debug My Perforated Model" workflow below
- **Option 3**: Confirm they want to replace existing code, then start from Step 1

**If no PAI integration found:**

Proceed with Step 1 below.

### Prerequisites: Install PerforatedAI Packages

Before doing anything else, the user must install the two required pip packages. These are not bundled with the skill — the user must install them in their Python environment.

**Instruct the user to run this command in their terminal:**

```bash
pip install perforatedai perforatedbp
```

- **`perforatedai`** — core dendrite library (`globals_perforatedai`, `utils_perforatedai`, etc.)
- **`perforatedbp`** — Perforated Backpropagation extension (required for dendrite scoring)

Ask them to confirm the install completed without errors before continuing.

### Step 1: Discovery

#### 1.1 Get Training Script and Analyze Model

**Ask:** "What's the path to your training script?"

- If they provide a path: Read the script and analyze it to determine:
  - Model architecture type (CNN, Transformer, ResNet, Custom, etc.)
  - Model version/size (e.g., ResNet18 vs ResNet50, GPT2-small vs GPT2-large)
  - Input dimensions and data format
  - Training loop structure
  - Optimization metric being used
  - Whether the script has configurable model selection (command-line arguments for model, architecture parameters, etc.)
- If they say they don't have a script yet: Tell them:

  > "PerforatedAI is an optimization tool for existing models. Please build an initial training setup first and get a baseline working before integrating dendrites. Once you have a working training script, come back and say 'Perforate my model' to add dendritic optimization."

  Then stop - do not proceed with integration.

#### 1.2 Check for Library-Managed Training

**Before proceeding with the standard integration steps, check whether the user's script uses a training library that requires a different integration path.**

Look for these patterns in their script:

- **HuggingFace Transformers Trainer**: imports of `Trainer` or `TrainingArguments` from `transformers`, or a call to `trainer.train()`
- **PyTorch Lightning**: imports of `pl.LightningModule`, `pl.Trainer`, or `lightning.pytorch`

**If you find HuggingFace Trainer usage:**

Tell them: "I see your script uses the HuggingFace `Trainer`. The PAI integration is slightly different when using Trainer — PAI's transformers library handles several things automatically that you'd otherwise do manually."

**IMMEDIATELY load and follow the library skill:**

Load the sibling skill **perforatedai-libraries-transformers** (installed alongside this one) and follow it in full. If your agent cannot load it by name, read the entire file from:

```
https://github.com/PerforatedAI/PerforatedAI/blob/main/skills/perforatedai-libraries-transformers/SKILL.md
```

**Follow every step in that skill. Do NOT continue with steps 1.2 onward in this skill.**

---

**If you find PyTorch Lightning usage:**

Tell them: "I see your script uses PyTorch Lightning. PAI has specific integration requirements for Lightning modules. Please check `examples/libraryexamples/pytorch_lightning/` for reference examples, and refer to the standard PAI API steps below with special attention to placing `add_validation_score` inside your `validation_epoch_end` hook."

Then continue with step 1.2 below — the standard integration steps apply, but flag to the user that optimizer setup goes through Lightning's `configure_optimizers` and the restructuring block goes in `validation_epoch_end`.

---

**If neither is found:** proceed normally with step 1.2.

---

#### 1.3 Ask About Optimization Goal

**Ask:** "What are you optimizing for?"

Options:

- **Accuracy / Loss / Decision Making** - Improve model performance metrics
- **Efficiency / Model Size** - Reduce parameters while maintaining performance

Based on their answer:

**If Accuracy/Loss/Decision Making:**

- Proceed to 1.4, then Step 2, with standard dendrite growth settings (maximize metric or minimize loss)

**If Efficiency/Model Size:**

**Educate them first:** "Important: Dendrites ADD parameters to your model, but they do so more efficiently than traditional approaches. To optimize for efficiency with PerforatedAI, you should start with a smaller base model and then add dendrites strategically. This will allow you to achieve better performance than a larger model with fewer parameters. Let's work together to find the right balance for your use case."

Then **read `efficiency-path.md` in this skill's directory and follow it.** It contains the full decision tree: how to check whether model selection is configurable, which smaller variant to recommend for each model family, and how to establish the downsized baseline. Come back to 1.4 afterwards.

Do not read that file for the accuracy path - it does not apply.

#### 1.4 Record the Baseline Number

**Do this before you edit anything.** You cannot tell whether dendrites helped if nobody wrote down where the model started. This is the single most common omission in a PAI integration, and it is unrecoverable later without rerunning the original code.

**Ask:** "Before I change anything - what does your model currently score on your validation metric, and how many parameters does it have?"

**If they have a number from the current, unmodified script:** record the metric value, the epoch it peaked at, total epochs, parameter count, and the hardware. Tell them you're holding it as the comparison point.

**If they don't have one, or it's from older code or different hardware:** tell them:

> "Run your script as-is once and tell me the best validation score it reaches. That's what everything after this gets compared against - without it we won't be able to tell whether the dendrites earned their keep."

Wait for the number before proceeding to Step 2.

**If they're on the efficiency path (1.3), you have two baselines:** the original model's score and the downsized model's score. Record both. The goal is recovering the original score from the downsized model plus dendrites.

**Carry this number forward.** You restate it in Step 10 when the first dendrite result comes in.

### Step 2: Add Imports

Add these imports at the top of their training script:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

These are the same for all model types. Make the change in their file.

### Step 3: Configure PAI Settings

Based on the analyzed model type, add the appropriate configuration **before** model initialization in their script:

#### For CNN/Vision Models

Add this configuration:

```python
# PAI Configuration for CNNs
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_module_names_to_perforate(["Conv2d", "Linear"])
GPA.pc.set_output_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
```

#### For Transformers/Sequence Models

Add this configuration:

```python
# PAI Configuration for Transformers
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_module_names_to_perforate(["Linear"])
GPA.pc.set_output_dimensions([-1, -1, 0])  # [batch, sequence, features]
```

#### For ResNet Models

Add this configuration:

```python
# PAI Configuration for ResNet
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_module_names_to_perforate(["BasicBlock", "Bottleneck", "Linear"])
GPA.pc.set_module_ids_to_track([".conv1", ".bn1"])
GPA.pc.set_output_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
```

#### For Custom/Unknown Models

If the model doesn't match the above patterns, analyze it:

1. **Determine input dimensions:**
   - If images (4D tensors): `[-1, 0, -1, -1]` (batch, channels, height, width)
   - If sequences/text (3D tensors): `[-1, -1, 0]` (batch, sequence, features)

2. **Identify convertible layers:**
   - Look for `nn.Linear`, `nn.Conv2d`, `nn.Conv1d` in their model
   - Start with converting `Linear` layers only for safety
   - Can expand to Conv layers if needed

3. **Add configuration based on analysis:**

```python
# PAI Configuration - analyzed from your model
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_max_dendrites(5)
GPA.pc.set_module_names_to_perforate(["Linear"])  # Start conservative
GPA.pc.set_output_dimensions([...])  # Based on your tensor shape
# May need to skip certain layers - we'll see from debug output
```

Explain your reasoning for each choice based on what you saw in their model when you make the change.

#### Important: Module ID Naming Convention

🚨 **CRITICAL:** When using `set_module_ids_to_track()` or `append_module_ids_to_track()`, all module IDs **MUST start with a "." (dot)**.

**Correct:**

```python
GPA.pc.set_module_ids_to_track([".layer1", ".conv1", ".bn1", ".output_projection"])
GPA.pc.append_module_ids_to_track([".layer2", ".fc"])
```

**Incorrect (will not work):**

```python
GPA.pc.set_module_ids_to_track(["layer1", "conv1", "bn1"])  # ❌ Missing dots
GPA.pc.append_module_ids_to_track(["layer2", "fc"])  # ❌ Missing dots
```

The dot prefix is required because PAI uses these as substring matches against the full module path. For example, ".layer1" will match modules like "model.layer1.conv1", "model.layer1.0.bn1", etc.

**→ Next: Proceed to Step 4.**

---

### Step 4: Initialize Model

Find where their model is created. Before adding PAI initialization, **first analyze their validation loop** to determine if they're maximizing or minimizing a metric.

**Look for their validation code to identify:**

- Metrics like `accuracy`, `acc`, `f1`, `precision`, `recall`, `auc` → maximizing
- Metrics like `loss`, `error`, `mse`, `mae`, `rmse`, `cross_entropy` → minimizing
- Check if they're using `max()`, `min()`, or comparing with `best_acc`, `best_loss`, etc.

**Determine maximizing_score:**

- `maximizing_score=True` if tracking accuracy, F1, precision, or any "higher is better" metric
- `maximizing_score=False` if tracking loss, MSE, MAE, error, or any "lower is better" metric

**Then add PAI initialization:**

Find pattern like:

```python
model = YourModel(...)
model = model.to(device)
```

Change it to:

```python
model = YourModel(...)
model = UPA.perforate_model(model, save_name="your_model_dendritic", maximizing_score=True)  # or False
model = model.to(device)
```

**Set maximizing_score based on what you found:**

- If they track `val_acc`, `accuracy`, `val_f1`, etc.: use `maximizing_score=True`
- If they track `val_loss`, `loss`, `val_mse`, `error`, etc.: use `maximizing_score=False`

Tell them: "I analyzed your validation loop and found you're tracking [metric_name]. I've set `maximizing_score=[True/False]` accordingly."

Make this change in their script.

**→ Next: Proceed to Step 5.**

---

### Step 5: Detect and Handle Multi-GPU Setup

🚨 **MANDATORY: Check for DataParallel or DistributedDataParallel BEFORE proceeding to optimizer setup.**

**Analyze their script to check if they're using DataParallel or DistributedDataParallel:**

**Search for these patterns:**

- `torch.nn.DataParallel(model, ...)` or `nn.DataParallel(model, ...)`
- `torch.nn.parallel.DistributedDataParallel(model, ...)` or `DDP(model, ...)`
- Command-line arguments like `--parallel`, `--multi-gpu`, `--distributed`, `--world-size`, `--local_rank`
- Environment checks like `torch.cuda.device_count() > 1`
- Import statements: `from torch.nn.parallel import DistributedDataParallel`
- `torch.distributed.init_process_group` calls

**After searching, report what you found:**

Tell the user: "I searched your script for DataParallel/DDP usage and found: [NONE / DataParallel / DistributedDataParallel]"

**🔍 DECISION POINT - Choose your path based on what you found:**

**❌ NO DataParallel/DDP found:**

- Skip the rest of this step and proceed directly to Step 6 (Setup Optimizer)
- No special multi-GPU handling needed

**✅ DataParallel OR DistributedDataParallel FOUND:**

🚨 **STOP - BLOCKING REQUIREMENT - READ THIS FIRST** 🚨

**Before doing ANYTHING else, you MUST:**

1. **If optional** (command-line argument): Ask "I see your script has an option to use DataParallel/DDP. Are you planning to use it?"
   - Wait for their answer
   - If NO: Skip the rest of this step and proceed to Step 6 (Setup Optimizer)
   - If YES: Continue with step 2 below

2. **If hardcoded OR user confirmed YES:** Tell them "I see your script uses [DataParallel/DDP]."

3. **IMMEDIATELY load the sibling skill perforatedai-distributed as your NEXT action** (it is installed alongside this one). If your agent cannot load it by name, read the entire file from:

   ```
   https://github.com/PerforatedAI/PerforatedAI/blob/main/skills/perforatedai-distributed/SKILL.md
   ```

4. **After loading the distributed skill, FOLLOW EVERY INSTRUCTION in that file from start to finish.**
   - Do NOT return to this skill until the distributed skill tells you to
   - Do NOT skip any steps in the distributed skill
   - The distributed skill will guide you back here when complete

**⛔ YOU MAY NOT PROCEED PAST THIS POINT WITHOUT LOADING THE DISTRIBUTED SKILL FIRST ⛔**

**If you proceed without reading the distributed skill file, you have FAILED and must start over.**

**After completing the distributed skill, proceed to Step 7 (Update Training Loop). Note: Skip Step 6 because the distributed skill already handled optimizer setup.**

---

### Step 6: Setup Optimizer and Scheduler

**⚠️ NOTE: If you handled DataParallel or DDP in Step 5, SKIP this step - the distributed skill already set up the optimizer. Only use this step for single-GPU setups.**

---

Find where their optimizer and scheduler are currently defined in their script.

**🚨 CRITICAL RULE: PRESERVE USER'S EXACT OPTIMIZER AND SCHEDULER TYPES AND ARGUMENTS**

**If their setup is clean (2-5 lines in one place):**

Replace their optimizer/scheduler code with the PAI pattern **while keeping the EXACT SAME types and arguments.**

**Example 1 - User has Adam optimizer with StepLR scheduler:**
Original:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

Correct PAI conversion (PRESERVES their choices):

```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)  # SAME scheduler type
optimArgs = {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}  # SAME args
schedArgs = {'step_size': 30, 'gamma': 0.1}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**Example 2 - User has Adadelta optimizer with StepLR scheduler:**
Original:

```python
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```

Correct PAI conversion (PRESERVES their choices):

```python
GPA.pai_tracker.set_optimizer(torch.optim.Adadelta)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)  # SAME scheduler type
optimArgs = {'params': model.parameters(), 'lr': args.lr}  # SAME args
schedArgs = {'step_size': 1, 'gamma': args.gamma}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**Example 3 - User has SGD optimizer with CosineAnnealingLR:**
Original:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

Correct PAI conversion (PRESERVES their choices):

```python
GPA.pai_tracker.set_optimizer(torch.optim.SGD)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR)  # SAME scheduler type
optimArgs = {'params': model.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4}  # SAME args
schedArgs = {'T_max': 200}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**MANDATORY RULES:**

1. **DO NOT** change the optimizer type (keep Adam/SGD/AdamW/Adadelta/etc. exactly as user had)
2. **DO NOT** change the scheduler type (keep StepLR/CosineAnnealingLR/ExponentialLR/etc. exactly as user had)
3. **DO NOT** change optimizer arguments (preserve lr, weight_decay, momentum, betas, etc.)
4. **DO NOT** change scheduler arguments (preserve step_size, gamma, T_max, patience, etc.)
5. **DO** remove any `scheduler.step()` calls in their training loop - PAI handles this automatically **ONLY IF** you are initializing the scheduler within setup_optimizer by passing schedArgs AND returning a scheduler from the function. If NOT using setup_optimizer with schedArgs, user must manage scheduler.step() themselves.
6. If user has NO scheduler, use `set_scheduler(None)` or omit the set_scheduler call

**If their setup is complex (scattered across functions, custom classes, framework-managed, etc.):**

Don't try to replace it. Instead, add this single line after their optimizer is fully created:

```python
# Their existing optimizer/scheduler setup stays unchanged
optimizer = ...  # Their code
scheduler = ...  # Their code (if they have one)

# Add only this line after optimizer creation
GPA.pai_tracker.set_optimizer_instance(optimizer)
```

Tell them: "Your optimizer setup is complex, so I'm using the simpler integration method. PAI will work with your existing optimizer configuration."

**IMPORTANT NOTE:** When using `set_optimizer_instance`, PAI will NOT handle the scheduler. You must keep all existing `scheduler.step()` calls in your training loop.

**🔍 CHECK: Does their code use multiple optimizers?**

Before calling `set_optimizer_instance`, scan their code for multiple optimizer definitions (e.g., separate optimizers for encoder/decoder, backbone/head, generator/discriminator, etc.).

**If they have multiple optimizers:**

`set_optimizer_instance` accepts an `additional_optimizers` list for this case:

```python
GPA.pai_tracker.set_optimizer_instance(main_optimizer, additional_optimizers=[optimizer2, optimizer3])
```

- The **first argument** (`main_optimizer`) must be the optimizer that contains the **perforated modules** (the ones listed in `set_module_names_to_perforate`). PAI will add dendrite parameters to this optimizer's param groups.
- **`additional_optimizers`** should contain all other optimizers in the training setup. PAI will handle freezing/unfreezing their parameters correctly across n/p mode switches, but will NOT add dendrites to them.

**🚨 IMPORTANT LIMITATION:** PAI currently only supports perforating modules that are covered by a single optimizer — the first argument to `set_optimizer_instance`. Modules whose parameters are split across multiple optimizers cannot be perforated. Design your `set_module_names_to_perforate` configuration accordingly so all perforated modules live in the main optimizer.

**🏆 BEST PRACTICE: Extract optimizer/scheduler setup into a helper function**

Whether using `setup_optimizer` or `set_optimizer_instance`, the optimizer and scheduler must be rebuilt identically after each `restructured=True` event. The safest way to guarantee this is to extract the setup into a shared function called from both the initial setup and the restructured block:

```python
def build_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return optimizer, scheduler

# Initial setup
optimizer, scheduler = build_optimizer_and_scheduler(model, args)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# After restructure
elif restructured and not training_complete:
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```

This prevents subtle divergence where the initial setup and restructured setup drift apart over time. Always recommend this pattern for any setup more complex than 2-3 lines.

---

### Step 7: Update Training Loop

Find their validation step in the training loop. You need to update it to use PAI's `add_validation_score` function.

**Pattern 1 - If they used PAI optimizer setup (Step 6, first option):**

Find where validation completes and they have a validation score. Add the PAI score tracking and restructuring logic.

Look for something like:

```python
val_acc = validate(model, val_loader)
# or
val_loss = compute_loss(model, val_loader)
```

After this, add:

```python
# Add PAI score tracking
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)  # Pass actual value (val_acc or val_loss)
model = model.to(device)  # Re-apply device settings

if training_complete:
    print("PAI training complete!")
    break

elif restructured and not training_complete:
    # Model was restructured (dendrites added/incorporated)
    # Reinitialize optimizer with EXACT SAME settings from Step 5
    # Example: if Step 5 used Adadelta with StepLR:
    optimArgs = {'params': model.parameters(), 'lr': args.lr}  # EXACT SAME as Step 5
    schedArgs = {'step_size': 1, 'gamma': args.gamma}  # EXACT SAME as Step 5
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**🚨 CRITICAL:**

- Use the EXACT SAME optimArgs and schedArgs from Step 6
- If Step 6 used StepLR, use StepLR args here (step_size, gamma)
- If Step 6 used CosineAnnealingLR, use those args here (T_max, etc.)
- DO NOT change optimizer/scheduler types or arguments in the restructured block
- Just pass the actual validation value (val_acc or val_loss). PAI handles maximization/minimization internally.

**Pattern 2 - If they used `set_optimizer_instance` (Step 6, second option):**

Find where validation completes. Add the PAI score tracking and restructuring logic.

**NOTE:** When using `set_optimizer_instance`, PAI does NOT handle the scheduler. Keep all existing `scheduler.step()` calls in your training loop.

After validation, add:

```python
# Add PAI score tracking
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)  # Pass actual value
model = model.to(device)  # Re-apply device settings

if training_complete:
    print("PAI training complete!")
    break

elif restructured and not training_complete:
    # Model was restructured - reinitialize optimizer exactly as they had it
    # Copy their ENTIRE original optimizer setup (including any complex initialization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Their exact setup
    # If they had scheduler, recreate it too
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)  # If they had one
    # Then tell PAI about it
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```

When `restructured=True`, recreate the optimizer exactly as it was originally defined - if their setup was complex with function calls or multiple steps, copy all of that.

**IMPORTANT: Modify their training loop to handle dynamic completion**

PAI automatically decides when training is complete (when dendrites stop improving validation score). Because of this, their training loop needs to be able to run indefinitely or for many epochs.

**Find their training loop:**

```python
for epoch in range(1, args.epochs + 1):
    # training code
```

**🚨 CRITICAL: Change it to this EXACT pattern (DO NOT use epoch = 0):**

```python
epoch = -1  # 🚨 MUST BE -1, NOT 0! (will increment to 0 at loop start)
while True:
    epoch += 1
    # training code
    # ... validation ...
    # PAI's training_complete will break the loop
```

**Why epoch = -1 and not 0:**

- The loop immediately increments `epoch += 1` BEFORE any training code
- Starting at -1 means the first epoch will be 0 (or 1 if they increment before use)
- This matches their original loop behavior where `range(1, args.epochs + 1)` starts at 1
- **NEVER use `epoch = 0` - this would make the first epoch 1, breaking compatibility**

Tell them: "I've modified your training loop to allow PAI to control when training ends via the `training_complete` flag. PAI will automatically break the loop when adding more dendrites no longer improves validation score."

**⚠️ WARNING:** If their code has epoch-dependent behavior (learning rate schedules, early stopping, etc.), make sure those still work correctly with the modified loop structure.

**After making all the code changes:** Tell them:

> "I've integrated PerforatedAI into your training script. Note: I set `set_testing_dendrite_capacity(True)` which is a debugging flag that helps verify dendrites are being added correctly. We'll change this to `False` for full training after confirming everything works."

### Step 8: Score Tracking Setup

This step is **instrumentation only** - it changes what you can see, not how the model trains, so it is safe to do before the first run.

Options that change _results_ - switch mode, module selection, layer targeting - are deliberately held until after the first run. See [Tuning Options](#tuning-options-only-after-a-first-result). Do not offer them here.

#### 8.1 Additional Score Tracking (Training and Test)

**Understanding the three dataset types:**

- **Training set**: Used for training the model (every epoch)
- **Validation set**: Used to track performance and make dendrite decisions (every epoch) - this is the MAIN metric
- **Test set**: Typically evaluated ONLY at the end for final results (optional)

**🚨 CRITICAL: The validation metric is ALREADY tracked by PAI via `add_validation_score()`. DO NOT duplicate it.**

**First, check if they have a test set:**

Look for a `test_loader`, `test_dataset`, or separate test evaluation function in their code.

**If they have a test set:**

Ask: "I see you have a test set that's normally evaluated at the end. Would you like me to add:

1. Training score tracking every epoch (helps detect overfitting)
2. Test score tracking every epoch (lets you see test scores for EVERY architecture/dendrite count in `_best_arch_scores.csv`, not just your final architecture)"

**If they DON'T have a test set (only train/val):**

Ask: "Would you like me to add training score tracking every epoch? This helps me make better optimization recommendations by comparing training vs validation trends (e.g., detecting overfitting)."

**If yes, modify their training loop:**

**Pattern with all three datasets (train, validation, test):**

```python
# Each epoch
train_acc = train(model, train_loader)
val_acc = validate(model, val_loader)
test_acc = test(model, test_loader)  # Usually only done at end, but can track every epoch

# Track extra scores (training and test)
GPA.pai_tracker.add_extra_score(train_acc, "train")
GPA.pai_tracker.add_extra_score(test_acc, "test")

# Main validation metric for dendrite decisions (NOT duplicated in add_extra_score)
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
```

**Pattern with only two datasets (train, validation):**

```python
# Each epoch
train_acc = train(model, train_loader)
val_acc = validate(model, val_loader)

# Track training score only
GPA.pai_tracker.add_extra_score(train_acc, "train")

# Main validation metric for dendrite decisions
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
```

**Implementation: For training scores** - Track during the training loop (don't run a separate evaluate):

```python
# Inside the training loop, accumulate metrics
train_loss_total = 0
train_correct = 0
train_total = 0

for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Accumulate stats during training
    train_loss_total += loss.item()
    train_correct += (outputs.argmax(1) == targets).sum().item()  # For classification
    train_total += targets.size(0)

# After training epoch, calculate training score
train_acc = train_correct / train_total  # Or use train_loss_total / len(train_loader) for loss
GPA.pai_tracker.add_extra_score(train_acc, "train")
```

**Implementation: For test scores** - Use their existing test/evaluate function:

```python
# After validation, evaluate on test set (if they have one)
test_acc = test(model, test_loader)  # Or evaluate(model, test_loader)
GPA.pai_tracker.add_extra_score(test_acc, "test")
```

**Advanced: If they track MULTIPLE metrics on the same dataset:**

For example, top-1 and top-5 accuracy:

```python
val_acc1 = evaluate_top1(model, val_loader)
val_acc5 = evaluate_top5(model, val_loader)

# Main metric for dendrite decisions
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc1, model)

# Secondary metric for analysis only
GPA.pai_tracker.add_extra_score(val_acc5, "validation_top5")
```

Make these changes in their script.

### Step 9: Verify Dendrite Integration

Tell them:

**"Now run your training script. PAI will automatically test dendrite integration for 7 epochs."**

**"Look for this success message from tracker_perforatedai:**

```
Successfully added 3 dendrites with GPA.pc.set_testing_dendrite_capacity(True) (default).
You may now set that to False and run a real experiment.
```

This message will appear automatically after 7 epochs if everything is working correctly."

**Wait for the user to run the script and report back. Do not run it yourself.**

**Once they see this success message:**

Say: "Great! The test completed successfully. I'll now switch to full training mode."

Find the line:

```python
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
```

Change it to:

```python
GPA.pc.set_testing_dendrite_capacity(False)  # Full training mode
```

Make this change in their script, then go to Step 10 to run full training.

**If the success message does NOT appear:**

This means errors occurred during the test. Tell them: "The test didn't complete successfully. Let's debug the issue."

Switch to debugging mode:

- Check error messages in the console output
- Review the configuration (dimensions, module names, input shapes)
- Load the **perforatedai-debugging** skill and find the error in its index
- Check the debugging documentation in [api/debugging.md](https://github.com/PerforatedAI/PerforatedAI/blob/main/api/debugging.md)

### Step 10: Run Full Training and Compare to the Baseline

Tell them: "You can now run full training. PAI will add dendrites dynamically and manage the training process."

**When it completes, your first action is to compare against the baseline from Step 1.4 - not to recommend tuning.**

|                | Baseline | + Dendrites |
| -------------- | -------- | ----------- |
| <their metric> |          |             |
| Parameters     |          |             |
| Epochs to best |          |             |

Then say plainly whether dendrites helped, and by how much.

- **If the gain is clear:** say so. Everything after this is optimization on top of a result that already works.
- **If it's marginal, or within this model's run-to-run variance:** say that plainly instead of immediately reaching for a tuning fix. A modest first result is common and usually means the config needs work - but the user is entitled to know where they actually stand before spending more GPU hours.
- **If no dendrites were added at all** (`noImprove_lr*` files present, or an empty `switch_epochs.csv`): that is an integration problem, not a tuning problem. Go to the debugging skill, not to the tuning options.

Only once they have this comparison in hand should you offer anything from the next section.

---

### Tuning Options (Only After a First Result)

🚨 **Do not offer anything in this section until the user has a completed run and a comparison against their baseline.** Every option here changes results. Raised before the first run, they pile variables onto an experiment that has not produced a number yet, and they spend the user's GPU hours on choices no evidence supports yet. Raised after, they are informed decisions measured against a known starting point.

**Skip this section entirely during a pilot study (the perforatedai-pilot skill)** - a pilot reports the first comparison and stops.

For recommendations driven by the run's actual data rather than the generic options below, use **perforatedai-analyze**. Prefer it: it reads the CSVs and tells you which of these are worth trying for this specific model.

#### Switch Mode

By default PAI uses history-based switching — it adds a dendrite when validation score stops improving. Ask if they want predictable/fixed-interval switching instead:

Ask: "Would you like to use fixed switch mode, where PAI switches between main and dendrite training on a fixed epoch schedule rather than waiting for plateau detection? This makes total training time more predictable."

If yes, add these lines to the PAI configuration block (before `perforate_model`):

```python
GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
GPA.pc.set_fixed_switch_num(30)       # epochs between each subsequent switch
GPA.pc.set_first_fixed_switch_num(30) # epochs before the very first switch
```

- `set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)` — enable fixed-interval switching mode
- `set_fixed_switch_num` — how many epochs between each switch after the first
- `set_first_fixed_switch_num` — how many epochs to train before the first switch (can differ from subsequent switches if a longer warmup is desired)

Set both to the same value for uniform switching throughout training.

#### Convert Higher-Level Modules

Analyze their model architecture. If they have structured modules (like ResNet blocks, Transformer layers, etc.):

Ask: "I see your model has [BlockType] modules (e.g., BasicBlock, TransformerLayer). Converting at the block level along with individual layers can sometimes work better. Would you like to add block-level conversion?"

**Examples:**

- ResNet: Add `["BasicBlock", "Bottleneck"]` to existing `["Linear", "Conv2d"]`
- Transformers: Add `["TransformerEncoderLayer"]` to existing `["Linear"]`
- Custom architectures: Add their custom module classes

If yes, update the configuration:

```python
# Add to existing module conversions (don't replace)
GPA.pc.append_module_names_to_perforate(["BasicBlock"])  # Or their block type
# This adds to the list - Linear and Conv2d layers inside BasicBlock won't be converted
# because BasicBlock will be converted first
```

Make this change and explain: "I've added [BlockType] to the conversion list. PAI will convert these modules first, so the layers inside them won't be separately converted. This captures higher-level feature patterns."

#### Convert Only Top Layers

Ask: "For parameter efficiency, would you like to convert only the top (deeper) layers of your network? Top layers often benefit more from dendrites than early layers."

If yes, analyze their model structure and identify deeper layers to convert while skipping early ones.

**For sequential models:**

```python
# Skip early layers, only convert later ones
GPA.pc.append_module_ids_to_track([".layer1", ".layer2", ".conv1", ".conv2"])  # Skip these
# Keep layer3, layer4, fc for conversion
```

**For named architectures (ResNet, VGG, etc.):**

```python
# ResNet example: Only convert layer3, layer4 (skip layer1, layer2)
GPA.pc.append_module_ids_to_track([".layer1", ".layer2", ".conv1", ".bn1"])
```

Make the changes and explain: "I've configured PAI to only add dendrites to your top [N] layers. This focuses dendrite resources where they typically have the most impact."

---

### Follow-up Support

- Say **"Analyze my perforated results"** for data-driven recommendations read off the run's CSVs (uses perforatedai-analyze skill). Prefer this over the generic tuning options above.
- Say **"Debug my perforated model"** for issues during training.

---

### 2. "Debug my perforated model" - Debug and Optimize

**This entry point lives in the perforatedai-debugging skill.**

When the user says **"Debug my perforated model"**, load the sibling skill **perforatedai-debugging** and follow it. It carries the triage workflow plus the full error-by-error reference (symptom - cause - fix) for every known PAI failure mode.

If your agent cannot load it by name, read it from:

```
https://github.com/PerforatedAI/PerforatedAI/blob/main/skills/perforatedai-debugging/SKILL.md
```

---

### 3. "Analyze my perforated results" - Review Training Outputs

**This functionality has been moved to a separate skill.**

When the user says **"Analyze my perforated results"**, they should use the **perforatedai-analyze** skill which provides:

- Comprehensive analysis of training CSV outputs
- Performance insights and dendrite impact assessment
- Optimization recommendations based on results
- Learning rate and scheduler tuning suggestions
- Module selection optimization

Tell them: "For analyzing your training results, please use the perforatedai-analyze skill. Just say 'Analyze my perforated results' and the analysis skill will be loaded automatically."

---

### 4. Deployment (Inference, Fine-tuning, and Export)

**These entry points live in the perforatedai-deploy skill.**

When the user says **"Load my perforated model for inference"** or **"Export my perforated model"** (ONNX / TFLite / TorchScript), load the sibling skill **perforatedai-deploy** and follow it.

If your agent cannot load it by name, read it from:

```
https://github.com/PerforatedAI/PerforatedAI/blob/main/skills/perforatedai-deploy/SKILL.md
```

---

## Core Concepts

### What are Artificial Dendrites?

In biological neurons, dendrites perform computation before signals reach the cell body. PerforatedAI adds artificial dendrites to neural network layers, enabling:

- **Dynamic Architecture Growth**: Automatically adds dendrites where needed during training
- **Improved Accuracy**: Better feature representation through dendritic computation
- **Minimal Code Changes**: Wrap your existing PyTorch model with ~10 lines of code

### Architecture

- **PAINeuronModule**: Wrapper that converts a standard PyTorch module (Conv2d, Linear, etc.) into one that can have dendritic copies
- **PAIDendriteModule**: Container for all dendrite modules added to a neuron module
- **Dendrite-to-Neuron Weights**: Learned parameters controlling how dendrite outputs combine with main neuron output

---

**For detailed guidance:**

- Say **"Perforate my model"** to start the interactive setup process
- Say **"Debug my perforated model"** to debug an existing integration (uses perforatedai-debugging skill)
- Say **"Analyze my perforated results"** to review your training outputs and get optimization recommendations (uses perforatedai-analyze skill)
- Say **"Load my perforated model for inference"** or **"Export my perforated model"** to deploy a trained model, or convert it to ONNX / TFLite / TorchScript (uses perforatedai-deploy skill)
- Say **"Run a perforated pilot"** for a controlled baseline-vs-dendrites comparison with no tuning (uses perforatedai-pilot skill)
