---
name: perforatedai
description: "Expert in PerforatedAI library for adding artificial dendrites to PyTorch neural networks. Triggers: 'Perforate my model' (start interactive setup), 'debug my perforated model' (debug/optimize existing integration), 'analyze my perforated results' (analyze training outputs). Also use when: debugging dendrite training, configuring PAI settings, troubleshooting dendrite issues, working with PAINeuronModule or PAIDendriteModule."
---

# PerforatedAI Dendrite Network Skill

## Available Resources

This skill has access to:
- **API Documentation**: [api-docs/](./api-docs/) - All API guides and references
- **Source Code**: [source-code/](./source-code/) - Complete PerforatedAI library implementation
- **Examples**: [examples/](./examples/) - Working code examples for various architectures

Feel free to reference these when helping users debug or understand implementation details.

## Entry Points

### 1. "Perforate my model" - Interactive Setup

When the user says **"Perforate my model"**, start the interactive setup process.

**IMPORTANT: Your role is to analyze code and make edits. Never run the user's training script - only ask them to run it and provide output.**

**First, check if PAI is already partially integrated:**

Ask for their training script path, then read it and check for:
- PAI imports: `from perforatedai import globals_perforatedai as GPA` and `utils_perforatedai as UPA`
- Configuration calls: `GPA.pc.set_*` functions
- Model initialization: `UPA.initialize_pai()` call
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

### Step 1: Discovery

#### 1.1 Get Training Script and Analyze Model

**Ask:** "What's the path to your training script?"

- If they provide a path: Read the script and analyze it to determine:
  - Model architecture type (CNN, Transformer, ResNet, Custom, etc.)
  - Model version/size (e.g., ResNet18 vs ResNet50, GPT2-small vs GPT2-large)
  - Input dimensions and data format
  - Training loop structure
  - Optimization metric being used
  
- If they say they don't have a script yet: Tell them:
  > "PerforatedAI is an optimization tool for existing models. Please build an initial training setup first and get a baseline working before integrating dendrites. Once you have a working training script, come back and say 'Perforate my model' to add dendritic optimization."
  
  Then stop - do not proceed with integration.

- If the script has multiple model options: **Ask which one they're using**
  - Multiple architectures (e.g., ResNet vs VGG)
  - Multiple versions of same architecture (e.g., ResNet18 vs ResNet34 vs ResNet50)
  - Configurable model sizes

#### 1.2 Ask About Optimization Goal

**Ask:** "What are you optimizing for?"

Options:
- **Accuracy / Loss / Decision Making** - Improve model performance metrics
- **Efficiency / Model Size** - Reduce parameters while maintaining performance

Based on their answer:

**If Accuracy/Loss/Decision Making:**
- Proceed to Step 2 with standard dendrite growth settings (maximize metric or minimize loss)

**If Efficiency/Model Size:**
- **Educate them first:** "Important: Dendrites ADD parameters to your model, but they do so more efficiently than traditional approaches. To optimize for efficiency with PerforatedAI, you should start with a smaller base model and then add dendrites strategically.  This will allow you to achieve better performance than a larger model with fewer parameters. Let's work together to find the right balance for your use case."

- **Then determine the path based on their model:**

  **For Pretrained Models (e.g., ResNet50, BERT-base):**
  - Tell them: "You're using [ModelName]. To optimize for efficiency, I recommend switching to [SmallerVariant] and adding dendrites. For example:
    - ResNet50 → ResNet34 + dendrites
    - ResNet34 → ResNet18 + dendrites
    - BERT-base → BERT-small + dendrites
    - GPT2-medium → GPT2-small + dendrites"
  
  - If they're already using the smallest common variant (e.g., ResNet18), suggest alternative efficient architectures:
    - "Since you're already using ResNet18, consider MobileNetV2 or EfficientNet-B0 as a more efficient base"
    - "For transformers, consider DistilBERT instead of BERT-small"
  
  - Modify their script to load the smaller pretrained model variant
  
  - **Before proceeding to Step 2:** Tell them:
    "I've updated your script to use [SmallerModel]. Before we add dendrites, please run a quick training test to confirm it loads correctly and trains without errors. This smaller model will have lower accuracy initially, which is expected. Dendrites will help recover performance."
  
  - Wait for confirmation before proceeding to Step 2
  
  **For Custom Models:**
  - Analyze their model architecture to identify:
    - Number of layers (depth)
    - Hidden dimensions/channels (width)
    - Whether they have command-line arguments or config files for these
  
  - Tell them: "To optimize for efficiency, let's make your model smaller first, then add dendrites. We can:
    - Reduce layer count (make it shallower)
    - Reduce hidden dimensions/channels (make it less wide)
    - Both"
  
  - Ask: "Would you like to reduce depth (fewer layers), width (fewer channels/neurons), or both?"
  
  - **Then implement the changes based on their code structure:**
  
    **If they already have configurable width/depth parameters:**
    - Identify where these are set (config file, command-line args, hardcoded values)
    - Update them to smaller values. For example:
      - `hidden_dim=512` → `hidden_dim=256`
      - `num_layers=12` → `num_layers=6`
    - Make the changes and tell them what you changed
    
    **If they have command-line arguments but not for width/depth:**
    - Add new command-line arguments for the parameters they want to adjust
    - Example: Add `--hidden_dim`, `--num_layers`, `--num_channels`, etc.
    - Set reasonable defaults that are smaller than their current hardcoded values
    - Update their model instantiation to use these new arguments
    
    **If they don't have command-line arguments:**
    - Add configurable settings at the top of their script or in a config section
    - Example:
      ```python
      # Model configuration (adjust these for efficiency)
      NUM_LAYERS = 6  # Original: 12
      HIDDEN_DIM = 256  # Original: 512
      NUM_CHANNELS = 32  # Original: 64
      ```
    - Update their model definition to use these variables instead of hardcoded values
  
  - **Before proceeding to Step 2:**
    - Tell them: "I've updated your model to be smaller. Before we add dendrites, please run your training script now to confirm:
      1. Training runs without errors
      2. The model is indeed smaller (check parameter count)
      3. Accuracy is somewhat lower than your original model (expected)
      
      This establishes a baseline. After we add dendrites, we'll aim to recover or exceed your original accuracy with this more efficient architecture."
    
    - Wait for them to confirm they've tested the smaller model before proceeding to dendrite integration

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
GPA.pc.set_max_dendrites(5)
GPA.pc.set_module_names_to_convert(["Conv2d", "Linear"])
GPA.pc.set_input_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
```

#### For Transformers/Sequence Models

Add this configuration:

```python
# PAI Configuration for Transformers
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_max_dendrites(5)
GPA.pc.set_module_names_to_convert(["Linear"])
GPA.pc.set_input_dimensions([-1, -1, 0])  # [batch, sequence, features]
GPA.pc.append_module_ids_to_track([".output_projection"])  # Skip final layer
```

#### For ResNet Models

Add this configuration:

```python
# PAI Configuration for ResNet
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_max_dendrites(3)
GPA.pc.set_module_names_to_convert(["BasicBlock", "Bottleneck"])
GPA.pc.append_module_ids_to_track([".layer1", ".layer2", ".conv1", ".fc"])
GPA.pc.set_input_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
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
GPA.pc.set_module_names_to_convert(["Linear"])  # Start conservative
GPA.pc.set_input_dimensions([...])  # Based on your tensor shape
# May need to skip certain layers - we'll see from debug output
```

Explain your reasoning for each choice based on what you saw in their model when you make the change.

### Step 4: Initialize Model with PAI

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
model = UPA.initialize_pai(model, save_name="your_model_dendritic", maximizing_score=True)  # or False
model = model.to(device)
```

**Set maximizing_score based on what you found:**
- If they track `val_acc`, `accuracy`, `val_f1`, etc.: use `maximizing_score=True`
- If they track `val_loss`, `loss`, `val_mse`, `error`, etc.: use `maximizing_score=False`

Tell them: "I analyzed your validation loop and found you're tracking [metric_name]. I've set `maximizing_score=[True/False]` accordingly."

Make this change in their script.

### Step 5: Setup Optimizer and Scheduler

Find where their optimizer and scheduler are currently defined in their script.

**If their setup is clean (2-5 lines in one place):**

Replace their optimizer/scheduler code with the PAI pattern. For example, if they have:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
```

Replace it with:
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)  # Their optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)  # Prefer ReduceLROnPlateau
optimArgs = {'params': model.parameters(), 'lr': 0.001}  # Their learning rate and other args
schedArgs = {'mode': 'max', 'patience': 5}  # 'max' if maximizing, 'min' if minimizing
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**Important when making this change:**
- Keep their optimizer type (Adam, SGD, AdamW, etc.)
- Preserve all their optimizer arguments (lr, weight_decay, momentum, betas, etc.) in optimArgs
- **Set scheduler mode to match Step 4's maximizing_score setting:**
  - If `maximizing_score=True`: use `schedArgs = {'mode': 'max', 'patience': 5}`
  - If `maximizing_score=False`: use `schedArgs = {'mode': 'min', 'patience': 5}`
- Prefer ReduceLROnPlateau scheduler (works best with PAI), or keep their scheduler type
- If keeping their scheduler, adapt schedArgs to their scheduler's parameters
- Find and remove any `scheduler.step()` calls in their training loop - PAI handles this automatically

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

### Step 6: Update Training Loop

Find their validation step in the training loop. You need to update it to use PAI's `add_validation_score` function.

**Pattern 1 - If they used PAI optimizer setup (Step 5, first option):**

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

elif restructured:
    # Model was restructured (dendrites added/incorporated)
    # Reinitialize optimizer with same settings from Step 5
    optimArgs = {'params': model.parameters(), 'lr': learning_rate}  # Copy from Step 5
    schedArgs = {'mode': 'max', 'patience': 5}  # Copy from Step 5
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**Important:** Just pass the actual validation value (accuracy or loss). PAI handles the maximization/minimization internally based on the `maximizing_score` setting from Step 4.

Make sure optimArgs and schedArgs match exactly what you used in Step 5.

**Pattern 2 - If they used `set_optimizer_instance` (Step 5, second option):**

Find where validation completes. Add the PAI score tracking and restructuring logic.

After validation, add:
```python
# Add PAI score tracking
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)  # Pass actual value
model = model.to(device)  # Re-apply device settings

if training_complete:
    print("PAI training complete!")
    break

elif restructured:
    # Model was restructured - reinitialize optimizer exactly as they had it
    # Copy their ENTIRE original optimizer setup (including any complex initialization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Their exact setup
    # If they had scheduler, recreate it too
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)  # If they had one
    # Then tell PAI about it
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```

When `restructured=True`, recreate the optimizer exactly as it was originally defined - if their setup was complex with function calls or multiple steps, copy all of that.

**After making all the code changes:** Tell them:
> "I've integrated PerforatedAI into your training script. Note: I set `set_testing_dendrite_capacity(True)` which is a debugging flag that helps verify dendrites are being added correctly. We'll change this to `False` for full training after confirming everything works."

### Step 7: Optional Configuration Tuning

Before running the first experiment, check with them about optional configurations that can improve results and analysis:

#### 7.1 Additional Score Tracking

Ask: "Would you like me to add training and test score tracking? This helps in two ways:
1. I can make better optimization recommendations by seeing training vs validation trends
2. You'll get automatic reports of best test scores for each architecture, not just validation scores"

If yes, modify their training loop to track scores:

**For training scores** - Track during the training loop (don't run a separate evaluate):
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

# After epoch, calculate and log training score
train_acc = train_correct / train_total  # Or train_loss_total / len(train_loader)
GPA.pai_tracker.add_extra_score(train_acc, "train")
```

**For test scores** - Run evaluation on test set:
```python
# After validation, if they have a test set
test_acc = evaluate(model, test_loader)  # Or test_loss
GPA.pai_tracker.add_extra_score(test_acc, "test")
```

Make these changes in their script.

#### 7.2 Convert Higher-Level Modules

Analyze their model architecture. If they have structured modules (like ResNet blocks, Transformer layers, etc.):

Ask: "I see your model has [BlockType] modules (e.g., BasicBlock, TransformerLayer). Converting at the block level along with individual layers can sometimes work better. Would you like to add block-level conversion?"

**Examples:**
- ResNet: Add `["BasicBlock", "Bottleneck"]` to existing `["Linear", "Conv2d"]`
- Transformers: Add `["TransformerEncoderLayer"]` to existing `["Linear"]`
- Custom architectures: Add their custom module classes

If yes, update the configuration:
```python
# Add to existing module conversions (don't replace)
GPA.pc.append_module_names_to_convert(["BasicBlock"])  # Or their block type
# This adds to the list - Linear and Conv2d layers inside BasicBlock won't be converted
# because BasicBlock will be converted first
```

Make this change and explain: "I've added [BlockType] to the conversion list. PAI will convert these modules first, so the layers inside them won't be separately converted. This captures higher-level feature patterns."

#### 7.3 Convert Only Top Layers

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

**After optional configurations:**

Tell them: "Configuration complete! Now let's verify the integration works."

### Step 8: Verify Dendrite Integration

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

Make this change in their script and tell them: "You can now run full training. PAI will add dendrites dynamically and manage the training process."

**If the success message does NOT appear:**

This means errors occurred during the test. Tell them: "The test didn't complete successfully. Let's debug the issue."

Switch to debugging mode:
- Check error messages in the console output
- Review the configuration (dimensions, module names, input shapes)
- Reference the [Debugging](#debugging-common-issues) section below for common issues
- Check the debugging documentation in [api-docs/debugging.md](./api-docs/debugging.md)

### Follow-up Support

After the initial setup and verification, you can now run full training.

**When your training is complete, come back and say: "Analyze my perforated results"**

---

### 2. "Debug my perforated model" - Debug and Optimize

When the user says **"Debug my perforated model"**, help them debug or optimize an existing PerforatedAI integration.

**IMPORTANT: Never run their training script. Only read files, analyze code, make edits, and ask the user to run the script and provide error messages.**

**Step 1: Get their training script**

Ask: "What's the path to your training script?"

Read the script and analyze the current PAI setup. **Do not run it.**

**Step 2: Check integration completeness**

Verify all required components are present:
- ✅ Imports (GPA, UPA)
- ✅ Configuration (set_testing_dendrite_capacity, set_max_dendrites, etc.)
- ✅ Model initialization (UPA.initialize_pai)
- ✅ Optimizer setup (setup_optimizer or set_optimizer_instance)
- ✅ Training loop (add_validation_score)

Report any missing components and offer to add them.

**Step 3: Enable debug mode if needed**

Check if `set_testing_dendrite_capacity` is currently set to `False` in their script.

**If set to False:**
- Ask: "I see you have `set_testing_dendrite_capacity(False)`. Does the crash/issue still occur when you set it to `True`?"

**Based on their answer:**
- **If they say "yes" or "not sure":** Change it to `True` in their script and tell them:
  > "I've temporarily set `testing_dendrite_capacity=True` for debugging. This runs a simplified 7-epoch test which helps isolate issues. We'll set it back to `False` after fixing the problem."
  
- **If they say "no, it works fine with True":** Tell them:
  > "The issue only occurs with `testing_dendrite_capacity=False`. This suggests the problem is related to full training mode. Let's debug with it set to `False` to reproduce the actual issue."
  > Keep it set to `False` for debugging.

**If already set to True:**
- Proceed with debugging - no changes needed

**Step 4: Identify the issue**

Ask: "What issue are you experiencing? If you're getting a crash or error, please copy-paste the full error message/traceback."

**Wait for the user to provide the issue/error. Do not try to run their script.**

Common scenarios:

**A. "Training errors / crashes"**
- Once you have the error traceback, analyze it for common issues:
  - Dimension mismatches in `set_input_dimensions()`
  - Wrong module names in `set_module_names_to_convert()`
  - Device placement issues (model not on correct device after restructuring)
  - Optimizer not reinitialized after restructuring
- Check [api-docs/debugging.md](./api-docs/debugging.md) for detailed debugging steps

**B. "Dendrites not being added"**
- Verify based on n_epochs_to_switch, improvement_threshold, and save_name_scores.csv that enough epochs have passed such that dendrites should have been added
- Check improvement_threshold - may be too strict
- Check epoch count that enough epochs have passed
- Verify validation scores are being passed to `add_validation_score()`

**C. "Dendrites added but performance worse"**
- Check if `maximizing_score` matches their metric (True for accuracy, False for loss)
- Verify they're passing the raw score value (not negated)

**Step 5: Make fixes**

Based on the identified issue, make the necessary code changes directly in their script.

**Step 6: Verification and restore settings**

After fixes, tell them what was changed and **ask them to run the training script**.

Say: "Please run your training script now and let me know:
- Does it run without errors?
- What output do you see?"

**Do not run the script yourself - wait for the user to run it and report back.**

**If you changed `testing_dendrite_capacity` to True in Step 3:**
- Tell them: "First, run a quick test with `testing_dendrite_capacity=True` to verify the fix works in debug mode."
- After they confirm it works, change it back to `False` and tell them:
  > "Great! I've set `testing_dendrite_capacity=False` back for full training. Run your training again to confirm everything works in production mode."

**If it was already True or you kept it False:**
- Just tell them what to look for when they run training with current settings

---

### 3. "Analyze my perforated results" - Review Training Outputs

When the user says **"Analyze my perforated results"**, perform a comprehensive analysis of their PAI training outputs.

### Step 1: Locate Result Files

Find the `save_name` from their training script by searching for the `UPA.initialize_pai()` call. The `save_name` parameter shows where results are stored.

**Only ask "What was your save_name?" if:**
- The script has a variable or argument for save_name that could change
- You cannot find the initialize_pai call in their script

The results are stored in: `{save_name}/{save_name}_*.csv`

Look for these files:
- `{save_name}/{save_name}_scores.csv` - Validation scores over epochs
- `{save_name}/{save_name}_best_arch_scores.csv` - Best architecture performance at each dendrite count
- `{save_name}/{save_name}_switch_epochs.csv` - Epochs when dendrites were added
- `{save_name}/{save_name}_learning_rate.csv` - Learning rate schedule
- `{save_name}/{save_name}Best_PBScores.csv` - Perforated Backpropagation scores (if enabled)

### Step 2: Read and Analyze Score Files

Read all available CSV files and analyze:

**From `{save_name}_scores.csv`:**
- Validation score progression over epochs
- Calculate overall improvement from baseline to final
- Identify training phases (neuron mode vs dendrite mode)
- Check for plateaus or instabilities

**From `{save_name}_switch_epochs.csv`:**
- Identify exactly when dendrites were added (epoch numbers)
- Correlate dendrite additions with score changes from _scores.csv
- Determine if dendrite additions coincided with improvements

**From `{save_name}_best_arch_scores.csv`:**
- Compare performance across different dendrite counts (0, 1, 2, 3, etc.)
- Identify if there are diminishing returns after a certain number of dendrites
- **Determine optimal dendrite count:** If score gains plateau or decrease after N dendrites, recommend setting `max_dendrites` to that value
- Example: If dendrites 0→1→2→3 show good gains but 4→5 show minimal improvement, recommend `max_dendrites=3`

**From `{save_name}_learning_rate.csv`:**
- Show learning rate schedule
- Correlate LR changes with score changes
- Identify if LR scheduling worked well with dendrite additions

**From `{save_name}Best_PBScores.csv` (if exists):**
- Analyze correlation scores for each module
- **Identify modules with low correlation scores** (< 0.3 or bottom 20%)
- **Recommendation:** If certain modules show very low correlation scores, they may not benefit from dendrites
  - Suggest adding those module IDs to `append_module_ids_to_track()` to skip them
  - Focus dendrite resources on high-correlation modules instead

### Step 3: Generate Insights

Provide a comprehensive summary including:

1. **Training Summary:**
   - Initial score vs final score (% improvement)
   - Total epochs trained
   - Number of dendrites added
   - Training phases observed

2. **Performance Analysis:**
   - Best validation score achieved
   - Score stability (variance in later epochs)
   - Whether training converged properly

3. **Dendrite Impact:**
   - Score improvement after each dendrite addition (from switch_epochs.csv + scores.csv)
   - Which dendrite additions had the most impact
   - Whether dendrites helped or hurt performance
   - Optimal dendrite count based on diminishing returns in best_arch_scores

4. **Comparison to Baseline:**
   - Baseline score is the first score in best_arch_scores
   - Calculate parameter count increase (if available)
   - Assess efficiency: did dendrites provide good accuracy/parameter ratio?

5. **Module-Level Analysis (if PB scores available):**
   - Which modules benefited most from dendrites (high correlation)
   - Which modules should be excluded (low correlation)

### Step 4: Show Visualizations

Tell them: "Training visualizations have been automatically generated at `{save_name}/{save_name}.png`. This shows:
- Score progression over epochs
- Learning rate schedule
- Dendrite addition timeline
- Architecture performance comparison"

### Step 5: Next Steps

Based on the analysis results:

**If training went well (dendrites improved performance):**

Say: "Your dendritic training was successful! Here's what I found worked well and recommendations for optimization."

Then direct them to the [Optimization Recommendations](#optimization-recommendations-after-successful-training) section below.

**If training had issues (dendrites didn't help or training was unstable):**

Say: "I see some issues in your training results. Let's debug:"

- **If dendrites didn't improve performance:**
  - Check if you're converting the right layers
  - Is improvement_threshold too strict? Try `[0]`
  - Try different input_dimensions or module configurations
  - Reference [Debugging](#debugging-common-issues) section
  
- **If training was unstable:**
  - Consider reducing learning rate
  - Adjust candidate_weight_initialization_multiplier lower (0.01 instead of 0.1)
  - Try a different scheduler
  - Check for dimension mismatches in [Debugging](#debugging-common-issues)

---

## Optimization Recommendations After Successful Training

When dendritic training has successfully improved performance, provide targeted recommendations based on the analysis:

### 1. Optimize Dendrite Count

Based on `best_arch_scores.csv` analysis:

**If diminishing returns detected:**
- Example: Dendrites 0→1→2→3 showed gains of +5%, +3%, +2%, but 4→5 showed only +0.1%
- **Recommend:** `GPA.pc.set_max_dendrites(3)` to focus resources on high-impact dendrites
- Explain: "Your results show the biggest improvements came from the first 3 dendrites. Setting max_dendrites=3 will make training more efficient without sacrificing performance."

**If all dendrites contributed equally:**
- Keep current max_dendrites or increase slightly
- Suggest running longer to see if more dendrites could help

### 2. Adjust Improvement Threshold

Based on score progression from `scores.csv` and `switch_epochs.csv`:

**If dendrites were added too frequently:**
- Current threshold may be too lenient
- **Recommend:** Tighten threshold, e.g., `[0.02, 0.01, 0.001, 0]` instead of `[0.01, 0.001, 0.0001, 0]`
- This makes dendrite additions more selective

**If dendrites were rarely added but helpful:**
- Threshold may be too strict
- **Recommend:** Relax threshold or set to `[0]` to always try adding dendrites when performance plateaus

### 3. Module Selection Optimization

Based on `Best_PBScores.csv` (if available):

**If certain modules show low correlation scores (< 0.3):**
- **Recommend:** Add those module IDs to exclusion list
- Example: If `.layer1` and `.conv1` show correlation < 0.2:
  ```python
  GPA.pc.append_module_ids_to_track([".layer1", ".conv1"])  # Skip these
  ```
- Explain: "These modules showed low dendrite correlation, meaning dendrites didn't help them much. Excluding them will focus resources on high-impact layers."

**If all modules show high correlation:**
- Current module selection is working well
- Consider expanding to convert additional layer types if any were excluded

### 4. Learning Rate and Scheduler Tuning

Based on `learning_rate.csv` and correlation with `scores.csv`:

**If learning rate dropped too quickly:**
- Scores plateaued before dendrites could be fully optimized
- **Recommend:** Increase scheduler patience or use slower decay
- Example: `schedArgs = {'mode': 'max', 'patience': 10}` instead of 5

**If learning rate stayed high too long:**
- Training may have been unstable during dendrite additions
- **Recommend:** Faster decay or lower initial learning rate

### 5. Weight Initialization for Dendrites

Based on training stability from `scores.csv`:

**If scores showed spikes/instability after dendrite additions:**
- Dendrite weights may be initialized too large
- **Recommend:** Lower `candidate_weight_initialization_multiplier`
- Example: `GPA.pc.set_candidate_weight_initialization_multiplier(0.01)` instead of 0.1

**If scores were very smooth:**
- Current initialization is working well
- Could potentially try slightly larger values for faster adaptation

### 6. Training Duration Recommendations

Based on total epochs and convergence:

**If training completed but scores still improving:**
- **Recommend:** Increase `set_n_epochs_to_switch()` to train longer in each phase
- Allow more time for dendrites to optimize before adding more

**If training converged early:**
- Current settings are efficient
- Could reduce epoch counts for faster experimentation

### 7. Architecture Expansion Strategies

If dendrites significantly improved performance:

**Consider converting additional layers:**
- If you only converted `["Linear"]`, try adding `["Conv2d", "Linear"]`
- If you skipped early layers, try including them: remove some IDs from `append_module_ids_to_track()`

**Consider increasing max_dendrites:**
- If best_arch_scores showed steady improvements across all dendrite counts
- Try max_dendrites=7 or 10 for potentially higher performance

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
- Say **"Debug my perforated model"** to debug or optimize an existing integration  
- Say **"Analyze my perforated results"** to review your training outputs and get optimization recommendations
